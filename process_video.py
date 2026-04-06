#!/usr/bin/env python3
"""
BDD100K HydraNet Video Inference
================================

Process a BDD100K dashcam video through the trained HydraNet model,
producing a side-by-side output with drivable area overlay + depth colormap.

Usage:
    python process_video.py --video path/to/video.mov --checkpoint best_model.pth
    python process_video.py --video path/to/video.mov --checkpoint best_model.pth --output result.mp4
    python process_video.py --video path/to/video.mov --checkpoint best_model.pth --fps 10  # faster processing

Requirements:
    pip install torch torchvision opencv-python-headless efficientnet_pytorch
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── Model Architecture (must match notebook) ─────────────────────────

INPUT_SIZE = (256, 512)

class Backbone(nn.Module):
    def __init__(self):
        from efficientnet_pytorch import EfficientNet
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')

    def forward(self, x):
        features = []
        x = self.base._swish(self.base._bn0(self.base._conv_stem(x)))
        prev_x = x
        for idx, block in enumerate(self.base._blocks):
            x = block(x)
            if idx == 0 or x.shape[-1] != prev_x.shape[-1]:
                features.append(prev_x)
            prev_x = x
        features.append(x)
        x = self.base._swish(self.base._bn1(self.base._conv_head(x)))
        features.append(x)
        return features


class SceneContext(nn.Module):
    def __init__(self, in_channels=1280):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 800),
            nn.ReLU(),
            nn.Linear(800, 128),
            nn.ReLU(),
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        gap = self.gap(x).view(b, -1)
        ctx = self.mlp(gap)
        ctx = ctx.view(b, 1, h, w)
        attn = self.reconstruct(ctx)
        return x * attn


class DepthContext(nn.Module):
    def __init__(self, in_channels=1280):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 800),
            nn.ReLU(),
            nn.Linear(800, 128),
            nn.ReLU(),
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        gap = self.gap(x).view(b, -1)
        ctx = self.mlp(gap)
        ctx = ctx.view(b, 1, h, w)
        attn = self.reconstruct(ctx)
        return x * attn


class SceneNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1280, 768, 2, stride=2),
            nn.BatchNorm2d(768), nn.ReLU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, 2, stride=2),
            nn.BatchNorm2d(512), nn.ReLU())
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, deep):
        x = self.up1(deep)
        x = self.up2(x)
        x = self.up3(x)
        return x


class DrivableAreaHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, num_classes, 1))

    def forward(self, x):
        return self.head(x)


class DepthHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.head(x)


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.drivable_context = SceneContext(1280)
        self.depth_context = DepthContext(1280)
        self.neck = SceneNeck()
        self.drivable_head = DrivableAreaHead()
        self.depth_head = DepthHead()

    def forward(self, x):
        features = self.backbone(x)
        deep = features[-1]

        drv_feat = self.drivable_context(deep)
        drv_dec = self.neck(drv_feat)
        drivable = self.drivable_head(drv_dec)
        drivable = F.interpolate(drivable, size=x.shape[2:],
                                 mode='bilinear', align_corners=False)

        dep_feat = self.depth_context(deep)
        dep_dec = self.neck(dep_feat)
        depth = self.depth_head(dep_dec)
        depth = F.interpolate(depth, size=x.shape[2:],
                              mode='bilinear', align_corners=False)

        return drivable, depth


# ── Visualization helpers ────────────────────────────────────────────

DRIVABLE_COLORS = np.array([
    [0, 0, 0],        # 0: not drivable
    [0, 180, 0],      # 1: direct (green)
    [0, 100, 255],    # 2: alternative (blue)
], dtype=np.uint8)


def overlay_drivable(frame, pred_mask, alpha=0.4):
    """Overlay drivable area prediction on the frame."""
    color_mask = DRIVABLE_COLORS[pred_mask]
    overlay = frame.copy()
    mask_active = pred_mask > 0
    overlay[mask_active] = cv2.addWeighted(
        frame[mask_active], 1 - alpha,
        color_mask[mask_active], alpha, 0
    )
    return overlay


def colorize_depth(depth_map):
    """Convert normalized depth to a colormap."""
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    return colored


# ── Main inference ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HydraNet video inference")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", default=None, help="Output video path (default: input_hydranet.mp4)")
    parser.add_argument("--fps", type=int, default=0, help="Process every Nth frame (0 = use all frames)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--no-display", action="store_true", help="Skip cv2 display window")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base}_hydranet.mp4"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = HydraNet().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {src_fps:.1f} fps, {total_frames} frames")

    # Output video writer — side by side (drivable overlay | depth colormap)
    out_w = width * 2
    out_h = height
    out_fps = src_fps if args.fps == 0 else src_fps / args.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (out_w, out_h))

    frame_idx = 0
    processed = 0
    t0 = time.time()

    print("Processing...")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames if requested
            if args.fps > 0 and (frame_idx % args.fps) != 0:
                continue

            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

            # Inference
            drivable_out, depth_out = model(input_tensor)

            # Post-process drivable
            drivable_pred = drivable_out.argmax(dim=1).squeeze(0).cpu().numpy()
            drivable_pred = cv2.resize(drivable_pred.astype(np.uint8),
                                       (width, height),
                                       interpolation=cv2.INTER_NEAREST)

            # Post-process depth
            depth_pred = depth_out.squeeze(0).squeeze(0).cpu().numpy()
            depth_pred = cv2.resize(depth_pred, (width, height),
                                    interpolation=cv2.INTER_LINEAR)

            # Render
            left = overlay_drivable(frame, drivable_pred)
            right = colorize_depth(depth_pred)

            # Side by side
            combined = np.concatenate([left, right], axis=1)
            writer.write(combined)

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                fps_actual = processed / elapsed
                print(f"  {processed} frames ({frame_idx}/{total_frames}) "
                      f"— {fps_actual:.1f} fps")

            if args.max_frames > 0 and processed >= args.max_frames:
                break

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"\nDone! {processed} frames in {elapsed:.1f}s "
          f"({processed / elapsed:.1f} fps)")
    print(f"Output: {args.output}")
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
