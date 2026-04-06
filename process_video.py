#!/usr/bin/env python3
"""
BDD100K HydraNet Video Inference
================================

Process a BDD100K dashcam video through the trained HydraNet model,
producing a side-by-side output with drivable area overlay + depth colormap.

Usage:
    python process_video.py --video path/to/video.mov --checkpoint hydranet_best.pth
    python process_video.py --video path/to/video.mov --checkpoint hydranet_best.pth --fps 5
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
from torchvision import models, transforms

# ── Must match notebook ──────────────────────────────────────────────

INPUT_SIZE = (320, 640)  # (H, W)


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Backbone, self).__init__()
        if pretrained:
            self.encoder = models.efficientnet_b0(
                weights='EfficientNet_B0_Weights.IMAGENET1K_V1'
            ).features
        else:
            self.encoder = models.efficientnet_b0(weights=None).features

    def forward(self, image):
        l0 = self.encoder[0](image)
        l1 = self.encoder[1](l0)
        l2 = self.encoder[2](l1)
        l3 = self.encoder[3](l2)
        l4 = self.encoder[4](l3)
        l5 = self.encoder[5](l4)
        l6 = self.encoder[6](l5)
        l7 = self.encoder[7](l6)
        l8 = self.encoder[8](l7)
        return [l0, l2, l3, l4, l8]


class SceneContext(nn.Module):
    def __init__(self):
        super(SceneContext, self).__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        spatial_size = (INPUT_SIZE[0] // 32) * (INPUT_SIZE[1] // 32)
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, spatial_size)

        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        b, c, h, w = features.shape
        feature_vector = torch.mean(features, dim=[2, 3])
        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))
        c3 = c2.view(b, 1, h, w)
        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))
        return c7 * features + features


class DepthContext(nn.Module):
    def __init__(self):
        super(DepthContext, self).__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        spatial_size = (INPUT_SIZE[0] // 32) * (INPUT_SIZE[1] // 32)
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, spatial_size)

        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        b, c, h, w = features.shape
        feature_vector = torch.mean(features, dim=[2, 3])
        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))
        c3 = c2.view(b, 1, h, w)
        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))
        return c7 * features + features


class SceneNeck(nn.Module):
    def __init__(self):
        super(SceneNeck, self).__init__()
        self.GeLU = nn.GELU()

        self.upsample_layer_0 = nn.ConvTranspose2d(1280, 1280, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(80, 1280, 1)
        self.decode_layer_0 = nn.Conv2d(1280, 768, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(768, 768, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(768, 768, 2, 2)
        self.skip_link_layer_1 = nn.Conv2d(40, 768, 1)
        self.decode_layer_2 = nn.Conv2d(768, 512, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.upsample_layer_2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.skip_link_layer_2 = nn.Conv2d(24, 512, 1)
        self.decode_layer_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.decode_layer_5 = nn.Conv2d(512, 256, 3, 1, 1)

    def forward(self, context, features):
        d0 = self.upsample_layer_0(context)
        d0 = d0 + self.skip_link_layer_0(features[3])
        d1 = self.GeLU(self.decode_layer_0(d0))
        d2 = self.GeLU(self.decode_layer_1(d1))

        d3 = self.upsample_layer_1(d2)
        d3 = d3 + self.skip_link_layer_1(features[2])
        d3 = self.GeLU(self.decode_layer_2(d3))
        d4 = self.GeLU(self.decode_layer_3(d3))

        d5 = self.upsample_layer_2(d4)
        d5 = d5 + self.skip_link_layer_2(features[1])
        d5 = self.GeLU(self.decode_layer_4(d5))
        neck = self.GeLU(self.decode_layer_5(d5))
        return neck


class DrivableAreaHead(nn.Module):
    def __init__(self, num_classes=3):
        super(DrivableAreaHead, self).__init__()
        self.GeLU = nn.GELU()

        self.upsample_layer_0  = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(32, 256, 1)
        self.decode_layer_0    = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1    = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_2   = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_3   = nn.Conv2d(128, 64, 3, 1, 1)

        self.output_layer = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, neck, features):
        d0 = self.upsample_layer_0(neck) + self.skip_link_layer_0(features[0])
        d0 = self.GeLU(self.decode_layer_0(d0))
        d1 = self.GeLU(self.decode_layer_1(d0))
        d2 = self.upsample_layer_1(d1)
        d2 = self.GeLU(self.decode_layer_2(d2))
        d3 = self.GeLU(self.decode_layer_3(d2))
        return self.output_layer(d3)


class DepthHead(nn.Module):
    def __init__(self):
        super(DepthHead, self).__init__()
        self.GeLU = nn.GELU()

        self.upsample_layer_0  = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(32, 256, 1)
        self.decode_layer_0    = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_1    = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_2   = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_3   = nn.Conv2d(128, 128, 3, 1, 1)

        self.output_layer = nn.Conv2d(128, 1, 3, 1, 1)

    def forward(self, neck, features):
        d0 = self.upsample_layer_0(neck) + self.skip_link_layer_0(features[0])
        d0 = self.GeLU(self.decode_layer_0(d0))
        d1 = self.GeLU(self.decode_layer_1(d0))
        d2 = self.upsample_layer_1(d1)
        d2 = self.GeLU(self.decode_layer_2(d2))
        d3 = self.GeLU(self.decode_layer_3(d2))
        return self.output_layer(d3)


class HydraNet(nn.Module):
    def __init__(self, num_drivable_classes=3):
        super(HydraNet, self).__init__()
        self.backbone         = Backbone(pretrained=True)
        self.drivable_context = SceneContext()
        self.depth_context    = DepthContext()
        self.neck             = SceneNeck()
        self.drivable_head    = DrivableAreaHead(num_classes=num_drivable_classes)
        self.depth_head       = DepthHead()

    def forward(self, image):
        features     = self.backbone(image)
        deep         = features[4]

        driv_ctx     = self.drivable_context(deep)
        driv_neck    = self.neck(driv_ctx, features)
        drivable_out = self.drivable_head(driv_neck, features)

        depth_ctx    = self.depth_context(deep)
        depth_neck   = self.neck(depth_ctx, features)
        depth_out    = self.depth_head(depth_neck, features)

        return drivable_out, depth_out


# ── Visualization ────────────────────────────────────────────────────

DRIVABLE_COLORS = np.array([
    [0, 0, 0],        # 0: not drivable
    [0, 180, 0],      # 1: direct (green)
    [0, 100, 255],    # 2: alternative (blue)
], dtype=np.uint8)


def overlay_drivable(frame, pred_mask, alpha=0.4):
    color_mask = DRIVABLE_COLORS[pred_mask]
    overlay = frame.copy()
    mask_active = pred_mask > 0
    overlay[mask_active] = cv2.addWeighted(
        frame[mask_active], 1 - alpha,
        color_mask[mask_active], alpha, 0
    )
    return overlay


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HydraNet video inference")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", default=None, help="Output video path (default: input_hydranet.mp4)")
    parser.add_argument("--fps", type=int, default=0, help="Process every Nth frame (0 = all)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base}_hydranet.mp4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = HydraNet().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {src_fps:.1f} fps, {total_frames} frames")

    out_fps = src_fps if args.fps == 0 else src_fps / args.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width * 2, height))

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

            if args.fps > 0 and (frame_idx % args.fps) != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            drivable_out, depth_out = model(input_tensor)

            drv = drivable_out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            drv = cv2.resize(drv, (width, height), interpolation=cv2.INTER_NEAREST)

            dep = depth_out.squeeze(0).squeeze(0).cpu().numpy()
            dep = cv2.resize(dep, (width, height), interpolation=cv2.INTER_LINEAR)

            left = overlay_drivable(frame, drv)
            right = cv2.applyColorMap((dep * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

            writer.write(np.concatenate([left, right], axis=1))
            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {processed} frames ({frame_idx}/{total_frames}) "
                      f"— {processed / elapsed:.1f} fps")

            if args.max_frames > 0 and processed >= args.max_frames:
                break

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\nDone! {processed} frames in {elapsed:.1f}s ({processed / elapsed:.1f} fps)")
    print(f"Output: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
