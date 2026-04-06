#!/usr/bin/env python3
"""
BDD100K HydraNet — Dataset Preparation Script
==============================================

Run this ONCE on Colab after downloading the Kaggle datasets.
It produces a clean bdd_hydranet.zip ready to host on S3.

Usage (on Colab):
    !pip install -q kaggle
    # Upload kaggle.json, then:
    !kaggle datasets download -d solesensei/solesensei_bdd100k -p data/ --unzip
    !kaggle datasets download -d thakurmayank5/depth-images-for-bdd100k-dataset -p data/bdd100k/labels/ --unzip
    !python prepare_dataset.py

Output:
    bdd_hydranet/
      train/
        images/     {id}.jpg        RGB images
        seg/        {id}.png        Semantic seg masks (19 Cityscapes classes)
        depth/      {id}.jpg        Pseudo-depth (Depth Anything v2, grayscale)
        lanes/      {id}.png        Lane markings rendered from poly2d annotations
        drivable/   {id}.png        Drivable area masks (0=bg, 1=direct, 2=alternative)
      val/
        images/     ...
        seg/        ...
        depth/      ...
        lanes/      ...
        drivable/   ...
      labels_train.json             Full BDD100K annotations (boxes, attributes, etc.)
      labels_val.json               for student lab challenges
    bdd_hydranet.zip                Ready to upload to S3!
"""

import os
import sys
import json
import shutil
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── Raw data paths (from Kaggle downloads) ──────────────────────────
RAW_IMG_ROOT   = "data/bdd100k/bdd100k/images/10k"
RAW_SEG_ROOT   = "data/bdd100k_seg/bdd100k/seg/labels"
RAW_DEPTH_ROOT = "data/bdd100k/labels/depth-images"
RAW_LABELS_DIR = "data/bdd100k_labels_release/bdd100k/labels"

OUTPUT_DIR = "bdd_hydranet"
ZIP_NAME   = "bdd_hydranet.zip"

# BDD100K original resolution
IMG_W, IMG_H = 1280, 720

# Depth maps are split differently from the 10k images — search all splits
DEPTH_SPLITS = ["train", "val", "test"]


# ── Helpers ──────────────────────────────────────────────────────────

def find_depth(basename):
    """Search all depth splits for a matching depth map."""
    for split in DEPTH_SPLITS:
        p = os.path.join(RAW_DEPTH_ROOT, split, basename + ".jpg")
        if os.path.exists(p):
            return p
    return None


def find_seg(basename):
    """Search train/val splits for a matching seg mask."""
    for split in ["train", "val"]:
        p = os.path.join(RAW_SEG_ROOT, split, basename + "_train_id.png")
        if os.path.exists(p):
            return p
    return None


def render_lane_mask(labels):
    """Render lane polylines from BDD100K annotations into a uint8 mask.

    Classes:
        0 = background
        1 = road curb / edge
        2 = lane marking (all other lane types)
    """
    mask = Image.new("L", (IMG_W, IMG_H), 0)
    draw = ImageDraw.Draw(mask)

    for label in labels:
        if label.get("category") != "lane":
            continue
        for poly in label.get("poly2d", []):
            verts = poly.get("vertices", [])
            if len(verts) < 2:
                continue
            points = [(int(round(v[0])), int(round(v[1]))) for v in verts]

            attrs = label.get("attributes", {})
            lane_type = attrs.get("laneType", "").lower()
            cls = 1 if "road curb" in lane_type else 2

            # Draw with width=8 so lines survive downscaling to 256x512
            draw.line(points, fill=cls, width=8)

    return np.array(mask, dtype=np.uint8)


def render_drivable_mask(labels):
    """Render drivable area polygons from BDD100K annotations into a uint8 mask.

    Classes:
        0 = not drivable (background)
        1 = directly drivable (ego lane)
        2 = alternatively drivable (other lanes)
    """
    mask = Image.new("L", (IMG_W, IMG_H), 0)
    draw = ImageDraw.Draw(mask)

    for label in labels:
        if label.get("category") != "drivable area":
            continue
        for poly in label.get("poly2d", []):
            verts = poly.get("vertices", [])
            if len(verts) < 3:
                continue
            points = [(int(round(v[0])), int(round(v[1]))) for v in verts]

            attrs = label.get("attributes", {})
            area_type = attrs.get("areaType", "")
            cls = 1 if area_type == "direct" else 2

            draw.polygon(points, fill=cls)

    return np.array(mask, dtype=np.uint8)


def load_labels_json(split):
    """Find and load the BDD100K detection labels JSON for a split.

    Returns dict: image_filename -> label entry
    """
    if not os.path.isdir(RAW_LABELS_DIR):
        print(f"  [warn] Labels dir not found: {RAW_LABELS_DIR}")
        return {}

    # Try common naming patterns
    candidates = [
        f"bdd100k_labels_images_{split}.json",
        f"det_{split}.json",
        f"{split}.json",
    ]
    # Also try any JSON file with the split name in it
    all_json = [f for f in os.listdir(RAW_LABELS_DIR) if f.endswith(".json")]
    for f in all_json:
        if split in f and f not in candidates:
            candidates.append(f)

    for fname in candidates:
        fpath = os.path.join(RAW_LABELS_DIR, fname)
        if os.path.exists(fpath):
            print(f"  Loading labels: {fname}")
            with open(fpath) as f:
                raw = json.load(f)
            return {entry["name"]: entry for entry in raw}

    print(f"  [warn] No labels JSON found for split '{split}'")
    print(f"         Looked in: {RAW_LABELS_DIR}")
    print(f"         Available: {all_json}")
    return {}


# ── Process one split ────────────────────────────────────────────────

def process_split(split):
    """Find matched images and copy/render all labels into clean structure."""
    img_dir = os.path.join(RAW_IMG_ROOT, split)
    if not os.path.isdir(img_dir):
        print(f"  [skip] Image dir not found: {img_dir}")
        return 0

    # Create output directories
    dirs = {}
    for sub in ["images", "seg", "depth", "lanes", "drivable"]:
        d = os.path.join(OUTPUT_DIR, split, sub)
        os.makedirs(d, exist_ok=True)
        dirs[sub] = d

    # Load JSON labels (for lanes, drivable, detection boxes, attributes)
    labels_by_name = load_labels_json(split)

    # List all images in this split
    all_images = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    print(f"  Source images: {len(all_images)}")

    matched = 0
    filtered_labels = []
    lane_count = 0
    drivable_count = 0

    for i, img_name in enumerate(all_images):
        basename = os.path.splitext(img_name)[0]

        # Check seg + depth exist
        seg_path = find_seg(basename)
        depth_path = find_depth(basename)
        if not seg_path or not depth_path:
            continue

        # ── Copy image ──
        shutil.copy2(
            os.path.join(img_dir, img_name),
            os.path.join(dirs["images"], f"{basename}.jpg"),
        )

        # ── Copy seg mask (strip _train_id suffix in source, save as {id}.png) ──
        shutil.copy2(seg_path, os.path.join(dirs["seg"], f"{basename}.png"))

        # ── Copy depth map ──
        shutil.copy2(depth_path, os.path.join(dirs["depth"], f"{basename}.jpg"))

        # ── Render lane + drivable masks from JSON ──
        entry = labels_by_name.get(img_name, {})
        img_labels = entry.get("labels", [])

        lane_mask = render_lane_mask(img_labels)
        Image.fromarray(lane_mask).save(
            os.path.join(dirs["lanes"], f"{basename}.png")
        )
        if lane_mask.any():
            lane_count += 1

        drivable_mask = render_drivable_mask(img_labels)
        Image.fromarray(drivable_mask).save(
            os.path.join(dirs["drivable"], f"{basename}.png")
        )
        if drivable_mask.any():
            drivable_count += 1

        # ── Collect filtered label entry ──
        if entry:
            filtered_labels.append(entry)
        else:
            # Minimal entry for images without JSON labels
            filtered_labels.append({
                "name": img_name,
                "attributes": {},
                "labels": [],
            })

        matched += 1
        if (matched % 500) == 0:
            print(f"    ... processed {matched} images")

    # Save filtered labels JSON
    labels_path = os.path.join(OUTPUT_DIR, f"labels_{split}.json")
    with open(labels_path, "w") as f:
        json.dump(filtered_labels, f)

    print(f"  Matched: {matched} / {len(all_images)} images")
    print(f"  With lane annotations:     {lane_count}")
    print(f"  With drivable annotations: {drivable_count}")
    print(f"  Labels JSON: {labels_path} ({len(filtered_labels)} entries)")

    return matched


# ── Zip it up ────────────────────────────────────────────────────────

def make_zip():
    """Create a zip archive of the clean dataset."""
    print(f"\nCreating {ZIP_NAME} ...")
    total_files = 0
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, ".")  # keeps bdd_hydranet/ prefix
                zf.write(fpath, arcname)
                total_files += 1
                if total_files % 2000 == 0:
                    print(f"  ... zipped {total_files} files")

    size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
    print(f"  Done: {total_files} files, {size_mb:.1f} MB")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BDD100K HydraNet — Dataset Preparation")
    print("=" * 60)

    # Sanity checks
    missing = []
    for name, path in [
        ("Images (10k)", RAW_IMG_ROOT),
        ("Seg masks", RAW_SEG_ROOT),
        ("Depth maps", RAW_DEPTH_ROOT),
    ]:
        if not os.path.isdir(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("\n[ERROR] Missing required data directories:")
        for m in missing:
            print(m)
        print("\nMake sure you've downloaded both Kaggle datasets:")
        print("  kaggle datasets download -d solesensei/solesensei_bdd100k -p data/ --unzip")
        print("  kaggle datasets download -d thakurmayank5/depth-images-for-bdd100k-dataset -p data/bdd100k/labels/ --unzip")
        sys.exit(1)

    # Clean output
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving old {OUTPUT_DIR}/ ...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Process each split
    total = 0
    for split in ["train", "val"]:
        print(f"\n{'─' * 40}")
        print(f"  Processing: {split}")
        print(f"{'─' * 40}")
        n = process_split(split)
        total += n

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {total} matched image-label sets")
    print(f"{'=' * 60}")

    # Show final structure
    print(f"\n{OUTPUT_DIR}/")
    for split in ["train", "val"]:
        split_dir = os.path.join(OUTPUT_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"  {split}/")
        for sub in ["images", "seg", "depth", "lanes", "drivable"]:
            sub_dir = os.path.join(split_dir, sub)
            if os.path.isdir(sub_dir):
                count = len(os.listdir(sub_dir))
                sample = sorted(os.listdir(sub_dir))[:1]
                print(f"    {sub:12s}  {count:5d} files   e.g. {sample[0] if sample else '(empty)'}")
    for f in ["labels_train.json", "labels_val.json"]:
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(fp):
            size_kb = os.path.getsize(fp) / 1024
            print(f"  {f}  ({size_kb:.0f} KB)")

    # Create zip
    make_zip()

    print(f"\n{'=' * 60}")
    print(f"  Done! Upload {ZIP_NAME} to your S3 bucket.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
