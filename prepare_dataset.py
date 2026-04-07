#!/usr/bin/env python3
"""
BDD100K HydraNet — Dataset Preparation Script
==============================================

Run this ONCE on Colab after downloading the Kaggle datasets.
Produces TWO clean zips ready to host on S3:

  bdd_hydranet_10k.zip   — 10k curated subset (8k train / 1k val / 1k test)
  bdd_hydranet_full.zip  — all 100k images (70k train / 10k val / 20k test)

Both share the same flat structure so notebooks work with either — just
change the download URL.

Full dataset (pre-built): https://drive.google.com/file/d/1tm41I4GvfmVxSxTFJEeoeKsn5RZ4eik7/view?usp=sharing

Usage (on Colab):
    !pip install -q kaggle
    !kaggle datasets download -d solesensei/solesensei_bdd100k -p data/ --unzip
    !kaggle datasets download -d thakurmayank5/depth-images-for-bdd100k-dataset -p data/bdd100k/labels/ --unzip
    !python prepare_dataset.py          # builds both zips
    !python prepare_dataset.py --10k    # 10k subset only (faster)
    !python prepare_dataset.py --full   # full 100k only

Output structure (identical for both):
    bdd_hydranet/
      train/
        images/     {id}.jpg        RGB images
        depth/      {id}.jpg        Pseudo-depth (grayscale)
        drivable/   {id}.png        Drivable area (0=bg, 1=direct, 2=alternative)
        lanes/      {id}.png        Lane markings (0=bg, 1=curb, 2=marking)
      val/
        ...
      test/
        ...
      labels_train.json             Full BDD100K annotations for student labs
      labels_val.json               (boxes, lanes, drivable, weather, scene, etc.)
      labels_test.json
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Raw data paths (from Kaggle downloads) ──────────────────────────
RAW_IMG_ROOT   = "data/bdd100k/bdd100k/images/100k"
RAW_DEPTH_ROOT = "data/bdd100k/labels/depth-images"
RAW_LABELS_DIR = "data/bdd100k_labels_release/bdd100k/labels"

# BDD100K original resolution
IMG_W, IMG_H = 1280, 720

# 10k subset split sizes
SUBSET_TRAIN = 8000
SUBSET_VAL   = 1000
SUBSET_TEST  = 1000

SEED = 42


# ── Rendering helpers ────────────────────────────────────────────────

def render_drivable_mask(labels):
    """Render drivable area polygons into a uint8 mask.

    0 = not drivable, 1 = directly drivable (ego lane), 2 = alternative
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
            area_type = label.get("attributes", {}).get("areaType", "")
            cls = 1 if area_type == "direct" else 2
            draw.polygon(points, fill=cls)
    return np.array(mask, dtype=np.uint8)


def render_lane_mask(labels):
    """Render lane polylines into a uint8 mask.

    0 = background, 1 = road curb/edge, 2 = lane marking
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
            lane_type = label.get("attributes", {}).get("laneType", "").lower()
            cls = 1 if "road curb" in lane_type else 2
            draw.line(points, fill=cls, width=8)
    return np.array(mask, dtype=np.uint8)


# ── Label loading ────────────────────────────────────────────────────

def load_labels_json(split):
    """Load BDD100K JSON labels for a split. Returns {filename: entry}."""
    if not os.path.isdir(RAW_LABELS_DIR):
        print(f"  [warn] Labels dir not found: {RAW_LABELS_DIR}")
        return {}

    # Try common naming patterns
    all_json = sorted(f for f in os.listdir(RAW_LABELS_DIR) if f.endswith(".json"))
    candidates = [
        f"bdd100k_labels_images_{split}.json",
        f"det_{split}.json",
        f"{split}.json",
    ] + [f for f in all_json if split in f]

    for fname in candidates:
        fpath = os.path.join(RAW_LABELS_DIR, fname)
        if os.path.exists(fpath):
            print(f"  Loading labels: {fname}")
            with open(fpath) as f:
                raw = json.load(f)
            return {entry["name"]: entry for entry in raw}

    print(f"  [warn] No labels JSON found for '{split}' in {RAW_LABELS_DIR}")
    print(f"         Available: {all_json}")
    return {}


# ── Image discovery ──────────────────────────────────────────────────

def discover_images(split):
    """Find all images in 100k/{split}/, handling subdirectories."""
    img_dir = os.path.join(RAW_IMG_ROOT, split)
    if not os.path.isdir(img_dir):
        return []
    return sorted(Path(img_dir).rglob("*.jpg"))


def find_depth(basename):
    """Search all depth splits for a matching depth map."""
    for split in ["train", "val", "test"]:
        p = os.path.join(RAW_DEPTH_ROOT, split, basename + ".jpg")
        if os.path.exists(p):
            return p
    return None


# ── Process images ───────────────────────────────────────────────────

def process_images(image_paths, labels_by_name, output_root, split):
    """Copy images + depth, render drivable + lane masks for a list of images."""
    dirs = {}
    for sub in ["images", "depth", "drivable", "lanes"]:
        d = os.path.join(output_root, split, sub)
        os.makedirs(d, exist_ok=True)
        dirs[sub] = d

    processed = 0
    skipped_no_depth = 0
    has_drivable = 0
    has_lanes = 0
    filtered_labels = []

    for img_path in image_paths:
        basename = img_path.stem
        img_name = img_path.name

        # Depth is required
        depth_path = find_depth(basename)
        if not depth_path:
            skipped_no_depth += 1
            continue

        # Copy image
        shutil.copy2(str(img_path), os.path.join(dirs["images"], f"{basename}.jpg"))

        # Copy depth
        shutil.copy2(depth_path, os.path.join(dirs["depth"], f"{basename}.jpg"))

        # Get JSON labels for this image
        entry = labels_by_name.get(img_name, {})
        img_labels = entry.get("labels", [])

        # Render drivable mask
        drivable = render_drivable_mask(img_labels)
        Image.fromarray(drivable).save(os.path.join(dirs["drivable"], f"{basename}.png"))
        if drivable.any():
            has_drivable += 1

        # Render lane mask
        lanes = render_lane_mask(img_labels)
        Image.fromarray(lanes).save(os.path.join(dirs["lanes"], f"{basename}.png"))
        if lanes.any():
            has_lanes += 1

        # Collect label entry
        if entry:
            filtered_labels.append(entry)
        else:
            filtered_labels.append({
                "name": img_name,
                "attributes": {},
                "labels": [],
            })

        processed += 1
        if processed % 2000 == 0:
            print(f"    ... {processed} images done")

    # Save filtered labels JSON
    labels_path = os.path.join(output_root, f"labels_{split}.json")
    with open(labels_path, "w") as f:
        json.dump(filtered_labels, f)

    print(f"  {split}: {processed} images processed"
          f" ({skipped_no_depth} skipped — no depth)")
    print(f"    drivable annotations: {has_drivable}/{processed}")
    print(f"    lane annotations:     {has_lanes}/{processed}")

    return processed


# ── Zip helper ───────────────────────────────────────────────────────

def make_zip(output_dir, zip_name):
    """Zip a directory into an archive."""
    print(f"\nCreating {zip_name} ...")
    total_files = 0
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(output_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, os.path.dirname(output_dir))
                zf.write(fpath, arcname)
                total_files += 1
                if total_files % 5000 == 0:
                    print(f"  ... {total_files} files zipped")

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"  {total_files} files, {size_mb:.1f} MB -> {zip_name}")
    return size_mb


# ── Print dataset summary ───────────────────────────────────────────

def print_summary(output_dir):
    print(f"\n  {os.path.basename(output_dir)}/")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"    {split}/")
        for sub in ["images", "depth", "drivable", "lanes"]:
            sub_dir = os.path.join(split_dir, sub)
            if os.path.isdir(sub_dir):
                count = len(os.listdir(sub_dir))
                sample = sorted(os.listdir(sub_dir))[:1]
                ext = sample[0].split(".")[-1] if sample else "?"
                print(f"      {sub:12s}  {count:6d} .{ext}")
    for split in ["train", "val", "test"]:
        lf = os.path.join(output_dir, f"labels_{split}.json")
        if os.path.exists(lf):
            kb = os.path.getsize(lf) / 1024
            print(f"    labels_{split}.json  ({kb:.0f} KB)")


# ── Build the full 100k dataset ─────────────────────────────────────

def build_full(output_dir="bdd_hydranet_full"):
    """Process ALL 100k images into a clean dataset."""
    print("\n" + "=" * 60)
    print("  Building FULL dataset (100k)")
    print("=" * 60)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    total = 0
    for split in ["train", "val", "test"]:
        print(f"\n  Discovering {split} images ...")
        image_paths = discover_images(split)
        if not image_paths:
            print(f"  [skip] No images found for {split}")
            continue
        print(f"  Found {len(image_paths)} source images")

        labels = load_labels_json(split)
        n = process_images(image_paths, labels, output_dir, split)
        total += n

    print(f"\n  TOTAL: {total} images")
    print_summary(output_dir)
    size = make_zip(output_dir, "bdd_hydranet_full.zip")
    return total, size


# ── Build the 10k subset ────────────────────────────────────────────

def build_10k(output_dir="bdd_hydranet_10k"):
    """Sample a balanced 10k subset from the 100k dataset."""
    print("\n" + "=" * 60)
    print(f"  Building 10k SUBSET ({SUBSET_TRAIN} train / "
          f"{SUBSET_VAL} val / {SUBSET_TEST} test)")
    print("=" * 60)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    random.seed(SEED)
    total = 0

    for split, target_n in [("train", SUBSET_TRAIN),
                             ("val", SUBSET_VAL),
                             ("test", SUBSET_TEST)]:
        print(f"\n  Discovering {split} images ...")
        image_paths = discover_images(split)
        if not image_paths:
            print(f"  [skip] No images found for {split}")
            continue
        print(f"  Found {len(image_paths)} source images, sampling {target_n}")

        # Filter to images that have depth (should be nearly all)
        with_depth = [p for p in image_paths if find_depth(p.stem)]
        print(f"  {len(with_depth)} have matching depth maps")

        # Sample
        if len(with_depth) <= target_n:
            sampled = with_depth
            print(f"  Using all {len(sampled)} (fewer than target {target_n})")
        else:
            sampled = sorted(random.sample(with_depth, target_n))

        labels = load_labels_json(split)
        n = process_images(sampled, labels, output_dir, split)
        total += n

    print(f"\n  TOTAL: {total} images")
    print_summary(output_dir)
    size = make_zip(output_dir, "bdd_hydranet_10k.zip")
    return total, size


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare BDD100K HydraNet dataset")
    parser.add_argument("--10k", dest="do_10k", action="store_true",
                        help="Build 10k subset only")
    parser.add_argument("--full", dest="do_full", action="store_true",
                        help="Build full 100k only")
    args = parser.parse_args()

    # Default: build both
    if not args.do_10k and not args.do_full:
        args.do_10k = True
        args.do_full = True

    print("=" * 60)
    print("  BDD100K HydraNet — Dataset Preparation")
    print("=" * 60)

    # Sanity checks
    missing = []
    for name, path in [
        ("Images (100k)", RAW_IMG_ROOT),
        ("Depth maps", RAW_DEPTH_ROOT),
        ("JSON labels", RAW_LABELS_DIR),
    ]:
        if not os.path.isdir(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("\n[ERROR] Missing required directories:")
        for m in missing:
            print(m)
        print("\nDownload from Kaggle first:")
        print("  kaggle datasets download -d solesensei/solesensei_bdd100k"
              " -p data/ --unzip")
        print("  kaggle datasets download -d thakurmayank5/"
              "depth-images-for-bdd100k-dataset -p data/bdd100k/labels/ --unzip")
        sys.exit(1)

    results = []

    if args.do_10k:
        n, size = build_10k()
        results.append(("bdd_hydranet_10k.zip", n, size))

    if args.do_full:
        n, size = build_full()
        results.append(("bdd_hydranet_full.zip", n, size))

    # Final summary
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    for name, n, size in results:
        print(f"  {name:30s}  {n:6d} images  {size:8.1f} MB")
    print("\nUpload to S3 and update the DATA_URL in your notebooks.")


if __name__ == "__main__":
    main()
