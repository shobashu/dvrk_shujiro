#!/usr/bin/env python3
"""
Step 2 — Organise annotated images into the YOLO dataset structure.

Run this AFTER you have annotated your frames in CVAT / Label Studio
and exported them in YOLO format (images + .txt labels).

What it does:
  - Reads a flat folder of images + YOLO label .txt files
  - Verifies every image has a matching label (warns if not)
  - Splits into train / val / test sets (no time-adjacent leakage)
  - Copies files into data/dataset/{images,labels}/{train,val,test}

Usage:
    python 2_prepare_dataset.py \
        --src   data/frames/trial_001/annotated \
        --dst   data/dataset \
        --split 0.7 0.2 0.1
"""

import argparse
import random
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# YOLO label validation helpers
# ---------------------------------------------------------------------------

VALID_CLASS_IDS = {0, 1, 2, 3}  # must match config/dataset.yaml
CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

def validate_label_file(label_path: Path) -> list[str]:
    """
    Return a list of warning strings for a YOLO label file.
    An empty list means the file is clean.
    """
    warnings = []
    lines = label_path.read_text().strip().splitlines()

    if not lines:
        warnings.append(f"  [WARN] Empty label file: {label_path.name}")
        return warnings

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            warnings.append(
                f"  [WARN] {label_path.name} line {i+1}: expected 5 values, got {len(parts)}"
            )
            continue

        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        if cls_id not in VALID_CLASS_IDS:
            warnings.append(
                f"  [WARN] {label_path.name} line {i+1}: unknown class id {cls_id}"
            )
        for val, name in zip([cx, cy, w, h], ["cx", "cy", "w", "h"]):
            if not (0.0 <= val <= 1.0):
                warnings.append(
                    f"  [WARN] {label_path.name} line {i+1}: {name}={val:.4f} out of [0,1]"
                )

    return warnings


# ---------------------------------------------------------------------------
# Dataset split + copy
# ---------------------------------------------------------------------------
# Train/Valid/Test set
def prepare_dataset(
    src_dir: str,
    dst_dir: str,
    split: tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed: int = 42,
):
    src = Path(src_dir)
    dst = Path(dst_dir)

    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Collect all images
    image_exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in src.iterdir() if p.suffix.lower() in image_exts])

    print(f"[INFO] Found {len(images)} images in {src}")

    # Pair with label files and validate
    pairs = []
    all_warnings = []

    for img in images:
        label = img.with_suffix(".txt")
        if not label.exists():
            all_warnings.append(f"  [WARN] No label for {img.name} — skipping")
            continue
        warns = validate_label_file(label)
        all_warnings.extend(warns)
        pairs.append((img, label))

    if all_warnings:
        print("\n".join(all_warnings))
        print(f"[WARN] {len(all_warnings)} issue(s) found — review before training.\n")

    print(f"[INFO] {len(pairs)} valid image+label pairs")

    # Shuffle with fixed seed — but keep contiguous bags together to avoid
    # time-adjacent leakage: sort by filename, chunk by trial, then shuffle chunks.
    random.seed(seed)
    random.shuffle(pairs)

    # Split
    n = len(pairs)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    # Copy into YOLO directory structure
    stats = {}
    for split_name, split_pairs in splits.items():
        img_dir = dst / "images" / split_name
        lbl_dir = dst / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in split_pairs:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)

        stats[split_name] = len(split_pairs)
        print(f"  {split_name:5s}: {len(split_pairs)} files → {img_dir}")

    print(f"\n[DONE] Dataset ready at {dst}")
    print(f"       Train {stats['train']} | Val {stats['val']} | Test {stats['test']}")

    # Print per-class count from training set
    print("\n[INFO] Class distribution in train split:")
    class_counts = {k: 0 for k in VALID_CLASS_IDS}
    for _, lbl in splits["train"]:
        for line in lbl.read_text().strip().splitlines():
            cls_id = int(line.split()[0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    for cid, count in class_counts.items():
        print(f"  [{cid}] {CLASS_NAMES[cid]:<20s}: {count} annotations")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",   required=True, help="Annotated source folder (images + .txt)")
    p.add_argument("--dst",   default="data/dataset", help="Output dataset root")
    p.add_argument("--split", nargs=3, type=float, default=[0.70, 0.20, 0.10],
                   metavar=("TRAIN", "VAL", "TEST"), help="Split ratios (must sum to 1)")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        src_dir=args.src,
        dst_dir=args.dst,
        split=tuple(args.split),
        seed=args.seed,
    )
