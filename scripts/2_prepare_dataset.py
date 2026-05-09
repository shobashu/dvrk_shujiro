#!/usr/bin/env python3
"""
Prepare the YOLO training dataset from two zip files:
  - Task_Pad_cylinder_pegs.yolov8.zip  : manually labeled data (ground truth)
  - remaining_frames.zip               : raw frames to be auto-labeled with best.pt

STEP 1 — Unzip both archives into a working directory
STEP 2 — Collect labeled pairs from the yolov8 zip (Roboflow format)
STEP 3 — Auto-label raw frames using best.pt
STEP 4 — Merge all pairs and split into train / val / test
STEP 5 — Write config/dataset.yaml

Usage:
    python scripts/2_prepare_dataset.py \
        --labeled  Task_Pad_cylinder_pegs.yolov8.zip \
        --frames   remaining_frames.zip \
        --weights  models/dvrk_v1/weights/best.pt \
        --out      data/dataset
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO


# ── Classes (must match the labeled dataset export) ─────────────────────────────
CLASSES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def unzip(zip_path: Path, dest: Path) -> Path:
    out = dest / zip_path.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    return out


def find_label(img: Path) -> Path | None:
    """
    Locate the YOLO .txt label for an image.
    Handles Roboflow layout (…/images/x.jpg → …/labels/x.txt)
    and flat layout (x.jpg → x.txt in the same folder).
    """
    # Flat: sibling .txt
    sibling = img.with_suffix(".txt")
    if sibling.exists():
        return sibling

    # Roboflow: replace the 'images' segment with 'labels'
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            label_parts = parts.copy()
            label_parts[i] = "labels"
            label_parts[-1] = img.stem + ".txt"
            candidate = Path(*label_parts)
            if candidate.exists():
                return candidate

    return None


# ── STEP 1 — Unzip ──────────────────────────────────────────────────────────────

def step1_unzip(labeled_zip: Path, frames_zip: Path, work: Path):
    print("\nSTEP 1 — Unzipping archives")
    labeled_root = unzip(labeled_zip, work)
    frames_root  = unzip(frames_zip,  work)
    print(f"  Labeled data : {labeled_root}")
    print(f"  Raw frames   : {frames_root}")
    return labeled_root, frames_root


# ── STEP 2 — Collect labeled pairs from the yolov8 zip ─────────────────────────

def step2_collect_labeled(labeled_root: Path) -> list[tuple[Path, Path]]:
    print("\nSTEP 2 — Collecting labeled pairs")
    pairs = []
    for img in sorted(labeled_root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = find_label(img)
        if label:
            pairs.append((img, label))
        else:
            print(f"  [WARN] No label for {img.name} — skipping")
    print(f"  {len(pairs)} labeled pairs found")
    return pairs


# ── STEP 3 — Auto-label raw frames ─────────────────────────────────────────────

def step3_auto_label(
    frames_root: Path,
    weights: Path,
    work: Path,
    conf: float,
) -> list[tuple[Path, Path]]:
    print(f"\nSTEP 3 — Auto-labeling raw frames  (conf ≥ {conf})")

    images = sorted([p for p in frames_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    print(f"  {len(images)} raw frames found")

    model = YOLO(str(weights))
    predict_out = work / "predict"

    model.predict(
        source=str(frames_root),
        save_txt=True,
        save_conf=False,
        conf=conf,
        project=str(predict_out),
        name="run",
        exist_ok=True,
        verbose=False,
    )

    # Index generated labels by stem (handles any sub-directory structure)
    labels_by_stem = {
        p.stem: p
        for p in (predict_out / "run" / "labels").rglob("*.txt")
    }

    pairs = []
    detected = 0
    empty_dir = predict_out / "run" / "labels"
    empty_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        label = labels_by_stem.get(img.stem)
        if label is None:
            # No detection → empty label (valid YOLO background frame)
            label = empty_dir / (img.stem + ".txt")
            label.write_text("")
        else:
            detected += 1
        pairs.append((img, label))

    print(f"  {detected}/{len(images)} frames had at least one detection")
    return pairs


# ── STEP 4 — Merge and split ────────────────────────────────────────────────────

def step4_merge_and_split(
    labeled_pairs: list[tuple[Path, Path]],
    auto_pairs:    list[tuple[Path, Path]],
    dst: Path,
    split: tuple[float, float, float],
    seed: int,
) -> dict:
    all_pairs = labeled_pairs + auto_pairs
    print(f"\nSTEP 4 — Merging and splitting")
    print(f"  {len(labeled_pairs)} labeled  +  {len(auto_pairs)} auto-labeled  =  {len(all_pairs)} total")
    print(f"  Split: {split[0]:.0%} train / {split[1]:.0%} val / {split[2]:.0%} test  (seed={seed})")

    random.seed(seed)
    random.shuffle(all_pairs)

    n       = len(all_pairs)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])

    splits = {
        "train": all_pairs[:n_train],
        "val":   all_pairs[n_train : n_train + n_val],
        "test":  all_pairs[n_train + n_val :],
    }

    for split_name, pairs in splits.items():
        img_dir = dst / "images" / split_name
        lbl_dir = dst / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in pairs:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / (img.stem + ".txt"))

        print(f"  {split_name:5s}: {len(pairs)} pairs  →  {img_dir}")

    return splits


# ── STEP 5 — Write config/dataset.yaml ─────────────────────────────────────────

def step5_write_yaml(dst: Path):
    print("\nSTEP 5 — Writing config/dataset.yaml")
    yaml_path = Path("config") / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {list(CLASSES.values())}\n"
    )
    print(f"  Written: {yaml_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled",  required=True,
                   help="Path to Task_Pad_cylinder_pegs.yolov8.zip")
    p.add_argument("--frames",   required=True,
                   help="Path to remaining_frames.zip")
    p.add_argument("--weights",  default="models/dvrk_v1/weights/best.pt",
                   help="YOLO weights used for auto-labeling")
    p.add_argument("--out",      default="data/dataset",
                   help="Output dataset root")
    p.add_argument("--work",     default="data/work",
                   help="Working directory for unzipped files and predictions")
    p.add_argument("--conf",     type=float, default=0.3,
                   help="Confidence threshold for auto-labeling (default: 0.3)")
    p.add_argument("--split",    nargs=3, type=float, default=[0.70, 0.20, 0.10],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train/val/test ratios (must sum to 1)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.work)
    out  = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)

    labeled_root, frames_root = step1_unzip(
        Path(args.labeled), Path(args.frames), work
    )
    labeled_pairs = step2_collect_labeled(labeled_root)
    auto_pairs    = step3_auto_label(frames_root, Path(args.weights), work, args.conf)
    splits        = step4_merge_and_split(
        labeled_pairs, auto_pairs, out, tuple(args.split), args.seed
    )
    step5_write_yaml(out)

    total = sum(len(v) for v in splits.values())
    print(f"\nDone — {total} pairs ready for training at {out}")


if __name__ == "__main__":
    main()
