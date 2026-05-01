#!/usr/bin/env python3
"""
scripts/fix_labels.py

Fixes corrupted YOLO label files that contain polygon annotations
instead of bounding boxes. This happens when Roboflow exports SAM2
polygon masks without converting them to bbox format.

Normal YOLO line (5 values):
    3 0.347 0.322 0.032 0.162

Corrupted polygon line (many values):
    3 0.653 0.283 0.665 0.355 0.669 0.349 ...

Fix: compute the min/max of polygon x,y points → convert to bbox.

Usage:
    python scripts/fix_labels.py --labels data/dataset/labels/train
    python scripts/fix_labels.py --labels data/dataset/labels/val
    python scripts/fix_labels.py --labels data/dataset/labels/test
"""

import argparse
import shutil
from pathlib import Path


def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """
    Convert flat list of polygon points [x1,y1,x2,y2,...] to YOLO bbox.
    Returns (cx, cy, w, h) normalized.
    """
    xs = values[0::2]   # every even index = x coordinate
    ys = values[1::2]   # every odd index  = y coordinate

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min

    return cx, cy, w, h


def fix_label_file(label_path: Path, dry_run: bool = False) -> dict:
    """
    Fix a single label file. Returns stats about what was changed.
    """
    lines    = label_path.read_text().strip().splitlines()
    fixed    = []
    n_fixed  = 0
    n_ok     = 0
    n_skip   = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Normal YOLO bbox: exactly 5 values
        if len(parts) == 5:
            fixed.append(line)
            n_ok += 1
            continue

        # Too few values — malformed, skip
        if len(parts) < 5:
            print(f"    [SKIP] Malformed line (only {len(parts)} values): {line[:60]}")
            n_skip += 1
            continue

        # Polygon: class_id + even number of coordinate pairs
        cls_id = parts[0]
        coords = parts[1:]

        if len(coords) % 2 != 0:
            print(f"    [SKIP] Odd number of coordinates: {line[:60]}")
            n_skip += 1
            continue

        try:
            coord_floats = [float(v) for v in coords]
            cx, cy, w, h = polygon_to_bbox(coord_floats)

            # Clamp to [0, 1] just in case
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            new_line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            fixed.append(new_line)
            n_fixed += 1

        except ValueError as e:
            print(f"    [ERROR] Could not parse: {line[:60]}  ({e})")
            n_skip += 1

    # Write fixed file
    if not dry_run and n_fixed > 0:
        # Backup original
        backup = label_path.with_suffix(".txt.bak")
        shutil.copy2(label_path, backup)
        label_path.write_text("\n".join(fixed) + "\n")

    return {"ok": n_ok, "fixed": n_fixed, "skipped": n_skip}


def fix_labels(labels_dir: str, dry_run: bool = False):
    labels = Path(labels_dir)

    if not labels.exists():
        print(f"[ERROR] Directory not found: {labels}")
        return

    label_files = sorted(labels.glob("*.txt"))
    print(f"[INFO] Found {len(label_files)} label files in {labels}")

    if dry_run:
        print("[INFO] DRY RUN — no files will be modified\n")

    total_ok    = 0
    total_fixed = 0
    total_skip  = 0
    files_with_issues = []

    for lf in label_files:
        stats = fix_label_file(lf, dry_run=dry_run)
        total_ok    += stats["ok"]
        total_fixed += stats["fixed"]
        total_skip  += stats["skipped"]

        if stats["fixed"] > 0 or stats["skipped"] > 0:
            files_with_issues.append((lf.name, stats))
            print(f"  {lf.name}: {stats['fixed']} polygon(s) converted, "
                  f"{stats['skipped']} line(s) skipped")

    print(f"\n{'─'*50}")
    print(f"  Total label files : {len(label_files)}")
    print(f"  Files with issues : {len(files_with_issues)}")
    print(f"  Normal bbox lines : {total_ok}")
    print(f"  Polygons fixed    : {total_fixed}")
    print(f"  Lines skipped     : {total_skip}")
    print(f"{'─'*50}")

    if total_fixed > 0 and not dry_run:
        print(f"\n  Original files backed up as .txt.bak")
        print(f"  Run again with --dry_run to verify before committing changes")

    if total_fixed == 0:
        print(f"\n  No issues found — all label files are clean.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fix polygon annotations in YOLO label files"
    )
    p.add_argument(
        "--labels", required=True,
        help="Path to labels folder (e.g. data/dataset/labels/train)"
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Preview fixes without modifying any files"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_labels(args.labels, dry_run=args.dry_run)
