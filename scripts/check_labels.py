#!/usr/bin/env python3
"""
scripts/check_labels.py

Opens each image with its bounding boxes drawn so you can visually
verify annotations are correct before training.

Controls (while image is open):
    SPACE or RIGHT arrow  →  next image
    LEFT arrow            →  previous image
    Q or ESC              →  quit

Usage:
    # Check train split
    python scripts/check_labels.py --images data/dataset/images/train --labels data/dataset/labels/train

    # Check only images that had polygons fixed (have a .bak file)
    python scripts/check_labels.py --images data/dataset/images/train --labels data/dataset/labels/train --fixed_only
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


CLASS_NAMES  = {0: "cylinder", 1: "peg_inactive", 2: "peg_lit", 3: "peg_lit"}
CLASS_COLORS = {
    0: (40,  160, 220),   # cylinder      — orange (BGR)
    1: (120, 120, 120),   # peg_inactive  — gray
    2: (200,  80,  30),   # peg_lit       — blue
    3: (200,  80,  30),   # peg_lit alt   — blue
}


def draw_boxes(image: np.ndarray, label_path: Path) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]

    if not label_path.exists():
        cv2.putText(out, "NO LABEL FILE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    lines = label_path.read_text().strip().splitlines()
    if not lines:
        cv2.putText(out, "EMPTY LABEL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return out

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            # Still corrupted — highlight in red
            cv2.putText(out, f"BAD LINE: {line[:40]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            continue

        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        # Convert normalized → pixel coords
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    return out


def check_labels(images_dir: str, labels_dir: str, fixed_only: bool = False):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_files = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )

    if fixed_only:
        # Only show images whose label file has a .bak (was modified)
        image_files = [
            img for img in image_files
            if (labels_dir / (img.stem + ".txt.bak")).exists()
        ]
        print(f"[INFO] Showing {len(image_files)} fixed images only")
    else:
        print(f"[INFO] {len(image_files)} images to review")

    if not image_files:
        print("[ERROR] No images found.")
        return

    print("\nControls:")
    print("  SPACE or → = next image")
    print("  ← = previous image")
    print("  Q or ESC = quit\n")

    idx = 0
    while True:
        img_path = image_files[idx]
        lbl_path = labels_dir / (img_path.stem + ".txt")

        frame = cv2.imread(str(img_path))
        if frame is None:
            idx = (idx + 1) % len(image_files)
            continue

        annotated = draw_boxes(frame, lbl_path)

        # Info overlay
        info = f"[{idx+1}/{len(image_files)}]  {img_path.name}"
        cv2.putText(annotated, info, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Was this file fixed?
        if (labels_dir / (img_path.stem + ".txt.bak")).exists():
            cv2.putText(annotated, "POLYGON FIXED", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1, cv2.LINE_AA)

        cv2.imshow("Label Check  |  SPACE=next  LEFT=prev  Q=quit", annotated)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):        # Q or ESC
            break
        elif key in (ord(' '), 83, 3):   # SPACE or RIGHT arrow
            idx = min(idx + 1, len(image_files) - 1)
        elif key in (81, 2):             # LEFT arrow
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()
    print(f"\n[DONE] Reviewed {idx+1} images.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images",  default="data/dataset/images/train")
    p.add_argument("--labels",  default="data/dataset/labels/train")
    p.add_argument("--fixed_only", action="store_true",
                   help="Only show images whose labels were modified by fix_labels.py")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_labels(args.images, args.labels, args.fixed_only)
