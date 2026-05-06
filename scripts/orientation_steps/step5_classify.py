#!/usr/bin/env python3
"""
Step 5 — Classification from boundary ratio.

Reads crops.json (from step 1), recomputes the boundary row for each crop
using the thresholds in params.py, classifies each cylinder, draws coloured
bounding boxes on the original images, and writes results.csv.

Bbox colours:
    green  = upright   (boundary_y_rel > UPRIGHT_THRESHOLD)
    orange = tilted
    red    = inverted  (boundary_y_rel < INVERTED_THRESHOLD)

Output:
    runs/orientation_steps/05_classify/annotated/<image_name>   (full images)
    runs/orientation_steps/05_classify/results.csv

Run:
    python3 step5_classify.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import params

WHITE_LOWER = np.array([params.WHITE_H_MIN, params.WHITE_S_MIN, params.WHITE_V_MIN], dtype=np.uint8)
WHITE_UPPER = np.array([params.WHITE_H_MAX, params.WHITE_S_MAX, params.WHITE_V_MAX], dtype=np.uint8)
BLUE_LOWER  = np.array([params.BLUE_H_MIN,  params.BLUE_S_MIN,  params.BLUE_V_MIN],  dtype=np.uint8)
BLUE_UPPER  = np.array([params.BLUE_H_MAX,  params.BLUE_S_MAX,  params.BLUE_V_MAX],  dtype=np.uint8)

ORIENT_COLORS = {
    "upright":  (  0, 200,   0),
    "tilted":   (  0, 165, 255),
    "inverted": (  0,   0, 220),
}
CSV_FIELDS = ["image_name", "cylinder_id", "orientation",
              "boundary_y_rel", "confidence", "blue_ratio"]


def boundary_from_crop(crop: np.ndarray):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bm  = cv2.inRange(hsv, BLUE_LOWER,  BLUE_UPPER)
    wm  = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    h, w  = crop.shape[:2]
    rows  = np.arange(h)
    blue_row   = np.array([np.count_nonzero(bm[r]) / max(w, 1) for r in rows])
    total_blue = blue_row.sum()

    bdy_rel    = (float((rows * blue_row).sum() / total_blue) / h
                  if total_blue > 0 else 0.5)
    blue_ratio = float(np.count_nonzero(bm)) / max(h * w, 1)
    return round(bdy_rel, 4), round(blue_ratio, 4)


def classify(bdy_rel: float) -> str:
    if bdy_rel > params.UPRIGHT_THRESHOLD:
        return "upright"
    if bdy_rel < params.INVERTED_THRESHOLD:
        return "inverted"
    return "tilted"


def draw_box(frame, x1, y1, x2, y2, label, bdy_rel, conf):
    color = ORIENT_COLORS[label]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}  {bdy_rel:.2f}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def main():
    crops_dir = Path(params.OUT_ROOT) / "01_crops"
    out_dir   = Path(params.OUT_ROOT) / "05_classify"
    ann_dir   = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((crops_dir / "crops.json").read_text())
    src_root = Path(params.SRC).expanduser()

    print(f"[INFO] {len(metadata)} crops  →  {out_dir}")
    print(f"  upright  : boundary_y_rel > {params.UPRIGHT_THRESHOLD}")
    print(f"  inverted : boundary_y_rel < {params.INVERTED_THRESHOLD}")

    by_image = defaultdict(list)
    for entry in metadata:
        by_image[entry["source_image"]].append(entry)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for source_img, entries in sorted(by_image.items()):
            frame = cv2.imread(str(src_root / source_img))
            if frame is None:
                continue

            counts = defaultdict(int)
            for entry in entries:
                crop = cv2.imread(str(crops_dir / entry["crop_file"]))
                if crop is None:
                    continue

                bdy_rel, blue_ratio = boundary_from_crop(crop)
                label = classify(bdy_rel)
                x1, y1, x2, y2 = entry["bbox"]
                draw_box(frame, x1, y1, x2, y2, label, bdy_rel, entry["confidence"])
                counts[label] += 1

                writer.writerow({
                    "image_name":     source_img,
                    "cylinder_id":    entry["cylinder_id"],
                    "orientation":    label,
                    "boundary_y_rel": bdy_rel,
                    "confidence":     entry["confidence"],
                    "blue_ratio":     blue_ratio,
                })

            cv2.imwrite(str(ann_dir / source_img), frame)
            summary = "  ".join(f"{v}x {k}" for k, v in counts.items())
            print(f"  {source_img}: {summary}")

    print(f"\n[DONE] annotated images  →  {ann_dir}/")
    print(f"       CSV               →  {csv_path}")


if __name__ == "__main__":
    main()
