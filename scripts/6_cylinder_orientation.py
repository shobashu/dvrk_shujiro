#!/usr/bin/env python3
"""
Step 6 — Cylinder orientation analysis via HSV color segmentation.

For each detected cylinder bbox crop, segments white (top) and blue (bottom)
regions in HSV, computes the axis angle relative to vertical, and labels each
detection as upright / tilted / inverted.

Bbox colors: green = upright, orange = tilted, red = inverted.

Usage:
    # Folder of images
    python3 6_cylinder_orientation.py \
        --src ~/data_shujiro/Task_Pad_Cylinder_Pegs_yolov8/train/images

    # Custom weights / output
    python3 6_cylinder_orientation.py \
        --src ~/data_shujiro/.../train/images \
        --weights models/best.pt \
        --output runs/orientation
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Reuse constants from detect_node / 4_infer ──────────────────────────────

CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

CYLINDER_CLASS_ID = 0
DEFAULT_WEIGHTS = "models/best.pt"

# Orientation thresholds (degrees from vertical; 0° = upright)
UPRIGHT_MAX  = 30.0
INVERTED_MIN = 150.0

# Minimum pixels required in a mask before trusting its centroid
MIN_MASK_PIXELS = 50

# HSV ranges (OpenCV: H 0–179, S 0–255, V 0–255)
WHITE_LOWER = np.array([  0,   0, 180], dtype=np.uint8)
WHITE_UPPER = np.array([179,  55, 255], dtype=np.uint8)

BLUE_LOWER  = np.array([ 90,  80,  40], dtype=np.uint8)
BLUE_UPPER  = np.array([130, 255, 255], dtype=np.uint8)

# Bbox colors per orientation label (BGR)
ORIENT_COLORS = {
    "upright":  (  0, 200,   0),  # green
    "tilted":   (  0, 165, 255),  # orange
    "inverted": (  0,   0, 220),  # red
}


# ── Orientation logic ────────────────────────────────────────────────────────

def _mask_centroid(mask: np.ndarray):
    """Return (x, y) centroid of a binary mask, or None if the mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def analyse_orientation(crop_bgr: np.ndarray):
    """
    Compute cylinder orientation from a bbox crop.

    Strategy:
      1. Blue mask centroid is reliable — it is dense and spatially concentrated.
         Its vertical position (blue_y_rel) gives a coarse upright/inverted signal.
      2. White mask is bimodal: white cylinder cap at one end PLUS white peg-board
         background at the other end. Using the full white centroid blends both and
         lands near blue, producing a meaningless angle.
         Fix: restrict the white search to the crop half OPPOSITE to blue.
         This isolates the cap and discards the background contamination.
      3. Angle = atan2(dx, -dy) on the (white_cap → blue) vector; 0° = upright.

    Returns:
        label     : "upright" | "tilted" | "inverted"
        angle_deg : float — 0° upright, ±180° inverted
        blue_ratio: float — blue pixel fraction of crop area
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    blue_mask  = cv2.inRange(hsv, BLUE_LOWER,  BLUE_UPPER)

    area       = crop_bgr.shape[0] * crop_bgr.shape[1]
    blue_ratio = float(np.count_nonzero(blue_mask)) / max(area, 1)
    crop_h     = crop_bgr.shape[0]

    blue_c = _mask_centroid(blue_mask)

    if blue_c is None or np.count_nonzero(blue_mask) < MIN_MASK_PIXELS:
        return "upright", 0.0, round(blue_ratio, 4)

    blue_y_rel = blue_c[1] / crop_h   # 0 = top of crop, 1 = bottom
    split_y    = int(blue_c[1])

    if blue_y_rel >= 0.5:
        # Blue in lower half → upright; look for white cap above blue centroid only
        white_cap_mask = white_mask[:split_y, :]
        white_cap_c    = _mask_centroid(white_cap_mask)
        if white_cap_c is None or np.count_nonzero(white_cap_mask) < MIN_MASK_PIXELS:
            label = "upright" if blue_y_rel > 0.6 else "tilted"
            return label, 0.0, round(blue_ratio, 4)
        white_c = white_cap_c                              # y already in [0, split_y]
    else:
        # Blue in upper half → inverted; look for white cap below blue centroid only
        white_cap_mask = white_mask[split_y:, :]
        white_cap_c    = _mask_centroid(white_cap_mask)
        if white_cap_c is None or np.count_nonzero(white_cap_mask) < MIN_MASK_PIXELS:
            label = "inverted" if blue_y_rel < 0.4 else "tilted"
            return label, 180.0, round(blue_ratio, 4)
        white_c = (white_cap_c[0], white_cap_c[1] + split_y)  # restore absolute y

    # atan2(dx, -dy): 0° = white directly above blue = upright
    dx = white_c[0] - blue_c[0]
    dy = white_c[1] - blue_c[1]   # negative when white is above blue (upright)
    angle_deg = math.degrees(math.atan2(dx, -dy))

    abs_angle = abs(angle_deg)
    if abs_angle < UPRIGHT_MAX:
        label = "upright"
    elif abs_angle > INVERTED_MIN:
        label = "inverted"
    else:
        label = "tilted"

    return label, round(angle_deg, 1), round(blue_ratio, 4)


# ── Drawing (mirrors detect_node._draw_detections style) ────────────────────

def draw_cylinder_box(frame, x1, y1, x2, y2, label, angle_deg, conf):
    color = ORIENT_COLORS[label]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}  {angle_deg:+.0f}deg  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
        (20, 20, 20), 1, cv2.LINE_AA,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(src, weights, conf, iou, imgsz, output_dir):
    model    = YOLO(weights)
    src_path = Path(src).expanduser()
    out_path = Path(output_dir)
    img_out  = out_path / "annotated"
    img_out.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if src_path.is_dir():
        sources = sorted(p for p in src_path.iterdir() if p.suffix.lower() in image_exts)
    else:
        sources = [src_path]

    print(f"[INFO] {len(sources)} images  |  weights={weights}  |  out={out_path}")

    csv_path = out_path / "orientation.csv"
    fields   = ["image_name", "cylinder_id", "orientation", "angle_deg", "confidence", "blue_ratio"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for img_path in sources:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [WARN] cannot read {img_path.name}")
                continue

            results = model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )[0]

            annotated  = frame.copy()
            cyl_count  = 0
            cyl_summary = []

            for box in results.boxes:
                if int(box.cls[0]) != CYLINDER_CLASS_ID:
                    continue

                det_conf        = round(float(box.conf[0]), 4)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                label, angle_deg, blue_ratio = analyse_orientation(crop)
                draw_cylinder_box(annotated, x1, y1, x2, y2, label, angle_deg, det_conf)

                writer.writerow({
                    "image_name":  img_path.name,
                    "cylinder_id": cyl_count,
                    "orientation": label,
                    "angle_deg":   angle_deg,
                    "confidence":  det_conf,
                    "blue_ratio":  blue_ratio,
                })
                cyl_summary.append(f"cyl{cyl_count}={label}({angle_deg:+.0f}deg)")
                cyl_count += 1

            cv2.imwrite(str(img_out / img_path.name), annotated)
            summary = "  ".join(cyl_summary) if cyl_summary else "no cylinders"
            print(f"  {img_path.name}: {cyl_count} cylinder(s) — {summary}")

    print(f"\n[DONE] annotated images → {img_out}/")
    print(f"       CSV             → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",     required=True,              help="Image file or folder")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,    help="YOLO .pt weights")
    p.add_argument("--conf",    type=float, default=0.45,   help="Detection confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45,   help="NMS IoU threshold")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--output",  default="runs/orientation", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        src=args.src,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        output_dir=args.output,
    )
