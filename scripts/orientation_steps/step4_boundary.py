#!/usr/bin/env python3
"""

To run with THE MOST RELEVANT IMAGES:
python3 step4_boundary.py --frames 0347 0353 0638 0942 1320 1717 2167 2>&1


Step 4 — Orientation angle + visual separation line.

For each crop:
  • Finds blue centroid (reliable anchor).
  • Finds white centroid restricted to the opposite half from blue
    (avoids peg-board background contamination).
  • Computes orientation angle from the white → blue vector:
        0°    white directly above blue  (upright)
       ±180°  white directly below blue  (inverted)
       +45°   white top-left / blue bottom-right
       -45°   white top-right / blue bottom-left
  • Draws on the crop:
        green line  — orientation axis (white centroid → blue centroid, extended)
        orange line — separation line  (perpendicular to axis at midpoint)
        white dot   — white centroid
        blue dot    — blue centroid
  • Saves the per-row colour density chart (matplotlib).

Output:
    runs/orientation_steps/04_boundary/<crop>.jpg
    runs/orientation_steps/04_boundary/<crop>_chart.png

Run:
    python3 step4_boundary.py                           # uses params.SAMPLE
    python3 step4_boundary.py --frames 0347 0638 0942  # specific frames
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import params

WHITE_LOWER = np.array([params.WHITE_H_MIN, params.WHITE_S_MIN, params.WHITE_V_MIN], dtype=np.uint8)
WHITE_UPPER = np.array([params.WHITE_H_MAX, params.WHITE_S_MAX, params.WHITE_V_MAX], dtype=np.uint8)
BLUE_LOWER  = np.array([params.BLUE_H_MIN,  params.BLUE_S_MIN,  params.BLUE_V_MIN],  dtype=np.uint8)
BLUE_UPPER  = np.array([params.BLUE_H_MAX,  params.BLUE_S_MAX,  params.BLUE_V_MAX],  dtype=np.uint8)

MIN_PIXELS = 50   # minimum mask pixels to trust a centroid


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centroid(mask: np.ndarray):
    """(x, y) median centre of the largest connected component, or None if too few pixels.

    Using the largest component so that a second smaller blob (e.g. peg-board
    background bleed) does not pull the result away from the main region.
    Median within that component avoids sub-blob outliers shifting the result.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2:  # label 0 is background
        return None

    # Pick the largest component (skip label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1  # +1 to re-align with full stats array

    if stats[largest, cv2.CC_STAT_AREA] < MIN_PIXELS:
        return None

    ys, xs = np.where(labels == largest)
    return float(np.median(xs)), float(np.median(ys))


# ── Orientation ───────────────────────────────────────────────────────────────

def compute_orientation(crop: np.ndarray):
    """
    Returns (angle_deg, blue_c, white_c, midpoint, blue_row, white_row).

    angle_deg convention (measured from downward vertical, clockwise +):
        0°   = white directly above blue   (upright)
       ±180° = white directly below blue   (inverted)
       +45°  = white top-left, blue bottom-right
       -45°  = white top-right, blue bottom-left
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    wm  = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    bm  = cv2.inRange(hsv, BLUE_LOWER,  BLUE_UPPER)

    h, w = crop.shape[:2]
    rows = np.arange(h)
    blue_row  = np.array([np.count_nonzero(bm[r]) / max(w, 1) for r in rows])
    white_row = np.array([np.count_nonzero(wm[r]) / max(w, 1) for r in rows])

    blue_c = _centroid(bm)
    if blue_c is None:
        return 0.0, None, None, None, blue_row, white_row

    # Restrict white to the half opposite to blue to exclude peg-board background
    split_y    = int(blue_c[1])
    blue_y_rel = blue_c[1] / h

    if blue_y_rel >= 0.5:
        # Blue in lower half → white cap should be in upper half
        white_c = _centroid(wm[:split_y, :])
        # y coordinates are already absolute (slice starts at 0)
    else:
        # Blue in upper half → white cap should be in lower half
        wc_rel  = _centroid(wm[split_y:, :])
        white_c = (wc_rel[0], wc_rel[1] + split_y) if wc_rel is not None else None

    if white_c is None:
        return 0.0, blue_c, None, None, blue_row, white_row

    # Vector from white centroid → blue centroid
    dx = blue_c[0] - white_c[0]
    dy = blue_c[1] - white_c[1]   # positive = blue is below white

    # atan2(dx, dy): 0° when dx=0, dy>0 (blue straight below white = upright)
    angle_deg = math.degrees(math.atan2(dx, dy))

    midpoint = ((white_c[0] + blue_c[0]) / 2,
                (white_c[1] + blue_c[1]) / 2)

    return angle_deg, blue_c, white_c, midpoint, blue_row, white_row


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_orientation(crop: np.ndarray, angle_deg: float,
                     blue_c, white_c, midpoint) -> np.ndarray:
    vis = crop.copy()
    h, w = vis.shape[:2]
    scale = max(h, w) * 2   # long enough to always exit the frame

    if blue_c is not None:
        cv2.circle(vis, (int(blue_c[0]),  int(blue_c[1])),  5, (200,  30,  30), -1)  # blue dot
    if white_c is not None:
        cv2.circle(vis, (int(white_c[0]), int(white_c[1])), 5, (  0,   0, 220), -1)  # white centroid → red

    if white_c is not None and midpoint is not None:
        # Direction of the white → blue axis
        dx = blue_c[0] - white_c[0]
        dy = blue_c[1] - white_c[1]
        length = math.hypot(dx, dy)

        if length > 1:
            nx, ny = dx / length, dy / length   # unit vector along axis

            # Green orientation axis (extended across the whole crop)
            p1 = (int(midpoint[0] - nx * scale), int(midpoint[1] - ny * scale))
            p2 = (int(midpoint[0] + nx * scale), int(midpoint[1] + ny * scale))
            cv2.line(vis, p1, p2, (0, 220, 0), 1)

            # Orange separation line (perpendicular to axis at midpoint)
            px, py = -ny, nx   # 90° rotation of (nx, ny)
            s1 = (int(midpoint[0] - px * scale), int(midpoint[1] - py * scale))
            s2 = (int(midpoint[0] + px * scale), int(midpoint[1] + py * scale))
            cv2.line(vis, s1, s2, (0, 165, 255), 2)

    # Angle label
    text = f"{angle_deg:+.1f} deg"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(vis, (2, h - th - 10), (tw + 6, h - 2), (0, 0, 0), -1)
    cv2.putText(vis, text, (4, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    return vis


# ── Density chart ─────────────────────────────────────────────────────────────

def save_density_chart(blue_row, white_row, blue_c, white_c,
                       angle_deg: float, crop_h: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, max(3.0, crop_h / 55)))
    rows = np.arange(crop_h)

    ax.barh(rows, white_row, color="orange", label="white", height=1.0, align="edge")
    ax.barh(rows, blue_row,  color="steelblue", label="blue",  height=1.0, align="edge", alpha=0.8)

    if blue_c is not None:
        ax.axhline(blue_c[1],   color="blue",  lw=1.2, ls="--", label=f"blue  median y={blue_c[1]:.0f}")
    if white_c is not None:
        ax.axhline(white_c[1],  color="yellow", lw=1.2, ls="--", label=f"white median y={white_c[1]:.0f}")

    ax.set_xlabel("pixel fraction per row")
    ax.set_ylabel("row  (0 = top of crop)")
    ax.set_ylim(crop_h, 0)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"per-row colour density  —  angle = {angle_deg:+.1f}°", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=90)
    plt.close(fig)


# ── CLI + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", nargs="+", metavar="FRAME_NUM",
                   help="Only process crops from these frame numbers "
                        "(e.g. --frames 0347 0353 0638). "
                        "Falls back to params.SAMPLE when omitted.")
    return p.parse_args()


def main():
    args      = parse_args()
    crops_dir = Path(params.OUT_ROOT) / "01_crops"
    out_dir   = Path(params.OUT_ROOT) / "04_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = sorted(crops_dir.glob("*_cyl*.jpg"))

    if args.frames:
        crops = [c for c in crops if any(f"frame_{n}" in c.name for n in args.frames)]
        print(f"[INFO] {len(crops)} crops matching frames {args.frames}  →  {out_dir}")
    elif params.SAMPLE:
        crops = crops[: params.SAMPLE]
        print(f"[INFO] {len(crops)} crops (SAMPLE={params.SAMPLE})  →  {out_dir}")
    else:
        print(f"[INFO] {len(crops)} crops  →  {out_dir}")

    for crop_path in crops:
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        angle_deg, blue_c, white_c, midpoint, blue_row, white_row = compute_orientation(crop)
        vis = draw_orientation(crop, angle_deg, blue_c, white_c, midpoint)

        cv2.imwrite(str(out_dir / crop_path.name), vis)
        save_density_chart(blue_row, white_row, blue_c, white_c, angle_deg,
                           crop.shape[0], out_dir / (crop_path.stem + "_chart.png"))

        print(f"  {crop_path.name}: {angle_deg:+.1f}°")

    print(f"[DONE] {out_dir}/")


if __name__ == "__main__":
    main()
