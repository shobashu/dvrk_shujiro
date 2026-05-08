#!/usr/bin/env python3
"""
Step 3 — HSV masking.

Applies the WHITE_* and BLUE_* thresholds from params.py and saves a
4-panel image per crop:
    [ original | white mask | blue mask | colour overlay ]

Workflow:
  1. Run this script.
  2. Open runs/orientation_steps/03_masks/ and inspect the panels.
  3. If the white mask bleeds into the background, raise WHITE_V_MIN
     or lower WHITE_S_MAX in params.py.
  4. If the blue mask is sparse, widen BLUE_H range or lower BLUE_S_MIN.
  5. Re-run until masks look clean, then proceed to step 4.

Output:
    runs/orientation_steps/03_masks/<crop_name>.jpg

Run:
    python3 step3_mask.py                          # all crops (respects SAMPLE)
    python3 step3_mask.py 0205 0219 0368           # only those frame numbers
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import params

WHITE_LOWER = np.array([params.WHITE_H_MIN, params.WHITE_S_MIN, params.WHITE_V_MIN], dtype=np.uint8)
WHITE_UPPER = np.array([params.WHITE_H_MAX, params.WHITE_S_MAX, params.WHITE_V_MAX], dtype=np.uint8)
BLUE_LOWER  = np.array([params.BLUE_H_MIN,  params.BLUE_S_MIN,  params.BLUE_V_MIN],  dtype=np.uint8)
BLUE_UPPER  = np.array([params.BLUE_H_MAX,  params.BLUE_S_MAX,  params.BLUE_V_MAX],  dtype=np.uint8)


def mask_panel(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    wm  = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    bm  = cv2.inRange(hsv, BLUE_LOWER,  BLUE_UPPER)

    area  = crop.shape[0] * crop.shape[1]
    w_pct = 100 * np.count_nonzero(wm) / area
    b_pct = 100 * np.count_nonzero(bm) / area

    # Colour overlay: paint matched pixels onto a copy of the crop
    overlay = crop.copy()
    overlay[wm > 0] = (230, 230, 230)   # white pixels  → light grey
    overlay[bm > 0] = ( 30,  30, 210)   # blue  pixels  → vivid blue

    wm_bgr = cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)
    bm_bgr = cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR)

    panels = [
        (crop.copy(),    "original"),
        (wm_bgr,         f"white {w_pct:.0f}%"),
        (bm_bgr,         f"blue  {b_pct:.0f}%"),
        (overlay,        "overlay"),
    ]
    for img, lbl in panels:
        cv2.putText(img, lbl, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return np.hstack([p for p, _ in panels])


def main():
    frame_filter = sys.argv[1:]  # e.g. ["0205", "0219", "0368"]

    crops_dir = Path(params.OUT_ROOT) / "01_crops"
    out_dir   = Path(params.OUT_ROOT) / "03_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = sorted(crops_dir.glob("*_cyl*.jpg"))
    if frame_filter:
        crops = [c for c in crops if any(f"frame_{n}_" in c.name for n in frame_filter)]
    elif params.SAMPLE:
        crops = crops[: params.SAMPLE]
    print(f"[INFO] {len(crops)} crops  →  {out_dir}")
    print(f"  white  H=[{params.WHITE_H_MIN},{params.WHITE_H_MAX}]"
          f"  S=[{params.WHITE_S_MIN},{params.WHITE_S_MAX}]"
          f"  V=[{params.WHITE_V_MIN},{params.WHITE_V_MAX}]")
    print(f"  blue   H=[{params.BLUE_H_MIN},{params.BLUE_H_MAX}]"
          f"  S=[{params.BLUE_S_MIN},{params.BLUE_S_MAX}]"
          f"  V=[{params.BLUE_V_MIN},{params.BLUE_V_MAX}]")

    for crop_path in crops:
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        cv2.imwrite(str(out_dir / crop_path.name), mask_panel(crop))

    print(f"[DONE] {out_dir}/")


if __name__ == "__main__":
    main()
