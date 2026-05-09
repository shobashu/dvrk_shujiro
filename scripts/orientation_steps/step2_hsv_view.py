#!/usr/bin/env python3
"""
Step 2 — HSV channel visualisation.

For each crop from step 1, saves a 4-panel image:
    [ BGR original | H channel | S channel | V channel ]

Use this to identify what H/S/V values the white cap and blue body
have so you can tune the thresholds in params.py before running step 3.

Reading the panels:
  H (hue)        — colour wheel mapped with COLORMAP_HSV; blue ≈ cyan band
  S (saturation) — bright = vivid colour, dark = grey/white
  V (value)      — bright = well-lit, dark = shadow

Output:
    runs/orientation_steps/02_hsv/<crop_name>.jpg

Run:
    python3 step2_hsv_view.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import params


def hsv_panel(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Scale H 0–179 → 0–255 so the full colormap range is used
    H_vis = cv2.applyColorMap((H.astype(np.uint16) * 255 // 179).astype(np.uint8),
                              cv2.COLORMAP_HSV)
    S_vis = cv2.cvtColor(S, cv2.COLOR_GRAY2BGR)
    V_vis = cv2.cvtColor(V, cv2.COLOR_GRAY2BGR)

    panels = [crop_bgr.copy(), H_vis, S_vis, V_vis]
    labels = ["BGR", "H (hue)", "S (sat)", "V (val)"]
    for panel, lbl in zip(panels, labels):
        cv2.putText(panel, lbl, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return np.hstack(panels)


def main():
    crops_dir = Path(params.OUT_ROOT) / "01_crops"
    out_dir   = Path(params.OUT_ROOT) / "02_hsv"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = sorted(crops_dir.glob("*_cyl*.jpg"))
    if params.SAMPLE:
        crops = crops[: params.SAMPLE]
    print(f"[INFO] {len(crops)} crops  →  {out_dir}")

    for crop_path in crops:
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        cv2.imwrite(str(out_dir / crop_path.name), hsv_panel(crop))

    print(f"[DONE] open {out_dir}/ and check H/S/V values,")
    print(f"       then update thresholds in params.py before running step 3.")


if __name__ == "__main__":
    main()
