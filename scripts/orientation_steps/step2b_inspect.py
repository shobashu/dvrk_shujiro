#!/usr/bin/env python3
"""
Step 2b — HSV inspection: distribution plots + interactive pixel picker.

Two modes (both run by default):

1. PLOTS (always saved)
   For each sample crop: H / S / V histograms with the current threshold
   ranges from params.py drawn as coloured spans, so you can judge whether
   the thresholds capture the right pixels.
   Also saves a 2-D H-vs-S scatter coloured by actual hue.
   Output: runs/orientation_steps/02b_inspect/<crop>_hist.png
           runs/orientation_steps/02b_inspect/<crop>_scatter.png

2. INTERACTIVE PICKER  (skipped with --no-interactive)
   Opens an OpenCV window per crop. Hover to see live HSV values overlaid
   on the image and printed in the title bar.  Also shows whether the pixel
   would be caught by the current white / blue threshold.
   Keys: n = next crop   p = previous crop   q = quit

Run:
    python3 step2b_inspect.py                 # plots + picker
    python3 step2b_inspect.py --no-interactive # plots only (headless)
"""

import argparse
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


# ── Histogram plots ───────────────────────────────────────────────────────────

def save_histograms(crop: np.ndarray, out_stem: Path):
    """H / S / V histograms with current threshold spans highlighted."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # H histogram
    axes[0].hist(H, bins=180, range=(0, 179), color="grey", alpha=0.6)
    axes[0].axvspan(params.BLUE_H_MIN,  params.BLUE_H_MAX,  alpha=0.25, color="steelblue", label="blue range")
    axes[0].set_xlabel("H  (0–179)"); axes[0].set_ylabel("pixel count")
    axes[0].set_title("Hue"); axes[0].legend(fontsize=7)

    # S histogram
    axes[1].hist(S, bins=128, range=(0, 255), color="grey", alpha=0.6)
    axes[1].axvspan(0,                    params.WHITE_S_MAX, alpha=0.25, color="gold",      label=f"white S ≤ {params.WHITE_S_MAX}")
    axes[1].axvspan(params.BLUE_S_MIN,    255,               alpha=0.25, color="steelblue", label=f"blue  S ≥ {params.BLUE_S_MIN}")
    axes[1].set_xlabel("S  (0–255)"); axes[1].set_title("Saturation"); axes[1].legend(fontsize=7)

    # V histogram
    axes[2].hist(V, bins=128, range=(0, 255), color="grey", alpha=0.6)
    axes[2].axvspan(params.WHITE_V_MIN, 255,               alpha=0.25, color="gold",      label=f"white V ≥ {params.WHITE_V_MIN}")
    axes[2].axvspan(params.BLUE_V_MIN,  255,               alpha=0.25, color="steelblue", label=f"blue  V ≥ {params.BLUE_V_MIN}")
    axes[2].set_xlabel("V  (0–255)"); axes[2].set_title("Value (brightness)"); axes[2].legend(fontsize=7)

    fig.suptitle(out_stem.stem, fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_stem.parent / (out_stem.stem + "_hist.png")), dpi=90)
    plt.close(fig)


def save_scatter(crop: np.ndarray, out_stem: Path):
    """2-D H-vs-S scatter coloured by actual hue, with threshold rectangle."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].ravel().astype(float)
    S = hsv[:, :, 1].ravel().astype(float)

    # Downsample for speed
    idx  = np.random.choice(len(H), size=min(3000, len(H)), replace=False)
    H_s, S_s = H[idx], S[idx]

    # Convert OpenCV hue (0–179) to matplotlib colour (0–1 in HSV)
    colors = plt.cm.hsv(H_s / 179.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(H_s, S_s, c=colors, s=2, alpha=0.5)

    # Draw blue threshold rectangle
    rect_b = plt.Rectangle((params.BLUE_H_MIN, params.BLUE_S_MIN),
                            params.BLUE_H_MAX - params.BLUE_H_MIN,
                            255 - params.BLUE_S_MIN,
                            linewidth=1.5, edgecolor="blue", facecolor="none",
                            label="blue threshold")
    ax.add_patch(rect_b)

    # White threshold: low S band (any H)
    ax.axhspan(0, params.WHITE_S_MAX, alpha=0.10, color="gold", label=f"white S ≤ {params.WHITE_S_MAX}")

    ax.set_xlabel("H  (0–179)"); ax.set_ylabel("S  (0–255)")
    ax.set_xlim(0, 179); ax.set_ylim(0, 255)
    ax.set_title(f"H vs S scatter — {out_stem.stem}"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(str(out_stem.parent / (out_stem.stem + "_scatter.png")), dpi=90)
    plt.close(fig)


# ── Interactive pixel picker ──────────────────────────────────────────────────

def _pixel_label(h, s, v):
    """Return 'WHITE', 'BLUE', or '----' based on current thresholds."""
    if (params.WHITE_H_MIN <= h <= params.WHITE_H_MAX and
            params.WHITE_S_MIN <= s <= params.WHITE_S_MAX and
            params.WHITE_V_MIN <= v <= params.WHITE_V_MAX):
        return "WHITE"
    if (params.BLUE_H_MIN <= h <= params.BLUE_H_MAX and
            params.BLUE_S_MIN <= s <= params.BLUE_S_MAX and
            params.BLUE_V_MIN <= v <= params.BLUE_V_MAX):
        return "BLUE"
    return "----"


def interactive_picker(crops: list):
    """OpenCV window with live HSV read-out on mouse hover."""
    idx   = [0]
    mouse = {"x": 0, "y": 0}

    def on_mouse(event, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y

    win = "HSV picker  [n=next  p=prev  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        crop = crops[idx[0]].copy()
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h2, w2 = crop.shape[:2]

        mx = min(max(mouse["x"], 0), w2 - 1)
        my = min(max(mouse["y"], 0), h2 - 1)
        b, g, r = crop[my, mx]
        hv, sv, vv = hsv[my, mx]
        label = _pixel_label(int(hv), int(sv), int(vv))

        # Overlay text on image
        disp = crop.copy()
        info = (f"xy=({mx},{my})  "
                f"BGR=({b},{g},{r})  "
                f"HSV=({hv},{sv},{vv})  "
                f"-> {label}")
        cv2.putText(disp, info, (6, h2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw crosshair
        cv2.line(disp, (mx, 0),  (mx, h2), (0, 255, 255), 1)
        cv2.line(disp, (0, my),  (w2, my), (0, 255, 255), 1)

        # Title bar
        crop_name = crops[idx[0]].shape  # placeholder; we pass name separately below
        cv2.setWindowTitle(win, f"[{idx[0]+1}/{len(crops)}]  {info}")
        cv2.imshow(win, disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            idx[0] = (idx[0] + 1) % len(crops)
        elif key == ord("p"):
            idx[0] = (idx[0] - 1) % len(crops)

    cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip OpenCV window, save plots only")
    return p.parse_args()


def main():
    args      = parse_args()
    crops_dir = Path(params.OUT_ROOT) / "01_crops"
    out_dir   = Path(params.OUT_ROOT) / "02b_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = sorted(crops_dir.glob("*_cyl*.jpg"))
    if params.SAMPLE:
        crop_paths = crop_paths[: params.SAMPLE]

    print(f"[INFO] {len(crop_paths)} crops")

    images = []
    for cp in crop_paths:
        img = cv2.imread(str(cp))
        if img is None:
            continue
        images.append(img)
        stem = out_dir / cp.stem
        save_histograms(img, stem)
        save_scatter(img,    stem)
        print(f"  saved plots for {cp.name}")

    print(f"[DONE] plots → {out_dir}/")

    if not args.no_interactive and images:
        print("\n[PICKER] opening OpenCV window …  n=next  p=prev  q=quit")
        print("         hover over a pixel to see its HSV value and whether it")
        print("         matches the current WHITE or BLUE threshold in params.py")
        interactive_picker(images)


if __name__ == "__main__":
    main()
