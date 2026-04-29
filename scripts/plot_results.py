#!/usr/bin/env python3
"""
Reproduce and customise the YOLOv8 results.png from results.csv.

Usage:
    python plot_results.py
    python plot_results.py --csv runs/detect/models/dvrk_v1-2/results.csv
    python plot_results.py --csv results.csv --out my_plot.png --smooth 0.6
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Which columns to plot and how to label them
# ---------------------------------------------------------------------------
PANELS = [
    # (csv_column,               subplot_title,          y_label)
    ("train/box_loss",           "Train Box Loss",        "Loss"),
    ("train/cls_loss",           "Train Class Loss",      "Loss"),
    ("train/dfl_loss",           "Train DFL Loss",        "Loss"),
    ("metrics/precision(B)",     "Precision",             ""),
    ("metrics/recall(B)",        "Recall",                ""),
    ("val/box_loss",             "Val Box Loss",          "Loss"),
    ("val/cls_loss",             "Val Class Loss",        "Loss"),
    ("val/dfl_loss",             "Val DFL Loss",          "Loss"),
    ("metrics/mAP50(B)",         "mAP @ IoU 0.50",        ""),
    ("metrics/mAP50-95(B)",      "mAP @ IoU 0.50–0.95",   ""),
]

LAYOUT = (2, 5)   # rows, cols — change to (5, 2) for a tall layout


def smooth_ewm(series: pd.Series, alpha: float) -> pd.Series:
    """Exponential weighted moving average (same idea as Ultralytics smoother)."""
    return series.ewm(alpha=alpha).mean()


def plot(csv_path: str, out_path: str, smooth_alpha: float, figsize: tuple,
         title: str, grid: bool, raw_color: str, smooth_color: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()   # strip any whitespace from header

    nrows, ncols = LAYOUT
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, (col, subplot_title, ylabel) in zip(axes, PANELS):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        y = df[col]
        x = df["epoch"]

        ax.plot(x, y, color=raw_color, linewidth=1.0, alpha=0.6, label="raw")
        ax.plot(x, smooth_ewm(y, smooth_alpha),
                color=smooth_color, linewidth=1.5, linestyle="--", label="smooth")

        ax.set_title(subplot_title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        if grid:
            ax.set_axisbelow(True)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.65)

    # Legend on first axis only
    axes[0].legend(fontsize=7)

    # Overall figure title
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[DONE] Saved → {out_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="runs/detect/models/dvrk_v1-2/results.csv")
    p.add_argument("--out",     default="results_custom.png")
    p.add_argument("--smooth",  type=float, default=0.3,
                   help="EWM alpha for smoothing (0=very smooth, 1=no smooth)")
    p.add_argument("--title",   default="dVRK YOLOv8 Training Results")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--figsize", default="18,8",
                   help="Figure size as W,H e.g. '18,8'")
    p.add_argument("--raw-color",    default="steelblue")
    p.add_argument("--smooth-color", default="tomato")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    w, h = map(float, args.figsize.split(","))
    plot(
        csv_path=args.csv,
        out_path=args.out,
        smooth_alpha=args.smooth,
        figsize=(w, h),
        title=args.title,
        grid=not args.no_grid,
        raw_color=args.raw_color,
        smooth_color=args.smooth_color,
    )
