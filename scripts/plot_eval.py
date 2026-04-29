#!/usr/bin/env python3
"""
Reproduce and customise YOLOv8 evaluation plots:
  - BoxPR_curve   (Precision vs Recall, per class + mean)
  - BoxF1_curve   (F1 vs Confidence)
  - BoxP_curve    (Precision vs Confidence)
  - BoxR_curve    (Recall vs Confidence)
  - confusion_matrix (raw + normalised)
  - labels        (dataset statistics: counts, xy, wh)

Requires ultralytics ≥ 8.4 installed. Run with:
    /home/cfxuser/miniconda3/bin/python plot_eval.py
    /home/cfxuser/miniconda3/bin/python plot_eval.py --out my_plots/ --title "Run v2"
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ── Style (edit freely) ──────────────────────────────────────────────────────
CLASS_COLORS = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]
ALL_COLOR    = "#1a1a8c"
GRID_KW      = dict(linestyle="--", linewidth=0.7, alpha=0.65)
DPI          = 150


# ── Validation ───────────────────────────────────────────────────────────────

def run_val(weights: str, data_yaml: str):
    """
    Returns:
        curves_results – list of (px, py_per_class, xlabel, ylabel) per curve
        curves_names   – list of curve name strings
        ap50           – np.array of AP@0.50 per class
        class_names    – dict {int: str}
        cm_matrix      – np.array (nc+1, nc+1) raw confusion matrix
    """
    import warnings
    warnings.filterwarnings("ignore")
    from ultralytics import YOLO

    validator_ref = {}

    import tempfile
    tmp_dir = tempfile.mkdtemp()

    model = YOLO(weights)
    model.add_callback("on_val_end", lambda v: validator_ref.update({"v": v}))
    # plots=True is required so ultralytics accumulates the confusion matrix
    results = model.val(data=data_yaml, plots=True, verbose=False,
                        save_json=False, project=tmp_dir, name="val")

    ap50         = np.array(results.box.ap50)
    class_names  = results.names                          # {0: 'cylinder', ...}
    curves_names = results.curves                         # list of 4 strings
    # each item: [px (1000,), py (nc, 1000), xlabel, ylabel]
    curves_data  = [(np.array(r[0]), np.array(r[1]), r[2], r[3])
                    for r in results.curves_results]

    cm_matrix = None
    v = validator_ref.get("v")
    if v is not None and hasattr(v, "confusion_matrix"):
        cm_matrix = v.confusion_matrix.matrix.copy()

    return curves_data, curves_names, ap50, class_names, cm_matrix


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _style(ax, title, xlabel, ylabel, grid, xlim=(0, 1), ylim=(0, 1.05)):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(**GRID_KW)


def plot_pr(ax, px, py, ap50, class_names, title, grid):
    nc = py.shape[0]
    for i in range(nc):
        label = f"{class_names[i]}  AP={ap50[i]:.3f}"
        ax.plot(px, py[i], color=CLASS_COLORS[i % len(CLASS_COLORS)],
                linewidth=1.2, label=label)
    mean_py = py.mean(axis=0)
    ax.plot(px, mean_py, color=ALL_COLOR, linewidth=2.5,
            label=f"all classes  mAP@0.5={ap50.mean():.3f}")
    ax.legend(fontsize=7, loc="lower left")
    _style(ax, title, "Recall", "Precision", grid)


def plot_mc(ax, px, py, class_names, title, xlabel, ylabel, grid):
    nc = py.shape[0]
    mean_py  = py.mean(axis=0)
    peak_idx = mean_py.argmax()
    for i in range(nc):
        ax.plot(px, py[i], color=CLASS_COLORS[i % len(CLASS_COLORS)],
                linewidth=1.2, label=class_names[i])
    ax.plot(px, mean_py, color=ALL_COLOR, linewidth=2.5,
            label=f"all classes {mean_py[peak_idx]:.2f} @ {px[peak_idx]:.3f}")
    ax.legend(fontsize=7, loc="lower left" if ylabel != "Precision" else "lower right")
    _style(ax, title, xlabel, ylabel, grid)


def plot_confusion(ax, cm, class_names, title, normalise, grid):
    labels = [class_names[i] for i in sorted(class_names)] + ["background"]
    n = len(labels)
    data = cm[:n, :n].copy().astype(float)

    if normalise:
        row_sum = data.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        data /= row_sum

    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("True", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(False)

    thresh = 0.5 if normalise else data.max() / 2
    for r in range(n):
        for c in range(n):
            v = data[r, c]
            ax.text(c, r, f"{v:.2f}" if normalise else f"{int(v)}",
                    ha="center", va="center", fontsize=7,
                    color="white" if v > thresh else "black")


def plot_labels(axes, label_dir: Path, class_names: dict, title: str, grid: bool):
    ax_bar, ax_xy, ax_wh = axes
    txts = list(label_dir.rglob("*.txt"))

    if not txts:
        for ax in axes:
            ax.text(0.5, 0.5, f"No labels found in\n{label_dir}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    cls_list, cx_list, cy_list, w_list, h_list = [], [], [], [], []
    for f in txts:
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls_list.append(int(parts[0]))
                cx_list.append(float(parts[1]))
                cy_list.append(float(parts[2]))
                w_list.append(float(parts[3]))
                h_list.append(float(parts[4]))

    cls_arr = np.array(cls_list)
    cx, cy, w, h = map(np.array, (cx_list, cy_list, w_list, h_list))
    nc = len(class_names)
    colors = [CLASS_COLORS[c % len(CLASS_COLORS)] for c in cls_arr]

    # bar chart
    counts = [(cls_arr == i).sum() for i in range(nc)]
    names  = [class_names[i] for i in range(nc)]
    bars   = ax_bar.bar(names, counts, color=CLASS_COLORS[:nc])
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                    str(cnt), ha="center", va="bottom", fontsize=8)
    ax_bar.set_title(f"{title} — Instance Counts", fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("Instances", fontsize=9)
    ax_bar.tick_params(axis="x", rotation=15, labelsize=8)
    if grid:
        ax_bar.set_axisbelow(True)
        ax_bar.grid(axis="y", **GRID_KW)

    # xy scatter
    ax_xy.scatter(cx, cy, c=colors, s=2, alpha=0.4)
    ax_xy.set_title("Box Centres (cx, cy)", fontsize=10, fontweight="bold")
    ax_xy.set_xlabel("cx", fontsize=9)
    ax_xy.set_ylabel("cy", fontsize=9)
    ax_xy.set_xlim(0, 1); ax_xy.set_ylim(0, 1)
    ax_xy.invert_yaxis()
    if grid:
        ax_xy.set_axisbelow(True)
        ax_xy.grid(**GRID_KW)
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=class_names[i]) for i in range(nc)]
    ax_xy.legend(handles=patches, fontsize=6, loc="upper right")

    # wh scatter
    ax_wh.scatter(w, h, c=colors, s=2, alpha=0.4)
    ax_wh.set_title("Box Sizes (w × h)", fontsize=10, fontweight="bold")
    ax_wh.set_xlabel("width", fontsize=9)
    ax_wh.set_ylabel("height", fontsize=9)
    if grid:
        ax_wh.set_axisbelow(True)
        ax_wh.grid(**GRID_KW)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = not args.no_grid

    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    class_names = {i: n for i, n in enumerate(cfg["names"])}
    label_dir   = Path(cfg["train"]).parent.parent / "labels" / "train"

    print("[INFO] Running model.val() — this may take a minute …")
    curves_data, curves_names, ap50, val_names, cm = run_val(args.weights, args.data)

    T = args.title

    # map curve name → data
    curve_map = {name: data for name, data in zip(curves_names, curves_data)}

    # ── PR curve ──────────────────────────────────────────────────────────────
    key = next((k for k in curve_map if "Precision-Recall" in k), None)
    if key:
        px, py, xlabel, ylabel = curve_map[key]
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_pr(ax, px, py, ap50, class_names, f"{T} — Precision-Recall Curve", grid)
        fig.tight_layout(); fig.savefig(out / "BoxPR_curve.png", dpi=DPI); plt.close(fig)
        print(f"[DONE] {out}/BoxPR_curve.png")

    # ── F1 curve ──────────────────────────────────────────────────────────────
    key = next((k for k in curve_map if "F1" in k), None)
    if key:
        px, py, xlabel, ylabel = curve_map[key]
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_mc(ax, px, py, class_names, f"{T} — F1-Confidence Curve", xlabel, ylabel, grid)
        fig.tight_layout(); fig.savefig(out / "BoxF1_curve.png", dpi=DPI); plt.close(fig)
        print(f"[DONE] {out}/BoxF1_curve.png")

    # ── Precision curve ───────────────────────────────────────────────────────
    key = next((k for k in curve_map if "Precision-Confidence" in k), None)
    if key:
        px, py, xlabel, ylabel = curve_map[key]
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_mc(ax, px, py, class_names, f"{T} — Precision-Confidence Curve", xlabel, ylabel, grid)
        fig.tight_layout(); fig.savefig(out / "BoxP_curve.png", dpi=DPI); plt.close(fig)
        print(f"[DONE] {out}/BoxP_curve.png")

    # ── Recall curve ──────────────────────────────────────────────────────────
    key = next((k for k in curve_map if "Recall-Confidence" in k), None)
    if key:
        px, py, xlabel, ylabel = curve_map[key]
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_mc(ax, px, py, class_names, f"{T} — Recall-Confidence Curve", xlabel, ylabel, grid)
        fig.tight_layout(); fig.savefig(out / "BoxR_curve.png", dpi=DPI); plt.close(fig)
        print(f"[DONE] {out}/BoxR_curve.png")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if cm is not None:
        for normalise, fname in [(True, "confusion_matrix_normalized.png"),
                                 (False, "confusion_matrix.png")]:
            suffix = "Normalised" if normalise else "Raw Counts"
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_confusion(ax, cm, class_names, f"{T} — Confusion Matrix ({suffix})",
                           normalise, grid)
            fig.tight_layout(); fig.savefig(out / fname, dpi=DPI); plt.close(fig)
            print(f"[DONE] {out}/{fname}")
    else:
        print("[WARN] Confusion matrix not captured — skipping")

    # ── Labels ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_labels(axes, label_dir, class_names, T, grid)
    fig.suptitle(f"{T} — Dataset Label Statistics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "labels.jpg", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {out}/labels.jpg")

    print(f"\nAll plots saved to: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/models/dvrk_v1-2/weights/best.pt")
    p.add_argument("--data",    default="data.yaml")
    p.add_argument("--out",     default="plots_custom/")
    p.add_argument("--title",   default="dVRK YOLOv8")
    p.add_argument("--no-grid", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
