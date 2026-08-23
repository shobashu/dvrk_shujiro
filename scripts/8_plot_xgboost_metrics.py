#!/usr/bin/env python3
"""
Step 8 — Generate report-ready figures from a 7_2_train_xgboost.py .metrics.json file.

All figures come from the out-of-fold (genuinely held-out) results saved in
the metrics file — never the training-set numbers, which are meaningless for
judging generalization (see 7_2_train_xgboost.py's own warning about that).

Usage:
    python3 scripts/8_plot_xgboost_metrics.py models/cylinder_state_xgb_v2.metrics.json
    python3 scripts/8_plot_xgboost_metrics.py models/cylinder_state_xgb_v2.metrics.json --out-dir figures/
    python3 scripts/8_plot_xgboost_metrics.py models/cylinder_state_xgb_v2.metrics.json \
        --compare models/cylinder_state_xgb.metrics.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("metrics_json", help="Path to a .metrics.json file written by 7_2_train_xgboost.py")
    p.add_argument("--out-dir", default=None,
                   help="Where to save the figures (default: same directory as metrics_json)")
    p.add_argument("--top-n", type=int, default=15, help="How many top features to show (default 15)")
    p.add_argument("--compare", default=None,
                   help="Optional: an earlier .metrics.json to compare F1-per-class against "
                        "(e.g. a model trained on fewer sessions), generates a before/after chart")
    return p.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_confusion_matrix(metrics: dict, out_path: Path):
    labels = metrics["class_labels"]
    cm = np.array(metrics["out_of_fold_confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    unit = "session" if metrics.get("cv_group_by") == "session" else "event"
    ax.set_title(f"Out-of-fold confusion matrix\n({metrics.get('n_splits', '?')}-fold, "
                 f"grouped by {unit})")

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(cm[i, j])
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)", ha="center", va="center",
                    color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_class_metrics(metrics: dict, out_path: Path):
    labels = metrics["class_labels"]
    report = metrics["out_of_fold_classification_report"]

    precision = [report[c]["precision"] for c in labels]
    recall    = [report[c]["recall"] for c in labels]
    f1        = [report[c]["f1-score"] for c in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1-score")

    for xs, vals in [(x - width, precision), (x, recall), (x + width, f1)]:
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Out-of-fold per-class performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_importance(metrics: dict, out_path: Path, top_n: int):
    top = metrics["top_feature_importances"][:top_n]
    names = [t["feature"] for t in top][::-1]
    values = [t["importance"] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(names))))
    ax.barh(names, values, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{len(names)} feature importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(metrics_before: dict, metrics_after: dict,
                     label_before: str, label_after: str, out_path: Path):
    labels = metrics_after["class_labels"]
    report_before = metrics_before["out_of_fold_classification_report"]
    report_after  = metrics_after["out_of_fold_classification_report"]

    f1_before = [report_before[c]["f1-score"] for c in labels]
    f1_after  = [report_after[c]["f1-score"] for c in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, f1_before, width, label=label_before, color="#999999")
    ax.bar(x + width / 2, f1_after,  width, label=label_after,  color="#4C72B0")

    for xi, v in zip(x - width / 2, f1_before):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, f1_after):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1-score (out-of-fold)")
    ax.set_title("F1-score per class: before vs. after adding sessions")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_json)
    metrics = load(metrics_path)

    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = metrics_path.name.replace(".metrics.json", "")

    print(f"── Generating figures from {metrics_path} ──")
    plot_confusion_matrix(metrics, out_dir / f"{stem}_confusion_matrix.png")
    plot_per_class_metrics(metrics, out_dir / f"{stem}_per_class_metrics.png")
    plot_feature_importance(metrics, out_dir / f"{stem}_feature_importance.png", args.top_n)

    if args.compare:
        compare_path = Path(args.compare)
        metrics_before = load(compare_path)
        label_before = compare_path.name.replace(".metrics.json", "")
        plot_comparison(metrics_before, metrics, label_before, stem,
                         out_dir / f"{stem}_vs_{label_before}_f1_comparison.png")

    print("Done.")


if __name__ == "__main__":
    main()
