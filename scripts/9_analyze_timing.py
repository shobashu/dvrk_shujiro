#!/usr/bin/env python3
"""
Step 9 — Analyze MEASURED (not assumed) pipeline latency from timing_log.csv,
written by ForceOrientationTracker (dvrk_shujiro/arduino/force_orientation_bridge.py)
during live trials with the force sensor and/or YOLO orientation check enabled.

Two event types are logged, one row each time they happen:
    orientation_check — Target hit -> orientation result resolved
                         (settle delay [fixed] + inference time [measured] + retries)
    force_popup       — trigger event -> popup actually starts rendering

Each row is also tagged with which capabilities were enabled that run
("force", "orientation", or "force+orientation"), so this script can compare
"used together" against "used alone" directly from one combined log file.

Usage:
    python3 scripts/9_analyze_timing.py timing_log.csv
    python3 scripts/9_analyze_timing.py timing_log.csv --out-dir figures/
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("timing_csv", help="Path to timing_log.csv")
    p.add_argument("--out-dir", default=None, help="Where to save figures (default: same dir as timing_csv)")
    return p.parse_args()


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("total_latency_s", "settle_delay_s", "inference_time_s"):
                row[key] = float(row[key]) if row.get(key) else None
            row["retries"] = int(row["retries"]) if row.get("retries") else None
            rows.append(row)
    return rows


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def print_table(rows):
    by_config = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_config[r["config"]][r["event"]].append(r)

    print("── Measured latency summary (assumed vs. measured) ──\n")
    for config, events in sorted(by_config.items()):
        print(f"Config: {config}")

        if "orientation_check" in events:
            ev = events["orientation_check"]
            total = summarize([r["total_latency_s"] for r in ev])
            infer = summarize([r["inference_time_s"] for r in ev])
            retries = summarize([r["retries"] for r in ev])
            settle = ev[0]["settle_delay_s"]
            print(f"  Orientation check (n={total['n']}):")
            print(f"    Assumed (settle delay only):  {settle:.3f}s")
            print(f"    Measured total latency:       {total['mean']:.3f}s ± {total['std']:.3f}s "
                  f"(min {total['min']:.3f}s, max {total['max']:.3f}s)")
            print(f"    Measured inference time:      {infer['mean']:.3f}s ± {infer['std']:.3f}s")
            if retries:
                print(f"    Retries per check:            {retries['mean']:.2f} avg "
                      f"(max {retries['max']})")

        if "force_popup" in events:
            ev = events["force_popup"]
            total = summarize([r["total_latency_s"] for r in ev])
            print(f"  Force popup delay (n={total['n']}):")
            print(f"    Assumed:   ~0s (instant)")
            print(f"    Measured:  {total['mean']:.3f}s ± {total['std']:.3f}s "
                  f"(min {total['min']:.3f}s, max {total['max']:.3f}s)")

        if "state_detection" in events:
            ev = events["state_detection"]
            total = summarize([r["total_latency_s"] for r in ev])
            max_fps = 1.0 / total["mean"] if total["mean"] > 0 else float("inf")
            print(f"  Cylinder state detection, per-frame (n={total['n']}):")
            print(f"    Measured:  {total['mean']:.3f}s ± {total['std']:.3f}s "
                  f"(min {total['min']:.3f}s, max {total['max']:.3f}s)")
            print(f"    Implied max throughput: ~{max_fps:.1f} fps "
                  f"(camera publishes faster than this -> frames get dropped, not queued)")
        print()


def plot_orientation_breakdown(rows, out_path):
    """Stacked bar per config: settle (fixed) + measured inference + whatever's
    left over (retry-added delay), so total = measured end-to-end latency."""
    by_config = defaultdict(list)
    for r in rows:
        if r["event"] == "orientation_check":
            by_config[r["config"]].append(r)
    if not by_config:
        print("  (no orientation_check rows — skipping breakdown chart)")
        return

    configs = sorted(by_config)
    settle, infer, retry_delay = [], [], []
    for c in configs:
        ev = by_config[c]
        s = ev[0]["settle_delay_s"]
        i = statistics.mean(r["inference_time_s"] for r in ev)
        total = statistics.mean(r["total_latency_s"] for r in ev)
        settle.append(s)
        infer.append(i)
        retry_delay.append(max(0.0, total - s - i))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(configs))
    ax.bar(x, settle, label="Settle delay (fixed)", color="#999999")
    ax.bar(x, infer, bottom=settle, label="Inference time (measured)", color="#4C72B0")
    bottom2 = [s + i for s, i in zip(settle, infer)]
    ax.bar(x, retry_delay, bottom=bottom2, label="Retry-added delay (measured)", color="#DD8452")

    totals = [s + i + r for s, i, r in zip(settle, infer, retry_delay)]
    ax.set_ylim(0, max(totals) * 1.2 if totals else 1)
    for xi, total in enumerate(totals):
        ax.text(xi, total + max(totals) * 0.02, f"{total:.2f}s", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(configs)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Orientation-check delay breakdown\n(measured, by config)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_event_comparison(rows, out_path):
    """Grouped bar: mean total_latency_s per event type, per config — shows
    whether running force+orientation together changes the numbers vs. alone."""
    by_event_config = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_event_config[r["event"]][r["config"]].append(r["total_latency_s"])

    events = sorted(by_event_config)
    all_configs = sorted({c for ev in by_event_config.values() for c in ev})
    if not events:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(all_configs), 1)
    for i, config in enumerate(all_configs):
        means = [statistics.mean(by_event_config[ev][config]) if by_event_config[ev].get(config) else 0.0
                 for ev in events]
        xs = [j + i * width for j in range(len(events))]
        ax.bar(xs, means, width, label=config)

    ax.set_xticks([j + width * (len(all_configs) - 1) / 2 for j in range(len(events))])
    ax.set_xticklabels(events)
    ax.set_ylabel("Mean total latency (s)")
    ax.set_title("Measured latency by event and configuration")
    ax.legend(title="Config")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    csv_path = Path(args.timing_csv)
    rows = load_rows(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    print_table(rows)
    plot_orientation_breakdown(rows, out_dir / f"{stem}_orientation_breakdown.png")
    plot_event_comparison(rows, out_dir / f"{stem}_event_comparison.png")
    print("Done.")


if __name__ == "__main__":
    main()
