#!/usr/bin/env python3
"""
Correlate PSM1 robot data with Arduino trial events for any subject.

Paths are resolved automatically from --subject:
  PSM1    → /home/stanford/dvrk_recordings/training/{subject}_PSM1.csv
  Arduino → /home/stanford/Arduino/experiment_data_{subject}.csv

Override either path explicitly with --psm1 / --arduino.

Arduino columns:  Trial, Target_Peg, Target_Color,
                  Unix_Cue_Time, Unix_Lift_Time, Unix_Place_Time

Phases per trial:
  React phase  = Cue_Time  → Lift_Time   (seeing target → picking up peg)
  Transport    = Lift_Time → Place_Time  (lifting → placing peg)

Usage:
    python3 plots_mk/plot_corr.py --subject fc12
    python3 plots_mk/plot_corr.py --subject tg14 --out tg14_corr.png
    python3 plots_mk/plot_corr.py --psm1 /path/to/X_PSM1.csv --arduino /path/to/data_X.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

PSM1_DIR    = "/home/stanford/dvrk_recordings/training"
ARDUINO_DIR = "/home/stanford/Arduino"

REACT_COLOR = "#a8d8ea"   # light blue  — Cue → Lift
TRANS_COLOR = "#ffcb77"   # amber       — Lift → Place
BLUE_COLOR  = "#4e79a7"
WHITE_COLOR = "#e15759"


def load_psm1(path):
    df = pd.read_csv(path)
    return df


def load_arduino(path):
    df = pd.read_csv(path)
    return df


def filter_to_trials(df, trials):
    """Keep only robot rows that fall within any trial window."""
    t_min = trials["Unix_Cue_Time"].min()
    t_max = trials["Unix_Place_Time"].max()
    return df[(df["timestamp_s"] >= t_min) & (df["timestamp_s"] <= t_max)].copy()


def shade_trials(ax, trials, ymin, ymax, alpha=0.25):
    for _, row in trials.iterrows():
        ax.axvspan(row["Unix_Cue_Time"],  row["Unix_Lift_Time"],
                   color=REACT_COLOR, alpha=alpha, linewidth=0)
        ax.axvspan(row["Unix_Lift_Time"], row["Unix_Place_Time"],
                   color=TRANS_COLOR, alpha=alpha, linewidth=0)


def per_trial_metric(signal_t, signal_v, t_start, t_end, func=np.mean):
    mask = (signal_t >= t_start) & (signal_t <= t_end)
    vals = signal_v[mask]
    return func(vals) if len(vals) > 0 else np.nan


def plot(psm1_csv, arduino_csv, subject, out):
    psm1    = load_psm1(psm1_csv)
    trials  = load_arduino(arduino_csv)

    cp  = filter_to_trials(psm1[psm1["topic"] == "measured_cp"].copy(), trials)
    cv  = filter_to_trials(psm1[psm1["topic"] == "measured_cv"].copy(), trials)
    js  = filter_to_trials(psm1[psm1["topic"] == "measured_js"].copy(), trials)
    jaw = filter_to_trials(psm1[psm1["topic"] == "jaw/measured_js"].copy(), trials)

    # Derived signals
    speed = np.sqrt(cv["lin_x"]**2 + cv["lin_y"]**2 + cv["lin_z"]**2) if not cv.empty else None
    jaw_deg = np.degrees(jaw["jaw_pos"]) if not jaw.empty else None

    # Use absolute Unix time on x-axis (shared across all time panels)
    t_min = trials["Unix_Cue_Time"].min()
    t_max = trials["Unix_Place_Time"].max()

    fig = plt.figure(figsize=(20, 26))
    fig.suptitle(f"{subject}  —  PSM1 ↔ Arduino Trial Correlation", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    legend_patches = [
        mpatches.Patch(color=REACT_COLOR, alpha=0.7, label="React phase (Cue→Lift)"),
        mpatches.Patch(color=TRANS_COLOR, alpha=0.7, label="Transport phase (Lift→Place)"),
    ]

    # ── 1. Speed + jaw overlay with trial bands ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    shade_trials(ax1, trials, 0, 1)
    if speed is not None:
        ax1.plot(cv["timestamp_s"], speed, color=BLUE_COLOR, linewidth=0.7,
                 label="Speed (m/s)", zorder=3)
    ax1_r = ax1.twinx()
    if jaw_deg is not None:
        ax1_r.plot(jaw["timestamp_s"], jaw_deg, color="#e15759",
                   linewidth=0.7, alpha=0.8, label="Jaw (°)", zorder=3)
    ax1.set_xlim(t_min - 1, t_max + 1)
    ax1.set_title("Cartesian Speed  +  Jaw Angle  (shaded = trial phases)")
    ax1.set_xlabel("Unix time (s)")
    ax1.set_ylabel("Speed (m/s)", color=BLUE_COLOR)
    ax1_r.set_ylabel("Jaw angle (°)", color="#e15759")
    ax1.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=BLUE_COLOR,  label="Speed (m/s)"),
        plt.Line2D([0], [0], color="#e15759",   label="Jaw (°)"),
    ], fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2)
    # mark trial numbers
    for _, row in trials.iterrows():
        ax1.text(row["Unix_Cue_Time"], ax1.get_ylim()[1] * 0.98,
                 str(int(row["Trial"])), fontsize=5.5, ha="center", va="top",
                 color="#555555")

    # ── 2. Insertion depth with trial bands ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    shade_trials(ax2, trials, 0, 1)
    if not js.empty and "insertion_pos" in js.columns:
        ax2.plot(js["timestamp_s"], js["insertion_pos"],
                 color="#59a14f", linewidth=0.8)
    ax2.set_xlim(t_min - 1, t_max + 1)
    ax2.set_title("Insertion Depth during Trials")
    ax2.set_xlabel("Unix time (s)")
    ax2.set_ylabel("Insertion (m)")
    ax2.legend(handles=legend_patches, fontsize=7)
    ax2.grid(True, alpha=0.2)

    # ── 3. Per-trial timing (reaction + transport) ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    react_times = (trials["Unix_Lift_Time"]  - trials["Unix_Cue_Time"]).values
    trans_times = (trials["Unix_Place_Time"] - trials["Unix_Lift_Time"]).values
    trial_nums  = trials["Trial"].values
    x = np.arange(len(trial_nums))
    bars_r = ax3.bar(x,           react_times, label="React (Cue→Lift)",      color=REACT_COLOR, edgecolor="gray", linewidth=0.4)
    bars_t = ax3.bar(x, trans_times, bottom=react_times, label="Transport (Lift→Place)", color=TRANS_COLOR, edgecolor="gray", linewidth=0.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(trial_nums, fontsize=6, rotation=45)
    ax3.set_title("Per-Trial Phase Durations (s)")
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("Duration (s)")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Peak speed per trial, split by colour ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if speed is not None:
        peak_speeds = []
        for _, row in trials.iterrows():
            mask = (cv["timestamp_s"] >= row["Unix_Lift_Time"]) & \
                   (cv["timestamp_s"] <= row["Unix_Place_Time"])
            v = speed.values[mask.values] if hasattr(mask, "values") else speed[mask]
            peak_speeds.append(np.max(v) if len(v) > 0 else np.nan)
        trials = trials.copy()
        trials["peak_speed"] = peak_speeds
        for color, label, marker in [("Blue", "Blue peg", "o"), ("White", "White peg", "s")]:
            sub = trials[trials["Target_Color"] == color]
            col = BLUE_COLOR if color == "Blue" else WHITE_COLOR
            ax4.scatter(sub["Trial"], sub["peak_speed"], color=col,
                        label=label, s=40, marker=marker, zorder=3)
        ax4.plot(trials["Trial"], trials["peak_speed"],
                 color="gray", linewidth=0.5, zorder=1)
    ax4.set_title("Peak Speed during Transport Phase\n(by target colour)")
    ax4.set_xlabel("Trial")
    ax4.set_ylabel("Peak speed (m/s)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. Total trial time over trials (learning curve) ──────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    total_time = (trials["Unix_Place_Time"] - trials["Unix_Cue_Time"]).values
    colors_seq = [BLUE_COLOR if c == "Blue" else WHITE_COLOR
                  for c in trials["Target_Color"]]
    ax5.bar(trials["Trial"], total_time, color=colors_seq, edgecolor="gray", linewidth=0.4)
    # rolling average
    window = min(5, len(total_time))
    roll = pd.Series(total_time).rolling(window, min_periods=1, center=True).mean()
    ax5.plot(trials["Trial"], roll, color="black", linewidth=1.5,
             linestyle="--", label=f"{window}-trial rolling avg")
    ax5.set_title("Total Trial Duration — Learning Curve\n(blue=Blue peg, red=White peg)")
    ax5.set_xlabel("Trial")
    ax5.set_ylabel("Duration (s)")
    ax5.legend(fontsize=8)
    ax5.grid(axis="y", alpha=0.3)

    # ── 6. Peak jaw closure per trial ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    shade_trials(ax6, trials, 0, 1)
    if jaw_deg is not None:
        ax6.plot(jaw["timestamp_s"], jaw_deg, color="#76b7b2", linewidth=0.7, zorder=3)
        # mark minimum jaw (tightest grasp) per trial
        for _, row in trials.iterrows():
            mask = (jaw["timestamp_s"] >= row["Unix_Cue_Time"]) & \
                   (jaw["timestamp_s"] <= row["Unix_Place_Time"])
            sub = jaw_deg.values[mask.values] if hasattr(mask, "values") else jaw_deg[mask]
            if len(sub) > 0:
                min_idx = np.argmin(sub)
                t_vals = jaw["timestamp_s"].values[mask.values] if hasattr(mask, "values") \
                         else jaw["timestamp_s"][mask].values
                ax6.scatter(t_vals[min_idx], sub[min_idx],
                            color="#e15759", s=25, zorder=5)
    ax6.set_xlim(t_min - 1, t_max + 1)
    ax6.set_title("Jaw Angle during Trials  (red dot = tightest grasp per trial)")
    ax6.set_xlabel("Unix time (s)")
    ax6.set_ylabel("Jaw angle (°)")
    ax6.legend(handles=legend_patches, fontsize=8)
    ax6.grid(True, alpha=0.2)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Correlate PSM1 robot data with Arduino trial events."
    )
    parser.add_argument("--subject", default=None,
                        help="Subject ID (e.g. fc12, tg14, aa7). "
                             "Auto-resolves PSM1 and Arduino paths.")
    parser.add_argument("--psm1", default=None,
                        help="Explicit path to PSM1 CSV (overrides --subject)")
    parser.add_argument("--arduino", default=None,
                        help="Explicit path to Arduino CSV (overrides --subject)")
    parser.add_argument("--out", default=None,
                        help="Save figure to this path instead of showing. "
                             "Defaults to <subject>_corr.png if --subject is given.")
    args = parser.parse_args()

    if args.subject is None and (args.psm1 is None or args.arduino is None):
        parser.error("Provide --subject OR both --psm1 and --arduino.")

    subject   = args.subject or os.path.basename(args.psm1).replace("_PSM1.csv", "")
    psm1_csv  = args.psm1    or os.path.join(PSM1_DIR,    f"{subject}_PSM1.csv")
    ard_csv   = args.arduino or os.path.join(ARDUINO_DIR, f"experiment_data_{subject}.csv")
    out       = args.out     or (f"{subject}_corr.png" if args.subject else None)

    for path, label in [(psm1_csv, "PSM1"), (ard_csv, "Arduino")]:
        if not os.path.exists(path):
            parser.error(f"{label} file not found: {path}")

    print(f"Subject : {subject}")
    print(f"PSM1    : {psm1_csv}")
    print(f"Arduino : {ard_csv}")
    print(f"Output  : {out or '(interactive window)'}")
    plot(psm1_csv, ard_csv, subject, out)


if __name__ == "__main__":
    main()
