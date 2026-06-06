#!/usr/bin/env python3
"""
Plot PSM1 recording data from tg14_PSM1.csv (or any similarly structured CSV).

Panels:
  1. Cartesian position (x, y, z)
  2. Insertion depth over time  ← clinically meaningful
  3. Jaw angle over time        ← grasp/release events
  4. Cartesian speed + jaw angle overlay  ← motion vs. grasping
  5. Joint positions (all 6 joints)
  6. 3-D trajectory coloured by jaw angle  ← where grasps happen in space
  7. XY top-down trajectory coloured by speed
  8. Jaw effort (force proxy)

Usage:
    python3 plot_psm1.py /home/stanford/dvrk_recordings/training/tg14_PSM1.csv
    python3 plot_psm1.py tg14_PSM1.csv --out tg14_PSM1.png
    python3 plot_psm1.py tg14_PSM1.csv --start 10 --end 60
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

JOINT_NAMES = ["yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw"]
JOINT_POS_COLS = [f"{j}_pos" for j in JOINT_NAMES]


def load(path: str):
    df = pd.read_csv(path)
    t0 = df["timestamp_s"].iloc[0]
    df["time_s"] = df["timestamp_s"] - t0
    return df


def filter_time(df, start, end):
    if start is not None:
        df = df[df["time_s"] >= start]
    if end is not None:
        df = df[df["time_s"] <= end]
    return df


def detect_grasps(jaw, threshold_rad=-0.1):
    """Return boolean series: True when jaw is closed (grasping)."""
    return jaw < threshold_rad


def plot(path: str, out: str | None, start: float | None, end: float | None):
    df = load(path)

    cp  = filter_time(df[df["topic"] == "measured_cp"].copy(),         start, end)
    cv  = filter_time(df[df["topic"] == "measured_cv"].copy(),         start, end)
    js  = filter_time(df[df["topic"] == "measured_js"].copy(),         start, end)
    jaw = filter_time(df[df["topic"] == "jaw/measured_js"].copy(),     start, end)

    fig = plt.figure(figsize=(20, 26))
    fig.suptitle(f"PSM1 Recording — {path.split('/')[-1]}", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. Cartesian position ─────────────────────────────────────────────────
    ax_pos = fig.add_subplot(gs[0, 0])
    for col, label, color in zip(
        ["pos_x", "pos_y", "pos_z"], ["X", "Y", "Z"],
        ["#e15759", "#4e79a7", "#59a14f"]
    ):
        ax_pos.plot(cp["time_s"], cp[col], label=label, linewidth=0.8, color=color)
    ax_pos.set_title("Cartesian Position (m)")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    # ── 2. Insertion depth ────────────────────────────────────────────────────
    ax_ins = fig.add_subplot(gs[0, 1])
    if not js.empty and "insertion_pos" in js.columns:
        ax_ins.plot(js["time_s"], js["insertion_pos"], color="#f28e2b", linewidth=0.9)
        ax_ins.fill_between(js["time_s"], js["insertion_pos"],
                            js["insertion_pos"].min(), alpha=0.15, color="#f28e2b")
    ax_ins.set_title("Insertion Depth (m)")
    ax_ins.set_xlabel("Time (s)")
    ax_ins.set_ylabel("Insertion (m)")
    ax_ins.grid(True, alpha=0.3)

    # ── 3. Jaw angle (grasp events) ───────────────────────────────────────────
    ax_jaw = fig.add_subplot(gs[1, 0])
    if not jaw.empty:
        grasping = detect_grasps(jaw["jaw_pos"])
        ax_jaw.plot(jaw["time_s"], np.degrees(jaw["jaw_pos"]),
                    color="#76b7b2", linewidth=0.9, zorder=3)
        ax_jaw.fill_between(jaw["time_s"], np.degrees(jaw["jaw_pos"]),
                            where=grasping, alpha=0.35, color="#e15759",
                            label="Grasping (jaw < −5.7°)")
        ax_jaw.axhline(np.degrees(-0.1), color="#e15759", linewidth=0.8,
                       linestyle="--", alpha=0.7)
    ax_jaw.set_title("Jaw Angle — Grasp Events")
    ax_jaw.set_xlabel("Time (s)")
    ax_jaw.set_ylabel("Jaw angle (°)")
    ax_jaw.legend(fontsize=8)
    ax_jaw.grid(True, alpha=0.3)

    # ── 4. Speed vs jaw overlay ───────────────────────────────────────────────
    ax_spd = fig.add_subplot(gs[1, 1])
    ax_jaw2 = ax_spd.twinx()
    if not cv.empty:
        speed = np.sqrt(cv["lin_x"]**2 + cv["lin_y"]**2 + cv["lin_z"]**2)
        ax_spd.plot(cv["time_s"], speed, color="#4e79a7", linewidth=0.8, label="Speed")
    if not jaw.empty:
        ax_jaw2.plot(jaw["time_s"], np.degrees(jaw["jaw_pos"]),
                     color="#e15759", linewidth=0.8, alpha=0.7, label="Jaw (°)")
    ax_spd.set_title("Cartesian Speed  vs  Jaw Angle")
    ax_spd.set_xlabel("Time (s)")
    ax_spd.set_ylabel("Speed (m/s)", color="#4e79a7")
    ax_jaw2.set_ylabel("Jaw angle (°)", color="#e15759")
    lines = [Line2D([0], [0], color="#4e79a7", label="Speed (m/s)"),
             Line2D([0], [0], color="#e15759", label="Jaw angle (°)")]
    ax_spd.legend(handles=lines, fontsize=8)
    ax_spd.grid(True, alpha=0.3)

    # ── 5. All joint positions ────────────────────────────────────────────────
    ax_jp = fig.add_subplot(gs[2, :])
    colors6 = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f", "#b07aa1", "#9c755f"]
    if not js.empty:
        for col, name, c in zip(JOINT_POS_COLS, JOINT_NAMES, colors6):
            if col in js.columns:
                ax_jp.plot(js["time_s"], js[col], label=name, linewidth=0.8, color=c)
    ax_jp.set_title("Joint Positions (rad / m for insertion)")
    ax_jp.set_xlabel("Time (s)")
    ax_jp.set_ylabel("Position")
    ax_jp.legend(fontsize=8, ncol=3)
    ax_jp.grid(True, alpha=0.3)

    # ── 6. 3-D trajectory coloured by jaw angle ───────────────────────────────
    ax_3d = fig.add_subplot(gs[3, 0], projection="3d")
    if not cp.empty and not jaw.empty:
        jaw_interp = np.interp(cp["time_s"], jaw["time_s"], jaw["jaw_pos"])
        sc = ax_3d.scatter(cp["pos_x"], cp["pos_y"], cp["pos_z"],
                           c=np.degrees(jaw_interp), cmap="RdYlGn_r",
                           s=1.5, alpha=0.7)
        cb = fig.colorbar(sc, ax=ax_3d, label="Jaw angle (°)", shrink=0.55, pad=0.1)
        cb.ax.tick_params(labelsize=7)
    elif not cp.empty:
        ax_3d.scatter(cp["pos_x"], cp["pos_y"], cp["pos_z"],
                      c=cp["time_s"], cmap="viridis", s=1.5, alpha=0.7)
    ax_3d.set_title("3-D Trajectory\n(colour = jaw angle, red=closed)", fontsize=9)
    ax_3d.set_xlabel("X (m)", fontsize=7)
    ax_3d.set_ylabel("Y (m)", fontsize=7)
    ax_3d.set_zlabel("Z (m)", fontsize=7)
    ax_3d.tick_params(labelsize=6)

    # ── 7. XY top-down trajectory coloured by speed ───────────────────────────
    ax_xy = fig.add_subplot(gs[3, 1])
    if not cp.empty and not cv.empty:
        speed_cp = np.interp(cp["time_s"], cv["time_s"],
                             np.sqrt(cv["lin_x"]**2 + cv["lin_y"]**2 + cv["lin_z"]**2))
        sc2 = ax_xy.scatter(cp["pos_x"], cp["pos_y"],
                            c=speed_cp, cmap="plasma", s=1.5, alpha=0.7)
        cb2 = fig.colorbar(sc2, ax=ax_xy, label="Speed (m/s)")
        cb2.ax.tick_params(labelsize=7)
    elif not cp.empty:
        ax_xy.plot(cp["pos_x"], cp["pos_y"], linewidth=0.6, color="#4e79a7")
    ax_xy.set_title("XY Top-down Trajectory\n(colour = speed)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot PSM1 CSV recording.")
    parser.add_argument("csv", help="Path to the CSV file")
    parser.add_argument("--out", default=None, help="Save figure instead of showing")
    parser.add_argument("--start", type=float, default=None, help="Start time offset (s)")
    parser.add_argument("--end",   type=float, default=None, help="End time offset (s)")
    args = parser.parse_args()

    plot(args.csv, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
