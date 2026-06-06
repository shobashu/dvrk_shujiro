#!/usr/bin/env python3
"""
Plot MTMR recording data from tg14_MTMR.csv (or any similarly structured CSV).

Topics plotted:
  measured_cp  — Cartesian position (x, y, z) and orientation quaternion (x, y, z, w)
  measured_cv  — Cartesian linear velocity (x, y, z) and angular velocity (x, y, z)
  measured_js  — joint positions for all 6 joints

Usage:
    python3 plot_mtmr.py /home/stanford/dvrk_recordings/training/tg14_MTMR.csv
    python3 plot_mtmr.py tg14_MTMR.csv --out tg14_MTMR.png
    python3 plot_mtmr.py tg14_MTMR.csv --start 10 --end 60
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

JOINT_NAMES = [
    "outer_yaw", "shoulder_pitch", "elbow_pitch",
    "wrist_platform", "wrist_pitch", "wrist_yaw",
]
JOINT_POS_COLS = [f"{j}_pos" for j in JOINT_NAMES]


def load(path: str):
    df = pd.read_csv(path)
    df["time_s"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    return df


def filter_time(df, start, end):
    if start is not None:
        df = df[df["time_s"] >= start]
    if end is not None:
        df = df[df["time_s"] <= end]
    return df


def plot(path: str, out: str | None, start: float | None, end: float | None):
    df = load(path)

    cp = filter_time(df[df["topic"] == "measured_cp"].copy(), start, end)
    cv = filter_time(df[df["topic"] == "measured_cv"].copy(), start, end)
    js = filter_time(df[df["topic"] == "measured_js"].copy(), start, end)

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(f"MTMR Recording — {path.split('/')[-1]}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Cartesian position ────────────────────────────────────────────────────
    ax_pos = fig.add_subplot(gs[0, 0])
    for col, label in zip(["pos_x", "pos_y", "pos_z"], ["x", "y", "z"]):
        ax_pos.plot(cp["time_s"], cp[col], label=label, linewidth=0.8)
    ax_pos.set_title("Cartesian Position (m)")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(loc="upper right", fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    # ── Orientation quaternion ────────────────────────────────────────────────
    ax_ori = fig.add_subplot(gs[0, 1])
    for col, label in zip(["ori_x", "ori_y", "ori_z", "ori_w"], ["qx", "qy", "qz", "qw"]):
        ax_ori.plot(cp["time_s"], cp[col], label=label, linewidth=0.8)
    ax_ori.set_title("Orientation (quaternion)")
    ax_ori.set_xlabel("Time (s)")
    ax_ori.legend(loc="upper right", fontsize=8)
    ax_ori.grid(True, alpha=0.3)

    # ── Cartesian path speed ──────────────────────────────────────────────────
    ax_speed = fig.add_subplot(gs[1, 0])
    if not cv.empty:
        speed = np.sqrt(cv["lin_x"]**2 + cv["lin_y"]**2 + cv["lin_z"]**2)
        ax_speed.plot(cv["time_s"], speed, color="steelblue", linewidth=0.8)
    ax_speed.set_title("Cartesian Speed |v| (m/s)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, alpha=0.3)

    # ── Angular velocity ──────────────────────────────────────────────────────
    ax_ang = fig.add_subplot(gs[1, 1])
    if not cv.empty:
        for col, label in zip(["ang_x", "ang_y", "ang_z"], ["ωx", "ωy", "ωz"]):
            ax_ang.plot(cv["time_s"], cv[col], label=label, linewidth=0.8)
    ax_ang.set_title("Angular Velocity (rad/s)")
    ax_ang.set_xlabel("Time (s)")
    ax_ang.legend(loc="upper right", fontsize=8)
    ax_ang.grid(True, alpha=0.3)

    # ── Joint positions ───────────────────────────────────────────────────────
    ax_jp = fig.add_subplot(gs[2, :])
    if not js.empty:
        for col, name in zip(JOINT_POS_COLS, JOINT_NAMES):
            if col in js.columns:
                ax_jp.plot(js["time_s"], js[col], label=name, linewidth=0.8)
    ax_jp.set_title("Joint Positions (rad or m)")
    ax_jp.set_xlabel("Time (s)")
    ax_jp.set_ylabel("Position")
    ax_jp.legend(loc="upper right", fontsize=8, ncol=3)
    ax_jp.grid(True, alpha=0.3)

    # ── 3-D trajectory ────────────────────────────────────────────────────────
    ax_3d = fig.add_subplot(gs[3, :], projection="3d")
    if not cp.empty:
        sc = ax_3d.scatter(
            cp["pos_x"], cp["pos_y"], cp["pos_z"],
            c=cp["time_s"], cmap="viridis", s=1, alpha=0.6,
        )
        fig.colorbar(sc, ax=ax_3d, label="Time (s)", shrink=0.5)
    ax_3d.set_title("3-D Cartesian Trajectory (colour = time)")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot MTMR CSV recording.")
    parser.add_argument("csv", help="Path to the CSV file")
    parser.add_argument("--out", default=None, help="Save figure to this path instead of showing")
    parser.add_argument("--start", type=float, default=None, help="Start time offset (s)")
    parser.add_argument("--end", type=float, default=None, help="End time offset (s)")
    args = parser.parse_args()

    plot(args.csv, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
