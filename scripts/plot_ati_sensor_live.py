#!/usr/bin/env python3
"""
Live plot of ATI force/torque sensor data — reads the CSV written by
ati_sensor_udp_receiver.py. No ROS2 required; works in any Python environment.

Usage:
    # Pass the CSV file being written by the receiver:
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv

    # Adjust visible time window or refresh rate:
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --window 10
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --rate 30
"""

import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV file being written by ati_sensor_udp_receiver.py")
    p.add_argument("--window", type=float, default=10.0, help="Visible time window in seconds (default 10)")
    p.add_argument("--rate",   type=int,   default=20,   help="Refresh rate in Hz (default 20)")
    return p.parse_args()


def read_csv_safe(path):
    """Read CSV, silently dropping any incomplete last line being written."""
    try:
        df = pd.read_csv(path, comment="#")
        df.columns = df.columns.str.strip()
        df = df.dropna()
        return df
    except Exception:
        return None


def main():
    args = parse_args()

    fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"ATI Sensor — live  ({args.csv})", fontsize=11)

    lf = {c: ax_f.plot([], [], label=c, linewidth=1)[0] for c in ("Fx", "Fy", "Fz")}
    lt = {c: ax_t.plot([], [], label=c, linewidth=1)[0] for c in ("Tx", "Ty", "Tz")}

    for ax, ylabel in ((ax_f, "Force (N)"), (ax_t, "Torque (N·m)")):
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_t.set_xlabel("Time (s)")

    status_text = ax_f.text(
        0.01, 0.97, "Waiting for data...",
        transform=ax_f.transAxes, fontsize=8,
        verticalalignment="top", color="gray")

    def update(_frame):
        df = read_csv_safe(args.csv)
        if df is None or df.empty:
            return list(lf.values()) + list(lt.values())

        # Time relative to first sample
        t = df["timestamp_s"].values - df["timestamp_s"].iloc[0]
        now = t[-1]
        t_min = now - args.window

        # Only feed the visible window to the lines (keeps rendering fast)
        mask = t >= t_min
        tv = t[mask]

        for line, col in zip(lf.values(), ("fx", "fy", "fz")):
            line.set_data(tv, df[col].values[mask])
        for line, col in zip(lt.values(), ("tx", "ty", "tz")):
            line.set_data(tv, df[col].values[mask])

        ax_f.set_xlim(t_min, now)
        ax_t.set_xlim(t_min, now)

        vis_f = pd.concat([df["fx"][mask], df["fy"][mask], df["fz"][mask]])
        vis_t = pd.concat([df["tx"][mask], df["ty"][mask], df["tz"][mask]])

        if not vis_f.empty:
            pad = max((vis_f.max() - vis_f.min()) * 0.1, 0.5)
            ax_f.set_ylim(vis_f.min() - pad, vis_f.max() + pad)
        if not vis_t.empty:
            pad = max((vis_t.max() - vis_t.min()) * 0.1, 0.01)
            ax_t.set_ylim(vis_t.min() - pad, vis_t.max() + pad)

        status_text.set_text(f"{len(df)} samples  |  t={now:.1f}s")
        status_text.set_color("black")

        return list(lf.values()) + list(lt.values())

    interval_ms = int(1000 / args.rate)
    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
