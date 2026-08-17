#!/usr/bin/env python3
"""
Live plot of ATI force/torque sensor data — reads the CSV written by
ati_sensor_udp_receiver.py. No ROS2 required; works in any Python environment.

The curve color transitions green → yellow → red as force increases toward
--max-force. A colorbar on the right shows the scale.

Usage:
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --window 10
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --max-force 8
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --smooth 5 --y-max 10
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV file being written by ati_sensor_udp_receiver.py")
    p.add_argument("--window",    type=float, default=10.0, help="Visible time window in seconds (default 10)")
    p.add_argument("--rate",      type=int,   default=20,   help="Refresh rate in Hz (default 20)")
    p.add_argument("--max-force", type=float, default=5.0,  help="Force value (N) that maps to full red (default 5)")
    p.add_argument("--smooth",    type=int,   default=5,
                   help="Trailing moving-average window (samples) to reduce sensor noise, "
                        "1 = no smoothing (default 5)")
    p.add_argument("--y-max",     type=float, default=10.0, help="Fixed y-axis upper limit in Newtons (default 10)")
    return p.parse_args()


def read_csv_safe(path):
    try:
        df = pd.read_csv(path, comment="#")
        df.columns = df.columns.str.strip()
        df = df.dropna()
        return df
    except Exception:
        return None


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def main():
    args = parse_args()

    plt.rcParams.update({"font.size": 13})

    cmap = cm.RdYlGn_r  # green (low) → yellow → red (high)
    norm = Normalize(vmin=0, vmax=args.max_force)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("ATI Sensor — Applied Force (live)", fontsize=16, fontweight="bold")

    lc = LineCollection([], cmap=cmap, norm=norm, linewidth=3, zorder=3)
    ax.add_collection(lc)

    ax.set_ylabel("Total Force  ‖F‖  (N)", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(True, alpha=0.3)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Force (N)", fontsize=12)

    mag_text = ax.text(
        0.01, 0.97, "Waiting for data...",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", color="gray",
        fontweight="bold"
    )

    def update(_frame):
        df = read_csv_safe(args.csv)
        if df is None or df.empty:
            return [lc]

        t = df["timestamp_s"].values - df["timestamp_s"].iloc[0]
        now = t[-1]
        t_min = now - args.window
        mask = t >= t_min
        tv = t[mask]

        mag = np.sqrt(
            df["fx"].values[mask] ** 2 +
            df["fy"].values[mask] ** 2 +
            df["fz"].values[mask] ** 2
        )
        if args.smooth > 1 and mag.size:
            # Trailing moving average — smooths sensor/electrical noise.
            # min_periods=1 so the window fills in gradually at the start
            # instead of producing NaNs.
            mag = pd.Series(mag).rolling(window=args.smooth, min_periods=1).mean().values

        if mag.size > 1:
            segs = make_segments(tv, mag)
            lc.set_segments(segs)
            lc.set_array(mag[:-1])

        ax.set_xlim(t_min, now)
        ax.set_ylim(0, args.y_max)

        current_mag = mag[-1] if mag.size else 0.0
        color = cmap(norm(current_mag))
        mag_text.set_text(f"‖F‖ = {current_mag:.3f} N  |  t = {now:.1f} s")
        mag_text.set_color(color)

        return [lc]

    interval_ms = int(1000 / args.rate)
    ani = animation.FuncAnimation(  # noqa: F841
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
