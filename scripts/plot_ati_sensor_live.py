#!/usr/bin/env python3
"""
Live plot of ATI force/torque sensor data — reads the CSV written by
ati_sensor_udp_receiver.py. No ROS2 required; works in any Python environment.

The curve color transitions green → yellow → red as force increases toward
--max-force. A colorbar on the right shows the scale.

Measures (not assumes) its own display latency: every frame it computes
data_age_s (how old the newest plotted sample already was when this frame
drew it — receiver-write to plot-visible) and read_time_s (how long re-
parsing the growing CSV took). Both are logged to --delay-log every frame,
and a summary (mean/std/min/max) prints to the terminal when you close the
plot window.

Usage:
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --window 10
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --max-force 8
    python3 plot_ati_sensor_live.py ati_sensor_20260603_120000.csv --smooth 5 --y-max 10
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

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
    p.add_argument("--delay-log", default=None,
                   help="Where to log measured data_age_s/read_time_s per frame "
                        "(default: <csv>_delay_log.csv)")
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

    delay_log_path = args.delay_log or f"{Path(args.csv).stem}_delay_log.csv"
    delay_log_file = open(delay_log_path, "w", newline="")
    delay_log_writer = csv.writer(delay_log_file)
    delay_log_writer.writerow(["unix_time", "data_age_s", "read_time_s", "row_count"])
    delay_samples = {"data_age_s": [], "read_time_s": []}

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
        read_t0 = time.time()
        df = read_csv_safe(args.csv)
        read_time_s = time.time() - read_t0
        if df is None or df.empty:
            return [lc]

        # Measured (not assumed) delay: how old was the newest sample,
        # in wall-clock terms, by the time this frame actually drew it.
        # timestamp_s is a raw time.time() value written by
        # ati_sensor_udp_receiver.py, so this captures UDP-to-file-to-here
        # end to end, including any file-buffering delay on the writer side.
        data_age_s = time.time() - df["timestamp_s"].iloc[-1]
        delay_samples["data_age_s"].append(data_age_s)
        delay_samples["read_time_s"].append(read_time_s)
        delay_log_writer.writerow([f"{time.time():.3f}", f"{data_age_s:.4f}",
                                    f"{read_time_s:.4f}", len(df)])

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
        mag_text.set_text(f"‖F‖ = {current_mag:.3f} N  |  t = {now:.1f} s  |  "
                           f"data age = {data_age_s * 1000:.0f} ms")
        mag_text.set_color(color)

        return [lc]

    def _print_summary(*_args):
        def stats(name):
            vals = delay_samples[name]
            if not vals:
                return f"  {name}: no samples"
            return (f"  {name}: mean={statistics.mean(vals):.4f}s "
                    f"± {statistics.pstdev(vals) if len(vals) > 1 else 0.0:.4f}s "
                    f"(min {min(vals):.4f}s, max {max(vals):.4f}s, n={len(vals)})")
        print(f"\n── Measured display latency (this session) ──")
        print(stats("data_age_s"))
        print(stats("read_time_s"))
        print(f"  Log saved: {delay_log_path}")
        delay_log_file.flush()
        delay_log_file.close()

    interval_ms = int(1000 / args.rate)
    ani = animation.FuncAnimation(  # noqa: F841
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    fig.canvas.mpl_connect("close_event", _print_summary)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
