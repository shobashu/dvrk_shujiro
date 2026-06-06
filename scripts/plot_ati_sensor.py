#!/usr/bin/env python3
"""
Plot ATI force/torque sensor data from a CSV produced by ati_sensor_udp_receiver.py.

Usage:
    python3 plot_ati_sensor.py ati_sensor_20260603_120000.csv
    python3 plot_ati_sensor.py ati_sensor_20260603_120000.csv --out ati_plot.png
    python3 plot_ati_sensor.py ati_sensor_20260603_120000.csv --start 5 --end 30
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV file from ati_sensor_udp_receiver.py")
    p.add_argument("--out",   default=None,  help="Save plot to this file instead of showing it")
    p.add_argument("--start", type=float, default=None, help="Start time offset (s) to zoom in")
    p.add_argument("--end",   type=float, default=None, help="End time offset (s) to zoom in")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv, comment="#")
    df.columns = df.columns.str.strip()

    # Convert absolute timestamp to seconds from start
    df["t"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]

    if args.start is not None:
        df = df[df["t"] >= args.start]
    if args.end is not None:
        df = df[df["t"] <= args.end]

    if df.empty:
        print("No data in the requested time range.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"ATI Sensor — {args.csv}", fontsize=12)

    # Force
    ax = axes[0]
    ax.plot(df["t"], df["fx"], label="Fx", linewidth=1)
    ax.plot(df["t"], df["fy"], label="Fy", linewidth=1)
    ax.plot(df["t"], df["fz"], label="Fz", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Torque
    ax = axes[1]
    ax.plot(df["t"], df["tx"], label="Tx", linewidth=1)
    ax.plot(df["t"], df["ty"], label="Ty", linewidth=1)
    ax.plot(df["t"], df["tz"], label="Tz", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Torque (N·m)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
