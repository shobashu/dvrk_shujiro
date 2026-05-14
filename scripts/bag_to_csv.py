#!/usr/bin/env python3
"""
Convert a dVRK ROS2 bag to two CSV files: one for PSM1, one for MTMR.

Camera topics are skipped. Each row has a timestamp and the subtopic name
so you can filter easily (e.g. keep only measured_cp rows in a spreadsheet).

Usage:
    python3 scripts/bag_to_csv.py ~/dvrk_recordings/training/trial_01
    python3 scripts/bag_to_csv.py ~/dvrk_recordings/training/trial_01 --out ~/my_csvs
"""

import argparse
import csv
from pathlib import Path

ARMS = ["PSM1", "MTMR"]

# Topics that carry useful kinematics data (camera topics are ignored)
WANTED_SUBTOPICS = {"measured_cp", "measured_cv", "measured_js", "jaw/measured_js"}


def parse_pose_stamped(msg, row):
    p = msg.pose.position
    o = msg.pose.orientation
    row.update({
        "pos_x": p.x, "pos_y": p.y, "pos_z": p.z,
        "ori_x": o.x, "ori_y": o.y, "ori_z": o.z, "ori_w": o.w,
    })


def parse_twist_stamped(msg, row):
    lv = msg.twist.linear
    av = msg.twist.angular
    row.update({
        "lin_x": lv.x, "lin_y": lv.y, "lin_z": lv.z,
        "ang_x": av.x, "ang_y": av.y, "ang_z": av.z,
    })


def parse_joint_state(msg, row):
    for i, name in enumerate(msg.name):
        col = name.replace("/", "_").replace(" ", "_")
        if i < len(msg.position):
            row[f"{col}_pos"] = msg.position[i]
        if i < len(msg.velocity):
            row[f"{col}_vel"] = msg.velocity[i]
        if i < len(msg.effort):
            row[f"{col}_eff"] = msg.effort[i]


def extract_row(subtopic, msg, timestamp_ns):
    ts = timestamp_ns * 1e-9
    row = {"timestamp_s": f"{ts:.6f}", "topic": subtopic}

    if subtopic == "measured_cp":
        parse_pose_stamped(msg, row)
    elif subtopic == "measured_cv":
        parse_twist_stamped(msg, row)
    elif subtopic in ("measured_js", "jaw/measured_js"):
        parse_joint_state(msg, row)

    return row


def write_csv(out_file, arm_rows):
    # Preserve column order: timestamp and topic first, then data columns
    fieldnames = ["timestamp_s", "topic"]
    seen = set(fieldnames)
    for r in arm_rows:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(arm_rows)


def main():
    parser = argparse.ArgumentParser(description="Extract PSM1/MTMR topics from a ROS2 bag to CSV.")
    parser.add_argument("bag", help="Path to the bag directory (e.g. ~/dvrk_recordings/training/trial_01)")
    parser.add_argument("--out", default=None, help="Output directory (default: same as bag parent)")
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_dir = Path(args.out).expanduser() if args.out else bag_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        print("ERROR: rosbags not installed. Run:  pip install rosbags")
        raise SystemExit(1)

    # Try Jazzy typestore, fall back to latest available
    try:
        typestore = get_typestore(Stores.ROS2_JAZZY)
    except AttributeError:
        typestore = get_typestore(Stores.LATEST)

    rows = {arm: [] for arm in ARMS}

    print(f"Reading bag: {bag_path}")
    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            topic = conn.topic
            arm = next((a for a in ARMS if topic.startswith(f"/{a}/")), None)
            if arm is None:
                continue

            subtopic = topic[len(f"/{arm}/"):]
            if subtopic not in WANTED_SUBTOPICS:
                continue

            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            rows[arm].append(extract_row(subtopic, msg, timestamp))

    for arm in ARMS:
        arm_rows = sorted(rows[arm], key=lambda r: r["timestamp_s"])
        if not arm_rows:
            print(f"  [{arm}] No data found, skipping.")
            continue

        out_file = out_dir / f"{bag_path.name}_{arm}.csv"
        write_csv(out_file, arm_rows)
        print(f"  [{arm}] {len(arm_rows):,} rows  →  {out_file}")


if __name__ == "__main__":
    main()
