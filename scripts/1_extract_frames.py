#!/usr/bin/env python3
"""
Step 1 — Extract frames from a ROS 2 bag file.

Reads a .db3 bag, pulls the camera image topic, and saves
sampled frames as .jpg files ready for annotation.

Optionally exports kinematics (joint states / PSM pose) to CSV
alongside the images for later use in scoring.

Usage:
    python 1_extract_frames.py \
        --bag  data/raw_bags/trial_001 \
        --out  data/frames/trial_001 \
        --topic /jhu_mtm/measured_cp \
        --fps  2
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

# ROS 2 imports — only needed when running inside a sourced ROS 2 workspace
try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] ROS 2 not found — running in offline/test mode only.")


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def open_bag(bag_path: str):
    """Open a ROS 2 bag and return (reader, type_map)."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    return reader, type_map


def extract_frames(
    bag_path: str,
    output_dir: str,
    image_topic: str = "/jhu_crsus/left/image_raw",
    kinematic_topics: list = None,
    target_fps: float = 2.0,
    image_quality: int = 95,
):
    """
    Extract sampled frames and optional kinematics from a bag.

    Args:
        bag_path:         Path to the ROS 2 bag directory.
        output_dir:       Where to save images and kinematics CSV.
        image_topic:      ROS topic name for the endoscope camera.
        kinematic_topics: List of ROS topics for joint/pose data.
        target_fps:       How many frames per second to save.
        image_quality:    JPEG quality (0-100).
    """
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS 2 is required for bag extraction.")

    kinematic_topics = kinematic_topics or []

    out_images = Path(output_dir) / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # CSV writer for kinematics
    kin_rows = []

    reader, type_map = open_bag(bag_path)

    # Work out minimum interval between saved frames (nanoseconds)
    min_interval_ns = int(1e9 / target_fps)
    last_saved_ns = -min_interval_ns  # ensure first frame is always saved

    frame_idx = 0
    print(f"[INFO] Extracting from: {bag_path}")
    print(f"[INFO] Image topic: {image_topic}")
    print(f"[INFO] Target fps: {target_fps}  →  saving every {1/target_fps:.2f}s")

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        # ── Image frames ──────────────────────────────────────────────────
        if topic == image_topic:
            if (timestamp_ns - last_saved_ns) < min_interval_ns:
                continue

            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            # Convert ROS Image to OpenCV BGR
            frame = ros_image_to_cv2(msg)
            if frame is None:
                continue

            fname = f"frame_{frame_idx:06d}.jpg"
            fpath = out_images / fname
            cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, image_quality])

            kin_rows.append({
                "frame_file": fname,
                "timestamp_ns": timestamp_ns,
                # Kinematic fields filled in below when topic matches
            })

            last_saved_ns = timestamp_ns
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  Saved {frame_idx} frames…")

        # ── Kinematics ────────────────────────────────────────────────────
        elif topic in kinematic_topics:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            # Store raw data — extend kin_rows or write a separate CSV per topic
            # (expand this based on your specific message types)
            _ = msg  # placeholder

    # Write kinematics CSV
    if kin_rows:
        csv_path = Path(output_dir) / "kinematics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=kin_rows[0].keys())
            writer.writeheader()
            writer.writerows(kin_rows)
        print(f"[INFO] Kinematics CSV → {csv_path}")

    print(f"[DONE] {frame_idx} frames saved to {out_images}")


def ros_image_to_cv2(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image to a BGR OpenCV array."""
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8", "mono8"):
        channels = 1 if enc == "mono8" else 3
        frame = data.reshape((msg.height, msg.width, channels))
        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif enc in ("bayer_rggb8", "bayer_bggr8", "bayer_gbrg8", "bayer_grbg8"):
        frame = data.reshape((msg.height, msg.width))
        code_map = {
            "bayer_rggb8": cv2.COLOR_BayerRG2BGR,
            "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
            "bayer_gbrg8": cv2.COLOR_BayerGB2BGR,
            "bayer_grbg8": cv2.COLOR_BayerGR2BGR,
        }
        frame = cv2.cvtColor(frame, code_map[enc])
    else:
        print(f"[WARN] Unsupported encoding: {enc}")
        return None

    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a ROS 2 bag.")
    p.add_argument("--bag",   required=True,  help="Path to rosbag2 directory")
    p.add_argument("--out",   required=True,  help="Output directory")
    p.add_argument("--topic", default="/jhu_crsus/left/image_raw",
                   help="Image topic name")
    p.add_argument("--kin_topics", nargs="*", default=[],
                   help="Kinematic topic names (space-separated)")
    p.add_argument("--fps",   type=float, default=2.0,
                   help="Frames per second to extract (default: 2)")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality 0-100 (default: 95)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(
        bag_path=args.bag,
        output_dir=args.out,
        image_topic=args.topic,
        kinematic_topics=args.kin_topics,
        target_fps=args.fps,
        image_quality=args.quality,
    )
