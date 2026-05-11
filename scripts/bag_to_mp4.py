#!/usr/bin/env python3
"""
Convert a ROS2 bag to MP4 (supports both image_raw and compressed topics).

Timestamps from the bag are used to drive output timing, so dropped frames
during recording are filled by repeating the last frame — the video plays
at true 1:1 wall-clock speed with no slowdowns or speedups.

Usage:
  python3.12 -s bag_to_mp4.py <bag_dir> [output_dir]

Example:
  python3.12 -s bag_to_mp4.py ~/dvrk_recordings/training/testwithcv2

source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && \
  export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 && \
  python3.12 -s ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts/bag_to_mp4.py \
    ~/dvrk_recordings/training/testwithcv2
"""

import sys
import os
import numpy as np
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CompressedImage
from rosidl_runtime_py.utilities import get_message


OUT_FPS = 30.0  # output video fps; timestamps control actual motion speed


def open_reader(bag_path):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def topic_type_map(reader):
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def raw_to_bgr(msg):
    enc = msg.encoding.lower()
    arr = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8",):
        frame = arr.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif enc in ("bgr8",):
        return arr.reshape(msg.height, msg.width, 3)
    elif enc in ("mono8",):
        gray = arr.reshape(msg.height, msg.width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif enc in ("yuv422", "yuv422_yuy2", "yuyv"):
        yuyv = arr.reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    elif enc in ("bayer_rggb8",):
        bayer = arr.reshape(msg.height, msg.width)
        return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
    else:
        channels = len(msg.data) // (msg.height * msg.width)
        return arr.reshape(msg.height, msg.width, channels)


def convert(bag_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cameras = {
        "left":  ["/camera_left/image_raw",   "/camera_left/compressed"],
        "right": ["/camera_right/image_raw",  "/camera_right/compressed"],
    }

    # First pass: find which topics exist
    reader = open_reader(bag_path)
    types  = topic_type_map(reader)
    del reader

    active = {}
    for cam, candidates in cameras.items():
        for topic in candidates:
            if topic in types:
                active[cam] = (topic, types[topic])
                print(f"  {cam}: {topic}  [{types[topic]}]")
                break
        else:
            print(f"  {cam}: no topic found, skipping")

    if not active:
        print("No camera topics found in bag.")
        sys.exit(1)

    out_dt_ns = int(1e9 / OUT_FPS)

    # Per-camera state for timestamp-based rendering
    writers  = {}
    last_frm = {cam: None for cam in active}
    next_out = {cam: None for cam in active}  # next output slot (nanoseconds)
    counts   = {cam: 0    for cam in active}

    def flush_to(cam, t_end_ns):
        """Repeat last_frm[cam] for every output slot before t_end_ns."""
        if last_frm[cam] is None or next_out[cam] is None:
            return
        while next_out[cam] < t_end_ns:
            writers[cam].write(last_frm[cam])
            counts[cam] += 1
            next_out[cam] += out_dt_ns

    # Second pass: stream frames, filling gaps with repeated frames
    reader = open_reader(bag_path)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t, _ in active.values()]))

    while reader.has_next():
        topic, data, t_bag = reader.read_next()

        for cam, (active_topic, type_str) in active.items():
            if topic != active_topic:
                continue

            msg_type = get_message(type_str)
            msg      = deserialize_message(data, msg_type)

            if "CompressedImage" in type_str:
                buf   = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            else:
                frame = raw_to_bgr(msg)

            if frame is None:
                continue

            if cam not in writers:
                h, w = frame.shape[:2]
                out_path    = os.path.join(output_dir, f"{cam}.mp4")
                writers[cam] = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (w, h)
                )
                next_out[cam] = t_bag
                print(f"  Writing {cam}.mp4  ({w}x{h} @ {OUT_FPS} fps)")

            # Fill all output slots before this frame's arrival with the previous frame
            flush_to(cam, t_bag)

            # Park the new frame — it will be written when the next frame arrives
            last_frm[cam] = frame

    # Write the final frame once for each camera
    for cam in active:
        if cam in writers and last_frm[cam] is not None:
            writers[cam].write(last_frm[cam])
            counts[cam] += 1

    for cam, writer in writers.items():
        writer.release()
        out_path = os.path.join(output_dir, f"{cam}.mp4")
        size     = os.path.getsize(out_path) / 1e6
        print(f"  {cam}.mp4  — {counts[cam]} frames, {size:.1f} MB")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bag_path   = os.path.expanduser(sys.argv[1])
    output_dir = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else bag_path

    print(f"Bag:    {bag_path}")
    print(f"Output: {output_dir}")
    print()
    convert(bag_path, output_dir)
