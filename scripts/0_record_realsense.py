#!/usr/bin/env python3
"""
Record color frames from the Intel RealSense depth camera for YOLO training.

Outputs frames in the same layout as 1_extract_frames.py so you can feed the
result directly into the existing 2_prepare_dataset.py → 3_train.py pipeline:

  Step 0  (this script):  python 0_record_realsense.py --out ~/data/frames/trial_rs_001
  Step 1  (skip):         frames are already extracted
  Step 2: python 2_prepare_dataset.py \\
              --labeled  Task_Pad_cylinder_pegs.yolov8.zip \\
              --frames   trial_rs_001_frames.zip \\
              --weights  models/best_v2.pt \\
              --out      data/dataset_rs
  Step 3: python 3_train.py

Output layout:
  <out>/images/left/frame_0000.jpg
  <out>/images/left/frame_0001.jpg
  ...
  <out>/metadata.yaml

Controls:
  Space      toggle recording on / off  (starts paused)
  s          save a single frame (snapshot, works any time)
  q          quit and write metadata
"""

import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_OUT_BASE = Path.home() / "dvrk_recordings" / "realsense"
DEFAULT_FPS      = 5       # frames saved per second (independent of camera FPS)
CAMERA_FPS       = 30      # RealSense stream rate
WIDTH, HEIGHT    = 640, 480


def parse_args():
    p = argparse.ArgumentParser(
        description="Record RealSense color frames for YOLO training"
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output directory (default: ~/dvrk_recordings/realsense/<timestamp>)"
    )
    p.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=f"Frames to save per second (default: {DEFAULT_FPS})"
    )
    p.add_argument(
        "--width",  type=int, default=WIDTH,  help="Stream width  (default: 640)"
    )
    p.add_argument(
        "--height", type=int, default=HEIGHT, help="Stream height (default: 480)"
    )
    p.add_argument(
        "--autostart", action="store_true",
        help="Start recording immediately (default: press Space to start)"
    )
    return p.parse_args()


def make_output_dir(base: str | None) -> Path:
    if base:
        out = Path(base).expanduser()
    else:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = DEFAULT_OUT_BASE / f"trial_{ts}"
    images_dir = out / "images" / "left"
    images_dir.mkdir(parents=True, exist_ok=True)
    return out


def draw_hud(frame, recording: bool, frame_count: int, target_fps: float):
    h, w = frame.shape[:2]
    bar_color = (0, 0, 180) if recording else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (w, 34), bar_color, -1)

    status = "REC" if recording else "PAUSED"
    rec_color = (0, 80, 255) if recording else (160, 160, 160)
    cv2.putText(frame, status, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2, cv2.LINE_AA)

    info = (f"  frames saved: {frame_count}"
            f"  |  {target_fps:.1f} fps"
            f"  |  Space=rec  s=snap  q=quit")
    cv2.putText(frame, info, (70, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (w - 18, 17), 8, (0, 0, 255), -1, cv2.LINE_AA)


def save_frame(color_img: np.ndarray, images_dir: Path, frame_count: int) -> Path:
    path = images_dir / f"frame_{frame_count:04d}.jpg"
    cv2.imwrite(str(path), color_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def main():
    args = parse_args()
    out_dir    = make_output_dir(args.out)
    images_dir = out_dir / "images" / "left"

    print(f"Output directory : {out_dir}")
    print(f"Save rate        : {args.fps} fps")
    print(f"Resolution       : {args.width}x{args.height}")
    print()
    print("Controls: Space = start/stop recording  |  s = snapshot  |  q = quit")
    print()

    # ── RealSense setup ───────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, args.width, args.height,
                         rs.format.z16,  CAMERA_FPS)

    print("Starting RealSense…")
    pipeline.start(config)

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)  # cool colormap

    cv2.namedWindow("RealSense Color", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RealSense Color", args.width, args.height)
    cv2.resizeWindow("RealSense Depth", args.width, args.height)

    recording    = args.autostart
    frame_count  = 0
    last_saved_t = None
    save_interval = 1.0 / args.fps
    start_time   = None

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            now       = color_frame.timestamp / 1e3   # ms → s
            color_img = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            if start_time is None:
                start_time = now

            # ── save frame at target fps ──────────────────────────────────────
            should_save = (
                recording
                and (last_saved_t is None or (now - last_saved_t) >= save_interval)
            )
            if should_save:
                path = save_frame(color_img, images_dir, frame_count)
                if frame_count % 10 == 0:
                    print(f"  Saved frame {frame_count:4d}  →  {path.name}")
                frame_count  += 1
                last_saved_t  = now

            # ── draw HUD ──────────────────────────────────────────────────────
            display = color_img.copy()
            draw_hud(display, recording, frame_count, args.fps)
            cv2.imshow("RealSense Color", display)
            cv2.imshow("RealSense Depth", depth_vis)

            # ── keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
                state_str = "STARTED" if recording else "PAUSED"
                print(f"  Recording {state_str}  (frames saved so far: {frame_count})")
            elif key == ord("s"):
                path = save_frame(color_img, images_dir, frame_count)
                print(f"  Snapshot → {path.name}")
                frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ── write metadata (matches 1_extract_frames.py format) ──────────────────
    metadata = {
        "source":          "realsense_direct",
        "extraction_date": datetime.datetime.now().isoformat(),
        "target_fps":      args.fps,
        "resolution":      f"{args.width}x{args.height}",
        "camera_topics":   {"left": "realsense/color"},
        "frames_extracted": {"left": frame_count},
    }
    meta_path = out_dir / "metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

    print()
    print(f"Done.  Saved {frame_count} frames → {images_dir}")
    print(f"       Metadata  → {meta_path}")
    print()
    print("Next steps:")
    print(f"  1. Zip the frames:  cd {out_dir} && zip -r frames.zip images/")
    print( "  2. Upload frames.zip to Roboflow, label cylinders/pegs, export YOLOv8 zip")
    print( "  3. Run: python 2_prepare_dataset.py \\")
    print( "              --labeled  <roboflow_export>.zip \\")
    print(f"              --frames   {out_dir}/images/frames.zip \\")
    print( "              --weights  models/best_v2.pt \\")
    print( "              --out      data/dataset_realsense")
    print( "  4. Run: python 3_train.py")


if __name__ == "__main__":
    main()
