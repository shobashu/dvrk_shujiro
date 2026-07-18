"""
04_record_session.py — Record a D405 color+depth .db3 file during a dVRK session.

Loads camera_settings.json, applies exposure / gain / WB, then shows a live
color + colorized-depth preview.  Press Space to start and stop each recording.
Multiple recordings can be made in one run; each gets its own timestamped file.

A .json metadata file is saved alongside every .db3 with the camera settings,
intrinsics, frame count, and duration — needed by the offline processing scripts.

Note on dVRK kinematics
  This script records the camera only.  PSM poses and gripper state are captured
  separately by the existing ROS2 recording infrastructure
  (scripts/record-surgery-compressed.sh or the task timer node).

Controls:
  Space   start recording  /  stop recording
  q       quit  (prompts confirmation if recording is active)

Output (written to recordings/ next to this script):
  recordings/<name>_YYYYMMDD_HHMMSS.db3    RealSense recording (color + depth)
  recordings/<name>_YYYYMMDD_HHMMSS.json   metadata

Run:
    python3 scripts/depth_camera/04_record_session.py
    python3 scripts/depth_camera/04_record_session.py --name trial_001
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── config ────────────────────────────────────────────────────────────────────
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 30
WARMUP_FRAMES = 30

SETTINGS_PATH  = Path(__file__).parent / "camera_settings.json"
RECORDINGS_DIR = Path(__file__).parent / "recordings"


# ── sensor helpers ────────────────────────────────────────────────────────────
def find_color_sensor(device):
    for s in device.query_sensors():
        if s.supports(rs.option.enable_auto_white_balance):
            return s
    return None


def load_settings(depth_sensor, color_sensor):
    if not SETTINGS_PATH.exists():
        print("  camera_settings.json not found — using camera defaults.")
        return {}
    s = json.loads(SETTINGS_PATH.read_text())
    mapping = [
        ("depth_exposure", depth_sensor, rs.option.exposure),
        ("depth_gain",     depth_sensor, rs.option.gain),
        ("laser_power",    depth_sensor, rs.option.laser_power),
        ("color_exposure", color_sensor, rs.option.exposure),
        ("white_balance",  color_sensor, rs.option.white_balance),
    ]
    for key, sensor, opt in mapping:
        val = s.get(key)
        if val is None:
            continue
        try:
            sensor.set_option(opt, val)
            print(f"  {key} = {val}")
        except Exception as exc:
            print(f"  [!] Could not set {key}: {exc}")
    return s


def intrinsics_to_dict(intr):
    return {
        "fx": intr.fx, "fy": intr.fy,
        "ppx": intr.ppx, "ppy": intr.ppy,
        "width": intr.width, "height": intr.height,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


# ── recording helpers ─────────────────────────────────────────────────────────
def make_stem(name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{ts}" if name else ts


def bag_size_mb(path):
    try:
        return os.path.getsize(path) / 1_048_576
    except OSError:
        return 0.0


def save_metadata(json_path, bag_stem, settings, intrinsics, n_frames, duration_s, depth_scale):
    meta = {
        "bag_file":       bag_stem + ".db3",
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s":     round(duration_s, 3),
        "n_frames":       n_frames,
        "camera_settings": settings,
        "intrinsics":     intrinsics,
        "depth_scale":    depth_scale,   # metres per raw depth unit — sensor-reported, not assumed
    }
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata → {json_path.name}")


# ── overlay ───────────────────────────────────────────────────────────────────
def draw_hud(frame, recording, rec_start, n_frames, bag_path):
    h, w = frame.shape[:2]

    if recording:
        elapsed = time.time() - rec_start
        mins, secs = divmod(int(elapsed), 60)
        size_mb = bag_size_mb(bag_path)

        # Pulsing red dot (toggles every 0.5 s)
        dot_color = (0, 0, 220) if int(elapsed * 2) % 2 == 0 else (60, 60, 180)
        cv2.circle(frame, (18, 18), 9, dot_color, -1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w, 30), (40, 0, 0), -1)
        cv2.putText(frame,
                    f"  REC  {mins:02d}:{secs:02d}  |  {n_frames} frames  |  "
                    f"{size_mb:.1f} MB  |  Space=stop  q=quit",
                    (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 180, 80), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (w, 30), (30, 30, 30), -1)
        cv2.putText(frame,
                    "STANDBY  |  Space = start recording  |  q = quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="",
                    help="Session name prefix for output files (default: timestamp only).")
    args = ap.parse_args()

    RECORDINGS_DIR.mkdir(exist_ok=True)

    # ── pipeline setup ────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)

    print("Starting RealSense pipeline…")
    profile = pipeline.start(cfg)
    device  = profile.get_device()

    depth_sensor = device.first_depth_sensor()
    color_sensor = find_color_sensor(device)

    depth_scale = depth_sensor.get_depth_scale()   # metres per raw depth unit (D405: ~0.0001)
    print(f"  Depth scale = {depth_scale} m/unit")

    depth_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

    print(f"  Loading {SETTINGS_PATH.name}…")
    settings   = load_settings(depth_sensor, color_sensor)
    intrinsics = intrinsics_to_dict(
        profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics()
    )

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)

    print(f"  Warming up ({WARMUP_FRAMES} frames)…", end="", flush=True)
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames(timeout_ms=5000)
    print(" ready.\n")

    print("Press Space to start recording.  Press Space again to stop.")
    print(f"Files will be saved to: {RECORDINGS_DIR.resolve()}\n")

    cv2.namedWindow("D405 — Recording",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("D405 — Recording", STREAM_WIDTH * 2, STREAM_HEIGHT)

    # ── recording state ───────────────────────────────────────────────────────
    recording  = False
    recorder   = None
    rec_start  = 0.0
    n_frames   = 0
    bag_path   = None
    json_path  = None
    stem       = None

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if recording:
                n_frames += 1

            color_img = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            display = np.concatenate([color_img, depth_vis], axis=1)
            draw_hud(display[:, :STREAM_WIDTH],   # overlay on color half only
                     recording, rec_start, n_frames, bag_path)

            cv2.imshow("D405 — Recording", display)
            key = cv2.waitKey(1) & 0xFF

            # ── space: toggle recording ───────────────────────────────────────
            if key == ord(" "):
                if not recording:
                    # Start a new recording
                    stem      = make_stem(args.name)
                    bag_path  = RECORDINGS_DIR / (stem + ".db3")
                    json_path = RECORDINGS_DIR / (stem + ".json")
                    recorder  = rs.recorder(str(bag_path), device)
                    rec_start = time.time()
                    n_frames  = 0
                    recording = True
                    print(f"  ● Recording started → {bag_path.name}")
                else:
                    # Stop current recording
                    duration  = time.time() - rec_start
                    recorder  = None           # deleting closes the bag cleanly
                    recording = False
                    print(f"  ■ Recording stopped  |  {n_frames} frames  |  "
                          f"{duration:.1f} s  |  {bag_size_mb(bag_path):.1f} MB")
                    save_metadata(json_path, stem, settings, intrinsics,
                                  n_frames, duration, depth_scale)
                    print(f"  Saved → {bag_path.name}\n")
                    print("Press Space to start a new recording.  q to quit.\n")

            # ── q: quit ───────────────────────────────────────────────────────
            elif key == ord("q"):
                if recording:
                    print("  Recording is active — stopping before quit…")
                    duration  = time.time() - rec_start
                    recorder  = None
                    recording = False
                    print(f"  ■ {n_frames} frames  |  {duration:.1f} s  |  "
                          f"{bag_size_mb(bag_path):.1f} MB")
                    save_metadata(json_path, stem, settings, intrinsics,
                                  n_frames, duration, depth_scale)
                    print(f"  Saved → {bag_path.name}")
                break

    finally:
        if recorder is not None:
            recorder = None    # ensure bag is closed if something raises
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
