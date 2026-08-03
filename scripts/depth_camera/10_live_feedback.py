"""
10_live_feedback.py — Preview tool: Space to record one trial, auto-analyze, see the graph.

No Arduino, no signal file, no multi-cylinder handling — just a live camera
preview where Space starts/stops a recording (same idea as
04_record_session.py), and on stop it automatically runs the existing
pipeline (05 extract -> 06 --auto --reuse-prompts -> 07 trajectory) and
shows the resulting 3-D trajectory graph in a window. Built to let you see
what the depth-camera feedback actually looks like before wiring it into a
real multi-trial/Arduino session.

Single-cylinder only (one --reuse-prompts template) — the 4-cylinder
--templates-dir routing from 09_live_trial_monitor.py isn't used here.

Reuses process_trial()/build_group_figure() from 09_live_trial_monitor.py
rather than reimplementing them, so fixes made there apply here too.

Controls:
  Space   start recording  /  stop recording (triggers analysis)
  q       quit

Run:
    conda activate dvrk_ml
    cd ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts/depth_camera
    python3 10_live_feedback.py \\
        --reuse-prompts recordings/valid12_20260731_221930_frames/masks_meta.json
"""

import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib.backends.backend_agg import FigureCanvasAgg

SCRIPT_DIR = Path(__file__).parent

# Reuse process_trial()/build_group_figure() (and their helpers) from
# 09_live_trial_monitor.py instead of reimplementing the pipeline/plot.
_spec = importlib.util.spec_from_file_location("live9", SCRIPT_DIR / "09_live_trial_monitor.py")
live9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live9)

STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 30
WARMUP_FRAMES = 30

SETTINGS_PATH  = SCRIPT_DIR / "camera_settings.json"
RECORDINGS_DIR = SCRIPT_DIR / "recordings" / "preview"


# ── sensor / metadata helpers (same as 04_record_session.py / 09) ───────────

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


def save_metadata(json_path, bag_stem, settings, intrinsics, depth_scale, n_frames, duration_s):
    meta = {
        "bag_file":        bag_stem + ".db3",
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s":      round(duration_s, 3),
        "n_frames":        n_frames,
        "camera_settings": settings,
        "intrinsics":      intrinsics,
        "depth_scale":     depth_scale,
    }
    json_path.write_text(json.dumps(meta, indent=2))


def draw_hud(frame, recording, rec_start, n_frames):
    h, w = frame.shape[:2]
    if recording:
        elapsed = time.time() - rec_start
        cv2.rectangle(frame, (0, 0), (w, 26), (40, 0, 0), -1)
        cv2.putText(frame, f"  REC  {elapsed:4.1f}s  |  {n_frames} frames  |  Space=stop",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 180, 80), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (w, 26), (30, 30, 30), -1)
        cv2.putText(frame, "STANDBY  |  Space = record a trial  |  q = quit",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)


# ── background: analyze one trial, render the figure to an image ───────────

def process_and_render(bag_path: Path, reuse_prompts: Path, pegs_path: Path,
                        extract_fps: float, result_queue: queue.Queue):
    print(f"[feedback] Analyzing {bag_path.name}...")
    t0 = time.time()
    csv_path = live9.process_trial(bag_path, reuse_prompts, pegs_path, extract_fps,
                                    out_dir=RECORDINGS_DIR)
    if csv_path is None:
        print("[feedback] pipeline failed — no graph to show.")
        return

    info = {"trial_id": "preview", "bag_path": bag_path, "center_peg": None}
    fig = live9.build_group_figure([info], [csv_path], pegs_path)

    canvas = FigureCanvasAgg(fig)   # headless, thread-safe render — no GUI backend needed here
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    result_queue.put(img_bgr)
    print(f"[feedback] Done in {time.time() - t0:.1f}s.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse-prompts", required=True, dest="reuse_prompts",
                    help="masks_meta.json template replayed headlessly (single cylinder)")
    ap.add_argument("--pegs", default=str(SCRIPT_DIR / "pegs.json"))
    ap.add_argument("--extract-fps", type=float, default=5.0, dest="extract_fps",
                    help="Frame extraction rate (default: 5)")
    args = ap.parse_args()

    reuse_prompts = Path(args.reuse_prompts).resolve()
    pegs_path     = Path(args.pegs).resolve()
    if not reuse_prompts.exists():
        sys.exit(f"[error] --reuse-prompts not found: {reuse_prompts}")
    if not pegs_path.exists():
        sys.exit(f"[error] pegs not found: {pegs_path}")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting RealSense pipeline...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)
    profile = pipeline.start(cfg)
    device  = profile.get_device()

    depth_sensor = device.first_depth_sensor()
    color_sensor = find_color_sensor(device)
    depth_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

    settings    = load_settings(depth_sensor, color_sensor)
    depth_scale = depth_sensor.get_depth_scale()
    intrinsics  = intrinsics_to_dict(
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
    print(f"  depth_scale = {depth_scale} m/unit")

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)

    print(f"  Warming up ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames(timeout_ms=5000)
    print("  Ready.\n")
    print("Press Space to record a trial, Space again to stop + analyze. 'q' to quit.\n")

    cv2.namedWindow("D405 — Live Feedback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("D405 — Live Feedback", STREAM_WIDTH * 2, STREAM_HEIGHT)
    cv2.namedWindow("Depth Feedback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Feedback", 700, 600)

    result_queue: queue.Queue = queue.Queue()

    recording = False
    recorder  = None
    rec_start = 0.0
    n_frames  = 0
    bag_path  = None
    json_path = None
    stem      = None

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
            draw_hud(display[:, :STREAM_WIDTH], recording, rec_start, n_frames)
            cv2.imshow("D405 — Live Feedback", display)

            try:
                img = result_queue.get_nowait()
                cv2.imshow("Depth Feedback", img)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    stem      = "preview_" + time.strftime("%Y%m%d_%H%M%S")
                    bag_path  = RECORDINGS_DIR / f"{stem}.db3"
                    json_path = bag_path.with_suffix(".json")
                    recorder  = rs.recorder(str(bag_path), device)
                    rec_start = time.time()
                    n_frames  = 0
                    recording = True
                    print(f"  ● Recording -> {bag_path.name}")
                else:
                    duration  = time.time() - rec_start
                    recorder  = None   # closes the bag
                    recording = False
                    save_metadata(json_path, stem, settings, intrinsics, depth_scale,
                                  n_frames, duration)
                    print(f"  ■ Stopped  |  {n_frames} frames  |  {duration:.1f}s")
                    threading.Thread(
                        target=process_and_render,
                        args=(bag_path, reuse_prompts, pegs_path, args.extract_fps, result_queue),
                        daemon=True).start()

            elif key == ord("q"):
                if recording:
                    recorder = None
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
