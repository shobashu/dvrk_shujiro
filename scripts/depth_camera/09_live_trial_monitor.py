"""
09_live_trial_monitor.py — Depth-camera trial recorder + every-3-trials 3D map.

Runs alongside dvrk_shujiro/arduino/read_arduino_with_force.py as a SEPARATE
process (that script needs the ros2_yolo_venv environment for rclpy; this one
needs dvrk_ml for pyrealsense2 + sam2 — the two can't merge into one process
because dvrk_ml's rclpy is broken at the C-extension level, and ros2_yolo_venv
has neither pyrealsense2 nor sam2). They coordinate through a small
cross-process signal file instead of a shared import.

Protocol — one JSON line per trial boundary, written by
read_arduino_with_force.py, tailed here:
    {"event": "trial_start", "trial": "3", "t": 1234567.89}
    {"event": "trial_end",   "trial": "3", "t": 1234571.23}
That file is truncated fresh every time read_arduino_with_force.py starts, so
if this script is (re)started mid-session it only reacts to trials from that
point forward — it does not replay/backfill older ones. If EITHER script
needs restarting mid-session, restart both together to stay in sync.

Per trial: opens a fresh RealSense recording on trial_start (same settings/
depth-scale capture as 04_record_session.py), closes it on trial_end.
Every --group-size (default 3) completed trials, spawns a background thread
that runs the existing CLI pipeline over those trials —
    05_extract_frames.py  ->  06_segment_cylinder.py --auto --reuse-prompts
        ->  07_cylinder_trajectory.py
— then pops up a 3D matplotlib view (one line per trial, over the peg board)
in a Tk window.

CAUTION — --auto replays saved click coordinates with ZERO human
verification (see 06_segment_cylinder.py's --auto docs). If the cylinder/
gripper doesn't start in about the same screen position every trial, or the
camera moves, a trial's mask — and therefore its line on the map — can be
silently wrong with nobody checking. This trades reviewability for being
fully hands-off, by explicit choice.

Timing: this is NOT sub-second real-time. SAM2 propagation alone measured
~5.2 frames/sec on this machine — a 3-trial batch of ~90s trials extracted
at 15fps is roughly (90s x 15fps / 5.2fps/s) ~= 4-5 minutes of background
work per trial, so the map appears a few minutes after the 3rd trial ends,
not instantly. It runs in the background so the next trial isn't blocked.

Run (separately from read_arduino_with_force.py, in its own terminal):
    conda activate dvrk_ml
    cd ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts/depth_camera
    python3 09_live_trial_monitor.py \\
        --reuse-prompts recordings/valid12_20260731_221930_frames/masks_meta.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SCRIPT_DIR = Path(__file__).parent
PYTHON     = sys.executable

TRIAL_SIGNAL_PATH = '/tmp/dvrk_depth_trial_signal.jsonl'   # must match read_arduino_with_force.py

STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 30
WARMUP_FRAMES = 30

SETTINGS_PATH  = SCRIPT_DIR / "camera_settings.json"
RECORDINGS_DIR = SCRIPT_DIR / "recordings" / "live"

TRIAL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ── sensor / recording helpers (mirrors 04_record_session.py) ───────────────

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


# ── board frame (same convention as 07b/07c/08) ──────────────────────────────

def build_board_frame(pegs_xyz: np.ndarray, up: np.ndarray):
    board_Z = up
    raw_X   = pegs_xyz[4] - pegs_xyz[0]
    board_X = raw_X - np.dot(raw_X, board_Z) * board_Z
    board_X /= np.linalg.norm(board_X)
    board_Y  = np.cross(board_Z, board_X)
    board_Y /= np.linalg.norm(board_Y)
    origin   = pegs_xyz[0]
    return origin, np.column_stack([board_X, board_Y, board_Z])


def to_board_mm(pts_cam: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R.T @ (pts_cam - origin).T).T * 1000.0


# ── live RealSense recorder, toggled by the signal-watch thread ─────────────

class LiveDepthRecorder:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)

        print("[depth] Starting RealSense pipeline...")
        profile = self.pipeline.start(cfg)
        device  = profile.get_device()

        depth_sensor = device.first_depth_sensor()
        color_sensor = find_color_sensor(device)
        depth_sensor.set_option(rs.option.enable_auto_exposure,      0)
        color_sensor.set_option(rs.option.enable_auto_exposure,      0)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

        self.device      = device
        self.settings    = load_settings(depth_sensor, color_sensor)
        self.depth_scale = depth_sensor.get_depth_scale()
        self.intrinsics  = intrinsics_to_dict(
            profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
        print(f"[depth] depth_scale = {self.depth_scale} m/unit")

        print(f"[depth] Warming up ({WARMUP_FRAMES} frames)...")
        for _ in range(WARMUP_FRAMES):
            self.pipeline.wait_for_frames(timeout_ms=5000)
        print("[depth] Ready.")

        self._lock      = threading.Lock()
        self._recorder   = None
        self._rec_start  = 0.0
        self._n_frames   = 0
        self._bag_path   = None
        self._json_path  = None
        self._stop       = threading.Event()
        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

    def _pump_loop(self):
        """Keeps frames flowing through the pipeline at all times — the
        rs.recorder object (once constructed) captures automatically
        whatever passes through while it exists, same as 04_record_session.py."""
        while not self._stop.is_set():
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                continue
            with self._lock:
                if self._recorder is not None:
                    self._n_frames += 1

    def start_trial(self, trial_id: str):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        bag_path = RECORDINGS_DIR / f"live_trial_{trial_id}_{ts}.db3"
        with self._lock:
            self._recorder  = rs.recorder(str(bag_path), self.device)
            self._rec_start = time.time()
            self._n_frames  = 0
            self._bag_path  = bag_path
            self._json_path = bag_path.with_suffix(".json")
        print(f"[depth] trial {trial_id} recording -> {bag_path.name}")

    def end_trial(self, trial_id: str, center_peg: int | None = None):
        with self._lock:
            if self._recorder is None:
                return None
            self._recorder = None   # closes the bag
            duration  = time.time() - self._rec_start
            n_frames  = self._n_frames
            bag_path  = self._bag_path
            json_path = self._json_path

        meta = {
            "bag_file":        bag_path.name,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s":      round(duration, 3),
            "n_frames":        n_frames,
            "camera_settings": self.settings,
            "intrinsics":      self.intrinsics,
            "depth_scale":     self.depth_scale,
            "center_peg":      center_peg,
        }
        json_path.write_text(json.dumps(meta, indent=2))
        print(f"[depth] trial {trial_id} stopped  ({n_frames} frames, {duration:.1f}s, "
              f"center_peg={center_peg})")
        return {"trial_id": trial_id, "bag_path": bag_path, "json_path": json_path,
                "center_peg": center_peg}

    def shutdown(self):
        self._stop.set()
        self._pump_thread.join(timeout=3)
        self.pipeline.stop()


# ── background pipeline over one group of trials ─────────────────────────────

def run_step(cmd: list, label: str) -> bool:
    print(f"[depth] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"[depth] [error] {label} failed (exit {result.returncode}) — skipping this trial.")
        return False
    return True


def template_for(center_peg: int | None, reuse_prompts: Path | None,
                  templates_dir: Path | None) -> Path | None:
    """Resolve which --reuse-prompts template applies to a trial.

    Single-template mode (--reuse-prompts) ignores center_peg entirely —
    only correct for a 1-cylinder setup. Multi-template mode (--templates-dir)
    looks up center<N>.json, matching the Arduino's 1-4 center-peg index.
    """
    if reuse_prompts is not None:
        return reuse_prompts
    if templates_dir is not None and center_peg is not None:
        candidate = templates_dir / f"center{center_peg}.json"
        if candidate.exists():
            return candidate
        print(f"[depth] [warn] no template for center_peg={center_peg} "
              f"(expected {candidate.name} in {templates_dir}) — skipping this trial.")
    return None


def process_trial(bag_path: Path, reuse_prompts: Path, pegs_path: Path,
                   extract_fps: float, out_dir: Path | None = None) -> Path | None:
    out_dir = out_dir or RECORDINGS_DIR
    frames_dir = bag_path.parent / f"{bag_path.stem}_frames"

    if not run_step([PYTHON, "05_extract_frames.py",
                      "--bag", str(bag_path), "--fps", str(extract_fps), "--depth"],
                     "frame extraction"):
        return None

    if not run_step([PYTHON, "06_segment_cylinder.py",
                      "--frames", str(frames_dir / "images"),
                      "--output", str(frames_dir / "masks"),
                      "--reuse-prompts", str(reuse_prompts),
                      "--auto"],
                     "segmentation"):
        return None

    traj_csv = out_dir / f"{bag_path.stem}_trajectory.csv"
    if not run_step([PYTHON, "07_cylinder_trajectory.py",
                      "--masks", str(frames_dir / "masks"),
                      "--depth", str(frames_dir / "depth"),
                      "--meta", str(frames_dir / "metadata.yaml"),
                      "--pegs", str(pegs_path),
                      "--output", str(traj_csv)],
                     "trajectory"):
        return None

    return traj_csv


def build_group_figure(trial_infos: list, traj_csvs: list, pegs_path: Path) -> Figure:
    pegs_data = json.loads(pegs_path.read_text())
    pegs_sorted = sorted(pegs_data["pegs"], key=lambda p: p["id"])
    pegs_xyz = np.array([p["xyz_m"] for p in pegs_sorted], dtype=np.float64)
    up = np.array(pegs_data["up_axis"], dtype=np.float64)
    up = up / np.linalg.norm(up)
    origin, R = build_board_frame(pegs_xyz, up)

    fig = Figure(figsize=(7, 6))
    ax  = fig.add_subplot(projection="3d")

    pegs_b = to_board_mm(pegs_xyz, origin, R)
    ax.scatter(pegs_b[:, 0], pegs_b[:, 1], pegs_b[:, 2],
               marker="*", s=140, color="black", zorder=5)
    for i, p in enumerate(pegs_b):
        ax.text(p[0], p[1], p[2], f" {i}", fontsize=7)

    for i, (info, csv_path) in enumerate(zip(trial_infos, traj_csvs)):
        color = TRIAL_COLORS[i % len(TRIAL_COLORS)]
        df = pd.read_csv(csv_path).dropna(subset=["X_m", "Y_m", "Z_m"])
        if df.empty:
            continue
        pts_cam = df[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=np.float64)
        pts_b   = to_board_mm(pts_cam, origin, R)
        cp_tag  = f" (cyl {info['center_peg']})" if info.get("center_peg") else ""
        ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                color=color, linewidth=1.4, alpha=0.85,
                label=f"trial {info['trial_id']}{cp_tag}")

    ax.set_xlim(-10, 80)
    ax.set_ylim(-10, 80)
    ax.set_zlim(-30, 30)
    ax.set_xlabel("X board (mm)")
    ax.set_ylabel("Y board (mm)")
    ax.set_zlabel("Z lift (mm)")
    ax.set_title(f"Trials {', '.join(i['trial_id'] for i in trial_infos)}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def process_group_and_show(trial_infos: list, popup, reuse_prompts: Path | None,
                            templates_dir: Path | None, pegs_path: Path, extract_fps: float):
    print(f"[depth] Processing group: trials {[i['trial_id'] for i in trial_infos]}")
    t0 = time.time()
    traj_csvs, good_infos = [], []
    for info in trial_infos:
        template = template_for(info.get("center_peg"), reuse_prompts, templates_dir)
        if template is None:
            print(f"[depth] [warn] trial {info['trial_id']}: no usable template — skipped.")
            continue
        csv_path = process_trial(info["bag_path"], template, pegs_path, extract_fps)
        if csv_path is not None:
            traj_csvs.append(csv_path)
            good_infos.append(info)

    if not traj_csvs:
        print("[depth] [error] no trials in this group produced a trajectory — skipping map.")
        return

    fig = build_group_figure(good_infos, traj_csvs, pegs_path)
    print(f"[depth] Group processed in {time.time() - t0:.1f}s — showing map.")
    popup.show_threadsafe(fig, f"Trials {', '.join(i['trial_id'] for i in good_infos)}")


# ── Tk popup ──────────────────────────────────────────────────────────────────

class MapPopup:
    def __init__(self, root):
        self._root = root

    def show_threadsafe(self, fig: Figure, title: str):
        self._root.after(0, lambda: self._show(fig, title))

    def _show(self, fig: Figure, title: str):
        win = tk.Toplevel(self._root)
        win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        tk.Button(win, text="Close", command=win.destroy).pack(pady=4)


# ── signal-file tail loop ─────────────────────────────────────────────────────

def signal_watch_loop(recorder: LiveDepthRecorder, popup: MapPopup,
                       reuse_prompts: Path | None, templates_dir: Path | None, pegs_path: Path,
                       extract_fps: float, group_size: int, stop_event: threading.Event):
    print(f"[depth] Waiting for signal file: {TRIAL_SIGNAL_PATH}")
    while not stop_event.is_set() and not os.path.exists(TRIAL_SIGNAL_PATH):
        time.sleep(0.5)
    if stop_event.is_set():
        return

    f = open(TRIAL_SIGNAL_PATH, "r")
    f.seek(0, os.SEEK_END)   # ignore anything written before we started watching
    print("[depth] Watching for trial events.")

    completed = []
    pending_center_peg = {}   # trial_id -> center_peg, set at trial_start, used at trial_end
    while not stop_event.is_set():
        line = f.readline()
        if not line:
            time.sleep(0.15)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        trial_id = str(evt["trial"])
        if evt["event"] == "trial_start":
            pending_center_peg[trial_id] = evt.get("center_peg")
            recorder.start_trial(trial_id)
        elif evt["event"] == "trial_end":
            center_peg = pending_center_peg.pop(trial_id, None)
            info = recorder.end_trial(trial_id, center_peg=center_peg)
            if info is None:
                continue
            completed.append(info)
            if len(completed) % group_size == 0:
                group = completed[-group_size:]
                threading.Thread(
                    target=process_group_and_show,
                    args=(group, popup, reuse_prompts, templates_dir, pegs_path, extract_fps),
                    daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--reuse-prompts", dest="reuse_prompts",
                       help="Single masks_meta.json template, replayed for every trial "
                            "regardless of which cylinder is active. Only correct for a "
                            "1-cylinder setup.")
    group.add_argument("--templates-dir", dest="templates_dir",
                       help="Directory containing center1.json .. center4.json (one "
                            "masks_meta.json per center-peg cylinder). Picks the matching "
                            "template each trial from the Arduino's center_peg signal. "
                            "Required for a real 4-cylinder setup.")
    ap.add_argument("--pegs", default=str(SCRIPT_DIR / "pegs.json"))
    ap.add_argument("--extract-fps", type=float, default=5.0, dest="extract_fps",
                    help="Frame extraction rate for live trials (default: 5)")
    ap.add_argument("--group-size", type=int, default=3, dest="group_size",
                    help="Show the 3-D map every N completed trials (default: 3)")
    args = ap.parse_args()

    reuse_prompts = Path(args.reuse_prompts).resolve() if args.reuse_prompts else None
    templates_dir = Path(args.templates_dir).resolve() if args.templates_dir else None
    pegs_path     = Path(args.pegs).resolve()
    if reuse_prompts is not None and not reuse_prompts.exists():
        sys.exit(f"[error] --reuse-prompts not found: {reuse_prompts}")
    if templates_dir is not None and not templates_dir.is_dir():
        sys.exit(f"[error] --templates-dir not found: {templates_dir}")
    if not pegs_path.exists():
        sys.exit(f"[error] pegs not found: {pegs_path}")

    print("=" * 60)
    print(" Live depth-camera trial monitor")
    if reuse_prompts is not None:
        print(f"  Reuse template : {reuse_prompts}  (single-cylinder mode)")
    else:
        print(f"  Templates dir  : {templates_dir}  (multi-cylinder mode)")
        missing = [n for n in (1, 2, 3, 4) if not (templates_dir / f"center{n}.json").exists()]
        if missing:
            print(f"  [warn] missing templates for center peg(s) {missing} — "
                  f"those trials will be skipped until recorded.")
    print(f"  Pegs           : {pegs_path}")
    print(f"  Extract fps    : {args.extract_fps}")
    print(f"  Map every      : {args.group_size} trials")
    print("  CAUTION: --auto segmentation runs with zero human verification.")
    print("=" * 60)

    recorder = LiveDepthRecorder()

    root = tk.Tk()
    root.withdraw()
    popup = MapPopup(root)

    stop_event = threading.Event()
    watch_thread = threading.Thread(
        target=signal_watch_loop,
        args=(recorder, popup, reuse_prompts, templates_dir, pegs_path, args.extract_fps,
              args.group_size, stop_event),
        daemon=True)
    watch_thread.start()

    def signal_handler(sig, frame):
        print("\n[depth] Shutting down...")
        stop_event.set()
        recorder.shutdown()
        root.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    def keep_alive():
        root.after(100, keep_alive)
    root.after(100, keep_alive)

    print("[depth] Press Ctrl+C to quit")
    root.mainloop()


if __name__ == "__main__":
    main()
