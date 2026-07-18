"""
05_extract_frames.py — Extract color (+ optional depth) frames from a D405 .db3 recording.

Plays back the RealSense recording offline (no camera required) and writes
sequentially-numbered JPEG frames ready for SAM2.

Usage:
    python3 scripts/depth_camera/05_extract_frames.py \\
        --bag recordings/trial_001_20260630_105549.db3

    python3 scripts/depth_camera/05_extract_frames.py \\
        --bag recordings/trial_001_20260630_105549.db3 \\
        --fps 5 \\
        --depth

Output (written next to the .db3 file):
    recordings/<stem>_frames/
        images/
            frame_0000.jpg
            ...
        depth/          (only with --depth)
            frame_0000.png   (16-bit Z16, millimetres)
            ...
        metadata.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


SCRIPT_DIR = Path(__file__).parent


def load_session_meta(bag_path: Path) -> dict:
    """Load the .json sidecar written by 04_record_session.py."""
    json_path = bag_path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {}


def extract(bag_path: Path, out_dir: Path, target_fps: float,
            save_depth: bool, jpeg_quality: int) -> None:

    session_meta = load_session_meta(bag_path)
    source_fps   = 30.0   # D405 always records at 30 fps
    step         = max(1, round(source_fps / target_fps))
    actual_fps   = source_fps / step

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    dep_dir = None
    if save_depth:
        dep_dir = out_dir / "depth"
        dep_dir.mkdir(parents=True, exist_ok=True)

    # ── open bag ──────────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_device_from_file(str(bag_path), repeat_playback=False)

    print(f"Opening {bag_path.name} …")
    profile  = pipeline.start(cfg)

    # disable real-time throttling so we drain the bag as fast as possible
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    n_raw   = 0   # total frames seen in bag
    n_saved = 0   # frames written to disk

    print(f"  Source FPS     : {source_fps:.0f}")
    print(f"  Target FPS     : {target_fps:.2f}  (keep every {step} frame{'s' if step > 1 else ''})")
    print(f"  Output         : {out_dir}")
    print(f"  Save depth     : {save_depth}")
    print()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                # bag exhausted
                break

            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame:
                continue

            n_raw += 1
            if (n_raw - 1) % step != 0:
                continue

            # ── color ─────────────────────────────────────────────────────────
            color_img = np.asanyarray(color_frame.get_data())   # BGR uint8
            fname     = f"{n_saved:05d}.jpg"
            cv2.imwrite(str(img_dir / fname), color_img,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            # ── depth (optional) ──────────────────────────────────────────────
            if save_depth and depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data())  # uint16, raw counts (D405: 0.0001 m/count)
                cv2.imwrite(str(dep_dir / f"{n_saved:05d}.png"), depth_img)

            n_saved += 1
            if n_saved % 50 == 0:
                print(f"  … {n_saved} frames extracted (raw frame {n_raw})")

    finally:
        pipeline.stop()

    # ── metadata.yaml ─────────────────────────────────────────────────────────
    intr = session_meta.get("intrinsics", {})
    meta = {
        "bag_file":       bag_path.name,
        "source_fps":     source_fps,
        "target_fps":     actual_fps,
        "frame_step":     step,
        "n_frames":       n_saved,
        "n_raw_frames":   n_raw,
        "jpeg_quality":   jpeg_quality,
        "has_depth":      save_depth,
        "intrinsics":     intr,
        "depth_scale":    session_meta.get("depth_scale"),  # metres/unit; None for pre-fix recordings
        "session":        session_meta,
    }
    yaml_path = out_dir / "metadata.yaml"
    yaml_path.write_text(yaml.dump(meta, default_flow_style=False, sort_keys=False))

    print()
    print(f"Done.  {n_saved} frames → {out_dir}")
    print(f"       metadata → {yaml_path.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract color frames from a D405 .db3 recording for SAM2.")
    ap.add_argument("--bag",   required=True,
                    help="Path to .db3 recording (absolute or relative to this script).")
    ap.add_argument("--fps",   type=float, default=5.0,
                    help="Target extraction FPS (default 5). Source is 30 fps.")
    ap.add_argument("--depth", action="store_true",
                    help="Also save 16-bit depth PNGs alongside color frames.")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality for color frames (default 95).")
    ap.add_argument("--out",
                    help="Output directory (default: <bag_stem>_frames/ next to .db3).")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.is_absolute():
        bag_path = SCRIPT_DIR / bag_path

    if not bag_path.exists():
        print(f"[error] Bag not found: {bag_path}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = bag_path.parent / (bag_path.stem + "_frames")

    extract(bag_path, out_dir, args.fps, args.depth, args.quality)


if __name__ == "__main__":
    main()
