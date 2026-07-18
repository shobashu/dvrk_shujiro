"""
06b_visualize_masks.py — Overlay SAM2 masks on color frames and play as video.

Green tint (50% alpha) where mask == 1.
Corner HUD shows frame index and whether a mask exists for that frame.

Controls
  Space       pause / resume
  Left / Right arrow  step one frame (when paused)
  q           quit

Run:
    conda activate dvrk_ml
    python3 scripts/depth_camera/06b_visualize_masks.py \\
        --frames recordings/trial_001_20260630_105549_frames/images/ \\
        --masks  recordings/trial_001_20260630_105549_frames/masks/

    python3 06b_visualize_masks.py \
    --frames recordings/trial_001_20260630_105549_frames/images/ \
    --masks masks/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WIN       = "mask overlay"
KEY_SPACE = ord(" ")
KEY_Q     = ord("q")
KEY_LEFT  = 65361   # Linux/X11 left arrow  (cv2.waitKeyEx)
KEY_RIGHT = 65363   # Linux/X11 right arrow


def collect_sorted(directory: Path, suffixes: tuple) -> list[Path]:
    files = [p for p in directory.iterdir() if p.suffix.lower() in suffixes]
    try:
        return sorted(files, key=lambda p: int(p.stem))
    except ValueError as e:
        sys.exit(
            f"[error] non-numeric filename in {directory}: {e}\n"
            f"        Files must be named 00000.jpg / 00000.png etc."
        )


def draw_hud(img: np.ndarray, frame_idx: int, total: int, has_mask: bool) -> None:
    label     = "MASK" if has_mask else "NO MASK"
    label_col = (0, 220, 0) if has_mask else (0, 80, 220)
    text      = f"frame {frame_idx:05d} / {total - 1:05d}   {label}"

    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (8, 8), (8 + tw + 6, 8 + th + bl + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (11, 8 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_col, 1, cv2.LINE_AA)


def apply_overlay(frame: np.ndarray, mask_path: Path):
    """Return (overlaid_frame, has_mask). mask_path may not exist."""
    if mask_path is None or not mask_path.exists():
        return frame, False

    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return frame, False

    binary = raw > 0          # values are 0 or 1 (step 6 output)
    if not binary.any():
        return frame, False

    green = frame.copy()
    green[binary] = (0, 200, 0)
    blended = cv2.addWeighted(frame, 0.5, green, 0.5, 0)
    return blended, True


def main():
    ap = argparse.ArgumentParser(
        description="Play color frames with SAM2 mask overlay.")
    ap.add_argument("--frames", default="./frames/",
                    help="Directory of color frames (00000.jpg …)")
    ap.add_argument("--masks",  default="./masks/",
                    help="Directory of binary mask PNGs (00000.png …)")
    ap.add_argument("--fps",    type=float, default=10.0,
                    help="Playback speed in frames per second (default 10)")
    args = ap.parse_args()

    frames_dir = Path(args.frames).resolve()
    masks_dir  = Path(args.masks).resolve()

    if not frames_dir.is_dir():
        sys.exit(f"[error] frames directory not found: {frames_dir}")

    frame_files = collect_sorted(frames_dir, (".jpg", ".jpeg"))
    if not frame_files:
        sys.exit(f"[error] no JPEG files found in {frames_dir}")

    # Build a stem → mask path lookup (masks dir may not exist / may be partial)
    mask_lookup: dict[str, Path] = {}
    if masks_dir.is_dir():
        for p in masks_dir.iterdir():
            if p.suffix.lower() == ".png":
                mask_lookup[p.stem] = p
    else:
        print(f"[warn] masks directory not found: {masks_dir} — playing frames only")

    total      = len(frame_files)
    delay_ms   = max(1, int(1000 / args.fps))
    paused     = False
    idx        = 0

    print(f"Frames : {total}  from {frames_dir.name}/")
    print(f"Masks  : {len(mask_lookup)} found  from {masks_dir.name}/")
    print(f"FPS    : {args.fps}")
    print()
    print("Space=pause  ←/→=step (when paused)  q=quit")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        frame_path = frame_files[idx]
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            print(f"[warn] could not read {frame_path.name} — skipping")
        else:
            mask_path = mask_lookup.get(frame_path.stem)
            display, has_mask = apply_overlay(bgr, mask_path)
            draw_hud(display, idx, total, has_mask)
            cv2.imshow(WIN, display)

        key = cv2.waitKeyEx(delay_ms if not paused else 0)

        if key == KEY_Q:
            break

        elif key == KEY_SPACE:
            paused = not paused
            print("paused" if paused else "playing")

        elif key == KEY_LEFT:
            idx = max(0, idx - 1)

        elif key == KEY_RIGHT:
            idx = min(total - 1, idx + 1)

        elif not paused:
            if idx < total - 1:
                idx += 1
            else:
                idx = 0          # loop

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
