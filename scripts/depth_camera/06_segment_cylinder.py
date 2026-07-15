"""
06_segment_cylinder.py — Interactively segment the peg-transfer cylinder with SAM 2.1.

Workflow:
  1. Show the start frame → click the cylinder (positive) and background
     (negative); mask preview updates live.  Press 'p' to confirm.
  2. Randomly sample --n-supervision additional frames (default 9).
     Each frame shows the same live mask preview.
     Enter/p = confirm,  s = skip (excluded from final propagation),
     r = reset this frame's clicks,  q = quit.
  3. After all supervision frames, the state is rebuilt from only the
     confirmed annotations and propagated through every frame.
  4. Binary masks saved to --output as 00000.png … (0=bg, 1=cylinder).
  5. masks_meta.json written next to the masks folder.

Run:
    conda activate dvrk_ml
    python3 scripts/depth_camera/06_segment_cylinder.py \\
        --frames recordings/trial_001_20260630_105549_frames/images/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

SCRIPT_DIR = Path(__file__).parent
WIN        = "cylinder segmentation"


def draw_overlay(frame: np.ndarray, mask, clicks: list,
                 hud: str | None = None) -> np.ndarray:
    display = frame.copy()
    if mask is not None:
        green = display.copy()
        green[mask == 1] = (0, 200, 0)
        display = cv2.addWeighted(display, 0.5, green, 0.5, 0)
    for u, v, lbl in clicks:
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(display, (u, v), 6, color, -1, cv2.LINE_AA)
        cv2.circle(display, (u, v), 6, (255, 255, 255), 1, cv2.LINE_AA)
    if hud:
        cv2.putText(display, hud,
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(display,
                    "Enter/p=confirm   s=skip   r=reset   a=positive   n=negative   q=quit",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return display


def collect_frame_files(frames_dir: Path) -> list:
    candidates = [
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    ]
    validated = []
    for p in candidates:
        try:
            int(p.stem)
        except ValueError:
            sys.exit(
                f"[error] SAM2 requires purely numeric filenames (e.g. 00000.jpg).\n"
                f"        Found '{p.name}' in {frames_dir}.\n"
                f"        Re-run extraction with numeric names, or rename the files."
            )
        validated.append(p)
    if not validated:
        sys.exit(f"[error] no JPEG files found in {frames_dir}")
    return sorted(validated, key=lambda p: int(p.stem))


def annotate_frame(predictor, state, frame_bgr: np.ndarray,
                   frame_idx: int, hud: str | None,
                   initial: bool = False) -> list | None:
    """
    Interactive annotation loop for one frame.
    Returns list of (u, v, label) tuples if confirmed, or None if skipped.
    If initial=True the user must confirm with 'p' (no skip option).
    """
    H, W  = frame_bgr.shape[:2]
    clicks: list = []
    mask         = None
    mode: int    = 1
    cb           = {"click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["click"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse, cb)

    while True:
        cv2.imshow(WIN, draw_overlay(frame_bgr, mask, clicks, hud))
        key = cv2.waitKey(20) & 0xFF

        # ── mouse click ───────────────────────────────────────────────────────
        if cb["click"] is not None:
            u, v = cb["click"]
            cb["click"] = None
            if 0 <= u < W and 0 <= v < H:
                clicks.append((u, v, mode))
                pts = np.array([[c[0], c[1]] for c in clicks], dtype=np.float32)
                lbs = np.array([c[2]         for c in clicks], dtype=np.int32)
                _, _, masks = predictor.add_new_points_or_box(
                    state, frame_idx=frame_idx, obj_id=1,
                    points=pts, labels=lbs, clear_old_points=True)
                mask = (masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
                label_str = "pos" if mode == 1 else "neg"
                print(f"    +{label_str} ({u},{v})  → {int(mask.sum())} px masked")

        # ── keyboard ──────────────────────────────────────────────────────────
        if key == ord("a"):
            mode = 1;  print("  Mode: positive")
        elif key == ord("n"):
            mode = 0;  print("  Mode: negative")

        elif key == ord("r"):
            clicks.clear();  mask = None
            print("  Reset — clicks cleared for this frame.")

        elif key in (13, ord("p")):     # Enter or p → confirm
            if not clicks:
                print("  [warn] no clicks yet — click the cylinder first")
            else:
                return clicks

        elif key == ord("s") and not initial:
            print("  Skipped.")
            return None

        elif key == ord("q"):
            cv2.destroyAllWindows();  sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="Interactive SAM2 cylinder segmentation for dVRK recordings.")
    ap.add_argument("--frames",        default="./frames/",
                    help="Directory of numerically-named JPEG frames (default: ./frames/)")
    ap.add_argument("--output",        default="./masks/",
                    help="Output directory for binary PNG masks (default: ./masks/)")
    ap.add_argument("--checkpoint",
                    default=str(SCRIPT_DIR / "checkpoints" / "sam2.1_hiera_large.pt"),
                    help="Path to SAM2.1 checkpoint (.pt)")
    ap.add_argument("--config",        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                    help="SAM2 Hydra config (relative to sam2 package)")
    ap.add_argument("--start-frame",   type=int, default=0, dest="start_frame",
                    help="First frame to annotate (default: 0)")
    ap.add_argument("--n-supervision", type=int, default=9,  dest="n_supervision",
                    help="Number of additional supervision frames (default: 9)")
    ap.add_argument("--seed",          type=int, default=None,
                    help="Random seed for frame selection (optional)")
    args = ap.parse_args()

    frames_dir = Path(args.frames).resolve()
    out_dir    = Path(args.output).resolve()
    ckpt_path  = Path(args.checkpoint).resolve()

    if not frames_dir.is_dir():
        sys.exit(f"[error] frames directory not found: {frames_dir}")
    if not ckpt_path.exists():
        sys.exit(f"[error] checkpoint not found: {ckpt_path}")

    frame_files = collect_frame_files(frames_dir)
    num_frames  = len(frame_files)
    start_frame = args.start_frame

    if start_frame >= num_frames:
        sys.exit(f"[error] --start-frame {start_frame} >= number of frames ({num_frames})")

    # ── build predictor ────────────────────────────────────────────────────────
    print(f"Loading SAM2  : {ckpt_path.name}")
    predictor = build_sam2_video_predictor(args.config, str(ckpt_path), device="cuda")

    print(f"Init state    : {num_frames} frames from {frames_dir.name}/")
    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )

    # ── pick supervision frames ────────────────────────────────────────────────
    candidates = [i for i in range(num_frames) if i != start_frame]
    n_sup      = min(args.n_supervision, len(candidates))
    rng        = np.random.default_rng(args.seed)
    sup_frames = sorted(rng.choice(candidates, n_sup, replace=False).tolist()) if n_sup > 0 else []

    all_frames = [start_frame] + sup_frames   # annotation order
    print(f"Frames to annotate ({len(all_frames)} total): {all_frames}")
    print()

    # ── annotation loop ────────────────────────────────────────────────────────
    # confirmed_annotations: {frame_idx: [(u, v, label), ...]}
    confirmed: dict[int, list] = {}

    for i, frame_idx in enumerate(all_frames):
        bgr = cv2.imread(str(frame_files[frame_idx]))
        if bgr is None:
            print(f"[warn] could not read frame {frame_idx} — skipping")
            continue

        is_initial = (frame_idx == start_frame)
        if is_initial:
            print(f"=== Initial frame {frame_idx:05d}  (p=confirm) ===")
            hud = None  # no extra HUD line on initial frame
        else:
            sup_i = sup_frames.index(frame_idx)
            print(f"=== Supervision {sup_i + 1}/{n_sup}  frame {frame_idx:05d} ===")
            hud = (f"Supervision {sup_i + 1}/{n_sup}   "
                   f"frame {frame_idx:05d}   "
                   f"a=pos  n=neg")

        result = annotate_frame(predictor, state, bgr, frame_idx,
                                hud=hud, initial=is_initial)

        if result is not None:
            confirmed[frame_idx] = result
            print(f"  Confirmed: {len(result)} click(s)")
        # else: skipped — will be excluded from final state

    cv2.destroyAllWindows()

    if not confirmed:
        sys.exit("[error] no frames confirmed — nothing to propagate.")

    # ── rebuild state with only confirmed annotations ─────────────────────────
    # This ensures skipped frames don't leak into propagation.
    print(f"\nRebuilding state with {len(confirmed)} confirmed frame(s): "
          f"{sorted(confirmed)}")
    predictor.reset_state(state)
    for frame_idx, clicks in sorted(confirmed.items()):
        pts = np.array([[c[0], c[1]] for c in clicks], dtype=np.float32)
        lbs = np.array([c[2]         for c in clicks], dtype=np.int32)
        predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=1,
            points=pts, labels=lbs, clear_old_points=True)

    # ── propagation ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Propagating through {num_frames} frames → {out_dir}")

    try:
        n_saved = 0
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
            mask = (video_res_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), mask)
            n_saved += 1
            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx:5d} / {num_frames - 1}")

    except torch.cuda.OutOfMemoryError:
        print()
        print("[CUDA OOM] Ran out of GPU memory during propagation.")
        print("  Try --config configs/sam2.1/sam2.1_hiera_b+.yaml with the base+ checkpoint.")
        sys.exit(1)

    # ── metadata ───────────────────────────────────────────────────────────────
    all_prompts = [
        {"frame": fidx, "u": u, "v": v, "label": lbl}
        for fidx, clicks in sorted(confirmed.items())
        for u, v, lbl in clicks
    ]
    meta = {
        "start_frame":          start_frame,
        "n_supervision":        n_sup,
        "annotated_frames":     sorted(confirmed.keys()),
        "prompts":              all_prompts,
        "num_frames_processed": n_saved,
        "checkpoint":           str(ckpt_path),
    }
    meta_path = out_dir.parent / "masks_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone.  {n_saved} masks → {out_dir}")
    print(f"Meta   → {meta_path}")


if __name__ == "__main__":
    main()
