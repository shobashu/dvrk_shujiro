"""
07c_show_cylinder_cloud.py — Show the full 3D point cloud of the detected cylinder.

Deprojects every valid mask pixel to 3D for one frame and renders the
complete visible surface of the cylinder as a scatter plot, alongside the
colour image.  The centroid (used in the trajectory) is marked separately
so you can see exactly which part of the cylinder is being tracked.

Left panel  : colour image with mask overlay and centroid cross-hair
Right panel : 3D scatter of all deprojected mask pixels, coloured by
              surface depth (nearest = yellow, farthest = blue).
              Centroid = red sphere.  Pegs = stars.

Controls (right-click the 3-D panel to rotate):
  ← / → arrow keys   step to previous / next frame that has a valid mask
  q                   quit

Run:
    conda activate dvrk_ml
    python3 07c_show_cylinder_cloud.py \\
        --frame 100 \\
        --masks   masks/ \\
        --depth   recordings/trial_001_20260630_105549_frames/depth/ \\
        --images  recordings/trial_001_20260630_105549_frames/images/ \\
        --meta    recordings/trial_001_20260630_105549_frames/metadata.yaml \\
        --pegs    pegs.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import yaml

SCRIPT_DIR  = Path(__file__).parent
MIN_VALID   = 10
MAX_DEPTH_MM = 5000

# D405 short-range sensors commonly report 0.0001 m/unit (10x finer than the
# 0.001 m/unit "standard" scale most other RealSense depth cameras use).
# Only used as a fallback for recordings made before depth_scale was captured
# in the metadata (04_record_session.py) — new recordings carry the real,
# sensor-reported value instead of this guess.
FALLBACK_DEPTH_SCALE = 0.0001


# ── depth / intrinsics helpers ────────────────────────────────────────────────

def load_intrinsics(meta_path: Path) -> dict:
    meta = yaml.safe_load(meta_path.read_text())
    intr = meta["intrinsics"]
    depth_scale = meta.get("depth_scale")
    if depth_scale is None:
        depth_scale = FALLBACK_DEPTH_SCALE
        print(f"[warn] no depth_scale in {meta_path.name} (recorded before this was tracked) — "
              f"assuming D405 default {FALLBACK_DEPTH_SCALE} m/unit. "
              f"Verify against pegs.json if this recording used a different sensor.")
    return {
        "fx":  float(intr["fx"]),
        "fy":  float(intr["fy"]),
        "ppx": float(intr["ppx"]),
        "ppy": float(intr["ppy"]),
        "depth_scale": float(depth_scale),
    }


def deproject_full(mask: np.ndarray, depth_raw: np.ndarray, fx, fy, ppx, ppy, depth_scale: float):
    """
    Return (pts, centroid) where pts is (N,3) of all valid mask pixels [m]
    and centroid is the median point (3,).  Returns (None, None) if too few.
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None, None

    depths_m = depth_raw[ys, xs].astype(np.float64) * depth_scale
    valid    = (depths_m > 0) & (depths_m < MAX_DEPTH_MM / 1000.0)
    if valid.sum() < MIN_VALID:
        return None, None

    d_m = depths_m[valid]
    X   = (xs[valid] - ppx) * d_m / fx
    Y   = (ys[valid] - ppy) * d_m / fy
    Z   = d_m

    pts      = np.stack([X, Y, Z], axis=1)
    centroid = np.median(pts, axis=0)
    return pts, centroid


# ── board frame (same logic as 07b) ──────────────────────────────────────────

def load_pegs(pegs_path: Path):
    data  = json.loads(pegs_path.read_text())
    pegs  = [np.array(p["xyz_m"]) for p in data["pegs"]]
    up    = np.array(data["up_axis"])
    up   /= np.linalg.norm(up)
    return pegs, up


def build_board_frame(pegs, up):
    board_Z = up
    raw_X   = pegs[4] - pegs[0]
    board_X = raw_X - np.dot(raw_X, board_Z) * board_Z
    board_X /= np.linalg.norm(board_X)
    board_Y  = np.cross(board_Z, board_X)
    board_Y /= np.linalg.norm(board_Y)
    return pegs[0], np.column_stack([board_X, board_Y, board_Z])


def to_board_mm(pts_cam, origin, R):
    return (R.T @ (pts_cam - origin).T).T * 1000.0


# ── collect valid frames ──────────────────────────────────────────────────────

def collect_valid_frames(masks_dir: Path, depth_dir: Path, images_dir: Path):
    """Return sorted list of frame indices that have mask + depth + image."""
    valid = []
    for p in sorted(masks_dir.iterdir(), key=lambda p: int(p.stem)):
        if p.suffix.lower() != ".png":
            continue
        fidx = int(p.stem)
        if (depth_dir / f"{fidx:05d}.png").exists() and \
           (images_dir / f"{fidx:05d}.jpg").exists():
            valid.append(fidx)
    return valid


# ── rendering ─────────────────────────────────────────────────────────────────

def render(fig, frame_idx, masks_dir, depth_dir, images_dir, intr, pegs_cam, origin, R):
    fig.clf()

    mask_path  = masks_dir  / f"{frame_idx:05d}.png"
    depth_path = depth_dir  / f"{frame_idx:05d}.png"
    image_path = images_dir / f"{frame_idx:05d}.jpg"

    mask      = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    bgr       = cv2.imread(str(image_path))

    if mask is None or depth_raw is None or bgr is None:
        fig.suptitle(f"frame {frame_idx:05d} — could not load files", color="red")
        fig.canvas.draw_idle()
        return

    fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    pts_cam, centroid_cam = deproject_full(mask, depth_raw, fx, fy, ppx, ppy, intr["depth_scale"])

    # ── left panel: colour image + mask overlay + centroid ────────────────────
    ax2d = fig.add_subplot(1, 2, 1)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    if mask is not None:
        overlay[mask == 1] = [0, 200, 0]
        rgb = (0.5 * rgb + 0.5 * overlay).astype(np.uint8)

    ax2d.imshow(rgb)
    if centroid_cam is not None:
        # project centroid back to pixel for cross-hair
        cx = centroid_cam[0] * fx / centroid_cam[2] + ppx
        cy = centroid_cam[1] * fy / centroid_cam[2] + ppy
        ax2d.plot(cx, cy, "r+", markersize=18, markeredgewidth=2.5)
        ax2d.plot(cx, cy, "ro", markersize=6, fillstyle="none", markeredgewidth=1.5)

    ax2d.set_title(f"Frame {frame_idx:05d} — colour + mask", fontsize=9)
    ax2d.axis("off")

    # ── right panel: 3-D point cloud ──────────────────────────────────────────
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    if pts_cam is None:
        ax3d.set_title("No valid depth in mask", fontsize=9)
        fig.canvas.draw_idle()
        return

    # transform to board frame [mm]
    pts_b      = to_board_mm(pts_cam, origin, R)
    centroid_b = to_board_mm(centroid_cam[np.newaxis], origin, R)[0]
    pegs_b     = to_board_mm(np.array(pegs_cam), origin, R)

    # colour by Z (distance from camera) — nearest = yellow, farthest = blue
    z_vals = pts_b[:, 2]
    norm   = mcolors.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    colors = cm.plasma(norm(z_vals))

    ax3d.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                 c=colors, s=4, alpha=0.6, linewidths=0)

    # centroid: large red sphere
    ax3d.scatter([centroid_b[0]], [centroid_b[1]], [centroid_b[2]],
                 color="red", s=120, zorder=10,
                 edgecolors="white", linewidths=1.0, label="centroid")

    # pegs: labelled stars
    peg_colors = plt.cm.tab10(np.linspace(0, 1, len(pegs_b)))
    for i, (pos, col) in enumerate(zip(pegs_b, peg_colors)):
        ax3d.scatter([pos[0]], [pos[1]], [pos[2]], marker="*", s=150,
                     color=col, edgecolors="k", linewidths=0.5, zorder=5)
        ax3d.text(pos[0], pos[1], pos[2], f"  {i}", fontsize=6, color=col)

    # depth colour bar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax3d, shrink=0.4, pad=0.1)
    cb.set_label("Z board (mm)", fontsize=7)

    # stats annotation
    spread_x = pts_b[:, 0].max() - pts_b[:, 0].min()
    spread_y = pts_b[:, 1].max() - pts_b[:, 1].min()
    spread_z = pts_b[:, 2].max() - pts_b[:, 2].min()
    ax3d.set_title(
        f"3-D cloud  ({len(pts_b)} px)\n"
        f"ΔX={spread_x:.1f} mm  ΔY={spread_y:.1f} mm  ΔZ={spread_z:.1f} mm",
        fontsize=8)
    ax3d.set_xlabel("X board (mm)", fontsize=7)
    ax3d.set_ylabel("Y board (mm)", fontsize=7)
    ax3d.set_zlabel("Z lift (mm)",  fontsize=7)
    ax3d.set_xlim(-10, 80)
    ax3d.set_ylim(-10, 80)
    ax3d.set_zlim(-30, 30)
    ax3d.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Cylinder surface cloud — frame {frame_idx:05d}   "
        f"centroid: ({centroid_b[0]:.1f}, {centroid_b[1]:.1f}, {centroid_b[2]:.1f}) mm  "
        f"[← / → to step frames]",
        fontsize=9)

    fig.canvas.draw_idle()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualise the full 3D cylinder point cloud for one or more frames.")
    ap.add_argument("--frame",   type=int, default=0,
                    help="Starting frame index (default: 0)")
    ap.add_argument("--masks",   default="./masks/")
    ap.add_argument("--depth",   required=True,
                    help="Depth PNG directory from step 5")
    ap.add_argument("--images",  required=True,
                    help="Colour JPEG directory from step 5")
    ap.add_argument("--meta",    required=True,
                    help="metadata.yaml from step 5")
    ap.add_argument("--pegs",    default="./pegs.json")
    args = ap.parse_args()

    masks_dir  = Path(args.masks).resolve()
    depth_dir  = Path(args.depth).resolve()
    images_dir = Path(args.images).resolve()
    meta_path  = Path(args.meta).resolve()
    pegs_path  = Path(args.pegs).resolve()

    for p, name in [(masks_dir, "masks"), (depth_dir, "depth"),
                    (images_dir, "images"), (meta_path, "meta"), (pegs_path, "pegs")]:
        if not p.exists():
            sys.exit(f"[error] {name} not found: {p}")

    intr             = load_intrinsics(meta_path)
    pegs_cam, up     = load_pegs(pegs_path)
    origin, R        = build_board_frame(pegs_cam, up)
    valid_frames     = collect_valid_frames(masks_dir, depth_dir, images_dir)

    if not valid_frames:
        sys.exit("[error] no frames found with mask + depth + image")

    # clamp starting frame to nearest valid
    start = min(valid_frames, key=lambda f: abs(f - args.frame))
    idx   = valid_frames.index(start)

    print(f"Valid frames : {len(valid_frames)}  (first={valid_frames[0]}, last={valid_frames[-1]})")
    print(f"Starting at  : frame {start}")
    print("← / → to step through frames,  q to quit")

    fig = plt.figure(figsize=(16, 7))

    def on_key(event):
        nonlocal idx
        if event.key == "right" and idx < len(valid_frames) - 1:
            idx += 1
        elif event.key == "left" and idx > 0:
            idx -= 1
        elif event.key == "q":
            plt.close("all")
            return
        else:
            return
        render(fig, valid_frames[idx], masks_dir, depth_dir, images_dir,
               intr, pegs_cam, origin, R)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render(fig, valid_frames[idx], masks_dir, depth_dir, images_dir,
           intr, pegs_cam, origin, R)

    plt.show()


if __name__ == "__main__":
    main()
