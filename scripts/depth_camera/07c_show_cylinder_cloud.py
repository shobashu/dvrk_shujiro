"""
07c_show_cylinder_cloud.py — Show the full 3D point cloud of the detected cylinder.

Deprojects every valid mask pixel to 3D for one frame and renders the
complete visible surface of the cylinder as a scatter plot, alongside the
colour image.  The centroid is marked separately so you can see exactly
which part of the cylinder is being tracked.

Coordinate frame: same board frame as 07b_visualize_trajectory.py
  Origin : peg 0  (bottom-left)
  X      : peg 0 → peg 4  (left → right)
  Y      : bottom row → top row
  Z      : up_axis (lift above board surface)

Peg layout:
  Row 1 (top):    Peg 3 ——— Peg 7
                  Peg 2     Peg 6
                  Peg 1     Peg 5
  Row 4 (bottom): Peg 0 ——— Peg 4   ← origin

Left panel  : colour image with mask overlay and centroid cross-hair.
Right panel : 3D point cloud of all deprojected mask pixels (colour =
              lift height Z), centroid as red sphere, pegs as stars with
              board grid.

Controls:
  ← / →   step to previous / next frame
  q        quit

Run:
    conda activate dvrk_ml
    python3 scripts/depth_camera/07c_show_cylinder_cloud.py \\
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

SCRIPT_DIR   = Path(__file__).parent
MIN_VALID    = 10
MAX_DEPTH_MM = 5000   # raw-count ceiling (also catches uint16 sentinel 65535)

# D405 depth scale: 0.0001 m per raw count (0.1 mm/count).
# Pegs are registered via the RealSense SDK which applies this scale internally,
# so cylinder deprojection must use the same value to align with pegs.json.
DEPTH_SCALE  = 0.0001  # m per raw count

# Board grid edges (pairs of peg IDs)
BOARD_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_intrinsics(meta_path: Path) -> dict:
    meta = yaml.safe_load(meta_path.read_text())
    intr = meta["intrinsics"]
    return {
        "fx":  float(intr["fx"]),
        "fy":  float(intr["fy"]),
        "ppx": float(intr["ppx"]),
        "ppy": float(intr["ppy"]),
    }


def load_pegs(pegs_path: Path):
    """Return list of (8,) xyz arrays [m, camera frame] and normalised up_axis."""
    data = json.loads(pegs_path.read_text())
    pegs = [np.array(p["xyz_m"]) for p in data["pegs"]]
    up   = np.array(data["up_axis"])
    up  /= np.linalg.norm(up)
    return pegs, up


def build_board_frame(pegs, up):
    """
    Board frame identical to 07b_visualize_trajectory.py.
    Returns (origin, R) where R columns = board X, Y, Z in camera frame.
    """
    board_Z = up
    raw_X   = pegs[4] - pegs[0]                              # peg0 → peg4
    board_X = raw_X - np.dot(raw_X, board_Z) * board_Z
    board_X /= np.linalg.norm(board_X)
    board_Y  = np.cross(board_Z, board_X)
    board_Y /= np.linalg.norm(board_Y)
    return pegs[0], np.column_stack([board_X, board_Y, board_Z])


def to_board_mm(pts_cam: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Camera-frame metres → board-frame millimetres.  pts_cam: (N,3) or (3,)."""
    return (R.T @ (np.atleast_2d(pts_cam) - origin).T).T * 1000.0


def deproject_full(mask: np.ndarray, depth_raw: np.ndarray, fx, fy, ppx, ppy):
    """
    Deproject every valid mask pixel to camera-frame 3D [m].
    Returns (pts (N,3), centroid (3,)) or (None, None) if too few valid pixels.
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None, None

    depths = depth_raw[ys, xs].astype(np.float64)
    valid  = (depths > 0) & (depths < MAX_DEPTH_MM)
    if valid.sum() < MIN_VALID:
        return None, None

    d_m = depths[valid] * DEPTH_SCALE       # raw counts → metres
    X   = (xs[valid] - ppx) * d_m / fx
    Y   = (ys[valid] - ppy) * d_m / fy
    Z   = d_m

    pts      = np.stack([X, Y, Z], axis=1)  # (N,3) metres
    centroid = np.median(pts, axis=0)
    return pts, centroid


def collect_valid_frames(masks_dir: Path, depth_dir: Path, images_dir: Path) -> list:
    valid = []
    for p in sorted(masks_dir.iterdir(), key=lambda p: int(p.stem)):
        if p.suffix.lower() != ".png":
            continue
        fidx = int(p.stem)
        if (depth_dir  / f"{fidx:05d}.png").exists() and \
           (images_dir / f"{fidx:05d}.jpg").exists():
            valid.append(fidx)
    return valid


# ── rendering ─────────────────────────────────────────────────────────────────

def draw_board_grid(ax, pegs_b):
    for a, b in BOARD_EDGES:
        ax.plot([pegs_b[a, 0], pegs_b[b, 0]],
                [pegs_b[a, 1], pegs_b[b, 1]],
                [pegs_b[a, 2], pegs_b[b, 2]],
                color="gray", lw=0.8, alpha=0.5, zorder=1)


def render(fig, frame_idx, masks_dir, depth_dir, images_dir,
           intr, pegs_cam, origin, R):
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
    pts_cam, centroid_cam = deproject_full(mask, depth_raw, fx, fy, ppx, ppy)

    # ── left: colour image + mask overlay + centroid cross-hair ──────────────
    ax2d = fig.add_subplot(1, 2, 1)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mask is not None and mask.any():
        overlay          = rgb.copy()
        overlay[mask == 1] = [0, 200, 0]
        rgb = (0.5 * rgb + 0.5 * overlay).astype(np.uint8)
    ax2d.imshow(rgb)

    if centroid_cam is not None:
        cx = centroid_cam[0] * fx / centroid_cam[2] + ppx
        cy = centroid_cam[1] * fy / centroid_cam[2] + ppy
        ax2d.plot(cx, cy, "r+", markersize=20, markeredgewidth=2.5)
        ax2d.plot(cx, cy, "ro", markersize=7,  fillstyle="none", markeredgewidth=1.5)

    ax2d.set_title(f"Frame {frame_idx:05d} — colour + mask  (red = centroid)", fontsize=9)
    ax2d.axis("off")

    # ── right: 3-D board-frame point cloud ────────────────────────────────────
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    if pts_cam is None:
        ax3d.set_title("No valid depth pixels in mask", fontsize=9)
        fig.canvas.draw_idle()
        return

    # transform to board frame [mm]
    pts_b      = to_board_mm(pts_cam,          origin, R)    # (N,3) mm
    centroid_b = to_board_mm(centroid_cam,     origin, R)[0] # (3,)  mm
    pegs_b     = to_board_mm(np.array(pegs_cam), origin, R)  # (8,3) mm

    # peg grid
    draw_board_grid(ax3d, pegs_b)

    # cloud coloured by Z (lift above board): low = blue, high = yellow
    z_vals = pts_b[:, 2]
    norm   = mcolors.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    ax3d.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                 c=cm.plasma(norm(z_vals)), s=5, alpha=0.6, linewidths=0)

    # centroid
    ax3d.scatter([centroid_b[0]], [centroid_b[1]], [centroid_b[2]],
                 color="red", s=140, zorder=10,
                 edgecolors="white", linewidths=1.2, label="centroid")

    # pegs
    peg_colors = plt.cm.tab10(np.linspace(0, 1, len(pegs_b)))
    for i, (pos, col) in enumerate(zip(pegs_b, peg_colors)):
        ax3d.scatter([pos[0]], [pos[1]], [pos[2]], marker="*", s=180,
                     color=col, edgecolors="k", linewidths=0.5, zorder=5)
        ax3d.text(pos[0], pos[1], pos[2], f"  {i}", fontsize=6, color=col)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.4, pad=0.1).set_label("Z lift (mm)", fontsize=7)

    spread_x = pts_b[:, 0].max() - pts_b[:, 0].min()
    spread_y = pts_b[:, 1].max() - pts_b[:, 1].min()
    spread_z = pts_b[:, 2].max() - pts_b[:, 2].min()
    ax3d.set_title(
        f"3-D cloud  {len(pts_b)} px   "
        f"ΔX={spread_x:.1f}  ΔY={spread_y:.1f}  ΔZ={spread_z:.1f} mm",
        fontsize=8)
    ax3d.set_xlabel("X board (mm)", fontsize=7)
    ax3d.set_ylabel("Y board (mm)", fontsize=7)
    ax3d.set_zlabel("Z lift (mm)",  fontsize=7)
    ax3d.set_xlim(-10, 90)
    ax3d.set_ylim(-10, 90)
    ax3d.set_zlim(-20, 50)
    ax3d.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Frame {frame_idx:05d}   centroid board-frame: "
        f"({centroid_b[0]:.1f}, {centroid_b[1]:.1f}, {centroid_b[2]:.1f}) mm   "
        f"[← / → step frames  |  q quit]",
        fontsize=9)

    fig.canvas.draw_idle()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Show full 3D point cloud of the detected cylinder per frame.")
    ap.add_argument("--frame",   type=int, default=0,
                    help="Starting frame index (default: 0)")
    ap.add_argument("--masks",   default="./masks/",
                    help="Binary mask directory from step 6 (default: ./masks/)")
    ap.add_argument("--depth",   required=True,
                    help="Depth PNG directory from step 5")
    ap.add_argument("--images",  required=True,
                    help="Colour JPEG directory from step 5")
    ap.add_argument("--meta",    required=True,
                    help="metadata.yaml from step 5")
    ap.add_argument("--pegs",    default="./pegs.json",
                    help="pegs.json from step 3 (default: ./pegs.json)")
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

    intr         = load_intrinsics(meta_path)
    pegs_cam, up = load_pegs(pegs_path)
    origin, R    = build_board_frame(pegs_cam, up)

    print(f"Depth scale  : {DEPTH_SCALE} m/count  (D405 default)")
    print(f"Board origin : peg 0  {pegs_cam[0].round(4)} m (camera frame)")
    print(f"Peg 4        : {pegs_cam[4].round(4)} m → defines board X axis")
    print()

    valid_frames = collect_valid_frames(masks_dir, depth_dir, images_dir)
    if not valid_frames:
        sys.exit("[error] no frames found with mask + depth + image")

    start = min(valid_frames, key=lambda f: abs(f - args.frame))
    idx   = valid_frames.index(start)

    print(f"Valid frames : {len(valid_frames)}  [{valid_frames[0]} … {valid_frames[-1]}]")
    print(f"Starting at  : frame {start}")
    print("← / → to step,  q to quit")

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
