"""
07b_visualize_trajectory.py — Cylinder trajectory in peg-board coordinate frame.

Coordinate frame (board frame, all values in mm):
  Origin : peg 0  (bottom-left corner of board)
  X      : peg 0 → peg 4  (left → right along bottom row)
  Y      : peg 0 → peg 3  (bottom row → top row)
  Z      : up_axis         (perpendicular to board surface = "lift height")

Peg layout (8 pegs, 2 columns × 4 rows, registration order from step 3):
  Row 1 (top):    Peg 3 ——— Peg 7
                  Peg 2     Peg 6
                  Peg 1     Peg 5
  Row 4 (bottom): Peg 0 ——— Peg 4
                  Left      Right   ← origin = Peg 0

Plots:
  • 3-D view   : trajectory coloured by time, pegs as stars, board grid
  • Top view   : X vs Y  (bird's-eye, shows 2×4 peg layout)
  • Side view  : X vs Z  (lift height vs left-right position)
  • Front view : Y vs Z  (lift height vs row position)

Run:
    conda activate dvrk_ml
    python3 scripts/depth_camera/07b_visualize_trajectory.py
    python3 scripts/depth_camera/07b_visualize_trajectory.py \\
        --trajectory cylinder_trajectory.csv --pegs pegs.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np

SCRIPT_DIR = Path(__file__).parent

# Board grid: pairs of peg IDs to connect with lines
BOARD_EDGES = [
    (0, 1), (1, 2), (2, 3),          # left column (bottom → top)
    (4, 5), (5, 6), (6, 7),          # right column
    (0, 4), (1, 5), (2, 6), (3, 7),  # horizontal rows
]


def load_trajectory(csv_path: Path):
    """Return (t, X_cam, Y_cam, Z_cam) arrays, empty rows skipped."""
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            if parts[idx["X_m"]] == "":
                continue
            rows.append((
                float(parts[idx["t_s"]]),
                float(parts[idx["X_m"]]),
                float(parts[idx["Y_m"]]),
                float(parts[idx["Z_m"]]),
            ))
    if not rows:
        sys.exit("[error] no valid rows in trajectory CSV")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def load_pegs(pegs_path: Path):
    """Return list of {"id", "xyz"} dicts (camera frame, metres) and up_axis."""
    data = json.loads(pegs_path.read_text())
    pegs = [{"id": p["id"], "xyz": np.array(p["xyz_m"])} for p in data["pegs"]]
    up   = np.array(data.get("up_axis", [0, -1, 0]))
    up  /= np.linalg.norm(up)
    return pegs, up


def build_board_frame(pegs, up):
    """
    Derive peg-board coordinate frame from registered peg positions.

    Returns:
        origin (3,) — peg 0 position in camera frame [m]
        R      (3,3) — columns are board X, Y, Z axes in camera frame

    Board X : peg0→peg4, projected onto board plane  (left → right)
    Board Z : up_axis (normal to board surface)
    Board Y : cross(Z, X)                            (bottom row → top row)
    """
    p0 = pegs[0]["xyz"]
    p4 = pegs[4]["xyz"]

    board_Z = up  # already normalised

    raw_X   = p4 - p0
    board_X = raw_X - np.dot(raw_X, board_Z) * board_Z
    board_X /= np.linalg.norm(board_X)

    board_Y = np.cross(board_Z, board_X)
    board_Y /= np.linalg.norm(board_Y)

    R = np.column_stack([board_X, board_Y, board_Z])
    return p0, R


def to_board_mm(pts_cam, origin, R):
    """Camera-frame metres → board-frame millimetres. pts_cam: (N,3) or (3,)."""
    return (R.T @ (pts_cam - origin).T).T * 1000.0


def scatter_line_3d(ax, X, Y, Z, t, lw=1.2, alpha=0.85):
    norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
    cmap = cm.plasma
    for i in range(len(X) - 1):
        ax.plot(X[i:i+2], Y[i:i+2], Z[i:i+2],
                color=cmap(norm(t[i])), lw=lw, alpha=alpha)
    return norm, cmap


def scatter_line_2d(ax, Xa, Ya, t):
    norm   = mcolors.Normalize(vmin=t.min(), vmax=t.max())
    points = np.array([Xa, Ya]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", norm=norm, linewidth=1.2, alpha=0.8)
    lc.set_array(t[:-1])
    ax.add_collection(lc)


def draw_board_grid_3d(ax, peg_board):
    for a, b in BOARD_EDGES:
        pa, pb = peg_board[a], peg_board[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color="gray", lw=0.7, alpha=0.5, zorder=1)


def draw_board_grid_2d(ax, peg_board, xi, yi):
    for a, b in BOARD_EDGES:
        ax.plot([peg_board[a][xi], peg_board[b][xi]],
                [peg_board[a][yi], peg_board[b][yi]],
                color="gray", lw=0.7, alpha=0.5, zorder=1)


def draw_pegs_3d(ax, peg_board):
    colors = plt.cm.tab10(np.linspace(0, 1, len(peg_board)))
    for i, (pos, col) in enumerate(zip(peg_board, colors)):
        ax.scatter([pos[0]], [pos[1]], [pos[2]], marker="*", s=200,
                   color=col, edgecolors="k", linewidths=0.6, zorder=5,
                   label=f"peg {i}")
        ax.text(pos[0], pos[1], pos[2], f"  {i}", fontsize=7, color=col)


def draw_pegs_2d(ax, peg_board, xi, yi):
    colors = plt.cm.tab10(np.linspace(0, 1, len(peg_board)))
    for i, (pos, col) in enumerate(zip(peg_board, colors)):
        ax.scatter([pos[xi]], [pos[yi]], marker="*", s=120,
                   color=col, edgecolors="k", linewidths=0.4, zorder=5)
        ax.annotate(str(i), (pos[xi], pos[yi]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")


def main():
    ap = argparse.ArgumentParser(
        description="Visualise cylinder trajectory in peg-board coordinate frame.")
    ap.add_argument("--trajectory", default=str(SCRIPT_DIR / "cylinder_trajectory.csv"))
    ap.add_argument("--pegs",       default=str(SCRIPT_DIR / "pegs.json"))
    ap.add_argument("--save",       default=None,
                    help="Save figure to this path instead of showing interactively.")
    args = ap.parse_args()

    traj_path = Path(args.trajectory)
    pegs_path = Path(args.pegs)
    for p, name in [(traj_path, "trajectory CSV"), (pegs_path, "pegs.json")]:
        if not p.exists():
            sys.exit(f"[error] {name} not found: {p}")

    # ── load & transform to board frame ──────────────────────────────────────
    t, Xc, Yc, Zc = load_trajectory(traj_path)
    pegs, up       = load_pegs(pegs_path)

    origin, R = build_board_frame(pegs, up)

    cam_pts  = np.stack([Xc, Yc, Zc], axis=1)       # (N,3) metres
    board_t  = to_board_mm(cam_pts, origin, R)        # (N,3) mm
    Xb, Yb, Zb = board_t[:, 0], board_t[:, 1], board_t[:, 2]

    peg_cam   = np.array([p["xyz"] for p in pegs])   # (8,3) metres
    peg_board = to_board_mm(peg_cam, origin, R)       # (8,3) mm

    print(f"Trajectory : {len(t)} frames  t=[{t[0]:.1f}s … {t[-1]:.1f}s]")
    print("Board frame [mm]:")
    print(f"  X (left→right) : [{Xb.min():.1f}, {Xb.max():.1f}]")
    print(f"  Y (bot→top)    : [{Yb.min():.1f}, {Yb.max():.1f}]")
    print(f"  Z (lift)       : [{Zb.min():.1f}, {Zb.max():.1f}]")
    print("Pegs in board frame [mm]:")
    for i, pos in enumerate(peg_board):
        print(f"  peg {i}: ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:5.1f})")

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Cylinder Trajectory — Board Frame  "
        "(X = left→right,  Y = bottom→top,  Z = lift above board)  [mm]",
        fontsize=12)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axXY = fig.add_subplot(3, 2, 2)   # top view
    axXZ = fig.add_subplot(3, 2, 4)   # side view
    axYZ = fig.add_subplot(3, 2, 6)   # front view

    # ── 3-D plot ──────────────────────────────────────────────────────────────
    draw_board_grid_3d(ax3d, peg_board)
    norm, cmap = scatter_line_3d(ax3d, Xb, Yb, Zb, t)
    ax3d.scatter([Xb[0]],  [Yb[0]],  [Zb[0]],  color="lime", s=60, zorder=6,
                 label="start", edgecolors="k", linewidths=0.5)
    ax3d.scatter([Xb[-1]], [Yb[-1]], [Zb[-1]], color="red",  s=60, zorder=6,
                 label="end",   edgecolors="k", linewidths=0.5)
    draw_pegs_3d(ax3d, peg_board)

    ax3d.set_xlabel("X board (mm)", fontsize=8)
    ax3d.set_ylabel("Y board (mm)", fontsize=8)
    ax3d.set_zlabel("Z lift (mm)",  fontsize=8)
    ax3d.set_title("3-D view (drag to rotate)", fontsize=9)
    ax3d.legend(fontsize=6, loc="upper left", markerscale=0.7)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.08).set_label("time (s)", fontsize=8)

    # ── 2-D projections ───────────────────────────────────────────────────────
    proj_specs = [
        (axXY, Xb, Yb, 0, 1, "X board (mm)", "Y board (mm)", "Top view  (XY)"),
        (axXZ, Xb, Zb, 0, 2, "X board (mm)", "Z lift (mm)",  "Side view  (XZ)"),
        (axYZ, Yb, Zb, 1, 2, "Y board (mm)", "Z lift (mm)",  "Front view  (YZ)"),
    ]
    for ax2, Xa, Ya, xi, yi, xl, yl, title in proj_specs:
        draw_board_grid_2d(ax2, peg_board, xi, yi)
        scatter_line_2d(ax2, Xa, Ya, t)
        ax2.scatter([Xa[0]],  [Ya[0]],  color="lime", s=30, zorder=5,
                    edgecolors="k", linewidths=0.4)
        ax2.scatter([Xa[-1]], [Ya[-1]], color="red",  s=30, zorder=5,
                    edgecolors="k", linewidths=0.4)
        draw_pegs_2d(ax2, peg_board, xi, yi)
        ax2.autoscale()
        ax2.set_xlabel(xl, fontsize=7)
        ax2.set_ylabel(yl, fontsize=7)
        ax2.set_title(title, fontsize=8)
        ax2.tick_params(labelsize=6)
        ax2.grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        print("\nClose the window to exit. Drag the 3-D panel to rotate.")
        plt.show()


if __name__ == "__main__":
    main()
