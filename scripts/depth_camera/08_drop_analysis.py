"""
08_drop_analysis.py — Detect drop / placement events in one trial's cylinder trajectory.

Consumes the CSV written by 07_cylinder_trajectory.py (frame, t_s, X_m, Y_m, Z_m,
nearest_peg_id, dist_to_peg_mm, lateral_mm, z_above_peg_mm, n_valid_px) plus
pegs.json, and produces a per-event table:

  - MOVING segments are found from 3-D velocity spikes (same spike/settle
    counter idea as 7_velocity_check.py, just applied to real 3-D speed
    instead of 2-D pixel speed).
  - Each segment's settle point is classified PLACEMENT (settled at a peg
    tip) or DROP (settled far from every peg tip).
  - For DROPs, the landing offset relative to the nearest peg is decomposed
    into the board's own X/Y axes (same frame as 07b/07c: X spans the two
    outer columns, Y spans the rows) — this gives a signed direction and
    magnitude, plus how far below the peg tip it landed.
  - A path-efficiency and reversal-count struggle metric is computed for
    every segment (straight-line distance vs. actual distance travelled,
    and how many times the motion changed direction before settling).

NOTE on direction sign: which physical side "+X" / "+Y" corresponds to is
fixed by peg0/peg4/up_axis in pegs.json, but hasn't been visually verified
against a known deliberate drop. Do one manual test (drop the cylinder to a
known side of a peg) and check the printed sign before trusting direction
labels for real feedback.

Run:
    conda activate dvrk_ml
    python3 08_drop_analysis.py \\
        --trajectory cylinder_trajectory_valid12.csv \\
        --pegs pegs.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STATE_SETTLED = "SETTLED"
STATE_MOVING  = "MOVING"


# ── board frame (same convention as 07b/07c) ─────────────────────────────────

def build_board_frame(pegs_xyz: np.ndarray, up: np.ndarray):
    board_Z = up
    raw_X   = pegs_xyz[4] - pegs_xyz[0]
    board_X = raw_X - np.dot(raw_X, board_Z) * board_Z
    board_X /= np.linalg.norm(board_X)
    board_Y  = np.cross(board_Z, board_X)
    board_Y /= np.linalg.norm(board_Y)
    origin   = pegs_xyz[0]
    return origin, np.column_stack([board_X, board_Y, board_Z])


def to_board_mm(pt_cam: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R.T @ (pt_cam - origin)) * 1000.0


# ── peg metadata from ID (matches the side-then-center click order) ─────────

def peg_meta(peg_id: int) -> dict:
    peg_id = int(peg_id)
    group  = "center" if 8 <= peg_id <= 11 else "side"
    column = "A" if 0 <= peg_id <= 3 else ("C" if 4 <= peg_id <= 7 else "B")
    row    = peg_id % 4
    row_label = "end" if row in (0, 3) else "mid"
    return {"group": group, "column": column, "row": row, "row_label": row_label}


def peg_spacing_mm(pegs_xyz: np.ndarray) -> float:
    """Median nearest-neighbour distance between registered pegs — a
    self-calibrated scale for 'close to a peg' that doesn't depend on
    knowing the physical board dimensions."""
    n = len(pegs_xyz)
    nn = []
    for i in range(n):
        d = [np.linalg.norm(pegs_xyz[i] - pegs_xyz[j]) for j in range(n) if j != i]
        nn.append(min(d))
    return float(np.median(nn)) * 1000.0


# ── event detection ───────────────────────────────────────────────────────────

def smooth_positions(pos: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average — reduces per-frame depth/mask jitter before
    velocity and direction-reversal are computed from it. window<=1 disables."""
    if window <= 1:
        return pos
    return pd.DataFrame(pos).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def detect_events(df: pd.DataFrame, pegs_xyz: np.ndarray, origin, R, spacing_mm: float,
                   vel_drop_mms: float, vel_stat_mms: float,
                   spike_frames: int, settle_frames: int,
                   place_radius_frac: float, seat_height_mm: float,
                   smooth_window: int = 3) -> list:
    pos_raw = df[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=np.float64) * 1000.0  # mm
    pos_sm  = smooth_positions(pos_raw, smooth_window)
    t   = df["t_s"].to_numpy(dtype=np.float64)
    n   = len(df)

    # Spike/settle detection needs the RAW signal — smoothing a short, genuine
    # spike (a real drop can be just 2-3 samples wide at 5-15 fps) dilutes its
    # peak below threshold. Smoothed positions are used below only for the
    # struggle metrics (reversal count, path efficiency), which is what
    # actually benefits from suppressing per-frame depth/mask jitter.
    vel = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt > 1e-6:
            vel[i] = np.linalg.norm(pos_raw[i] - pos_raw[i - 1]) / dt   # mm/s

    events = []
    state         = STATE_SETTLED
    spike_count   = 0
    settle_count  = 0
    move_start_i  = None

    for i in range(1, n):
        if vel[i] >= vel_drop_mms:
            spike_count += 1
            if move_start_i is None:
                move_start_i = i - 1
        else:
            spike_count = 0

        if vel[i] < vel_stat_mms:
            settle_count += 1
        else:
            settle_count = 0

        if spike_count >= spike_frames and state != STATE_MOVING:
            state = STATE_MOVING

        if state == STATE_MOVING and settle_count >= settle_frames:
            settle_i = i
            start_i  = move_start_i if move_start_i is not None else max(0, i - 1)

            row       = df.iloc[settle_i]
            peg_id    = int(row["nearest_peg_id"])
            dist_mm   = float(row["dist_to_peg_mm"])
            height_mm = float(row["z_above_peg_mm"])

            is_placement = (dist_mm <= place_radius_frac * spacing_mm
                             and abs(height_mm) <= seat_height_mm)

            # struggle metrics over the motion span (smoothed — jitter-resistant)
            span_pos   = pos_sm[start_i:settle_i + 1]
            straight   = np.linalg.norm(span_pos[-1] - span_pos[0])
            travelled  = float(np.sum(np.linalg.norm(np.diff(span_pos, axis=0), axis=1)))
            efficiency = (straight / travelled) if travelled > 1e-6 else None

            reversals = 0
            if len(span_pos) >= 3:
                main_dir = span_pos[-1] - span_pos[0]
                nrm = np.linalg.norm(main_dir)
                if nrm > 1e-6:
                    main_dir /= nrm
                    steps = np.diff(span_pos, axis=0)
                    proj  = steps @ main_dir
                    signs = np.sign(proj[np.abs(proj) > 1e-6])
                    reversals = int(np.sum(signs[1:] != signs[:-1]))

            event = {
                "frame_start":   int(df.iloc[start_i]["frame"]),
                "frame_settle":  int(df.iloc[settle_i]["frame"]),
                "t_start_s":     round(float(t[start_i]), 3),
                "t_settle_s":    round(float(t[settle_i]), 3),
                "nearest_peg_id": peg_id,
                "peg_group":     peg_meta(peg_id)["group"],
                "peg_column":    peg_meta(peg_id)["column"],
                "peg_row_label": peg_meta(peg_id)["row_label"],
                "event_type":    "PLACEMENT" if is_placement else "DROP",
                "dist_to_peg_mm": round(dist_mm, 1),
                "height_mm":      round(height_mm, 1),
                "path_efficiency": round(efficiency, 3) if efficiency is not None else "",
                "n_reversals":    reversals,
            }

            if not is_placement:
                settle_m  = pos_sm[settle_i] / 1000.0   # smoothed, back to camera-frame metres
                peg_xyz_m = pegs_xyz[peg_id]
                offset_board = to_board_mm(settle_m, origin, R) - to_board_mm(peg_xyz_m, origin, R)
                dx, dy, dz = offset_board
                dominant   = "column" if abs(dx) >= abs(dy) else "row"
                event["offset_x_mm"]      = round(float(dx), 1)   # board X: across the two outer columns
                event["offset_y_mm"]      = round(float(dy), 1)   # board Y: along the rows
                event["dominant_axis"]    = dominant
                event["dominant_sign"]    = "+" if (dx if dominant == "column" else dy) >= 0 else "-"
                event["offset_mag_mm"]    = round(float(np.hypot(dx, dy)), 1)
            else:
                event["offset_x_mm"] = event["offset_y_mm"] = ""
                event["dominant_axis"] = event["dominant_sign"] = event["offset_mag_mm"] = ""

            events.append(event)

            state        = STATE_SETTLED
            spike_count  = 0
            settle_count = 0
            move_start_i = None

    return events


def main():
    ap = argparse.ArgumentParser(
        description="Detect drop/placement events from a cylinder_trajectory CSV.")
    ap.add_argument("--trajectory", required=True, help="CSV from 07_cylinder_trajectory.py")
    ap.add_argument("--pegs",       default="./pegs.json")
    ap.add_argument("--output",     default=None,
                    help="Output events CSV (default: <trajectory-stem>_events.csv)")
    ap.add_argument("--vel-drop-mms",  type=float, default=250.0, dest="vel_drop_mms",
                    help="3-D speed (mm/s) above which motion counts as a spike (default: 250)")
    ap.add_argument("--vel-stat-mms",  type=float, default=20.0,  dest="vel_stat_mms",
                    help="3-D speed (mm/s) below which motion counts as settled (default: 20)")
    ap.add_argument("--spike-frames",  type=int,   default=2, dest="spike_frames",
                    help="Consecutive spike samples before declaring MOVING (default: 2)")
    ap.add_argument("--settle-frames", type=int,   default=5, dest="settle_frames",
                    help="Consecutive settled samples before declaring settle (default: 5)")
    ap.add_argument("--place-radius-frac", type=float, default=0.5, dest="place_radius_frac",
                    help="Fraction of median peg spacing counted as 'at a peg' (default: 0.5)")
    ap.add_argument("--seat-height-mm", type=float, default=10.0, dest="seat_height_mm",
                    help="Height tolerance (mm) around a peg tip to call it seated (default: 10)")
    ap.add_argument("--smooth-window", type=int, default=3, dest="smooth_window",
                    help="Centered moving-average window (samples) on 3-D position before "
                         "computing velocity/reversals — reduces depth/mask jitter. "
                         "1 disables smoothing (default: 3)")
    args = ap.parse_args()

    traj_path = Path(args.trajectory).resolve()
    pegs_path = Path(args.pegs).resolve()
    if not traj_path.exists():
        sys.exit(f"[error] trajectory not found: {traj_path}")
    if not pegs_path.exists():
        sys.exit(f"[error] pegs not found: {pegs_path}")

    out_path = (Path(args.output).resolve() if args.output
                else traj_path.parent / f"{traj_path.stem}_events.csv")

    df = pd.read_csv(traj_path)
    df = df.dropna(subset=["X_m", "Y_m", "Z_m", "nearest_peg_id"]).reset_index(drop=True)
    if len(df) < 3:
        sys.exit(f"[error] not enough valid rows in {traj_path.name} ({len(df)})")

    pegs_data = json.loads(pegs_path.read_text())
    pegs_sorted = sorted(pegs_data["pegs"], key=lambda p: p["id"])
    pegs_xyz = np.array([p["xyz_m"] for p in pegs_sorted], dtype=np.float64)
    up = np.array(pegs_data["up_axis"], dtype=np.float64)
    up = up / np.linalg.norm(up)

    origin, R = build_board_frame(pegs_xyz, up)
    spacing_mm = peg_spacing_mm(pegs_xyz)

    print(f"Trajectory   : {traj_path.name}  ({len(df)} valid rows)")
    print(f"Pegs         : {len(pegs_xyz)} registered  (median spacing = {spacing_mm:.1f} mm)")
    print(f"Spike/settle : >= {args.vel_drop_mms:.0f} mm/s x{args.spike_frames}  /  "
          f"< {args.vel_stat_mms:.0f} mm/s x{args.settle_frames}")
    print()

    events = detect_events(
        df, pegs_xyz, origin, R, spacing_mm,
        args.vel_drop_mms, args.vel_stat_mms,
        args.spike_frames, args.settle_frames,
        args.place_radius_frac, args.seat_height_mm,
        smooth_window=args.smooth_window)

    if not events:
        print("No settle events detected — check thresholds against this trial's speeds.")
        return

    n_drop = sum(1 for e in events if e["event_type"] == "DROP")
    n_place = len(events) - n_drop
    print(f"── {len(events)} event(s): {n_place} placement(s), {n_drop} drop(s) ──\n")

    header = (f"{'type':<10}{'peg':>4}{'grp':>7}{'row':>5}{'dist_mm':>9}{'height_mm':>11}"
              f"{'axis':>8}{'sign':>5}{'mag_mm':>8}{'eff':>7}{'revs':>6}")
    print(header)
    for e in events:
        print(f"{e['event_type']:<10}{e['nearest_peg_id']:>4}{e['peg_group']:>7}"
              f"{e['peg_row_label']:>5}{e['dist_to_peg_mm']:>9}{e['height_mm']:>11}"
              f"{e['dominant_axis']:>8}{e['dominant_sign']:>5}{e['offset_mag_mm']:>8}"
              f"{e['path_efficiency']:>7}{e['n_reversals']:>6}")

    pd.DataFrame(events).to_csv(out_path, index=False)
    print(f"\nEvents → {out_path}")


if __name__ == "__main__":
    main()
