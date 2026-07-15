"""
07_cylinder_trajectory.py — Turn SAM2 masks + depth frames into a 3D cylinder trajectory.

For each mask frame:
  1. Load the binary mask and aligned depth PNG (uint16, mm).
  2. Deproject every valid mask pixel to 3D using camera intrinsics.
  3. Take the median 3D point → cylinder centroid in camera frame (metres).
  4. Find nearest registered peg; compute lateral distance and height along
     the scene up-axis defined in pegs.json.
  5. Append a row to the output CSV.

NOTE: peg positions in pegs.json are in the *registration-session* camera
frame (Step 3).  If the camera moved between registration and recording the
lateral/height columns will be in a shifted coordinate frame.  Verify by
checking that cylinder Z values at rest on the peg board match the peg Z
values in pegs.json.

Output CSV columns:
  frame, t_s, X_m, Y_m, Z_m,
  nearest_peg_id, dist_to_peg_mm, lateral_mm, z_above_peg_mm,
  n_valid_px

Run:
    conda activate dvrk_ml
    python3 scripts/depth_camera/07_cylinder_trajectory.py \\
        --masks  masks/ \\
        --depth  recordings/trial_001_20260630_105549_frames/depth/ \\
        --meta   recordings/trial_001_20260630_105549_frames/metadata.yaml \\
        --pegs   pegs.json \\
        --output cylinder_trajectory.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

MIN_VALID_PX  = 10      # skip frame if fewer valid depth pixels than this
MAX_DEPTH_MM  = 5000    # ignore depth readings beyond 5 m (also catches uint16 sentinel 65535)


def load_intrinsics(meta_path: Path) -> dict:
    meta = yaml.safe_load(meta_path.read_text())
    intr = meta["intrinsics"]
    return {
        "fx":  float(intr["fx"]),
        "fy":  float(intr["fy"]),
        "ppx": float(intr["ppx"]),
        "ppy": float(intr["ppy"]),
        "target_fps": float(meta["target_fps"]),
    }


def deproject(mask: np.ndarray, depth_mm: np.ndarray, fx, fy, ppx, ppy):
    """
    Return (centroid_xyz_m, n_valid) where centroid_xyz_m is shape (3,)
    or None if not enough valid pixels.
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None, 0

    depths = depth_mm[ys, xs].astype(np.float64)
    valid  = (depths > 0) & (depths < MAX_DEPTH_MM)
    n_valid = int(valid.sum())

    if n_valid < MIN_VALID_PX:
        return None, n_valid

    d_m = depths[valid] / 1000.0
    X   = (xs[valid] - ppx) * d_m / fx
    Y   = (ys[valid] - ppy) * d_m / fy
    Z   = d_m

    pts     = np.stack([X, Y, Z], axis=1)   # (n_valid, 3)
    centroid = np.median(pts, axis=0)        # (3,)
    return centroid, n_valid


def nearest_peg(centroid: np.ndarray, pegs: list):
    """Return (peg_id, dist_m, nearest_xyz) for the closest peg."""
    best_dist = float("inf")
    best_id   = -1
    best_xyz  = None
    for peg in pegs:
        xyz  = np.array(peg["xyz_m"], dtype=np.float64)
        dist = float(np.linalg.norm(centroid - xyz))
        if dist < best_dist:
            best_dist = dist
            best_id   = peg["id"]
            best_xyz  = xyz
    return best_id, best_dist, best_xyz


def lateral_and_height(centroid: np.ndarray, peg_xyz: np.ndarray, up: np.ndarray):
    """
    Decompose (centroid - peg) into components along `up` and perpendicular.
    Returns (lateral_m, height_m).
    Positive height_m means the cylinder is above the peg tip.
    """
    delta    = centroid - peg_xyz
    height_m = float(np.dot(delta, up))
    lateral_m = float(np.linalg.norm(delta - height_m * up))
    return lateral_m, height_m


def collect_masks(masks_dir: Path) -> list:
    files = [p for p in masks_dir.iterdir() if p.suffix.lower() == ".png"]
    try:
        return sorted(files, key=lambda p: int(p.stem))
    except ValueError as e:
        sys.exit(f"[error] non-numeric mask filename: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Build 3D cylinder trajectory from SAM2 masks + depth frames.")
    ap.add_argument("--masks",  default="./masks/",
                    help="Directory of binary mask PNGs from step 6 (default: ./masks/)")
    ap.add_argument("--depth",  required=True,
                    help="Directory of uint16 depth PNGs from step 5 (e.g. recordings/.../depth/)")
    ap.add_argument("--meta",   required=True,
                    help="metadata.yaml written by step 5 (contains intrinsics + fps)")
    ap.add_argument("--pegs",   default="./pegs.json",
                    help="pegs.json written by step 3 (default: ./pegs.json)")
    ap.add_argument("--output", default="./cylinder_trajectory.csv",
                    help="Output CSV path (default: ./cylinder_trajectory.csv)")
    args = ap.parse_args()

    masks_dir  = Path(args.masks).resolve()
    depth_dir  = Path(args.depth).resolve()
    meta_path  = Path(args.meta).resolve()
    pegs_path  = Path(args.pegs).resolve()
    out_path   = Path(args.output).resolve()

    for p, name in [(masks_dir, "masks"), (depth_dir, "depth"),
                    (meta_path, "meta"), (pegs_path, "pegs")]:
        if not p.exists():
            sys.exit(f"[error] {name} not found: {p}")

    # ── load config ───────────────────────────────────────────────────────────
    intr = load_intrinsics(meta_path)
    fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    target_fps = intr["target_fps"]

    import json
    pegs_data = json.loads(pegs_path.read_text())
    pegs_list = pegs_data["pegs"]
    up = np.array(pegs_data["up_axis"], dtype=np.float64)
    up = up / np.linalg.norm(up)

    mask_files = collect_masks(masks_dir)
    if not mask_files:
        sys.exit(f"[error] no PNG files found in {masks_dir}")

    print(f"Masks      : {len(mask_files)}  from {masks_dir.name}/")
    print(f"Depth      : {depth_dir.name}/")
    print(f"Intrinsics : fx={fx:.1f}  fy={fy:.1f}  pp=({ppx:.1f},{ppy:.1f})")
    print(f"Pegs       : {len(pegs_list)} registered")
    print(f"Up axis    : {up.round(3).tolist()}")
    print(f"Output     : {out_path}")
    print()

    # ── process frames ────────────────────────────────────────────────────────
    rows       = []
    n_skipped  = 0

    for mask_path in mask_files:
        frame_idx = int(mask_path.stem)
        t_s       = frame_idx / target_fps

        depth_path = depth_dir / f"{mask_path.stem}.png"
        if not depth_path.exists():
            print(f"[warn] no depth for frame {mask_path.stem} — skipping")
            n_skipped += 1
            continue

        mask     = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

        if mask is None or depth_mm is None:
            print(f"[warn] could not read frame {mask_path.stem} — skipping")
            n_skipped += 1
            continue

        centroid, n_valid = deproject(mask, depth_mm, fx, fy, ppx, ppy)

        if centroid is None:
            rows.append({
                "frame": frame_idx, "t_s": round(t_s, 4),
                "X_m": "", "Y_m": "", "Z_m": "",
                "nearest_peg_id": "", "dist_to_peg_mm": "",
                "lateral_mm": "", "z_above_peg_mm": "",
                "n_valid_px": n_valid,
            })
            n_skipped += 1
            continue

        peg_id, dist_m, peg_xyz = nearest_peg(centroid, pegs_list)
        lat_m, hgt_m             = lateral_and_height(centroid, peg_xyz, up)

        rows.append({
            "frame":          frame_idx,
            "t_s":            round(t_s, 4),
            "X_m":            round(float(centroid[0]), 6),
            "Y_m":            round(float(centroid[1]), 6),
            "Z_m":            round(float(centroid[2]), 6),
            "nearest_peg_id": peg_id,
            "dist_to_peg_mm": round(dist_m * 1000, 2),
            "lateral_mm":     round(lat_m * 1000, 2),
            "z_above_peg_mm": round(hgt_m * 1000, 2),
            "n_valid_px":     n_valid,
        })

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx:05d}  centroid=({centroid[0]:.3f}, "
                  f"{centroid[1]:.3f}, {centroid[2]:.3f}) m  "
                  f"peg={peg_id}  lat={lat_m*1000:.1f} mm  "
                  f"h={hgt_m*1000:.1f} mm  n={n_valid}")

    # ── write CSV ─────────────────────────────────────────────────────────────
    fields = ["frame", "t_s", "X_m", "Y_m", "Z_m",
              "nearest_peg_id", "dist_to_peg_mm", "lateral_mm",
              "z_above_peg_mm", "n_valid_px"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ── summary ───────────────────────────────────────────────────────────────
    good = [r for r in rows if r["Z_m"] != ""]
    print()
    print(f"Done.  {len(good)} frames with valid centroid, "
          f"{n_skipped} skipped.")
    if good:
        zs  = [r["Z_m"]            for r in good]
        lats = [r["lateral_mm"]    for r in good]
        hgts = [r["z_above_peg_mm"] for r in good]
        print(f"  Z range      : {min(zs):.3f} – {max(zs):.3f} m")
        print(f"  lateral range: {min(lats):.1f} – {max(lats):.1f} mm")
        print(f"  height range : {min(hgts):.1f} – {max(hgts):.1f} mm")
    print(f"CSV → {out_path}")


if __name__ == "__main__":
    main()
