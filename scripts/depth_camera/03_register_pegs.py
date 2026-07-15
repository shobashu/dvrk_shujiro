"""
03_register_pegs.py — Register peg-top 3D positions with alignment verification.

Three-phase workflow
────────────────────
Phase 1 — CALIBRATE
  Click 3 points you know are at the same real height (e.g. three corners of
  the task board).  The script fits a plane through those points and uses its
  normal as the true "up" axis, correcting for any camera tilt or roll.
  Skip with 's' to use raw camera −Y as the height axis.

Phase 2 — REGISTER
  Left-click each peg top.  Each click captures 30 aligned frames, pools the
  7×7 depth patch, filters to [0.05, 0.40] m, takes the median, and
  deprojects to (X, Y, Z) in the camera frame.

Phase 3 — MAP  (blocking matplotlib window)
  3-D scatter of all registered peg positions with a reference height plane.
  Height differences between every pair of pegs are printed to the terminal
  so you can spot outliers.
    Enter / S  →  save pegs.json and exit
    R          →  back to registration (keep existing pegs; add / undo more)
    Q          →  quit without saving

Controls (CALIBRATE):
  left-click   record a same-height reference point  (need exactly 3)
  u            undo last calibration point
  s            skip calibration, use default −Y axis
  q            quit

Controls (REGISTER):
  left-click   register a peg top  (triggers 30-frame capture)
  u            undo last peg
  r            reset all pegs
  Enter        open 3-D map
  q            quit without saving

Run:
    python3 scripts/depth_camera/03_register_pegs.py
"""

import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── config ────────────────────────────────────────────────────────────────────
STREAM_WIDTH       = 640   # 848×480 not supported simultaneously on this setup
STREAM_HEIGHT      = 480
FPS                = 30
WARMUP_FRAMES      = 30
CAPTURE_FRAMES     = 30    # frames pooled per peg click
WINDOW_HALF        = 3     # 7×7 patch = (2*3+1)²
DEPTH_MIN_M        = 0.05
DEPTH_MAX_M        = 0.40
LOW_CONF_THRESHOLD = 50    # warn if fewer valid samples than this

SETTINGS_PATH = Path(__file__).parent / "camera_settings.json"
PEGS_PATH     = Path(__file__).parent / "pegs.json"

PEG_PALETTE = [
    (0,   255, 128),
    (255, 128,   0),
    (0,   180, 255),
    (220,   0, 255),
    (0,   255, 255),
    (255, 220,   0),
    (180,   0, 255),
    (0,   220, 180),
]

PHASE_CALIBRATE = "calibrate"
PHASE_REGISTER  = "register"

# ── shared state ──────────────────────────────────────────────────────────────
state = {
    "phase":         PHASE_CALIBRATE,
    "calib_pts":     [],
    "up_axis":       np.array([0., -1., 0.]),   # default: camera −Y = world up
    "pegs":          [],
    "pending_click": None,    # (u, v) set by REGISTER-phase click
    "depth_frame":   None,    # latest frame, used by CALIBRATE-phase click
    "intrinsics":    None,
    "pipeline":      None,
    "align":         None,
}


# ── sensor helpers ────────────────────────────────────────────────────────────
def find_color_sensor(device):
    for s in device.query_sensors():
        if s.supports(rs.option.enable_auto_white_balance):
            return s
    return None


def load_settings(depth_sensor, color_sensor):
    if not SETTINGS_PATH.exists():
        print("  camera_settings.json not found — using camera defaults.")
        return
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
            print(f"  {key} = {val}")
        except Exception as exc:
            print(f"  [!] Could not set {key}: {exc}")


# ── depth / geometry helpers ──────────────────────────────────────────────────
def sample_patch_single(depth_frame, u, v):
    """7×7 valid depths from one frame (for calibration phase)."""
    h, w = depth_frame.get_height(), depth_frame.get_width()
    u0, u1 = max(0, u - WINDOW_HALF), min(w - 1, u + WINDOW_HALF)
    v0, v1 = max(0, v - WINDOW_HALF), min(h - 1, v + WINDOW_HALF)
    vals = []
    for row in range(v0, v1 + 1):
        for col in range(u0, u1 + 1):
            d = depth_frame.get_distance(col, row)
            if DEPTH_MIN_M <= d <= DEPTH_MAX_M:
                vals.append(d)
    return vals


def get_xyz_single(u, v):
    """Deproject pixel to 3-D using the current stored depth frame (single frame)."""
    df   = state["depth_frame"]
    intr = state["intrinsics"]
    if df is None or intr is None:
        return None
    vals = sample_patch_single(df, u, v)
    if not vals:
        print(f"  [!] No valid depth at ({u}, {v})")
        return None
    d = float(np.median(vals))
    return tuple(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], d))


def capture_peg_xyz(u, v):
    """
    Pool CAPTURE_FRAMES aligned frames, median-filter the 7×7 patch,
    deproject.  Returns (xyz_list, n_valid) or (None, 0).
    """
    all_depths = []
    for _ in range(CAPTURE_FRAMES):
        frames  = state["pipeline"].wait_for_frames(timeout_ms=5000)
        aligned = state["align"].process(frames)
        df = aligned.get_depth_frame()
        if df:
            all_depths.extend(sample_patch_single(df, u, v))

    n = len(all_depths)
    if n == 0:
        print(f"  [!] No valid depth at ({u}, {v}) — check range / occlusion.")
        return None, 0
    if n < LOW_CONF_THRESHOLD:
        print(f"  [!] Low confidence: only {n}/{CAPTURE_FRAMES * 49} valid samples.")

    d   = float(np.median(all_depths))
    xyz = rs.rs2_deproject_pixel_to_point(state["intrinsics"], [float(u), float(v)], d)
    return list(xyz), n


def compute_up_axis(pt_a, pt_b, pt_c):
    """Fit a plane through three co-planar same-height points; return its normal."""
    v1  = np.array(pt_b) - np.array(pt_a)
    v2  = np.array(pt_c) - np.array(pt_a)
    up  = np.cross(v1, v2)
    nrm = np.linalg.norm(up)
    if nrm < 1e-6:
        print("  [!] Calibration points are collinear — keeping default axis.")
        return state["up_axis"].copy()
    up /= nrm
    if up[1] > 0:    # ensure it points "up" (−Y in camera frame when roughly level)
        up = -up
    return up


def height_diff(xyz_from, xyz_to):
    d = np.array(xyz_to) - np.array(xyz_from)
    return float(np.dot(d, state["up_axis"]))


# ── mouse callback ────────────────────────────────────────────────────────────
def on_mouse(event, u, v, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state["phase"] == PHASE_CALIBRATE:
        if len(state["calib_pts"]) >= 3:
            return
        xyz = get_xyz_single(u, v)
        if xyz is None:
            return
        state["calib_pts"].append({"pixel": (u, v), "xyz": xyz})
        n = len(state["calib_pts"])
        X, Y, Z = xyz
        print(f"  Cal point {n}/3 → X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m")
        if n == 3:
            up = compute_up_axis(*[p["xyz"] for p in state["calib_pts"]])
            state["up_axis"] = up
            print(f"  Up axis (camera frame): [{up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f}]")
            print("  Calibration done — advancing to peg registration.\n")
            state["phase"] = PHASE_REGISTER

    elif state["phase"] == PHASE_REGISTER:
        state["pending_click"] = (u, v)


# ── overlays ──────────────────────────────────────────────────────────────────
def draw_calib_markers(frame):
    for i, pt in enumerate(state["calib_pts"]):
        u, v = pt["pixel"]
        cv2.drawMarker(frame, (u, v), (0, 255, 200), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(frame, f"Ref {i+1}", (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)


def draw_peg_markers(frame):
    intr = state["intrinsics"]
    for peg in state["pegs"]:
        # Re-project from stored 3-D position so overlay stays geometrically locked
        px, py = rs.rs2_project_point_to_pixel(intr, peg["xyz_m"])
        px, py = int(round(px)), int(round(py))
        color  = peg["color_bgr"]
        cv2.circle(frame, (px, py), 10, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py),  3, color, -1, cv2.LINE_AA)
        cv2.putText(frame, str(peg["id"]), (px + 12, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_calibrate_hud(frame):
    n = len(state["calib_pts"])
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 26), (40, 20, 20), -1)
    cv2.putText(frame,
                f"CALIBRATE  |  {n}/3 ref points  |  "
                "click 3 same-height pts  |  u=undo  s=skip  q=quit",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    draw_calib_markers(frame)


def draw_register_hud(frame):
    n = len(state["pegs"])
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 26), (20, 30, 20), -1)
    cv2.putText(frame,
                f"REGISTER  |  {n} peg(s)  |  "
                "left-click=add  u=undo  r=reset  Enter=review map  q=quit",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    draw_peg_markers(frame)


# ── 3-D map ───────────────────────────────────────────────────────────────────
def show_3d_map_blocking(pegs):
    """
    Blocking matplotlib 3-D review.  Also prints a height-difference table.
    Returns 'save', 'register', or 'quit'.
    """
    result = {"action": "save"}

    # ── print height-difference table ─────────────────────────────────────────
    print("\n── Peg height differences (positive = second peg is higher) ──────")
    for i, pa in enumerate(pegs):
        for pb in pegs[i + 1:]:
            dh = height_diff(pa["xyz_m"], pb["xyz_m"])
            flag = "  ← outlier?" if abs(dh) > 0.005 else ""
            print(f"  Peg {pa['id']} → Peg {pb['id']}:  Δh = {dh*1000:+.1f} mm{flag}")
    print("────────────────────────────────────────────────────────────────────\n")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    for peg in pegs:
        X, Y, Z = peg["xyz_m"]
        r, g, b = [c / 255 for c in peg["color_bgr"][::-1]]   # BGR → RGB
        ax.scatter([X], [Y], [Z], c=[[r, g, b]], s=120, depthshade=False)
        ax.text(X, Y, Z, f"  {peg['id']}  ({peg['n_samples']}smp)", fontsize=8)

    # Reference plane at mean peg height along calibrated up axis
    up   = state["up_axis"]
    pts  = np.array([p["xyz_m"] for p in pegs])
    h_mean  = float(np.dot(pts, up).mean())
    h_pt    = h_mean * up
    ref     = np.array([1., 0., 0.]) if abs(up[0]) < 0.9 else np.array([0., 1., 0.])
    t1      = np.cross(up, ref);  t1 /= np.linalg.norm(t1)
    t2      = np.cross(up, t1);   t2 /= np.linalg.norm(t2)
    span    = 0.07
    corners = [h_pt + a * t1 + b * t2 for a in (-span, span) for b in (-span, span)]
    gx = np.array([[corners[0][0], corners[1][0]], [corners[2][0], corners[3][0]]])
    gy = np.array([[corners[0][1], corners[1][1]], [corners[2][1], corners[3][1]]])
    gz = np.array([[corners[0][2], corners[1][2]], [corners[2][2], corners[3][2]]])
    ax.plot_surface(gx, gy, gz, alpha=0.18, color="cyan")

    cal_note = ("calibrated tilt" if not np.allclose(up, [0., -1., 0.])
                else "default −Y axis")
    ax.set_xlabel("X (right, m)")
    ax.set_ylabel("Y (down, m)")
    ax.set_zlabel("Z (depth, m)")
    ax.set_title(
        f"Registered peg positions  ({len(pegs)} pegs)  [{cal_note}]\n"
        "Cyan plane = mean peg height\n"
        "Enter / S = save & exit   |   R = back to register   |   Q = quit",
        fontsize=9,
    )
    ax.invert_yaxis()

    def on_key(event):
        k = (event.key or "").lower()
        if k in ("enter", "s"):
            result["action"] = "save"
            plt.close(fig)
        elif k == "r":
            result["action"] = "register"
            plt.close(fig)
        elif k == "q":
            result["action"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show(block=True)
    return result["action"]


# ── save ──────────────────────────────────────────────────────────────────────
def save_pegs():
    intr = state["intrinsics"]
    out  = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "intrinsics": {
            "fx":     intr.fx,
            "fy":     intr.fy,
            "ppx":    intr.ppx,
            "ppy":    intr.ppy,
            "width":  intr.width,
            "height": intr.height,
            "model":  str(intr.model),
            "coeffs": list(intr.coeffs),
        },
        "up_axis": state["up_axis"].tolist(),
        "pegs": [
            {
                "id":       p["id"],
                "u":        p["u"],
                "v":        p["v"],
                "xyz_m":    p["xyz_m"],
                "n_samples": p["n_samples"],
            }
            for p in state["pegs"]
        ],
    }
    PEGS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved {len(state['pegs'])} peg(s) → {PEGS_PATH.resolve()}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)

    print("Starting RealSense pipeline…")
    profile = pipeline.start(cfg)
    device  = profile.get_device()

    depth_sensor = device.first_depth_sensor()
    color_sensor = find_color_sensor(device)

    depth_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

    print(f"  Loading {SETTINGS_PATH.name}…")
    load_settings(depth_sensor, color_sensor)

    align      = rs.align(rs.stream.color)
    intrinsics = (profile.get_stream(rs.stream.color)
                         .as_video_stream_profile()
                         .get_intrinsics())

    state["pipeline"]   = pipeline
    state["align"]      = align
    state["intrinsics"] = intrinsics

    print(f"  Warming up ({WARMUP_FRAMES} frames)…", end="", flush=True)
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames(timeout_ms=5000)
    print(" ready.\n")

    print("--- PHASE 1: CALIBRATE ---")
    print("Click 3 points at the SAME real height (e.g. task board corners).")
    print("'s' to skip and use default camera axis.  'u' to undo.\n")

    cv2.namedWindow("register", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("register", STREAM_WIDTH, STREAM_HEIGHT)
    cv2.setMouseCallback("register", on_mouse)

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            state["depth_frame"] = depth_frame   # used by calibration clicks
            frame = np.asanyarray(color_frame.get_data())

            # ── CALIBRATE phase ───────────────────────────────────────────────
            if state["phase"] == PHASE_CALIBRATE:
                draw_calibrate_hud(frame)
                cv2.imshow("register", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("u") and state["calib_pts"]:
                    state["calib_pts"].pop()
                    print("  Removed last calibration point.")
                elif key == ord("s"):
                    print("  Calibration skipped — using default −Y axis.\n")
                    state["phase"] = PHASE_REGISTER
                    print("--- PHASE 2: REGISTER ---")
                    print("Left-click each peg top.  Enter = review 3-D map.\n")

            # ── REGISTER phase ────────────────────────────────────────────────
            else:
                # Process a pending click (30-frame capture)
                if state["pending_click"] is not None:
                    u, v = state["pending_click"]
                    state["pending_click"] = None
                    peg_id = len(state["pegs"])
                    print(f"  Registering peg {peg_id} at ({u}, {v})…", end="", flush=True)
                    xyz, n = capture_peg_xyz(u, v)
                    if xyz is not None:
                        X, Y, Z = xyz
                        print(f"  X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m  "
                              f"[{n}/{CAPTURE_FRAMES * 49} samples]")
                        color_bgr = PEG_PALETTE[peg_id % len(PEG_PALETTE)]
                        state["pegs"].append({
                            "id":        peg_id,
                            "u":         u,
                            "v":         v,
                            "xyz_m":     xyz,
                            "n_samples": n,
                            "color_bgr": color_bgr,
                        })
                        if len(state["pegs"]) > 1:
                            print("  Height vs previous pegs:")
                            for prev in state["pegs"][:-1]:
                                dh = height_diff(prev["xyz_m"], xyz)
                                print(f"    peg {peg_id} − peg {prev['id']}: "
                                      f"Δh = {dh*1000:+.1f} mm")
                    else:
                        print()

                draw_register_hud(frame)
                cv2.imshow("register", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Quit without saving.")
                    break
                elif key == ord("u"):
                    if state["pegs"]:
                        removed = state["pegs"].pop()
                        print(f"  Undid peg {removed['id']}.")
                    else:
                        print("  Nothing to undo.")
                elif key == ord("r"):
                    state["pegs"].clear()
                    print("  All pegs cleared.")
                elif key in (13, 10):   # Enter
                    if not state["pegs"]:
                        print("  [!] Register at least one peg first.")
                        continue
                    print(f"\n--- PHASE 3: MAP ({len(state['pegs'])} peg(s)) ---")
                    print("  Click the matplotlib window, then use keyboard.\n")
                    action = show_3d_map_blocking(state["pegs"])

                    if action == "save":
                        save_pegs()
                        break
                    elif action == "register":
                        print("\n--- PHASE 2: REGISTER (back) ---")
                        print("Pegs kept. Add/undo with click / u. Enter when done.\n")
                    else:
                        print("Quit without saving.")
                        break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
