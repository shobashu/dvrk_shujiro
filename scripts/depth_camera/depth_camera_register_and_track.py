"""
Three-phase workflow: level calibration → peg registration → cylinder tracking.

Phase 1a — CALIBRATE
  Click 3 points you know are at the same real height (e.g. three corners of
  the task board, or any three same-level landmarks).  The script fits a plane
  through those points and uses its normal as the true "up" axis, correcting
  for any camera tilt or roll.
  Skip with --skip-cal to use raw camera Y as the height axis.

Phase 1b — REGISTER
  Click the top of each peg to record its 3D position.

Phase 2 — MAP  (optional, skip with --skip-map)
  Blocking 3-D matplotlib window to review peg positions before tracking.

Phase 3 — TRACK
  YOLO detects the cylinder every frame; live 3D position is shown.

Controls  (CALIBRATE phase):
  Left-click   record a same-height reference point (need exactly 3)
  u            undo the last calibration point
  s            skip calibration, use default Y axis
  q            quit

Controls  (REGISTER phase):
  Left-click   record a peg top
  u            undo the last peg
  Enter        open 3-D map (or go to tracking if --skip-map)
  q            quit

Controls  (MAP window — click the matplotlib window first):
  Enter / T    start tracking
  R            back to registration
  Q            quit

Controls  (TRACK phase):
  q            quit

Coordinate frame  (origin = camera optical centre):
  X → right   Y → down   Z → forward (metres)

Usage:
  python register_and_track.py
  python register_and_track.py --skip-cal           # skip tilt calibration
  python register_and_track.py --skip-map           # skip 3-D map review
  python register_and_track.py --weights best.pt --conf 0.5 --imgsz 320
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ultralytics import YOLO

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS     = "models/best_v2.pt"
DEPTH_SAMPLE_RADIUS = 5
CYLINDER_CLASS_ID   = 0
SMOOTH_ALPHA        = 0.35   # EMA weight for new measurement (lower = smoother)

CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}
CLASS_COLORS = {
    0: (255, 100,   0),
    1: (180, 180, 180),
    2: (0,   200, 255),
    3: (255, 255,   0),
}

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
PHASE_TRACK     = "track"

# ── shared state ──────────────────────────────────────────────────────────────
state = {
    "phase":      PHASE_CALIBRATE,
    "calib_pts":  [],          # up to 2 same-height reference points
    "up_axis":    np.array([0., -1., 0.]),  # default: camera -Y = up (level camera)
    "pegs":       [],
    "depth_frame": None,
    "intrinsics":  None,
}


# ── cylinder EMA smoother ─────────────────────────────────────────────────────
class CylinderTracker:
    """Exponential moving average over the cylinder's 3D position."""
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        self.xyz   = None

    def update(self, xyz):
        if xyz is None:
            return self.xyz
        m = np.array(xyz, dtype=float)
        self.xyz = m if self.xyz is None else self.alpha * m + (1 - self.alpha) * self.xyz
        return tuple(self.xyz)

    def reset(self):
        self.xyz = None


# ── depth helper ──────────────────────────────────────────────────────────────
def get_median_depth(depth_frame, u, v, radius=DEPTH_SAMPLE_RADIUS):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    u0, u1 = max(0, u - radius), min(w - 1, u + radius)
    v0, v1 = max(0, v - radius), min(h - 1, v + radius)
    depths = [
        depth_frame.get_distance(col, row)
        for row in range(v0, v1 + 1)
        for col in range(u0, u1 + 1)
        if depth_frame.get_distance(col, row) > 0
    ]
    return float(np.median(depths)) if depths else 0.0


def get_xyz(x, y):
    """Deproject pixel (x,y) to 3D using current depth frame. Returns xyz or None."""
    depth_frame = state["depth_frame"]
    intrinsics  = state["intrinsics"]
    if depth_frame is None or intrinsics is None:
        return None
    depth_m = get_median_depth(depth_frame, x, y)
    if depth_m <= 0:
        raw = depth_frame.get_distance(x, y)
        print(f"  [!] No valid depth at ({x},{y}) — raw={raw:.3f} m  (too close / reflective?)")
        return None
    return tuple(rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth_m))


def get_bbox_depth(depth_frame, x1, y1, x2, y2, inner_pct=0.5):
    """Median depth sampled from the inner inner_pct of the bounding box.

    Sampling from the bbox interior avoids edge pixels that mix foreground
    and background depth, giving a cleaner reading on the cylinder body.
    The weighted centroid pixel is returned alongside the depth for accurate
    3D deprojection.
    """
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = max(2, int((x2 - x1) * inner_pct / 2))
    hh = max(2, int((y2 - y1) * inner_pct / 2))
    h_img, w_img = depth_frame.get_height(), depth_frame.get_width()
    u0 = max(0, int(cx) - hw);  u1 = min(w_img - 1, int(cx) + hw)
    v0 = max(0, int(cy) - hh);  v1 = min(h_img - 1, int(cy) + hh)
    pts = []
    for row in range(v0, v1 + 1):
        for col in range(u0, u1 + 1):
            d = depth_frame.get_distance(col, row)
            if d > 0:
                pts.append((d, col, row))
    if not pts:
        return 0.0, int(cx), int(cy)
    depths = [d for d, _, _ in pts]
    med    = float(np.median(depths))
    # Use the pixel whose depth is closest to the median for deprojection
    best   = min(pts, key=lambda t: abs(t[0] - med))
    return med, best[1], best[2]


# ── height helper ─────────────────────────────────────────────────────────────
def height_diff(xyz_from, xyz_to):
    """Signed height difference using the calibrated up axis.
    Positive = xyz_to is higher than xyz_from in real world."""
    d = np.array(xyz_to) - np.array(xyz_from)
    return float(np.dot(d, state["up_axis"]))


def compute_up_axis(pt_a, pt_b, pt_c):
    """Derive world-up in camera space from three co-planar same-height 3D points.
    Fits a plane and returns its normal (no roll assumption needed)."""
    v1 = np.array(pt_b) - np.array(pt_a)
    v2 = np.array(pt_c) - np.array(pt_a)
    up = np.cross(v1, v2)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        print("  [!] Calibration points collinear — keeping default axis.")
        return state["up_axis"].copy()
    up /= norm
    # Ensure it points "up" (negative camera-Y when roughly level)
    if up[1] > 0:
        up = -up
    return up


# ── mouse callback ────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state["phase"] == PHASE_CALIBRATE:
        if len(state["calib_pts"]) >= 3:
            return
        xyz = get_xyz(x, y)
        if xyz is None:
            return
        state["calib_pts"].append({"pixel": (x, y), "xyz": xyz})
        n = len(state["calib_pts"])
        X, Y, Z = xyz
        print(f"  Cal point {n}/3: X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m")

        if n == 3:
            up = compute_up_axis(state["calib_pts"][0]["xyz"],
                                 state["calib_pts"][1]["xyz"],
                                 state["calib_pts"][2]["xyz"])
            state["up_axis"] = up
            print(f"  Calibrated up axis (camera frame): "
                  f"[{up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f}]")
            print("  Calibration done — advancing to peg registration.\n")
            print("--- PHASE 1b: REGISTER ---")
            print("Left-click peg tops. 'u' = undo. Enter = open map / start tracking.\n")
            state["phase"] = PHASE_REGISTER

    elif state["phase"] == PHASE_REGISTER:
        xyz = get_xyz(x, y)
        if xyz is None:
            return
        n     = len(state["pegs"]) + 1
        color = PEG_PALETTE[(n - 1) % len(PEG_PALETTE)]
        label = f"Peg {n}"
        state["pegs"].append({"pixel": (x, y), "xyz": xyz, "color": color, "label": label})

        X, Y, Z = xyz
        print(f"  Registered {label}  X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m")

        if len(state["pegs"]) > 1:
            print("  Height differences vs previous pegs:")
            for prev in state["pegs"][:-1]:
                dh = height_diff(prev["xyz"], xyz)
                print(f"    {label} vs {prev['label']}:  Δh={dh:+.4f} m  ({dh*1000:+.1f} mm)")


# ── overlay helpers ───────────────────────────────────────────────────────────
def draw_dot_marker(frame, pixel, color, label):
    u, v = pixel
    cv2.drawMarker(frame, (u, v), color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.circle(frame, (u, v), 7, color, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tx = max(0, min(u - tw // 2, frame.shape[1] - tw - 4))
    ty = v - 16
    if ty < th + 4:
        ty = v + 26
    cv2.rectangle(frame, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_peg_markers(frame):
    for peg in state["pegs"]:
        draw_dot_marker(frame, peg["pixel"], peg["color"], peg["label"])


def draw_calib_markers(frame):
    for i, pt in enumerate(state["calib_pts"]):
        draw_dot_marker(frame, pt["pixel"], (0, 255, 200), f"Ref {i+1}")


def draw_axes_legend(frame):
    h, w = frame.shape[:2]
    ox, oy = 50, h - 65
    L = 35

    # Background box — taller when calibration info is shown
    cal_active = not np.allclose(state["up_axis"], [0., -1., 0.])
    box_h = 18 if not cal_active else 30
    cv2.rectangle(frame, (ox - 15, oy - L - 10), (ox + L + 55, oy + L + box_h),
                  (20, 20, 20), -1)
    cv2.rectangle(frame, (ox - 15, oy - L - 10), (ox + L + 55, oy + L + box_h),
                  (80, 80, 80), 1)

    # X → right (red)
    cv2.arrowedLine(frame, (ox, oy), (ox + L, oy), (0, 0, 220), 2,
                    tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.putText(frame, "X", (ox + L + 4, oy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

    # Y → down (green)
    cv2.arrowedLine(frame, (ox, oy), (ox, oy + L), (0, 200, 0), 2,
                    tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.putText(frame, "Y", (ox - 14, oy + L + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

    # Z → forward (blue dot)
    cv2.circle(frame, (ox, oy), 7, (220, 100, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (ox, oy), 3, (220, 100, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, "Z", (ox - 14, oy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 100, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "origin = camera centre", (ox - 13, oy + L + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)

    if cal_active:
        u = state["up_axis"]
        cv2.putText(frame,
                    f"up=[{u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f}]",
                    (ox - 13, oy + L + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 220, 180), 1, cv2.LINE_AA)


def draw_calibrate_hud(frame):
    n = len(state["calib_pts"])
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 30), (40, 20, 20), -1)
    cv2.putText(frame,
                f"LEVEL CAL  |  {n}/3 ref points  |  "
                "left-click 3 same-height pts  |  u = undo  |  s = skip  |  q = quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    draw_axes_legend(frame)
    draw_calib_markers(frame)


def draw_register_hud(frame):
    n = len(state["pegs"])
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 30), (30, 30, 30), -1)
    cv2.putText(frame,
                f"REGISTER  |  {n} peg(s)  |  "
                "left-click = add  |  u = undo  |  Enter = map/track  |  q = quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    draw_axes_legend(frame)
    draw_peg_markers(frame)


def draw_track_hud(frame, detections):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 40), -1)
    cv2.putText(frame, "TRACKING  |  q = quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    for det in detections:
        cls_id = det["cls_id"]
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        color  = CLASS_COLORS.get(cls_id, (0, 255, 0))
        name   = CLASS_NAMES.get(cls_id, str(cls_id))
        xyz    = det.get("xyz")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        header = f"{name} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, header, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if xyz:
            X, Y, Z = xyz
            pos_str = f"X:{X:+.3f} Y:{Y:+.3f} Z:{Z:.3f}m"
        else:
            pos_str = "NO DEPTH"
        cv2.putText(frame, pos_str, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    draw_peg_markers(frame)


# ── detection helper ──────────────────────────────────────────────────────────
def run_detections(model, color_img, depth_frame, intrinsics, conf, imgsz, classes,
                   cylinder_only=True, tracker=None):
    """Run YOLO and return detections.

    When cylinder_only=True, only the highest-confidence cylinder box is kept.
    Depth is sampled from the inner 50 % of the bounding box for accuracy.
    If tracker is provided, the cylinder XYZ is EMA-smoothed.
    """
    results = model.predict(color_img, conf=conf, imgsz=imgsz,
                            classes=classes, verbose=False)[0]
    detections = []
    if results.boxes is None:
        return detections

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2

        if cylinder_only and cls_id != CYLINDER_CLASS_ID:
            continue

        xyz = None
        if cls_id == CYLINDER_CLASS_ID:
            depth_m, px, py = get_bbox_depth(depth_frame, x1, y1, x2, y2, inner_pct=0.5)
            if depth_m > 0:
                raw_xyz = tuple(rs.rs2_deproject_pixel_to_point(
                    intrinsics, [float(px), float(py)], depth_m))
                xyz = tracker.update(raw_xyz) if tracker else raw_xyz
        else:
            depth_m = get_median_depth(depth_frame, cx, cy)
            if depth_m > 0:
                xyz = tuple(rs.rs2_deproject_pixel_to_point(
                    intrinsics, [float(cx), float(cy)], depth_m))

        detections.append({
            "cls_id": cls_id, "conf": confidence,
            "bbox": (x1, y1, x2, y2), "center": (cx, cy), "xyz": xyz,
        })

    if cylinder_only and len(detections) > 1:
        # Keep only the highest-confidence cylinder to avoid ambiguity
        detections = [max(detections, key=lambda d: d["conf"])]

    return detections


# ── 3-D map ───────────────────────────────────────────────────────────────────
def show_3d_map_blocking(pegs):
    """Blocking 3-D map in main thread. Returns 'track', 'register', or 'quit'."""
    result = {"action": "track"}

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    xs     = [p["xyz"][0] for p in pegs]
    ys     = [p["xyz"][1] for p in pegs]
    zs     = [p["xyz"][2] for p in pegs]
    colors = [(r/255, g/255, b/255) for p in pegs for r, g, b in [p["color"]]]

    ax.scatter(xs, ys, zs, c=colors, s=100, depthshade=False)
    for p in pegs:
        ax.text(p["xyz"][0], p["xyz"][1], p["xyz"][2], f"  {p['label']}", fontsize=8)

    # Reference plane perpendicular to the calibrated up axis at the mean peg height
    up  = state["up_axis"]
    pts = np.array([p["xyz"] for p in pegs])
    mean_h  = float(np.dot(pts, up).mean())   # mean height along up axis
    mean_pt = mean_h * up                     # point on the plane (closest to origin)

    # Build two tangent vectors in the plane
    ref = np.array([1., 0., 0.]) if abs(up[0]) < 0.9 else np.array([0., 1., 0.])
    t1  = np.cross(up, ref);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(up, t1);   t2 /= np.linalg.norm(t2)
    span = 0.06
    corners = [mean_pt + a*t1 + b*t2
               for a in (-span, span) for b in (-span, span)]
    px = np.array([[corners[0][0], corners[1][0]],
                   [corners[2][0], corners[3][0]]])
    py = np.array([[corners[0][1], corners[1][1]],
                   [corners[2][1], corners[3][1]]])
    pz = np.array([[corners[0][2], corners[1][2]],
                   [corners[2][2], corners[3][2]]])
    ax.plot_surface(px, py, pz, alpha=0.20, color="cyan")

    ax.set_xlabel("X  (right, m)")
    ax.set_ylabel("Y  (down, m)")
    ax.set_zlabel("Z  (depth, m)")
    cal_note = ("calibrated tilt correction"
                if not np.allclose(up, [0., -1., 0.]) else "default Y axis")
    ax.set_title(
        f"Registered peg positions  ({len(pegs)} pegs)\n"
        f"Cyan plane = mean height level  [{cal_note}]\n"
        "Enter / T = start tracking   |   R = re-register   |   Q = quit",
        fontsize=9,
    )
    ax.invert_yaxis()

    def on_key(event):
        k = (event.key or "").lower()
        if k in ("enter", "t"):
            result["action"] = "track"
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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default=DEFAULT_WEIGHTS)
    p.add_argument("--conf",     type=float, default=0.6)
    p.add_argument("--imgsz",    type=int,   default=320)
    p.add_argument("--classes",  type=int,   nargs="+", default=[CYLINDER_CLASS_ID],
                   help="Class IDs to detect (default: 0=cylinder only). "
                        "0=cylinder 1=peg_inactive 2=peg_lit_blue 3=peg_lit_white")
    p.add_argument("--all-classes", action="store_true",
                   help="Detect all classes (overrides --classes).")
    p.add_argument("--smooth", type=float, default=SMOOTH_ALPHA, metavar="ALPHA",
                   help=f"EMA smoothing for cylinder XYZ (0=max smooth, 1=raw). "
                        f"Default: {SMOOTH_ALPHA}")
    p.add_argument("--skip-cal", action="store_true",
                   help="Skip level calibration; use raw camera Y as height axis.")
    p.add_argument("--skip-map", action="store_true",
                   help="Skip Phase 2 (3-D map review).")
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    args = p.parse_args()

    if args.all_classes:
        args.classes = None

    cylinder_only = (args.classes is not None and
                     args.classes == [CYLINDER_CLASS_ID])
    tracker = CylinderTracker(alpha=args.smooth)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}")
        return

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    cls_label = "cylinder only" if cylinder_only else (
        str(args.classes) if args.classes else "all")
    print(f"Confidence: {args.conf}  |  Inference size: {args.imgsz}  |  "
          f"Classes: {cls_label}  |  EMA alpha: {args.smooth}")

    pipeline = rs.pipeline()
    config   = rs.config()
    sw, sh = args.width, args.height
    config.enable_stream(rs.stream.color, sw, sh, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, sw, sh, rs.format.z16,  30)

    print("Starting RealSense streams…")
    profile = pipeline.start(config)
    state["intrinsics"] = (
        profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics()
    )

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", sw, sh)
    cv2.resizeWindow("Depth",  sw, sh)
    cv2.setMouseCallback("Camera", on_mouse)

    if args.skip_cal:
        state["phase"] = PHASE_REGISTER
        print("\n--- PHASE 1: REGISTER (calibration skipped — using camera Y axis) ---")
        print("Left-click peg tops. 'u' = undo. Enter when done.\n")
    else:
        print("\n--- PHASE 1a: LEVEL CALIBRATION ---")
        print("Left-click 3 points you know are at the SAME real height")
        print("(e.g. three corners of the task board at the same level).")
        print("'u' = undo last point  |  's' = skip calibration\n")

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            state["depth_frame"] = depth_frame

            color_img = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            frame     = color_img.copy()

            # ── CALIBRATE ─────────────────────────────────────────────────────
            if state["phase"] == PHASE_CALIBRATE:
                draw_calibrate_hud(frame)
                cv2.imshow("Camera", frame)
                cv2.imshow("Depth",  depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("u") and state["calib_pts"]:
                    state["calib_pts"].pop()
                    print("  Removed last calibration point.")
                elif key == ord("s"):
                    print("  Calibration skipped — using default camera Y axis.\n")
                    print("--- PHASE 1b: REGISTER ---")
                    print("Left-click peg tops. 'u' = undo. Enter when done.\n")
                    state["phase"] = PHASE_REGISTER

            # ── REGISTER ──────────────────────────────────────────────────────
            elif state["phase"] == PHASE_REGISTER:
                draw_register_hud(frame)
                cv2.imshow("Camera", frame)
                cv2.imshow("Depth",  depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("u") and state["pegs"]:
                    removed = state["pegs"].pop()
                    print(f"  Removed {removed['label']}")
                elif key in (13, 10):  # Enter
                    if not state["pegs"]:
                        print("  [!] No pegs registered yet — click at least one.")
                    elif args.skip_map:
                        tracker.reset()
                        state["phase"] = PHASE_TRACK
                        print(f"\n--- PHASE 3: TRACK ---")
                        print(f"Tracking with {len(state['pegs'])} peg reference(s). q = quit.\n")
                    else:
                        print(f"\n--- PHASE 2: MAP ({len(state['pegs'])} peg(s)) ---")
                        print("  Click on the matplotlib window, then use keyboard:\n"
                              "  Enter/T = track  |  R = re-register  |  Q = quit\n")
                        action = show_3d_map_blocking(state["pegs"])
                        if action == "register":
                            print("\n--- PHASE 1b: REGISTER (back) ---")
                            print("Pegs kept. Add/undo with left-click / u. Enter when done.\n")
                        elif action == "track":
                            tracker.reset()
                            state["phase"] = PHASE_TRACK
                            print(f"\n--- PHASE 3: TRACK ---")
                            print(f"Tracking with {len(state['pegs'])} peg reference(s). q = quit.\n")
                        else:
                            break

            # ── TRACK ─────────────────────────────────────────────────────────
            else:
                detections = run_detections(
                    model, color_img, depth_frame, state["intrinsics"],
                    args.conf, args.imgsz, args.classes,
                    cylinder_only=cylinder_only, tracker=tracker,
                )
                draw_track_hud(frame, detections)
                cv2.imshow("Camera", frame)
                cv2.imshow("Depth",  depth_vis)

                for det in detections:
                    name = CLASS_NAMES.get(det["cls_id"], str(det["cls_id"]))
                    xyz  = det["xyz"]
                    if xyz:
                        X, Y, Z = xyz
                        print(f"  [{name}] conf={det['conf']:.2f}  "
                              f"X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
