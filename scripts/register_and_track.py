"""
Three-phase workflow for peg registration + 3-D evaluation + cylinder tracking.

Phase 1 — REGISTER
  Click on the top of each side peg to lock its 3D position.
  The positions are saved as static reference markers.

Phase 2 — MAP
  A 3-D matplotlib window shows the registered peg positions so you can
  evaluate the registration before starting tracking.

Phase 3 — TRACK
  YOLO runs on every frame.  The cylinder's live 3D position is shown
  together with the registered peg markers.

Controls  (REGISTER phase):
  Left-click   record a peg top
  u            undo the last peg
  Enter        finish registration → open 3-D map
  q            quit

Controls  (MAP phase):
  Enter        close map → start tracking
  q            quit

Controls  (TRACK phase):
  q            quit

Coordinate frame  (origin = camera optical centre):
  X → right   Y → down   Z → forward (metres)

Usage:
  python register_and_track.py
  python register_and_track.py --weights best.pt --conf 0.5 --imgsz 320
"""

import argparse
import threading
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
DEFAULT_WEIGHTS = "models/best_v2.pt"
DEPTH_SAMPLE_RADIUS = 5

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

# Distinct colours cycled across registered pegs
PEG_PALETTE = [
    (0,   255, 128),   # green
    (255, 128,   0),   # orange
    (0,   180, 255),   # sky blue
    (220,   0, 255),   # magenta
    (0,   255, 255),   # cyan
    (255, 220,   0),   # gold
    (180,   0, 255),   # purple
    (0,   220, 180),   # teal
]

PHASE_REGISTER = "register"
PHASE_MAP      = "map"
PHASE_TRACK    = "track"


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


# ── shared state ──────────────────────────────────────────────────────────────
state = {
    "phase":       PHASE_REGISTER,
    "pegs":        [],     # [{"pixel":(u,v), "xyz":(X,Y,Z), "color":(r,g,b), "label":"Peg 1"}, …]
    "depth_frame": None,
    "intrinsics":  None,
}


# ── mouse callback (REGISTER phase only) ─────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state["phase"] != PHASE_REGISTER:
        return

    depth_frame = state["depth_frame"]
    intrinsics  = state["intrinsics"]
    if depth_frame is None or intrinsics is None:
        return

    depth_m = get_median_depth(depth_frame, x, y)
    if depth_m <= 0:
        raw = depth_frame.get_distance(x, y)
        print(f"  [!] No valid depth at ({x},{y}) — raw={raw:.3f} m  (too close / reflective?)")
        return

    xyz   = tuple(rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth_m))
    n     = len(state["pegs"]) + 1
    color = PEG_PALETTE[(n - 1) % len(PEG_PALETTE)]
    label = f"Peg {n}"

    state["pegs"].append({"pixel": (x, y), "xyz": xyz, "color": color, "label": label})

    X, Y, Z = xyz
    print(f"  Registered {label}  X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m")

    if len(state["pegs"]) > 1:
        print("  Height differences (Y) vs previous pegs:")
        for prev in state["pegs"][:-1]:
            dy = Y - prev["xyz"][1]
            print(f"    {label} vs {prev['label']}:  ΔY={dy:+.4f} m  ({dy*1000:+.1f} mm)")


# ── overlay helpers ───────────────────────────────────────────────────────────
def draw_peg_markers(frame):
    """Draw a crosshair + label for every registered peg."""
    for peg in state["pegs"]:
        u, v   = peg["pixel"]
        color  = peg["color"]
        label  = peg["label"]

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


def draw_register_hud(frame):
    n = len(state["pegs"])
    w = frame.shape[1]

    cv2.rectangle(frame, (0, 0), (w, 30), (30, 30, 30), -1)
    cv2.putText(frame,
                f"REGISTER  |  {n} peg(s) recorded  |  "
                "left-click = add  |  u = undo  |  Enter = start tracking  |  q = quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    draw_peg_markers(frame)


def draw_map_hud(frame):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 30), (20, 40, 20), -1)
    cv2.putText(frame,
                f"MAP  |  {len(state['pegs'])} peg(s) registered  |  "
                "Enter = start tracking  |  q = quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    draw_peg_markers(frame)


def draw_track_hud(frame, detections):
    w = frame.shape[1]

    cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 40), -1)
    cv2.putText(frame, "TRACKING  |  q = quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    # YOLO detection overlays
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

    # Peg reference markers (static)
    draw_peg_markers(frame)


# ── detection helper (TRACK phase) ───────────────────────────────────────────
def run_detections(model, color_img, depth_frame, intrinsics, conf, imgsz, classes):
    results = model.predict(
        color_img,
        conf=conf,
        imgsz=imgsz,
        classes=classes,
        verbose=False,
    )[0]

    detections = []
    if results.boxes is None:
        return detections

    for box in results.boxes:
        cls_id      = int(box.cls[0])
        confidence  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy      = (x1 + x2) // 2, (y1 + y2) // 2
        depth_m     = get_median_depth(depth_frame, cx, cy)

        xyz = None
        if depth_m > 0:
            xyz = tuple(rs.rs2_deproject_pixel_to_point(
                intrinsics, [float(cx), float(cy)], depth_m
            ))

        detections.append({
            "cls_id": cls_id,
            "conf":   confidence,
            "bbox":   (x1, y1, x2, y2),
            "center": (cx, cy),
            "xyz":    xyz,
        })

    return detections


# ── 3-D map ───────────────────────────────────────────────────────────────────
def _plot_3d(pegs):
    """Runs in a background thread so OpenCV and matplotlib event loops are independent."""
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    xs     = [p["xyz"][0] for p in pegs]
    ys     = [p["xyz"][1] for p in pegs]
    zs     = [p["xyz"][2] for p in pegs]
    colors = [(r/255, g/255, b/255) for p in pegs for r, g, b in [p["color"]]]

    ax.scatter(xs, ys, zs, c=colors, s=100, depthshade=False)
    for p in pegs:
        ax.text(p["xyz"][0], p["xyz"][1], p["xyz"][2], f"  {p['label']}", fontsize=8)

    # cyan plane at mean Y to show the ideal level
    mean_y = float(np.mean(ys))
    xx, zz = np.meshgrid(
        [min(xs) - 0.02, max(xs) + 0.02],
        [min(zs) - 0.02, max(zs) + 0.02],
    )
    ax.plot_surface(xx, np.full_like(xx, mean_y), zz, alpha=0.15, color="cyan")

    ax.set_xlabel("X  (right, m)")
    ax.set_ylabel("Y  (down, m)")
    ax.set_zlabel("Z  (depth, m)")
    ax.set_title("Registered peg positions\n(cyan plane = mean height)\n"
                 "drag to rotate  |  scroll to zoom")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show(block=True)   # blocks only this thread, not OpenCV


def show_3d_map(pegs):
    if not pegs:
        return
    t = threading.Thread(target=_plot_3d, args=(list(pegs),), daemon=True)
    t.start()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf",    type=float, default=0.6)
    p.add_argument("--imgsz",   type=int,   default=320)
    p.add_argument("--classes", type=int,   nargs="+", default=None,
                   help="Limit detection to these class IDs (default: all). "
                        "0=cylinder 1=peg_inactive 2=peg_lit_blue 3=peg_lit_white")
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}")
        return

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    print(f"Confidence: {args.conf}  |  Inference size: {args.imgsz}  |  "
          f"Classes: {args.classes if args.classes else 'all'}")

    # RealSense
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)

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
    cv2.resizeWindow("Camera", 640, 480)
    cv2.resizeWindow("Depth",  640, 480)
    cv2.setMouseCallback("Camera", on_mouse)

    print("\n--- PHASE 1: REGISTER ---")
    print("Left-click the top of each side peg to record its 3D position.")
    print("Press 'u' to undo the last peg.")
    print("Press Enter when done to start tracking.\n")

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

            # ── render ────────────────────────────────────────────────────────
            frame = color_img.copy()

            if state["phase"] == PHASE_REGISTER:
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
                    else:
                        state["phase"] = PHASE_MAP
                        show_3d_map(state["pegs"])
                        print(f"\n--- PHASE 2: MAP ---")
                        print(f"3-D map open. Press Enter to start tracking or q to quit.\n")

            elif state["phase"] == PHASE_MAP:
                draw_map_hud(frame)
                cv2.imshow("Camera", frame)
                cv2.imshow("Depth",  depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (13, 10):  # Enter
                    plt.close("all")
                    state["phase"] = PHASE_TRACK
                    print(f"\n--- PHASE 3: TRACK ---")
                    print(f"Tracking cylinder with {len(state['pegs'])} peg reference(s).")
                    print("Press 'q' to quit.\n")

            else:  # PHASE_TRACK
                detections = run_detections(
                    model, color_img, depth_frame, state["intrinsics"],
                    args.conf, args.imgsz, args.classes
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
