#!/usr/bin/env python3
"""
Live HSV tuner — subscribe to /camera_left/compressed and adjust
WHITE and BLUE thresholds interactively with trackbars.

Layout:
  [ Original ]  [ White mask ]  [ Blue mask ]  [ Combined ]

Keys:
  s  — save current values to orientation_steps/params.py
  q  — quit

Run:
    # Make sure the camera stream is running first:
    #   ./scripts/camera-stream-compressed-transport.sh
    source /opt/ros/jazzy/setup.bash
    python3 scripts/hsv_tuner_live.py

    # Optional: choose a different topic
    python3 scripts/hsv_tuner_live.py --topic /camera_right/compressed
"""

import argparse
import re
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

# ── ROS2 imports ──────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
except ImportError:
    print("ERROR: rclpy not found. Source your ROS2 setup first:")
    print("  source /opt/ros/jazzy/setup.bash")
    sys.exit(1)

_here = Path(__file__).resolve().parent
PARAMS_FILE = _here / "params.py" if (_here / "params.py").exists() else _here / "orientation_steps" / "params.py"
WINDOW = "HSV Tuner  |  s=save  q=quit"


# ── Trackbar helpers ──────────────────────────────────────────────────────────

def nothing(_):
    pass


def make_trackbars(win, white, blue):
    """Create all trackbars in `win` with initial values from params dicts."""
    cv2.createTrackbar("W  H min", win, white["H_MIN"], 179, nothing)
    cv2.createTrackbar("W  H max", win, white["H_MAX"], 179, nothing)
    cv2.createTrackbar("W  S min", win, white["S_MIN"], 255, nothing)
    cv2.createTrackbar("W  S max", win, white["S_MAX"], 255, nothing)
    cv2.createTrackbar("W  V min", win, white["V_MIN"], 255, nothing)
    cv2.createTrackbar("W  V max", win, white["V_MAX"], 255, nothing)
    cv2.createTrackbar("B  H min", win, blue["H_MIN"],  179, nothing)
    cv2.createTrackbar("B  H max", win, blue["H_MAX"],  179, nothing)
    cv2.createTrackbar("B  S min", win, blue["S_MIN"],  255, nothing)
    cv2.createTrackbar("B  S max", win, blue["S_MAX"],  255, nothing)
    cv2.createTrackbar("B  V min", win, blue["V_MIN"],  255, nothing)
    cv2.createTrackbar("B  V max", win, blue["V_MAX"],  255, nothing)


def read_trackbars(win):
    w = {k: cv2.getTrackbarPos(f"W  {k.replace('_', ' ')}", win)
         for k in ("H min", "H max", "S min", "S max", "V min", "V max")}
    b = {k: cv2.getTrackbarPos(f"B  {k.replace('_', ' ')}", win)
         for k in ("H min", "H max", "S min", "S max", "V min", "V max")}
    # Normalise key names
    wh = {k.replace(" ", "_").upper(): v for k, v in w.items()}
    bl = {k.replace(" ", "_").upper(): v for k, v in b.items()}
    return wh, bl


# ── params.py writer ──────────────────────────────────────────────────────────

def save_params(white, blue):
    text = PARAMS_FILE.read_text()

    replacements = {
        "WHITE_H_MIN": white["H_MIN"], "WHITE_H_MAX": white["H_MAX"],
        "WHITE_S_MIN": white["S_MIN"], "WHITE_S_MAX": white["S_MAX"],
        "WHITE_V_MIN": white["V_MIN"], "WHITE_V_MAX": white["V_MAX"],
        "BLUE_H_MIN":  blue["H_MIN"],  "BLUE_H_MAX":  blue["H_MAX"],
        "BLUE_S_MIN":  blue["S_MIN"],  "BLUE_S_MAX":  blue["S_MAX"],
        "BLUE_V_MIN":  blue["V_MIN"],  "BLUE_V_MAX":  blue["V_MAX"],
    }

    # Match pairs like: WHITE_H_MIN, WHITE_H_MAX =  0, 179
    def replace_pair(m):
        k1, k2 = m.group(1), m.group(2)
        v1 = replacements.get(k1, m.group(3))
        v2 = replacements.get(k2, m.group(4))
        return f"{k1}, {k2} = {v1:>4}, {v2:<4}"

    text = re.sub(
        r"(WHITE_[HSV]_MIN|BLUE_[HSV]_MIN),\s*(WHITE_[HSV]_MAX|BLUE_[HSV]_MAX)\s*=\s*(\d+),\s*(\d+)",
        replace_pair,
        text,
    )

    PARAMS_FILE.write_text(text)
    print(f"[SAVED] {PARAMS_FILE}")
    print(f"  WHITE  H {white['H_MIN']:>3}–{white['H_MAX']:<3}  "
          f"S {white['S_MIN']:>3}–{white['S_MAX']:<3}  "
          f"V {white['V_MIN']:>3}–{white['V_MAX']:<3}")
    print(f"  BLUE   H {blue['H_MIN']:>3}–{blue['H_MAX']:<3}  "
          f"S {blue['S_MIN']:>3}–{blue['S_MAX']:<3}  "
          f"V {blue['V_MIN']:>3}–{blue['V_MAX']:<3}")


# ── ROS2 subscriber node ──────────────────────────────────────────────────────

class FrameGrabber(Node):
    def __init__(self, topic):
        super().__init__("hsv_tuner_live")
        self.latest = None
        self._lock = threading.Lock()
        self.create_subscription(CompressedImage, topic, self._cb, 1)

    def _cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._lock:
                self.latest = frame

    def get_frame(self):
        with self._lock:
            return self.latest.copy() if self.latest is not None else None


# ── Image processing ──────────────────────────────────────────────────────────

def draw_info_bar(width, white, blue):
    """Black bar showing current HSV values for both colours."""
    bar = np.zeros((80, width, 3), dtype=np.uint8)
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    w_text = (f"WHITE  H {white['H_MIN']:>3}-{white['H_MAX']:<3}   "
              f"S {white['S_MIN']:>3}-{white['S_MAX']:<3}   "
              f"V {white['V_MIN']:>3}-{white['V_MAX']:<3}")
    b_text = (f"BLUE   H {blue['H_MIN']:>3}-{blue['H_MAX']:<3}   "
              f"S {blue['S_MIN']:>3}-{blue['S_MAX']:<3}   "
              f"V {blue['V_MIN']:>3}-{blue['V_MAX']:<3}")
    cv2.putText(bar, w_text, (12, 28), font, scale, (220, 220, 220), th)
    cv2.putText(bar, b_text, (12, 62), font, scale, (60, 160, 255), th)
    return bar


def label(img, text):
    cv2.putText(img, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return img


def apply_masks(frame, white, blue):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    w_lo = np.array([white["H_MIN"], white["S_MIN"], white["V_MIN"]])
    w_hi = np.array([white["H_MAX"], white["S_MAX"], white["V_MAX"]])
    b_lo = np.array([blue["H_MIN"],  blue["S_MIN"],  blue["V_MIN"]])
    b_hi = np.array([blue["H_MAX"],  blue["S_MAX"],  blue["V_MAX"]])

    m_white = cv2.inRange(hsv, w_lo, w_hi)
    m_blue  = cv2.inRange(hsv, b_lo, b_hi)
    m_both  = cv2.bitwise_or(m_white, m_blue)

    def overlay(mask, color_bgr):
        colored = np.zeros_like(frame)
        colored[mask > 0] = color_bgr
        return cv2.addWeighted(frame, 0.5, colored, 0.5, 0)

    vis_white = label(overlay(m_white, (255, 255, 255)), "WHITE")
    vis_blue  = label(overlay(m_blue,  (255, 100,   0)), "BLUE")
    vis_both  = label(overlay(m_both,  (  0, 200, 200)), "COMBINED")
    vis_orig  = label(frame.copy(), "ORIGINAL")

    top    = np.hstack([vis_orig,  vis_white])
    bottom = np.hstack([vis_blue,  vis_both])
    grid   = np.vstack([top, bottom])
    bar    = draw_info_bar(grid.shape[1], white, blue)
    return np.vstack([grid, bar])


# ── Main ──────────────────────────────────────────────────────────────────────

def load_current_params():
    """Read current threshold values from params.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("params", PARAMS_FILE)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    white = {
        "H_MIN": p.WHITE_H_MIN, "H_MAX": p.WHITE_H_MAX,
        "S_MIN": p.WHITE_S_MIN, "S_MAX": p.WHITE_S_MAX,
        "V_MIN": p.WHITE_V_MIN, "V_MAX": p.WHITE_V_MAX,
    }
    blue = {
        "H_MIN": p.BLUE_H_MIN,  "H_MAX": p.BLUE_H_MAX,
        "S_MIN": p.BLUE_S_MIN,  "S_MAX": p.BLUE_S_MAX,
        "V_MIN": p.BLUE_V_MIN,  "V_MAX": p.BLUE_V_MAX,
    }
    return white, blue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera_left/compressed",
                        help="Compressed image topic to subscribe to")
    args = parser.parse_args()

    white, blue = load_current_params()

    rclpy.init()
    node = FrameGrabber(args.topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 800)
    make_trackbars(WINDOW, white, blue)

    print(f"Subscribing to {args.topic}")
    print("Waiting for first frame...")

    placeholder = np.zeros((240, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (400, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)

    try:
        while True:
            white, blue = read_trackbars(WINDOW)
            frame = node.get_frame()

            if frame is None:
                cv2.imshow(WINDOW, placeholder)
            else:
                cv2.imshow(WINDOW, apply_masks(frame, white, blue))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_params(white, blue)

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
