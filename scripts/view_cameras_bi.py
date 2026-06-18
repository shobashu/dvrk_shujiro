#!/usr/bin/env python3
"""
dVRK stereo camera viewer with pedal-triggered USB camera overlay.

When the configured pedal is pressed, a small PIP (picture-in-picture) window
showing the USB Logitech camera pops up in the top-right corner of each screen
(left, right, and supervisor). The normal stereo view is never interrupted.
Pressing the pedal again hides the overlay windows.

── Configuration (edit these three lines) ────────────────────────────────────
  PEDAL_SOURCE     "ros"  → dVRK ROS topic (set BI_PEDAL_TOPIC below)
                   "usb"  → PCsensor FootSwitch plugged into the computer
  BI_PEDAL_TOPIC   ROS topic to use when PEDAL_SOURCE = "ros"
  USB_PEDAL_DEVICE evdev path to use when PEDAL_SOURCE = "usb"
──────────────────────────────────────────────────────────────────────────────

Run via:
  bash view_cameras_bi.sh

Press 'q' or Escape to quit.
"""

import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Joy

# ── User-configurable ─────────────────────────────────────────────────────────
PEDAL_SOURCE     = "usb"          # "ros" or "usb"
BI_PEDAL_TOPIC   = "/console/camera"   # used when PEDAL_SOURCE = "ros"
USB_PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
USB_CAMERA_INDEX = 0              # /dev/video0  — Logitech HD 1080p
# ─────────────────────────────────────────────────────────────────────────────

LEFT_X,  LEFT_Y  = 0,    0   # DP-2  640x480
RIGHT_X, RIGHT_Y = 640,  0   # DP-0  640x480
SUPER_X, SUPER_Y = 1280, 0   # supervisor monitor
MON_W,   MON_H   = 640, 480

PIP_W,   PIP_H   = 320, 240  # overlay PIP window size (top-right of each screen)
_PIP_NAMES = ("left_pip", "right_pip", "supervisor_pip")


def fill(frame: np.ndarray) -> np.ndarray:
    """Scale to fill MON_W×MON_H, center-crop excess — no squishing, no bars."""
    fh, fw = frame.shape[:2]
    if fw == MON_W and fh == MON_H:
        return frame
    scale = max(MON_W / fw, MON_H / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))
    x0 = (nw - MON_W) // 2
    y0 = (nh - MON_H) // 2
    return resized[y0:y0 + MON_H, x0:x0 + MON_W]


# ── Pedal state (shared between pedal thread and display loop) ────────────────
_pedal_lock   = threading.Lock()
_pedal_active = False


def _set_pedal(state: bool):
    global _pedal_active
    with _pedal_lock:
        _pedal_active = state


def _get_pedal() -> bool:
    with _pedal_lock:
        return _pedal_active


# ── USB pedal reader (evdev) ──────────────────────────────────────────────────
def _usb_pedal_thread():
    """Toggles pedal state on each press of the PCsensor FootSwitch via evdev."""
    try:
        import evdev
        from evdev import categorize, ecodes
        dev = evdev.InputDevice(USB_PEDAL_DEVICE)
        print(f"[pedal] USB footswitch opened: {dev.name}")
        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY:
                key = categorize(event)
                if key.keystate == 1:  # press only, ignore release and repeat
                    with _pedal_lock:
                        global _pedal_active
                        _pedal_active = not _pedal_active
                    print(f"[pedal] view {'→ USB camera' if _pedal_active else '→ stereo'}")
    except Exception as e:
        print(f"[pedal] USB pedal error: {e}")


# ── ROS node (stereo camera + optional ROS pedal) ─────────────────────────────
class StereoViewerBI(Node):
    def __init__(self):
        super().__init__("stereo_viewer_bi")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._lock        = threading.Lock()
        self._frame_left  = None
        self._frame_right = None

        self.create_subscription(CompressedImage, "camera_left/compressed",  self._cb_left,  qos)
        self.create_subscription(CompressedImage, "camera_right/compressed", self._cb_right, qos)

        if PEDAL_SOURCE == "ros":
            self.create_subscription(Joy, BI_PEDAL_TOPIC, self._cb_ros_pedal, 10)
            self.get_logger().info(f"Pedal source: ROS topic '{BI_PEDAL_TOPIC}'")
        else:
            self.get_logger().info(f"Pedal source: USB evdev '{USB_PEDAL_DEVICE}'")

        self.get_logger().info("Press 'q' or Escape to quit.")

    def _decode(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _cb_left(self, msg):
        frame = self._decode(msg)
        if frame is not None:
            with self._lock:
                self._frame_left = frame

    def _cb_right(self, msg):
        frame = self._decode(msg)
        if frame is not None:
            with self._lock:
                self._frame_right = frame

    def _cb_ros_pedal(self, msg: Joy):
        _set_pedal(len(msg.buttons) > 0 and msg.buttons[0] == 1)

    def get_frames(self):
        with self._lock:
            l = self._frame_left.copy()  if self._frame_left  is not None else None
            r = self._frame_right.copy() if self._frame_right is not None else None
        return l, r


def main():
    rclpy.init()
    node = StereoViewerBI()

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    if PEDAL_SOURCE == "usb":
        threading.Thread(target=_usb_pedal_thread, daemon=True).start()

    usb_cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not usb_cap.isOpened():
        node.get_logger().error(f"Could not open USB camera at index {USB_CAMERA_INDEX}")

    placeholder = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    usb_placeholder = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
    cv2.putText(usb_placeholder, "USB cam unavailable", (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    for name, x, y, fullscreen in [
        ("left_cam",   LEFT_X,  LEFT_Y,  True),
        ("right_cam",  RIGHT_X, RIGHT_Y, True),
        ("supervisor", SUPER_X, SUPER_Y, False),
    ]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, placeholder)
        cv2.waitKey(100)
        cv2.moveWindow(name, x, y)
        cv2.waitKey(100)
        if fullscreen:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(100)

    overlay_shown = False

    while True:
        left, right = node.get_frames()
        pedal = _get_pedal()

        # Normal feeds are always displayed regardless of pedal state
        left_frame = fill(left if left is not None else placeholder)
        cv2.imshow("left_cam",   left_frame)
        cv2.waitKey(1)
        cv2.imshow("right_cam",  fill(right if right is not None else placeholder))
        cv2.waitKey(1)
        cv2.imshow("supervisor", left_frame)

        # PIP overlay: show USB camera in top-right corner of each screen
        if pedal and usb_cap.isOpened():
            ret, usb_frame = usb_cap.read()
            pip = cv2.resize(usb_frame if ret else usb_placeholder, (PIP_W, PIP_H))

            if not overlay_shown:
                pip_positions = [
                    ("left_pip",       LEFT_X  + MON_W - PIP_W, LEFT_Y),
                    ("right_pip",      RIGHT_X + MON_W - PIP_W, RIGHT_Y),
                    ("supervisor_pip", SUPER_X + MON_W - PIP_W, SUPER_Y),
                ]
                for name, x, y in pip_positions:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(name, PIP_W, PIP_H)
                    cv2.imshow(name, pip)
                    cv2.waitKey(1)
                    cv2.moveWindow(name, x, y)
                    cv2.waitKey(1)
                overlay_shown = True
            else:
                for name in _PIP_NAMES:
                    cv2.imshow(name, pip)
        elif overlay_shown:
            for name in _PIP_NAMES:
                cv2.destroyWindow(name)
            overlay_shown = False

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    usb_cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
