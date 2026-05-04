#!/usr/bin/env python3
"""
dVRK stereo camera viewer.
Left window  : x=0,   y=0, 640x480
Right window : x=640, y=0, 640x480

Run via:
  bash view_cameras.sh

Press 'q' or Escape to quit.
"""

import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

LEFT_X,  LEFT_Y  = 0,   0   # DP-2
RIGHT_X, RIGHT_Y = 640, 0   # DP-0


class StereoViewer(Node):
    def __init__(self):
        super().__init__("stereo_viewer")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._lock = threading.Lock()
        self._frame_left  = None
        self._frame_right = None

        self.create_subscription(CompressedImage, "camera_left/compressed",  self._cb_left,  qos)
        self.create_subscription(CompressedImage, "camera_right/compressed", self._cb_right, qos)
        self.get_logger().info("Subscribed — press 'q' or Escape to quit.")

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

    def get_frames(self):
        with self._lock:
            l = self._frame_left.copy()  if self._frame_left  is not None else None
            r = self._frame_right.copy() if self._frame_right is not None else None
        return l, r


def main():
    rclpy.init()
    node = StereoViewer()

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    for name, x, y in [("left_cam", LEFT_X, LEFT_Y), ("right_cam", RIGHT_X, RIGHT_Y)]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, x, y)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, placeholder)
    cv2.waitKey(1)

    while True:
        left, right = node.get_frames()

        cv2.imshow("left_cam",  left  if left  is not None else placeholder)
        cv2.waitKey(1)
        cv2.imshow("right_cam", right if right is not None else placeholder)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
