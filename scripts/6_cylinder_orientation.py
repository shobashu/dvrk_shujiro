#!/usr/bin/env python3
"""
Step 6 — Real-time cylinder orientation on dVRK camera stream.

Subscribes to /camera_left/compressed, runs YOLO detection, and for each
detected cylinder draws the orientation axis on the live feed.

Axis drawing (per crop):
    green line   — orientation axis (white centroid → blue centroid, extended)
    orange line  — separation line  (perpendicular at midpoint)
    red dot      — blue-body centroid
    blue dot     — white-cap centroid
    cyan text    — angle in degrees  (0° = upright, ±180° = inverted)

Usage:
    python3 6_cylinder_orientation.py
    python3 6_cylinder_orientation.py --topic /camera_left/compressed
    python3 6_cylinder_orientation.py --weights models/best.pt --conf 0.45
"""

import argparse
import math
import re
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO

CYLINDER_CLASS_ID = 0
DEFAULT_WEIGHTS   = "models/best.pt"
DEFAULT_TOPIC     = "/camera_left/compressed"
MIN_MASK_PIXELS   = 50

# HSV thresholds (tuned via orientation_steps/params.py)
WHITE_LOWER = np.array([  0,   0, 150], dtype=np.uint8)
WHITE_UPPER = np.array([179, 200, 220], dtype=np.uint8)
BLUE_LOWER  = np.array([ 80,  80,  80], dtype=np.uint8)
BLUE_UPPER  = np.array([120, 160, 150], dtype=np.uint8)


# ── Orientation helpers ───────────────────────────────────────────────────────

def _centroid(mask: np.ndarray):
    """Median centre of the largest connected component, or None."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2:
        return None
    areas   = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    if stats[largest, cv2.CC_STAT_AREA] < MIN_MASK_PIXELS:
        return None
    ys, xs = np.where(labels == largest)
    return float(np.median(xs)), float(np.median(ys))


def compute_orientation(crop: np.ndarray):
    """
    Returns (angle_deg, blue_c, white_c, midpoint).

    Angle convention:
        0°   = white cap directly above blue body  (upright)
       ±180° = white cap directly below blue body  (inverted)
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    wm  = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    bm  = cv2.inRange(hsv, BLUE_LOWER,  BLUE_UPPER)

    h = crop.shape[0]

    blue_c = _centroid(bm)
    if blue_c is None:
        return 0.0, None, None, None

    split_y    = int(blue_c[1])
    blue_y_rel = blue_c[1] / h

    if blue_y_rel >= 0.5:
        white_c = _centroid(wm[:split_y, :])
    else:
        wc_rel  = _centroid(wm[split_y:, :])
        white_c = (wc_rel[0], wc_rel[1] + split_y) if wc_rel is not None else None

    if white_c is None:
        return 0.0, blue_c, None, None

    dx = blue_c[0] - white_c[0]
    dy = blue_c[1] - white_c[1]
    angle_deg = math.degrees(math.atan2(dx, dy))

    midpoint = ((white_c[0] + blue_c[0]) / 2,
                (white_c[1] + blue_c[1]) / 2)

    return angle_deg, blue_c, white_c, midpoint


def draw_orientation(crop: np.ndarray, angle_deg: float,
                     blue_c, white_c, midpoint) -> np.ndarray:
    """Draw axis, separator, centroids, and angle text onto a crop in-place copy."""
    vis   = crop.copy()
    h, w  = vis.shape[:2]
    scale = max(h, w) * 2

    if blue_c is not None:
        cv2.circle(vis, (int(blue_c[0]),  int(blue_c[1])),  5, (200,  30,  30), -1)
    if white_c is not None:
        cv2.circle(vis, (int(white_c[0]), int(white_c[1])), 5, (  0,   0, 220), -1)

    if white_c is not None and midpoint is not None:
        dx     = blue_c[0] - white_c[0]
        dy     = blue_c[1] - white_c[1]
        length = math.hypot(dx, dy)
        if length > 1:
            nx, ny = dx / length, dy / length
            p1 = (int(midpoint[0] - nx * scale), int(midpoint[1] - ny * scale))
            p2 = (int(midpoint[0] + nx * scale), int(midpoint[1] + ny * scale))
            cv2.line(vis, p1, p2, (0, 220, 0), 1)

            px, py = -ny, nx
            s1 = (int(midpoint[0] - px * scale), int(midpoint[1] - py * scale))
            s2 = (int(midpoint[0] + px * scale), int(midpoint[1] + py * scale))
            cv2.line(vis, s1, s2, (0, 165, 255), 2)

    text = f"{angle_deg:+.1f}deg"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(vis, (2, h - th - 10), (tw + 6, h - 2), (0, 0, 0), -1)
    cv2.putText(vis, text, (4, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    return vis


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class CylinderOrientationNode(Node):
    def __init__(self, topic: str, model: YOLO, conf: float, imgsz: int):
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', topic)
        safe = re.sub(r'_+', '_', safe).strip('_')
        super().__init__(f"cylinder_orientation_{safe}")

        self.model  = model
        self.conf   = conf
        self.imgsz  = imgsz

        self._raw_frame    = None
        self._raw_frame_id = 0
        self._last_infer_id   = -1
        self._annotated_frame = None
        self._raw_lock = threading.Lock()
        self._ann_lock = threading.Lock()

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.sub = self.create_subscription(CompressedImage, topic, self._cb, qos)
        self.get_logger().info(f"Subscribed to {topic}")

        self._stop        = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

    def _cb(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._raw_lock:
            self._raw_frame    = frame
            self._raw_frame_id += 1

    def _infer_loop(self):
        while not self._stop.is_set():
            with self._raw_lock:
                frame_id = self._raw_frame_id
                frame    = self._raw_frame

            if frame is None or frame_id == self._last_infer_id:
                self._stop.wait(0.005)
                continue
            self._last_infer_id = frame_id

            results   = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz,
                classes=[CYLINDER_CLASS_ID], verbose=False,
            )[0]
            annotated = self._draw(frame, results)

            with self._ann_lock:
                self._annotated_frame = annotated

    def _draw(self, frame: np.ndarray, results) -> np.ndarray:
        out = frame.copy()
        if results.boxes is None:
            return out

        h, w = frame.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            angle_deg, blue_c, white_c, midpoint = compute_orientation(crop)
            vis_crop = draw_orientation(crop, angle_deg, blue_c, white_c, midpoint)

            out[y1:y2, x1:x2] = vis_crop
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)

        return out

    def show(self, window_name: str):
        with self._ann_lock:
            frame = self._annotated_frame
        if frame is not None:
            cv2.imshow(window_name, frame)

    def stop(self):
        self._stop.set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rclpy.init()
    model = YOLO(args.weights)
    print(f"[INFO] Loaded weights: {args.weights}")
    print(f"[INFO] Topic:          {args.topic}")
    print(f"[INFO] Confidence:     {args.conf}")
    print(f"[INFO] Inference size: {args.imgsz}")
    print("[INFO] Press 'q' to quit.\n")

    window = "Cylinder Orientation"
    node   = CylinderOrientationNode(args.topic, model, args.conf, args.imgsz)

    executor    = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    try:
        while rclpy.ok():
            node.show(window)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        cv2.destroyAllWindows()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topic",   default=DEFAULT_TOPIC,   help="Compressed image topic")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO .pt weights")
    p.add_argument("--conf",    type=float, default=0.45, help="Detection confidence threshold")
    p.add_argument("--imgsz",   type=int,   default=320,  help="YOLO inference resolution")
    return p.parse_args()


if __name__ == "__main__":
    main()
