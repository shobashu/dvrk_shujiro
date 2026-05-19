#!/usr/bin/env python3
"""
Step 7 — Real-time cylinder velocity / state check on dVRK camera stream.

Subscribes to /camera_left/compressed, runs YOLO detection, tracks the
centre of the cylinder bounding box over time, computes pixel velocity,
and classifies each frame into one of three states:

    STATIONARY — cylinder still (velocity < --vel-stat)
    HELD       — smooth controlled motion (between --vel-stat and --vel-drop)
    DROPPED    — sudden velocity spike, or tracking lost while held

On every state change a line is printed to stdout, e.g.:
    [12:34:56.789] HELD -> DROPPED  vel=73.2px/s  pos=(312,240)

Usage:
    python3 7_velocity_check.py
    python3 7_velocity_check.py --topic /camera_left/compressed
    python3 7_velocity_check.py --weights models/best.pt --conf 0.45
    python3 7_velocity_check.py --vel-stat 6 --vel-drop 80 --lost 4
    python3 7_velocity_check.py --log cylinder_states.csv
"""

import argparse
import re
import threading
import time
from collections import deque

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

STATE_STATIONARY = "STATIONARY"
STATE_HELD       = "HELD"
STATE_DROPPED    = "DROPPED"

STATE_COLORS = {
    STATE_STATIONARY: (200, 200, 200),
    STATE_HELD:       (0,   220, 0),
    STATE_DROPPED:    (0,    30, 255),
}

SPARK_W      = 150
SPARK_H      = 50
SPARK_SAMPLES = 60


# ── Visualization helpers ─────────────────────────────────────────────────────

def _stamp() -> str:
    """Wall-clock HH:MM:SS.mmm timestamp string."""
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


def draw_sparkline(frame: np.ndarray, samples, vel_drop: float):
    """Draw a velocity sparkline in the bottom-left corner, in-place."""
    h = frame.shape[0]
    x0, y0 = 6, h - SPARK_H - 6
    canvas = np.full((SPARK_H, SPARK_W, 3), 255, dtype=np.uint8)

    # scale so both the data range and the drop threshold stay visible
    vmax = max(vel_drop * 1.2, max(samples) if samples else 1.0, 1.0)

    drop_y = int(SPARK_H - 1 - (vel_drop / vmax) * (SPARK_H - 1))
    for dx in range(0, SPARK_W, 8):
        cv2.line(canvas, (dx, drop_y), (dx + 4, drop_y), (0, 0, 255), 1)

    if len(samples) >= 2:
        n   = len(samples)
        pts = []
        for i, v in enumerate(samples):
            px = int(i / (n - 1) * (SPARK_W - 1))
            py = int(SPARK_H - 1 - min(v, vmax) / vmax * (SPARK_H - 1))
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 0, 0), 1)

    cv2.rectangle(canvas, (0, 0), (SPARK_W - 1, SPARK_H - 1), (120, 120, 120), 1)
    frame[y0:y0 + SPARK_H, x0:x0 + SPARK_W] = canvas


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class CylinderVelocityNode(Node):
    def __init__(self, topic: str, model: YOLO, conf: float, imgsz: int,
                 vel_stat: float, vel_drop: float, lost_frames: int,
                 vel_window: int, log_path):
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', topic)
        safe = re.sub(r'_+', '_', safe).strip('_')
        super().__init__(f"cylinder_velocity_{safe}")

        self.model = model
        self.conf  = conf
        self.imgsz = imgsz

        self.vel_stat    = vel_stat
        self.vel_drop    = vel_drop
        self.lost_frames = lost_frames

        self._raw_frame    = None
        self._raw_frame_id = 0
        self._last_infer_id   = -1
        self._annotated_frame = None
        self._raw_lock = threading.Lock()
        self._ann_lock = threading.Lock()

        # state machine (lives entirely in the inference thread)
        self._state       = STATE_STATIONARY
        self._prev_center = None
        self._prev_time   = None
        self._lost_count  = 0
        self._event_count = 0
        self._vel_hist    = deque(maxlen=vel_window)
        self._spark_hist  = deque(maxlen=SPARK_SAMPLES)

        self._log_file = open(log_path, "w") if log_path else None
        if self._log_file is not None:
            self._log_file.write("timestamp_s,state,vel_px_s,cx,cy\n")
        self._start_time = time.time()

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

            results = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz,
                classes=[CYLINDER_CLASS_ID], verbose=False,
            )[0]
            annotated = self._process(frame, results)

            with self._ann_lock:
                self._annotated_frame = annotated

    def _largest_box(self, results, w: int, h: int):
        """Return (cx, cy, x1, y1, x2, y2) of the biggest cylinder box, or None."""
        if results.boxes is None or len(results.boxes) == 0:
            return None
        best, best_area = None, -1.0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if best is None or best_area <= 0:
            return None
        x1, y1, x2, y2 = best
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x1, y1, x2, y2

    def _set_state(self, new_state: str, vel: float, center):
        if new_state == self._state:
            return
        self._event_count += 1
        cx = int(center[0]) if center is not None else -1
        cy = int(center[1]) if center is not None else -1
        print(f"[{_stamp()}] {self._state} -> {new_state}  "
              f"vel={vel:.1f}px/s  pos=({cx},{cy})")
        self._state = new_state

    def _process(self, frame: np.ndarray, results) -> np.ndarray:
        out  = frame.copy()
        h, w = frame.shape[:2]
        now  = time.time()

        det = self._largest_box(results, w, h)

        if det is None:
            # tracking loss: a sustained miss after a hold counts as a drop
            self._lost_count += 1
            if (self._lost_count >= self.lost_frames
                    and self._state in (STATE_HELD, STATE_DROPPED)):
                self._set_state(STATE_DROPPED, 0.0, None)
            self._prev_center = None
            self._prev_time   = None
            smoothed = self._vel_hist[-1] if self._vel_hist else 0.0
            self._finish(out, None, smoothed, now)
            return out

        self._lost_count = 0
        cx, cy, x1, y1, x2, y2 = det
        center = (cx, cy)

        raw_vel = 0.0
        if self._prev_center is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-6:
                raw_vel = float(np.hypot(cx - self._prev_center[0],
                                         cy - self._prev_center[1]) / dt)
        self._prev_center = center
        self._prev_time   = now

        self._vel_hist.append(raw_vel)
        smoothed = float(np.mean(self._vel_hist))

        # classification — spike test uses raw velocity, levels use smoothed
        if raw_vel >= self.vel_drop:
            self._set_state(STATE_DROPPED, raw_vel, center)
        elif self._state == STATE_DROPPED:
            # re-acquired after a drop → settle back to stationary
            self._set_state(STATE_STATIONARY, smoothed, center)
        elif smoothed < self.vel_stat:
            self._set_state(STATE_STATIONARY, smoothed, center)
        else:
            self._set_state(STATE_HELD, smoothed, center)

        color = STATE_COLORS[self._state]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{self._state} {smoothed:.1f}px/s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1, th + 6)
        cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
        cv2.putText(out, label, (x1 + 2, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        self._finish(out, center, smoothed, now)
        return out

    def _finish(self, out: np.ndarray, center, smoothed: float, now: float):
        """Sparkline, HUD text, and CSV row — shared by both detection paths."""
        self._spark_hist.append(smoothed)
        draw_sparkline(out, list(self._spark_hist), self.vel_drop)

        hud = f"State: {self._state} | Events: {self._event_count}"
        cv2.rectangle(out, (4, 4), (12 + len(hud) * 11, 30), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if self._log_file is not None:
            cx = int(center[0]) if center is not None else -1
            cy = int(center[1]) if center is not None else -1
            self._log_file.write(
                f"{now - self._start_time:.3f},{self._state},"
                f"{smoothed:.3f},{cx},{cy}\n")

    def show(self, window_name: str):
        with self._ann_lock:
            frame = self._annotated_frame
        if frame is not None:
            cv2.imshow(window_name, frame)

    def stop(self):
        self._stop.set()
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rclpy.init()
    model = YOLO(args.weights)
    print(f"[INFO] Loaded weights: {args.weights}")
    print(f"[INFO] Topic:          {args.topic}")
    print(f"[INFO] Confidence:     {args.conf}")
    print(f"[INFO] Inference size: {args.imgsz}")
    print(f"[INFO] Stationary <    {args.vel_stat} px/s")
    print(f"[INFO] Drop >=         {args.vel_drop} px/s")
    print(f"[INFO] Lost frames:    {args.lost}")
    print(f"[INFO] Vel window:     {args.vel_window}")
    if args.log:
        print(f"[INFO] CSV log:        {args.log}")
    print("[INFO] Press 'q' to quit.\n")

    window = "Cylinder Velocity Check"
    node   = CylinderVelocityNode(
        args.topic, model, args.conf, args.imgsz,
        args.vel_stat, args.vel_drop, args.lost, args.vel_window, args.log)

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
    p.add_argument("--topic",      default=DEFAULT_TOPIC,   help="Compressed image topic")
    p.add_argument("--weights",    default=DEFAULT_WEIGHTS, help="YOLO .pt weights")
    p.add_argument("--conf",       type=float, default=0.45, help="Detection confidence threshold")
    p.add_argument("--imgsz",      type=int,   default=320,  help="YOLO inference resolution")
    p.add_argument("--vel-stat",   type=float, default=8.0,
                   help="Stationary threshold px/s")
    p.add_argument("--vel-drop",   type=float, default=60.0,
                   help="Drop (velocity spike) threshold px/s")
    p.add_argument("--lost",       type=int,   default=5,
                   help="Consecutive missed detections before tracking-loss -> DROPPED")
    p.add_argument("--vel-window", type=int,   default=5,
                   help="Rolling average window for velocity smoothing")
    p.add_argument("--log",        default=None,
                   help="Optional path for CSV output")
    return p.parse_args()


if __name__ == "__main__":
    main()
