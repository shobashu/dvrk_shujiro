#!/usr/bin/env python3
"""
Step 8 — Multi-cylinder tracking with persistent IDs (YOLO + ByteTrack).

Uses Ultralytics built-in tracking (model.track) so each cylinder gets a
stable ID across frames — no need for a separate DeepSORT install.

Each cylinder is drawn with a unique colour and its track ID, so you can
visually confirm that IDs stay consistent even when cylinders move or
temporarily disappear.

Usage:
    python3 scripts/8_multi_cylinder_track.py
    python3 scripts/8_multi_cylinder_track.py --weights models/best_v2.pt --conf 0.45
    python3 scripts/8_multi_cylinder_track.py --log track_log.csv --record track.mp4
    python3 scripts/8_multi_cylinder_track.py --tracker botsort   # appearance-based (more robust)

Bytetrack is faster but can lose IDs if cylinders occlude each other; botsort is more robust but slightly slower.
Botsort is slightly more robust to occlusions (e.g. one cylinder briefly passing in front of another) since it uses an appearance model, but it requires more compute and may not run at full frame rate on older machines. Bytetrack is very fast but can lose track IDs in crowded scenes. You can switch between them with the --tracker argument.

"""

import argparse
import re
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO

CYLINDER_CLASS_ID = 0
DEFAULT_WEIGHTS   = "models/best_v2.pt"
DEFAULT_TOPIC     = "/camera_left/compressed"

# one distinct colour per track ID (cycles if > 20 cylinders)
_PALETTE = [
    (0, 220, 0),   (0, 120, 255), (255, 60, 0),   (0, 220, 220),
    (220, 0, 220), (255, 200, 0), (0, 180, 120),   (180, 0, 80),
    (80, 180, 0),  (0, 80, 220),  (220, 120, 0),   (120, 0, 220),
    (0, 220, 80),  (220, 80, 0),  (80, 220, 220),  (220, 220, 0),
    (0, 0, 220),   (220, 0, 0),   (0, 220, 160),   (160, 220, 0),
]

def _track_color(track_id: int):
    return _PALETTE[int(track_id) % len(_PALETTE)]

def _stamp() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class MultiCylinderTrackNode(Node):
    def __init__(self, topic: str, model: YOLO, conf: float, imgsz: int,
                 tracker: str, log_path):
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', topic)
        safe = re.sub(r'_+', '_', safe).strip('_')
        super().__init__(f"multi_cylinder_track_{safe}")

        self.model  = model
        self.conf   = conf
        self.imgsz  = imgsz
        self.tracker_cfg = f"{tracker}.yaml"

        self._raw_frame    = None
        self._raw_frame_id = 0
        self._last_infer_id   = -1
        self._annotated_frame = None
        self._raw_lock = threading.Lock()
        self._ann_lock = threading.Lock()

        # track_id → last known centre
        self._tracks: dict[int, tuple] = {}

        self._log_file  = open(log_path, "w") if log_path else None
        self._start_time = time.time()
        if self._log_file:
            self._log_file.write(f"# recorded {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._log_file.write("timestamp_s,track_id,cx,cy,x1,y1,x2,y2,conf\n")

        self._stop = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.sub = self.create_subscription(CompressedImage, topic, self._cb, qos)
        self.get_logger().info(f"Subscribed to {topic}  tracker={self.tracker_cfg}")

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

            # model.track keeps internal state across calls (persist=True)
            results = self.model.track(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=[CYLINDER_CLASS_ID],
                tracker=self.tracker_cfg,
                persist=True,
                verbose=False,
            )[0]

            annotated = self._process(frame, results)
            with self._ann_lock:
                self._annotated_frame = annotated

    def _process(self, frame: np.ndarray, results) -> np.ndarray:
        out = frame.copy()
        now = time.time() - self._start_time
        h, w = frame.shape[:2]

        active_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id, conf_val in zip(
                    results.boxes.xyxy,
                    results.boxes.id,
                    results.boxes.conf):

                tid  = int(track_id.item())
                conf_score = float(conf_val.item())
                x1, y1, x2, y2 = map(int, box.tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                color = _track_color(tid)

                active_ids.add(tid)
                self._tracks[tid] = (cx, cy)

                # bounding box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                # centre cross
                icx, icy = int(cx), int(cy)
                cv2.line(out, (icx - 8, icy), (icx + 8, icy), color, 2)
                cv2.line(out, (icx, icy - 8), (icx, icy + 8), color, 2)
                cv2.circle(out, (icx, icy), 4, (255, 255, 255), -1)
                cv2.circle(out, (icx, icy), 4, color, 1)

                # ID label
                label = f"Cyl #{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = max(y1, th + 8)
                cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 6, ly), color, -1)
                cv2.putText(out, label, (x1 + 3, ly - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                # coord label
                cv2.putText(out, f"({icx},{icy})", (icx + 10, icy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                if self._log_file:
                    self._log_file.write(
                        f"{now:.3f},{tid},{cx:.1f},{cy:.1f},"
                        f"{x1},{y1},{x2},{y2},{conf_score:.3f}\n")

        # HUD
        hud = f"Tracking {len(active_ids)} cylinder(s)  [{_stamp()}]"
        cv2.rectangle(out, (4, 4), (14 + len(hud) * 11, 30), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # legend: one colour swatch per known track ID
        for i, (tid, _) in enumerate(sorted(self._tracks.items())):
            col = _track_color(tid)
            lx, ly = w - 110, 12 + i * 22
            cv2.rectangle(out, (lx, ly), (lx + 18, ly + 14), col, -1)
            cv2.putText(out, f"Cyl #{tid}", (lx + 22, ly + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        return out

    def get_annotated_frame(self):
        with self._ann_lock:
            return self._annotated_frame

    def stop(self):
        self._stop.set()
        if self._log_file:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topic",    default=DEFAULT_TOPIC)
    p.add_argument("--weights",  default=DEFAULT_WEIGHTS)
    p.add_argument("--conf",     type=float, default=0.45)
    p.add_argument("--imgsz",    type=int,   default=320)
    p.add_argument("--tracker",  default="bytetrack",
                   choices=["bytetrack", "botsort"],
                   help="bytetrack=fast no appearance model; botsort=appearance-based more robust")
    p.add_argument("--log",      default=None,
                   help="CSV log path (optional)")
    p.add_argument("--record",   default=None,
                   help="MP4 output path (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    model = YOLO(args.weights)
    print(f"[INFO] Weights:  {args.weights}")
    print(f"[INFO] Topic:    {args.topic}")
    print(f"[INFO] Tracker:  {args.tracker}")
    print(f"[INFO] Press 'q' to quit.\n")

    node = MultiCylinderTrackNode(
        args.topic, model, args.conf, args.imgsz, args.tracker, args.log)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    window = "Multi-Cylinder Tracking"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    video_writer = None

    try:
        while rclpy.ok():
            frame = node.get_annotated_frame()
            if frame is not None:
                cv2.imshow(window, frame)
                if args.record:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(args.record, fourcc, 30.0, (w, h))
                        print(f"[INFO] Recording → {args.record}")
                    video_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved → {args.record}")
        cv2.destroyAllWindows()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
