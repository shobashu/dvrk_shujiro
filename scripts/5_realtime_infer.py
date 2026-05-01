#!/usr/bin/env python3
"""
Step 5 — Real-time YOLOv8 inference on dVRK camera streams.

Subscribes to ROS2 camera topics, runs YOLO on each frame, and displays
annotated results in OpenCV windows.

Terminal 1 — start the dVRK + cameras:
# however you normally start the dVRK console

Terminal 2 — start the cameras (if not already started by dVRK launch):
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
./camera-stream-raw.sh

Terminal 3 — run the inference:
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
cd /home/stanford/dvrk_shujiro_ws/src/dvrk_shujiro/scripts
python3 5_realtime_infer.py

Usage:
    # Both cameras, all classes (default)
    python3 5_realtime_infer.py

    # Cylinder only (faster, less noise)
    python3 5_realtime_infer.py --classes 0

    # Right camera only
    python3 5_realtime_infer.py --camera right

    # Smaller inference resolution for less lag
    python3 5_realtime_infer.py --imgsz 320

    # Custom weights / confidence
    python3 5_realtime_infer.py --weights models/best.pt --conf 0.4

    # Also publish annotated images back to ROS2 topics
    python3 5_realtime_infer.py --publish

Prerequisites:
    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    # Cameras must be running (camera-stream-raw.sh)
"""

import argparse
import re
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CompressedImage, Image

from ultralytics import YOLO

# 4 objects classes
CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

CLASS_COLORS = {
    0: (0,   200, 255), # orange
    1: (180, 180, 180), # gray
    2: (255, 100,   0), # blue
    3: (255, 255,   0), # yellow
}

DEFAULT_WEIGHTS = "models/best.pt"


# ---------------------------------------------------------------------------
# Node that subscribes to a camera topic, runs YOLO inference in a separate thread,
class YOLOCameraNode(Node):
    def __init__(self, topic: str, window_name: str, model: YOLO,
                 conf: float, imgsz: int, classes: list, publish: bool,
                 compressed: bool = False):
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', window_name)
        safe = re.sub(r'_+', '_', safe).strip('_')
        super().__init__(f"yolo_{safe}")

        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes or None
        self.window_name = window_name

        self._raw_frame = None
        self._raw_frame_id = 0        # incremented on every new frame
        self._last_infer_id = -1      # last frame id the inference thread processed
        self._annotated_frame = None
        self._raw_lock = threading.Lock()
        self._ann_lock = threading.Lock()

        qos = QoSPresetProfiles.SENSOR_DATA.value
        if compressed:
            self.sub = self.create_subscription(
                CompressedImage, topic, self._compressed_image_cb, qos)
        else:
            self.sub = self.create_subscription(
                Image, topic, self._image_cb, qos)

        self.pub = None
        if publish:
            out_topic = topic.replace("/compressed", "/image_yolo").replace("/image_raw", "/image_yolo")
            self.pub = self.create_publisher(Image, out_topic, 10)
            self.get_logger().info(f"Publishing annotated frames to {out_topic}")

        self.get_logger().info(f"Subscribed to {topic}")


        # Inference runs in its own thread so the display loop never blocks on YOLO
        self._stop = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "mono8":
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        with self._raw_lock:
            self._raw_frame = frame.copy()
            self._raw_frame_id += 1

    def _compressed_image_cb(self, msg: CompressedImage):
        # msg.data is raw JPEG/PNG bytes — cv2.imdecode handles any format
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._raw_lock:
            self._raw_frame = frame
            self._raw_frame_id += 1

    def _infer_loop(self):
        # This runs in a separate thread and continuously processes the latest frame.
        while not self._stop.is_set():
            with self._raw_lock:
                frame_id = self._raw_frame_id
                frame = self._raw_frame

            if frame is None or frame_id == self._last_infer_id:
                self._stop.wait(0.005)
                continue
            self._last_infer_id = frame_id

            results = self.model.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=self.classes,
                verbose=False,
            )[0]
            annotated = self._draw(frame, results)

            with self._ann_lock:
                self._annotated_frame = annotated

            if self.pub is not None:
                self._publish(annotated)

    def show(self):
        with self._ann_lock:
            frame = self._annotated_frame
        if frame is not None:
            # Display the annotated frame in the OpenCV window
            cv2.imshow(self.window_name, frame)

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------

    def _draw(self, frame: np.ndarray, results) -> np.ndarray:
        out = frame.copy()
        if results.boxes is None:
            return out
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color  = CLASS_COLORS.get(cls_id, (0, 255, 0))
            label  = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _publish(self, frame: np.ndarray):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        self.pub.publish(msg)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    rclpy.init()
    model = YOLO(args.weights)
    print(f"[INFO] Loaded weights:   {args.weights}")
    print(f"[INFO] Confidence:       {args.conf}")
    print(f"[INFO] Inference size:   {args.imgsz}")
    print(f"[INFO] Classes filter:   {args.classes if args.classes else 'all'}")
    print("[INFO] Press 'q' in any window to quit.\n")

    if args.compressed:
        camera_map = {
            "left":  ("/camera_left/compressed",  "Left Camera  — YOLO"),
            "right": ("/camera_right/compressed", "Right Camera — YOLO"),
        }
    else:
        camera_map = {
            "left":  ("/camera_left/image_raw",  "Left Camera  — YOLO"),
            "right": ("/camera_right/image_raw", "Right Camera — YOLO"),
        }

    cameras = ["left", "right"] if args.camera == "both" else [args.camera]
    nodes = []
    for cam in cameras:
        topic, window = camera_map[cam]
        node = YOLOCameraNode(topic, window, model,
                              args.conf, args.imgsz, args.classes, args.publish,
                              compressed=args.compressed)
        nodes.append(node)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 640, 480)

    executor = rclpy.executors.MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            for node in nodes:
                node.show()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.stop()
        cv2.destroyAllWindows()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf",    type=float, default=0.5)
    p.add_argument("--imgsz",   type=int,   default=320,
                   help="YOLO inference resolution (smaller = faster, default 320)")
    p.add_argument("--classes", type=int,   nargs="+", default=None,
                   help="Class IDs to detect: 0=cylinder 1=peg_inactive 2=peg_lit_blue 3=peg_lit_white")
    p.add_argument("--camera",     choices=["left", "right", "both"], default="both")
    p.add_argument("--compressed", action="store_true",
                   help="Subscribe to /compressed topics (CompressedImage) instead of /image_raw")
    p.add_argument("--publish", action="store_true",
                   help="Publish annotated frames to /camera_*/image_yolo")
    return p.parse_args()


if __name__ == "__main__":
    main()
