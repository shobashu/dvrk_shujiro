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
cd /home/cfxuser/dvrk_shujiro/scripts                                                                                                                                    
python 5_realtime_infer.py                                                                                                                                               
                                                                                                                                                                        
Two OpenCV windows will pop up (left and right camera) showing the live feed with bounding boxes drawn over the detected objects in real time. You control the arms as
normal — the inference runs independently in the background.                                                                                                             
                                                                            
One thing to note: the script runs on CPU right now (same as training, since the CUDA driver is outdated). Inference on CPU should still be fast enough for a live view, 
but if it feels laggy let me know and I can add a frame-skip option to reduce load.

Usage:
    # Both cameras (default)
    python 5_realtime_infer.py

    # Right camera only
    python 5_realtime_infer.py --camera right

    # Custom weights / confidence
    python 5_realtime_infer.py --weights models/best.pt --conf 0.4

    # Also publish annotated images back to ROS2 topics
    python 5_realtime_infer.py --publish

Prerequisites:
    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    # Cameras must be running (camera-stream-raw.sh)
"""

import argparse
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

from ultralytics import YOLO

CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

# Colours per class (BGR)
CLASS_COLORS = {
    0: (0,   200, 255),   # cylinder     — orange
    1: (180, 180, 180),   # peg_inactive — grey
    2: (255, 100,   0),   # peg_lit_blue — blue
    3: (255, 255, 255),   # peg_lit_white — white
}

DEFAULT_WEIGHTS = "models/best.pt"


# ---------------------------------------------------------------------------

class YOLOCameraNode(Node):
    def __init__(self, topic: str, window_name: str, model: YOLO,
                 conf: float, publish: bool):
        super().__init__(f"yolo_{window_name.replace(' ', '_')}")
        self.model = model
        self.conf = conf
        self.window_name = window_name
        self.latest_frame = None
        self.lock = threading.Lock()

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.sub = self.create_subscription(Image, topic, self._image_cb, qos)

        self.pub = None
        if publish:
            out_topic = topic.replace("/image_raw", "/image_yolo")
            from sensor_msgs.msg import Image as Img
            self.pub = self.create_publisher(Img, out_topic, 10)
            self.get_logger().info(f"Publishing annotated frames to {out_topic}")

        self.get_logger().info(f"Subscribed to {topic}")

    def _image_cb(self, msg: Image):
        # Convert ROS Image to numpy BGR
        dtype = np.uint8
        frame = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding in ("mono8",):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        with self.lock:
            self.latest_frame = frame.copy()

    def process_and_show(self) -> bool:
        """Run YOLO on latest frame and show it. Returns False if window closed."""
        with self.lock:
            frame = self.latest_frame

        if frame is None:
            return True

        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        annotated = self._draw(frame, results)

        cv2.imshow(self.window_name, annotated)

        if self.pub is not None:
            self._publish(annotated)

        return True

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
    print(f"[INFO] Loaded weights: {args.weights}")
    print(f"[INFO] Confidence threshold: {args.conf}")
    print("[INFO] Press 'q' in any window to quit.\n")

    nodes = []
    camera_map = {
        "left":  ("/camera_left/image_raw",  "Left Camera  — YOLO"),
        "right": ("/camera_right/image_raw", "Right Camera — YOLO"),
    }

    cameras = ["left", "right"] if args.camera == "both" else [args.camera]
    for cam in cameras:
        topic, window = camera_map[cam]
        node = YOLOCameraNode(topic, window, model, args.conf, args.publish)
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
                node.process_and_show()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--camera",  choices=["left", "right", "both"], default="both")
    p.add_argument("--publish", action="store_true",
                   help="Publish annotated frames to /camera_*/image_yolo")
    return p.parse_args()


if __name__ == "__main__":
    main()
