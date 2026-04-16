#!/usr/bin/env python3
"""
Step 4 — ROS 2 node: real-time cylinder & peg detection.

Subscribes to the endoscope camera topic, runs YOLO inference,
draws annotated bounding boxes, and re-publishes the result.

Run (from your ROS 2 workspace, after colcon build):
    ros2 run dvrk_detector detect_node \
        --ros-args \
        -p weights:=/path/to/models/dvrk_v1/weights/best.pt \
        -p image_topic:=/jhu_crsus/left/image_raw \
        -p conf_threshold:=0.45
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

import cv2
import json
import numpy as np


# ---------------------------------------------------------------------------
# Color palette for each class (BGR)
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    0: (180, 120,  40),   # cylinder       — amber
    1: ( 80,  80,  80),   # peg_inactive   — gray
    2: (200,  80,  30),   # peg_lit_blue   — blue  (BGR!)
    3: (255, 230, 100),   # peg_lit_white  — white/cream
}

CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class DetectNode(Node):

    def __init__(self):
        super().__init__("dvrk_detect_node")

        # ── Parameters (overridable via --ros-args -p key:=value) ─────────
        self.declare_parameter("weights",     "models/dvrk_v1/weights/best.pt")
        self.declare_parameter("image_topic", "/jhu_crsus/left/image_raw")
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("iou_threshold",  0.50)
        self.declare_parameter("publish_json",   True)   # publish detections as JSON

        weights    = self.get_parameter("weights").value
        img_topic  = self.get_parameter("image_topic").value
        self.conf  = self.get_parameter("conf_threshold").value
        self.iou   = self.get_parameter("iou_threshold").value
        self.pub_json = self.get_parameter("publish_json").value

        # ── Load model ────────────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO weights: {weights}")
        self.model = YOLO(weights)
        # Warm up — avoids latency spike on first real frame
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        self.get_logger().info("Model ready.")

        # ── ROS interfaces ────────────────────────────────────────────────
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, img_topic, self._image_callback, 10
        )

        # Annotated image output (view with: ros2 run rqt_image_view rqt_image_view)
        self.pub_image = self.create_publisher(Image, "/dvrk/detection/image", 10)

        # Structured detections as JSON string (subscribe for downstream scoring)
        if self.pub_json:
            self.pub_detections = self.create_publisher(
                String, "/dvrk/detection/objects", 10
            )

        self.get_logger().info(f"Subscribed to: {img_topic}")
        self.get_logger().info("Publishing annotated image to: /dvrk/detection/image")

        self._frame_count = 0

    # ── Callback ──────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        # Convert ROS Image → OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Run inference
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        # Draw and publish annotated frame
        annotated = self._draw_detections(frame.copy(), results)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header   # keep original timestamp
            self.pub_image.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish image error: {e}")

        # Publish structured detections as JSON
        if self.pub_json:
            det_list = self._results_to_dict(results, msg.header.stamp)
            json_msg = String()
            json_msg.data = json.dumps(det_list)
            self.pub_detections.publish(json_msg)

        self._frame_count += 1
        if self._frame_count % 100 == 0:
            self.get_logger().info(f"Processed {self._frame_count} frames")

    # ── Drawing helpers ────────────────────────────────────────────────────

    def _draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        h, w = frame.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

            # Label text (dark for readability on bright backgrounds)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA,
            )

        # Frame counter overlay (top-left)
        cv2.putText(
            frame, f"frame {self._frame_count}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (220, 220, 220), 1, cv2.LINE_AA,
        )
        return frame

    def _results_to_dict(self, results, stamp) -> list[dict]:
        """
        Convert YOLO results to a list of dicts for downstream consumers.

        Each dict:
          {
            "class_id":   int,
            "class_name": str,
            "conf":       float,
            "bbox_xyxy":  [x1, y1, x2, y2],   # pixel coords
            "bbox_center":[cx, cy],
            "timestamp":  {"sec": int, "nanosec": int}
          }
        """
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "class_id":    cls_id,
                "class_name":  CLASS_NAMES.get(cls_id, "unknown"),
                "conf":        round(float(box.conf[0]), 4),
                "bbox_xyxy":   [x1, y1, x2, y2],
                "bbox_center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "timestamp":   {"sec": stamp.sec, "nanosec": stamp.nanosec},
            })
        return detections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
