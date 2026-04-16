#!/usr/bin/env python3
"""
dvrk_shujiro/camera/yolo_detector.py

YOLO inference wrapper — kept separate from the ROS node so it can
also be used standalone (e.g. running inference on saved frames offline).

Used by: dvrk_shujiro/nodes/detect_node.py
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# ── Class definitions — must match config/dataset.yaml ────────────────────

CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

# BGR colors for overlay drawing
CLASS_COLORS = {
    0: (40,  140, 220),   # cylinder      — orange-amber
    1: (100, 100, 100),   # peg_inactive  — gray
    2: (200,  80,  30),   # peg_lit_blue  — blue  (BGR order!)
    3: (200, 220, 255),   # peg_lit_white — warm white
}


class YoloDetector:
    """
    Thin wrapper around an Ultralytics YOLO model.

    Example
    -------
    detector = YoloDetector("models/dvrk_v1/weights/best.pt")
    detections = detector.detect(frame)
    annotated  = detector.draw(frame, detections)
    """

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.45,
        iou_threshold: float  = 0.50,
        device: str = "",          # "" = auto (GPU if available)
    ):
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        self.model = YOLO(str(weights_path))
        self.conf  = conf_threshold
        self.iou   = iou_threshold
        self.device = device or None

        # Warm-up pass to avoid latency spike on first real frame
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    # ── Inference ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns a list of detection dicts:
          {
            "class_id":    int,
            "class_name":  str,
            "conf":        float,
            "bbox_xyxy":   [x1, y1, x2, y2],   # pixel coords
            "bbox_center": [cx, cy],             # pixel coords
          }
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "class_id":    cls_id,
                "class_name":  CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "conf":        round(float(box.conf[0]), 4),
                "bbox_xyxy":   [x1, y1, x2, y2],
                "bbox_center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            })
        return detections

    # ── Drawing ───────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of frame.
        Returns the annotated copy.
        """
        out = frame.copy()
        for det in detections:
            cls_id = det["class_id"]
            x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label = f"{det['class_name']} {det['conf']:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label pill background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 1, cv2.LINE_AA,
            )
        return out

    # ── Convenience filters ───────────────────────────────────────────────

    def get_cylinders(self, detections: list[dict]) -> list[dict]:
        return [d for d in detections if d["class_id"] == 0]

    def get_lit_pegs(self, detections: list[dict]) -> list[dict]:
        return [d for d in detections if d["class_id"] in (2, 3)]

    def get_all_pegs(self, detections: list[dict]) -> list[dict]:
        return [d for d in detections if d["class_id"] in (1, 2, 3)]
