#!/usr/bin/env python3
"""
Step 1 — YOLO detection and crop extraction.

Runs YOLO on all images in params.SRC, extracts each detected cylinder
bounding box as a separate crop image, and saves a crops.json metadata file.

Output:
    runs/orientation_steps/01_crops/<image_stem>_cyl<id>.jpg
    runs/orientation_steps/01_crops/crops.json

Run:
    cd scripts/orientation_steps
    python3 step1_detect_crop.py
"""

import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import params

CYLINDER_CLASS_ID = 0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    src     = Path(params.SRC).expanduser()
    out_dir = Path(params.OUT_ROOT) / "01_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    model   = YOLO(params.WEIGHTS)
    sources = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"[INFO] {len(sources)} images  →  {out_dir}")

    metadata    = []
    total_crops = 0

    for img_path in sources:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [WARN] cannot read {img_path.name}")
            continue

        results = model.predict(
            source=str(img_path),
            conf=params.CONF,
            iou=params.IOU,
            imgsz=params.IMGSZ,
            verbose=False,
        )[0]

        cyl_id = 0
        for box in results.boxes:
            if int(box.cls[0]) != CYLINDER_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{img_path.stem}_cyl{cyl_id}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)

            metadata.append({
                "source_image": img_path.name,
                "crop_file":    crop_name,
                "cylinder_id":  cyl_id,
                "bbox":         [x1, y1, x2, y2],
                "confidence":   round(float(box.conf[0]), 4),
            })
            cyl_id     += 1
            total_crops += 1

        print(f"  {img_path.name}: {cyl_id} cylinder(s)")

    json_path = out_dir / "crops.json"
    json_path.write_text(json.dumps(metadata, indent=2))

    print(f"\n[DONE] {total_crops} crops saved")
    print(f"       crops    →  {out_dir}/")
    print(f"       metadata →  {json_path}")


if __name__ == "__main__":
    main()
