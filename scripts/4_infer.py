#!/usr/bin/env python3
"""
Step 4 — Run inference with a trained YOLOv8 model.

Usage:
    # Single image
    python 4_infer.py --src path/to/image.jpg

    # Folder of images
    python 4_infer.py --src path/to/folder/

    # Custom weights and confidence threshold
    python 4_infer.py --src path/to/folder/ --weights runs/detect/models/dvrk_v1-2/weights/best.pt --conf 0.4

    # Don't save — just display results in a window
    python 4_infer.py --src path/to/image.jpg --no-save --show
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = {
    0: "cylinder",
    1: "peg_inactive",
    2: "peg_lit_blue",
    3: "peg_lit_white",
}

DEFAULT_WEIGHTS = "runs/detect/models/dvrk_v1-2/weights/best.pt"


def infer(
    src: str,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    save: bool = True,
    show: bool = False,
    output_dir: str = "runs/infer",
):
    model = YOLO(weights)
    src_path = Path(src)

    if src_path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        sources = sorted([str(p) for p in src_path.iterdir() if p.suffix.lower() in image_exts])
        if not sources:
            print(f"[WARN] No images found in {src_path}")
            return
        print(f"[INFO] Found {len(sources)} images in {src_path}")
    else:
        sources = [str(src_path)]

    results = model.predict(
        source=sources,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save,
        show=show,
        project=output_dir,
        name="results",
        exist_ok=True,
        verbose=False,
    )

    # Print per-image summary
    for r in results:
        img_name = Path(r.path).name
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {img_name}: no detections")
            continue
        counts = {}
        for cls_id in boxes.cls.tolist():
            name = CLASS_NAMES.get(int(cls_id), str(int(cls_id)))
            counts[name] = counts.get(name, 0) + 1
        summary = ", ".join(f"{v}x {k}" for k, v in counts.items())
        print(f"  {img_name}: {summary}")

    if save:
        print(f"\n[DONE] Results saved to {output_dir}/results/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",     required=True, help="Image file or folder of images")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pt weights file")
    p.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--no-save", action="store_true", help="Don't save annotated images")
    p.add_argument("--show",    action="store_true", help="Display results in a window")
    p.add_argument("--output",  default="runs/infer", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(
        src=args.src,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=not args.no_save,
        show=args.show,
        output_dir=args.output,
    )
