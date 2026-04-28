#!/usr/bin/env python3
"""
Step 3 — Train a YOLOv8 detector on the dVRK dataset.

Usage:
    python 3_train.py                       # default settings
    python 3_train.py --model yolov8s.pt    # larger model
    python 3_train.py --epochs 150 --batch 8

Model size guide:
    yolov8n  — fastest, ~3ms/frame on GPU. Start here.
    yolov8s  — better accuracy, ~5ms/frame. Use if nano misses small pegs.
    yolov8m  — high accuracy, ~10ms/frame. Only if s is not enough.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_name: str = "yolov8n.pt",
    data_yaml: str = "config/dataset.yaml",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "models",
    run_name: str = "dvrk_v1",
    device: str = "",   # "" = auto-detect GPU
):
    """
    Fine-tune a pretrained YOLOv8 model on the dVRK dataset.

    All training artefacts (weights, plots, confusion matrix) land in:
        models/dvrk_v1/
    The best checkpoint is at:
        models/dvrk_v1/weights/best.pt
    """
    print(f"[INFO] Model   : {model_name}")
    print(f"[INFO] Dataset : {data_yaml}")
    print(f"[INFO] Epochs  : {epochs}   Batch: {batch}   imgsz: {imgsz}")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=run_name,
        device=device if device else None,

        # Augmentations — suitable for endoscope footage
        hsv_h=0.015,      # hue shift (catches LED color variation)
        hsv_s=0.5,        # saturation shift
        hsv_v=0.3,        # brightness shift (variable endoscope lighting)
        fliplr=0.5,       # horizontal flip (symmetric task setup)
        flipud=0.0,       # no vertical flip (pegs are always upright)
        scale=0.3,        # zoom in/out (cylinder distance varies)
        translate=0.1,
        mosaic=0.5,       # mosaic augmentation (helps small peg detection)
        degrees=5.0,      # small rotations (camera isn't perfectly level)

        # Training settings
        patience=30,      # early stop if no improvement for 30 epochs
        save_period=10,   # checkpoint every 10 epochs
        val=True,
        plots=True,       # saves confusion matrix, PR curve, etc.

        # Logging
        verbose=True,
    )

    best = Path(project) / run_name / "weights" / "best.pt"
    print(f"\n[DONE] Training complete.")
    print(f"       Best weights → {best}")
    return results


# ---------------------------------------------------------------------------
# Quick validation on test split
# ---------------------------------------------------------------------------

def validate(weights_path: str, data_yaml: str = "config/dataset.yaml"):
    """Run YOLO val on the test split and print metrics."""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="test", plots=True)

    print("\n── Test set metrics ──────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    # Per-class breakdown
    print("\n── Per-class mAP50 ───────────────────────")
    names = {0: "cylinder", 1: "peg_inactive", 2: "peg_lit_blue", 3: "peg_lit_white"}
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  [{i}] {names.get(i, '?'):<20s}: {ap:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="yolov8n.pt")
    p.add_argument("--data",    default="data.yaml")
    p.add_argument("--epochs",  type=int,   default=80)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--project", default="models")
    p.add_argument("--name",    default="dvrk_v1")
    p.add_argument("--device",  default="", help="e.g. '0' for GPU 0, 'cpu'")
    p.add_argument("--val_only", default="",
                   help="Skip training; run validation on this weights path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.val_only:
        validate(args.val_only, args.data)
    else:
        train(
            model_name=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            run_name=args.name,
            device=args.device,
        )
        # Auto-validate on test split after training
        best = str(Path(args.project) / args.name / "weights" / "best.pt")
        validate(best, args.data)
