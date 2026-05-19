#!/usr/bin/env python3
"""
Train a YOLOv8 model on the dVRK peg-transfer dataset.

Run AFTER 2_prepare_dataset.py has built data/dataset/ and config/dataset.yaml.

STEP 1 — Verify the dataset exists and count images
STEP 2 — Load the base YOLOv8 model (pretrained on COCO)
STEP 3 — Train with dVRK-specific augmentations
STEP 4 — Evaluate on the held-out test split
STEP 5 — Report where the best weights were saved

Usage:
    python scripts/3_train.py
    python scripts/3_train.py --model yolov8s.pt --epochs 150 --batch 8

Model size guide:
    yolov8n  — fastest inference (~3 ms/frame on GPU).  Start here.
    yolov8s  — better accuracy (~5 ms/frame).           Use if nano misses small pegs.
    yolov8m  — highest accuracy (~10 ms/frame).         Only if small is not enough.
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


# ── Training configuration ───────────────────────────────────────────────────────
# Edit these defaults before running, or override with CLI flags.

DEFAULT_CONFIG = {
    "model":   "yolov8s.pt",   # base weights (downloaded automatically if absent)
    "data":    "config/dataset.yaml",
    "epochs":  100,
    "imgsz":   640,
    "batch":   16,
    "device":  "",             # "" = auto (GPU if available, else CPU)
    "project": "models",
    "name":    "dvrk_v2",
}

# Augmentations tuned for endoscope / LED peg-transfer footage
AUGMENTATION_CONFIG = {
    "hsv_h":    0.015,   # hue shift  — catches LED colour variation
    "hsv_s":    0.5,     # saturation — variable endoscope lighting
    "hsv_v":    0.3,     # brightness — variable endoscope lighting
    "fliplr":   0.5,     # horizontal flip — symmetric task setup
    "flipud":   0.0,     # no vertical flip — pegs are always upright
    "scale":    0.3,     # zoom in/out — cylinder distance varies
    "translate": 0.1,
    "mosaic":   0.5,     # helps detect small pegs
    "degrees":  5.0,     # small rotations — camera isn't perfectly level
}


# ── STEP 1 — Verify dataset ──────────────────────────────────────────────────────

def step1_verify_dataset(data_yaml: str):
    print("\nSTEP 1 — Verifying dataset")
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {yaml_path}\n"
            "Run 2_prepare_dataset.py first."
        )

    cfg = yaml.safe_load(yaml_path.read_text())
    dataset_root = Path(cfg["path"])

    counts = {}
    for split in ("train", "val", "test"):
        img_dir = dataset_root / "images" / split
        counts[split] = len(list(img_dir.glob("*"))) if img_dir.exists() else 0

    print(f"  Config  : {yaml_path}")
    print(f"  Classes : {cfg['names']}")
    print(f"  Train   : {counts['train']} images")
    print(f"  Val     : {counts['val']} images")
    print(f"  Test    : {counts['test']} images")

    if counts["train"] == 0:
        raise RuntimeError("Train split is empty — did 2_prepare_dataset.py complete successfully?")


# ── STEP 2 — Load model ──────────────────────────────────────────────────────────

def step2_load_model(model_name: str) -> YOLO:
    print(f"\nSTEP 2 — Loading base model: {model_name}")
    model = YOLO(model_name)
    print(f"  Model loaded (pretrained on COCO, will be fine-tuned on dVRK data)")
    return model


# ── STEP 3 — Train ──────────────────────────────────────────────────────────────

def step3_train(model: YOLO, cfg: dict) -> str:
    print(f"\nSTEP 3 — Training")
    print(f"  Epochs : {cfg['epochs']}   Batch : {cfg['batch']}   imgsz : {cfg['imgsz']}")
    print(f"  Output : {cfg['project']}/{cfg['name']}/")
    print()

    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        project=cfg["project"],
        name=cfg["name"],
        device=cfg["device"] if cfg["device"] else None,

        # Augmentations
        **AUGMENTATION_CONFIG,

        # Training settings
        patience=30,      # early-stop after 30 epochs without improvement
        save_period=10,   # save a checkpoint every 10 epochs
        val=True,
        plots=True,       # saves confusion matrix, PR curve, etc.
        verbose=True,
    )

    best = str(Path(results.save_dir) / "weights" / "best.pt")
    return best


# ── STEP 4 — Evaluate on test split ─────────────────────────────────────────────

def step4_evaluate(weights_path: str, data_yaml: str):
    print(f"\nSTEP 4 — Evaluating on test split")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="test", plots=True)

    print("\n  ── Test metrics ──────────────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    print("\n  ── Per-class mAP50 ───────────────────────────")
    names = {0: "cylinder", 1: "peg_inactive", 2: "peg_lit_blue", 3: "peg_lit_white"}
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  [{i}] {names.get(i, '?'):<20s}: {ap:.4f}")

    return metrics


# ── STEP 5 — Report ─────────────────────────────────────────────────────────────

def step5_report(best_weights: str):
    best_src = Path(best_weights)
    named    = Path("models") / "best_v2.pt"
    shutil.copy2(best_src, named)

    print(f"\nSTEP 5 — Done")
    print(f"  Best weights : {best_weights}")
    print(f"  Named copy   : {named}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default=DEFAULT_CONFIG["model"])
    p.add_argument("--data",    default=DEFAULT_CONFIG["data"])
    p.add_argument("--epochs",  type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--imgsz",   type=int,   default=DEFAULT_CONFIG["imgsz"])
    p.add_argument("--batch",   type=int,   default=DEFAULT_CONFIG["batch"])
    p.add_argument("--device",  default=DEFAULT_CONFIG["device"],
                   help="GPU id (e.g. '0') or 'cpu'. Empty = auto-detect.")
    p.add_argument("--project", default=DEFAULT_CONFIG["project"])
    p.add_argument("--name",    default=DEFAULT_CONFIG["name"])
    p.add_argument("--val_only", default="",
                   help="Skip training; run test evaluation on this weights path.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "model":   args.model,
        "data":    args.data,
        "epochs":  args.epochs,
        "imgsz":   args.imgsz,
        "batch":   args.batch,
        "device":  args.device,
        "project": args.project,
        "name":    args.name,
    }

    if args.val_only:
        step4_evaluate(args.val_only, cfg["data"])
        return

    step1_verify_dataset(cfg["data"])
    model      = step2_load_model(cfg["model"])
    best       = step3_train(model, cfg)
    step4_evaluate(best, cfg["data"])
    step5_report(best)


if __name__ == "__main__":
    main()
