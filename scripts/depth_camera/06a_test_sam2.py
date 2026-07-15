"""
06a_test_sam2.py — Smoke-test that SAM2 loads on the GPU.

Prints the device it runs on and the total parameter count.
No segmentation is performed here.

Run:
    conda activate dvrk_ml
    python3 scripts/depth_camera/06a_test_sam2.py
"""

from pathlib import Path
import torch
from sam2.build_sam import build_sam2

SCRIPT_DIR = Path(__file__).parent
CKPT       = SCRIPT_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
CFG        = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    print(f"Device  : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    print(f"Loading : {CKPT.name} …")
    model = build_sam2(CFG, str(CKPT), device=DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}")
    print("SAM2 ready.")


if __name__ == "__main__":
    main()
