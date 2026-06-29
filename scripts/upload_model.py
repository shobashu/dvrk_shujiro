import shutil
import tempfile
from pathlib import Path

import roboflow

API_KEY = "kNjei7eJkL2YOs0ir5Il"
WORKSPACE = "shujiros-workspace"
PROJECT = "task_pad_cylinder_2"
VERSION = 1  # version number in your Roboflow project

MODEL_PT = Path(__file__).parent / "models" / "best_v3.pt"

rf = roboflow.Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

# Roboflow deploy() expects a directory with weights/best.pt inside
with tempfile.TemporaryDirectory() as tmp:
    weights_dir = Path(tmp) / "weights"
    weights_dir.mkdir()
    shutil.copy(MODEL_PT, weights_dir / "best.pt")
    version.deploy(model_type="yolov8", model_path=tmp + "/")

print("Upload complete.")
