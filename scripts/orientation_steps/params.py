"""
Shared parameters for the orientation pipeline.
Edit this file to tune thresholds, then re-run the relevant step.
"""
from pathlib import Path

_HERE    = Path(__file__).parent   # orientation_steps/
_SCRIPTS = _HERE.parent            # scripts/

# ── Input ─────────────────────────────────────────────────────────────────────
SRC     = str(Path.home() / "data_shujiro/Task_Pad_Cylinder_Pegs_yolov8/train/images")
WEIGHTS = str(_SCRIPTS / "models/best.pt")
CONF    = 0.45
IOU     = 0.45
IMGSZ   = 640

# ── Pipeline output root ──────────────────────────────────────────────────────
OUT_ROOT = str(_SCRIPTS / "runs/orientation_steps")

# ── Quick-test limiter ────────────────────────────────────────────────────────
# Set to a small number while tuning (e.g. 4), None to process all crops.
SAMPLE = 4

# ── HSV thresholds ────────────────────────────────────────────────────────────
# OpenCV scale: H 0–179, S 0–255, V 0–255
#
# After running step2_hsv_view.py, open a panel and note:
#   • What H value the blue part of the cylinder shows  (expect ~100–120)
#   • What S value separates white (low S) from coloured areas
#   • What V value the white cap has  (expect >180)

# WHITE_H_MIN, WHITE_H_MAX =   0, 179   # white has no dominant hue → keep full range
# WHITE_S_MIN, WHITE_S_MAX =   0,  55   # low saturation  → white/grey
# WHITE_V_MIN, WHITE_V_MAX = 180, 255   # high brightness → white

# BLUE_H_MIN,  BLUE_H_MAX  =  90, 130   # blue hue range
# BLUE_S_MIN,  BLUE_S_MAX  =  100, 170   # medium-high saturation
# BLUE_V_MIN,  BLUE_V_MAX  =  40, 255   # any brightness (handles shadows)

WHITE_H_MIN, WHITE_H_MAX =   0, 179   # white has no dominant hue → keep full range
WHITE_S_MIN, WHITE_S_MAX =   0, 200   # low saturation  → white/grey
WHITE_V_MIN, WHITE_V_MAX = 150, 220   # high brightness → white

BLUE_H_MIN,  BLUE_H_MAX  =  80, 120   # blue hue range
BLUE_S_MIN,  BLUE_S_MAX  =  80, 160   # medium-high saturation
BLUE_V_MIN,  BLUE_V_MAX  =  80, 150   # any brightness (handles shadows)

# ── Classification thresholds (step 5) ───────────────────────────────────────
# boundary_y_rel = blue centroid row / crop height   (0 = top, 1 = bottom)
UPRIGHT_THRESHOLD  = 0.55   # boundary_y_rel > this  →  upright
INVERTED_THRESHOLD = 0.45   # boundary_y_rel < this  →  inverted
                             # in between             →  tilted
