"""Configuration constants for dVRK timer"""

# Timer settings
TIMER_RATE_HZ = 200
TIMER_INTERVAL = 1.0 / TIMER_RATE_HZ
MAX_TIME_SEC = 30

# GUI settings

WINDOW_ALPHA = 0.1
WINDOW_WIDTH = 340
WINDOW_HEIGHT = 95

# Display settings
FONT_STATUS = ("Arial", 8)
FONT_TIME = ("Arial", 18, "bold")
FONT_PATH = ("Arial", 10)

# Colors
COLOR_GREEN = "green"
COLOR_ORANGE = "orange"
COLOR_RED = "red"
COLOR_PSM1 = "blue"
COLOR_PSM2 = "purple"

# Progress thresholds (%)
PROGRESS_YELLOW_THRESHOLD = 70
PROGRESS_RED_THRESHOLD = 90

# ROS topics
TOPIC_TELEOP_ENABLED = '/console/teleop/enabled'
TOPIC_OPERATOR_PRESENT = '/console/operator_present'
TOPIC_PSM1_POSE = '/PSM1/measured_cp'
TOPIC_PSM2_POSE = '/PSM2/measured_cp'

# ── Arduino settings ──────────────────────────────────────────────────────────
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600
ARDUINO_CSV  = "arduino_data.csv"

# ── Session defaults (main.py asks once at startup, Enter accepts these) ──────
# TRIGGER_SOURCE: "mono_teleop" -> trial starts/stops on teleop-enabled + MONO pedal
#                 "arduino"     -> trial starts on Arduino 'LIFTED', ends per ARDUINO_END_MODE
TRIGGER_SOURCE = "mono_teleop"

# ARDUINO_END_MODE (only used when TRIGGER_SOURCE == "arduino"):
#   "target_placed"    -> trial ends when the cylinder is placed on the target peg
#   "return_to_center" -> trial ends only once it's carried back to the center peg
ARDUINO_END_MODE = "target_placed"

# Which metrics to compute/report each trial
TRACK_PATH_LENGTH = True
TRACK_ORIENTATION = True

# ── Force sensor + YOLO cylinder-orientation options (only used when
#    TRIGGER_SOURCE == "arduino") ──────────────────────────────────────────────
# These are independent — enable either alone or both together:
#
# ENABLE_FORCE_SENSOR: live ATI force capture — peak ‖F‖ and how many times
#   it crosses FORCE_THRESHOLD_N per trial.
#   Needs: ati_sensor_udp_receiver.py running (publishes /ati_sensor/wrench).
#   Does NOT need cv2/ultralytics or the YOLO venv.
#
# ENABLE_ORIENTATION_CHECK: YOLO-based "was the cylinder placed upright, with
#   the correct color facing down?" check, cropped from the left camera.
#   Needs: camera-stream-compressed-transport.sh running, AND this terminal
#   itself running in the YOLO venv (cv2/ultralytics are only imported when
#   this is actually enabled — see read_arduino_with_force.py's docstring).
#
# Both are triggered off the same Arduino serial stream as the trial trigger.
# Note: results are captured over the Arduino's FULL lift -> place -> pickup
# -> return cycle regardless of ARDUINO_END_MODE, so with
# ARDUINO_END_MODE="target_placed" this summary prints a little after that
# trial's timer/GUI summary already did.
ENABLE_FORCE_SENSOR = False
ENABLE_ORIENTATION_CHECK = False

TOPIC_WRENCH = '/ati_sensor/wrench'
TOPIC_CAMERA_LEFT = '/camera_left/compressed'

# scripts/6_cylinder_orientation.py is loaded from here at runtime (it isn't
# an importable package) and YOLO_WEIGHTS defaults under here too.
SCRIPTS_DIR = '/home/stanford/dvrk_shujiro_ws/src/dvrk_shujiro/scripts'
YOLO_WEIGHTS = SCRIPTS_DIR + '/models/best_v2.pt'
YOLO_CONF = 0.5
YOLO_IMGSZ = 320

ORIENTATION_TOLERANCE_DEG = 20.0    # |angle| <= this many degrees from upright (0°) counts as OK
ORIENTATION_DELAY_SEC = 0.5         # wait after "Target hit" before cropping, so the piece settles
ORIENTATION_MAX_ATTEMPTS = 3        # retry if no cylinder is detected (e.g. hand/gripper in the way)
ORIENTATION_RETRY_DELAY_SEC = 0.5

# Outer peg index (1-8) -> which half of the left-camera frame it's in
PEG_CLASSIFICATIONS = {1: "L", 2: "L", 3: "L", 4: "L", 5: "R", 6: "R", 7: "R", 8: "R"}

# Trailing moving-average window (samples) applied to ‖F‖ before computing
# peak/crossings and before plotting — smooths out sensor/electrical noise.
# Applied once, so the graph, the values logged to the terminal, and the
# CSV row all reflect the same smoothed signal. 1 = no smoothing. Larger
# values cut more noise but also flatten brief genuine force spikes.
FORCE_SMOOTHING_WINDOW = 5

# Force threshold (Newtons) used to count contact events per trial — how
# many times the (smoothed) force rises above this value, rather than an
# average force (which isn't a meaningful number for a signal that's mostly
# near zero with occasional contact spikes).
FORCE_THRESHOLD_N = 5.0

# ── Force/orientation result popup display (the window shown on all monitors
#    at end of trial) ───────────────────────────────────────────────────────
FORCE_POPUP_SHOW_SEC = 8.0   # how long it stays visible before auto-hiding
FORCE_POPUP_MARGIN_X = 100    # offset (pixels) from each monitor's LEFT edge
FORCE_POPUP_MARGIN_Y = 100    # offset (pixels) from each monitor's TOP edge

FORCE_ORIENTATION_CSV = "force_orientation_data.csv"

# Measured (not assumed) latency log — one row per orientation-check, per
# force-popup, and per state-detection event, tagged with which capabilities
# were enabled that run so "force+orientation together" vs "orientation
# alone" (etc.) can be compared later. See scripts/9_analyze_timing.py.
TIMING_LOG_CSV = "timing_log.csv"

# ── Cylinder state detection (STATIONARY/HELD/DROPPED/LOST) ────────────────
# Independent of TRIGGER_SOURCE — only needs /camera_left/compressed and
# /PSM1/measured_cp, both already used regardless of trigger. Always uses
# the trained XGBoost classifier (see scripts/7_2_train_xgboost.py) — no
# rule-based fallback. LOST is still a deterministic "not detected for
# STATE_LOST_FRAMES frames" rule, never model-predicted — see
# state_detection/cylinder_state_tracker.py for why (same reasoning as the
# fix in scripts/7_velocity_check.py).
ENABLE_STATE_DETECTION = False

# Trained model to load — retrain with scripts/7_2_train_xgboost.py and
# point this at the new file when you have a better one.
STATE_XGB_MODEL_PATH = SCRIPTS_DIR + '/models/cylinder_state_xgb_v2.json'
STATE_XGB_WINDOW = 5          # lag window — MUST match the value used to train STATE_XGB_MODEL_PATH
STATE_VEL_WINDOW = 5          # rolling-average window (frames) for velocity smoothing
STATE_MAX_JUMP = 800.0        # px/s — reject detections implying an impossible jump (spurious/misdetection)
STATE_LOST_FRAMES = 10        # consecutive missed detections before -> LOST

# Cross-process signal file for scripts/depth_camera/09_live_trial_monitor.py
# (only relevant if you also run that script) — see read_arduino_with_force.py
# module docstring for why this exists.
TRIAL_SIGNAL_PATH = '/tmp/dvrk_depth_trial_signal.jsonl'
