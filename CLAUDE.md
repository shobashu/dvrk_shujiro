# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS2 (Jazzy) Python package for real-time multimodal surgical performance assessment on the da Vinci Research Kit (dVRK). The package records and analyzes stereo camera streams, robot kinematics (PSM1/PSM2/ECM), and task timing data. Deactivate conda before running ROS2 commands.

## Build & Run

**Build the ROS2 workspace:**
```bash
cd ~/dvrk_shujiro_ws
colcon build --packages-select dvrk_shujiro
source install/setup.bash
```

**Run the task timer GUI (main entry point):**
```bash
# Via ROS2 (after colcon build):
ros2 run dvrk_shujiro task_timer_gui

# Directly (no build required):
cd ~/dvrk_shujiro_ws/src/dvrk_shujiro
python3 -m dvrk_shujiro.main
```

**Run YOLO detection node:**
```bash
ros2 run dvrk_shujiro detect_node \
    --ros-args \
    -p weights:=models/dvrk_v1/weights/best.pt \
    -p image_topic:=/jhu_crsus/left/image_raw \
    -p conf_threshold:=0.45
```

**ML pipeline scripts (run from repo root):**
```bash
# 1. Extract frames from a ROS bag recording
python3 scripts/1_extract_frames.py \
    --bag compressed/test6 \
    --out ~/dvrk_shujiro_ws/data/frames/trial_001 \
    --fps 5

# 2. Unzip labeled data + raw frames, auto-label with best.pt, split train/val/test
#    Inputs: Task_Pad_cylinder_pegs.yolov8.zip (Roboflow export) + remaining_frames.zip
#    Outputs: data/dataset/  and  config/dataset.yaml
python3 scripts/2_prepare_dataset.py \
    --labeled Task_Pad_cylinder_pegs.yolov8.zip \
    --frames  remaining_frames.zip \
    --weights models/dvrk_v1/weights/best.pt \
    --out     data/dataset

# 3. Train YOLOv8 (reads config/dataset.yaml written by step 2)
python3 scripts/3_train.py
```

## Tests

```bash
cd ~/dvrk_shujiro_ws
colcon test --packages-select dvrk_shujiro
# Tests cover: copyright, flake8, pep257
```

## Camera & Recording

```bash
# Start camera streams (keep running):
./scripts/camera-stream-compressed-transport.sh

# Record a session:
./scripts/record-surgery-compressed.sh        # prompts for name

# Convert ROS bag → MP4:
./scripts/compressed-bag-to-mp4.sh <name> both

# Run + record in one step:
./scripts/run_and_record_raw.sh
```

Recordings land in `~/dvrk_recordings/compressed/`. Use `/camera_left/compressed` (30 fps) not `/camera_left/image_raw` (19 fps) for viewing and recording.

## Architecture

The system runs two parallel execution contexts that share state via the `TimerGUI` object:

- **Main thread** — Tkinter GUI (`TimerGUI` / `TimerWindow` in `dvrk_shujiro/gui/`). Blocked on `mainloop()`. Display updates poll every 100 ms via `root.after()`.
- **Background thread** — `rclpy.spin()` launched by `TaskTimerNode.start_spinning()`. Callbacks update GUI attributes directly (no lock — Tkinter is not thread-safe; updates are read-only scalars so this works in practice).

**Data flow:**
1. ROS topics → `TaskTimerNode` callbacks (background thread)
2. Node callbacks → update `TimerGUI` scalar attributes (path lengths, angular metrics)
3. `TimerGUI._update_display()` reads those attributes every 100 ms (main thread)
4. `TaskTimerNode.update_timer()` fires at 200 Hz via `create_timer()`, calls `gui.tick(dt)` to advance elapsed time

**Key classes:**
- `TimerGUI` (`gui/timer_window.py`) — dual-window display manager; owns `is_running`, `elapsed`, and all accumulated metrics
- `TaskTimerNode` (`nodes/task_timer_node.py`) — ROS node; owns two `MetricsTracker` instances (PSM1/PSM2)
- `MetricsTracker` (`metrics/metrics_tracker.py`) — stateful accumulator for path length (Euclidean sum) and angular displacement (quaternion diff via `quaternion_math.py`)
- `DetectNode` (`nodes/detect_node.py`) — standalone YOLO inference node; subscribes to raw image, publishes annotated image + JSON detections
- `YoloDetector` (`camera/yolo_detector.py`) — thin Ultralytics YOLO wrapper; can be used offline without ROS

**ROS topics subscribed by `TaskTimerNode`:**
| Topic | Type | Purpose |
|---|---|---|
| `/console/teleop/enabled` | `Bool` | Enables/disables tracking; logs trial on disable |
| `/console/operator_present` | `Joy` | MONO pedal state (`buttons[0]`) |
| `/PSM1/measured_cp` | `PoseStamped` | Right arm kinematics |
| `/PSM2/measured_cp` | `PoseStamped` | Left arm kinematics |

**Timer/GUI legacy variants** (in `dvrk_shujiro/launch/`): `task_timer.py`, `task_timer_bar.py`, `task_timer_path_gui.py`, etc. are older standalone scripts. `main.py` + the `nodes/` + `gui/` refactor is the current architecture.

## Configuration

All tuneable constants live in `dvrk_shujiro/config.py`:
- `MAX_TIME_SEC` — trial duration (default 120 s)
- `TIMER_RATE_HZ` — ROS timer frequency (default 200 Hz)
- `ARDUINO_PORT` — serial port for Arduino LED controller (default `/dev/ttyACM1`)
- ROS topic names
- GUI color thresholds and window dimensions

## Hardware notes

- Blackmagic DeckLink: device 0 = right camera, device 1 = left camera
- Displays: DP-2, DP-0 at 640×480 @ 59.94 Hz (set via `scripts/setup-monitors.sh`)
- Arduino on `/dev/ttyACM1` controls peg LEDs; Arduino reader and trial popup are commented out in `main.py` but code exists in `dvrk_shujiro/arduino/`

## Dependencies

System packages (Ubuntu/ROS2 Jazzy):
```bash
sudo apt install ros-jazzy-gscam ros-jazzy-compressed-image-transport \
  ros-jazzy-image-transport-plugins gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad ffmpeg vlc
```

Python packages: `pip install -r requirements.txt` (numpy, opencv, rosbags, PyQt5, ultralytics when using YOLO)
