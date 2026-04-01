# dVRK Data Collection Tools

**Author:** Shujiro  
**Project:** dVRK surgical robot data collection and analysis  
**Duration:** 6 months (March 2026 - September 2026)

---

## Overview

Real-time multimodal surgical performance assessment system for the da Vinci Research Kit (dVRK).

**Components:**
- 📹 **Camera streaming** (30 fps stereo HD video)
- ⏱️ **Task timing** with visual feedback GUI
- 🤖 **Kinematics tracking** (PSM1, PSM2, ECM positions)
- 📊 **Performance metrics** (path length, smoothness, scoring)
- 🎥 **Data recording** (compressed bags, video export)

**Goal:** Real-time surgical skill assessment using computer vision (YOLO, DeepLabCut) and robot kinematics analysis.

---

## Quick Start

### Daily Workflow

**1. Setup displays (once at boot):**
```bash
cd ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts
./setup-monitors.sh
```

**2. Start camera streaming:**
```bash
./camera-stream-compressed-transport.sh
```
- Publishes at 30 fps (compressed)
- Keep this terminal running!

**3. Launch task timer GUI (optional):**
```bash
cd ~/dvrk_shujiro_ws
ros2 run dvrk_shujiro task_timer_gui
```

**4. View cameras (new terminal):**
```bash
rqt
# Plugins → Visualization → Image View
# Select: /camera_left/compressed
```

**5. Record session:**
```bash
cd ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts
./record-surgery-compressed.sh
# Enter recording name
# Perform task...
# Press Ctrl+C when done
```

---

## Tools & Scripts

### Task Timer GUI

Floating semi-transparent timer window that tracks time and total path length when MONO pedal is pressed.

**Features:**
- Dual-window display for stereo endoscope viewing
- Color-coded progress bar (green → yellow → red)
- Only counts active manipulation time (MONO pedal pressed)
- Configurable time limit (default: 2 minutes)
- Real-time path length calculation

**Usage:**
```bash
ros2 run dvrk_shujiro task_timer_gui
```

**Topics subscribed:**
- `/console/teleop/enabled` - Detects MONO pedal state
- `/PSM1/measured_cp` - Left tool position
- `/PSM2/measured_cp` - Right tool position

---

### Camera System (30 fps)

**Start cameras:**
```bash
./scripts/camera-stream-compressed-transport.sh
```

**Published topics:**
- `/camera_left/compressed` ⭐ (30 fps, JPEG)
- `/camera_right/compressed` ⭐ (30 fps, JPEG)
- `/camera_left/image_raw` (19 fps, full quality)
- `/camera_right/image_raw` (19 fps, full quality)

**Performance:**
- Streaming: 30 fps compressed
- Recording: ~50-100 MB/minute
- Quality: JPEG 90% (excellent for ML)

**See:** `scripts/README.md` for detailed camera documentation

---

### Recording System

**Record compressed data:**
```bash
./scripts/record-surgery-compressed.sh
```

**What gets recorded:**
- Camera streams (30 fps compressed)
- Robot kinematics (PSM1, PSM2, ECM)
- Task metrics

**Storage:** `~/dvrk_recordings/compressed/`

**File size:** ~100 MB/minute (vs 1.8 GB/min for raw)

---

### Video Conversion

**Convert ROS bag to MP4:**
```bash
./scripts/compressed-bag-to-mp4.sh <recording_name> both
```

**Output:**
- `~/dvrk_recordings/compressed/<name>_left.mp4`
- `~/dvrk_recordings/compressed/<name>_right.mp4`

**Ready for:** YOLO annotation, DeepLabCut training

---

## Project Structure

```
dvrk_shujiro/
├── dvrk_shujiro/          # Python package
│   ├── gui/               # Task timer interface
│   ├── metrics/           # Performance metrics
│   ├── nodes/             # ROS2 nodes
│   └── utils/             # Utilities (quaternion math)
├── scripts/               # Camera & recording scripts
│   ├── camera-stream-compressed-transport.sh
│   ├── record-surgery-compressed.sh
│   ├── compressed-bag-to-mp4.sh
│   ├── setup-monitors.sh
│   ├── README.md          # Detailed camera docs
│   └── archive/           # Old experimental scripts
├── docs/                  # Documentation
│   ├── camera_setup.md
│   └── session_notes.md
└── README.md              # This file
```

---

## Hardware

- **Robot:** da Vinci Si surgical system (dVRK modified)
- **Cameras:** Stereo HD endoscope (1920×1080 @ 30 fps)
- **Capture:** Blackmagic DeckLink video cards
  - Device 0 = Right camera
  - Device 1 = Left camera
- **Displays:** DP-2, DP-0 manipulation screens (640×480 @ 59.94Hz)

---

## Software Stack

- **ROS2:** Jazzy
- **Vision:** gscam, compressed_image_transport
- **GUI:** PyQt5 (task timer)
- **Video:** GStreamer, ffmpeg
- **ML (planned):** YOLO, DeepLabCut

---

## Dependencies

### Install required packages
```bash
sudo apt install \
  ros-jazzy-gscam \
  ros-jazzy-compressed-image-transport \
  ros-jazzy-image-transport-plugins \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  ffmpeg \
  vlc
```

---

## Development Roadmap

### ✅ Phase 1: Camera System (Complete)
- [x] Camera streaming at 30 fps
- [x] Compressed recording pipeline
- [x] Bag-to-video conversion
- [x] Performance optimization

### ✅ Phase 2: Task Metrics (Complete)
- [x] Task timer GUI
- [x] Path length tracking
- [x] Active time measurement
- [x] Visual feedback

### 🔄 Phase 3: Kinematics Analysis (In Progress)
- [ ] Smoothness metrics
- [ ] Path efficiency
- [ ] Tremor detection
- [ ] Workspace violations

### 📋 Phase 4: Vision ML (Next)
- [ ] YOLO integration for tool detection
- [ ] DeepLabCut for pose estimation
- [ ] Real-time inference pipeline
- [ ] Stereo 3D reconstruction

### 📋 Phase 5: Assessment System
- [ ] Multimodal scoring engine
- [ ] Real-time feedback
- [ ] Performance visualization
- [ ] Data analysis tools

---

## Troubleshooting

### Cameras not working
```bash
pkill -f gscam
./camera-stream-compressed-transport.sh
```

### Low frame rate
- Use `/camera_left/compressed` (30 fps) ✅
- NOT `/camera_left/image_raw` (only 19 fps) ❌

### GUI not showing
```bash
# Make sure ROS2 is sourced
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run dvrk_shujiro task_timer_gui
```

### Recording has no video
- Start cameras BEFORE recording
- Check: `ros2 bag info <recording>` shows compressed topics

**See:** `scripts/README.md` and `docs/` for detailed troubleshooting

---

## Data Management

### Storage Requirements
- **Per recording:** ~100 MB/min compressed
- **Per video:** ~10-20 MB/min (MP4)
- **Recommended:** 1-2 TB external drive for 100+ recordings

### File Organization
```
~/dvrk_recordings/compressed/
├── 2026-03-31_trial_01/
├── 2026-03-31_trial_01_left.mp4
├── 2026-03-31_trial_01_right.mp4
└── ...
```

---

## Session History

- **2026-03-23:** Initial camera setup, brightness troubleshooting
- **2026-03-31:** Optimized to 30 fps, compressed recording system

See `docs/session_*.md` for detailed session notes

---

## License

Research use only - Stanford dVRK Team

---

## Acknowledgments

- **dVRK Community:** Open-source surgical robotics platform
- **Mentor:** Camera system optimization guidance
- **Previous Students:** Baseline camera configuration

---

## Contact

dVRK Research Team - Stanford University
Shujiro Shobayashi
```
