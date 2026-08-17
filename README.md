# dVRK Data Collection Tools

**Author:** Shujiro  
**Project:** dVRK surgical robot data collection and analysis  
**Duration:** 6 months (March 2026 - September 2026)

---

## Overview

Real-time multimodal performance assessment system for the da Vinci Research Kit (dVRK).

**Components:**
- 📹 **Camera streaming** (30 fps stereo HD video)
- ⏱️ **Task timing** with visual feedback GUI
- 🤖 **Kinematics tracking** (PSM1, MTMR)
- 📊 **Performance metrics** (path length, smoothness, scoring)
- 🎥 **Data recording** (compressed bags, video export)

**Goal:** Real-time surgical skill assessment using computer vision and robot kinematics analysis.

---

## Quick Start

### Daily Workflow

**1. Start the dVRK console / teleop** (separate workspace):
```bash
source /opt/ros/jazzy/setup.bash
source ~/dvrk_ros2_new_ws/install/setup.bash
cd ~/dvrk_ros2_new_ws/src/dvrk/dvrk_config_stanford
ros2 run dvrk_robot dvrk_system -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-Teleop.json
```
Home PSM1/MTMR from the console before continuing.

> Note: the console now lives in `dvrk_ros2_new_ws`, not the old `ros2_ws`.
> The SRC dVRK no longer relays PSM1/MTMR over Bluetooth — the dESSJ/Arduino
> BLE boards were replaced with native ESSJ + new dSIB boards — so this is
> the only way to bring the arms up.

**2. Start camera streaming (new terminal, keep running):**
```bash
bash ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts/camera-stream-compressed-transport.sh
```
Publishes `/camera_left/compressed` and `/camera_right/compressed` at ~30 fps.

**3. View cameras (new terminal):**
```bash
bash ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts/view_cameras.sh
```
Opens the stereo viewer window(s) directly — this replaced the old rqt
Image View workflow.

**4. Record session:**
```bash
cd ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts
./record-surgery-safe.sh
# Enter recording name
# Perform task...
# Press Ctrl+C when done
```
`record-surgery-safe.sh` is the current recording script (replaces
`record-surgery-compressed.sh` — see [Recording System](#recording-system)
below for why).

---

## Tools & Scripts

### Task Timer GUI

Floating semi-transparent timer window that tracks time, path length, and
orientation-change rate for a trial. Two ways to trigger a trial (chosen at
launch, see below): MONO pedal + teleop, or an Arduino peg board.

**Features:**
- Dual-window display for stereo endoscope viewing
- Color-coded progress bar (green → yellow → red)
- Configurable time limit (default: 2 minutes, `config.py: MAX_TIME_SEC`) —
  timing out marks the trial **failed**, logs the summary, resets, and
  waits for the next trial (the windows never close on their own)
- Real-time path length (mm) shown in the GUI, per arm (PSM1/PSM2)
- Orientation angular displacement + rate — **printed to the terminal
  once per second while running, not drawn in the GUI window**
- Optional Arduino-driven force sensor + YOLO cylinder-orientation check
  (see [below](#force-sensor--orientation-option-arduino-trigger-only))

**How to run:**
```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/dvrk_shujiro_ws/src/dvrk_shujiro
python3 -m dvrk_shujiro.main
```

**Session settings prompt:** on every launch, before the timer windows
open, the terminal asks you to accept or override a few settings —
press Enter on each to take the `config.py` default:
- **Trial trigger** — `mono_teleop` (MONO pedal + teleop enabled) or
  `arduino` (cylinder lift/place on the peg board)
- **Arduino trial end** (only asked if trigger is `arduino`) —
  `target_placed` (ends when placed on the target peg) or
  `return_to_center` (ends only once carried back to the center peg)
- **Report path length?** / **Report angular displacement / orientation
  rate?** — turns those sections of the terminal output on/off
- **Enable force sensor?** / **Enable YOLO cylinder-orientation check?**
  (only asked if trigger is `arduino`, independent of each other) — see
  [Force sensor + orientation
  option](#force-sensor--orientation-option-arduino-trigger-only) below

These are one-shot choices for that run only. To change the *defaults* so
you don't have to re-type them every time, edit `TRIGGER_SOURCE`,
`ARDUINO_END_MODE`, `TRACK_PATH_LENGTH`, `TRACK_ORIENTATION`,
`ENABLE_FORCE_SENSOR`, `ENABLE_ORIENTATION_CHECK` in `config.py` — same
single-source-of-truth file as `MAX_TIME_SEC`.

**MONO+teleop trigger:** requires the dVRK console/teleop stack already
running with PSM1 and PSM2 homed (see [Quick Start](#quick-start) step 1),
so `/console/teleop/enabled`, `/console/operator_present`,
`/PSM1/measured_cp`, `/PSM2/measured_cp` are all publishing.

**Arduino trigger mode:** instead of MONO+teleop, the trial starts when
the Arduino reports the cylinder lifted off the center peg (`LIFTED`) and
ends per the `arduino_end_mode` you chose. Requires the peg board wired up
and reachable at `ARDUINO_PORT` (`config.py`). Type `s` + Enter in the
same terminal to start an Arduino trial block, `q` + Enter to abort it.
(If you just want the standalone Arduino trial popup — without ROS/PSM
tracking — that's still available separately via `python3
dvrk_shujiro/arduino/read_arduino_with_popup.py`, which has the same
target-placed/return-to-center prompt.)

**Usage:**
1. Start the script above, answer the session-settings prompt — two
   floating timer windows appear.
2. **MONO+teleop:** on the dVRK console GUI, enable teleoperation, then
   press and hold the MONO pedal and start moving.
   **Arduino:** type `s` + Enter, then lift the cylinder off the center peg.
   Either way, the clock and progress bar start.
3. Watch the **terminal** (not the GUI window) for live path length +
   orientation rate, logged once per second, e.g.:
   ```
   [12.0s @ 199.8Hz] Path: R=145mm L=98mm | Orient: R=32° (2.8°/s) L=21° (1.9°/s)
   ```
4. The trial ends one of these ways — either way the timer windows stay
   open, reset, and wait for the next trial:
   - **Time runs out** (`MAX_TIME_SEC`) — marked **failed**.
   - **MONO+teleop:** you disable teleoperation on the console GUI before
     time is up — marked complete. (Releasing MONO without disabling
     teleop only *pauses* the clock — it resumes if you press MONO again
     while teleop is still enabled.)
   - **Arduino:** the configured end event (placement, or placement +
     return) happens before time is up — marked complete.
5. Either way, the terminal prints the trial summary: **duration, path
   length (PSM1/PSM2 + total), angular displacement, and average
   orientation rate** (whichever sections you enabled at the prompt).

**Known gotcha (MONO+teleop only):** if the clock stops responding to MONO
presses mid-session (pedal held but bar doesn't move), toggle teleop **OFF
then ON** again on the console. This resets the enable state and the GUI
starts responding to MONO again — this is a console-side state issue, not
a bug in this code.

#### Force sensor + orientation option (Arduino trigger only)

Two **independent** yes/no prompts — enable either alone or both together:

**Force sensor** (`ENABLE_FORCE_SENSOR`) — live ATI force capture: peak ‖F‖
(N), and how many times the force crosses above `FORCE_THRESHOLD_N`
(`config.py`, default 5N — a count of distinct contact events, not an
average, since the signal is mostly near zero with occasional contact
spikes) — over the trial window, matching `ARDUINO_END_MODE`: just
lift → place with `target_placed`, or the full lift → place → pickup →
return cycle with `return_to_center`. The raw signal is smoothed first
(`FORCE_SMOOTHING_WINDOW`, default 5 samples) to cut sensor noise before
computing peak/crossings and plotting — see `config.py` to tune both.
Needs only the ATI receiver running:
```bash
# ATI force sensor UDP → ROS2 bridge (publishes /ati_sensor/wrench)
python3 scripts/ati_sensor_udp_receiver.py
```
No camera, no YOLO venv — `cv2`/`ultralytics` are never imported if this is
the only one you enable.

**YOLO cylinder-orientation check** (`ENABLE_ORIENTATION_CHECK`) — checks
whether the cylinder was placed **upright** and **the correct color facing
down**, cropped from the left camera around the target peg, triggered off
the Arduino's `Target hit` line. Needs the camera stream running, and this
terminal (the one running `python3 -m dvrk_shujiro.main`) to be in the YOLO
venv — `cv2`/`ultralytics` are only imported when this option is enabled:
```bash
# Camera stream (publishes /camera_left/compressed)
./scripts/camera-stream-compressed-transport.sh

# THIS terminal must be in the YOLO venv, not the plain ROS2 environment:
source ~/ros2_yolo_venv/bin/activate
```

Whichever of the two you enable, a result popup (force graph and/or
orientation text — whichever's active) appears on all monitors at the same
moment `ARDUINO_END_MODE` ends the trial itself — right at placement with
`target_placed`, or once the cylinder is carried back to the center peg with
`return_to_center`. A CSV log (`FORCE_ORIENTATION_CSV` in `config.py`,
default `force_orientation_data.csv`) records peak force, threshold-crossing
count, orientation angle, and placement success per trial either way (columns for whichever
capability is off are left blank, not zero) — that row is always written
once the Arduino's full physical cycle finishes, regardless of `ARDUINO_END_MODE`.

**Note:** with `target_placed` + orientation check both enabled, the popup
doesn't appear the instant the trial ends — it waits (a short beat, up to
~`ORIENTATION_DELAY_SEC` + retries) for the YOLO orientation result to
resolve first, so the force graph and orientation result always show up
together rather than the orientation field appearing blank.

**Topics subscribed:**
- `/PSM1/measured_cp`, `/PSM2/measured_cp` (`geometry_msgs/PoseStamped`) —
  arm kinematics, always subscribed (feed path length / orientation tracking
  regardless of trigger source)
- `/console/teleop/enabled` (`std_msgs/Bool`), `/console/operator_present`
  (`sensor_msgs/Joy`, MONO pedal in `buttons[0]`) — only subscribed in
  `mono_teleop` trigger mode
- `/ati_sensor/wrench` (`geometry_msgs/WrenchStamped`) — only subscribed if
  the force sensor option is enabled
- `/camera_left/compressed` (`sensor_msgs/CompressedImage`) — only
  subscribed if the YOLO orientation-check option is enabled

---

### Camera System (30 fps)

**Start cameras:**
```bash
./scripts/camera-stream-compressed-transport.sh
```

**View cameras:**
```bash
./scripts/view_cameras.sh
```
Opens a native OpenCV stereo viewer (`view_cameras.py`), positioned across
the manipulation displays (DP-2 / DP-0) plus a supervisor monitor. Press
`q` or `Esc` to quit. (`view_cameras_bi.sh` is a variant for a two-window,
no-supervisor-monitor layout.)

**Published topics:**
- `/camera_left/compressed` ⭐ (30 fps, JPEG)
- `/camera_right/compressed` ⭐ (30 fps, JPEG)
- `/camera_left/image_raw` (19 fps, full quality)
- `/camera_right/image_raw` (19 fps, full quality)

**Performance:**
- Streaming: 30 fps compressed
- Recording: ~50-100 MB/minute
- Quality: JPEG 90% (excellent for ML)

---

### Recording System

**Record (current script):**
```bash
./scripts/record-surgery-safe.sh
```

`record-surgery-safe.sh` replaced `record-surgery-compressed.sh`. Differences:
- Verifies every required topic has a live publisher (15s timeout) before
  recording starts — fails fast with a clear message instead of silently
  recording camera-only footage if the console isn't up / arms aren't homed.
- Actually records `/camera_right/compressed` (the old script listed it in
  its printed summary but never passed it to `ros2 bag record`).
- Sources `~/dvrk_ros2_new_ws` instead of `~/ros2_ws`, matching the console
  now living in the new workspace (see [Quick Start](#quick-start) step 1).

**What gets recorded:**
- `/camera_left/compressed`, `/camera_right/compressed` (30 fps compressed)
- `/PSM1/measured_cp`, `/PSM1/measured_cv`, `/PSM1/measured_js`, `/PSM1/jaw/measured_js`
- `/MTMR/measured_cp`, `/MTMR/measured_cv`, `/MTMR/measured_js`, `/MTMR/gripper/measured_js`

Note this is PSM1 + MTMR only — PSM2/ECM are not currently recorded.

**Storage:** `~/dvrk_recordings/training/`

**File size:** ~50-100 MB/minute (vs ~1.8 GB/min for raw)

---

### Video Conversion

**Convert ROS bag to MP4:**
```bash
./scripts/compressed-bag-to-mp4.sh <recording_name> both
```

**Output:**
- `~/dvrk_recordings/compressed/<name>_left.mp4`
- `~/dvrk_recordings/compressed/<name>_right.mp4`

> ⚠️ **Known mismatch:** this script still looks for the bag under
> `~/dvrk_recordings/compressed/<name>`, but `record-surgery-safe.sh` saves
> to `~/dvrk_recordings/training/<name>`. Move or symlink the bag into
> `~/dvrk_recordings/compressed/` first, or point `BAG_DIR` in the script
> at `~/dvrk_recordings/training/` before running.

**Ready for:** YOLO annotation, DeepLabCut training

---

## Project Structure

```
dvrk_shujiro/
├── dvrk_shujiro/          # Python package
│   ├── gui/               # Task timer interface (Tkinter)
│   ├── metrics/           # Performance metrics
│   ├── nodes/             # ROS2 nodes
│   └── utils/             # Utilities (quaternion math)
├── scripts/               # Camera, recording & ML pipeline scripts
│   ├── camera-stream-compressed-transport.sh
│   ├── view_cameras.sh / view_cameras.py
│   ├── record-surgery-safe.sh     # current recording script
│   ├── compressed-bag-to-mp4.sh
│   ├── 1_extract_frames.py … 8_multi_cylinder_track.py   # ML pipeline
│   └── archive/            # old experimental scripts
└── README.md               # This file
```

---

## Hardware

- **Robot:** da Vinci Si surgical system (dVRK modified)
- **Cameras:** Stereo HD endoscope (1920×1080 @ 30 fps)
- **Capture:** Blackmagic DeckLink video cards
  - Device 0 = Right camera
  - Device 1 = Left camera
- **Arm comms:** native ESSJ + dSIB boards (Bluetooth/BLE dESSJ+Arduino path
  has been removed)

---

## Software Stack

- **ROS2:** Jazzy
- **Console/teleop:** `dvrk_robot` (`dvrk_ros2_new_ws`, separate workspace from this package)
- **Vision:** gscam, compressed_image_transport
- **GUI:** Tkinter (task timer)
- **Video:** GStreamer, ffmpeg, OpenCV
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

### Console / arms not homing, or `record-surgery-safe.sh` aborts on missing topics
The dVRK console likely isn't up yet. In another terminal:
```bash
source /opt/ros/jazzy/setup.bash
source ~/dvrk_ros2_new_ws/install/setup.bash
cd ~/dvrk_ros2_new_ws/src/dvrk/dvrk_config_stanford
ros2 run dvrk_robot dvrk_system -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-Teleop.json
```
Home PSM1/MTMR, confirm with `ros2 topic list`, then retry.

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
# Make sure ROS2 is sourced and conda is deactivated
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/dvrk_shujiro_ws/src/dvrk_shujiro
python3 -m dvrk_shujiro.main
```

### Timer bar doesn't move even though MONO is held down
Teleop enable state on the console got stuck. Toggle teleop **OFF then ON**
again on the console. The GUI will start responding to MONO again
immediately. (The GUI logs `⏱️ Timer STARTED (MONO pressed)` to its
terminal the moment it correctly detects both conditions; if that line
never appears, this is almost always the cause.)

### Recording has no video
- Start cameras BEFORE recording
- Check: `ros2 bag info <recording>` shows compressed topics

### `compressed-bag-to-mp4.sh` says recording not found
It looks in `~/dvrk_recordings/compressed/`, but `record-surgery-safe.sh`
saves to `~/dvrk_recordings/training/`. See
[Video Conversion](#video-conversion) above.

---

## Data Management

### Storage Requirements
- **Per recording:** ~50-100 MB/min compressed
- **Per video:** ~10-20 MB/min (MP4)
- **Recommended:** 1-2 TB external drive for 100+ recordings

### File Organization
```
~/dvrk_recordings/training/
├── trial_01_compressed/
└── ...
~/dvrk_recordings/compressed/        # bag-to-mp4 output
├── trial_01_compressed_left.mp4
├── trial_01_compressed_right.mp4
└── ...
```

---

## Session History

- **2026-03-23:** Initial camera setup, brightness troubleshooting
- **2026-03-31:** Optimized to 30 fps, compressed recording system

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
