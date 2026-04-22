#!/usr/bin/env bash
# run-and-record-raw.sh
# Starts both cameras (raw) and records raw (+ optional robot topics).
# Usage: ./run_and_record_raw.sh

###  Install window control tool --> Notes: sudo apt-get install -y wmctrl

set -e

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

read -p "Enter recording name: " NAME
RECORDINGS_DIR=~/dvrk_recordings/training_raw
mkdir -p "$RECORDINGS_DIR"

# Monitor positions (from your xrandr)
# DP-2: +0+0  640x480
# DP-0: +640+0 640x480
LEFT_WIN_X=0
LEFT_WIN_Y=0
LEFT_WIN_W=640
LEFT_WIN_H=480

RIGHT_WIN_X=640
RIGHT_WIN_Y=0
RIGHT_WIN_W=640
RIGHT_WIN_H=480

cleanup() {
  echo ""
  echo "Stopping..."
  kill ${BAG_PID:-} ${LEFT_VIEW_PID:-} ${RIGHT_VIEW_PID:-} ${RIGHT_CAM_PID:-} ${LEFT_CAM_PID:-} 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Saved to: $RECORDINGS_DIR/$NAME"
}
trap cleanup SIGINT SIGTERM

echo "Starting right camera (device 0)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_right \
  -r camera/image_raw:=camera_right/image_raw \
  -r camera/camera_info:=camera_right/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=0 ! videoconvert ! videobalance brightness=0.1 contrast=1.2 saturation=1.5 hue=0.0" \
  -p camera_name:=camera_right \
  -p frame_id:=camera &
RIGHT_CAM_PID=$!

echo "Starting left camera (device 1)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_left \
  -r camera/image_raw:=camera_left/image_raw \
  -r camera/camera_info:=camera_left/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=1 ! videoconvert ! videobalance brightness=0.1 contrast=1.2 saturation=1.5 hue=0.0" \
  -p camera_name:=camera_left \
  -p frame_id:=camera &
LEFT_CAM_PID=$!

echo "Waiting for camera topics..."
sleep 3

################################################################################################
# Start viewers (titles usually include "showimage")
echo "Starting viewers..."
ros2 run image_tools showimage --ros-args -r image:=/camera_left/image_raw  &
LEFT_VIEW_PID=$!
ros2 run image_tools showimage --ros-args -r image:=/camera_right/image_raw &
RIGHT_VIEW_PID=$!

# Move/resize windows by title.
# We retry a few times because windows appear a moment after process start.
move_window() {
  local title="$1" x="$2" y="$3" w="$4" h="$5"
  for i in {1..20}; do
    if wmctrl -l | grep -i "$title" >/dev/null; then
      wmctrl -r "$title" -e "0,$x,$y,$w,$h"
      return 0
    fi
    sleep 0.2
  done
  echo "Warning: could not find window with title matching: $title"
  return 1
}

# Try to match window titles. If these don’t match on your system, see note below.
move_window "showimage" $LEFT_WIN_X  $LEFT_WIN_Y  $LEFT_WIN_W  $LEFT_WIN_H || true
# second showimage window often has same title; we instead move “active” after focusing:
# focus+move the most recently created window
wmctrl -a "showimage" || true
sleep 0.2

# If both are still stacked, you can manually adjust titles (see note).
# For now, move all showimage windows to a cascading layout as fallback:
wmctrl -l | grep -i "showimage" | awk '{print $1}' | while read -r wid; do
  wmctrl -i -r "$wid" -e "0,$LEFT_WIN_X,$LEFT_WIN_Y,$LEFT_WIN_W,$LEFT_WIN_H"
  LEFT_WIN_X=$((LEFT_WIN_X+640))
done

################################################################################################

echo "Recording to: $RECORDINGS_DIR/$NAME"
cd "$RECORDINGS_DIR"
ros2 bag record -o "$NAME" --topics \
  /camera_left/image_raw \
  /camera_right/image_raw \
  /camera_left/camera_info \
  /camera_right/camera_info \
  /PSM1/measured_cp \
  /PSM2/measured_cp \
  /ECM/measured_cp &
BAG_PID=$!

wait "$BAG_PID"