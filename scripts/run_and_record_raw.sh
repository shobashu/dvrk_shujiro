#!/usr/bin/env bash
# run-and-record-raw.sh
# Starts both cameras (raw) and records raw (+ optional robot topics).
# Usage: ./run_and_record_raw.sh

set -e

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

read -p "Enter recording name: " NAME
RECORDINGS_DIR=~/dvrk_recordings/training_raw
mkdir -p "$RECORDINGS_DIR"
cd "$RECORDINGS_DIR"

cleanup() {
  echo ""
  echo "Stopping recorder and cameras..."
  [[ -n "${BAG_PID:-}" ]] && kill "$BAG_PID" 2>/dev/null || true
  [[ -n "${RIGHT_CAM_PID:-}" ]] && kill "$RIGHT_CAM_PID" 2>/dev/null || true
  [[ -n "${LEFT_CAM_PID:-}" ]] && kill "$LEFT_CAM_PID" 2>/dev/null || true
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

echo "Waiting for topics to appear..."
sleep 3

echo "Recording raw topics to: $RECORDINGS_DIR/$NAME"
echo "Press Ctrl+C to stop."

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
