#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo "═══════════════════════════════════════"
echo "  dVRK Surgery Recording (RAW)"
echo "═══════════════════════════════════════"
echo ""
read -p "Enter recording name: " NAME

RECORDINGS_DIR=~/dvrk_recordings/training_raw
mkdir -p "$RECORDINGS_DIR"
cd "$RECORDINGS_DIR"

echo ""
echo "Recording: $NAME"
echo "Location:  $RECORDINGS_DIR/$NAME"
echo ""
echo "Topics being recorded:"
echo "  /camera_left/image_raw"
echo "  /camera_right/image_raw"
echo "  /camera_left/camera_info"
echo "  /camera_right/camera_info"
echo "  /PSM1/measured_cp"
echo "  /PSM2/measured_cp"
echo "  /ECM/measured_cp"
echo ""
echo "Press Ctrl+C to stop recording"
echo ""

ros2 bag record -o "$NAME" --topics \
  /camera_left/image_raw \
  /camera_right/image_raw \
  /camera_left/camera_info \
  /camera_right/camera_info \
  /PSM1/measured_cp \
  /PSM2/measured_cp \
  /ECM/measured_cp

echo ""
echo "✓ Recording saved to: $RECORDINGS_DIR/$NAME"
ros2 bag info "$NAME"