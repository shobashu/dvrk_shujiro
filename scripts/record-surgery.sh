#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo "═══════════════════════════════════════"
echo "  dVRK Recording"
echo "═══════════════════════════════════════"
echo ""
read -p "Enter recording name: " NAME

mkdir -p ~/dvrk_recordings
cd ~/dvrk_recordings

echo ""
echo "Recording: $NAME"
echo "Location: ~/dvrk_recordings/$NAME"
echo ""
echo "Press Ctrl+C to stop"
echo ""

ros2 bag record -o $NAME \
  /camera/left/image_raw \
  /camera/right/image_raw
