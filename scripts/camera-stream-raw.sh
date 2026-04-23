#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

cleanup() {
  echo ""
  echo "Stopping cameras..."
  kill ${RIGHT_CAM_PID:-} ${LEFT_CAM_PID:-} 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

echo "========================================"
echo "  dVRK Camera RAW Streaming (no compress)"
echo "========================================"
echo ""

echo "Starting right camera (device 0)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_right \
  -r /camera/image_raw:=/camera_right/image_raw \
  -r /camera/camera_info:=/camera_right/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=0 ! videoconvert ! videobalance brightness=0.1 contrast=1.2 saturation=1.5 hue=0.0" \
  -p frame_id:=camera \
  -p use_sensor_data_qos:=true &
RIGHT_CAM_PID=$!

echo "Starting left camera (device 1)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_left \
  -r /camera/image_raw:=/camera_left/image_raw \
  -r /camera/camera_info:=/camera_left/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=1 ! videoconvert ! videobalance brightness=0.1 contrast=1.2 saturation=1.5 hue=0.0" \
  -p frame_id:=camera \
  -p use_sensor_data_qos:=true &
LEFT_CAM_PID=$!

sleep 2

echo ""
echo "Raw topics:"
echo "  /camera_left/image_raw"
echo "  /camera_right/image_raw"
echo ""
echo "Press Ctrl+C to stop."
echo ""

wait




