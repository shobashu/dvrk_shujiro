#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

cleanup() {
    echo ""
    echo "Stopping cameras..."
    kill $RIGHT_CAM_PID $LEFT_CAM_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "════════════════════════════════════════"
echo "  dVRK Camera Stream (Compressed)"
echo "════════════════════════════════════════"
echo ""

# Right camera with JPEG compression
echo "Starting right camera (device 0) with compression..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_right \
  -r camera/image_raw:=/camera/right/image_raw \
  -r camera/camera_info:=/camera/right/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=0 ! videoconvert ! jpegenc quality=85 ! jpegdec ! videoconvert" \
  -p image_encoding:=rgb8 \
  -p frame_id:=right_camera_frame &
RIGHT_CAM_PID=$!

# Left camera with JPEG compression
echo "Starting left camera (device 1) with compression..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_left \
  -r camera/image_raw:=/camera/left/image_raw \
  -r camera/camera_info:=/camera/left/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=1 ! videoconvert ! jpegenc quality=85 ! jpegdec ! videoconvert" \
  -p image_encoding:=rgb8 \
  -p frame_id:=left_camera_frame &
LEFT_CAM_PID=$!

sleep 3

echo ""
echo "✓ Cameras started with compression!"
echo ""
echo "Mode: Compressed (should improve FPS)"
echo ""
echo "Published topics:"
echo "  • /camera/left/image_raw"
echo "  • /camera/right/image_raw"
echo ""
echo "Press Ctrl+C to stop cameras"
echo ""

wait $RIGHT_CAM_PID $LEFT_CAM_PID
