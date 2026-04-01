#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

SHOW_VIEWERS=${1:-false}

cleanup() {
    echo ""
    echo "Stopping cameras..."
    kill $RIGHT_CAM_PID $LEFT_CAM_PID $RIGHT_VIEW_PID $LEFT_VIEW_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "════════════════════════════════════════"
echo "  dVRK Camera Stream (Optimized)"
echo "════════════════════════════════════════"
echo ""

# Right camera with JPEG compression
echo "Starting right camera (device 0)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_right \
  -r camera/image_raw:=camera_right/image_raw \
  -r camera/camera_info:=camera_right/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=0 ! videoconvert ! jpegenc quality=90 ! jpegdec" \
  -p camera_name:=camera_right \
  -p frame_id:=camera &
RIGHT_CAM_PID=$!

# Left camera with JPEG compression
echo "Starting left camera (device 1)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_left \
  -r camera/image_raw:=camera_left/image_raw \
  -r camera/camera_info:=camera_left/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=1 ! videoconvert ! jpegenc quality=90 ! jpegdec" \
  -p camera_name:=camera_left \
  -p frame_id:=camera &
LEFT_CAM_PID=$!

sleep 3

echo ""
echo "✓ Cameras started!"
echo ""
echo "Mode: Compressed (JPEG quality=90)"
echo ""
echo "Published topics:"
echo "  • camera_right/image_raw"
echo "  • camera_left/image_raw"
echo ""

# Check FPS
echo "Checking publishing rate..."
sleep 2
LEFT_RATE=$(timeout 3 ros2 topic hz /camera_left/image_raw 2>/dev/null | grep "average rate" | awk '{print $3}')

if [ ! -z "$LEFT_RATE" ]; then
    echo "  ✓ Publishing at: ${LEFT_RATE} Hz"
else
    echo "  ⧗ Cameras starting..."
fi

if [ "$SHOW_VIEWERS" = "true" ]; then
    echo ""
    echo "Launching image viewers..."
    
    ros2 run image_view image_view --ros-args \
      -r image:=camera_left/image_raw \
      -p autosize:=false &
    LEFT_VIEW_PID=$!
    
    ros2 run image_view image_view --ros-args \
      -r image:=camera_right/image_raw \
      -p autosize:=false &
    RIGHT_VIEW_PID=$!
fi

echo ""
echo "════════════════════════════════════════"
echo "To view cameras, use rqt:"
echo "  rqt"
echo "════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop cameras"
echo ""

wait $RIGHT_CAM_PID $LEFT_CAM_PID