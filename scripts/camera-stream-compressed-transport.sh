# Start Camera Streaming
# ./camera-stream-compressed-transport.sh

# View cameras (new terminal)
# source /opt/ros/jazzy/setup.bash
# source ~/ros2_ws/install/setup.bash
# rqt

#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

cleanup() {
    echo ""
    echo "Stopping cameras and compression..."
    kill $RIGHT_CAM_PID $LEFT_CAM_PID $RIGHT_COMP_PID $LEFT_COMP_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "════════════════════════════════════════"
echo "  dVRK Camera with ROS2 Compression"
echo "════════════════════════════════════════"
echo ""

# Right camera - simple pipeline
echo "Starting right camera (device 0)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_right \
  -r camera/image_raw:=camera_right/image_raw \
  -r camera/camera_info:=camera_right/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=0 ! videoconvert ! videobalance brightness=0.1 contrast=1.2 saturation=1.5 hue=0.0" \
  -p camera_name:=camera_right \
  -p frame_id:=camera &
RIGHT_CAM_PID=$!

# Left camera - simple pipeline
echo "Starting left camera (device 1)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_left \
  -r camera/image_raw:=camera_left/image_raw \
  -r camera/camera_info:=camera_left/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=1 ! videoconvert" \
  -p camera_name:=camera_left \
  -p frame_id:=camera &
LEFT_CAM_PID=$!

sleep 3

echo ""
echo "Starting ROS2 image compression..."

# Compress right camera stream
ros2 run image_transport republish raw compressed --ros-args \
  -r in:=camera_right/image_raw \
  -r out/compressed:=camera_right/compressed \
  -p disable_pub_plugins:="['compressedDepth', 'theora', 'zstd']" &
RIGHT_COMP_PID=$!

# Compress left camera stream
ros2 run image_transport republish raw compressed --ros-args \
  -r in:=camera_left/image_raw \
  -r out/compressed:=camera_left/compressed \
  -p disable_pub_plugins:="['compressedDepth', 'theora', 'zstd']" &
LEFT_COMP_PID=$!

sleep 2

echo ""
echo "✓ Cameras and compression started!"
echo ""
echo "Raw topics (slow, ~19 fps):"
echo "  • camera_right/image_raw"
echo "  • camera_left/image_raw"
echo ""
echo "Compressed topics (fast, ~25-30 fps):"
echo "  • camera_right/compressed"
echo "  • camera_left/compressed"
echo ""

# Check FPS
echo "Checking compressed stream rate..."
sleep 2
LEFT_COMP_RATE=$(timeout 3 ros2 topic hz /camera_left/compressed 2>/dev/null | grep "average rate" | awk '{print $3}')

if [ ! -z "$LEFT_COMP_RATE" ]; then
    echo "  Compressed stream: ${LEFT_COMP_RATE} Hz"
else
    echo "  Compression starting..."
fi

echo ""
echo "════════════════════════════════════════"
echo "To view compressed stream in rqt:"
echo "  1. Launch rqt"
echo "  2. Plugins → Visualization → Image View"
echo "  3. Select: camera_left/compressed"
echo "════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop"
echo ""

wait $RIGHT_CAM_PID $LEFT_CAM_PID $RIGHT_COMP_PID $LEFT_COMP_PID
