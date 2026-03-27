## Run with this commmand:
# cd ~/dvrk_shujiro_ws/src/dvrk_shujiro/scripts
# ./camera-stream-publish.sh

## Open new terminal and run:
# source /opt/ros/jazzy/setup.bash
# source ~/ros2_ws/install/setup.bash
# rqt #launch viewer

# In the rqt window:

#     Click: Plugins → Visualization → Image View
#     Select topic: /camera/left/image_raw or /camera/right/image_raw


# To verify if cameras are working
# source /opt/ros/jazzy/setup.bash
# source ~/ros2_ws/install/setup.bash
# ros2 topic hz /camera/left/image_raw  # Should show ~30 Hz

######################################################################################

# !/usr/bin/env bash

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping cameras..."
    kill $RIGHT_CAM_PID $LEFT_CAM_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "════════════════════════════════════════"
echo "  dVRK Camera Stream Publisher"
echo "════════════════════════════════════════"
echo ""

# Right camera
echo "Starting right camera (device 0)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_right \
  -r camera/image_raw:=/camera/right/image_raw \
  -r camera/camera_info:=/camera/right/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=0 ! videoconvert ! videobalance brightness=0 contrast=1.3 saturation=1.2 ! gamma gamma=1.8" \
  -p image_encoding:=rgb8 \
  -p frame_id:=right_camera_frame &
RIGHT_CAM_PID=$!

# Left camera
echo "Starting left camera (device 1)..."
ros2 run gscam gscam_node --ros-args \
  -r __node:=gscam_left \
  -r camera/image_raw:=/camera/left/image_raw \
  -r camera/camera_info:=/camera/left/camera_info \
  -p gscam_config:="decklinkvideosrc device-number=1 ! videoconvert ! videobalance brightness=0 contrast=1.3 saturation=1.2 ! gamma gamma=1.8" \
  -p image_encoding:=rgb8 \
  -p frame_id:=left_camera_frame &
LEFT_CAM_PID=$!

sleep 3

echo ""
echo "✓ Cameras started successfully!"
echo ""
echo "Published ROS2 topics:"
echo "  • /camera/left/image_raw"
echo "  • /camera/right/image_raw"
echo ""
echo "────────────────────────────────────────"
echo "To view cameras, run in another terminal:"
echo "  rqt"
echo "────────────────────────────────────────"
echo ""
echo "Press Ctrl+C to stop cameras"

# Only wait for cameras, NOT viewers
wait $RIGHT_CAM_PID $LEFT_CAM_PID
