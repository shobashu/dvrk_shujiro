# Record Surgery
# ./record-surgery-compressed.sh
# Enter recording name: trial_01_compressed

#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo "═══════════════════════════════════════"
echo "  dVRK Surgery Recording (Compressed)"
echo "═══════════════════════════════════════"
echo ""
read -p "Enter recording name: " NAME

RECORDINGS_DIR=~/dvrk_recordings/compressed
mkdir -p $RECORDINGS_DIR
cd $RECORDINGS_DIR

echo ""
echo "Recording: $NAME"
echo "Location: $RECORDINGS_DIR/$NAME"
echo ""
echo "Topics being recorded:"
echo "  📹 /camera_left/compressed (~30 fps, JPEG)"
echo "  📹 /camera_right/compressed (~30 fps, JPEG)"
echo "  🤖 /PSM1/measured_cp (robot kinematics)"
echo "  🤖 /PSM2/measured_cp (robot kinematics)"
echo "  🤖 /ECM/measured_cp (camera arm position)"
echo ""
echo "💾 File size: ~50-100 MB/min (vs 1.8 GB/min raw)"
echo ""
echo "Press Ctrl+C to stop recording"
echo ""

ros2 bag record \
  -o $NAME \
  /camera_left/compressed \
  /camera_right/compressed \
  /PSM1/measured_cp \
  /PSM2/measured_cp \
  /ECM/measured_cp

echo ""
echo "✓ Recording saved to: $RECORDINGS_DIR/$NAME"
echo ""

# Show recording info
ros2 bag info $NAME

