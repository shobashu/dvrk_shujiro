#!/usr/bin/env bash
# Record Surgery (safe version)
# ./record-surgery-safe.sh
# Enter recording name: trial_01_compressed
#
# Difference from record-surgery-compressed.sh:
#   - Verifies every required topic has a live publisher before recording starts.
#     If the dVRK console isn't up / arms aren't homed, this fails fast with a
#     clear message instead of silently recording camera-only footage.
#   - Actually records /camera_right/compressed (the old script listed it but
#     never passed it to `ros2 bag record`).
#   - Sources dvrk_ros2_new_ws instead of ros2_ws: the SRC dVRK no longer relays
#     PSM1/MTMR readings over Bluetooth (dESSJ/Arduino BLE boards were replaced
#     with native ESSJ + new dSIB boards, firmware reprogrammed). The console is
#     now started separately via dvrk_system from dvrk_ros2_new_ws; topic names
#     (/PSM1/*, /MTMR/*) are unchanged.

set -eo pipefail

# ROS's setup.bash files reference unset variables internally, so nounset
# must be off while sourcing them.
set +u
source /opt/ros/jazzy/setup.bash
source ~/dvrk_ros2_new_ws/install/setup.bash
set -u

TOPICS=(
  /camera_left/compressed
  /camera_right/compressed
  /PSM1/measured_cp
  /PSM1/measured_cv
  /PSM1/measured_js
  /PSM1/jaw/measured_js
  /MTMR/measured_cp
  /MTMR/measured_cv
  /MTMR/measured_js
  /MTMR/gripper/measured_js
)

WAIT_TIMEOUT_SEC=15

echo "═══════════════════════════════════════"
echo "  dVRK Surgery Recording (safe)"
echo "═══════════════════════════════════════"
echo ""

echo "Checking that all required topics have a live publisher (timeout ${WAIT_TIMEOUT_SEC}s)..."
missing=("${TOPICS[@]}")
elapsed=0
while [ "$elapsed" -lt "$WAIT_TIMEOUT_SEC" ]; do
  still_missing=()
  for t in "${missing[@]}"; do
    # "Publisher count: N" with N > 0 means something is actually publishing,
    # not just that the topic name is known to the graph. `ros2 topic info`
    # exits non-zero for a wholly unknown topic, so don't let that (or
    # pipefail) kill the script under set -e.
    pub_count=$(ros2 topic info "$t" --verbose 2>/dev/null | awk -F': ' '/Publisher count/{print $2}') || true
    if [ -z "$pub_count" ] || [ "$pub_count" -eq 0 ] 2>/dev/null; then
      still_missing+=("$t")
    fi
  done
  missing=("${still_missing[@]}")
  if [ "${#missing[@]}" -eq 0 ]; then
    break
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo ""
  echo "✗ Aborting: the following topics have no publisher after ${WAIT_TIMEOUT_SEC}s:"
  for t in "${missing[@]}"; do
    echo "    $t"
  done
  echo ""
  echo "This almost always means dvrk_system isn't running yet, or the arms"
  echo "aren't homed. In another terminal, run:"
  echo "    source /opt/ros/jazzy/setup.bash"
  echo "    source ~/dvrk_ros2_new_ws/install/setup.bash"
  echo "    cd ~/dvrk_ros2_new_ws/src/dvrk/dvrk_config_stanford"
  echo "    ros2 run dvrk_robot dvrk_system -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-Teleop.json"
  echo "then home PSM1/MTMR, confirm with 'ros2 topic list' that these topics"
  echo "appear, and re-run this script. Nothing was recorded."
  exit 1
fi

echo "✓ All required topics are live."
echo ""

read -p "Enter recording name: " NAME

RECORDINGS_DIR=~/dvrk_recordings/training
mkdir -p "$RECORDINGS_DIR"
cd "$RECORDINGS_DIR"

echo ""
echo "Recording: $NAME"
echo "Location: $RECORDINGS_DIR/$NAME"
echo ""
echo "Topics being recorded:"
for t in "${TOPICS[@]}"; do
  echo "  $t"
done
echo ""
echo "File size: ~50-100 MB/min (vs 1.8 GB/min raw)"
echo ""
echo "Press Ctrl+C to stop recording"
echo ""

ros2 bag record -o "$NAME" "${TOPICS[@]}"

echo ""
echo "✓ Recording saved to: $RECORDINGS_DIR/$NAME"
echo ""

ros2 bag info "$NAME"
