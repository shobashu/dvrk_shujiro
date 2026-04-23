#!/usr/bin/env bash
set -e

# Monitor geometry (from your xrandr)
LEFT_X=0;   LEFT_Y=0;  LEFT_W=640; LEFT_H=480
RIGHT_X=640; RIGHT_Y=0; RIGHT_W=640; RIGHT_H=480

# Launch the two viewers
ros2 run rqt_image_view rqt_image_view --ros-args -r image:=/camera_left/image_raw &
PID_LEFT=$!

ros2 run rqt_image_view rqt_image_view --ros-args -r image:=/camera_right/image_raw &
PID_RIGHT=$!

# Give Qt time to create the windows
sleep 2

# Get the two newest rqt_image_view windows (IDs)
mapfile -t IDS < <(wmctrl -l | awk '/rqt_image_view__ImageView - rqt/ {print $1}' | tail -n 2)

if [ "${#IDS[@]}" -ne 2 ]; then
  echo "ERROR: expected 2 rqt_image_view windows, found ${#IDS[@]}."
  echo "Run: wmctrl -l | grep -i rqt"
  exit 1
fi

# If your system consistently creates them in the same order, set mapping here:
# You said: first ID = RIGHT view, second ID = LEFT view
ID_RIGHT="${IDS[0]}"
ID_LEFT="${IDS[1]}"

# Move/resize windows to your two 640x480 monitors
wmctrl -i -r "$ID_LEFT"  -e "0,${LEFT_X},${LEFT_Y},${LEFT_W},${LEFT_H}"
wmctrl -i -r "$ID_RIGHT" -e "0,${RIGHT_X},${RIGHT_Y},${RIGHT_W},${RIGHT_H}"

echo "Placed LEFT  ($ID_LEFT) at  ${LEFT_X},${LEFT_Y} ${LEFT_W}x${LEFT_H}"
echo "Placed RIGHT ($ID_RIGHT) at ${RIGHT_X},${RIGHT_Y} ${RIGHT_W}x${RIGHT_H}"

# Keep script alive while viewers run
wait "$PID_LEFT" "$PID_RIGHT"