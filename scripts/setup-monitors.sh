#!/bin/bash

echo "Configuring dVRK manipulation displays..."

xrandr --output DP-2 --mode 640x480 --rate 59.94
xrandr --output DP-0 --mode 640x480 --rate 59.94
xrandr --output DP-0 --left-of HDMI-1
xrandr --output DP-2 --left-of DP-0

echo "  Display configuration complete"
echo "  DP-2 (left manipulation screen): 640x480 @ 59.94Hz"
echo "  DP-0 (right manipulation screen): 640x480 @ 59.94Hz"
