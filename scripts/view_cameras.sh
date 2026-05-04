#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Prevent snap's old libpthread (core20) from being loaded instead of the system one
export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3.12 -s "${SCRIPT_DIR}/view_cameras.py" "$@"
