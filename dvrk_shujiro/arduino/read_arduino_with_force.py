"""
Arduino trial reader + live ATI force feedback + cylinder orientation popup.
------------------------------------------------------------------------------
Combines the Arduino serial trial protocol (see automated_leds_new.ino) with:
  - the live /ati_sensor/wrench ROS2 topic (ati_sensor_udp_receiver.py) for
    force feedback, and
  - the left camera + YOLO cylinder detection (see 6_cylinder_orientation.py)
    for an orientation check, triggered on the Arduino's "Target hit" line.

At the end of each trial — triggered by the Arduino's "Object returned
successfully." line — shows a force-vs-time graph, peak/mean force, and the
orientation result (OK/OFF + angle) in TrialPopup on all monitors.

Run in this order (deactivate conda first — see repo CLAUDE.md):
    1. python3 ati_sensor_udp_receiver.py                    (publishes /ati_sensor/wrench)
    2. ./camera-stream-compressed-transport.sh                (publishes /camera_left/compressed)
    3. source ~/ros2_yolo_venv/bin/activate                   (YOLO needs this venv)
       python3 read_arduino_with_force.py

Type 's' + Enter in this script's terminal to start the experiment on the
Arduino (forwarded over serial), 'q' + Enter to abort.
"""
import sys
import os
import time
import csv
import select
import threading
import signal
import tkinter as tk

import cv2
import numpy as np
import serial
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

sys.path.insert(0, '/home/stanford/dvrk_shujiro_ws/src/dvrk_shujiro')
from dvrk_shujiro.gui.trial_popup import TrialPopup

# Load compute_orientation() from scripts/6_cylinder_orientation.py without
# turning scripts/ into an importable package.
import importlib.util as _ilu
_SCRIPTS_DIR = '/home/stanford/dvrk_shujiro_ws/src/dvrk_shujiro/scripts'
_spec = _ilu.spec_from_file_location(
    "cylinder_orientation", os.path.join(_SCRIPTS_DIR, "6_cylinder_orientation.py"))
_cyl_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cyl_mod)
compute_orientation = _cyl_mod.compute_orientation

# --- CONFIGURATION ---
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE    = 9600
CSV_FILENAME = 'experiment_data.csv'
WRENCH_TOPIC = '/ati_sensor/wrench'

CAMERA_TOPIC              = '/camera_left/compressed'
YOLO_WEIGHTS              = os.path.join(_SCRIPTS_DIR, 'models', 'best_v2.pt')
YOLO_CONF                 = 0.5
YOLO_IMGSZ                = 320
ORIENTATION_DELAY_SEC     = 0.5    # wait after "Target hit" before cropping, so the piece settles
ORIENTATION_TOLERANCE_DEG = 20.0   # |angle| <= this many degrees from upright (0°) counts as OK
ORIENTATION_MAX_ATTEMPTS  = 3      # retry this many times if no cylinder is detected (e.g. hand/gripper in the way)
ORIENTATION_RETRY_DELAY_SEC = 0.5  # wait between retries

# Outer peg index (1-8) -> which half of the left-camera frame it's in
PEG_CLASSIFICATIONS = {1: "L", 2: "L", 3: "L", 4: "L", 5: "R", 6: "R", 7: "R", 8: "R"}


class WrenchListener(Node):
    """Subscribes to /ati_sensor/wrench and buffers ‖F‖ samples while a trial is active."""

    def __init__(self):
        super().__init__('force_trial_listener')
        self._lock   = threading.Lock()
        self._active = False
        self._t0     = 0.0
        self._buffer = []  # list of (t_rel_seconds, magnitude_newtons)
        self.create_subscription(WrenchStamped, WRENCH_TOPIC, self._on_wrench, 50)

    def _on_wrench(self, msg):
        if not self._active:
            return
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        mag = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        t_rel = time.time() - self._t0
        with self._lock:
            self._buffer.append((t_rel, mag))

    def start_trial(self):
        with self._lock:
            self._buffer = []
            self._t0 = time.time()
            self._active = True

    def end_trial(self):
        """Stop buffering and return (times, forces, peak, mean) for the trial just finished."""
        with self._lock:
            self._active = False
            samples = list(self._buffer)
        if not samples:
            return [], [], 0.0, 0.0
        times  = [s[0] for s in samples]
        forces = [s[1] for s in samples]
        peak   = max(forces)
        mean   = sum(forces) / len(forces)
        return times, forces, peak, mean


class CameraFrameNode(Node):
    """Subscribes to the left camera and buffers only the latest frame.

    Orientation checks are on-demand (triggered once per "Target hit" event),
    not continuous inference, so no background inference thread is needed here.
    """

    def __init__(self):
        super().__init__('force_trial_camera')
        self._lock  = threading.Lock()
        self._frame = None
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(CompressedImage, CAMERA_TOPIC, self._on_image, qos)

    def _on_image(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._lock:
            self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame


def check_orientation(camera_node, model, side):
    """Crop the latest left-camera frame to the given peg side, run YOLO + orientation.

    Returns (angle_deg, is_upright, down_color), or (None, None, None) if no
    frame/cylinder is available. down_color is "blue" or "white" — whichever
    centroid sits lower (larger y) in the crop, i.e. which color is facing down.
    """
    frame = camera_node.get_frame()
    if frame is None:
        return None, None, None

    h, w = frame.shape[:2]
    crop = frame[:, :int(w * 0.4)] if side == "L" else frame[:, int(w * 0.55):]

    results = model.predict(crop, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, classes=[0], verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None, None, None

    ch, cw = crop.shape[:2]
    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(cw, x2), min(ch, y2)
    sub = crop[y1:y2, x1:x2]
    if sub.size == 0:
        return None, None, None

    angle_deg, blue_c, white_c, _ = compute_orientation(sub)
    if white_c is None or blue_c is None:
        return None, None, None

    down_color = "blue" if blue_c[1] > white_c[1] else "white"

    return angle_deg, abs(angle_deg) <= ORIENTATION_TOLERANCE_DEG, down_color


def arduino_loop(popup, wrench_listener, camera_node, model):
    """Runs in a background thread. Reads Arduino serial, drives the popup and force capture."""
    try:
        print(f"Connecting to {ARDUINO_PORT}...")
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)  # Give Arduino time to reset

        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'Target_Peg', 'Target_Color',
                             'Unix_Cue_Time', 'Unix_Lift_Time', 'Unix_Place_Time',
                             'Peak_Force_N', 'Mean_Force_N',
                             'Orientation_Deg', 'Orientation_OK', 'Down_Color', 'Placement_Success'])

            print("\n--- Python Monitor Ready ---")
            print(f"Data will be saved to: {CSV_FILENAME}")
            print("Type 's' and press Enter to START the experiment.")
            print("Type 'q' and press Enter to ABORT.")
            print("----------------------------\n")

            python_sync_time      = 0
            arduino_sync_millis   = 0
            current_trial         = None
            last_force_result     = ([], [], 0.0, 0.0)
            last_orientation      = (None, None, None, None)  # (angle_deg, is_upright, down_color, success) for the trial just finished
            last_lifted_peg       = None           # outer peg index (1-8) from "Object lifted..." line
            pending_orientation   = None           # {"time": float, "side": "L"/"R"} while waiting out the delay
            current_target_color  = None           # "Blue"/"White" from the "CUE," line, known before the piece is lifted

            while True:
                # Fire the orientation check once its scheduled time has elapsed,
                # independent of whether new serial data has arrived. Initially
                # scheduled ORIENTATION_DELAY_SEC after "Target hit"; retried up to
                # ORIENTATION_MAX_ATTEMPTS times (spaced by ORIENTATION_RETRY_DELAY_SEC)
                # if no cylinder is detected, e.g. the gripper is still in the frame.
                if pending_orientation is not None and \
                        time.time() >= pending_orientation["next_check"]:
                    side         = pending_orientation["side"]
                    target_color = pending_orientation["target_color"]
                    attempt      = pending_orientation["attempt"]
                    angle_deg, is_upright, down_color = check_orientation(camera_node, model, side)

                    if angle_deg is None and attempt < ORIENTATION_MAX_ATTEMPTS:
                        print(f"[ORIENTATION] no cylinder detected in crop "
                              f"(attempt {attempt}/{ORIENTATION_MAX_ATTEMPTS}), retrying...")
                        pending_orientation["attempt"]     = attempt + 1
                        pending_orientation["next_check"]  = time.time() + ORIENTATION_RETRY_DELAY_SEC
                    else:
                        pending_orientation = None
                        success = None
                        if down_color is not None and target_color is not None:
                            success = down_color.lower() == target_color.lower()
                        last_orientation = (angle_deg, is_upright, down_color, success)
                        if angle_deg is not None:
                            print(f"[ORIENTATION] {angle_deg:+.1f}°  ({'OK' if is_upright else 'OFF'})  "
                                  f"{down_color} facing down  —  target was {target_color}  "
                                  f"({'SUCCESS' if success else 'FAIL'})")
                        else:
                            print(f"[ORIENTATION] no cylinder detected in crop "
                                  f"after {attempt} attempts")

                # 1. Forward keyboard input to the Arduino (non-blocking)
                if sys.platform != "win32":
                    i, _, _ = select.select([sys.stdin], [], [], 0.0001)
                    if i:
                        cmd = sys.stdin.readline().strip()
                        if cmd:
                            arduino.write(cmd.encode('utf-8'))

                # 2. Read Arduino output
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8', errors='replace').strip()
                    if not line:
                        continue

                    if line.startswith("SYNC,"):
                        parts = line.split(",")
                        arduino_sync_millis = int(parts[1])
                        python_sync_time    = time.time()

                    elif line.startswith("--- Trial"):
                        # e.g. "--- Trial 3 ---" — marks the start of a new trial
                        digits = ''.join(ch for ch in line if ch.isdigit())
                        current_trial = digits or current_trial
                        wrench_listener.start_trial()
                        last_orientation = (None, None, None, None)
                        last_lifted_peg = None
                        pending_orientation = None
                        current_target_color = None

                    elif line.startswith("CUE,"):
                        # e.g. "CUE,Blue" — sent as soon as the target LEDs light up,
                        # well before the piece is lifted.
                        current_target_color = line.split(",", 1)[-1].strip()
                        print(f"[ARDUINO] Target color: {current_target_color}")

                    elif line == "LIFTED":
                        print("[ARDUINO] Cylinder lifted!")
                        popup.show_threadsafe()

                    elif "Object lifted. Move to Outer Peg" in line:
                        print(f"[ARDUINO] {line}")
                        try:
                            last_lifted_peg = int(line.split("Object lifted. Move to Outer Peg")[-1].strip())
                        except ValueError:
                            last_lifted_peg = None

                    elif "Target hit" in line:
                        # Cylinder just placed on the target peg — freezes the progress
                        # bar / scores the placement, and schedules the orientation
                        # check to fire after ORIENTATION_DELAY_SEC (let the piece settle).
                        print(f"[ARDUINO] {line}")
                        popup.complete_threadsafe()
                        side = PEG_CLASSIFICATIONS.get(last_lifted_peg)
                        if side is not None:
                            pending_orientation = {"next_check": time.time() + ORIENTATION_DELAY_SEC,
                                                    "side": side,
                                                    "target_color": current_target_color,
                                                    "attempt": 1}

                    elif line == "Object returned successfully.":
                        # Trial is fully complete — snapshot the force applied during it
                        print(f"[ARDUINO] {line}")
                        last_force_result = wrench_listener.end_trial()
                        times, forces, peak, mean = last_force_result
                        angle_deg, is_upright, down_color, success = last_orientation
                        popup.show_force_result_threadsafe(
                            current_trial or "?", times, forces, peak, mean,
                            angle_deg, is_upright, down_color, success)

                    elif line.startswith("DATA,"):
                        parts      = line.split(",")
                        trial      = parts[1]
                        peg        = parts[2]
                        color      = parts[3]
                        cue_unix   = python_sync_time + ((int(parts[4]) - arduino_sync_millis) / 1000.0)
                        lift_unix  = python_sync_time + ((int(parts[5]) - arduino_sync_millis) / 1000.0)
                        place_unix = python_sync_time + ((int(parts[6]) - arduino_sync_millis) / 1000.0)

                        _, _, peak, mean = last_force_result
                        angle_deg, is_upright, down_color, success = last_orientation
                        writer.writerow([trial, peg, color,
                                         f"{cue_unix:.3f}", f"{lift_unix:.3f}", f"{place_unix:.3f}",
                                         f"{peak:.3f}", f"{mean:.3f}",
                                         f"{angle_deg:.1f}" if angle_deg is not None else "",
                                         is_upright if is_upright is not None else "",
                                         down_color if down_color is not None else "",
                                         success if success is not None else ""])
                        file.flush()
                        print(f"[CSV LOGGED] Trial {trial} saved (peak={peak:.2f} N, mean={mean:.2f} N).")

                    else:
                        print(f"[ARDUINO] {line}")

    except serial.SerialException as e:
        print(f"Error connecting to {ARDUINO_PORT}: {e}")
    except Exception as e:
        print(f"[Arduino thread error] {e}")


def main():
    rclpy.init()

    print(f"[INFO] Loading YOLO weights: {YOLO_WEIGHTS}")
    model = YOLO(YOLO_WEIGHTS)

    wrench_listener = WrenchListener()
    camera_node     = CameraFrameNode()

    # Both nodes must share a single executor spun in one thread — calling
    # rclpy.spin() separately per node crashes with "Executor is already
    # spinning" since they share the same default rclpy context.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(wrench_listener)
    executor.add_node(camera_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Hidden tkinter root just to host the popup windows
    root = tk.Tk()
    root.withdraw()
    root.attributes('-alpha', 0.0)
    popup = TrialPopup(root=root)

    def signal_handler(sig, frame):
        print('\n\n[SHUTDOWN] Ctrl+C detected, exiting...')
        popup.print_final_score()
        executor.shutdown()
        wrench_listener.destroy_node()
        camera_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        root.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    thread = threading.Thread(
        target=arduino_loop, args=(popup, wrench_listener, camera_node, model), daemon=True)
    thread.start()

    def keep_alive():
        root.after(100, keep_alive)
    root.after(100, keep_alive)

    print("[INFO] Press Ctrl+C to quit")
    root.mainloop()


if __name__ == '__main__':
    main()
