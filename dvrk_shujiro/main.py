#!/usr/bin/env python3
"""
dVRK Real-Time Task Timer & Kinematics Metrics GUI — Main Entry Point
=======================================================================
This is the CURRENT, active entry point for the real-time GUI, trial
clock, path length, and orientation-rate tracking. The scripts in
`archive/` (task_timer.py, task_timer_gui.py, etc.) are superseded
legacy standalone versions kept for reference only — do not run those.

Run with:
    conda deactivate
    source /opt/ros/jazzy/setup.bash
    cd ~/dvrk_shujiro_ws/src/dvrk_shujiro
    python3 -m dvrk_shujiro.main

On launch it prompts for the session settings (Enter accepts the
config.py default for each — see TRIGGER_SOURCE / ARDUINO_END_MODE /
TRACK_PATH_LENGTH / TRACK_ORIENTATION / ENABLE_STATE_DETECTION /
ENABLE_FORCE_SENSOR / ENABLE_ORIENTATION_CHECK in config.py to change
the defaults):
    - Trial trigger: MONO pedal + teleop, or Arduino cylinder lift/place
    - (if Arduino) trial end: placed on target peg, or placed + returned to center
    - whether to report path length / angular displacement & orientation rate
    - whether to enable cylinder state detection (STATIONARY/HELD/DROPPED/
      LOST via a trained XGBoost model — see scripts/7_2_train_xgboost.py
      and scripts/7_velocity_check.py, which this is a headless port of;
      needs the camera stream + YOLO venv, but NOT the Arduino trigger)
    - (if Arduino) whether to enable ATI force capture (needs the ATI UDP
      receiver running)
    - (if Arduino) whether to enable the YOLO cylinder-orientation check
      (needs the camera stream running AND this terminal to be in the
      YOLO venv — see read_arduino_with_force.py's docstring and README)

Requires (already running before you start this):
    - dVRK console / teleop stack active, PSM1 & PSM2 homed
    - /console/teleop/enabled, /console/operator_present,
      /PSM1/measured_cp, /PSM2/measured_cp all publishing
      (only needed for the MONO+teleop trigger; the Arduino trigger
      needs the peg board connected at ARDUINO_PORT instead)

Usage:
    1. Start this script (opens two floating timer windows) and answer
       the session-settings prompt.
    2a. MONO+teleop mode: enable teleop and hold MONO to start the clock.
    2b. Arduino mode: type 's' + Enter to start a trial block; lifting
        the cylinder starts the clock, the configured end event stops it.
    3. Path length (mm) is shown live in the GUI windows (if enabled).
       Orientation angle/rate is printed once per second to THIS
       terminal (not drawn in the GUI window) — watch the console.
    4. Trial end (timeout, teleop-disable, or Arduino end event) logs the
       full trial summary to this terminal, resets, and waits for the
       next trial — the timer windows stay open the whole session.

Known gotcha: if the trial clock stops responding to MONO presses
mid-session, toggle teleop OFF then ON again on the console — this
resets the enable state and the clock starts responding again.
"""
import rclpy

from .config import (
    MAX_TIME_SEC, TRIGGER_SOURCE, ARDUINO_END_MODE,
    TRACK_PATH_LENGTH, TRACK_ORIENTATION,
    ENABLE_FORCE_SENSOR, ENABLE_ORIENTATION_CHECK, ENABLE_STATE_DETECTION,
)
from .gui.timer_window import TimerGUI
from .nodes.task_timer_node import TaskTimerNode


def _prompt_choice(label, options, default):
    """options: list of (key, description). Enter accepts `default`."""
    print(label)
    for i, (key, desc) in enumerate(options, start=1):
        marker = ' [default]' if key == default else ''
        print(f'  {i}) {desc}{marker}')
    keys = [key for key, _ in options]
    choice = input(f'Choose 1-{len(options)} [{keys.index(default) + 1}]: ').strip()
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return keys[int(choice) - 1]
    return default


def _prompt_bool(label, default):
    d = 'Y/n' if default else 'y/N'
    ans = input(f'{label} [{d}]: ').strip().lower()
    if not ans:
        return default
    return ans.startswith('y')


def prompt_session_settings():
    """Show the config.py defaults and let the user accept (Enter) or override
    them for this run. Nothing here is persisted — edit config.py to change
    the defaults for next time."""
    print('\nSession settings (Enter accepts the config.py default):')

    trigger_source = _prompt_choice(
        'Trial trigger:',
        [('mono_teleop', 'MONO pedal + teleop enabled'),
         ('arduino', 'Arduino cylinder lift/place')],
        TRIGGER_SOURCE)

    arduino_end_mode = ARDUINO_END_MODE
    if trigger_source == 'arduino':
        arduino_end_mode = _prompt_choice(
            'Arduino trial end:',
            [('target_placed', 'placed on target peg'),
             ('return_to_center', 'placed on target AND carried back to center')],
            ARDUINO_END_MODE)

    track_path = _prompt_bool('Report path length?', TRACK_PATH_LENGTH)
    track_orientation = _prompt_bool('Report angular displacement / orientation rate?', TRACK_ORIENTATION)

    # Independent of trigger_source — only needs the camera + PSM1 pose,
    # both already available in either mode.
    enable_state_detection = _prompt_bool(
        'Enable cylinder state detection (STATIONARY/HELD/DROPPED/LOST)?',
        ENABLE_STATE_DETECTION)

    enable_force_sensor = False
    enable_orientation_check = False
    if trigger_source == 'arduino':
        enable_force_sensor = _prompt_bool('Enable force sensor?', ENABLE_FORCE_SENSOR)
        enable_orientation_check = _prompt_bool('Enable YOLO orientation check?', ENABLE_ORIENTATION_CHECK)

    print(f'-> trigger={trigger_source}'
          + (f', arduino_end_mode={arduino_end_mode}' if trigger_source == 'arduino' else '')
          + f', path_length={track_path}, orientation={track_orientation}'
          + f', state_detection={enable_state_detection}'
          + (f', force_sensor={enable_force_sensor}, orientation_check={enable_orientation_check}'
             if trigger_source == 'arduino' else '')
          + '\n')

    if enable_force_sensor or enable_orientation_check or enable_state_detection:
        print('Make sure these are already running before you continue:')
        if enable_force_sensor:
            print('  - python3 scripts/ati_sensor_udp_receiver.py        (publishes /ati_sensor/wrench)')
        if enable_orientation_check or enable_state_detection:
            print('  - ./scripts/camera-stream-compressed-transport.sh   (publishes /camera_left/compressed)')
            print('  - this terminal itself must be in the YOLO venv (source ~/ros2_yolo_venv/bin/activate)')
        print()

    return {
        'trigger_source': trigger_source,
        'arduino_end_mode': arduino_end_mode,
        'track_path': track_path,
        'track_orientation': track_orientation,
        'enable_state_detection': enable_state_detection,
        'enable_force_sensor': enable_force_sensor,
        'enable_orientation_check': enable_orientation_check,
    }


def main(args=None):
    """
    Main entry point - starts GUI and ROS node together

    Usage:
        python3 -m dvrk_shujiro.main
    """
    print("Starting dVRK Task Timer...")

    settings = prompt_session_settings()

    # ── ROS 2 ─────────────────────────────────────────────────────────────────
    rclpy.init(args=args)

    # ── GUI ────────────────────────────────────────────────────────
    gui = TimerGUI(max_time=MAX_TIME_SEC)

    # ── ROS node (also starts the Arduino trigger thread, and the force/
    #    orientation tracker + its own popup, if selected) ────────────────────
    node = TaskTimerNode(gui, settings)

    # ── Start ROS spinning in background thread ───────────────────────────────
    node.start_spinning()

    # Run GUI in main thread (blocking)
    try:
        if settings['trigger_source'] == 'arduino':
            print("GUI started. Type 's' + Enter to start an Arduino trial block.")
        else:
            print("GUI started. Press MONO pedal to begin.")
        gui.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    
    print("Shutdown complete.")


if __name__ == '__main__':
    main()