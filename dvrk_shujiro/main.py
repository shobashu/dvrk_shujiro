# Run with this command:
# cd ~/dvrk_shujiro_ws/src/dvrk_shujiro
# python3 -m dvrk_shujiro.main

#!/usr/bin/env python3
"""
dVRK Task Timer - Main Entry Point
Run this file to start everything
"""
import rclpy
import threading

from .config import ARDUINO_PORT, MAX_TIME_SEC
from .gui.timer_window import TimerGUI
from .gui.trial_popup import TrialPopup
from .nodes.task_timer_node import TaskTimerNode
from .arduino.read_arduino import ArduinoReader



def main(args=None):
    """
    Main entry point - starts GUI and ROS node together
    
    Usage:
        python3 -m dvrk_shujiro.main
    """
    print("Starting dVRK Task Timer...")
    
    # ── ROS 2 ─────────────────────────────────────────────────────────────────
    rclpy.init(args=args)
    
    # ── GUI ────────────────────────────────────────────────────────
    gui = TimerGUI(max_time=MAX_TIME_SEC)

    # ── Trial popup ───────────────────────────────────────────────────────────
    # popup = TrialPopup(root=gui.window_left.root)  # Use left window root for thread-safe popups    
    
    # ── ROS node ───────────────────────────────────────────────────
    node = TaskTimerNode(gui)

    # ── Arduino reader ────────────────────────────────────────────────────────
    # arduino = ArduinoReader(port=ARDUINO_PORT, callback=popup.on_arduino_event)
    # arduino.start()

    # ── Start ROS spinning in background thread ─────────────────────────────────────────────
    node.start_spinning()  
    

    # Run GUI in main thread (blocking)
    try:
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