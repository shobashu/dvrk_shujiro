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
    popup = TrialPopup(root=gui.window_left.root)  # Use left window root for thread-safe popups    
    
    # ── ROS node ───────────────────────────────────────────────────
    node = TaskTimerNode(gui)

    # ── Arduino callback ──────────────────────────────────────────────────────
    def on_arduino_event(event):
        """
        Called by ArduinoReader every time a valid event arrives.
        Runs in the Arduino background thread.

        event.location_type : "CENTER" or "PEG"
        event.event         : "LIFTED" or "PLACED"
        event.pin_index     : decoded index (raw number - 48)
        event.arduino_ms    : milliseconds since Arduino powered on
        """
        if event.location_type == "CENTER" and event.event == "LIFTED":
            print(f"[Main] Cylinder lifted from center {event.pin_index}")
            popup.show_threadsafe()

        elif event.location_type == "PEG" and event.event == "PLACED":
            print(f"[Main] Cylinder placed on peg {event.pin_index}")
            popup.complete_threadsafe()

    # ── Arduino reader ────────────────────────────────────────────────────────
    arduino = ArduinoReader(port=ARDUINO_PORT, callback=on_arduino_event)
    arduino.start()
    
    # ── ROS spinning in background thread ──────────────────────────
    def spin_node():
        try:
            rclpy.spin(node)
        except Exception as e:
            print(f"ROS error: {e}")
    
    ros_thread = threading.Thread(target=spin_node, daemon=True)
    ros_thread.start()
    
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