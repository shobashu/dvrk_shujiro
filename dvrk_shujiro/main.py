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

from .config import MAX_TIME_SEC
from .gui.timer_window import TimerGUI
from .nodes.task_timer_node import TaskTimerNode


def main(args=None):
    """
    Main entry point - starts GUI and ROS node together
    
    Usage:
        python3 -m dvrk_shujiro.main
    """
    print("Starting dVRK Task Timer...")
    
    # Initialize ROS 2
    rclpy.init(args=args)
    
    # Create GUI
    gui = TimerGUI(max_time=MAX_TIME_SEC)
    
    # Create ROS node
    node = TaskTimerNode(gui)
    
    # Run ROS in background thread
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