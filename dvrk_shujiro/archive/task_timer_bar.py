#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import sys
import time

class TaskTimerWithBar(Node):
    def __init__(self):
        super().__init__('task_timer_bar')
        
        self.subscription = self.create_subscription(
            Bool,
            '/console/teleop/enabled',
            self.teleop_callback,
            10)
        
        self.start_time = None
        self.is_running = False
        self.max_time = 120.0  # 2 minutes in seconds
        
        # Create a timer to update the progress bar
        self.timer = self.create_timer(0.1, self.update_display)  # 10 Hz update
        
        self.get_logger().info(f'Task timer ready. Time limit: {self.max_time}s')

    def teleop_callback(self, msg):
        if msg.data and not self.is_running:
            self.start_time = self.get_clock().now()
            self.is_running = True
            self.get_logger().info('Task STARTED')
            
        elif not msg.data and self.is_running:
            end_time = self.get_clock().now()
            duration = (end_time - self.start_time).nanoseconds / 1e9
            self.get_logger().info(f'\n Task STOPPED. Duration: {duration:.2f}s')
            self.is_running = False

    def update_display(self):
        if not self.is_running:
            return
        
        # Calculate elapsed time
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        
        # Calculate progress (0 to 1)
        progress = min(elapsed / self.max_time, 1.0)
        
        # Create progress bar
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Color coding
        if progress < 0.7:
            color = '\033[92m'  # Green
        elif progress < 0.9:
            color = '\033[93m'  # Yellow
        else:
            color = '\033[91m'  # Red
        
        reset = '\033[0m'
        
        # Print with carriage return (overwrites same line)
        sys.stdout.write(f'\r{color}[{bar}]{reset} {elapsed:.1f}s / {self.max_time:.0f}s')
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = TaskTimerWithBar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()