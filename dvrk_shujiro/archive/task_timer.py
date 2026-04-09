#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class TaskTimer(Node):
    def __init__(self):
        super().__init__('task_timer')
        
        # Subscribe to teleop enabled status
        self.subscription = self.create_subscription(
            Bool,
            '/console/teleop/enabled',
            self.teleop_callback,
            10)
        
        self.start_time = None
        self.is_running = False
        self.get_logger().info('Task timer ready. Waiting for teleop to start...')

    def teleop_callback(self, msg):
        if msg.data and not self.is_running:
            # Task started
            self.start_time = self.get_clock().now()
            self.is_running = True
            self.get_logger().info('Task STARTED')
            
        elif not msg.data and self.is_running:
            # Task stopped
            end_time = self.get_clock().now()
            duration = (end_time - self.start_time).nanoseconds / 1e9
            self.get_logger().info(f'Task STOPPED. Duration: {duration:.2f} seconds')
            self.is_running = False

def main(args=None):
    rclpy.init(args=args)
    node = TaskTimer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
