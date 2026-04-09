"""ROS 2 node for task timer"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped

from ..config import *
from ..metrics.metrics_tracker import MetricsTracker


class TaskTimerNode(Node):
    """Main ROS 2 node"""
    
    def __init__(self, gui):
        super().__init__('task_timer_gui')
        self.gui = gui
        
        # State
        self.teleop_active = False
        self.teleop_enabled = False
        self.mono_pressed = False
        self.was_running = False
        
        # Performance monitoring
        self.last_time = None
        self.current_hz = 0.0
        self.sample_count = 0
        
        # Use MetricsTracker objects (cleaner!)
        self.tracker_psm1 = MetricsTracker("PSM1")
        self.tracker_psm2 = MetricsTracker("PSM2")
        
        # Subscriptions
        self.teleop_sub = self.create_subscription(
            Bool, TOPIC_TELEOP_ENABLED, self.teleop_callback, 10)
        
        self.mono_sub = self.create_subscription(
            Joy, TOPIC_OPERATOR_PRESENT, self.mono_callback, 10)
        
        self.pose_sub_psm1 = self.create_subscription(
            PoseStamped, TOPIC_PSM1_POSE, self.pose_callback_psm1, 10)
        
        self.pose_sub_psm2 = self.create_subscription(
            PoseStamped, TOPIC_PSM2_POSE, self.pose_callback_psm2, 10)
        
        self.timer = self.create_timer(TIMER_INTERVAL, self.update_timer)
        
        self.get_logger().info('Task timer ready. Turn on teleoperation & press MONO.')
    
    def teleop_callback(self, msg):
        old_state = self.teleop_enabled
        self.teleop_enabled = msg.data
        
        if old_state and not msg.data:
            self._log_trial_results()
            self._reset_all_metrics()
        
        self.update_state()
    
    def mono_callback(self, msg):
        if len(msg.buttons) > 0:
            self.mono_pressed = (msg.buttons[0] == 1)
        else:
            self.mono_pressed = False
        self.update_state()
    
    def pose_callback_psm1(self, msg):
        if not self.gui.is_running:
            self.tracker_psm1.last_position = None
            self.tracker_psm1.last_orientation = None
            self.tracker_psm1.last_timestamp = None
            return
        
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w]
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Update tracker
        self.tracker_psm1.update_position(position)
        self.tracker_psm1.update_orientation(orientation, timestamp)
        
        # Update GUI
        self.gui.path_length_psm1 = self.tracker_psm1.path_length
        self.gui.angular_displacement_psm1 = self.tracker_psm1.get_angular_displacement_rad()
        self.gui.orientation_rate_psm1 = self.tracker_psm1.get_orientation_rate_rad()
    
    def pose_callback_psm2(self, msg):
        if not self.gui.is_running:
            self.tracker_psm2.last_position = None
            self.tracker_psm2.last_orientation = None
            self.tracker_psm2.last_timestamp = None
            return
        
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w]
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Update tracker
        self.tracker_psm2.update_position(position)
        self.tracker_psm2.update_orientation(orientation, timestamp)
        
        # Update GUI
        self.gui.path_length_psm2 = self.tracker_psm2.path_length
        self.gui.angular_displacement_psm2 = self.tracker_psm2.get_angular_displacement_rad()
        self.gui.orientation_rate_psm2 = self.tracker_psm2.get_orientation_rate_rad()
    
    def update_state(self):
        should_run = self.teleop_enabled and self.mono_pressed
        
        if should_run and not self.was_running:
            self.gui.start()
            self.sample_count = 0
            self.get_logger().info('⏱️  Timer STARTED (MONO pressed)')
            self.was_running = True
        elif not should_run and self.was_running:
            duration = self.gui.stop()
            if duration is not None:
                path1 = self.tracker_psm1.get_path_mm()
                path2 = self.tracker_psm2.get_path_mm()
                self.get_logger().info(
                    f'⏸️  PAUSED. Time: {duration:.2f}s | '
                    f'PSM1: {path1:.1f}mm | PSM2: {path2:.1f}mm'
                )
            self.was_running = False
    
    def update_timer(self):
        self.gui.tick(TIMER_INTERVAL)
        
        if self.gui.is_running:
            self.sample_count += 1
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.last_time is not None:
                dt = current_time - self.last_time
                self.current_hz = 1.0 / dt if dt > 0 else 0.0
                
                if self.sample_count % TIMER_RATE_HZ == 0:
                    self._log_periodic_update()
            
            self.last_time = current_time
    
    def _log_periodic_update(self):
        path1 = self.tracker_psm1.get_path_mm()
        path2 = self.tracker_psm2.get_path_mm()
        ang1 = self.tracker_psm1.get_angular_displacement_deg()
        ang2 = self.tracker_psm2.get_angular_displacement_deg()
        rate1 = self.tracker_psm1.get_orientation_rate_deg()
        rate2 = self.tracker_psm2.get_orientation_rate_deg()
        
        self.get_logger().info(
            f'[{self.gui.elapsed:.1f}s @ {self.current_hz:.1f}Hz] Path: R={path1:.0f}mm L={path2:.0f}mm | '
            f'Orient: R={ang1:.0f}° ({rate1:.1f}°/s) L={ang2:.0f}° ({rate2:.1f}°/s)'
        )
    
    def _log_trial_results(self):
        if self.gui.elapsed <= 0:
            return
        
        duration = self.gui.elapsed
        path1 = self.tracker_psm1.get_path_mm()
        path2 = self.tracker_psm2.get_path_mm()
        total_path = path1 + path2
        
        ang1_rad = self.tracker_psm1.get_angular_displacement_rad()
        ang2_rad = self.tracker_psm2.get_angular_displacement_rad()
        rate1_rad = self.tracker_psm1.get_orientation_rate_rad()
        rate2_rad = self.tracker_psm2.get_orientation_rate_rad()
        
        self.get_logger().info('='*60)
        self.get_logger().info('✅ TRIAL COMPLETE')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Duration: {duration:.2f} s')
        self.get_logger().info('')
        self.get_logger().info('Path Length:')
        self.get_logger().info(f'  PSM1 (R): {path1:.1f} mm')
        self.get_logger().info(f'  PSM2 (L): {path2:.1f} mm')
        self.get_logger().info(f'  Total:    {total_path:.1f} mm')
        self.get_logger().info('')
        self.get_logger().info('Angular Displacement:')
        self.get_logger().info(f'  PSM1 (R): {ang1_rad:.3f} rad')
        self.get_logger().info(f'  PSM2 (L): {ang2_rad:.3f} rad')
        self.get_logger().info('')
        self.get_logger().info('Average Orientation Rate:')
        self.get_logger().info(f'  PSM1 (R): {rate1_rad:.4f} rad/s')
        self.get_logger().info(f'  PSM2 (L): {rate2_rad:.4f} rad/s')
        self.get_logger().info('='*60)
    
    def _reset_all_metrics(self):
        self.gui.reset()
        self.sample_count = 0
        self.tracker_psm1.reset()
        self.tracker_psm2.reset()

    def start_spinning(self):
        """Start ROS spinning in a background thread."""
        import threading
        thread = threading.Thread(target=self._spin, daemon=True)
        thread.start()

    def _spin(self):
        try:
            rclpy.spin(self)
        except Exception as e:
            print(f"ROS error: {e}")