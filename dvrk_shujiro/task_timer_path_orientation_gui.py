# WITH ORIENTATION PATH LENGTH

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
import tkinter as tk
from tkinter import ttk
import threading
import math
import numpy as np

# ============ Configuration ============
TIMER_RATE_HZ = 200        # Sampling rate (Hz)
TIMER_INTERVAL = 1.0 / TIMER_RATE_HZ
MAX_TIME_SEC = 120        # Time limit (seconds)
WINDOW_ALPHA = 0.85       # Transparency (0.0-1.0)
# =======================================


# ============ Quaternion Math Functions ============

def quaternion_conjugate(q):
    """Return conjugate (inverse for unit quaternions) of quaternion [x, y, z, w]"""
    return [-q[0], -q[1], -q[2], q[3]]

def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def quaternion_to_angle(q):
    """
    Extract rotation angle θ from quaternion q = [x, y, z, w]
    Following equation (6): θ = 2 * arccos(w)
    w is the scalar part (q4 in the paper, but index 3 in [x,y,z,w])
    """
    w = q[3]
    # Clamp to [-1, 1] to avoid numerical errors in arccos
    w = max(-1.0, min(1.0, w))
    theta = 2.0 * math.acos(w)
    return theta


class TimerGUI:
    def __init__(self, max_time=120):
        self.max_time = max_time
        self.elapsed = 0.0
        self.path_length_psm1 = 0.0  # PSM1 path in millimeters
        self.path_length_psm2 = 0.0  # PSM2 path in millimeters
        self.is_running = False

        # NEW: Orientation metrics
        self.angular_displacement_psm1 = 0.0  # A (total angle change)
        self.angular_displacement_psm2 = 0.0
        self.orientation_rate_psm1 = 0.0  # C (average rate)
        self.orientation_rate_psm2 = 0.0
        
        # Window size
        # self.width = 350
        # self.height = 110  # Increased for path length display
        
        # Create TWO windows (left and right)
        self.root_left = self.create_window("dVRK Timer", is_left=True)
        self.root_right = self.create_window("dVRK Timer", is_left=False)
        
        # Update loop
        self.update_display()
    
    def create_window(self, title, is_left=True):
        root = tk.Tk()
        root.title(title)
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        half_width = screen_width
        
        self.width = 340
        self.height = 95
        
        if is_left:
            x_position = half_width - self.width - 20
            y_position = 20
        else:
            x_position = screen_width - self.width - 20
            y_position = 20
        
        root.geometry(f"{self.width}x{self.height}+{x_position}+{y_position}")
        root.attributes('-topmost', True)
        root.attributes('-alpha', WINDOW_ALPHA)
        
        # Status label
        status_label = tk.Label(
            root, 
            text="Waiting for MONO...", 
            font=("Arial", 8),
            fg="gray"
        )
        status_label.pack(pady=0)
        
        # Canvas for custom-drawn progress bar + text
        canvas = tk.Canvas(root, width=300, height=35, 
                          bg=root.cget('bg'), highlightthickness=1, 
                          highlightbackground='gray')
        canvas.pack(pady=3, padx=5)
        
        # Draw progress bar background (gray)
        progress_bg = canvas.create_rectangle(
            0, 0, 300, 35,
            fill='#d0d0d0',
            outline='gray'
        )
        
        # Draw progress bar fill (will expand as time passes)
        progress_fill = canvas.create_rectangle(
            0, 0, 0, 35,  # Width starts at 0
            fill='green',
            outline=''
        )
        
        # Time text with black outline for visibility
        time_shadow = canvas.create_text(
            151, 18,
            text="00:00 / 02:00",
            font=("Arial", 18, "bold"),
            fill="black"
        )
        
        time_text = canvas.create_text(
            150, 17,
            text="00:00 / 02:00",
            font=("Arial", 18, "bold"),
            fill="white"
        )
        
        # PSM paths
        path_frame = tk.Frame(root, bg=root.cget('bg'))
        path_frame.pack(pady=1)
        
        path2_label = tk.Label(
            path_frame,
            text="L: 0 mm",
            font=("Arial", 10),
            fg="purple"
        )
        path2_label.pack(side='left', padx=10)
        
        path1_label = tk.Label(
            path_frame,
            text="R: 0 mm",
            font=("Arial", 10),
            fg="blue"
        )
        path1_label.pack(side='right', padx=10)
        
        # Store widgets
        root.status_label = status_label
        root.canvas = canvas
        root.progress_fill = progress_fill
        root.time_shadow = time_shadow
        root.time_text = time_text
        root.path1_label = path1_label
        root.path2_label = path2_label
        
        return root
    
    def update_display(self):
        if self.is_running:
            minutes = int(self.elapsed // 60)
            seconds = int(self.elapsed % 60)
            max_min = int(self.max_time // 60)
            max_sec = int(self.max_time % 60)
            
            time_text = f"{minutes:02d}:{seconds:02d} / {max_min:02d}:{max_sec:02d}"
            status_text = "⚡ Controlling"
            
            path1_mm = self.path_length_psm1 * 1000
            path2_mm = self.path_length_psm2 * 1000
            
            path1_text = f"R: {path1_mm:.0f} mm"
            path2_text = f"L: {path2_mm:.0f} mm"
            
            progress_pct = min(100, (self.elapsed / self.max_time) * 100)
            
            # Calculate progress bar width (0 to 300 pixels)
            bar_width = int(300 * progress_pct / 100)
            
            # Determine color
            if progress_pct < 70:
                bar_color = "green"
            elif progress_pct < 90:
                bar_color = "orange"
            else:
                bar_color = "red"
            
            # Update BOTH windows
            for root in [self.root_left, self.root_right]:
                root.status_label.config(text=status_text, fg="blue")
                
                # Update progress bar fill
                root.canvas.coords(root.progress_fill, 0, 0, bar_width, 35)
                root.canvas.itemconfig(root.progress_fill, fill=bar_color)
                
                # Update text
                root.canvas.itemconfig(root.time_shadow, text=time_text)
                root.canvas.itemconfig(root.time_text, text=time_text)
                
                root.path1_label.config(text=path1_text)
                root.path2_label.config(text=path2_text)
        else:
            for root in [self.root_left, self.root_right]:
                root.status_label.config(text="Waiting for MONO...", fg="gray")
                root.canvas.coords(root.progress_fill, 0, 0, 0, 35)  # Reset bar
        
        self.root_left.after(100, self.update_display)
    
    def start(self):
        if not self.is_running:
            self.is_running = True
    
    def stop(self):
        if self.is_running:
            self.is_running = False
            return self.elapsed
        return None
    
    def tick(self, dt=0.1):
        if self.is_running:
            self.elapsed += dt
    
    def add_path_psm1(self, distance):
        """Add incremental distance to PSM1 path length"""
        self.path_length_psm1 += distance
    
    def add_path_psm2(self, distance):
        """Add incremental distance to PSM2 path length"""
        self.path_length_psm2 += distance
    
    def reset(self):
        self.elapsed = 0.0
        self.path_length_psm1 = 0.0
        self.path_length_psm2 = 0.0
    
    def run(self):
        # Start event loop for left window (right window is synced)
        self.root_left.mainloop()

class TaskTimerNode(Node):
    def __init__(self, gui):
        super().__init__('task_timer_gui')
        self.gui = gui
        
        # Track both states
        self.teleop_enabled = False
        self.mono_pressed = False
        self.was_running = False
        
        # For Hz measurement
        self.last_time = None
        self.current_hz = 0.0
        self.sample_count = 0
        
        # For path length tracking (separate for each PSM)
        self.last_position_psm1 = None
        self.last_position_psm2 = None
        self.pose_sample_count_psm1 = 0
        self.pose_sample_count_psm2 = 0

        # NEW: For orientation tracking
        self.last_orientation_psm1 = None
        self.last_orientation_psm2 = None
        self.last_time_psm1 = None
        self.last_time_psm2 = None
        self.angle_sum_psm1 = 0.0
        self.angle_sum_psm2 = 0.0
        self.angle_time_sum_psm1 = 0.0  # Σ(θ/Δt)
        self.angle_time_sum_psm2 = 0.0
        self.orientation_sample_count_psm1 = 0
        self.orientation_sample_count_psm2 = 0
        
        # Subscribe to teleop state
        self.teleop_sub = self.create_subscription(
            Bool,
            '/console/teleop/enabled',
            self.teleop_callback,
            10)
        
        # Subscribe to MONO pedal (operator_present)
        self.mono_sub = self.create_subscription(
            Joy,
            '/console/operator_present',
            self.mono_callback,
            10)
        
        # Subscribe to PSM1 pose (right arm)
        self.pose_sub_psm1 = self.create_subscription(
            PoseStamped,
            '/PSM1/measured_cp',
            self.pose_callback_psm1,
            10)
        
        # Subscribe to PSM2 pose (left arm)
        self.pose_sub_psm2 = self.create_subscription(
            PoseStamped,
            '/PSM2/measured_cp',
            self.pose_callback_psm2,
            10)
        
        self.timer = self.create_timer(TIMER_INTERVAL, self.update_timer)
        self.get_logger().info('Task timer GUI ready (dual PSM). Turn on the Tele operation & Press MONO pedal to start timing.')

    def teleop_callback(self, msg):
        old_state = self.teleop_enabled
        self.teleop_enabled = msg.data
        
        # Reset timer when teleop is disabled (new trial)
        if old_state and not msg.data:
            duration = self.gui.elapsed
            path1_mm = self.gui.path_length_psm1 * 1000
            path2_mm = self.gui.path_length_psm2 * 1000
            total_mm = path1_mm + path2_mm

            # NEW: Log orientation metrics
            ang_disp1_deg = math.degrees(self.gui.angular_displacement_psm1)
            ang_disp2_deg = math.degrees(self.gui.angular_displacement_psm2)
            rate1_deg = math.degrees(self.gui.orientation_rate_psm1)
            rate2_deg = math.degrees(self.gui.orientation_rate_psm2)

            # Keep metrics in radians
            ang_disp1_rad = self.gui.angular_displacement_psm1
            ang_disp2_rad = self.gui.angular_displacement_psm2
            rate1_rad = self.gui.orientation_rate_psm1
            rate2_rad = self.gui.orientation_rate_psm2

            if duration > 0:
                self.get_logger().info('='*60)
                self.get_logger().info('✅ TRIAL COMPLETE')
                self.get_logger().info('='*60)
                self.get_logger().info(f'Duration: {duration:.2f} s')
                self.get_logger().info(f'')
                self.get_logger().info(f'Path Length:')
                self.get_logger().info(f'  PSM1 (R): {path1_mm:.1f} mm')
                self.get_logger().info(f'  PSM2 (L): {path2_mm:.1f} mm')
                self.get_logger().info(f'  Total:    {total_mm:.1f} mm')
                self.get_logger().info(f'')
                self.get_logger().info(f'Angular Displacement:')
                self.get_logger().info(f'  PSM1 (R): {ang_disp1_deg:.1f}°')
                self.get_logger().info(f'  PSM2 (L): {ang_disp2_deg:.1f}°')
                self.get_logger().info(f'')
                # self.get_logger().info(f'Average Orientation Rate:')
                # self.get_logger().info(f'  PSM1 (R): {rate1_deg:.2f} °/s')
                # self.get_logger().info(f'  PSM2 (L): {rate2_deg:.2f} °/s')
                self.get_logger().info(f'Average Orientation Rate:')
                self.get_logger().info(f'  PSM1 (R): {rate1_rad:.4f} rad/s')
                self.get_logger().info(f'  PSM2 (L): {rate2_rad:.4f} rad/s')
                self.get_logger().info('='*60)
                
                # Reset everything
                self.gui.reset()
                self.sample_count = 0
                self.pose_sample_count_psm1 = 0
                self.pose_sample_count_psm2 = 0
                self.last_position_psm1 = None
                self.last_position_psm2 = None

                # NEW: Reset orientation tracking
                self.last_orientation_psm1 = None
                self.last_orientation_psm2 = None
                self.last_time_psm1 = None
                self.last_time_psm2 = None
                self.angle_sum_psm1 = 0.0
                self.angle_sum_psm2 = 0.0
                self.angle_time_sum_psm1 = 0.0
                self.angle_time_sum_psm2 = 0.0
                self.orientation_sample_count_psm1 = 0
                self.orientation_sample_count_psm2 = 0
        
        self.update_state()
    
    def mono_callback(self, msg):
        # MONO pedal: buttons[0] == 1 when pressed
        if len(msg.buttons) > 0:
            self.mono_pressed = (msg.buttons[0] == 1)
        else:
            self.mono_pressed = False
        
        self.update_state()
    
    def pose_callback_psm1(self, msg):
        """PSM1 pose callback at ~198 Hz"""
        if not self.gui.is_running:
            self.last_position_psm1 = None
            self.last_orientation_psm1 = None
            self.last_time_psm1 = None
            return
        
        current_pos = msg.pose.position
        current_position = [current_pos.x, current_pos.y, current_pos.z]

        current_ori = msg.pose.orientation
        current_orientation = [current_ori.x, current_ori.y, current_ori.z, current_ori.w]
        
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.last_position_psm1 is not None:
            dx = current_position[0] - self.last_position_psm1[0]
            dy = current_position[1] - self.last_position_psm1[1]
            dz = current_position[2] - self.last_position_psm1[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            self.gui.add_path_psm1(distance)
            self.pose_sample_count_psm1 += 1
        
        # Orientation calculation (NEW)
        if self.last_orientation_psm1 is not None and self.last_time_psm1 is not None:
            # Q_{j,j+1} = Q_{j+1} * Q_j^{-1}
            q_j_inv = quaternion_conjugate(self.last_orientation_psm1)
            q_diff = quaternion_multiply(current_orientation, q_j_inv)
            
            # Extract angle θ
            theta = quaternion_to_angle(q_diff)
            
            # Time difference
            dt = current_time - self.last_time_psm1
            
            if dt > 0:
                # Accumulate angular displacement (equation 7)
                self.angle_sum_psm1 += theta
                
                # Accumulate for rate calculation (equation 8)
                self.angle_time_sum_psm1 += theta / dt
                
                self.orientation_sample_count_psm1 += 1
                
                # Update GUI metrics
                self.gui.angular_displacement_psm1 = self.angle_sum_psm1
                
                if self.orientation_sample_count_psm1 > 0:
                    self.gui.orientation_rate_psm1 = self.angle_time_sum_psm1 / self.orientation_sample_count_psm1
        
        # Update last values
        self.last_position_psm1 = current_position
        self.last_orientation_psm1 = current_orientation
        self.last_time_psm1 = current_time
    
    def pose_callback_psm2(self, msg):
        """PSM2 pose callback at ~198 Hz"""
        if not self.gui.is_running:
            self.last_position_psm2 = None
            self.last_orientation_psm2 = None
            self.last_time_psm2 = None
            return
        
        current_pos = msg.pose.position
        current_position = [current_pos.x, current_pos.y, current_pos.z]
        
        current_ori = msg.pose.orientation
        current_orientation = [current_ori.x, current_ori.y, current_ori.z, current_ori.w]
        
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Path length calculation (existing)
        if self.last_position_psm2 is not None:
            dx = current_position[0] - self.last_position_psm2[0]
            dy = current_position[1] - self.last_position_psm2[1]
            dz = current_position[2] - self.last_position_psm2[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            self.gui.add_path_psm2(distance)
            self.pose_sample_count_psm2 += 1
        
        # Orientation calculation (NEW)
        if self.last_orientation_psm2 is not None and self.last_time_psm2 is not None:
            # Q_{j,j+1} = Q_{j+1} * Q_j^{-1}
            q_j_inv = quaternion_conjugate(self.last_orientation_psm2)
            q_diff = quaternion_multiply(current_orientation, q_j_inv)
            
            # Extract angle θ
            theta = quaternion_to_angle(q_diff)
            
            # Time difference
            dt = current_time - self.last_time_psm2
            
            if dt > 0:
                # Accumulate angular displacement (equation 7)
                self.angle_sum_psm2 += theta
                
                # Accumulate for rate calculation (equation 8)
                self.angle_time_sum_psm2 += theta / dt
                
                self.orientation_sample_count_psm2 += 1
                
                # Update GUI metrics
                self.gui.angular_displacement_psm2 = self.angle_sum_psm2
                
                if self.orientation_sample_count_psm2 > 0:
                    self.gui.orientation_rate_psm2 = self.angle_time_sum_psm2 / self.orientation_sample_count_psm2
        
        # Update last values
        self.last_position_psm2 = current_position
        self.last_orientation_psm2 = current_orientation
        self.last_time_psm2 = current_time
    
    def update_state(self):
        should_run = self.teleop_enabled and self.mono_pressed
        
        if should_run and not self.was_running:
            self.gui.start()
            self.sample_count = 0
            self.get_logger().info('⏱️  Timer STARTED (MONO pressed)')
            self.was_running = True
            
        elif not should_run and self.was_running:
            duration = self.gui.stop()
            path1_mm = self.gui.path_length_psm1 * 1000
            path2_mm = self.gui.path_length_psm2 * 1000
            total_mm = path1_mm + path2_mm
            if duration is not None:
                self.get_logger().info(
                    f'⏸️  PAUSED. Time: {duration:.2f}s | PSM1: {path1_mm:.1f}mm | PSM2: {path2_mm:.1f}mm | Total: {total_mm:.1f}mm'
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
                
                # Log every second
                if self.sample_count % TIMER_RATE_HZ == 0:
                    path1_mm = self.gui.path_length_psm1 * 1000
                    path2_mm = self.gui.path_length_psm2 * 1000
                    total_mm = path1_mm + path2_mm
                    
                    # Convert orientation metrics to degrees
                    ang_disp1_deg = math.degrees(self.gui.angular_displacement_psm1)
                    ang_disp2_deg = math.degrees(self.gui.angular_displacement_psm2)
                    rate1_deg = math.degrees(self.gui.orientation_rate_psm1)
                    rate2_deg = math.degrees(self.gui.orientation_rate_psm2)
                    
                    self.get_logger().info(
                        f'Recording @ {self.current_hz:.1f} Hz | Time: {self.gui.elapsed:.1f}s'
                    )
                    self.get_logger().info(
                        f'  Path: R={path1_mm:.1f}mm L={path2_mm:.1f}mm Total={total_mm:.1f}mm'
                    )
                    self.get_logger().info(
                        f'  Orient: R={ang_disp1_deg:.1f}° ({rate1_deg:.1f}°/s) | L={ang_disp2_deg:.1f}° ({rate2_deg:.1f}°/s)'
                    )
                    # show in radians as well
                    self.get_logger().info(
                        f'  Orient (rad): R={self.gui.angular_displacement_psm1:.4f} rad ({self.gui.orientation_rate_psm1:.4f} rad/s) | L={self.gui.angular_displacement_psm2:.4f} rad ({self.gui.orientation_rate_psm2:.4f} rad/s)'
                    )
                    
            self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    
    gui = TimerGUI(max_time=MAX_TIME_SEC)
    node = TaskTimerNode(gui)
    
    def spin_node():
        try:
            rclpy.spin(node)
        except Exception:
            pass
    
    ros_thread = threading.Thread(target=spin_node, daemon=True)
    ros_thread.start()
    
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()