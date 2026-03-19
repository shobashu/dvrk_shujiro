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

# ============ Configuration ============
TIMER_RATE_HZ = 100        # Sampling rate (Hz)
TIMER_INTERVAL = 1.0 / TIMER_RATE_HZ
MAX_TIME_SEC = 120        # Time limit (seconds)
WINDOW_ALPHA = 0.85       # Transparency (0.0-1.0)
# =======================================

class TimerGUI:
    def __init__(self, max_time=120):
        self.max_time = max_time
        self.elapsed = 0.0
        self.path_length_psm1 = 0.0  # PSM1 path in meters
        self.path_length_psm2 = 0.0  # PSM2 path in meters
        self.is_running = False
        
        # Window size
        self.width = 350
        self.height = 180  # Increased for path length display
        
        # Create TWO windows (left and right)
        self.root_left = self.create_window("dVRK Timer", is_left=True)
        self.root_right = self.create_window("dVRK Timer", is_left=False)
        
        # Update loop
        self.update_display()
    
    def create_window(self, title, is_left=True):
        root = tk.Tk()
        root.title(title)
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Assume stereo display is split in half
        half_width = screen_width
        
        if is_left:
            # Position in top-right of LEFT half
            x_position = half_width - self.width - 20
            y_position = 20
        else:
            # Position in top-right of RIGHT half
            x_position = screen_width - self.width - 20
            y_position = 20
        
        root.geometry(f"{self.width}x{self.height}+{x_position}+{y_position}")
        root.attributes('-topmost', True)
        root.attributes('-alpha', WINDOW_ALPHA)
        
        # Status label
        status_label = tk.Label(
            root, 
            text="Waiting for MONO pedal...", 
            font=("Arial", 10),
            fg="gray"
        )
        status_label.pack(pady=3)
        
        # Time label
        time_label = tk.Label(
            root, 
            text="00:00 / 02:00", 
            font=("Arial", 22, "bold"),
            fg="green"
        )
        time_label.pack(pady=2)
        
        # PSM1 path length label
        path1_label = tk.Label(
            root,
            text="PSM (R): 0.0 mm",
            font=("Arial", 14),
            fg="blue"
        )
        path1_label.pack(pady=1)
        
        # PSM2 path length label
        path2_label = tk.Label(
            root,
            text="PSM (L): 0.0 mm",
            font=("Arial", 14),
            fg="purple"
        )
        path2_label.pack(pady=1)
        
        # Total path label
        total_path_label = tk.Label(
            root,
            text="Total: 0.0 mm",
            font=("Arial", 12, "bold"),
            fg="darkgreen"
        )
        total_path_label.pack(pady=1)
        
        # Progress bar
        style = ttk.Style(root)
        style.theme_use('default')
        style.configure("green.Horizontal.TProgressbar", background='green')
        style.configure("yellow.Horizontal.TProgressbar", background='orange')
        style.configure("red.Horizontal.TProgressbar", background='red')
        
        progress = ttk.Progressbar(
            root, 
            length=300, 
            mode='determinate',
            maximum=100,
            style="green.Horizontal.TProgressbar"
        )
        progress.pack(pady=3)
        
        # Store widgets as attributes
        root.status_label = status_label
        root.time_label = time_label
        root.path1_label = path1_label
        root.path2_label = path2_label
        root.total_path_label = total_path_label
        root.progress = progress
        
        return root
    
    def update_display(self):
        if self.is_running:
            minutes = int(self.elapsed // 60)
            seconds = int(self.elapsed % 60)
            max_min = int(self.max_time // 60)
            max_sec = int(self.max_time % 60)
            
            time_text = f"{minutes:02d}:{seconds:02d} / {max_min:02d}:{max_sec:02d}"
            status_text = "⚡ Controlling (MONO)"
            status_color = "blue"
            
            # Path lengths in mm
            path1_mm = self.path_length_psm1 * 1000
            path2_mm = self.path_length_psm2 * 1000
            total_mm = (self.path_length_psm1 + self.path_length_psm2) * 1000
            
            path1_text = f"PSM1: {path1_mm:.1f} mm"
            path2_text = f"PSM2: {path2_mm:.1f} mm"
            total_text = f"Total: {total_mm:.1f} mm"
            
            progress_pct = min(100, (self.elapsed / self.max_time) * 100)
            
            if progress_pct < 70:
                time_color = "green"
                bar_style = "green.Horizontal.TProgressbar"
            elif progress_pct < 90:
                time_color = "orange"
                bar_style = "yellow.Horizontal.TProgressbar"
            else:
                time_color = "red"
                bar_style = "red.Horizontal.TProgressbar"
            
            # Update BOTH windows
            for root in [self.root_left, self.root_right]:
                root.time_label.config(text=time_text, fg=time_color)
                root.status_label.config(text=status_text, fg=status_color)
                root.path1_label.config(text=path1_text)
                root.path2_label.config(text=path2_text)
                root.total_path_label.config(text=total_text)
                root.progress['value'] = progress_pct
                root.progress.config(style=bar_style)
        else:
            # Update BOTH windows
            for root in [self.root_left, self.root_right]:
                root.status_label.config(text="Waiting for MONO...", fg="gray")
        
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
        
        # Subscribe to PSM1 pose
        self.pose_sub_psm1 = self.create_subscription(
            PoseStamped,
            '/PSM1/measured_cp',
            self.pose_callback_psm1,
            10)
        
        # Subscribe to PSM2 pose
        self.pose_sub_psm2 = self.create_subscription(
            PoseStamped,
            '/PSM2/measured_cp',
            self.pose_callback_psm2,
            10)
        
        self.timer = self.create_timer(TIMER_INTERVAL, self.update_timer)
        self.get_logger().info('Task timer GUI ready (dual PSM). Press MONO pedal to start timing.')

    def teleop_callback(self, msg):
        old_state = self.teleop_enabled
        self.teleop_enabled = msg.data
        
        # Reset timer when teleop is disabled (new trial)
        if old_state and not msg.data:
            duration = self.gui.elapsed
            path1_mm = self.gui.path_length_psm1 * 1000
            path2_mm = self.gui.path_length_psm2 * 1000
            total_mm = path1_mm + path2_mm
            if duration > 0:
                self.get_logger().info(f'✅ Trial complete. Time: {duration:.2f}s')
                self.get_logger().info(f'   PSM1 path: {path1_mm:.1f} mm ({self.pose_sample_count_psm1} samples)')
                self.get_logger().info(f'   PSM2 path: {path2_mm:.1f} mm ({self.pose_sample_count_psm2} samples)')
                self.get_logger().info(f'   Total path: {total_mm:.1f} mm')
                self.gui.reset()
                self.sample_count = 0
                self.pose_sample_count_psm1 = 0
                self.pose_sample_count_psm2 = 0
                self.last_position_psm1 = None
                self.last_position_psm2 = None
        
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
            return
        
        current_pos = msg.pose.position
        current_position = [current_pos.x, current_pos.y, current_pos.z]
        
        if self.last_position_psm1 is not None:
            dx = current_position[0] - self.last_position_psm1[0]
            dy = current_position[1] - self.last_position_psm1[1]
            dz = current_position[2] - self.last_position_psm1[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            self.gui.add_path_psm1(distance)
            self.pose_sample_count_psm1 += 1
        
        self.last_position_psm1 = current_position
    
    def pose_callback_psm2(self, msg):
        """PSM2 pose callback at ~198 Hz"""
        if not self.gui.is_running:
            self.last_position_psm2 = None
            return
        
        current_pos = msg.pose.position
        current_position = [current_pos.x, current_pos.y, current_pos.z]
        
        if self.last_position_psm2 is not None:
            dx = current_position[0] - self.last_position_psm2[0]
            dy = current_position[1] - self.last_position_psm2[1]
            dz = current_position[2] - self.last_position_psm2[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            self.gui.add_path_psm2(distance)
            self.pose_sample_count_psm2 += 1
        
        self.last_position_psm2 = current_position
    
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
                    self.get_logger().info(
                        f'Recording @ {self.current_hz:.1f} Hz | Time: {self.gui.elapsed:.1f}s | '
                        f'PSM (R): {path1_mm:.1f}mm | PSM (L): {path2_mm:.1f}mm | Total: {total_mm:.1f}mm'
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