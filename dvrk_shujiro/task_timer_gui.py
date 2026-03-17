#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
import tkinter as tk
from tkinter import ttk
import threading

class TimerGUI:
    def __init__(self, max_time=120):
        self.max_time = max_time
        self.elapsed = 0.0
        self.is_running = False
        
        # Window size
        self.width = 450
        self.height = 120
        
        # Create TWO windows (left and right)
        self.root_left = self.create_window("dVRK Timer (L)", is_left=True)
        self.root_right = self.create_window("dVRK Timer (R)", is_left=False)
        
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
        root.attributes('-alpha', 0.85)
        
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
        time_label.pack(pady=3)
        
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
        progress.pack(pady=5)
        
        # Store widgets as attributes
        root.status_label = status_label
        root.time_label = time_label
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
    
    def reset(self):
        self.elapsed = 0.0
    
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

        # # Subscribe to clutch/pedal (Joy message)
        # self.clutch_sub = self.create_subscription(
        #     Joy,
        #     '/console/clutch',
        #     self.clutch_callback,
        #     10)
        
        self.timer = self.create_timer(0.1, self.update_timer)
        self.get_logger().info('Task timer GUI ready. Press MONO pedal to start timing.')

    def teleop_callback(self, msg):
        old_state = self.teleop_enabled
        self.teleop_enabled = msg.data
        
        # Reset timer when teleop is disabled (new trial)
        if old_state and not msg.data:
            duration = self.gui.elapsed
            if duration > 0:
                self.get_logger().info(f'✅ Trial complete. Total time: {duration:.2f}s')
                self.gui.reset()
        
        self.update_state()
    
    def mono_callback(self, msg):
        # MONO pedal: buttons[0] == 1 when pressed
        if len(msg.buttons) > 0:
            self.mono_pressed = (msg.buttons[0] == 1)
        else:
            self.mono_pressed = False
        
        self.update_state()
    
    def update_state(self):
        # Timer runs only when BOTH teleop enabled AND MONO pressed
        should_run = self.teleop_enabled and self.mono_pressed
        
        if should_run and not self.was_running:
            # Start/resume timing
            self.gui.start()
            self.get_logger().info('⏱️  Timer STARTED (MONO pressed)')
            self.was_running = True
            
        elif not should_run and self.was_running:
            # Pause timing (MONO released)
            duration = self.gui.stop()
            if duration is not None:
                self.get_logger().info(f'⏸️  Timer PAUSED (MONO released). Elapsed: {duration:.2f}s')
            self.was_running = False
    
    def update_timer(self):
        self.gui.tick(0.1)

def main(args=None):
    rclpy.init(args=args)
    
    gui = TimerGUI(max_time=120)
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
        
 



# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Bool
# import tkinter as tk
# from tkinter import ttk
# import threading

# class TimerGUI:
#     def __init__(self, max_time=120):
#         self.max_time = max_time
#         self.elapsed = 0.0
#         self.is_running = False
        
#         # Create window
#         self.root = tk.Tk()
#         self.root.title("dVRK Task Timer")
#         self.root.geometry("450x120")
#         self.root.attributes('-topmost', True)  # Always on top
        
#         # Optional: make it semi-transparent
#         # self.root.attributes('-alpha', 0.9)
        
#         # Status label
#         self.status_label = tk.Label(
#             self.root, 
#             text="Waiting for task...", 
#             font=("Arial", 12),
#             fg="gray"
#         )
#         self.status_label.pack(pady=5)
        
#         # Time label
#         self.time_label = tk.Label(
#             self.root, 
#             text="00:00 / 02:00", 
#             font=("Arial", 28, "bold"),
#             fg="green"
#         )
#         self.time_label.pack(pady=5)
        
#         # Progress bar
#         style = ttk.Style()
#         style.theme_use('default')
#         style.configure("green.Horizontal.TProgressbar", background='green')
#         style.configure("yellow.Horizontal.TProgressbar", background='orange')
#         style.configure("red.Horizontal.TProgressbar", background='red')
        
#         self.progress = ttk.Progressbar(
#             self.root, 
#             length=400, 
#             mode='determinate',
#             maximum=100,
#             style="green.Horizontal.TProgressbar"
#         )
#         self.progress.pack(pady=10)
        
#         # Update loop
#         self.update_display()
    
#     def update_display(self):
#         if self.is_running:
#             # Format time
#             minutes = int(self.elapsed // 60)
#             seconds = int(self.elapsed % 60)
#             max_min = int(self.max_time // 60)
#             max_sec = int(self.max_time % 60)
            
#             self.time_label.config(
#                 text=f"{minutes:02d}:{seconds:02d} / {max_min:02d}:{max_sec:02d}"
#             )
            
#             self.status_label.config(text="Task in progress...", fg="blue")
            
#             # Update progress bar
#             progress_pct = min(100, (self.elapsed / self.max_time) * 100)
#             self.progress['value'] = progress_pct
            
#             # Color coding
#             if progress_pct < 70:
#                 self.time_label.config(fg="green")
#                 self.progress.config(style="green.Horizontal.TProgressbar")
#             elif progress_pct < 90:
#                 self.time_label.config(fg="orange")
#                 self.progress.config(style="yellow.Horizontal.TProgressbar")
#             else:
#                 self.time_label.config(fg="red")
#                 self.progress.config(style="red.Horizontal.TProgressbar")
#         else:
#             self.status_label.config(text="Waiting for task...", fg="gray")
        
#         self.root.after(100, self.update_display)  # Update every 100ms
    
#     def start(self):
#         self.elapsed = 0.0
#         self.is_running = True
    
#     def stop(self):
#         self.is_running = False
    
#     def tick(self, dt=0.1):
#         if self.is_running:
#             self.elapsed += dt
    
#     def run(self):
#         self.root.mainloop()

# class TaskTimerNode(Node):
#     def __init__(self, gui):
#         super().__init__('task_timer_gui')
#         self.gui = gui
        
#         self.subscription = self.create_subscription(
#             Bool,
#             '/console/teleop/enabled',
#             self.teleop_callback,
#             10)
        
#         self.timer = self.create_timer(0.1, self.update_timer)
#         self.get_logger().info('Task timer GUI ready. Time limit: 120s')

#     def teleop_callback(self, msg):
#         if msg.data and not self.gui.is_running:
#             self.gui.start()
#             self.get_logger().info('⏱️  Task STARTED')
#         elif not msg.data and self.gui.is_running:
#             self.gui.stop()
#             self.get_logger().info(f'✅ Task STOPPED. Duration: {self.gui.elapsed:.2f}s')
    
#     def update_timer(self):
#         self.gui.tick(0.1)

# def main(args=None):
#     rclpy.init(args=args)
    
#     # Create GUI
#     gui = TimerGUI(max_time=120)  # 2 minutes
    
#     # Create ROS node
#     node = TaskTimerNode(gui)
    
#     # Run ROS in separate thread
#     def spin_node():
#         try:
#             rclpy.spin(node)
#         except Exception:
#             pass
    
#     ros_thread = threading.Thread(target=spin_node, daemon=True)
#     ros_thread.start()
    
#     # Run GUI in main thread
#     try:
#         gui.run()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         try:
#             node.destroy_node()
#         except Exception:
#             pass
#         try:
#             if rclpy.ok():
#                 rclpy.shutdown()
#         except Exception:
#             pass

# if __name__ == '__main__':
#     main()

# if __name__ == '__main__':
#     main()