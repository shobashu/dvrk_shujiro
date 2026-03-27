"""Timer GUI window"""
import tkinter as tk
from ..config import *


class TimerWindow:
    """Single timer window (left or right)"""
    
    # Dim color scheme
    COLORS = {
        'bg': '#0d0d0d',              # Very dark background
        'canvas_bg': '#1a1a1a',       # Canvas background
        'canvas_fill': '#252525',     # Progress bar background
        'border': '#2a2a2a',          # Borders
        'text_primary': '#606060',    # Main text (dimmed)
        'text_secondary': '#404040',  # Secondary text
        'text_shadow': '#000000',     # Text shadow
        'status_waiting': '#3a3a3a',  # Waiting status
        'status_active': '#4a5a6a',   # Active status (dim blue)
        'green': '#1a3a0d',           # Dim green
        'orange': '#4a3510',          # Dim orange
        'red': '#3a0a0a',             # Dim red
        'psm1': '#1a2a4a',            # Dim blue for PSM1
        'psm2': '#4a1a2a',            # Dim pink for PSM2
    }
    
    def __init__(self, title, is_left=True):
        self.root = tk.Tk()
        self.root.title(title)
        
        # Dark background
        self.root.configure(bg=self.COLORS['bg'])
        
        screen_width = self.root.winfo_screenwidth()
        
        # Position
        if is_left:
            x_position = screen_width - WINDOW_WIDTH - 20
        else:
            x_position = screen_width - WINDOW_WIDTH - 20
        
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x_position}+20")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.75)  # Adjust 0.5-1.0 for transparency
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Status label (dimmed)
        self.status_label = tk.Label(
            self.root, 
            text="Waiting for MONO...", 
            font=FONT_STATUS, 
            fg=self.COLORS['status_waiting'],
            bg=self.COLORS['bg']
        )
        self.status_label.pack(pady=0)
        
        # Progress canvas (dark)
        self.canvas = tk.Canvas(
            self.root, 
            width=300, 
            height=35,
            bg=self.COLORS['bg'],
            highlightthickness=1,
            highlightbackground=self.COLORS['border']
        )
        self.canvas.pack(pady=3, padx=5)
        
        # Dark background bar
        self.canvas.create_rectangle(
            0, 0, 300, 35, 
            fill=self.COLORS['canvas_fill'], 
            outline=self.COLORS['border']
        )
        
        self.progress_fill = self.canvas.create_rectangle(
            0, 0, 0, 35, 
            fill=self.COLORS['green'], 
            outline=''
        )
        
        # Dimmed text
        self.time_shadow = self.canvas.create_text(
            151, 18, 
            text="00:00 / 02:00",
            font=FONT_TIME, 
            fill=self.COLORS['text_shadow']
        )
        self.time_text = self.canvas.create_text(
            150, 17, 
            text="00:00 / 02:00",
            font=FONT_TIME, 
            fill=self.COLORS['text_primary']
        )
        
        # Path labels (dark background)
        path_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        path_frame.pack(pady=1)
        
        self.path_left = tk.Label(
            path_frame, 
            text="L: 0 mm", 
            font=FONT_PATH, 
            fg=self.COLORS['psm2'],
            bg=self.COLORS['bg']
        )
        self.path_left.pack(side='left', padx=10)
        
        self.path_right = tk.Label(
            path_frame, 
            text="R: 0 mm",
            font=FONT_PATH, 
            fg=self.COLORS['psm1'],
            bg=self.COLORS['bg']
        )
        self.path_right.pack(side='right', padx=10)
    
    def update(self, time_text, status_text, path1_mm, path2_mm, 
              progress_pct, bar_color):
        """Update all display elements"""
        # Update status with dimmed color
        self.status_label.config(
            text=status_text, 
            fg=self.COLORS['status_active']
        )
        
        # Update progress bar
        bar_width = int(300 * progress_pct / 100)
        self.canvas.coords(self.progress_fill, 0, 0, bar_width, 35)
        self.canvas.itemconfig(self.progress_fill, fill=bar_color)
        
        # Update time text
        self.canvas.itemconfig(self.time_shadow, text=time_text)
        self.canvas.itemconfig(self.time_text, text=time_text)
        
        # Update path labels
        self.path_right.config(text=f"R: {path1_mm:.0f} mm")
        self.path_left.config(text=f"L: {path2_mm:.0f} mm")
    
    def reset_display(self):
        """Reset to waiting state"""
        self.status_label.config(
            text="Waiting for MONO...", 
            fg=self.COLORS['status_waiting']
        )
        self.canvas.coords(self.progress_fill, 0, 0, 0, 35)


class TimerGUI:
    """Dual-window timer GUI manager"""
    
    def __init__(self, max_time=MAX_TIME_SEC):
        self.max_time = max_time
        self.elapsed = 0.0
        self.is_running = False
        
        # Metrics (will be updated by node)
        self.path_length_psm1 = 0.0
        self.path_length_psm2 = 0.0
        self.angular_displacement_psm1 = 0.0
        self.angular_displacement_psm2 = 0.0
        self.orientation_rate_psm1 = 0.0
        self.orientation_rate_psm2 = 0.0
        
        # Create windows
        self.window_left = TimerWindow("dVRK Timer", is_left=True)
        self.window_right = TimerWindow("dVRK Timer", is_left=False)
        
        self._start_update_loop()
    
    def _start_update_loop(self):
        """Start periodic display update"""
        self._update_display()
    
    def _update_display(self):
        if self.is_running:
            # Format time
            minutes = int(self.elapsed // 60)
            seconds = int(self.elapsed % 60)
            max_min = int(self.max_time // 60)
            max_sec = int(self.max_time % 60)
            time_text = f"{minutes:02d}:{seconds:02d} / {max_min:02d}:{max_sec:02d}"
            status_text = "⚡ Controlling"
            
            # Get metrics
            path1_mm = self.path_length_psm1 * 1000
            path2_mm = self.path_length_psm2 * 1000
            
            # Calculate progress
            progress_pct = min(100, (self.elapsed / self.max_time) * 100)
            
            # Use dimmed colors
            if progress_pct < PROGRESS_YELLOW_THRESHOLD:
                bar_color = TimerWindow.COLORS['green']
            elif progress_pct < PROGRESS_RED_THRESHOLD:
                bar_color = TimerWindow.COLORS['orange']
            else:
                bar_color = TimerWindow.COLORS['red']
            
            # Update both windows
            for window in [self.window_left, self.window_right]:
                window.update(time_text, status_text, path1_mm, path2_mm,
                            progress_pct, bar_color)
        else:
            for window in [self.window_left, self.window_right]:
                window.reset_display()
        
        self.window_left.root.after(100, self._update_display)
    
    def start(self):
        if not self.is_running:
            self.is_running = True
    
    def stop(self):
        if self.is_running:
            self.is_running = False
            return self.elapsed
        return None
    
    def tick(self, dt):
        if self.is_running:
            self.elapsed += dt
    
    def add_path_psm1(self, distance):
        self.path_length_psm1 += distance
    
    def add_path_psm2(self, distance):
        self.path_length_psm2 += distance
    
    def reset(self):
        self.elapsed = 0.0
        self.path_length_psm1 = 0.0
        self.path_length_psm2 = 0.0
        self.angular_displacement_psm1 = 0.0
        self.angular_displacement_psm2 = 0.0
        self.orientation_rate_psm1 = 0.0
        self.orientation_rate_psm2 = 0.0
    
    def run(self):
        self.window_left.root.mainloop()