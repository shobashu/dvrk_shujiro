"""
Trial Popup Window
------------------
A floating window that appears when a cylinder is lifted from a center peg.
Shows a live progress bar counting up to 1 minutes (MAX_TIME_SEC from config).
Freezes and disappears when the cylinder is placed on a target peg.

Lifecycle:
    1. show_threadsafe()     → called from Arduino thread when CENTER LIFTED
    2. complete_threadsafe() → called from Arduino thread when PEG PLACED
    3. (auto-hide)           → window disappears after 2 seconds
"""

import tkinter as tk
import time

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MAX_TIME_SEC = 60 # time limit (seconds)

# ── Popup display settings ────────────────────────────────────────────────────
POPUP_WIDTH     = 300
POPUP_HEIGHT    = 100
POPUP_SHOW_MS   = 2000      # how long result stays visible before hiding (ms)
UPDATE_MS       = 33        # refresh rate (30fps = 33ms, 20fps = 50ms)

# ── Force result popup settings ───────────────────────────────────────────────
FORCE_POPUP_WIDTH  = 360
FORCE_POPUP_HEIGHT = 290
FORCE_POPUP_SHOW_MS = 4000  # longer than the plain popup — there's a graph to read

# Time thresholds for bar color (seconds) — matches your existing timer logic
THRESHOLD_ORANGE = MAX_TIME_SEC * 0.70   # 70% → orange  (same as PROGRESS_YELLOW_THRESHOLD)
THRESHOLD_RED    = MAX_TIME_SEC * 0.90   # 90% → red     (same as PROGRESS_RED_THRESHOLD)

# Colors — matches your existing dark theme from timer_window.py
COLORS = {
    'bg'         : '#0d0d0d',
    'border'     : '#2a2a2a',
    'bar_bg'     : '#252525',
    'bar_green'  : '#1a3a0d',
    'bar_orange' : '#4a3510',
    'bar_red'    : '#3a0a0a',
    'bar_done'   : '#2a6a10',
    'bar_failed' : '#5a0a0a',    
    'text_dim'   : '#606060',
    'text_bright': '#909090',
    'text_done'  : '#80c060',
    'text_failed': '#c06060',
    'force_line' : '#e0904a',
    'force_peak' : '#e05050',
}


class TrialPopup:
    """
    Floating trial timer popup window.

    Must be created from the main tkinter thread.
    Use show_threadsafe() and complete_threadsafe() from any thread.
    """

    def __init__(self, root):
        """
        Args:
            root : the main tkinter root window (from TimerGUI).
                Needed to schedule callbacks safely on the main thread.
        """
        self._root       = root
        self._win        = None
        self._start_time = None
        self._running    = False
        self._force_canvases = []

        # Score tracking
        self.trial_count   = 0
        self.placed_count  = 0
        self.failed_count  = 0
        self.trial_times   = []
        self.failed_times  = []

        # Color-match (target color facing down) success tracking
        self.color_checked_count = 0   # trials where a down_color/success reading was available
        self.color_success_count = 0

    # ── Thread-safe API (call these from the Arduino background thread) ────────

    def show_threadsafe(self):
        """Call from Arduino thread when CENTER LIFTED is detected."""
        self._root.after(0, self._show)

    def complete_threadsafe(self):
        """Call from Arduino thread when PEG PLACED is detected."""
        self._root.after(0, self._complete)

    def show_force_result_threadsafe(self, trial_num, times, forces, peak, mean,
                                      orientation_deg=None, orientation_ok=None,
                                      down_color=None, success=None):
        """Call from Arduino thread when 'Object returned successfully.' is detected.

        times/forces: parallel lists — elapsed seconds since trial start, and
        the ATI sensor's total force magnitude ‖F‖ (N) at that instant.
        orientation_deg/orientation_ok: optional cylinder orientation angle (0° = upright)
        and whether it was within tolerance, measured at the "Target hit" event mid-trial.
        down_color: "blue" or "white" — whichever color was facing down at that same moment.
        success: whether down_color matched the trial's target color — the actual
        placement-correctness signal used for the running success rate.
        Pass None for all four if no orientation reading is available for this trial.
        """
        self._root.after(0, lambda: self._show_force_result(
            trial_num, times, forces, peak, mean, orientation_deg, orientation_ok,
            down_color, success))

    # ── Score summary (call at end of session) ────────────────────────────────

    def print_final_score(self):
        """Print session summary to terminal. Called by main.py on shutdown."""
        print("\n" + "=" * 40)
        print("        FINAL SESSION SCORE")
        print("=" * 40)
        print(f"  Trials started   : {self.trial_count}")
        print(f"  Placed        : {self.placed_count}")
        print(f"  Failed        : {self.failed_count}")
        
        if self.trial_times:
            avg  = sum(self.trial_times) / len(self.trial_times)
            best = min(self.trial_times)
            print(f"\n  Successful trials:")
            print(f"    Average time   : {self._fmt(avg)}")
            print(f"    Best time      : {self._fmt(best)}")
        
        if self.failed_times:
            avg_fail = sum(self.failed_times) / len(self.failed_times)
            print(f"\n  Failed trials:")
            print(f"    Average time   : {self._fmt(avg_fail)}")
        
        # Calculate success rate
        if self.trial_count > 0:
            success_rate = (self.placed_count / self.trial_count) * 100
            print(f"\n  Success rate     : {success_rate:.1f}%")

        if self.color_checked_count > 0:
            color_rate = (self.color_success_count / self.color_checked_count) * 100
            print(f"  Color-match rate : {color_rate:.1f}%  "
                  f"({self.color_success_count}/{self.color_checked_count})")

        print("=" * 40 + "\n")

    # ── Internal — main thread only ───────────────────────────────────────────

    def _show(self):
        """Create and show the popup on ALL THREE MONITORS simultaneously."""
        # Destroy any leftover popups from previous trial
        if hasattr(self, '_wins') and self._wins:
            for win in self._wins:
                try:
                    win.destroy()
                except Exception:
                    pass

        self._start_time = time.time()
        self._running    = True
        self.trial_count += 1

        # ── Monitor positions from xrandr ─────────────────────────────────────────
        # Monitor 0 (HDMI-1): 3840x2160 at +1280+0
        # Monitor 1 (DP-0):   640x480   at +640+0
        # Monitor 2 (DP-2):   640x480   at +0+0
        
        monitors = [
            {'name': 'HDMI-1', 'x': 1280, 'y': 0, 'width': 3840, 'height': 2160},  # Monitor 0
            {'name': 'DP-0',   'x': 640,  'y': 0, 'width': 640,  'height': 480},   # Monitor 1
            {'name': 'DP-2',   'x': 0,    'y': 0, 'width': 640,  'height': 480},   # Monitor 2
        ]

        # Top-left offset (margin from edges)
        MARGIN_X = 20
        MARGIN_Y = 20

        # Create one popup window per monitor
        self._wins = []
        self._canvases = []
        self._bars = []
        self._time_labels = []
        self._titles = []
        self._statuses = []

        for monitor in monitors:
            win = self._create_popup_window(
                x=monitor['x'] + MARGIN_X,
                y=monitor['y'] + MARGIN_Y
            )
            self._wins.append(win)

        # Start the update loop
        self._update()


    def _create_popup_window(self, x, y):
        """Helper function to create a single popup window at position (x, y)."""
        win = tk.Toplevel(self._root)
        win.title("")
        win.configure(bg=COLORS['bg'])
        win.attributes('-topmost', True)
        win.attributes('-alpha', 0.90)
        win.overrideredirect(True)
        win.lift()
        win.focus_force()

        win.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}+{x}+{y}")

        # Thin coloured border
        border = tk.Frame(win, bg=COLORS['border'], padx=1, pady=1)
        border.pack(fill='both', expand=True)

        inner = tk.Frame(border, bg=COLORS['bg'], padx=12, pady=10)
        inner.pack(fill='both', expand=True)

        # Title
        title = tk.Label(
            inner,
            text=f"TRIAL {self.trial_count} IN PROGRESS",
            font=("Arial", 9, "bold"),
            fg=COLORS['text_dim'],
            bg=COLORS['bg']
        )
        title.pack(anchor='w')
        self._titles.append(title)

        # Progress bar canvas
        bar_width = POPUP_WIDTH - 28
        canvas = tk.Canvas(
            inner,
            width=bar_width,
            height=28,
            bg=COLORS['bg'],
            highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        canvas.pack(pady=(6, 4))
        self._canvases.append(canvas)

        # Bar background
        canvas.create_rectangle(
            0, 0, bar_width, 28,
            fill=COLORS['bar_bg'],
            outline=COLORS['border']
        )

        # Bar fill (starts empty, grows right)
        bar = canvas.create_rectangle(
            0, 0, 0, 28,
            fill=COLORS['bar_green'],
            outline=''
        )
        self._bars.append(bar)

        # Time text centered on bar
        time_label = canvas.create_text(
            bar_width // 2, 14,
            text="00:00.0",
            font=("Arial", 11, "bold"),
            fill=COLORS['text_bright']
        )
        self._time_labels.append(time_label)

        # Status text below bar
        status = tk.Label(
            inner,
            text="Place on target peg & back",
            font=("Arial", 9),
            fg=COLORS['text_dim'],
            bg=COLORS['bg']
        )
        status.pack(anchor='w')
        self._statuses.append(status)

        return win

    def _update(self):
        """Refresh all bars and times every UPDATE_MS milliseconds."""
        if not self._running or not self._wins:
            return

        elapsed  = time.time() - self._start_time
        bar_w    = POPUP_WIDTH - 28
        fill_w   = int(bar_w * min(1.0, elapsed / MAX_TIME_SEC))

        # ── Check for timeout (FAILURE) ───────────────────────────────────────
        if elapsed >= MAX_TIME_SEC:
            self._fail()  # Trigger failure
            return

        # Bar color changes with time
        if elapsed < THRESHOLD_ORANGE:
            color = COLORS['bar_green']
        elif elapsed < THRESHOLD_RED:
            color = COLORS['bar_orange']
        else:
            color = COLORS['bar_red']

        # Update ALL canvases
        for i, canvas in enumerate(self._canvases):
            canvas.coords(self._bars[i], 0, 0, fill_w, 28)
            canvas.itemconfig(self._bars[i], fill=color)
            canvas.itemconfig(self._time_labels[i], text=self._fmt(elapsed))

        # Schedule next update
        self._root.after(UPDATE_MS, self._update)

    def _fail(self):
        """Called when time limit is exceeded. Show failure message."""
        if not self._running:
            return

        self._running = False
        duration      = time.time() - self._start_time
        self.failed_count += 1
        self.failed_times.append(duration)

        # Terminal log
        print(f"\n[Trial {self.trial_count}] FAILED - Time limit exceeded ({self._fmt(duration)})")
        print(f"[Score] {self.placed_count} placed / {self.failed_count} failed / {self.trial_count} total\n")

        if not self._wins:
            return

        # Update ALL windows to show failure
        for i in range(len(self._wins)):
            self._titles[i].config(text="  TIME LIMIT EXCEEDED!", fg='#8a2a2a')
            self._statuses[i].config(
                text=f"Trial {self.trial_count} FAILED — {self._fmt(duration)}",
                fg=COLORS['text_failed']
            )

            # Fill bar completely in failure color
            bar_w = POPUP_WIDTH - 28
            self._canvases[i].coords(self._bars[i], 0, 0, bar_w, 28)
            self._canvases[i].itemconfig(self._bars[i], fill=COLORS['bar_failed'])
            self._canvases[i].itemconfig(
                self._time_labels[i],
                text=self._fmt(MAX_TIME_SEC),
                fill=COLORS['text_failed']
            )

        # Auto-hide after POPUP_SHOW_MS
        self._root.after(POPUP_SHOW_MS, self._hide)

    def _complete(self):
        """Freeze the display and show placement confirmation on ALL monitors."""
        if not self._running:
            return  # Already completed or failed

        self._running = False
        duration      = time.time() - self._start_time
        
        # ── Check if placement happened after timeout ────────────────────────
        if duration >= MAX_TIME_SEC:
            # Late placement - still counts as failure
            print(f"\n[Trial {self.trial_count}] Placed after timeout ({self._fmt(duration)}) - counted as FAILURE")
            self.failed_count += 1
            self.failed_times.append(duration)
            self._fail()  # Show failure UI
            return

        self.placed_count += 1
        self.trial_times.append(duration)

        # Terminal log
        print(f"\n[Trial {self.trial_count}] Placed in {self._fmt(duration)}")
        print(f"[Score] {self.placed_count}/{self.trial_count} placements\n")

        if not self._wins:
            return

        # Update ALL windows
        for i in range(len(self._wins)):
            self._titles[i].config(text="PLACED !", fg='#4a8a2a')
            self._statuses[i].config(
                text=f"Trial {self.trial_count}  —  {self._fmt(duration)}",
                fg=COLORS['text_bright']
            )

            # Freeze bar at current fill, switch to done color
            bar_w  = POPUP_WIDTH - 28
            fill_w = int(bar_w * min(1.0, duration / MAX_TIME_SEC))
            self._canvases[i].coords(self._bars[i], 0, 0, fill_w, 28)
            self._canvases[i].itemconfig(self._bars[i], fill=COLORS['bar_done'])
            self._canvases[i].itemconfig(
                self._time_labels[i],
                text=self._fmt(duration),
                fill=COLORS['text_done']
            )

        # Auto-hide after POPUP_SHOW_MS
        self._root.after(POPUP_SHOW_MS, self._hide)

    def _show_force_result(self, trial_num, times, forces, peak, mean,
                            orientation_deg=None, orientation_ok=None,
                            down_color=None, success=None):
        """Replace the progress popup with a force-vs-time graph + peak/mean, on all monitors."""
        self._running = False
        self._hide()

        monitors = [
            {'name': 'HDMI-1', 'x': 1280, 'y': 0, 'width': 3840, 'height': 2160},
            {'name': 'DP-0',   'x': 640,  'y': 0, 'width': 640,  'height': 480},
            {'name': 'DP-2',   'x': 0,    'y': 0, 'width': 640,  'height': 480},
        ]
        MARGIN_X = 20
        MARGIN_Y = 20

        self._wins = []
        for monitor in monitors:
            win = self._create_force_popup_window(
                x=monitor['x'] + MARGIN_X, y=monitor['y'] + MARGIN_Y,
                trial_num=trial_num, times=times, forces=forces, peak=peak, mean=mean,
                orientation_deg=orientation_deg, orientation_ok=orientation_ok,
                down_color=down_color, success=success
            )
            self._wins.append(win)

        print(f"\n[Trial {trial_num}] Peak force: {peak:.2f} N  |  Mean force: {mean:.2f} N")
        if orientation_deg is not None:
            print(f"[Trial {trial_num}] Orientation: {'OK' if orientation_ok else 'OFF'} "
                  f"({orientation_deg:+.1f}°)  —  {down_color} facing down")

        if success is not None:
            self.color_checked_count += 1
            if success:
                self.color_success_count += 1
            rate = 100.0 * self.color_success_count / self.color_checked_count
            print(f"[Trial {trial_num}] Placement: {'SUCCESS' if success else 'FAIL'}")
            print(f"[Score] Color-match success rate: "
                  f"{self.color_success_count}/{self.color_checked_count} ({rate:.1f}%)")

        self._root.after(FORCE_POPUP_SHOW_MS, self._hide)

    def _create_force_popup_window(self, x, y, trial_num, times, forces, peak, mean,
                                    orientation_deg=None, orientation_ok=None,
                                    down_color=None, success=None):
        """Helper to build one force-result popup (title + peak/mean + orientation + graph)."""
        win = tk.Toplevel(self._root)
        win.title("")
        win.configure(bg=COLORS['bg'])
        win.attributes('-topmost', True)
        win.attributes('-alpha', 0.92)
        win.overrideredirect(True)
        win.lift()
        win.geometry(f"{FORCE_POPUP_WIDTH}x{FORCE_POPUP_HEIGHT}+{x}+{y}")

        border = tk.Frame(win, bg=COLORS['border'], padx=1, pady=1)
        border.pack(fill='both', expand=True)
        inner = tk.Frame(border, bg=COLORS['bg'], padx=10, pady=8)
        inner.pack(fill='both', expand=True)

        title = tk.Label(
            inner, text=f"TRIAL {trial_num} — FORCE APPLIED",
            font=("Arial", 10, "bold"), fg=COLORS['text_bright'], bg=COLORS['bg']
        )
        title.pack(anchor='w')

        stats = tk.Label(
            inner, text=f"Peak: {peak:.2f} N     Mean: {mean:.2f} N",
            font=("Arial", 11, "bold"), fg=COLORS['text_done'], bg=COLORS['bg']
        )
        stats.pack(anchor='w', pady=(2, 2))

        if orientation_deg is not None:
            if success is not None:
                ori_color = COLORS['text_done'] if success else COLORS['text_failed']
                ori_text  = f"{'SUCCESS' if success else 'FAIL'}  —  {down_color} down  ({orientation_deg:+.1f}°)"
            else:
                ori_color = COLORS['text_dim']
                ori_text  = f"Orientation: {orientation_deg:+.1f}°"
                if down_color is not None:
                    ori_text += f"  ({down_color} down)"
            ori_label = tk.Label(
                inner, text=ori_text,
                font=("Arial", 11, "bold"), fg=ori_color, bg=COLORS['bg']
            )
            ori_label.pack(anchor='w', pady=(0, 4))

        fig = Figure(figsize=(FORCE_POPUP_WIDTH / 100, 1.6), dpi=100)
        fig.patch.set_facecolor(COLORS['bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg'])

        if times:
            ax.plot(times, forces, color=COLORS['force_line'], linewidth=1.8)
            peak_idx = forces.index(max(forces))
            ax.plot(times[peak_idx], forces[peak_idx], 'o',
                    color=COLORS['force_peak'], markersize=5, zorder=3)
        else:
            ax.text(0.5, 0.5, "no force data", ha='center', va='center',
                    color=COLORS['text_dim'], fontsize=8, transform=ax.transAxes)

        ax.tick_params(colors=COLORS['text_dim'], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
        ax.set_xlabel("time (s)", color=COLORS['text_dim'], fontsize=7)
        ax.set_ylabel("‖F‖ (N)", color=COLORS['text_dim'], fontsize=7)
        fig.tight_layout(pad=0.3)

        canvas = FigureCanvasTkAgg(fig, master=inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self._force_canvases.append(canvas)

        return win

    def _hide(self):
        """Destroy all popup windows."""
        if hasattr(self, '_wins') and self._wins:
            for win in self._wins:
                try:
                    win.destroy()
                except Exception:
                    pass
            self._wins = []
            self._canvases = []
            self._bars = []
            self._time_labels = []
            self._titles = []
            self._statuses = []
        self._force_canvases = []

    def _fmt(self, seconds):
        """Format seconds as MM:SS.T  e.g. 75.3 → '01:15.3'"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        t = int((seconds % 1) * 10)
        return f"{m:02d}:{s:02d}.{t}"
    
    def on_arduino_event(self, event):
        """
        Callback passed to ArduinoReader.
        Called from Arduino background thread on every valid event.

        event.event_type : "LIFTED" → cylinder picked up, start timer
                        "DATA"   → trial complete, stop timer
        """
        if event.event_type == "LIFTED":
            print(f"[Popup] Cylinder lifted — starting timer")
            self.show_threadsafe()

        elif event.event_type == "DATA":
            print(f"[Popup] Trial {event.trial} complete — peg {event.target_peg} {event.color}")
            self.complete_threadsafe()
