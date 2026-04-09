"""
Trial Popup Window
------------------
A floating window that appears when a cylinder is lifted from a center peg.
Shows a live progress bar counting up to 2 minutes (MAX_TIME_SEC from config).
Freezes and disappears when the cylinder is placed on a target peg.

Lifecycle:
    1. show_threadsafe()     → called from Arduino thread when CENTER LIFTED
    2. complete_threadsafe() → called from Arduino thread when PEG PLACED
    3. (auto-hide)           → window disappears after 2 seconds
"""

import tkinter as tk
import time

try:
    from ..config import MAX_TIME_SEC
except ImportError:
    MAX_TIME_SEC = 120


# ── Popup display settings ────────────────────────────────────────────────────
POPUP_WIDTH     = 340
POPUP_HEIGHT    = 130
POPUP_SHOW_MS   = 2000      # how long result stays visible before hiding (ms)
UPDATE_MS       = 50        # refresh rate (50ms = 20fps)

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
    'text_dim'   : '#606060',
    'text_bright': '#909090',
    'text_done'  : '#80c060',
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
        self._win        = None         # the Toplevel popup window
        self._start_time = None         # time.time() when trial started
        self._running    = False        # is the timer currently counting?

        # Score tracking
        self.trial_count   = 0
        self.placed_count  = 0
        self.trial_times   = []         # duration (seconds) of each completed trial

    # ── Thread-safe API (call these from the Arduino background thread) ────────

    def show_threadsafe(self):
        """Call from Arduino thread when CENTER LIFTED is detected."""
        self._root.after(0, self._show)

    def complete_threadsafe(self):
        """Call from Arduino thread when PEG PLACED is detected."""
        self._root.after(0, self._complete)

    # ── Score summary (call at end of session) ────────────────────────────────

    def print_final_score(self):
        """Print session summary to terminal. Called by main.py on shutdown."""
        print("\n" + "=" * 40)
        print("        FINAL SESSION SCORE")
        print("=" * 40)
        print(f"  Trials started   : {self.trial_count}")
        print(f"  Cylinders placed : {self.placed_count}")
        if self.trial_times:
            avg  = sum(self.trial_times) / len(self.trial_times)
            best = min(self.trial_times)
            print(f"  Average time     : {self._fmt(avg)}")
            print(f"  Best time        : {self._fmt(best)}")
        print("=" * 40 + "\n")

    # ── Internal — main thread only ───────────────────────────────────────────

    def _show(self):
        """Create and show the popup. Start counting."""
        # Destroy any leftover popup from previous trial
        if self._win is not None:
            try:
                self._win.destroy()
            except Exception:
                pass

        self._start_time = time.time()
        self._running    = True
        self.trial_count += 1

        # ── Build the window ──────────────────────────────────────────────────
        self._win = tk.Toplevel(self._root)
        self._win.title("")
        self._win.configure(bg=COLORS['bg'])
        self._win.attributes('-topmost', True)
        self._win.attributes('-alpha', 0.90)
        self._win.overrideredirect(True)    # borderless, no title bar

        # Position: bottom-center of screen
        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        x  = (sw - POPUP_WIDTH) // 2
        y  = sh - POPUP_HEIGHT - 60
        self._win.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}+{x}+{y}")

        # Thin coloured border
        border = tk.Frame(self._win, bg=COLORS['border'], padx=1, pady=1)
        border.pack(fill='both', expand=True)

        inner = tk.Frame(border, bg=COLORS['bg'], padx=12, pady=10)
        inner.pack(fill='both', expand=True)

        # Title
        self._title = tk.Label(
            inner,
            text=f"⏱  TRIAL {self.trial_count} IN PROGRESS",
            font=("Arial", 9, "bold"),
            fg=COLORS['text_dim'],
            bg=COLORS['bg']
        )
        self._title.pack(anchor='w')

        # Progress bar canvas
        bar_width = POPUP_WIDTH - 28
        self._canvas = tk.Canvas(
            inner,
            width=bar_width,
            height=28,
            bg=COLORS['bg'],
            highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        self._canvas.pack(pady=(6, 4))

        # Bar background
        self._canvas.create_rectangle(
            0, 0, bar_width, 28,
            fill=COLORS['bar_bg'],
            outline=COLORS['border']
        )

        # Bar fill (starts empty, grows right)
        self._bar = self._canvas.create_rectangle(
            0, 0, 0, 28,
            fill=COLORS['bar_green'],
            outline=''
        )

        # Time text centered on bar
        self._time_label = self._canvas.create_text(
            bar_width // 2, 14,
            text="00:00.0",
            font=("Arial", 11, "bold"),
            fill=COLORS['text_bright']
        )

        # Status text below bar
        self._status = tk.Label(
            inner,
            text="Cylinder lifted — place on target peg",
            font=("Arial", 9),
            fg=COLORS['text_dim'],
            bg=COLORS['bg']
        )
        self._status.pack(anchor='w')

        # Start the update loop
        self._update()

    def _update(self):
        """Refresh bar and time every UPDATE_MS milliseconds."""
        if not self._running or self._win is None:
            return

        elapsed  = time.time() - self._start_time
        bar_w    = POPUP_WIDTH - 28
        fill_w   = int(bar_w * min(1.0, elapsed / MAX_TIME_SEC))

        # Bar color changes with time — same thresholds as your main timer
        if elapsed < THRESHOLD_ORANGE:
            color = COLORS['bar_green']
        elif elapsed < THRESHOLD_RED:
            color = COLORS['bar_orange']
        else:
            color = COLORS['bar_red']

        self._canvas.coords(self._bar, 0, 0, fill_w, 28)
        self._canvas.itemconfig(self._bar, fill=color)
        self._canvas.itemconfig(self._time_label, text=self._fmt(elapsed))

        self._win.after(UPDATE_MS, self._update)

    def _complete(self):
        """Freeze the display and show placement confirmation."""
        if not self._running:
            return

        self._running = False
        duration      = time.time() - self._start_time
        self.placed_count += 1
        self.trial_times.append(duration)

        # Terminal log
        print(f"\n[Trial {self.trial_count}] Placed in {self._fmt(duration)}")
        print(f"[Score] {self.placed_count}/{self.trial_count} placements\n")

        if self._win is None:
            return

        # Update title and status
        self._title.config(text="✅  PLACED!", fg='#4a8a2a')
        self._status.config(
            text=f"Trial {self.trial_count}  —  {self._fmt(duration)}",
            fg=COLORS['text_bright']
        )

        # Freeze bar at current fill, switch to done color
        bar_w  = POPUP_WIDTH - 28
        fill_w = int(bar_w * min(1.0, duration / MAX_TIME_SEC))
        self._canvas.coords(self._bar, 0, 0, fill_w, 28)
        self._canvas.itemconfig(self._bar, fill=COLORS['bar_done'])
        self._canvas.itemconfig(
            self._time_label,
            text=self._fmt(duration),
            fill=COLORS['text_done']
        )

        # Auto-hide after POPUP_SHOW_MS
        self._win.after(POPUP_SHOW_MS, self._hide)

    def _hide(self):
        """Destroy the popup window."""
        if self._win is not None:
            try:
                self._win.destroy()
            except Exception:
                pass
            self._win = None

    @staticmethod
    def _fmt(seconds):
        """Format seconds as MM:SS.T  e.g. 75.3 → '01:15.3'"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        t = int((seconds % 1) * 10)
        return f"{m:02d}:{s:02d}.{t}"
    
    def on_arduino_event(self, event):
        """
        Callback passed to ArduinoReader.
        Called from Arduino background thread on every valid event.

        event.location_type : "CENTER" or "PEG"
        event.event         : "LIFTED" or "PLACED"
        event.pin_index     : decoded index (raw number - 48)
        """
        if event.location_type == "CENTER" and event.event == "LIFTED":
            print(f"[Arduino] Cylinder lifted from center {event.pin_index}")
            self.show_threadsafe()

        elif event.location_type == "PEG" and event.event == "PLACED":
            print(f"[Arduino] Cylinder placed on peg {event.pin_index}")
            self.complete_threadsafe()
