"""
read_arduino.py  —  Adaptive dVRK Arduino interface (full version)
==================================================================
Bidirectional serial bridge between Python and the Arduino.

Session structure
-----------------
  Block 1 : 20 trials  (type 's' to start)
  Break   : type 'c' to continue into block 2
  Block 2 : 20 trials  (adaptation continues from block 1 level)

Keyboard commands (type + Enter)
---------------------------------
  s  → start / board check
  c  → continue after break (block 2)
  q  → abort
  d  → demo mode  (DemoController, cycles levels 1–8)
  t  → test mode  (same as demo but prints extra debug)

Protocol  Python → Arduino
--------------------------
  "TRIAL:<center>,<target>,<color>\n"
  "s\n" / "q\n"

Protocol  Arduino → Python
--------------------------
  "READY"                              → send first TRIAL command
  "TRIAL_DONE"                         → score last trial, send next TRIAL command
  "Object lifted. Move to Outer Peg N" → push to lifted_queue
  "Target hit"                         → push to lifted_queue
  "DATA,..."                           → parse, log CSV, notify controller
  "SYNC,<ms>"                          → time sync
  "FILENAME,<name>"                    → switch CSV filename

Interfaces with adaptive_controller.AdaptiveController.
Falls back to _DemoController when no controller is injected.
"""

import csv
import os
import queue as _queue
import select
import serial
import sys
import time
import threading

# --- CONFIGURATION ---
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE    = 9600
DEFAULT_CSV  = 'experiment_data.csv'
BLOCK_SIZE   = 20    # trials per block


# =============================================================================
# Fallback demo controller (no adaptation)
# =============================================================================
class _DemoController:
    """Cycles through all 8 levels in order — no scoring, no adaptation."""
    _SEQUENCE = [
        (0,0,0),(0,1,0),(0,2,0),(0,3,0),
        (0,4,1),(0,5,1),(0,6,1),(0,7,1),
    ]
    def __init__(self, verbose=False):
        self._idx     = 0
        self._verbose = verbose

    def get_next_trial(self):
        cfg = self._SEQUENCE[self._idx % len(self._SEQUENCE)]
        self._idx += 1
        return cfg

    def record_result(self, d):
        if self._verbose:
            print(f"[DEMO] {d}")

    # stub notifiers so duck-typing works
    def notify_drop(self):        pass
    def notify_force(self, f):    pass
    def notify_orientation(self, b): pass
    def start_break(self):        pass
    def end_break(self):          pass

    @property
    def is_block_complete(self):  return False
    @property
    def is_session_complete(self): return False
    @property
    def current_level(self):      return 0
    @property
    def trial_number(self):       return self._idx


# =============================================================================
# Helper: write CSV header if file is new
# =============================================================================
def _ensure_header(filename):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            csv.writer(f).writerow([
                'Trial','Target_Peg','Target_Color',
                'Unix_Cue_Time','Unix_Lift_Time','Unix_Place_Time',
                'Trial_Time_s'
            ])


# =============================================================================
# Core loop — called as daemon thread from 5_realtime_infer.py
# =============================================================================
def arduino_loop(lifted_queue: _queue.Queue,
                 stop_event:   threading.Event,
                 controller=None):
    """
    Parameters
    ----------
    lifted_queue : queue.Queue
        Consumed by batch_infer_loop in 5_realtime_infer.py.
        Pushes: "Object lifted. Move to Outer Peg <N>", "Target hit"
    stop_event : threading.Event
    controller : AdaptiveController | None
        If None → _DemoController is used.
    """
    if controller is None:
        controller = _DemoController()
        print("[ARDUINO] No controller injected — using DemoController.")

    # Time-sync state
    python_sync_time    = 0.0
    arduino_sync_millis = 0

    # Session state
    current_csv   = DEFAULT_CSV
    in_break      = False      # True while waiting for 'c' between blocks
    win_buf       = ""         # Windows keyboard buffer

    _ensure_header(current_csv)

    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print("[ARDUINO] Connected.")
        _print_menu()

        while not stop_event.is_set():

            # ---- 1. Forward keyboard input to Arduino ----
            cmd = _read_keyboard(win_buf)
            if cmd:
                if sys.platform == "win32" and len(cmd) == 1:
                    win_buf = ""   # reset after sending
                _dispatch_keyboard(cmd, arduino, controller,
                                   in_break, lifted_queue)
                if cmd.strip().lower() == 'c' and in_break:
                    in_break = False
                    controller.end_break()

            # ---- 2. Read from Arduino ----
            if arduino.in_waiting == 0:
                continue

            raw = arduino.readline().decode('utf-8', errors='replace').strip()
            if not raw:
                continue

            # ---- 3. Dispatch by message type ----

            if raw.startswith("FILENAME,"):
                parts = raw.split(",", 1)
                if len(parts) > 1:
                    current_csv = parts[1].strip()
                    _ensure_header(current_csv)
                    print(f"[ARDUINO] Log file → {current_csv}")

            elif raw.startswith("SYNC,"):
                parts = raw.split(",")
                arduino_sync_millis = int(parts[1])
                python_sync_time    = time.time()
                print(f"[ARDUINO] SYNC at {python_sync_time:.3f}")

            elif raw in ("READY", "TRIAL_DONE"):
                # Check for mid-session break
                if raw == "TRIAL_DONE":
                    if controller.is_session_complete:
                        print("[ARDUINO] Session complete — all 40 trials done!")
                        continue
                    if controller.is_block_complete and not in_break:
                        in_break = True
                        controller.start_break()
                        print("\n" + "="*50)
                        print("  BLOCK 1 COMPLETE — take a break!")
                        print("  Type 'c' + Enter to start Block 2.")
                        print("="*50 + "\n")
                        continue   # do NOT send next trial — wait for 'c'

                if not in_break:
                    _send_next_trial(arduino, controller)

            elif "Object lifted. Move to Outer Peg" in raw:
                lifted_queue.put(raw)
                print(f"[ARDUINO] {raw}")

            elif "Target hit" in raw:
                lifted_queue.put(raw)
                print(f"[ARDUINO] {raw}")

            elif raw.startswith("DATA,"):
                _handle_data(raw, python_sync_time, arduino_sync_millis,
                             current_csv, controller, lifted_queue)

            else:
                print(f"[ARDUINO] {raw}")

    except serial.SerialException as e:
        print(f"[ARDUINO] Serial error: {e}")
    finally:
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()
        if hasattr(controller, 'close'):
            controller.close()


# =============================================================================
# Internal helpers
# =============================================================================

def _send_next_trial(arduino, controller):
    center, target, color = controller.get_next_trial()
    cmd = f"TRIAL:{center},{target},{color}\n"
    arduino.write(cmd.encode('utf-8'))
    color_name = "Blue" if color == 0 else "White"
    print(f"[ARDUINO] → {cmd.strip()}  "
          f"(center={center+1} target={target+1} {color_name}  "
          f"level={controller.current_level}  "
          f"trial={controller.trial_number+1}/40)")


def _handle_data(raw, python_sync_time, arduino_sync_millis,
                 csv_path, controller, lifted_queue):
    """Parse DATA line, log to CSV, notify controller."""
    parts = raw.split(",")
    if len(parts) < 7:
        print(f"[ARDUINO] Malformed DATA: {raw}")
        return

    try:
        trial_num  = int(parts[1])
        target_peg = int(parts[2])
        color_str  = parts[3]
        cue_ms     = int(parts[4])
        lift_ms    = int(parts[5])
        place_ms   = int(parts[6])
    except ValueError:
        print(f"[ARDUINO] Could not parse DATA: {raw}")
        return

    def ms_to_unix(ms):
        return python_sync_time + ((ms - arduino_sync_millis) / 1000.0)

    cue_unix   = ms_to_unix(cue_ms)
    lift_unix  = ms_to_unix(lift_ms)
    place_unix = ms_to_unix(place_ms)
    trial_time = place_unix - lift_unix

    # Append to CSV
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            trial_num, target_peg, color_str,
            f"{cue_unix:.3f}", f"{lift_unix:.3f}", f"{place_unix:.3f}",
            f"{trial_time:.3f}",
        ])
    print(f"[CSV] Trial {trial_num} → {csv_path}  (t={trial_time:.2f}s)")

    # Notify adaptive controller
    controller.record_result({
        "trial":        trial_num,
        "target_peg":   target_peg,
        "color":        color_str,
        "cue_unix":     cue_unix,
        "lift_unix":    lift_unix,
        "place_unix":   place_unix,
        "trial_time_s": trial_time,
    })

    # Push summary to YOLO inference thread
    lifted_queue.put(
        f"DATA_SUMMARY:{trial_num},{target_peg},{color_str},{trial_time:.3f}"
    )


def _read_keyboard(win_buf):
    """Non-blocking keyboard read. Returns stripped command string or ''."""
    if sys.platform != "win32":
        r, _, _ = select.select([sys.stdin], [], [], 0.0001)
        if r:
            return sys.stdin.readline().strip()
    else:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getche()
            if ch in (b'\r', b'\n'):
                print()
                return win_buf
            try:
                return ch.decode('utf-8')
            except UnicodeDecodeError:
                pass
    return ""


def _dispatch_keyboard(cmd, arduino, controller, in_break, lifted_queue):
    """Handle local keyboard commands and forward relevant ones to Arduino."""
    cmd_lower = cmd.lower()
    if cmd_lower in ('s', 'q', 'c', 'd', 't'):
        arduino.write((cmd + '\n').encode('utf-8'))
        if cmd_lower == 'q':
            print("[ARDUINO] Abort sent.")
        elif cmd_lower == 'd':
            print("[ARDUINO] Demo mode activated.")
        elif cmd_lower == 't':
            print("[ARDUINO] Test mode activated.")


def _print_menu():
    print("\n--- Adaptive dVRK Monitor ---")
    print("  s + Enter  →  start (board check)")
    print("  c + Enter  →  continue after break")
    print("  q + Enter  →  abort")
    print("  d + Enter  →  demo mode")
    print("  t + Enter  →  test mode")
    print("-----------------------------\n")


# =============================================================================
# Standalone entry point (no YOLO)
# =============================================================================
def main():
    # Try to import real adaptive controller; fall back to demo
    try:
        from adaptive_controller import AdaptiveController
        ctrl = AdaptiveController(log_csv="adaptive_log.csv", start_level=5)
        print("[MAIN] AdaptiveController loaded.")
    except ImportError:
        ctrl = _DemoController(verbose=True)
        print("[MAIN] adaptive_controller.py not found — using DemoController.")

    stop = threading.Event()
    q    = _queue.Queue()

    print("Standalone mode (no YOLO). Ctrl+C to stop.\n")
    try:
        arduino_loop(q, stop, controller=ctrl)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")


if __name__ == '__main__':
    main()
