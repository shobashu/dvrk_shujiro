"""
Arduino Reader
--------------
Reads serial data from the Arduino and saves trial data to a CSV file.
Can be run standalone OR imported by main.py.

When imported, use the ArduinoReader class:
    from .arduino.read_arduino import ArduinoReader
    reader = ArduinoReader(callback=my_function)
    reader.start()

When run standalone:
    python3 read_arduino.py

Messages received from Arduino:
    --- System Ready ---        → plain text, printed to terminal, ignored
    SYNC,7138                   → time sync pulse sent once at experiment start
    LIFTED                      → cylinder lifted in real time (triggers popup)
    DATA,1,5,White,7138,9200,15400  → full trial result sent after placement
    2 second pause...           → plain text, printed to terminal, ignored

CSV format saved (one row per trial):
    Trial, Target_Peg, Target_Color, Unix_Cue_Time, Unix_Lift_Time, Unix_Place_Time
"""

import serial
import csv
import threading
import time

# ── Settings (imported from config when used inside the package) ───────────────
try:
    from ..config import ARDUINO_PORT as PORT, ARDUINO_BAUD as BAUD, ARDUINO_CSV as FILENAME
except ImportError:
    # Fallback when running standalone
    PORT     = '/dev/ttyACM0'
    BAUD     = 9600
    FILENAME = "experiment_data.csv"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ArduinoEvent — one parsed event from the Arduino
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ArduinoEvent:
    """
    Represents one event received from the Arduino.

    Two types of events:

    1. LIFTED — real time signal when cylinder is picked up
        event_type  = "LIFTED"
        trial       = None
        target_peg  = None
        color       = None
        unix_cue    = None
        unix_lift   = None
        unix_place  = None

    2. DATA — full trial result sent after placement
        event_type  = "DATA"
        trial       = trial number (int)
        target_peg  = target peg index (int, 1-8)
        color       = "Blue" or "White"
        unix_cue    = Unix time of cue (float)
        unix_lift   = Unix time of lift (float)
        unix_place  = Unix time of placement (float)
    """

    def __init__(self, event_type, trial=None, target_peg=None, color=None,
                 unix_cue=None, unix_lift=None, unix_place=None):
        self.event_type = event_type    # "LIFTED" or "DATA"
        self.trial      = trial
        self.target_peg = target_peg
        self.color      = color
        self.unix_cue   = unix_cue
        self.unix_lift  = unix_lift
        self.unix_place = unix_place

    def __repr__(self):
        if self.event_type == "LIFTED":
            return "ArduinoEvent(LIFTED)"
        return (f"ArduinoEvent(DATA trial={self.trial} peg={self.target_peg}"
                f" color={self.color})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ArduinoReader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ArduinoReader:
    """
    Reads Arduino serial data in a background thread.
    Parses each line and calls callback(event) for LIFTED and DATA events.

    Usage:
        reader = ArduinoReader(callback=my_function)
        reader.start()
        reader.send_command('s')   # start experiment
    """

    def __init__(self, callback=None, port=PORT, baud=BAUD, filename=FILENAME):
        self.port     = port
        self.baud     = baud
        self.filename = filename
        self.callback = callback

        self.ser      = None
        self._thread  = None

        # Time sync — set when SYNC message arrives from Arduino
        self._python_sync_time    = 0
        self._arduino_sync_millis = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Open serial connection and start background reading thread."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2)   # give Arduino time to reset after connection
            self._thread = threading.Thread(
                target=self._read_loop,
                daemon=True,
                name="ArduinoReader"
            )
            self._thread.start()
            print(f"[Arduino] Connected on {self.port} — logging to {self.filename}")
        except serial.SerialException as e:
            print(f"[Arduino] Could not connect on {self.port}: {e}")

    def stop(self):
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[Arduino] Connection closed.")

    def send_command(self, cmd):
        """
        Send a command to Arduino.
        's' = start experiment
        'q' = abort experiment
        """
        if self.ser and self.ser.is_open:
            self.ser.write(cmd.encode('utf-8'))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_loop(self):
        """Background thread — reads lines from Arduino and processes them."""
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)

            # Write CSV header if file is new/empty
            f.seek(0, 2)    # seek to end
            if f.tell() == 0:
                writer.writerow([
                    'Trial', 'Target_Peg', 'Target_Color',
                    'Unix_Cue_Time', 'Unix_Lift_Time', 'Unix_Place_Time'
                ])

            print(f"--- Logging started. Saving to {self.filename} ---")

            while True:
                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8').strip()
                    except Exception as e:
                        print(f"[Arduino] Read error: {e}")
                        continue

                    if not line:
                        continue

                    # ── SYNC — record time reference ──────────────────────────
                    if line.startswith("SYNC,"):
                        parts = line.split(",")
                        self._arduino_sync_millis = int(parts[1])
                        self._python_sync_time    = time.time()
                        print(f"[Arduino] Time sync received at {self._python_sync_time:.3f}")

                    # ── LIFTED — real time lift signal ────────────────────────
                    elif line == "LIFTED":
                        print("[Arduino] Cylinder LIFTED")
                        if self.callback:
                            self.callback(ArduinoEvent(event_type="LIFTED"))

                    # ── DATA — full trial result ──────────────────────────────
                    elif line.startswith("DATA,"):
                        event = self._parse_data(line)
                        if event:
                            # Save to CSV
                            writer.writerow([
                                event.trial,
                                event.target_peg,
                                event.color,
                                f"{event.unix_cue:.3f}",
                                f"{event.unix_lift:.3f}",
                                f"{event.unix_place:.3f}"
                            ])
                            f.flush()
                            print(f"[CSV] Trial {event.trial} saved — "
                                  f"peg {event.target_peg} {event.color}")
                            if self.callback:
                                self.callback(event)

                    # ── Everything else — print to terminal ───────────────────
                    else:
                        print(f"[Arduino] {line}")

    def _parse_data(self, line):
        """
        Parse a DATA line into an ArduinoEvent.
        Format: DATA,trial,peg,color,cue_ms,lift_ms,place_ms

        Returns None if parsing fails.
        """
        try:
            parts = line.split(",")
            if len(parts) != 7:
                return None

            trial      = int(parts[1])
            target_peg = int(parts[2])
            color      = parts[3]
            cue_ms     = int(parts[4])
            lift_ms    = int(parts[5])
            place_ms   = int(parts[6])

            # Convert Arduino milliseconds to Unix time using SYNC reference
            unix_cue   = self._python_sync_time + ((cue_ms   - self._arduino_sync_millis) / 1000.0)
            unix_lift  = self._python_sync_time + ((lift_ms  - self._arduino_sync_millis) / 1000.0)
            unix_place = self._python_sync_time + ((place_ms - self._arduino_sync_millis) / 1000.0)

            return ArduinoEvent(
                event_type = "DATA",
                trial      = trial,
                target_peg = target_peg,
                color      = color,
                unix_cue   = unix_cue,
                unix_lift  = unix_lift,
                unix_place = unix_place
            )
        except Exception as e:
            print(f"[Arduino] Parse error on '{line}': {e}")
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    reader = ArduinoReader()
    reader.start()

    print("\n--- Python Monitor Ready ---")
    print(f"Data will be saved to: {FILENAME}")
    print("Type 's' and press Enter to START the experiment.")
    print("Type 'q' and press Enter to ABORT.")
    print("----------------------------\n")

    try:
        while True:
            cmd = input("> ")
            if cmd.strip():
                reader.send_command(cmd.strip())
    except KeyboardInterrupt:
        print("\nExiting...")
        reader.stop()