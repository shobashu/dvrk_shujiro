"""
Arduino Reader
--------------
Reads serial data from the Arduino and saves it to a CSV file.
Can be run standalone (original behaviour) OR imported by main.py.

When imported, use the ArduinoReader class:
    from .arduino.read_arduino import ArduinoReader
    reader = ArduinoReader(callback=my_function)
    reader.start()

When run standalone (original behaviour):
    python3 read_arduino.py

Message format received from Arduino:
    CENTER 51,LIFTED,7138,0
    PEG 52,PLACED,76191,0

CSV format saved (one extra column added by Python):
    CENTER 51,LIFTED,7138,0,1773436001.566
    └─location─┘ └─event─┘ └─ms─┘ └manual┘ └─timestamp─┘
"""

import serial
import csv
import threading
from datetime import datetime, timedelta

# ── Settings (imported from config when used inside the package) ───────────────
try:
    from ..config import ARDUINO_PORT as PORT, ARDUINO_BAUD as BAUD, ARDUINO_CSV as FILENAME
except ImportError:
    # Fallback when running standalone
    PORT     = '/dev/ttyACM1'   # Change to your Arduino port
    BAUD     = 9600
    FILENAME = "arduino_data.csv"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ArduinoEvent — one parsed line from the Arduino
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ArduinoEvent:
    """
    Represents one event received from the Arduino.

    Attributes:
        location_type  "CENTER" or "PEG"
        pin_raw        raw number from Arduino message (e.g. 51)
        pin_index      decoded index (pin_raw - 48), e.g. 51 → 3
        event          "LIFTED" or "PLACED"
        arduino_ms     milliseconds since Arduino powered on
        manual         0 = normal event, 1 = manually triggered
        timestamp      Unix wall-clock time (float)
    """

    def __init__(self, location_type, pin_raw, event, arduino_ms, manual, timestamp):
        self.location_type = location_type
        self.pin_raw       = int(pin_raw)
        self.pin_index     = int(pin_raw) - 48   # Arduino adds ASCII offset of 48
        self.event         = event
        self.arduino_ms    = int(arduino_ms)
        self.manual        = int(manual)
        self.timestamp     = float(timestamp)

    def __repr__(self):
        return (f"ArduinoEvent({self.location_type} {self.pin_index}"
                f" — {self.event} at {self.arduino_ms}ms)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ArduinoReader — class wrapper around the original read_from_arduino logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ArduinoReader:
    """
    Wraps the original Arduino reading logic in a class so it can be
    started by main.py alongside the ROS node and GUI.

    Usage inside main.py:
        reader = ArduinoReader(callback=on_arduino_event)
        reader.start()

    The callback receives one ArduinoEvent every time a valid line arrives.
    """

    def __init__(self, callback=None, port=PORT, baud=BAUD, filename=FILENAME):
        self.port     = port
        self.baud     = baud
        self.filename = filename
        self.callback = callback        # called with ArduinoEvent on each event

        # Serial connection (same as original)
        self.ser = None

        # Time sync variables (same logic as original read_from_arduino)
        self.initialized        = False
        self.startTime          = datetime.now()
        self.initialMilliSeconds = 0

        self._thread = None

    def start(self):
        """Open serial connection and start background reading thread."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self._thread = threading.Thread(
                target=self._read_from_arduino,
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
        """Send a manual command to Arduino (triggers manualIntervention flag)."""
        if self.ser and self.ser.is_open:
            self.ser.write((cmd + '\n').encode('utf-8'))

    # ── Internal — original read_from_arduino logic, now inside the class ─────

    def _read_from_arduino(self):
        """
        Background thread — reads lines from Arduino, saves to CSV.
        This is the same logic as the original read_from_arduino() function,
        with one addition: calls self.callback(event) on each valid line.
        """
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            print(f"--- Logging started. Saving to {self.filename} ---")

            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        line = line.split(',')

                        # ── Time sync (original logic) ────────────────────────
                        if not self.initialized:
                            self.startTime           = datetime.now()
                            currTime                 = self.startTime
                            self.initialized         = True
                            self.initialMilliSeconds = int(line[2])
                        else:
                            currTime = self.startTime + timedelta(
                                milliseconds=(int(line[2]) - self.initialMilliSeconds)
                            )

                        line.append(currTime.timestamp())

                        # Print to terminal (original behaviour)
                        print(f"\n[Arduino]: {line}")

                        # Save to CSV (original behaviour)
                        writer.writerow(line)
                        f.flush()   # write to disk immediately

                        # ── New: parse and fire callback ──────────────────────
                        if self.callback:
                            event = self._parse(line)
                            if event:
                                self.callback(event)

    def _parse(self, line):
        """
        Parse a split CSV line into an ArduinoEvent.
        Returns None if the line is not a recognised event.

        line is already split, e.g.:
            ['CENTER 51', 'LIFTED', '7138', '0', 1773436001.566]
        """
        try:
            location_parts = line[0].strip().split(' ')
            if len(location_parts) != 2:
                return None

            location_type = location_parts[0]   # "CENTER" or "PEG"
            pin_raw       = location_parts[1]    # "51", "52", etc.
            event         = line[1].strip()      # "LIFTED" or "PLACED"
            arduino_ms    = line[2].strip()
            manual        = line[3].strip()
            timestamp     = line[4]

            if location_type not in ("CENTER", "PEG"):
                return None
            if event not in ("LIFTED", "PLACED"):
                return None

            return ArduinoEvent(location_type, pin_raw, event,
                                arduino_ms, manual, timestamp)
        except Exception as e:
            print(f"[Arduino] Parse error: {e}")
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone mode — original behaviour when run directly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    # This runs the original standalone behaviour:
    # reads Arduino and lets you type commands, exactly as before.
    reader = ArduinoReader()
    reader.start()

    print("--- Type your command and press Enter to send to Arduino ---")
    try:
        while True:
            cmd = input("> ")
            reader.send_command(cmd)
    except KeyboardInterrupt:
        print("\nClosing connection...")
        reader.stop()
