"""Background serial reader that turns Arduino peg-transfer events into
trial start/end callbacks for TaskTimerNode.

Firmware protocol (see arduino/automated_leds_new.ino):
    LIFTED          -> cylinder picked up off the center peg
    "Target hit"    -> cylinder placed on the outer/target peg
    DATA,...        -> full trial cycle done (placed AND carried back to center)
"""
import select
import sys
import threading
import time

import serial


class ArduinoTrigger:
    """Reads the dVRK peg board over serial and fires on_lifted/on_end.

    end_mode : "target_placed"    -> on_end fires on the 'Target hit' line
               "return_to_center" -> on_end fires on the 'DATA,' line (full cycle)
    """

    def __init__(self, port, baud, end_mode, on_lifted, on_end, logger=None,
                 on_raw_line=None, on_poll=None):
        """
        on_raw_line : optional, called with every raw serial line (including
            ones handled internally below) — lets a second subsystem (e.g.
            force/orientation capture) observe the full Arduino protocol
            without opening a second connection to the same port.
        on_poll : optional, called once per loop iteration (~every 10ms or
            whenever serial/stdin is checked) — for subsystems that need to
            fire time-delayed work (e.g. "check orientation 0.5s after
            Target hit") independent of new serial data arriving.
        """
        self.port = port
        self.baud = baud
        self.end_mode = end_mode
        self.on_lifted = on_lifted
        self.on_end = on_end
        self.on_raw_line = on_raw_line
        self.on_poll = on_poll
        self._log = logger.info if logger else print
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        try:
            arduino = serial.Serial(self.port, self.baud, timeout=0.1)
        except serial.SerialException as e:
            self._log(f'[ARDUINO] Could not open {self.port}: {e}')
            return

        time.sleep(2)  # let the board reset after opening the port
        self._log(f"[ARDUINO] Connected on {self.port}. "
                   f"Type 's' + Enter in this terminal to start a trial block "
                   f"('q' to abort).")

        try:
            while not self._stop_event.is_set():
                if self.on_poll:
                    self.on_poll()

                # Forward terminal keystrokes (s/q) to the board
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if ready:
                    cmd = sys.stdin.readline().strip()
                    if cmd:
                        arduino.write(cmd.encode('utf-8'))

                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8', errors='replace').strip()
                    if not line:
                        continue

                    if self.on_raw_line:
                        self.on_raw_line(line)

                    if line == 'LIFTED':
                        self.on_lifted()
                    elif 'Target hit' in line:
                        if self.end_mode == 'target_placed':
                            self.on_end()
                    elif line.startswith('DATA,'):
                        if self.end_mode == 'return_to_center':
                            self.on_end()
                    else:
                        self._log(f'[ARDUINO] {line}')
                else:
                    time.sleep(0.01)
        finally:
            arduino.close()
