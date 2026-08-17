"""Optional force-sensor and/or YOLO cylinder-orientation capture.

Ported from the standalone read_arduino_with_force.py: same Arduino protocol
lines (see automated_leds_new.ino), same orientation-check logic
(scripts/6_cylinder_orientation.py), same CSV schema — but as a passive
observer layered on top of ArduinoTrigger's serial stream (arduino_bridge.py)
instead of owning its own serial connection, so it can run alongside the
normal trial timer rather than as a separate process.

enable_force and enable_orientation are independent — enable either alone or
both. Only enable_orientation needs cv2/ultralytics/the YOLO venv; those are
only imported here when it's True.

Feed it:
    on_wrench(fx, fy, fz)           <- a /ati_sensor/wrench subscription
                                        (only if enable_force)
    on_compressed_image(jpeg_bytes) <- a /camera_left/compressed subscription
                                        (only if enable_orientation)
    on_raw_line(line)               <- ArduinoTrigger(on_raw_line=...)
    poll()                          <- ArduinoTrigger(on_poll=...)

on_force_result(...) fires once per trial, at whichever point end_mode
marks the trial's end ("Target hit" for target_placed, "Object returned
successfully." for return_to_center) — but if an orientation check is still
pending at that point, it's held until poll() resolves it, so the force
graph and orientation result reach the popup together instead of the
orientation field showing up blank.
"""
import csv
import importlib.util as _ilu
import json
import os
import threading
from collections import deque
import time


class ForceOrientationTracker:

    def __init__(self, enable_force, enable_orientation, end_mode,
                 yolo_weights, yolo_conf, yolo_imgsz, scripts_dir,
                 orientation_tolerance_deg, orientation_delay_sec,
                 orientation_max_attempts, orientation_retry_delay_sec,
                 peg_classifications, csv_path, trial_signal_path,
                 on_force_result, logger=None, smoothing_window=1,
                 force_threshold=5.0):
        """
        end_mode : same value as ArduinoTrigger's end_mode — "target_placed"
            stops force capture and shows the result popup right at
            "Target hit" (so the graph only covers lift->place, matching
            what the trial timer itself reports); "return_to_center" stops
            it at "Object returned successfully." (the full cycle), as
            before. The depth-camera trial_end signal always fires at the
            physical "Object returned successfully." regardless of this,
            since that downstream consumer tracks physical completion.
        """
        self.enable_force = enable_force
        self.enable_orientation = enable_orientation
        self.end_mode = end_mode
        self.smoothing_window = smoothing_window
        self.force_threshold = force_threshold
        self._log = logger.info if logger else print
        self.on_force_result = on_force_result

        self.yolo_conf = yolo_conf
        self.yolo_imgsz = yolo_imgsz
        self.orientation_tolerance_deg = orientation_tolerance_deg
        self.orientation_delay_sec = orientation_delay_sec
        self.orientation_max_attempts = orientation_max_attempts
        self.orientation_retry_delay_sec = orientation_retry_delay_sec
        self.peg_classifications = peg_classifications

        self.model = None
        self._compute_orientation = None
        self._cv2 = None
        self._np = None
        if self.enable_orientation:
            # Deferred imports: cv2/numpy/ultralytics are only needed for the
            # orientation check, not for force capture or the base timer app,
            # and normally require a dedicated venv (see config.py's comment
            # on ENABLE_ORIENTATION_CHECK).
            import cv2
            import numpy as np
            from ultralytics import YOLO

            self._cv2 = cv2
            self._np = np
            self._log(f'[ORIENTATION] Loading YOLO weights: {yolo_weights}')
            self.model = YOLO(yolo_weights)

            # Load compute_orientation() from scripts/6_cylinder_orientation.py
            # without turning scripts/ into an importable package.
            spec = _ilu.spec_from_file_location(
                'cylinder_orientation', os.path.join(scripts_dir, '6_cylinder_orientation.py'))
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._compute_orientation = mod.compute_orientation

        # ── Wrench buffer (only used if enable_force) ──────────────────────
        self._force_lock = threading.Lock()
        self._force_active = False
        self._force_t0 = 0.0
        self._force_buffer = []

        # ── Latest camera frame (only used if enable_orientation) ──────────
        self._frame_lock = threading.Lock()
        self._frame = None

        # ── CSV + depth-camera signal file ─────────────────────────────────
        self._csv_file = open(csv_path, mode='a', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['Trial', 'Target_Peg', 'Target_Color',
                                    'Unix_Cue_Time', 'Unix_Lift_Time', 'Unix_Place_Time',
                                    'Peak_Force_N', 'Force_Crossings_Above_Threshold',
                                    'Orientation_Deg', 'Orientation_OK', 'Down_Color',
                                    'Placement_Success'])
        self._signal_file = open(trial_signal_path, 'w')  # fresh each run

        # ── Trial-parsing state (mirrors read_arduino_with_force.py) ───────
        self._python_sync_time = 0
        self._arduino_sync_millis = 0
        self.current_trial = None
        self.last_force_result = ([], [], 0.0, 0)
        self.last_orientation = (None, None, None, None)
        self.last_lifted_peg = None
        self.pending_orientation = None
        self.current_target_color = None
        self._report_pending = False  # end_mode's trigger fired, waiting on poll() to resolve orientation

        self._log(f'[FORCE] Ready (force={enable_force}, orientation={enable_orientation}). '
                  f'CSV -> {csv_path}')

    # ── Sensor feed (call from ROS subscription callbacks) ─────────────────

    def on_wrench(self, fx, fy, fz):
        if not self.enable_force or not self._force_active:
            return
        mag = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        t_rel = time.time() - self._force_t0
        with self._force_lock:
            self._force_buffer.append((t_rel, mag))

    def on_compressed_image(self, jpeg_bytes):
        if not self.enable_orientation:
            return
        buf = self._np.frombuffer(jpeg_bytes, dtype=self._np.uint8)
        frame = self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._frame_lock:
            self._frame = frame

    # ── Internal helpers ─────────────────────────────────────────────────

    def _start_force_capture(self):
        with self._force_lock:
            self._force_buffer = []
            self._force_t0 = time.time()
            self._force_active = True

    def _end_force_capture(self):
        with self._force_lock:
            self._force_active = False
            samples = list(self._force_buffer)
        if not samples:
            return [], [], 0.0, 0
        times = [s[0] for s in samples]
        forces = self._smooth([s[1] for s in samples])
        crossings = self._count_crossings(forces, self.force_threshold)
        return times, forces, max(forces), crossings

    def _smooth(self, forces):
        """Trailing moving average, window = self.smoothing_window samples.
        Applied once here, so the returned peak/crossings and the plotted
        line (and the CSV row, which reads off this same result) all agree —
        smooths out sensor/electrical noise, at the cost of flattening any
        genuine force spike briefer than the window."""
        w = self.smoothing_window
        if w <= 1 or not forces:
            return list(forces)
        window = deque(maxlen=w)
        smoothed = []
        for f in forces:
            window.append(f)
            smoothed.append(sum(window) / len(window))
        return smoothed

    @staticmethod
    def _count_crossings(forces, threshold):
        """Count rising-edge crossings of `threshold` — i.e. distinct contact
        events (force rises from at-or-below to above threshold), not a raw
        sample count, so it doesn't just track how long contact was held."""
        count = 0
        below = True
        for f in forces:
            if below and f > threshold:
                count += 1
                below = False
            elif f <= threshold:
                below = True
        return count

    def _stop_force_if_needed(self):
        """Stop force capture (if enabled). Called once, at whichever serial
        line matches self.end_mode."""
        if self.enable_force:
            self.last_force_result = self._end_force_capture()
            peak, crossings = self.last_force_result[2], self.last_force_result[3]
            self._log(f'[FORCE] Trial {self.current_trial}: peak={peak:.2f}N, '
                      f'crossed {self.force_threshold:.1f}N {crossings}x')

    def _report(self):
        """Fire on_force_result with whatever force/orientation data is
        known right now — so the popup shows the force graph and the
        orientation result together."""
        times, forces, peak, crossings = self.last_force_result
        angle_deg, is_upright, down_color, success = self.last_orientation
        if self.on_force_result:
            self.on_force_result(self.current_trial or '?', times, forces, peak, crossings,
                                  angle_deg, is_upright, down_color, success)

    def _maybe_report_or_wait(self):
        """Called once self.end_mode's trigger event happens. Reports right
        away unless an orientation check is still pending — in that case
        poll() finishes the report once it resolves, so the force graph and
        orientation result show up together in the same popup instead of
        the orientation field showing up blank."""
        if self.enable_orientation and self.pending_orientation is not None:
            self._report_pending = True
            return
        self._report()

    def _emit_signal(self, event, trial, **extra):
        payload = {'event': event, 'trial': trial, 't': time.time()}
        payload.update(extra)
        self._signal_file.write(json.dumps(payload) + '\n')
        self._signal_file.flush()

    def _check_orientation(self, side):
        """Crop the latest left-camera frame to `side` ("L"/"R"), run YOLO + orientation.

        Returns (angle_deg, is_upright, down_color), or (None, None, None) if
        no frame/cylinder is available.
        """
        with self._frame_lock:
            frame = self._frame
        if frame is None:
            return None, None, None

        h, w = frame.shape[:2]
        crop = frame[:, :int(w * 0.4)] if side == 'L' else frame[:, int(w * 0.55):]

        results = self.model.predict(
            crop, conf=self.yolo_conf, imgsz=self.yolo_imgsz, classes=[0], verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return None, None, None

        ch, cw = crop.shape[:2]
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(cw, x2), min(ch, y2)
        sub = crop[y1:y2, x1:x2]
        if sub.size == 0:
            return None, None, None

        angle_deg, blue_c, white_c, _ = self._compute_orientation(sub)
        if white_c is None or blue_c is None:
            return None, None, None

        down_color = 'blue' if blue_c[1] > white_c[1] else 'white'
        return angle_deg, abs(angle_deg) <= self.orientation_tolerance_deg, down_color

    # ── Driven by ArduinoTrigger ────────────────────────────────────────────

    def poll(self):
        """Call this every ~10ms (via ArduinoTrigger(on_poll=...)) to fire the
        delayed orientation check independent of new serial data arriving."""
        if not self.enable_orientation:
            return
        if self.pending_orientation is None or time.time() < self.pending_orientation['next_check']:
            return

        side = self.pending_orientation['side']
        target_color = self.pending_orientation['target_color']
        attempt = self.pending_orientation['attempt']
        angle_deg, is_upright, down_color = self._check_orientation(side)

        if angle_deg is None and attempt < self.orientation_max_attempts:
            self._log(f'[ORIENTATION] no cylinder detected in crop '
                      f'(attempt {attempt}/{self.orientation_max_attempts}), retrying...')
            self.pending_orientation['attempt'] = attempt + 1
            self.pending_orientation['next_check'] = time.time() + self.orientation_retry_delay_sec
            return

        self.pending_orientation = None
        success = None
        if down_color is not None and target_color is not None:
            success = down_color.lower() == target_color.lower()
        self.last_orientation = (angle_deg, is_upright, down_color, success)

        if angle_deg is not None:
            if success is True:
                result_word = 'SUCCESS'
            elif success is False:
                result_word = 'FAIL'
            else:
                result_word = 'UNKNOWN — no target color captured for this trial'
            self._log(f'[ORIENTATION] {angle_deg:+.1f}°  ({"OK" if is_upright else "OFF"})  '
                      f'{down_color} facing down  —  target was {target_color}  '
                      f'({result_word})')
        else:
            self._log(f'[ORIENTATION] no cylinder detected in crop after {attempt} attempts')

        if self._report_pending:
            self._report_pending = False
            self._report()

    def on_raw_line(self, line):
        """Feed every raw Arduino serial line here (ArduinoTrigger(on_raw_line=...))."""
        if line.startswith('SYNC,'):
            parts = line.split(',')
            self._arduino_sync_millis = int(parts[1])
            self._python_sync_time = time.time()

        elif line.startswith('--- Trial'):
            digits = ''.join(ch for ch in line if ch.isdigit())
            self.current_trial = digits or self.current_trial
            if self.enable_force:
                self._start_force_capture()
            self.last_orientation = (None, None, None, None)
            self.last_lifted_peg = None
            self.pending_orientation = None
            self.current_target_color = None
            self._report_pending = False

        elif 'Waiting for lift from Center Peg' in line:
            try:
                center_peg = int(line.split('Waiting for lift from Center Peg')[-1].strip())
            except ValueError:
                center_peg = None
            self._emit_signal('trial_start', self.current_trial, center_peg=center_peg)

        elif line.startswith('CUE,'):
            self.current_target_color = line.split(',', 1)[-1].strip()

        elif 'Object lifted. Move to Outer Peg' in line:
            try:
                self.last_lifted_peg = int(
                    line.split('Object lifted. Move to Outer Peg')[-1].strip())
            except ValueError:
                self.last_lifted_peg = None

        elif 'Target hit' in line:
            if self.enable_orientation:
                side = self.peg_classifications.get(self.last_lifted_peg)
                if side is not None:
                    self.pending_orientation = {
                        'next_check': time.time() + self.orientation_delay_sec,
                        'side': side,
                        'target_color': self.current_target_color,
                        'attempt': 1,
                    }
            if self.end_mode == 'target_placed':
                # The graph/popup covers only lift->place, matching what the
                # trial timer itself reports in this mode. Force capture ends
                # right here; if orientation checking is also pending, the
                # popup waits for it (via poll()) so both show up together.
                self._stop_force_if_needed()
                self._maybe_report_or_wait()

        elif line == 'Object returned successfully.':
            # Always emitted regardless of end_mode — the depth-camera signal
            # tracks physical completion, not the software end_mode choice.
            self._emit_signal('trial_end', self.current_trial)
            if self.end_mode == 'return_to_center':
                self._stop_force_if_needed()
                self._maybe_report_or_wait()

        elif line.startswith('DATA,'):
            parts = line.split(',')
            trial, peg, color = parts[1], parts[2], parts[3]
            cue_unix = self._python_sync_time + ((int(parts[4]) - self._arduino_sync_millis) / 1000.0)
            lift_unix = self._python_sync_time + ((int(parts[5]) - self._arduino_sync_millis) / 1000.0)
            place_unix = self._python_sync_time + ((int(parts[6]) - self._arduino_sync_millis) / 1000.0)

            _, _, peak, crossings = self.last_force_result
            angle_deg, is_upright, down_color, success = self.last_orientation
            self._csv_writer.writerow([
                trial, peg, color, f'{cue_unix:.3f}', f'{lift_unix:.3f}', f'{place_unix:.3f}',
                f'{peak:.3f}' if self.enable_force else '',
                crossings if self.enable_force else '',
                f'{angle_deg:.1f}' if angle_deg is not None else '',
                is_upright if is_upright is not None else '',
                down_color if down_color is not None else '',
                success if success is not None else ''])
            self._csv_file.flush()
            summary = f'Trial {trial} CSV saved'
            if self.enable_force:
                summary += f' (peak={peak:.2f}N, crossed {self.force_threshold:.1f}N {crossings}x)'
            self._log(f'[FORCE] {summary}.')
