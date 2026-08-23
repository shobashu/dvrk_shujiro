"""Cylinder state (STATIONARY/HELD/DROPPED/LOST) detection via a trained
XGBoost classifier — ported from scripts/7_velocity_check.py's
CylinderVelocityNode, minus its OpenCV live-plot windows and CSV/video
export (that standalone script still exists separately if you want those).
Only the trained model is supported here — no rule-based threshold path.

LOST is still a deterministic rule ("not detected for lost_frames
consecutive frames"), never model-predicted, for the same reason
force_orientation_bridge.py's on_raw_line() overrides the orientation
model on "det is None" frames: there's no real detection for the
classifier to reason about, and (per the out-of-fold evaluation in
7_2_train_xgboost.py) the model is unreliable on exactly that transition.

Feed it:
    on_compressed_image(jpeg_bytes) <- a /camera_left/compressed subscription
    on_psm1_pose(msg)               <- a /PSM1/measured_cp subscription

Runs YOLO detection + XGBoost classification in its own background thread
(mirroring CylinderVelocityNode's _infer_loop) so it never blocks the ROS
callbacks that feed it frames/poses. on_state_change(old, new) fires on
every transition; a per-frame inference-latency callback lets the caller
log measured (not assumed) detection delay.
"""
import csv
import os
import threading
import time
from collections import deque


class CylinderStateTracker:

    STATE_STATIONARY = 'STATIONARY'
    STATE_HELD = 'HELD'
    STATE_DROPPED = 'DROPPED'
    STATE_LOST = 'LOST'

    def __init__(self, yolo_weights, yolo_conf, yolo_imgsz,
                 xgb_model_path, xgb_window, vel_window, max_jump, lost_frames,
                 on_state_change=None, logger=None,
                 timing_log_path=None, timing_config=''):
        # Deferred imports: only needed when this tracker is actually
        # constructed (ENABLE_STATE_DETECTION), and require the YOLO venv —
        # same reasoning as force_orientation_bridge.py's enable_orientation.
        import cv2
        import numpy as np
        from ultralytics import YOLO
        from xgboost import XGBClassifier
        import joblib

        self._cv2 = cv2
        self._np = np
        self._log = logger.info if logger else print
        self.on_state_change = on_state_change

        self.yolo_conf = yolo_conf
        self.yolo_imgsz = yolo_imgsz
        self.max_jump = max_jump
        self.lost_frames = lost_frames
        self.xgb_window = xgb_window

        self._log(f'[STATE] Loading YOLO weights: {yolo_weights}')
        self.model = YOLO(yolo_weights)

        self._log(f'[STATE] Loading XGBoost model: {xgb_model_path}')
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(xgb_model_path)
        le_path = xgb_model_path.replace('.json', '.labels.pkl')
        self.xgb_le = joblib.load(le_path)

        # ── Measured-latency log (shares the schema/file used by
        #    force_orientation_bridge.py's ForceOrientationTracker, tagged
        #    with the same combined "config" label so runs are comparable
        #    via scripts/9_analyze_timing.py) ────────────────────────────
        self._timing_lock = threading.Lock()
        self._timing_log_writer = None
        self._timing_config = timing_config
        if timing_log_path:
            is_new = not os.path.exists(timing_log_path) or os.path.getsize(timing_log_path) == 0
            self._timing_log_file = open(timing_log_path, mode='a', newline='')
            self._timing_log_writer = csv.writer(self._timing_log_file)
            if is_new:
                self._timing_log_writer.writerow([
                    'trial', 'event', 'config', 'total_latency_s',
                    'settle_delay_s', 'inference_time_s', 'retries', 'unix_time'])

        self._raw_lock = threading.Lock()
        self._raw_frame = None
        self._raw_frame_id = 0
        self._last_infer_id = -1

        self._psm1_lock = threading.Lock()
        self._psm1_pose = None

        self._state = self.STATE_STATIONARY
        self._prev_center = None
        self._prev_time = None
        self._lost_count = 0
        self._vel_hist = deque(maxlen=vel_window)
        self._feat_buf = deque(maxlen=xgb_window)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()

        self._log('[STATE] Ready. Tracking cylinder state in the background.')

    @property
    def state(self):
        return self._state

    # ── Sensor feed (call from ROS subscription callbacks) ─────────────────

    def on_compressed_image(self, jpeg_bytes):
        buf = self._np.frombuffer(jpeg_bytes, dtype=self._np.uint8)
        frame = self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._raw_lock:
            self._raw_frame = frame
            self._raw_frame_id += 1

    def on_psm1_pose(self, msg):
        with self._psm1_lock:
            self._psm1_pose = msg

    def stop(self):
        self._stop_event.set()

    # ── Background inference thread ─────────────────────────────────────

    def _infer_loop(self):
        while not self._stop_event.is_set():
            with self._raw_lock:
                frame_id = self._raw_frame_id
                frame = self._raw_frame
            if frame is None or frame_id == self._last_infer_id:
                self._stop_event.wait(0.005)
                continue
            self._last_infer_id = frame_id

            t0 = time.time()
            h, w = frame.shape[:2]
            results = self.model.predict(
                frame, conf=self.yolo_conf, imgsz=self.yolo_imgsz, classes=[0], verbose=False)[0]
            self._process(results, w, h)
            self._log_timing(time.time() - t0)

    def _log_timing(self, latency_s):
        if self._timing_log_writer is None:
            return
        with self._timing_lock:
            self._timing_log_writer.writerow([
                '', 'state_detection', self._timing_config, f'{latency_s:.4f}',
                '', f'{latency_s:.4f}', '', f'{time.time():.3f}'])
            self._timing_log_file.flush()

    # ── Detection + classification (ported from CylinderVelocityNode) ──────

    def _largest_box(self, results, w, h):
        if results.boxes is None or len(results.boxes) == 0:
            return None
        best, best_area = None, -1.0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if best is None or best_area <= 0:
            return None
        x1, y1, x2, y2 = best
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _xgb_classify(self, smoothed, cx, cy):
        """Build the lag-window feature row, push it, return the predicted
        state, or None if the pose isn't available yet or the buffer hasn't
        filled up to xgb_window samples yet."""
        with self._psm1_lock:
            pose = self._psm1_pose
        if pose is None:
            return None
        p, q = pose.pose.position, pose.pose.orientation
        feat_row = self._np.array(
            [smoothed, cx, cy, p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=self._np.float32)
        self._feat_buf.append(feat_row)
        if len(self._feat_buf) < self.xgb_window:
            return None
        arr = self._np.array(self._feat_buf, dtype=self._np.float32)
        current = arr[-1]
        lags = arr[:-1][::-1].flatten()
        delta = current - arr[-2]
        feat = self._np.concatenate([current, lags, delta]).reshape(1, -1)
        pred_int = self.xgb_model.predict(feat)[0]
        return self.xgb_le.inverse_transform([pred_int])[0]

    def _set_state(self, new_state):
        if new_state == self._state:
            return
        old_state = self._state
        self._state = new_state
        self._log(f'[STATE] {old_state} -> {new_state}')
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def _process(self, results, w, h):
        now = time.time()
        det = self._largest_box(results, w, h)

        if det is None:
            self._lost_count += 1
            self._prev_center = None
            self._prev_time = None
            smoothed = self._vel_hist[-1] if self._vel_hist else 0.0
            # Keep the lag buffer warm with the same cx=cy=-1 sentinel
            # convention used in the training data, but LOST itself is
            # decided by the rule below, never by this call's prediction.
            self._xgb_classify(smoothed, -1.0, -1.0)
            if self._lost_count >= self.lost_frames:
                self._set_state(self.STATE_LOST)
            return

        cx, cy = det

        # Plausibility gate: reject detections implying a physically
        # impossible jump (misdetection), same as CylinderVelocityNode.
        if self._prev_center is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-6:
                implied_speed = ((cx - self._prev_center[0]) ** 2 +
                                  (cy - self._prev_center[1]) ** 2) ** 0.5 / dt
                if implied_speed > self.max_jump:
                    self._lost_count += 1
                    return

        self._lost_count = 0
        raw_disp = 0.0
        if self._prev_center is not None:
            raw_disp = ((cx - self._prev_center[0]) ** 2 + (cy - self._prev_center[1]) ** 2) ** 0.5
        self._prev_center = (cx, cy)
        self._prev_time = now

        self._vel_hist.append(raw_disp)
        smoothed = sum(self._vel_hist) / len(self._vel_hist)

        new_state = self._xgb_classify(smoothed, cx, cy)
        if new_state is not None and new_state != self.STATE_LOST:
            # LOST is exclusively the no-detection rule above — a real
            # detection this frame means we are definitely not LOST,
            # regardless of what the model says (it was never trained to
            # see LOST with real cx/cy, so this shouldn't fire anyway;
            # guarded for safety).
            self._set_state(new_state)
