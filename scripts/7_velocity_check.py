#!/usr/bin/env python3
"""
Step 7 — Real-time cylinder velocity / state check on dVRK camera stream.

Subscribes to /camera_left/compressed, runs YOLO detection, tracks the
centre of the cylinder bounding box over time, computes pixel velocity,
and classifies each frame into one of three states:

    STATIONARY — cylinder still (velocity < --vel-stat)
    HELD       — smooth controlled motion (between --vel-stat and --vel-drop)
    DROPPED    — sudden velocity spike
    LOST       — cylinder not visible for >= --lost consecutive frames

On every state change a line is printed to stdout, e.g.:
    [12:34:56.789] HELD -> DROPPED  vel=73.2px/s  pos=(312,240)

Usage:
    python3 7_velocity_check.py
    python3 7_velocity_check.py --topic /camera_left/compressed
    python3 7_velocity_check.py --weights models/best.pt --conf 0.45
    python3 7_velocity_check.py --vel-stat 6 --vel-drop 80 --lost 4
    python3 7_velocity_check.py --log cylinder_states.csv
"""

import argparse
import re
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

from ultralytics import YOLO

CYLINDER_CLASS_ID = 0
DEFAULT_WEIGHTS   = "models/best_v2.pt"
DEFAULT_TOPIC     = "/camera_left/compressed"

STATE_STATIONARY = "STATIONARY"
STATE_HELD       = "HELD"
STATE_DROPPED    = "DROPPED"
STATE_LOST       = "LOST"

STATE_COLORS = {
    STATE_STATIONARY: (200, 200, 200),
    STATE_HELD:       (0,   220, 0),
    STATE_DROPPED:    (0,    30, 255),
    STATE_LOST:       (0,   165, 255),
}

SPARK_W      = 150
SPARK_H      = 50
SPARK_SAMPLES = 60

PLOT_W       = 700
PLOT_H       = 280
PLOT_PAD_L   = 58   # left margin for Y-axis labels
PLOT_PAD_B   = 36   # bottom margin for X-axis labels
PLOT_PAD_R   = 12
PLOT_PAD_T   = 30
PLOT_HISTORY = 15.0  # seconds of history shown

DIST_W     = 700
DIST_H     = 350
DIST_BINS  = 60


# ── Visualization helpers ─────────────────────────────────────────────────────

def _stamp() -> str:
    """Wall-clock HH:MM:SS.mmm timestamp string."""
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"

# live camera view
def draw_sparkline(frame: np.ndarray, samples, vel_drop: float):
    """Draw a velocity sparkline in the bottom-left corner, in-place."""
    h = frame.shape[0]
    x0, y0 = 6, h - SPARK_H - 6
    canvas = np.full((SPARK_H, SPARK_W, 3), 255, dtype=np.uint8)

    # scale so both the data range and the drop threshold stay visible
    vmax = max(vel_drop * 1.2, max(samples) if samples else 1.0, 1.0)

    drop_y = int(SPARK_H - 1 - (vel_drop / vmax) * (SPARK_H - 1))
    for dx in range(0, SPARK_W, 8):
        cv2.line(canvas, (dx, drop_y), (dx + 4, drop_y), (0, 0, 255), 1)

    if len(samples) >= 2:
        n   = len(samples)
        pts = []
        for i, v in enumerate(samples):
            px = int(i / (n - 1) * (SPARK_W - 1))
            py = int(SPARK_H - 1 - min(v, vmax) / vmax * (SPARK_H - 1))
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 0, 0), 1)

    cv2.rectangle(canvas, (0, 0), (SPARK_W - 1, SPARK_H - 1), (120, 120, 120), 1)
    frame[y0:y0 + SPARK_H, x0:x0 + SPARK_W] = canvas


def draw_velocity_plot(plot_hist, vel_stat: float, vel_drop: float,
                       current_state: str) -> np.ndarray:
    """Render a PLOT_W x PLOT_H velocity-over-time chart as a numpy image."""
    canvas = np.full((PLOT_H, PLOT_W, 3), 245, dtype=np.uint8)

    gx0 = PLOT_PAD_L
    gx1 = PLOT_W - PLOT_PAD_R
    gy0 = PLOT_PAD_T
    gy1 = PLOT_H - PLOT_PAD_B
    gw  = gx1 - gx0
    gh  = gy1 - gy0

    # dynamic Y range
    vmax_data = max((v for _, v, _ in plot_hist), default=0.0)
    vmax = max(vel_drop * 1.4, vmax_data * 1.1, 10.0)

    def to_px(t_rel, v):
        x = gx0 + int(t_rel / PLOT_HISTORY * gw)
        y = gy1 - int(min(v, vmax) / vmax * gh)
        return x, y

    # grid lines
    for i in range(5):
        gy = gy0 + int(i / 4 * gh)
        cv2.line(canvas, (gx0, gy), (gx1, gy), (210, 210, 210), 1)
        v_label = f"{vmax * (1 - i / 4):.0f}"
        cv2.putText(canvas, v_label, (4, gy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    for i in range(6):
        gx = gx0 + int(i / 5 * gw)
        cv2.line(canvas, (gx, gy0), (gx, gy1), (210, 210, 210), 1)
        t_label = f"-{PLOT_HISTORY * (1 - i / 5):.0f}s" if i < 5 else "now"
        cv2.putText(canvas, t_label, (gx - 10, gy1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 80, 80), 1, cv2.LINE_AA)

    # threshold lines
    _, stat_y = to_px(0, vel_stat)
    cv2.line(canvas, (gx0, stat_y), (gx1, stat_y), (0, 180, 0), 1)
    cv2.putText(canvas, f"stat {vel_stat:.0f}", (gx1 - 54, stat_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 150, 0), 1, cv2.LINE_AA)

    _, drop_y = to_px(0, vel_drop)
    for dx in range(gx0, gx1, 10):
        cv2.line(canvas, (dx, drop_y), (min(dx + 6, gx1), drop_y), (0, 30, 220), 1)
    cv2.putText(canvas, f"drop {vel_drop:.0f}", (gx1 - 60, drop_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 30, 200), 1, cv2.LINE_AA)

    # velocity trace (state-colored segments)
    if len(plot_hist) >= 2:
        now_t = plot_hist[-1][0]
        pts_by_state = []
        for abs_t, v, state in plot_hist:
            t_rel = abs_t - (now_t - PLOT_HISTORY)
            if t_rel < 0:
                continue
            pts_by_state.append((to_px(t_rel, v), STATE_COLORS[state]))

        for i in range(1, len(pts_by_state)):
            p1, _ = pts_by_state[i - 1]
            p2, col = pts_by_state[i]
            cv2.line(canvas, p1, p2, col, 2)

        # current value dot
        last_pt, last_col = pts_by_state[-1]
        cv2.circle(canvas, last_pt, 5, last_col, -1)
        cv2.circle(canvas, last_pt, 5, (0, 0, 0), 1)
        vel_now = plot_hist[-1][1]
        cv2.putText(canvas, f"{vel_now:.1f} px/s", (last_pt[0] + 6, last_pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, last_col, 1, cv2.LINE_AA)

    # axes border
    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (60, 60, 60), 1)

    # title
    title = f"Cylinder displacement  |  state: {current_state}"
    cv2.putText(canvas, title, (gx0, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    # Y-axis label (rotated via transpose trick)
    label_img = np.full((14, 70, 3), 245, dtype=np.uint8)
    cv2.putText(label_img, "displacement (px)", (0, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
    label_rot = cv2.rotate(label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lh, lw = label_rot.shape[:2]
    canvas[gy0 + (gh - lh) // 2: gy0 + (gh - lh) // 2 + lh, 0:lw] = label_rot

    return canvas

# use it by running "--mode dist"
def draw_distribution_plot(displacements: list) -> np.ndarray:
    """Render a live histogram of raw per-frame pixel displacements (no dt division)."""
    PAD_L, PAD_B, PAD_R, PAD_T = 58, 46, 16, 35
    canvas = np.full((DIST_H, DIST_W, 3), 245, dtype=np.uint8)
    gx0, gx1 = PAD_L, DIST_W - PAD_R
    gy0, gy1 = PAD_T, DIST_H - PAD_B
    gw, gh = gx1 - gx0, gy1 - gy0

    n = len(displacements)
    cv2.putText(canvas, f"Displacement distribution  (N={n} samples)  [px/frame]",
                (gx0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    if n < 2:
        cv2.putText(canvas, "Collecting data...", (gx0 + 20, gy0 + gh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (60, 60, 60), 1)
        return canvas

    arr = np.array(displacements, dtype=np.float32)
    p99 = float(np.percentile(arr, 99))
    xmax = max(p99 * 1.1, 1.0)

    counts, edges = np.histogram(arr, bins=DIST_BINS, range=(0.0, xmax))
    max_count = max(int(counts.max()), 1)

    # bars
    bar_w = gw / DIST_BINS
    for i, c in enumerate(counts):
        bx0 = gx0 + int(i * bar_w)
        bx1 = gx0 + int((i + 1) * bar_w) - 1
        by0 = gy1 - int(c / max_count * gh)
        if by0 < gy1:
            cv2.rectangle(canvas, (bx0, by0), (bx1, gy1), (100, 150, 220), -1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, gy1), (180, 180, 180), 1)

    # Y grid + count labels
    for i in range(5):
        gy = gy0 + int(i / 4 * gh)
        cv2.line(canvas, (gx0, gy), (gx1, gy), (210, 210, 210), 1)
        cv2.putText(canvas, f"{int(max_count * (1 - i / 4))}",
                    (4, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (80, 80, 80), 1, cv2.LINE_AA)

    # X grid + displacement labels
    for i in range(6):
        gx = gx0 + int(i / 5 * gw)
        cv2.line(canvas, (gx, gy0), (gx, gy1), (210, 210, 210), 1)
        cv2.putText(canvas, f"{xmax * i / 5:.1f}",
                    (gx - 8, gy1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "displacement (px)", (gx0 + gw // 2 - 50, gy1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    # stats panel (top-right)
    mean   = float(np.mean(arr))
    std    = float(np.std(arr))
    median = float(np.median(arr))
    p95    = float(np.percentile(arr, 95))
    for i, txt in enumerate([f"mean  {mean:.2f}", f"std   {std:.2f}",
                              f"med   {median:.2f}", f"p95   {p95:.2f}"]):
        cv2.putText(canvas, txt, (gx1 - 110, gy0 + 14 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (60, 60, 60), 1)
    return canvas


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class CylinderVelocityNode(Node):
    def __init__(self, topic: str, model: YOLO, conf: float, imgsz: int,
                 vel_stat: float, vel_drop: float, lost_frames: int,
                 vel_window: int, spike_frames: int, settle_frames: int,
                 max_jump: float, drop_timeout: float, log_path, settings: dict = None):
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', topic)
        safe = re.sub(r'_+', '_', safe).strip('_')
        super().__init__(f"cylinder_velocity_{safe}")

        self.model = model
        self.conf  = conf
        self.imgsz = imgsz

        self.vel_stat    = vel_stat
        self.vel_drop    = vel_drop
        self.lost_frames = lost_frames
        self.max_jump    = max_jump

        self._raw_frame    = None
        self._raw_frame_id = 0
        self._last_infer_id   = -1
        self._annotated_frame = None
        self._raw_lock = threading.Lock()
        self._ann_lock = threading.Lock()

        self.spike_frames  = spike_frames
        self.settle_frames = settle_frames
        self.drop_timeout  = drop_timeout

        # state machine (lives entirely in the inference thread)
        self._state        = STATE_STATIONARY
        self._prev_center  = None
        self._prev_time    = None
        self._lost_count   = 0
        self._spike_count  = 0   # consecutive frames above vel_drop
        self._settle_count = 0   # consecutive frames below vel_stat
        self._event_count  = 0
        self._dropped_at   = None  # wall-clock time when DROPPED was entered
        self._vel_hist     = deque(maxlen=vel_window)
        self._spark_hist   = deque(maxlen=SPARK_SAMPLES)
        self._plot_hist    = deque(maxlen=2000)  # (abs_time, smoothed_vel, state)
        self._plot_frame   = None
        self._plot_lock    = threading.Lock()

        self._dist_hist          = []   # all raw_vel samples, unbounded
        self._dist_frame         = None
        self._dist_lock          = threading.Lock()
        self._last_dist_update   = 0.0  # throttle histogram redraws

        # PSM1 kinematic state (updated by ROS callback)
        self._psm1_lock = threading.Lock()
        self._psm1_pose = None  # latest geometry_msgs/PoseStamped

        self._log_file = open(log_path, "w") if log_path else None
        if self._log_file is not None:
            if settings and settings.get("title"):
                self._log_file.write(f"# title={settings['title']}\n")
            self._log_file.write(f"# recorded {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if settings:
                for k, v in settings.items():
                    if k != "title":
                        self._log_file.write(f"# {k}={v}\n")
            self._log_file.write(
                "timestamp_s,state,vel_px_s,cx,cy,"
                "psm1_x,psm1_y,psm1_z,psm1_qx,psm1_qy,psm1_qz,psm1_qw\n")
        self._start_time = time.time()

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.sub = self.create_subscription(CompressedImage, topic, self._cb, qos)
        self.sub_psm1 = self.create_subscription(
            PoseStamped, "/PSM1/measured_cp", self._psm1_cb, qos)
        self.get_logger().info(f"Subscribed to {topic}")

        self._stop        = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

    def _psm1_cb(self, msg: PoseStamped):
        with self._psm1_lock:
            self._psm1_pose = msg

    # called whenever a new camera frame arrives (~30 fps)
    def _cb(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._raw_lock:
            self._raw_frame    = frame
            self._raw_frame_id += 1

    # runs in a background thread
    def _infer_loop(self):
        while not self._stop.is_set():
            with self._raw_lock:
                frame_id = self._raw_frame_id
                frame    = self._raw_frame

            if frame is None or frame_id == self._last_infer_id:
                self._stop.wait(0.005)
                continue
            self._last_infer_id = frame_id

            results = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz,
                classes=[CYLINDER_CLASS_ID], verbose=False,
            )[0]
            annotated = self._process(frame, results)

            with self._ann_lock:
                self._annotated_frame = annotated

    def _largest_box(self, results, w: int, h: int):
        """Return (cx, cy, x1, y1, x2, y2) of the biggest cylinder box, or None."""
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
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x1, y1, x2, y2

    # handles state transitions 
    def _set_state(self, new_state: str, vel: float, center):
        if new_state == self._state:
            return
        self._event_count += 1
        cx = int(center[0]) if center is not None else -1
        cy = int(center[1]) if center is not None else -1
        print(f"[{_stamp()}] {self._state} -> {new_state}  "
              f"disp={vel:.1f}px  pos=({cx},{cy})")
        self._state = new_state
        self._dropped_at = time.time() if new_state == STATE_DROPPED else None

    # state machine, called every frame
    def _process(self, frame: np.ndarray, results) -> np.ndarray:
        out  = frame.copy()
        h, w = frame.shape[:2]
        now  = time.time()

        det = self._largest_box(results, w, h)

        if det is None:
            self._lost_count  += 1
            self._spike_count  = 0
            self._settle_count = 0
            if (self._lost_count >= self.lost_frames
                    and self._state != STATE_LOST):
                self._set_state(STATE_LOST, 0.0, None)
            self._prev_center = None
            self._prev_time   = None
            smoothed = self._vel_hist[-1] if self._vel_hist else 0.0
            self._finish(out, None, smoothed, now)
            return out

        cx, cy, x1, y1, x2, y2 = det

        # plausibility gate: reject detections that imply physically impossible speed
        if (self._prev_center is not None and self._prev_time is not None):
            dt = now - self._prev_time
            if dt > 1e-6:
                implied_speed = float(np.hypot(cx - self._prev_center[0],
                                               cy - self._prev_center[1])) / dt
                if implied_speed > self.max_jump:
                    self._lost_count += 1
                    smoothed = self._vel_hist[-1] if self._vel_hist else 0.0
                    self._finish(out, None, smoothed, now)
                    return out

        self._lost_count = 0
        center = (cx, cy)

        raw_disp = 0.0
        if self._prev_center is not None:
            raw_disp = float(np.hypot(cx - self._prev_center[0],
                                      cy - self._prev_center[1]))
        self._prev_center = center
        self._prev_time   = now

        if raw_disp > 0.0:
            self._dist_hist.append(raw_disp)

        self._vel_hist.append(raw_disp)
        smoothed = float(np.mean(self._vel_hist))

        # spike counter: require N consecutive raw-disp spikes before DROPPED
        if raw_disp >= self.vel_drop:
            self._spike_count += 1
        else:
            self._spike_count = 0

        # settle counter: require N consecutive frames below vel_stat before STATIONARY
        if smoothed < self.vel_stat:
            self._settle_count += 1
        else:
            self._settle_count = 0

        if self._spike_count >= self.spike_frames:
            self._set_state(STATE_DROPPED, raw_disp, center)
            self._spike_count = 0
        elif self._state == STATE_DROPPED:
            # exit via settle OR via timeout (cylinder stopped bouncing)
            timed_out = (self._dropped_at is not None and
                         now - self._dropped_at >= self.drop_timeout)
            if self._settle_count >= self.settle_frames or timed_out:
                self._set_state(STATE_STATIONARY, smoothed, center)
        elif self._state == STATE_LOST:
            # cylinder re-appeared → go to HELD immediately, settle to STATIONARY normally
            self._set_state(STATE_HELD, smoothed, center)
        elif self._settle_count >= self.settle_frames:
            self._set_state(STATE_STATIONARY, smoothed, center)
        else:
            self._set_state(STATE_HELD, smoothed, center)

        color = STATE_COLORS[self._state]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # center marker
        icx, icy = int(cx), int(cy)
        arm = 8
        cv2.line(out, (icx - arm, icy), (icx + arm, icy), color, 2)
        cv2.line(out, (icx, icy - arm), (icx, icy + arm), color, 2)
        cv2.circle(out, (icx, icy), 4, (255, 255, 255), -1)
        cv2.circle(out, (icx, icy), 4, color, 1)
        cv2.putText(out, f"({icx},{icy})", (icx + 8, icy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        label = f"{self._state} {smoothed:.1f}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1, th + 6)
        cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
        cv2.putText(out, label, (x1 + 2, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        self._finish(out, center, smoothed, now)
        return out

    # called at the end of every _process() pass,
    def _finish(self, out: np.ndarray, center, smoothed: float, now: float):
        """Sparkline, HUD text, and CSV row — shared by both detection paths."""
        self._spark_hist.append(smoothed)
        draw_sparkline(out, list(self._spark_hist), self.vel_drop)

        self._plot_hist.append((now, smoothed, self._state))
        plot = draw_velocity_plot(
            list(self._plot_hist), self.vel_stat, self.vel_drop, self._state)
        with self._plot_lock:
            self._plot_frame = plot

        if now - self._last_dist_update > 0.25:
            self._last_dist_update = now
            dist = draw_distribution_plot(self._dist_hist)
            with self._dist_lock:
                self._dist_frame = dist

        hud = f"State: {self._state} | Events: {self._event_count}"
        cv2.rectangle(out, (4, 4), (12 + len(hud) * 11, 30), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if self._log_file is not None:
            cx = int(center[0]) if center is not None else -1
            cy = int(center[1]) if center is not None else -1
            with self._psm1_lock:
                pose = self._psm1_pose
            if pose is not None:
                p = pose.pose.position
                q = pose.pose.orientation
                psm1_str = f"{p.x:.6f},{p.y:.6f},{p.z:.6f},{q.x:.6f},{q.y:.6f},{q.z:.6f},{q.w:.6f}"
            else:
                psm1_str = ",,,,,,,"
            self._log_file.write(
                f"{now - self._start_time:.3f},{self._state},"
                f"{smoothed:.3f},{cx},{cy},{psm1_str}\n")

    def get_annotated_frame(self):
        with self._ann_lock:
            return self._annotated_frame

    def show(self, window_name: str):
        with self._ann_lock:
            frame = self._annotated_frame
        if frame is not None:
            cv2.imshow(window_name, frame)

    def show_plot(self, window_name: str):
        with self._plot_lock:
            frame = self._plot_frame
        if frame is not None:
            cv2.imshow(window_name, frame)

    def show_dist_plot(self, window_name: str):
        with self._dist_lock:
            frame = self._dist_frame
        if frame is not None:
            cv2.imshow(window_name, frame)

    def stop(self):
        self._stop.set()
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rclpy.init()
    model = YOLO(args.weights)
    print(f"[INFO] Loaded weights: {args.weights}")
    print(f"[INFO] Topic:          {args.topic}")
    print(f"[INFO] Confidence:     {args.conf}")
    print(f"[INFO] Inference size: {args.imgsz}")
    print(f"[INFO] Stationary <    {args.vel_stat} px/frame")
    print(f"[INFO] Drop >=         {args.vel_drop} px/frame")
    print(f"[INFO] Lost frames:    {args.lost}")
    print(f"[INFO] Vel window:     {args.vel_window}")
    print(f"[INFO] Spike frames:   {args.spike_frames}")
    print(f"[INFO] Settle frames:  {args.settle_frames}")
    print(f"[INFO] Max jump:       {args.max_jump} px/s")
    print(f"[INFO] Drop timeout:   {args.drop_timeout} s")
    if args.log == "__auto__":
        ts = time.strftime("%Y%m%d_%H%M%S")
        slug = args.title.replace(" ", "_") if args.title else "session"
        args.log = f"{slug}_{ts}.csv"
    if args.log:
        print(f"[INFO] CSV log:        {args.log}")
    if args.record == "__auto__":
        ts = time.strftime("%Y%m%d_%H%M%S")
        slug = args.title.replace(" ", "_") if args.title else "session"
        args.record = f"{slug}_{ts}.mp4"
    if args.record:
        print(f"[INFO] Video record:   {args.record}")
    print("[INFO] Press 'q' to quit.\n")

    settings = {
        "title":         args.title,
        "weights":       args.weights,
        "topic":         args.topic,
        "conf":          args.conf,
        "imgsz":         args.imgsz,
        "vel_stat":      args.vel_stat,
        "vel_drop":      args.vel_drop,
        "lost":          args.lost,
        "vel_window":    args.vel_window,
        "spike_frames":  args.spike_frames,
        "settle_frames": args.settle_frames,
        "max_jump":      args.max_jump,
        "drop_timeout":  args.drop_timeout,
    }

    window      = "Cylinder Velocity Check"
    plot_window = "Cylinder Velocity Plot" if args.mode == "velocity" else "Velocity Distribution"
    node   = CylinderVelocityNode(
        args.topic, model, args.conf, args.imgsz,
        args.vel_stat, args.vel_drop, args.lost, args.vel_window,
        args.spike_frames, args.settle_frames, args.max_jump,
        args.drop_timeout, args.log, settings)

    executor    = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)
    pw = DIST_W if args.mode == "dist" else PLOT_W
    ph = DIST_H if args.mode == "dist" else PLOT_H
    cv2.namedWindow(plot_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(plot_window, pw, ph)

    video_writer = None

    try:
        while rclpy.ok():
            frame = node.get_annotated_frame()
            if frame is not None:
                cv2.imshow(window, frame)
                if args.record:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(args.record, fourcc, 30.0, (w, h))
                        print(f"[INFO] Recording started → {args.record}")
                    video_writer.write(frame)
            if args.mode == "dist":
                node.show_dist_plot(plot_window)
            else:
                node.show_plot(plot_window)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved → {args.record}")
        cv2.destroyAllWindows()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topic",                     default=DEFAULT_TOPIC,    help="Compressed image topic")
    p.add_argument("--weights",                   default=DEFAULT_WEIGHTS,  help="YOLO .pt weights")
    p.add_argument("--conf",          type=float, default=0.45,             help="Detection confidence threshold")
    p.add_argument("--imgsz",         type=int,   default=320,              help="YOLO inference resolution")
    p.add_argument("--vel-stat",      type=float, default=2,              help="Stationary threshold px/frame")
    p.add_argument("--vel-drop",      type=float, default=50.0,             help="Drop threshold px/frame")
    p.add_argument("--lost",          type=int,   default=10,               help="Consecutive missed detections before tracking-loss -> DROPPED")
    p.add_argument("--vel-window",    type=int,   default=5,                help="Rolling average window for velocity smoothing") # the higher, the more stable but the longer the lag
    p.add_argument("--spike-frames",  type=int,   default=2,                help="Consecutive frames above vel-drop before DROPPED")
    p.add_argument("--settle-frames", type=int,   default=5,                help="Consecutive frames below vel-stat before STATIONARY")
    p.add_argument("--log",           nargs="?",  default=None, const="__auto__",
                   help="CSV output path; omit value to auto-generate from title + timestamp")
    p.add_argument("--record",        nargs="?",  default=None, const="__auto__",
                   help="MP4 output path; omit value to auto-generate from title + timestamp")
    p.add_argument("--title",                     default=None,             help="Optional session title written as first comment in the CSV")
    p.add_argument("--max-jump",      type=float, default=800.0,            help="Max implied speed (px/s) before a detection is rejected as spurious")
    p.add_argument("--drop-timeout",  type=float, default=2.0,              help="Seconds before DROPPED auto-resets to STATIONARY if settle never triggers")
    p.add_argument("--mode",                      default="velocity", 
                                        choices=["velocity", "dist"],       help="'velocity' shows time-series plot; 'dist' shows histogram of all samples")
    return p.parse_args()


if __name__ == "__main__":
    main()
