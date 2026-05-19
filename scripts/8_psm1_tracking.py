#!/usr/bin/env python3
"""
PSM1 gripper position tracker — projects the 3D gripper position directly
onto the camera image, plus an interactive 3D trail view.

Requirements:
  1. /camera_left/camera_info must be published (gives intrinsics K).
  2. A TF transform from the robot base frame to the camera optical frame
     must be available (hand-eye calibration).  The robot frame is read
     from the header.frame_id of the first /PSM1/measured_cp message.
     Set --camera-frame to match your TF tree (e.g. camera_left_optical_frame).

Usage:
    python3 8_psm1_tracking.py
    python3 8_psm1_tracking.py --camera-frame camera_left_optical_frame
    python3 8_psm1_tracking.py --no-cam        # 3D window only
    python3 8_psm1_tracking.py --trail 1000
"""

import argparse
import threading
from collections import deque

import cv2
import numpy as np
import rclpy
import rclpy.duration
import tf2_geometry_msgs          # registers PoseStamped transform support
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, CompressedImage

DEFAULT_CAM_TOPIC   = "/camera_left/compressed"
DEFAULT_INFO_TOPIC  = "/camera_left/camera_info"
DEFAULT_POSE_TOPIC  = "/PSM1/measured_cp"
DEFAULT_CAM_FRAME   = "ECM"   # PSM1 pose is published in ECM frame

VIEW_W = 560
VIEW_H = 560


# ── 3D view (unchanged from previous version) ────────────────────────────────

class View3D:
    def __init__(self, window_name: str):
        self.az = -0.6
        self.el =  0.45
        self._drag, self._last = False, None
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag, self._last = True, (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False
        elif event == cv2.EVENT_MOUSEMOVE and self._drag and self._last:
            dx, dy = x - self._last[0], y - self._last[1]
            self.az += dx * 0.008
            self.el  = float(np.clip(self.el - dy * 0.008, -1.55, 1.55))
            self._last = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.az, self.el = -0.6, 0.45

    def rotation_matrix(self):
        caz, saz = np.cos(self.az), np.sin(self.az)
        cel, sel = np.cos(self.el), np.sin(self.el)
        Rz = np.array([[caz,-saz,0],[saz,caz,0],[0,0,1]], dtype=np.float64)
        Rx = np.array([[1,0,0],[0,cel,-sel],[0,sel,cel]], dtype=np.float64)
        return Rx @ Rz


def draw_3d_view(trail, current, view: View3D) -> np.ndarray:
    canvas = np.full((VIEW_H, VIEW_W, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, "PSM1  3D trail",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "drag: rotate   right-click: reset",
                (10, VIEW_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (65,65,65), 1, cv2.LINE_AA)

    all_pts = list(trail) + ([current] if current else [])
    if not all_pts:
        cv2.putText(canvas, "waiting for pose ...",
                    (20, VIEW_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (90,90,90), 1, cv2.LINE_AA)
        return canvas

    arr      = np.array(all_pts, dtype=np.float64)
    centroid = arr.mean(axis=0)
    extent   = max(np.ptp(arr, axis=0).max(), 1e-4)
    scale    = min(VIEW_W, VIEW_H) * 0.38 / extent
    R        = view.rotation_matrix()
    cx_px, cy_px = VIEW_W//2, VIEW_H//2

    def proj(pts):
        p  = np.atleast_2d(pts).astype(np.float64) - centroid
        r  = (R @ p.T).T
        px = (r[:,0]*scale + cx_px).astype(np.int32)
        py = (-r[:,1]*scale + cy_px).astype(np.int32)
        return px, py, r[:,2]

    # floor grid
    zmin = arr[:,2].min()
    gr   = extent * 0.7
    for i in range(-5, 6):
        t = i/5*gr
        for ax in range(2):
            p1 = centroid.copy(); p1[ax] -= gr; p1[1-ax] += t; p1[2] = zmin
            p2 = centroid.copy(); p2[ax] += gr; p2[1-ax] += t; p2[2] = zmin
            gx, gy, _ = proj(np.array([p1, p2]))
            cv2.line(canvas, (gx[0],gy[0]), (gx[1],gy[1]), (38,38,38), 1)

    # axes
    alen = extent * 0.55
    for vec, col, lbl in (
        (np.array([alen,0,0]), (60,60,220), "X"),
        (np.array([0,alen,0]), (60,200,60), "Y"),
        (np.array([0,0,alen]), (220,60,60), "Z"),
    ):
        ax_pts = np.array([centroid, centroid+vec])
        ax, ay, _ = proj(ax_pts)
        cv2.arrowedLine(canvas, (ax[0],ay[0]), (ax[1],ay[1]), col, 2, tipLength=0.18)
        cv2.putText(canvas, lbl, (ax[1]+5,ay[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # trail
    n = len(all_pts)
    if n >= 2:
        px, py, depth = proj(arr)
        segs = []
        for i in range(1, n):
            t = i/n
            if t < 0.5:
                b = int(55 + 140*(t*2)); col = (b,b,b)
            else:
                u = (t-0.5)*2; col = (0, int(160*u), int(70+185*u))
            segs.append(((depth[i-1]+depth[i])/2,
                          (px[i-1],py[i-1]), (px[i],py[i]), col))
        for _, p1, p2, col in sorted(segs, key=lambda s: s[0]):
            cv2.line(canvas, p1, p2, col, 1)

    if current:
        cpx, cpy, _ = proj(np.array([current]))
        cv2.circle(canvas, (cpx[0],cpy[0]), 6, (0,255,200), -1)
        cv2.circle(canvas, (cpx[0],cpy[0]), 6, (255,255,255), 1)
        x, y, z = current
        cv2.putText(canvas, f"X {x*1000:+.1f}  Y {y*1000:+.1f}  Z {z*1000:+.1f} mm",
                    (10, VIEW_H-22), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0,210,170), 1, cv2.LINE_AA)
    return canvas


# ── camera projection helpers ─────────────────────────────────────────────────

def project_point(K: np.ndarray, dist: np.ndarray,
                  x: float, y: float, z: float):
    """Pin-hole projection of a single 3-D point (already in camera frame)."""
    if z <= 0:
        return None
    pts = np.array([[[x, y, z]]], dtype=np.float64)
    uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    return int(uv[0, 0, 0]), int(uv[0, 0, 1])


def draw_projected_gripper(frame: np.ndarray, uv, pos,
                           in_frame: bool, tf_error: str = "") -> np.ndarray:
    """Overlay the projected gripper dot and status text on the camera frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    if pos is not None:
        x, y, z = pos
        lines = [f"PSM1  X: {x*1000:+7.2f} mm",
                 f"      Y: {y*1000:+7.2f} mm",
                 f"      Z: {z*1000:+7.2f} mm"]
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (8, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,255,200), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "PSM1: waiting for pose ...", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 1, cv2.LINE_AA)

    if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
        u, v = uv
        # outer ring + filled dot
        cv2.circle(out, (u, v), 14, (0, 255, 200),  2)
        cv2.circle(out, (u, v),  5, (0, 255, 200), -1)
        # crosshair lines
        cv2.line(out, (u-20, v), (u-8,  v), (0,255,200), 1)
        cv2.line(out, (u+8,  v), (u+20, v), (0,255,200), 1)
        cv2.line(out, (u, v-20), (u,  v-8), (0,255,200), 1)
        cv2.line(out, (u,  v+8), (u, v+20), (0,255,200), 1)
    elif uv is None and pos is not None:
        if not in_frame:
            cv2.putText(out, "TF lookup failed — see terminal for available frames",
                        (8, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 140, 255), 1, cv2.LINE_AA)
            if tf_error:
                cv2.putText(out, tf_error,
                            (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            (0, 100, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(out, "gripper behind camera (Z <= 0)",
                        (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 100, 200), 1, cv2.LINE_AA)
    elif uv is not None:
        cv2.putText(out, "gripper outside image bounds",
                    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0,100,200), 1, cv2.LINE_AA)

    return out


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class PSM1TrackerNode(Node):
    def __init__(self, cam_topic: str, info_topic: str, pose_topic: str,
                 trail_len: int, camera_frame: str, use_cam: bool,
                 manual_K: np.ndarray | None):
        super().__init__("psm1_tracker")

        self._camera_frame = camera_frame

        # camera intrinsics (filled once CameraInfo arrives)
        self._K    = None
        self._dist = None

        # pose state
        self._position   = None
        self._pose_frame = None   # frame_id from the PoseStamped header
        self._trail      = deque(maxlen=trail_len)
        self._pose_lock  = threading.Lock()

        # annotated camera frame
        self._cam_frame  = None
        self._cam_lock   = threading.Lock()

        # TF2 (only used when pose is NOT already in the camera frame)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._last_tf_error     = ""
        self._tf_diag_done      = False
        self._pose_frame_logged = False
        self._diag_timer = self.create_timer(3.0, self._diag_tf_once)

        # manual intrinsics override (used when camera_info publishes zeros)
        self._manual_K = manual_K

        qos = QoSPresetProfiles.SENSOR_DATA.value

        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self._pose_cb, qos)
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self._info_cb,
            rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value)

        self.get_logger().info(f"Pose:         {pose_topic}")
        self.get_logger().info(f"Camera frame: {camera_frame}")

        if use_cam:
            self.cam_sub = self.create_subscription(
                CompressedImage, cam_topic, self._cam_cb, qos)
            self.get_logger().info(f"Camera topic: {cam_topic}")
        else:
            self.cam_sub = None
            self.create_timer(1/30, self._timer_cb)

    # ── subscribers ──────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self._K is not None:
            return
        K_msg = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K_msg[0, 0] == 0.0:
            self.get_logger().warn(
                "camera_info has fx=0 — intrinsics not calibrated in ROS. "
                "Use --fx --fy --cx --cy to set them manually.")
            if self._manual_K is not None:
                self._K    = self._manual_K
                self._dist = np.zeros(5)
                self.get_logger().info(
                    f"Using manual K  fx={self._K[0,0]:.1f}  fy={self._K[1,1]:.1f}"
                    f"  cx={self._K[0,2]:.1f}  cy={self._K[1,2]:.1f}")
            return
        self._K    = K_msg
        self._dist = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(
            f"Camera intrinsics received  fx={self._K[0,0]:.1f}  "
            f"fy={self._K[1,1]:.1f}  cx={self._K[0,2]:.1f}  cy={self._K[1,2]:.1f}")
        self.destroy_subscription(self.info_sub)

    def _diag_tf_once(self):
        if self._tf_diag_done:
            return
        self._tf_diag_done = True
        frames = self.tf_buffer.all_frames_as_string()
        self.get_logger().info(
            f"\n── TF frames known to buffer ──\n{frames}"
            f"\n── looking for target: '{self._camera_frame}' ──\n"
            f"Tip: run  ros2 run tf2_tools view_frames  for a graph.")
        self.destroy_timer(self._diag_timer)

    def _pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        with self._pose_lock:
            self._position   = (p.x, p.y, p.z)
            self._pose_frame = msg.header.frame_id
            self._trail.append((p.x, p.y, p.z))
        if not self._pose_frame_logged:
            self._pose_frame_logged = True
            self.get_logger().info(
                f"PSM1 pose frame_id = '{msg.header.frame_id}'")

    def _cam_cb(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._pose_lock:
            pos        = self._position
            pose_frame = self._pose_frame
        uv, in_tf = self._project(pos, pose_frame)
        with self._cam_lock:
            self._cam_frame = draw_projected_gripper(
                frame, uv, pos, in_tf, self._last_tf_error)

    def _timer_cb(self):
        # no-cam mode: nothing to display from camera
        pass

    # ── projection ───────────────────────────────────────────────────────────

    def _project(self, pos, pose_frame):
        """
        Project pos to pixel (u,v).
        If the pose is already expressed in the camera frame (e.g. ECM),
        skip TF and project directly.  Otherwise try TF2.
        Returns (uv_or_None, tf_ok: bool).
        """
        if pos is None or self._K is None:
            return None, False

        x, y, z = pos

        # pose is already in the camera body frame — project directly
        if pose_frame == self._camera_frame:
            return project_point(self._K, self._dist, x, y, z), True

        # try TF
        ps = PoseStamped()
        ps.header.frame_id    = pose_frame or ""
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.position.z    = z
        ps.pose.orientation.w = 1.0

        try:
            ps_cam = self.tf_buffer.transform(
                ps, self._camera_frame,
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception as e:
            self._last_tf_error = str(e).split("\n")[0][:80]
            return None, False

        p = ps_cam.pose.position
        return project_point(self._K, self._dist, p.x, p.y, p.z), True

    # ── accessors for main loop ───────────────────────────────────────────────

    def get_trail(self):
        with self._pose_lock:
            return list(self._trail), self._position

    def show_camera(self, window_name: str):
        with self._cam_lock:
            frame = self._cam_frame
        if frame is not None:
            cv2.imshow(window_name, frame)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rclpy.init()

    print(f"[INFO] Pose topic:    {args.pose}")
    print(f"[INFO] Camera frame:  {args.camera_frame}")
    if not args.no_cam:
        print(f"[INFO] Camera topic:  {args.cam}")
        print(f"[INFO] CameraInfo:    {args.info}")
    print(f"[INFO] Trail length:  {args.trail}")
    print("[INFO] Drag 3D window to rotate, right-click to reset.")
    print("[INFO] Press 'q' to quit.\n")

    manual_K = None
    if args.fx > 0:
        manual_K = np.array([[args.fx, 0, args.cx],
                              [0, args.fy, args.cy],
                              [0,  0,   1]], dtype=np.float64)
        print(f"[INFO] Manual K:      fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")

    node = PSM1TrackerNode(
        args.cam, args.info, args.pose,
        args.trail, args.camera_frame, not args.no_cam, manual_K)

    executor    = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    win_3d  = "PSM1 3D View"
    win_cam = "PSM1 Camera"

    view = View3D(win_3d)
    cv2.resizeWindow(win_3d, VIEW_W, VIEW_H)

    if not args.no_cam:
        cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_cam, 640, 480)

    try:
        while rclpy.ok():
            trail, current = node.get_trail()
            cv2.imshow(win_3d, draw_3d_view(trail, current, view))
            if not args.no_cam:
                node.show_camera(win_cam)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Project PSM1 gripper onto camera image + interactive 3D trail.")
    p.add_argument("--cam",          default=DEFAULT_CAM_TOPIC)
    p.add_argument("--info",         default=DEFAULT_INFO_TOPIC,
                   help="CameraInfo topic for intrinsics")
    p.add_argument("--pose",         default=DEFAULT_POSE_TOPIC)
    p.add_argument("--camera-frame", default=DEFAULT_CAM_FRAME,
                   help="TF frame name of the camera optical frame")
    p.add_argument("--trail",        type=int, default=600)
    p.add_argument("--no-cam",  action="store_true",
                   help="Show 3D view only (no camera feed)")
    p.add_argument("--fx", type=float, default=0.0,
                   help="Manual camera focal length x (px) — use when camera_info has zeros")
    p.add_argument("--fy", type=float, default=0.0,
                   help="Manual camera focal length y (px)")
    p.add_argument("--cx", type=float, default=320.0,
                   help="Manual principal point x (px, default: 320)")
    p.add_argument("--cy", type=float, default=240.0,
                   help="Manual principal point y (px, default: 240)")
    return p.parse_args()


if __name__ == "__main__":
    main()
