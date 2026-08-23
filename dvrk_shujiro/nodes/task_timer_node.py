"""ROS 2 node for task timer"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy, CompressedImage
from geometry_msgs.msg import PoseStamped, WrenchStamped

from ..config import *
from ..metrics.metrics_tracker import MetricsTracker
from ..arduino.arduino_bridge import ArduinoTrigger


class TaskTimerNode(Node):
    """Main ROS 2 node"""

    def __init__(self, gui, settings=None):
        super().__init__('task_timer_gui')
        self.gui = gui

        settings = settings or {}
        self.trigger_source = settings.get('trigger_source', TRIGGER_SOURCE)
        self.track_path = settings.get('track_path', TRACK_PATH_LENGTH)
        self.track_orientation = settings.get('track_orientation', TRACK_ORIENTATION)
        # Force sensor + YOLO orientation only make sense driven by the Arduino
        # protocol (see config.py's ENABLE_FORCE_SENSOR/ENABLE_ORIENTATION_CHECK
        # comment) — ignore them under mono_teleop rather than wiring to nothing.
        # Independent of each other: force capture doesn't need cv2/ultralytics.
        self.enable_force_sensor = (
            settings.get('enable_force_sensor', ENABLE_FORCE_SENSOR)
            and self.trigger_source == 'arduino')
        self.enable_orientation_check = (
            settings.get('enable_orientation_check', ENABLE_ORIENTATION_CHECK)
            and self.trigger_source == 'arduino')
        # Cylinder state detection (STATIONARY/HELD/DROPPED/LOST) only needs
        # the camera + PSM1 pose — both already available regardless of
        # trigger_source — so unlike the two above, it's not Arduino-only.
        self.enable_state_detection = settings.get('enable_state_detection', ENABLE_STATE_DETECTION)

        # One combined label so every measured-latency row this session —
        # from either tracker — is tagged consistently (see
        # scripts/9_analyze_timing.py).
        self._timing_config = '+'.join(
            name for name, on in [('force', self.enable_force_sensor),
                                   ('orientation', self.enable_orientation_check),
                                   ('state', self.enable_state_detection)] if on
        ) or 'none'

        # State
        self.teleop_active = False
        self.teleop_enabled = False
        self.mono_pressed = False
        self.was_running = False

        # Performance monitoring
        self.last_time = None
        self.current_hz = 0.0
        self.sample_count = 0

        # Use MetricsTracker objects (cleaner!)
        self.tracker_psm1 = MetricsTracker("PSM1")
        self.tracker_psm2 = MetricsTracker("PSM2")

        # Pose subscriptions feed the metrics trackers regardless of trigger source
        self.pose_sub_psm1 = self.create_subscription(
            PoseStamped, TOPIC_PSM1_POSE, self.pose_callback_psm1, 10)

        self.pose_sub_psm2 = self.create_subscription(
            PoseStamped, TOPIC_PSM2_POSE, self.pose_callback_psm2, 10)

        self.arduino = None
        self.force_tracker = None
        self.force_popup = None
        self.state_tracker = None
        if self.trigger_source == 'arduino':
            arduino_end_mode = settings.get('arduino_end_mode', ARDUINO_END_MODE)

            if self.enable_force_sensor or self.enable_orientation_check:
                from ..gui.trial_popup import TrialPopup
                from ..arduino.force_orientation_bridge import ForceOrientationTracker

                self.force_popup = TrialPopup(
                    root=self.gui.window_left.root,
                    force_threshold=FORCE_THRESHOLD_N,
                    force_popup_show_ms=int(FORCE_POPUP_SHOW_SEC * 1000),
                    force_popup_margin_x=FORCE_POPUP_MARGIN_X,
                    force_popup_margin_y=FORCE_POPUP_MARGIN_Y)
                self.force_tracker = ForceOrientationTracker(
                    enable_force=self.enable_force_sensor,
                    enable_orientation=self.enable_orientation_check,
                    end_mode=arduino_end_mode,
                    yolo_weights=YOLO_WEIGHTS, yolo_conf=YOLO_CONF, yolo_imgsz=YOLO_IMGSZ,
                    scripts_dir=SCRIPTS_DIR,
                    orientation_tolerance_deg=ORIENTATION_TOLERANCE_DEG,
                    orientation_delay_sec=ORIENTATION_DELAY_SEC,
                    orientation_max_attempts=ORIENTATION_MAX_ATTEMPTS,
                    orientation_retry_delay_sec=ORIENTATION_RETRY_DELAY_SEC,
                    peg_classifications=PEG_CLASSIFICATIONS,
                    csv_path=FORCE_ORIENTATION_CSV, trial_signal_path=TRIAL_SIGNAL_PATH,
                    on_force_result=self._on_force_result, logger=self.get_logger(),
                    smoothing_window=FORCE_SMOOTHING_WINDOW, force_threshold=FORCE_THRESHOLD_N,
                    timing_log_path=TIMING_LOG_CSV, timing_config=self._timing_config)
                # Wire the popup's measured-render-delay callback into the
                # tracker's timing log now that both exist.
                self.force_popup.on_popup_shown = self.force_tracker.log_popup_delay

                if self.enable_force_sensor:
                    self.wrench_sub = self.create_subscription(
                        WrenchStamped, TOPIC_WRENCH, self._on_wrench, 50)

            self.arduino = ArduinoTrigger(
                port=ARDUINO_PORT, baud=ARDUINO_BAUD, end_mode=arduino_end_mode,
                on_lifted=self._arduino_start, on_end=self._arduino_end,
                logger=self.get_logger(),
                on_raw_line=(self.force_tracker.on_raw_line if self.force_tracker else None),
                on_poll=(self.force_tracker.poll if self.force_tracker else None))
            self.arduino.start()
        else:
            # MONO pedal + teleop-enabled drive the trial
            self.teleop_sub = self.create_subscription(
                Bool, TOPIC_TELEOP_ENABLED, self.teleop_callback, 10)

            self.mono_sub = self.create_subscription(
                Joy, TOPIC_OPERATOR_PRESENT, self.mono_callback, 10)

        if self.enable_state_detection:
            from ..state_detection.cylinder_state_tracker import CylinderStateTracker
            self.state_tracker = CylinderStateTracker(
                yolo_weights=YOLO_WEIGHTS, yolo_conf=YOLO_CONF, yolo_imgsz=YOLO_IMGSZ,
                xgb_model_path=STATE_XGB_MODEL_PATH, xgb_window=STATE_XGB_WINDOW,
                vel_window=STATE_VEL_WINDOW, max_jump=STATE_MAX_JUMP,
                lost_frames=STATE_LOST_FRAMES, logger=self.get_logger(),
                timing_log_path=TIMING_LOG_CSV, timing_config=self._timing_config,
                on_state_change=self._on_cylinder_state_change)

        # One shared camera subscription feeds whichever of the two YOLO-
        # based trackers is active, regardless of trigger_source.
        if self.enable_orientation_check or self.enable_state_detection:
            self.camera_sub = self.create_subscription(
                CompressedImage, TOPIC_CAMERA_LEFT, self._on_camera_frame,
                QoSPresetProfiles.SENSOR_DATA.value)

        self.timer = self.create_timer(TIMER_INTERVAL, self.update_timer)
        self.gui.on_timeout = self._handle_timeout

        if self.trigger_source == 'arduino':
            self.get_logger().info('Task timer ready. Waiting for cylinder lift (Arduino)...')
        else:
            self.get_logger().info('Task timer ready. Turn on teleoperation & press MONO.')
    
    def teleop_callback(self, msg):
        old_state = self.teleop_enabled
        self.teleop_enabled = msg.data
        
        if old_state and not msg.data:
            self._log_trial_results()
            self._reset_all_metrics()
        
        self.update_state()
    
    def mono_callback(self, msg):
        if len(msg.buttons) > 0:
            self.mono_pressed = (msg.buttons[0] == 1)
        else:
            self.mono_pressed = False
        self.update_state()
    
    def pose_callback_psm1(self, msg):
        if self.state_tracker:
            # Runs continuously, independent of whether a trial is active —
            # cylinder state isn't a trial-scoped concept.
            self.state_tracker.on_psm1_pose(msg)

        if not self.gui.is_running:
            self.tracker_psm1.last_position = None
            self.tracker_psm1.last_orientation = None
            self.tracker_psm1.last_timestamp = None
            return
        
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w]
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Update tracker
        self.tracker_psm1.update_position(position)
        self.tracker_psm1.update_orientation(orientation, timestamp)
        
        # Update GUI
        self.gui.path_length_psm1 = self.tracker_psm1.path_length
        self.gui.angular_displacement_psm1 = self.tracker_psm1.get_angular_displacement_rad()
        self.gui.orientation_rate_psm1 = self.tracker_psm1.get_orientation_rate_rad()
    
    def pose_callback_psm2(self, msg):
        if not self.gui.is_running:
            self.tracker_psm2.last_position = None
            self.tracker_psm2.last_orientation = None
            self.tracker_psm2.last_timestamp = None
            return
        
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w]
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Update tracker
        self.tracker_psm2.update_position(position)
        self.tracker_psm2.update_orientation(orientation, timestamp)
        
        # Update GUI
        self.gui.path_length_psm2 = self.tracker_psm2.path_length
        self.gui.angular_displacement_psm2 = self.tracker_psm2.get_angular_displacement_rad()
        self.gui.orientation_rate_psm2 = self.tracker_psm2.get_orientation_rate_rad()
    
    def update_state(self):
        should_run = self.teleop_enabled and self.mono_pressed
        
        if should_run and not self.was_running:
            self.gui.start()
            self.sample_count = 0
            self.get_logger().info('⏱️  Timer STARTED (MONO pressed)')
            self.was_running = True
        elif not should_run and self.was_running:
            duration = self.gui.stop()
            if duration is not None:
                path1 = self.tracker_psm1.get_path_mm()
                path2 = self.tracker_psm2.get_path_mm()
                self.get_logger().info(
                    f'⏸️  PAUSED. Time: {duration:.2f}s | '
                    f'PSM1: {path1:.1f}mm | PSM2: {path2:.1f}mm'
                )
            self.was_running = False
    
    def _arduino_start(self):
        """Called from the Arduino thread when the cylinder is lifted."""
        if not self.was_running:
            self.gui.start()
            self.sample_count = 0
            self.get_logger().info('⏱️  Timer STARTED (Arduino: cylinder lifted)')
            self.was_running = True

    def _arduino_end(self):
        """Called from the Arduino thread on the configured end event."""
        if self.was_running:
            self.gui.stop()
            self.was_running = False
            self._log_trial_results()
            self._reset_all_metrics()

    def _on_wrench(self, msg):
        self.force_tracker.on_wrench(msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)

    def _on_cylinder_state_change(self, old_state, new_state):
        # Fires on CylinderStateTracker's background inference thread. Just a
        # plain attribute write — same lock-free pattern the rest of the GUI
        # metrics already use; TimerGUI reads it on the Tk thread every 100ms.
        self.gui.cylinder_state = new_state

    def _on_camera_frame(self, msg):
        # Shared subscription — route to whichever tracker(s) actually need it.
        if self.force_tracker:
            self.force_tracker.on_compressed_image(msg.data)
        if self.state_tracker:
            self.state_tracker.on_compressed_image(msg.data)

    def _on_force_result(self, trial, times, forces, peak, crossings,
                          angle_deg, is_upright, down_color, success, report_fired_at=None):
        """Called (from the Arduino thread) once a trial's full lift->place->
        return cycle is done — independent of when the trial timer itself
        ended if ARDUINO_END_MODE is 'target_placed'."""
        self.force_popup.show_force_result_threadsafe(
            trial, times, forces, peak, crossings, angle_deg, is_upright, down_color, success,
            report_fired_at=report_fired_at)

    def update_timer(self):
        self.gui.tick(TIMER_INTERVAL)
        
        if self.gui.is_running:
            self.sample_count += 1
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.last_time is not None:
                dt = current_time - self.last_time
                self.current_hz = 1.0 / dt if dt > 0 else 0.0
                
                if self.sample_count % TIMER_RATE_HZ == 0:
                    self._log_periodic_update()
            
            self.last_time = current_time
    
    def _log_periodic_update(self):
        parts = [f'[{self.gui.elapsed:.1f}s @ {self.current_hz:.1f}Hz]']

        if self.track_path:
            path1 = self.tracker_psm1.get_path_mm()
            path2 = self.tracker_psm2.get_path_mm()
            parts.append(f'Path: R={path1:.0f}mm L={path2:.0f}mm')

        if self.track_orientation:
            ang1 = self.tracker_psm1.get_angular_displacement_deg()
            ang2 = self.tracker_psm2.get_angular_displacement_deg()
            rate1 = self.tracker_psm1.get_orientation_rate_deg()
            rate2 = self.tracker_psm2.get_orientation_rate_deg()
            parts.append(f'Orient: R={ang1:.0f}° ({rate1:.1f}°/s) L={ang2:.0f}° ({rate2:.1f}°/s)')

        self.get_logger().info(' | '.join(parts))
    
    def _handle_timeout(self):
        """Called (on the Tk main thread) once elapsed reaches MAX_TIME_SEC."""
        self._log_trial_results(outcome='⛔ TRIAL FAILED — TIME LIMIT REACHED')
        self._reset_all_metrics()
        self.get_logger().info('Ready for next trial.')

    def _log_trial_results(self, outcome='✅ TRIAL COMPLETE'):
        if self.gui.elapsed <= 0:
            return

        duration = self.gui.elapsed

        self.get_logger().info('='*60)
        self.get_logger().info(outcome)
        self.get_logger().info('='*60)
        self.get_logger().info(f'Duration: {duration:.2f} s')

        if self.track_path:
            path1 = self.tracker_psm1.get_path_mm()
            path2 = self.tracker_psm2.get_path_mm()
            total_path = path1 + path2
            self.get_logger().info('')
            self.get_logger().info('Path Length:')
            self.get_logger().info(f'  PSM1 (R): {path1:.1f} mm')
            self.get_logger().info(f'  PSM2 (L): {path2:.1f} mm')
            self.get_logger().info(f'  Total:    {total_path:.1f} mm')

        if self.track_orientation:
            ang1_rad = self.tracker_psm1.get_angular_displacement_rad()
            ang2_rad = self.tracker_psm2.get_angular_displacement_rad()
            rate1_rad = self.tracker_psm1.get_orientation_rate_rad()
            rate2_rad = self.tracker_psm2.get_orientation_rate_rad()
            self.get_logger().info('')
            self.get_logger().info('Angular Displacement:')
            self.get_logger().info(f'  PSM1 (R): {ang1_rad:.3f} rad')
            self.get_logger().info(f'  PSM2 (L): {ang2_rad:.3f} rad')
            self.get_logger().info('')
            self.get_logger().info('Average Orientation Rate:')
            self.get_logger().info(f'  PSM1 (R): {rate1_rad:.4f} rad/s')
            self.get_logger().info(f'  PSM2 (L): {rate2_rad:.4f} rad/s')

        self.get_logger().info('='*60)
    
    def _reset_all_metrics(self):
        self.gui.reset()
        self.sample_count = 0
        self.tracker_psm1.reset()
        self.tracker_psm2.reset()

    def start_spinning(self):
        """Start ROS spinning in a background thread."""
        import threading
        thread = threading.Thread(target=self._spin, daemon=True)
        thread.start()

    def _spin(self):
        try:
            rclpy.spin(self)
        except Exception as e:
            print(f"ROS error: {e}")