from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, LaserScan

from limo_test.camera_lane_detector import LaneResult, OpenCVLaneDetector
from limo_test.lidar_obstacle_avoidance import LidarObstacleAvoidance

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover - robot image dependency.
    CvBridge = None


class LimoAutonomousDrive(Node):
    """LIMO 자율주행 통합 노드.

    파일별 책임:
    - camera_lane_detector.py: OpenCV로 차선을 보고 LaneResult를 만든다.
    - lidar_obstacle_avoidance.py: LaserScan으로 장애물 회피/AEB/fallback을 만든다.
    - autonomous_drive.py: 센서 콜백을 받고 두 판단을 합쳐 /cmd_vel을 발행한다.
    """

    def __init__(self):
        super().__init__("limo_autonomous_drive")
        self._declare_core_parameters()

        self.bridge = CvBridge() if CvBridge else None
        if self.bridge is None:
            self.get_logger().warn(
                "cv_bridge is not available. Install ros-humble-cv-bridge."
            )

        # 알고리즘 구현은 별도 파일에 있습니다.
        # 이 노드는 결과를 연결만 합니다.
        self.camera = OpenCVLaneDetector(self)
        self.lidar = LidarObstacleAvoidance(self)

        self.image_topic = self.get_parameter("image_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.debug_window_created = False

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 2)

        # 카메라 콜백: OpenCV 차선 인식 결과만 저장합니다.
        # 제어 주기는 timer에서 고정해
        # 센서 콜백 지연이 조향 주기를 흔들지 않게 합니다.
        self.image_subs = []
        for qos_profile in self.image_qos_profiles("image_qos"):
            self.image_subs.append(
                self.create_subscription(
                    Image,
                    self.image_topic,
                    self.image_callback,
                    qos_profile,
                )
            )

        # 라이다 콜백: 최신 LaserScan만 저장합니다.
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.latest_lane: Optional[LaneResult] = None
        self.latest_scan: Optional[LaserScan] = None
        self.latest_debug_image = None
        self.last_image_time = None
        self.last_scan_time = None

        # PID와 차선 유지 상태는 통합 제어의 기억값입니다.
        self.prev_error = 0.0
        self.last_steering = 0.0
        self.last_lane_error = 0.0
        self.last_lane_seen_time = None
        self.integral = 0.0
        self.prev_time = self.get_clock().now()

        self.get_logger().info(
            "LIMO autonomous drive ready: "
            f"image={self.image_topic}, scan={self.scan_topic}, "
            f"cmd={self.cmd_vel_topic}"
        )

    def _declare_core_parameters(self):
        # 토픽 이름은 launch argument 또는 YAML에서 덮어쓸 수 있습니다.
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("debug_image_topic", "/limo/autonomy/debug_image")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("display_debug_window", True)
        self.declare_parameter("debug_window_name", "LIMO lane detection")
        self.declare_parameter("debug_window_scale", 1.0)
        self.declare_parameter("publish_drive_log", True)
        self.declare_parameter("drive_log_period_sec", 0.5)
        self.declare_parameter("image_qos", "auto")
        self.declare_parameter("depth_qos", "auto")

        # 주행 속도와 조향 제한입니다.
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed", 1.40)
        self.declare_parameter("min_speed", 0.18)
        self.declare_parameter("lane_follow_min_speed", 0.55)
        self.declare_parameter("caution_speed", 0.35)
        self.declare_parameter("straight_min_speed", 1.40)
        self.declare_parameter("straight_steering_threshold", 0.25)
        self.declare_parameter("max_angular", 1.35)
        self.declare_parameter("steering_smoothing", 0.22)

        # 차선 중심 오차를 조향으로 바꾸는 PID입니다.
        self.declare_parameter("kp", 1.70)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.22)
        self.declare_parameter("integral_limit", 0.45)
        self.declare_parameter("max_lane_error_step", 0.35)

        # 차선이 순간적으로 끊겼을 때
        # 마지막 정상 방향을 짧게 유지합니다.
        self.declare_parameter("lane_hold_time_sec", 0.60)
        self.declare_parameter("lane_hold_speed", 0.75)
        self.declare_parameter("lane_hold_min_confidence", 0.12)

        # 차선이 사라지면 라이다 fallback으로 열린 방향을 찾습니다.
        self.declare_parameter("enable_lane_lost_drive", True)
        self.declare_parameter("lane_lost_steering_decay", 0.45)
        self.declare_parameter("lane_lost_use_lidar", True)

        # 센서가 오래되면 잘못된 명령을 내지 않도록 제한합니다.
        self.declare_parameter("image_timeout_sec", 0.7)
        self.declare_parameter("scan_timeout_sec", 0.7)
        self.declare_parameter("depth_timeout_sec", 0.7)
        self.declare_parameter("low_obstacle_speed", 0.10)

    def image_qos_profiles(self, parameter_name: str):
        mode = str(self.get_parameter(parameter_name).value).lower()
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        if mode == "reliable":
            return [reliable]
        if mode == "best_effort":
            return [best_effort]

        # auto: 카메라 드라이버 QoS가 제각각이라 둘 다 열어둡니다.
        return [reliable, best_effort]

    def image_callback(self, msg: Image):
        if self.bridge is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert image: {exc}")
            return

        self.latest_lane, self.latest_debug_image = self.camera.process(frame)
        self.show_debug_window(self.latest_debug_image)
        self.last_image_time = self.get_clock().now()

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

    def control_loop(self):
        """카메라 차선 추종을 기본으로 하고 라이다 회피를 우선 적용."""

        now = self.get_clock().now()
        image_ok = self.is_recent(self.last_image_time, "image_timeout_sec")
        scan_ok = self.is_recent(self.last_scan_time, "scan_timeout_sec")

        if not image_ok:
            self.publish_stop("Waiting for camera image")
            return

        lane = self.latest_lane
        recovery = self.lidar.recovery_command(now, scan_ok, self.latest_scan)
        if recovery is not None:
            speed, steering, drive_state = recovery
            self.publish_command(speed, steering)
            self.log_drive_status(lane, self.lane_is_valid(lane), drive_state, scan_ok)
            return

        dt = max((now - self.prev_time).nanoseconds / 1e9, 1e-3)
        self.prev_time = now

        speed, steering, lane_valid, lane_lost_active, drive_state = (
            self.lane_follow_command(now, lane, dt, scan_ok)
        )
        if drive_state == "stop":
            return
        if drive_state == "camera_invalid":
            self.publish_command(0.0, 0.0)
            self.log_drive_status(lane, lane_valid, drive_state, scan_ok)
            return

        # 라이다 판단은 차선 추종보다 우선순위가 높습니다.
        if scan_ok and self.latest_scan is not None:
            speed, steering, drive_state = self.apply_lidar_avoidance(
                speed,
                steering,
                lane_valid,
                lane_lost_active,
                drive_state,
                now,
            )
        else:
            speed = min(speed, float(self.get_parameter("caution_speed").value))
            drive_state = "scan_timeout"

        if self.should_slow_for_camera_obstacle(lane, lane_valid, speed):
            speed = min(speed, float(self.get_parameter("low_obstacle_speed").value))
            if drive_state == "lane_follow":
                drive_state = "camera_low_obstacle"

        if lane_lost_active and scan_ok and drive_state not in ("aeb_stop", "camera_invalid"):
            speed = float(self.get_parameter("lane_lost_min_speed").value)

        self.publish_command(speed, steering)
        self.log_drive_status(lane, lane_valid, drive_state, scan_ok)
        self.publish_debug_image()

    def lane_follow_command(self, now, lane, dt: float, scan_ok: bool):
        lane_valid = self.lane_is_valid(lane)
        lane_lost_active = False
        drive_state = "lane_follow"

        if lane_valid:
            lane_error = self.stable_lane_error(lane.center_error)
            self.last_lane_error = lane_error
            self.last_lane_seen_time = now
            steering = self.smoothed_steering(self.pid_steering(lane_error, dt))
            speed = self.speed_from_lane(lane, steering)
            return speed, steering, True, False, drive_state

        if self.can_hold_lane(now, lane):
            steering = self.smoothed_steering(self.pid_steering(self.last_lane_error, dt))
            speed = min(
                self.speed_from_lane(lane, steering),
                float(self.get_parameter("lane_hold_speed").value),
            )
            return speed, steering, True, False, "lane_hold"

        if lane is not None and not lane.camera_valid:
            self.integral = 0.0
            return 0.0, 0.0, False, False, "camera_invalid"

        if bool(self.get_parameter("enable_lane_lost_drive").value):
            lane_lost_active = True
            self.integral = 0.0
            if (
                scan_ok
                and self.latest_scan is not None
                and bool(self.get_parameter("lane_lost_use_lidar").value)
            ):
                speed, steering = self.lidar.lane_lost_command(self.latest_scan)
                self.get_logger().warn(
                    "Lane not detected. Using lidar fallback.",
                    throttle_duration_sec=1.0,
                )
                return speed, steering, False, lane_lost_active, "lane_lost_lidar"

            decay = float(self.get_parameter("lane_lost_steering_decay").value)
            return (
                float(self.get_parameter("lane_lost_speed").value),
                self.last_steering * decay,
                False,
                lane_lost_active,
                "lane_lost",
            )

        self.publish_stop("Lane not detected")
        return 0.0, 0.0, False, False, "stop"

    def apply_lidar_avoidance(
        self,
        speed: float,
        steering: float,
        lane_valid: bool,
        lane_lost_active: bool,
        drive_state: str,
        now,
    ):
        obstacle_speed, obstacle_steering, mode = self.lidar.obstacle_command(
            self.latest_scan
        )

        if mode == "clear":
            return speed, steering, drive_state

        if mode == "aeb_stop":
            if bool(self.get_parameter("enable_aeb_recovery").value):
                self.lidar.start_recovery(now, self.latest_scan)
                recovery = self.lidar.recovery_command(now, True, self.latest_scan)
                if recovery is not None:
                    return recovery
            return obstacle_speed, obstacle_steering, mode

        if mode == "stop_turn":
            if lane_lost_active:
                speed = float(self.get_parameter("lane_lost_min_speed").value)
            elif lane_valid:
                speed = float(self.get_parameter("caution_speed").value)
            else:
                speed = obstacle_speed
            return speed, obstacle_steering, mode

        if mode in ("slow_avoid", "side_avoid"):
            if lane_valid:
                speed = max(
                    min(speed, obstacle_speed),
                    float(self.get_parameter("lane_follow_min_speed").value),
                )
            else:
                speed = min(speed, obstacle_speed)
            return speed, steering + obstacle_steering, mode

        if mode == "tunnel_center":
            return speed, steering + obstacle_steering, mode

        if mode == "slalom_gap":
            if lane_lost_active:
                speed = max(speed, float(self.get_parameter("lane_lost_min_speed").value))
                steering = obstacle_steering
            else:
                speed = max(
                    min(speed, obstacle_speed),
                    float(self.get_parameter("lane_follow_min_speed").value),
                )
                steering = 0.35 * steering + obstacle_steering
            return speed, steering, mode

        return speed, steering, drive_state

    def lane_is_valid(self, lane: Optional[LaneResult]) -> bool:
        if lane is None:
            return False
        return lane.confidence >= float(self.get_parameter("min_lane_confidence").value)

    def can_hold_lane(self, now, lane: Optional[LaneResult]) -> bool:
        if self.last_lane_seen_time is None or lane is None or not lane.camera_valid:
            return False

        age = (now - self.last_lane_seen_time).nanoseconds / 1e9
        if age > float(self.get_parameter("lane_hold_time_sec").value):
            return False

        min_hold_conf = float(self.get_parameter("lane_hold_min_confidence").value)
        return lane.confidence >= min_hold_conf or lane.geometry_valid or lane.road_valid

    def stable_lane_error(self, error: float) -> float:
        if self.last_lane_seen_time is None:
            return error
        max_step = float(self.get_parameter("max_lane_error_step").value)
        delta = float(np.clip(error - self.last_lane_error, -max_step, max_step))
        return float(np.clip(self.last_lane_error + delta, -1.0, 1.0))

    def pid_steering(self, error: float, dt: float) -> float:
        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)
        limit = float(self.get_parameter("integral_limit").value)

        self.integral = float(np.clip(self.integral + error * dt, -limit, limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return kp * error + ki * self.integral + kd * derivative

    def smoothed_steering(self, raw_steering: float) -> float:
        smoothing = float(self.get_parameter("steering_smoothing").value)
        steering = (1.0 - smoothing) * raw_steering + smoothing * self.last_steering
        self.last_steering = steering
        return steering

    def speed_from_lane(self, lane: LaneResult, steering: float) -> float:
        max_speed = float(self.get_parameter("max_speed").value)
        min_speed = float(self.get_parameter("min_speed").value)
        lane_min_speed = float(self.get_parameter("lane_follow_min_speed").value)
        straight_min_speed = float(self.get_parameter("straight_min_speed").value)
        straight_threshold = float(
            self.get_parameter("straight_steering_threshold").value
        )

        turn_factor = 1.0 - min(abs(steering) / 1.4, 0.75)
        confidence_factor = 0.45 + 0.55 * lane.confidence
        speed = max(min_speed, lane_min_speed, max_speed * turn_factor * confidence_factor)
        if abs(steering) <= straight_threshold:
            speed = max(speed, straight_min_speed)
        return min(speed, max_speed)

    def should_slow_for_camera_obstacle(
        self,
        lane: Optional[LaneResult],
        lane_valid: bool,
        speed: float,
    ) -> bool:
        return (
            lane is not None
            and lane.road_valid
            and lane.camera_obstacle
            and lane_valid
            and speed > 0.0
            and bool(self.get_parameter("enable_camera_low_obstacle").value)
        )

    def publish_command(self, speed: float, steering: float):
        msg = Twist()
        max_angular = float(self.get_parameter("max_angular").value)
        msg.linear.x = float(speed)
        msg.angular.z = float(np.clip(steering, -max_angular, max_angular))
        self.cmd_pub.publish(msg)

    def publish_debug_image(self):
        if (
            self.bridge is None
            or not bool(self.get_parameter("publish_debug_image").value)
            or self.latest_debug_image is None
        ):
            return
        try:
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(self.latest_debug_image, encoding="bgr8")
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish debug image: {exc}")

    def show_debug_window(self, image):
        """OpenCV GUI 창에 차선 인식 결과를 표시합니다."""

        if image is None or not bool(self.get_parameter("display_debug_window").value):
            return

        window_name = str(self.get_parameter("debug_window_name").value)
        scale = float(self.get_parameter("debug_window_scale").value)

        try:
            if not self.debug_window_created:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                self.debug_window_created = True

            display_image = image
            if scale > 0.0 and abs(scale - 1.0) > 1e-3:
                display_image = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(window_name, display_image)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to show OpenCV debug window: {exc}",
                throttle_duration_sec=1.0,
            )

    def close_debug_window(self):
        if not self.debug_window_created:
            return
        try:
            cv2.destroyWindow(str(self.get_parameter("debug_window_name").value))
            cv2.waitKey(1)
        except Exception:
            pass

    def is_recent(self, stamp, timeout_param: str) -> bool:
        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        return age <= float(self.get_parameter(timeout_param).value)

    def publish_stop(self, reason: str):
        try:
            self.cmd_pub.publish(Twist())
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish stop command: {exc}")
        self.get_logger().warn(reason, throttle_duration_sec=1.0)

    def log_drive_status(
        self,
        lane: Optional[LaneResult],
        lane_valid: bool,
        drive_state: str,
        scan_ok: bool,
    ):
        if not bool(self.get_parameter("publish_drive_log").value):
            return

        lane_text = "인식" if lane_valid else "미인식"
        aeb_text = "작동" if drive_state.startswith("aeb") else "미작동"
        scan_text = "정상" if scan_ok else "끊김"
        self.get_logger().info(
            f"차선: {lane_text} | 라이다: {scan_text} | "
            f"AEB: {aeb_text} | 상태: {drive_state}",
            throttle_duration_sec=float(
                self.get_parameter("drive_log_period_sec").value
            ),
        )


def main(args=None):
    rclpy.init(args=args)
    node = LimoAutonomousDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C 시 마지막 정지 명령을 보내
        # LIMO가 이전 속도 명령을 물고 있지 않게 합니다.
        if rclpy.ok():
            try:
                node.cmd_pub.publish(Twist())
                rclpy.spin_once(node, timeout_sec=0.05)
            except Exception as exc:
                node.get_logger().warn(f"Failed to publish final stop: {exc}")
        node.close_debug_window()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
