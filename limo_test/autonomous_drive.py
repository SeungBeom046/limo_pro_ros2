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
        """ROS2 pub/sub, 알고리즘 객체, 제어 상태 변수를 초기화합니다.

        이 생성자는 실제 주행 판단을 하지 않습니다. 카메라/라이다 콜백은
        최신 센서값만 저장하고, 주행 명령 계산은 control_loop()에서
        일정 주기로만 수행합니다.
        """

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
        self.last_cmd_speed = 0.0
        self.last_cmd_steering = 0.0

        self.get_logger().info(
            "LIMO autonomous drive ready: "
            f"image={self.image_topic}, scan={self.scan_topic}, "
            f"cmd={self.cmd_vel_topic}"
        )

    def _declare_core_parameters(self):
        """통합 노드가 직접 사용하는 ROS 파라미터를 선언합니다.

        카메라 인식 전용 파라미터는 camera_lane_detector.py에서,
        라이다 회피 전용 파라미터는 lidar_obstacle_avoidance.py에서
        선언합니다. 여기에는 토픽, 속도, PID, fallback 정책처럼
        두 알고리즘을 합칠 때 필요한 값만 둡니다.
        """

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
        self.declare_parameter("kp", 2.05)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.22)
        self.declare_parameter("integral_limit", 0.45)
        self.declare_parameter("max_lane_error_step", 0.18)

        # 차선이 순간적으로 끊겼을 때
        # 마지막 정상 방향을 짧게 유지합니다.
        self.declare_parameter("lane_hold_time_sec", 1.20)
        self.declare_parameter("lane_hold_speed", 0.75)
        self.declare_parameter("lane_hold_min_confidence", 0.12)
        self.declare_parameter("lane_switch_reject_error", 0.45)

        # 차선이 사라지면 라이다 fallback으로 열린 방향을 찾습니다.
        self.declare_parameter("enable_lane_lost_drive", True)
        self.declare_parameter("lane_lost_steering_decay", 0.45)
        self.declare_parameter("lane_lost_use_lidar", True)
        self.declare_parameter("allow_lidar_drive_when_camera_invalid", True)
        self.declare_parameter("allow_lidar_drive_without_camera", True)
        self.declare_parameter("lane_priority_lidar_steering_limit", 0.28)
        self.declare_parameter("lane_priority_tunnel_speed", 0.35)

        # 센서가 오래되면 잘못된 명령을 내지 않도록 제한합니다.
        self.declare_parameter("image_timeout_sec", 0.7)
        self.declare_parameter("scan_timeout_sec", 0.7)
        self.declare_parameter("depth_timeout_sec", 0.7)
        self.declare_parameter("low_obstacle_speed", 0.10)
        self.declare_parameter("camera_obstacle_avoid_gain", 0.45)

    def image_qos_profiles(self, parameter_name: str):
        """카메라 구독에 사용할 QoS 후보를 반환합니다.

        LIMO 카메라 드라이버마다 RELIABLE/BEST_EFFORT 설정이 다를 수 있어
        기본 auto 모드에서는 두 QoS subscription을 모두 열어둡니다.
        """

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
        """카메라 Image 메시지를 OpenCV 프레임으로 바꿉니다.

        변환한 프레임으로 차선 인식을 수행합니다.

        반환된 LaneResult와 debug image만 저장합니다. 실제 속도/조향 계산은
        control_loop()에서 라이다 상태와 함께 통합합니다.
        """

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
        """최신 LaserScan을 저장합니다.

        라이다 콜백에서는 무거운 판단을 하지 않고, control_loop()에서
        최신 scan을 이용해 AEB/회피/fallback 판단을 합니다.
        """

        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

    def control_loop(self):
        """최종 /cmd_vel을 만드는 메인 제어 루프입니다.

        흐름은 다음 순서입니다.
        1. 센서 timeout 확인
        2. AEB 복구 동작이 진행 중이면 그 명령을 최우선 적용
        3. 카메라 차선 기반 기본 속도/조향 계산
        4. 라이다 안전 판단을 섞되, 차선 인식 중에는 차선을 우선 유지
        5. 카메라 하단 낮은 장애물 보조 회피 적용
        6. Twist를 publish하고 테스트 로그/debug image 출력
        """

        now = self.get_clock().now()
        image_ok = self.is_recent(self.last_image_time, "image_timeout_sec")
        scan_ok = self.is_recent(self.last_scan_time, "scan_timeout_sec")

        if not image_ok:
            if (
                bool(self.get_parameter("allow_lidar_drive_without_camera").value)
                and scan_ok
                and self.latest_scan is not None
            ):
                speed, steering = self.lidar.lane_lost_command(self.latest_scan)
                speed, steering, drive_state = self.apply_lidar_avoidance(
                    speed,
                    steering,
                    False,
                    True,
                    "no_camera_lidar",
                    now,
                )
                self.publish_command(speed, steering)
                self.log_drive_status(None, False, drive_state, scan_ok)
                return
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
            # 라이다가 고깔 상부만 봐서 늦게 반응하는 경우를 보완합니다.
            # 카메라 하단 중앙에 도로가 아닌 물체가 보이면
            # 그 반대 방향으로 약하게 조향을 더합니다.
            steering += (
                float(self.get_parameter("camera_obstacle_avoid_gain").value)
                * lane.camera_obstacle_error
            )
            if drive_state == "lane_follow":
                drive_state = "camera_low_obstacle"

        if (
            lane_lost_active
            and scan_ok
            and drive_state not in ("aeb_stop", "camera_invalid")
        ):
            speed = float(self.get_parameter("lane_lost_min_speed").value)

        self.publish_command(speed, steering)
        self.log_drive_status(lane, lane_valid, drive_state, scan_ok)
        self.publish_debug_image()

    def lane_follow_command(self, now, lane, dt: float, scan_ok: bool):
        """차선 인식 결과만 보고 기본 주행 명령을 만듭니다.

        반환값은 speed, steering, lane_valid, lane_lost_active, drive_state입니다.
        차선이 안정적으로 보이면 PID로 조향하고, 차선이 잠깐 끊기면
        lane_hold를 사용합니다. 차선이 계속 안 보이면 라이다 fallback으로
        넘길 수 있도록 lane_lost_active를 True로 반환합니다.
        """

        lane_valid = self.lane_is_valid(lane)
        if lane_valid and self.is_suspicious_lane_switch(now, lane):
            lane_valid = False
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
            if (
                bool(self.get_parameter("allow_lidar_drive_when_camera_invalid").value)
                and scan_ok
                and self.latest_scan is not None
            ):
                # 카메라 화면이 과노출/암전 등으로 차선 판단 불가여도
                # 라이다가 살아 있으면 정지 고착 대신
                # 열린 방향으로 저속 탐색합니다.
                speed, steering = self.lidar.lane_lost_command(self.latest_scan)
                self.get_logger().warn(
                    "Camera invalid. Using lidar fallback.",
                    throttle_duration_sec=1.0,
                )
                return speed, steering, False, True, "camera_invalid_lidar"
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
        """라이다 판단을 카메라 기반 속도/조향에 반영합니다.

        AEB는 항상 최우선입니다. 다만 차선이 정상 인식되는 동안에는
        slalom/gap/tunnel 회피가 차선을 벗어나게 만들지 않도록
        라이다 조향 보정량을 제한합니다. 차선을 잃었을 때만 라이다
        회피 조향이 더 크게 주행 방향을 결정합니다.
        """

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
                steering = self.limit_lidar_steering(steering, obstacle_steering)
                return speed, steering, "lane_priority_slow"
            else:
                speed = obstacle_speed
            return speed, obstacle_steering, mode

        if mode in ("passable_avoid", "slalom_gap") and lane_valid:
            speed = min(
                speed,
                max(
                    obstacle_speed,
                    float(self.get_parameter("lane_priority_tunnel_speed").value),
                ),
            )
            steering = self.limit_lidar_steering(steering, obstacle_steering)
            return speed, steering, "lane_priority_keep_lane"

        if mode == "tunnel_center" and lane_valid:
            speed = min(
                speed,
                float(self.get_parameter("lane_priority_tunnel_speed").value),
            )
            steering = self.limit_lidar_steering(steering, obstacle_steering)
            return speed, steering, "lane_priority_tunnel"

        if mode in ("slow_avoid", "side_avoid", "corner_guard", "passable_avoid"):
            if lane_valid:
                speed = min(
                    speed,
                    max(
                        obstacle_speed,
                        float(self.get_parameter("lane_priority_tunnel_speed").value),
                    ),
                )
                steering = self.limit_lidar_steering(steering, obstacle_steering)
                return speed, steering, "lane_priority_obstacle"
            else:
                speed = min(speed, obstacle_speed)
            return speed, steering + obstacle_steering, mode

        if mode == "tunnel_center":
            return speed, steering + obstacle_steering, mode

        if mode == "escape_bias":
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

    def is_suspicious_lane_switch(self, now, lane: LaneResult) -> bool:
        """갑자기 다른 차선을 잡은 것으로 보이면 True를 반환합니다.

        직전 정상 차선의 center_error와 현재 center_error 차이가 너무 크면
        차선 이탈 중 옆 차선을 따라가기 시작한 상황으로 보고,
        즉시 따라가지 않고 lane_hold가 직전 차선을 잠시 유지하게 합니다.
        """

        if self.last_lane_seen_time is None:
            return False

        hold_time = float(self.get_parameter("lane_hold_time_sec").value)
        age = (now - self.last_lane_seen_time).nanoseconds / 1e9
        if age > hold_time:
            return False

        reject_error = float(self.get_parameter("lane_switch_reject_error").value)
        if abs(lane.center_error - self.last_lane_error) <= reject_error:
            return False

        self.get_logger().warn(
            "차선 중심이 급격히 바뀌어 다른 차선으로 판단했습니다. "
            "직전 차선을 잠시 유지합니다.",
            throttle_duration_sec=0.5,
        )
        return True

    def limit_lidar_steering(self, lane_steering: float, lidar_steering: float) -> float:
        """차선 추종 중 라이다 조향 보정량을 제한합니다.

        터널 벽이나 통과 가능한 물체를 피하려다가
        차선 밖으로 나가는 일을 막기 위한 안전장치입니다.
        """

        limit = float(self.get_parameter("lane_priority_lidar_steering_limit").value)
        correction = float(np.clip(lidar_steering, -limit, limit))
        return lane_steering + correction

    def lane_is_valid(self, lane: Optional[LaneResult]) -> bool:
        """현재 LaneResult가 주행에 사용할 만큼 신뢰도 높은지 확인합니다."""

        if lane is None:
            return False
        return lane.confidence >= float(self.get_parameter("min_lane_confidence").value)

    def can_hold_lane(self, now, lane: Optional[LaneResult]) -> bool:
        """차선이 잠깐 끊긴 상황에서 직전 차선을 유지할지 판단합니다."""

        if self.last_lane_seen_time is None or lane is None or not lane.camera_valid:
            return False

        age = (now - self.last_lane_seen_time).nanoseconds / 1e9
        if age > float(self.get_parameter("lane_hold_time_sec").value):
            return False

        min_hold_conf = float(self.get_parameter("lane_hold_min_confidence").value)
        return lane.confidence >= min_hold_conf or lane.geometry_valid or lane.road_valid

    def stable_lane_error(self, error: float) -> float:
        """차선 중심 오차의 프레임 간 변화량을 제한합니다."""

        if self.last_lane_seen_time is None:
            return error
        max_step = float(self.get_parameter("max_lane_error_step").value)
        delta = float(np.clip(error - self.last_lane_error, -max_step, max_step))
        return float(np.clip(self.last_lane_error + delta, -1.0, 1.0))

    def pid_steering(self, error: float, dt: float) -> float:
        """차선 중심 오차를 angular.z 조향값으로 변환합니다."""

        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)
        limit = float(self.get_parameter("integral_limit").value)

        self.integral = float(np.clip(self.integral + error * dt, -limit, limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return kp * error + ki * self.integral + kd * derivative

    def smoothed_steering(self, raw_steering: float) -> float:
        """PID 조향값을 직전 조향과 섞어 급격한 흔들림을 줄입니다."""

        smoothing = float(self.get_parameter("steering_smoothing").value)
        steering = (1.0 - smoothing) * raw_steering + smoothing * self.last_steering
        self.last_steering = steering
        return steering

    def speed_from_lane(self, lane: LaneResult, steering: float) -> float:
        """차선 신뢰도와 조향량을 이용해 기본 전진 속도를 계산합니다.

        조향이 작고 차선 신뢰도가 높으면 빠르게 갑니다.
        조향이 크거나 신뢰도가 낮으면 느리게 갑니다.
        """

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
        """카메라 하단 중앙의 낮은 장애물 의심 영역을 확인합니다.

        고깔 하부처럼 라이다가 늦게 볼 수 있는 물체를 보조적으로
        감지합니다. 실제 감속/회피 적용은 control_loop()에서 수행합니다.
        """

        return (
            lane is not None
            and lane.road_valid
            and lane.camera_obstacle
            and lane_valid
            and speed > 0.0
            and bool(self.get_parameter("enable_camera_low_obstacle").value)
        )

    def publish_command(self, speed: float, steering: float):
        """계산된 속도/조향을 Twist 메시지로 publish합니다."""

        msg = Twist()
        max_angular = float(self.get_parameter("max_angular").value)
        msg.linear.x = float(speed)
        msg.angular.z = float(np.clip(steering, -max_angular, max_angular))
        self.last_cmd_speed = msg.linear.x
        self.last_cmd_steering = msg.angular.z
        self.cmd_pub.publish(msg)

    def publish_debug_image(self):
        """OpenCV debug image를 ROS Image 토픽으로 발행합니다."""

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
        """노드 종료 시 OpenCV debug 창을 정리합니다."""

        if not self.debug_window_created:
            return
        try:
            cv2.destroyWindow(str(self.get_parameter("debug_window_name").value))
            cv2.waitKey(1)
        except Exception:
            pass

    def is_recent(self, stamp, timeout_param: str) -> bool:
        """센서 timestamp가 timeout 이내인지 확인합니다."""

        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        return age <= float(self.get_parameter(timeout_param).value)

    def publish_stop(self, reason: str):
        """즉시 정지 Twist를 publish하고 정지 이유를 로그로 남깁니다."""

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
        """테스트 중 보기 쉬운 두 줄짜리 한글 주행 로그를 출력합니다.

        첫 줄은 차선 인식 유무, AEB 작동 여부, 현재 속도입니다.
        둘째 줄은 좌/우 차선 검출, 신뢰도, 중심 오차,
        카메라/도로 상태입니다.
        """

        if not bool(self.get_parameter("publish_drive_log").value):
            return

        lane_text = "인식" if lane_valid else "미인식"
        aeb_text = "작동" if drive_state.startswith("aeb") else "미작동"
        lane_status = self.format_lane_status(lane, lane_valid, drive_state)
        self.get_logger().info(
            f"\n차선: {lane_text} | AEB: {aeb_text} | "
            f"속도: {self.last_cmd_speed:.2f} m/s\n"
            f"차선 상태: {lane_status}",
            throttle_duration_sec=float(
                self.get_parameter("drive_log_period_sec").value
            ),
        )

    def format_lane_status(
        self,
        lane: Optional[LaneResult],
        lane_valid: bool,
        drive_state: str,
    ) -> str:
        """LaneResult를 사람이 읽기 쉬운 한글 상태 문자열로 변환합니다."""

        if lane is None:
            return f"카메라 프레임 없음, 현재 주행 상태는 {drive_state}"

        left_text = "보임" if lane.left_x is not None else "안 보임"
        right_text = "보임" if lane.right_x is not None else "안 보임"
        validity = "주행 기준 충족" if lane_valid else "신뢰도 부족"
        camera_text = "정상" if lane.camera_valid else "판단 불가"
        road_text = "확인" if lane.road_valid else "부족"
        obstacle_text = "감지" if lane.camera_obstacle else "없음"

        return (
            f"왼쪽 차선 {left_text}, 오른쪽 차선 {right_text}, "
            f"신뢰도 {lane.confidence:.2f}({validity}), "
            f"중심 오차 {lane.center_error:+.2f}, "
            f"카메라 {camera_text}, 도로영역 {road_text}, "
            f"하단 장애물 {obstacle_text}, 상태 {drive_state}"
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
