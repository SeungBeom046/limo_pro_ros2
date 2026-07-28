import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover - depends on the robot image.
    CvBridge = None


@dataclass
class LaneResult:
    # 카메라 한 프레임에서 추정한 차선 상태입니다.
    # center_error: -1.0~1.0 범위.
    # 양수면 차선 중심이 화면 왼쪽에 있어 우회전 보정.
    # confidence: 차선 검출 신뢰도. 낮으면 안전하게 정지합니다.
    center_error: float
    confidence: float
    left_x: Optional[float]
    right_x: Optional[float]


class LimoAutonomousDrive(Node):
    """Camera lane keeping with 2D lidar obstacle avoidance for AgileX LIMO."""

    def __init__(self):
        super().__init__("limo_autonomous_drive")

        # 토픽 이름은 LIMO 세팅마다 조금씩 다를 수 있어
        # 파라미터로 열어둡니다.
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("debug_image_topic", "/limo/autonomy/debug_image")
        self.declare_parameter("publish_debug_image", True)

        # 속도는 실차 첫 주행 기준으로 보수적으로 잡았습니다.
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed", 0.35)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("caution_speed", 0.12)
        self.declare_parameter("max_angular", 1.35)

        # 차선 중심 오차를 조향값으로 바꾸는 PID 계수입니다.
        # 흔들리면 kd를 올리고,
        # 반응이 과하면 kp를 낮추는 순서로 튜닝하세요.
        self.declare_parameter("kp", 1.45)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.18)
        self.declare_parameter("integral_limit", 0.45)

        # 카메라 하단부만 차선 ROI로 사용합니다.
        # 위쪽 배경/벽/사람 검출을 줄이기 위함입니다.
        self.declare_parameter("roi_top_ratio", 0.55)
        self.declare_parameter("expected_lane_width_ratio", 0.48)
        self.declare_parameter("single_lane_offset_ratio", 0.24)
        self.declare_parameter("min_lane_area", 180)

        # 라이다는 전방과 좌/우 측면 섹터로 나누어
        # 가장 가까운 장애물을 봅니다.
        self.declare_parameter("front_sector_deg", 28.0)
        self.declare_parameter("side_sector_deg", 55.0)
        self.declare_parameter("stop_distance", 0.42)
        self.declare_parameter("slow_distance", 0.85)
        self.declare_parameter("avoid_gain", 0.85)

        # 센서 데이터가 오래되면 잘못된 명령을 내리지 않도록 제한합니다.
        self.declare_parameter("image_timeout_sec", 0.7)
        self.declare_parameter("scan_timeout_sec", 0.7)

        self.bridge = CvBridge() if CvBridge else None
        if self.bridge is None:
            self.get_logger().warn(
                "cv_bridge is not available. Install ros-humble-cv-bridge."
            )

        self.image_topic = self.get_parameter("image_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 2)

        # 카메라와 라이다 콜백은 최신 센서 상태만 저장하고,
        # 실제 주행 명령 계산은 control_loop에서 주기적으로 수행합니다.
        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
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

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = self.get_clock().now()

        self.get_logger().info(
            "LIMO autonomous drive ready: "
            f"image={self.image_topic}, scan={self.scan_topic}, "
            f"cmd={self.cmd_vel_topic}"
        )

    def image_callback(self, msg: Image):
        if self.bridge is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert image: {exc}")
            return

        self.latest_lane, self.latest_debug_image = self.detect_lane(frame)
        self.last_image_time = self.get_clock().now()

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

    def detect_lane(self, frame) -> Tuple[LaneResult, np.ndarray]:
        height, width = frame.shape[:2]
        roi_top = int(height * float(self.get_parameter("roi_top_ratio").value))
        roi = frame[roi_top:, :]

        # HSV 색공간에서 흰색/노란색 차선 후보를 분리합니다.
        # 조명에 따라 이 범위는 현장에서 가장 많이 튜닝하게 됩니다.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 65, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 55, 70]), np.array([40, 255, 255]))
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 작은 노이즈는 제거하고 끊긴 차선 조각은 어느 정도 이어줍니다.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        lane_result = self.mask_to_lane_result(mask, width)
        debug = self.draw_debug(frame, mask, roi_top, lane_result)
        return lane_result, debug

    def mask_to_lane_result(self, mask, width: int) -> LaneResult:
        min_area = int(self.get_parameter("min_lane_area").value)
        expected_lane_width = width * float(
            self.get_parameter("expected_lane_width_ratio").value
        )
        single_lane_offset = width * float(
            self.get_parameter("single_lane_offset_ratio").value
        )

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        left_candidates = []
        right_candidates = []
        image_center = width / 2.0

        # 연결 성분 단위로 차선 후보를 찾습니다.
        # 화면 아래쪽에 가까운 후보일수록
        # 실제 주행 경로에 더 중요하므로 점수를 더 줍니다.
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x, y = centroids[label]
            score = area * (1.0 + y / max(mask.shape[0], 1))
            if x < image_center:
                left_candidates.append((score, x))
            else:
                right_candidates.append((score, x))

        left_x = max(left_candidates)[1] if left_candidates else None
        right_x = max(right_candidates)[1] if right_candidates else None

        confidence = 0.0
        if left_x is not None and right_x is not None:
            # 양쪽 차선이 보이면 두 차선의 중간을 주행 중심으로 봅니다.
            lane_width = max(right_x - left_x, 1.0)
            lane_center = (left_x + right_x) / 2.0
            width_score = 1.0 - min(
                abs(lane_width - expected_lane_width) / expected_lane_width,
                1.0,
            )
            confidence = 0.55 + 0.45 * width_score
        elif left_x is not None:
            # 한쪽 차선만 보이면 예상 차선 폭으로 반대쪽을 가정합니다.
            lane_center = left_x + single_lane_offset
            confidence = 0.45
        elif right_x is not None:
            lane_center = right_x - single_lane_offset
            confidence = 0.45
        else:
            lane_center = image_center

        center_error = (image_center - lane_center) / image_center
        center_error = float(np.clip(center_error, -1.0, 1.0))
        return LaneResult(center_error, confidence, left_x, right_x)

    def draw_debug(self, frame, mask, roi_top: int, lane: LaneResult):
        debug = frame.copy()
        height, width = frame.shape[:2]
        image_center = width // 2
        lane_center = int(image_center - lane.center_error * image_center)

        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug[roi_top:, :] = cv2.addWeighted(
            debug[roi_top:, :],
            0.65,
            colored_mask,
            0.35,
            0,
        )

        cv2.line(debug, (image_center, roi_top), (image_center, height), (255, 0, 0), 2)
        cv2.line(debug, (lane_center, roi_top), (lane_center, height), (0, 255, 0), 2)

        if lane.left_x is not None:
            cv2.circle(debug, (int(lane.left_x), roi_top + 20), 8, (0, 255, 255), -1)
        if lane.right_x is not None:
            cv2.circle(debug, (int(lane.right_x), roi_top + 20), 8, (0, 255, 255), -1)

        cv2.putText(
            debug,
            f"err={lane.center_error:+.2f} conf={lane.confidence:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return debug

    def control_loop(self):
        now = self.get_clock().now()
        image_ok = self.is_recent(self.last_image_time, "image_timeout_sec")
        scan_ok = self.is_recent(self.last_scan_time, "scan_timeout_sec")

        msg = Twist()
        # 카메라가 없으면 차선 기반 주행을 할 수 없으므로
        # 즉시 정지합니다.
        if not image_ok:
            self.publish_stop("Waiting for camera image")
            return

        lane = self.latest_lane
        if lane is None or lane.confidence <= 0.01:
            self.publish_stop("Lane not detected")
            return

        dt = max((now - self.prev_time).nanoseconds / 1e9, 1e-3)
        self.prev_time = now

        steering = self.pid_steering(lane.center_error, dt)
        speed = self.speed_from_lane(lane, steering)

        # 장애물 회피는 차선 추종보다 우선순위가 높습니다.
        # 가까운 장애물은 정지+회전,
        # 조금 먼 장애물은 감속+회피 조향으로 처리합니다.
        if scan_ok and self.latest_scan is not None:
            obstacle_speed, obstacle_steering, mode = self.obstacle_command(
                self.latest_scan
            )
            if mode == "stop_turn":
                speed = obstacle_speed
                steering = obstacle_steering
            elif mode == "slow_avoid":
                speed = min(speed, obstacle_speed)
                steering += obstacle_steering
        else:
            # 라이다가 잠시 끊겼을 때 완전 정지 대신
            # 저속 제한을 걸어 회복 여지를 둡니다.
            speed = min(speed, float(self.get_parameter("caution_speed").value))

        max_angular = float(self.get_parameter("max_angular").value)
        msg.linear.x = float(max(speed, 0.0))
        msg.angular.z = float(np.clip(steering, -max_angular, max_angular))
        self.cmd_pub.publish(msg)

        if (
            self.bridge is not None
            and bool(self.get_parameter("publish_debug_image").value)
            and self.latest_debug_image is not None
        ):
            try:
                self.debug_pub.publish(
                    self.bridge.cv2_to_imgmsg(self.latest_debug_image, encoding="bgr8")
                )
            except Exception as exc:
                self.get_logger().warn(f"Failed to publish debug image: {exc}")

    def pid_steering(self, error: float, dt: float) -> float:
        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)
        limit = float(self.get_parameter("integral_limit").value)

        self.integral = float(np.clip(self.integral + error * dt, -limit, limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return kp * error + ki * self.integral + kd * derivative

    def speed_from_lane(self, lane: LaneResult, steering: float) -> float:
        max_speed = float(self.get_parameter("max_speed").value)
        min_speed = float(self.get_parameter("min_speed").value)
        turn_factor = 1.0 - min(abs(steering) / 1.4, 0.75)
        confidence_factor = 0.45 + 0.55 * lane.confidence
        return max(min_speed, max_speed * turn_factor * confidence_factor)

    def obstacle_command(self, scan: LaserScan) -> Tuple[float, float, str]:
        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")
        # LIMO 라이다 좌표계 기준:
        # 정면 0rad, 왼쪽 양수, 오른쪽 음수로 가정합니다.
        front = self.sector_min(scan, -front_angle, front_angle)
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)

        stop_distance = float(self.get_parameter("stop_distance").value)
        slow_distance = float(self.get_parameter("slow_distance").value)
        avoid_gain = float(self.get_parameter("avoid_gain").value)
        caution_speed = float(self.get_parameter("caution_speed").value)

        if front < stop_distance:
            # 너무 가까우면 전진하지 않고 더 넓은 쪽으로 회전합니다.
            turn_direction = 1.0 if left > right else -1.0
            return 0.0, turn_direction * min(1.0, avoid_gain + 0.25), "stop_turn"

        if front < slow_distance:
            # 전방 여유가 작으면 좌우 공간 차이만큼 회피 조향을 더합니다.
            clearance_delta = np.clip((left - right) / slow_distance, -1.0, 1.0)
            return caution_speed, avoid_gain * clearance_delta, "slow_avoid"

        return 0.0, 0.0, "clear"

    def sector_min(
        self,
        scan: LaserScan,
        start_angle: float,
        end_angle: float,
    ) -> float:
        ranges = []
        for i, value in enumerate(scan.ranges):
            # inf/nan 또는 라이다 유효 범위 밖의 값은 판단에서 제외합니다.
            if not math.isfinite(value):
                continue
            if value < scan.range_min or value > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if start_angle <= angle <= end_angle:
                ranges.append(value)

        if not ranges:
            return float("inf")
        # 최소값 하나만 쓰면 튀는 노이즈에 민감하므로
        # 15퍼센타일 값을 사용합니다.
        return float(np.percentile(ranges, 15))

    def param_rad(self, name: str) -> float:
        return math.radians(float(self.get_parameter(name).value))

    def is_recent(self, stamp, timeout_param: str) -> bool:
        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        return age <= float(self.get_parameter(timeout_param).value)

    def publish_stop(self, reason: str):
        self.cmd_pub.publish(Twist())
        self.get_logger().warn(reason, throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = LimoAutonomousDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
