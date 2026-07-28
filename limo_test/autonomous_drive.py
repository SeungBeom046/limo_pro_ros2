import math
from dataclasses import dataclass
from typing import Optional, Tuple

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

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover - depends on the robot image.
    CvBridge = None


@dataclass
class LaneResult:
    # 카메라 한 프레임에서 추정한 차선 상태입니다.
    # center_error: -1.0~1.0 범위.
    # 양수면 차선 중심이 화면 왼쪽에 있어 우회전 보정.
    # confidence: 차선 검출 신뢰도.
    # 낮으면 라이다 기반 저속 탐색으로 전환합니다.
    center_error: float
    confidence: float
    left_x: Optional[float]
    right_x: Optional[float]
    mask_ratio: float
    left_count: int
    right_count: int
    rejected_large_count: int
    road_ratio: float
    camera_obstacle_ratio: float
    camera_obstacle: bool
    road_valid: bool
    geometry_valid: bool
    road_between_ratio: float
    lane_width_ratio: float
    camera_valid: bool


@dataclass
class LaneTrack:
    x: Optional[float]
    count: int
    std_ratio: float
    width_ratio: float
    valid: bool
    points: list


@dataclass
class LanePairTrack:
    left: LaneTrack
    right: LaneTrack
    valid: bool


class LimoAutonomousDrive(Node):
    """Camera lane keeping with 2D lidar obstacle avoidance for AgileX LIMO."""

    def __init__(self):
        super().__init__("limo_autonomous_drive")

        # 토픽 이름은 LIMO 세팅마다 조금씩 다를 수 있어
        # 파라미터로 열어둡니다.
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("debug_image_topic", "/limo/autonomy/debug_image")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_drive_log", True)
        self.declare_parameter("drive_log_period_sec", 0.5)
        self.declare_parameter("publish_lane_log", False)
        self.declare_parameter("lane_log_period_sec", 0.5)
        self.declare_parameter("image_qos", "auto")
        self.declare_parameter("depth_qos", "auto")

        # 속도는 실차 주행 기준으로 너무 답답하지 않게 잡되,
        # 장애물이나 차선 신뢰도에 따라 아래에서 자동으로 줄입니다.
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed", 1.40)
        self.declare_parameter("min_speed", 0.18)
        self.declare_parameter("lane_follow_min_speed", 0.55)
        self.declare_parameter("caution_speed", 0.35)
        self.declare_parameter("straight_min_speed", 1.40)
        self.declare_parameter("straight_steering_threshold", 0.25)
        self.declare_parameter("max_angular", 1.35)
        self.declare_parameter("steering_smoothing", 0.35)

        # 차선 중심 오차를 조향값으로 바꾸는 PID 계수입니다.
        # 흔들리면 kd를 올리고,
        # 반응이 과하면 kp를 낮추는 순서로 튜닝하세요.
        self.declare_parameter("kp", 1.45)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.18)
        self.declare_parameter("integral_limit", 0.45)

        # 중간 시야부터 아래까지 보면서
        # 원근에 따른 차선 폭 변화를 반영합니다.
        self.declare_parameter("roi_top_ratio", 0.38)
        self.declare_parameter("expected_lane_width_ratio", 0.48)
        self.declare_parameter("top_lane_width_ratio", 0.22)
        self.declare_parameter("bottom_lane_width_ratio", 0.62)
        self.declare_parameter("lookahead_ratio", 0.62)
        self.declare_parameter("single_lane_offset_ratio", 0.24)
        self.declare_parameter("min_lane_area", 180)
        self.declare_parameter("max_lane_area_ratio", 0.18)
        self.declare_parameter("min_lane_confidence", 0.30)
        self.declare_parameter("white_lane_value_min", 105)
        self.declare_parameter("white_lane_saturation_max", 170)
        self.declare_parameter("white_lane_relative_margin", 28)
        self.declare_parameter("lane_search_band_count", 9)
        self.declare_parameter("lane_search_band_height", 20)
        self.declare_parameter("lane_min_band_pixels", 8)
        self.declare_parameter("min_lane_band_count", 3)
        self.declare_parameter("lane_histogram_margin_ratio", 0.08)
        self.declare_parameter("max_lane_peak_width_ratio", 0.09)
        self.declare_parameter("max_lane_x_std_ratio", 0.14)
        self.declare_parameter("max_lane_band_jump_ratio", 0.16)
        self.declare_parameter("lane_pair_width_tolerance_ratio", 0.24)
        self.declare_parameter("lane_pair_center_prior_ratio", 0.28)
        self.declare_parameter("min_lane_width_ratio", 0.18)
        self.declare_parameter("max_lane_width_ratio", 0.85)
        self.declare_parameter("max_lane_mask_ratio", 0.24)
        self.declare_parameter("blank_white_mask_ratio", 0.65)
        self.declare_parameter("blank_black_road_ratio", 1.10)
        self.declare_parameter("blank_black_mask_ratio", 0.0)
        self.declare_parameter("black_road_value_max", 120)
        self.declare_parameter("black_road_saturation_min", 0)
        self.declare_parameter("min_road_ratio_for_lane", 0.08)
        self.declare_parameter("reflected_road_ratio_for_lane", 0.05)
        self.declare_parameter("min_road_between_ratio", 0.08)
        self.declare_parameter("reflected_road_between_ratio", 0.03)
        self.declare_parameter("camera_obstacle_roi_top_ratio", 0.72)
        self.declare_parameter("camera_obstacle_center_width_ratio", 0.46)
        self.declare_parameter("camera_obstacle_min_ratio", 0.06)
        self.declare_parameter("enable_camera_low_obstacle", False)

        # 차선이 안 잡혀도 실내 바닥에서
        # 바로 멈추지 않도록 하는 탐색 모드입니다.
        # 라이다가 안전하면 저속으로 전진하고,
        # 가까운 장애물은 기존 회피 로직을 씁니다.
        self.declare_parameter("enable_lane_lost_drive", True)
        self.declare_parameter("lane_lost_speed", 0.50)
        self.declare_parameter("lane_lost_min_speed", 0.50)
        self.declare_parameter("lane_lost_steering_decay", 0.45)
        self.declare_parameter("lane_lost_use_lidar", True)
        self.declare_parameter("lane_lost_lidar_angle_deg", 95.0)
        self.declare_parameter("lane_lost_clearance_distance", 1.20)
        self.declare_parameter("lane_lost_open_distance", 2.50)
        self.declare_parameter("lane_lost_gap_gain", 0.85)
        self.declare_parameter("lane_lost_obstacle_gain", 0.75)

        # 라이다는 전방과 좌/우 측면 섹터로 나누어
        # 가장 가까운 장애물을 봅니다.
        self.declare_parameter("aeb_sector_deg", 45.0)
        self.declare_parameter("front_sector_deg", 45.0)
        self.declare_parameter("side_sector_deg", 100.0)
        self.declare_parameter("closest_sample_count", 1)
        self.declare_parameter("aeb_distance", 0.35)
        self.declare_parameter("enable_aeb_recovery", True)
        self.declare_parameter("aeb_recovery_clear_distance", 0.45)
        self.declare_parameter("aeb_recovery_reverse_sec", 0.7)
        self.declare_parameter("aeb_recovery_turn_sec", 0.6)
        self.declare_parameter("aeb_recovery_reverse_speed", -0.18)
        self.declare_parameter("aeb_recovery_turn_speed", 0.05)
        self.declare_parameter("aeb_recovery_turn_angular", 0.85)
        self.declare_parameter("stop_distance", 0.50)
        self.declare_parameter("slow_distance", 1.20)
        self.declare_parameter("side_obstacle_distance", 0.34)
        self.declare_parameter("tunnel_side_distance", 0.70)
        self.declare_parameter("tunnel_balance_tolerance", 0.28)
        self.declare_parameter("tunnel_centering_gain", 0.35)
        self.declare_parameter("low_obstacle_speed", 0.10)
        self.declare_parameter("avoid_gain", 0.85)
        self.declare_parameter("enable_lidar_slalom", True)
        self.declare_parameter("slalom_view_deg", 90.0)
        self.declare_parameter("slalom_obstacle_distance", 0.75)
        self.declare_parameter("slalom_min_gap_indices", 5)
        self.declare_parameter("slalom_kp", 1.0)
        self.declare_parameter("slalom_speed", 0.50)
        self.declare_parameter("slalom_lane_mix", 0.35)
        self.declare_parameter("enable_depth_obstacle", False)
        self.declare_parameter("depth_timeout_sec", 0.7)
        self.declare_parameter("depth_near_distance", 0.45)
        self.declare_parameter("depth_slow_distance", 0.75)
        self.declare_parameter("depth_stop_ratio", 0.12)
        self.declare_parameter("depth_slow_ratio", 0.06)
        self.declare_parameter("depth_roi_top_ratio", 0.58)
        self.declare_parameter("depth_roi_bottom_ratio", 0.95)
        self.declare_parameter("depth_center_width_ratio", 0.46)
        self.declare_parameter("depth_side_width_ratio", 0.27)
        self.declare_parameter("depth_avoid_gain", 0.65)

        # 센서 데이터가 오래되면 잘못된 명령을 내리지 않도록 제한합니다.
        self.declare_parameter("image_timeout_sec", 0.7)
        self.declare_parameter("scan_timeout_sec", 0.7)

        self.bridge = CvBridge() if CvBridge else None
        if self.bridge is None:
            self.get_logger().warn(
                "cv_bridge is not available. Install ros-humble-cv-bridge."
            )

        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 2)

        # 카메라와 라이다 콜백은 최신 센서 상태만 저장하고,
        # 실제 주행 명령 계산은 control_loop에서 주기적으로 수행합니다.
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

        # depth 카메라는 연산량이 크고 현재 트랙에서는 오탐으로 속도를 죽일 수 있어
        # 기본 비활성화합니다. enable_depth_obstacle을 true로 바꿀 때만 구독합니다.
        self.depth_subs = []
        if bool(self.get_parameter("enable_depth_obstacle").value):
            for qos_profile in self.image_qos_profiles("depth_qos"):
                self.depth_subs.append(
                    self.create_subscription(
                        Image,
                        self.depth_topic,
                        self.depth_callback,
                        qos_profile,
                    )
                )

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
        self.latest_depth_obstacle = (False, False, 0.0, 0.0)
        self.latest_debug_image = None
        self.last_obstacle_distances = (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
        self.last_image_time = None
        self.last_depth_time = None
        self.last_scan_time = None

        self.prev_error = 0.0
        self.last_steering = 0.0
        self.integral = 0.0
        self.prev_time = self.get_clock().now()
        self.aeb_recovery_active = False
        self.aeb_recovery_start_sec = 0.0
        self.aeb_recovery_turn_direction = 1.0

        self.get_logger().info(
            "LIMO autonomous drive ready: "
            f"image={self.image_topic}, depth={self.depth_topic}, "
            f"scan={self.scan_topic}, "
            f"cmd={self.cmd_vel_topic}"
        )

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

        # auto: 카메라 드라이버마다 RELIABLE/BEST_EFFORT가 달라서 둘 다 엽니다.
        return [reliable, best_effort]

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

    def depth_callback(self, msg: Image):
        if self.bridge is None:
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        self.latest_depth_obstacle = self.detect_depth_obstacle(depth)
        self.last_depth_time = self.get_clock().now()

    def detect_lane(self, frame) -> Tuple[LaneResult, np.ndarray]:
        height, width = frame.shape[:2]
        roi_top = int(height * float(self.get_parameter("roi_top_ratio").value))
        roi = frame[roi_top:, :]

        # 트랙은 검은 도로 + 흰색 차선이므로
        # 흰색 라인과 검은 도로를 분리합니다.
        # 조명에 따라 흰색/검은색 임계값은 현장에서 튜닝하게 됩니다.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        white_value_min = int(self.get_parameter("white_lane_value_min").value)
        white_saturation_max = int(
            self.get_parameter("white_lane_saturation_max").value
        )
        relative_margin = int(
            self.get_parameter("white_lane_relative_margin").value
        )
        low_saturation_mask = cv2.inRange(
            hsv[:, :, 1],
            0,
            white_saturation_max,
        )
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, white_value_min]),
            np.array([180, white_saturation_max, 255]),
        )

        # 트랙 위 실제 카메라는 자동 노출 때문에 흰 차선도 어둡게 보일 수 있습니다.
        # 그래서 고정 밝기 기준 외에 ROI 평균보다 확실히 밝은 저채도 픽셀도
        # 차선 후보로 함께 사용합니다.
        relative_threshold = int(
            np.clip(np.mean(gray) + relative_margin, white_value_min, 245)
        )
        relative_white = cv2.inRange(gray, relative_threshold, 255)
        relative_white = cv2.bitwise_and(relative_white, low_saturation_mask)
        adaptive_white = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            -8,
        )
        adaptive_white = cv2.bitwise_and(adaptive_white, low_saturation_mask)
        mask = cv2.bitwise_or(white_mask, relative_white)
        mask = cv2.bitwise_or(mask, adaptive_white)

        # 작은 노이즈는 제거하고 끊긴 차선 조각은 어느 정도 이어줍니다.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        road_mask = self.detect_black_road_mask(hsv)
        camera_obstacle_ratio, camera_obstacle = self.detect_camera_obstacle(
            roi,
            road_mask,
            mask,
        )

        lane_result = self.mask_to_lane_result(
            mask,
            width,
            road_mask,
            camera_obstacle_ratio,
            camera_obstacle,
        )
        self.log_lane_status(lane_result, width, height, roi_top)
        debug = self.draw_debug(frame, mask, roi_top, lane_result)
        return lane_result, debug

    def detect_black_road_mask(self, hsv):
        value_max = int(self.get_parameter("black_road_value_max").value)
        saturation_min = int(self.get_parameter("black_road_saturation_min").value)
        return cv2.inRange(
            hsv,
            np.array([0, saturation_min, 0]),
            np.array([180, 255, value_max]),
        )

    def detect_camera_obstacle(self, roi, road_mask, lane_mask) -> Tuple[float, bool]:
        height, width = roi.shape[:2]
        top = int(
            height
            * float(self.get_parameter("camera_obstacle_roi_top_ratio").value)
        )
        center_width = int(
            width
            * float(
                self.get_parameter("camera_obstacle_center_width_ratio").value
            )
        )
        left = max(0, (width - center_width) // 2)
        right = min(width, left + center_width)

        near_roi = roi[top:, left:right]
        near_road = road_mask[top:, left:right]
        near_lane = lane_mask[top:, left:right]
        if near_roi.size == 0:
            return 0.0, False

        gray = cv2.cvtColor(near_roi, cv2.COLOR_BGR2GRAY)
        non_road = cv2.bitwise_not(near_road)
        # 중앙 주행 영역 안에서 차선 마스크는 빼고,
        # 낮은 턱/물체처럼 검은 도로가 아닌 덩어리만 봅니다.
        obstacle_mask = cv2.bitwise_and(non_road, cv2.bitwise_not(near_lane))
        obstacle_mask = cv2.bitwise_and(
            obstacle_mask,
            cv2.inRange(gray, 55, 255),
        )
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        ratio = float(cv2.countNonZero(obstacle_mask)) / float(obstacle_mask.size)
        threshold = float(self.get_parameter("camera_obstacle_min_ratio").value)
        return ratio, ratio >= threshold

    def detect_depth_obstacle(self, depth) -> Tuple[bool, bool, float, float]:
        depth_m = self.depth_to_meters(depth)
        if depth_m is None:
            return False, False, 0.0, 0.0

        height, width = depth_m.shape[:2]
        top = int(height * float(self.get_parameter("depth_roi_top_ratio").value))
        bottom = int(
            height * float(self.get_parameter("depth_roi_bottom_ratio").value)
        )
        center_width = int(
            width * float(self.get_parameter("depth_center_width_ratio").value)
        )
        side_width = int(
            width * float(self.get_parameter("depth_side_width_ratio").value)
        )
        center_left = max(0, (width - center_width) // 2)
        center_right = min(width, center_left + center_width)

        center_roi = depth_m[top:bottom, center_left:center_right]
        left_roi = depth_m[top:bottom, 0:side_width]
        right_roi = depth_m[top:bottom, width - side_width:width]

        near_distance = float(self.get_parameter("depth_near_distance").value)
        slow_distance = float(self.get_parameter("depth_slow_distance").value)
        stop_ratio_threshold = float(self.get_parameter("depth_stop_ratio").value)
        slow_ratio_threshold = float(self.get_parameter("depth_slow_ratio").value)

        stop_ratio = self.depth_close_ratio(center_roi, near_distance)
        slow_ratio = self.depth_close_ratio(center_roi, slow_distance)
        left_clearance = self.depth_clearance(left_roi)
        right_clearance = self.depth_clearance(right_roi)
        clearance_delta = np.clip(
            (left_clearance - right_clearance) / max(slow_distance, 1e-3),
            -1.0,
            1.0,
        )

        stop = stop_ratio >= stop_ratio_threshold
        slow = slow_ratio >= slow_ratio_threshold
        return stop, slow, float(clearance_delta), float(slow_ratio)

    def depth_to_meters(self, depth):
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        if np.issubdtype(depth.dtype, np.integer):
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.05] = 0.0
        return depth_m

    def depth_close_ratio(self, roi, threshold: float) -> float:
        if roi.size == 0:
            return 0.0
        valid = roi > 0.05
        if not np.any(valid):
            return 0.0
        close = np.logical_and(valid, roi < threshold)
        return float(np.count_nonzero(close)) / float(np.count_nonzero(valid))

    def depth_clearance(self, roi) -> float:
        if roi.size == 0:
            return 0.0
        valid = roi[roi > 0.05]
        if valid.size == 0:
            return 0.0
        return float(np.percentile(valid, 20))

    def mask_to_lane_result(
        self,
        mask,
        width: int,
        road_mask,
        camera_obstacle_ratio: float,
        camera_obstacle: bool,
    ) -> LaneResult:
        min_area = int(self.get_parameter("min_lane_area").value)
        max_area = int(
            mask.shape[0]
            * mask.shape[1]
            * float(self.get_parameter("max_lane_area_ratio").value)
        )
        image_center = width / 2.0
        pair_track = self.find_lane_pair_tracks(mask, road_mask, width)
        if pair_track.valid:
            left_track = pair_track.left
            right_track = pair_track.right
        else:
            left_track = self.find_lane_x_from_bands(mask, 0, int(image_center))
            right_track = self.find_lane_x_from_bands(
                mask,
                int(image_center),
                width,
            )
        left_x = left_track.x if left_track.valid else None
        right_x = right_track.x if right_track.valid else None
        left_count = left_track.count if left_track.valid else 0
        right_count = right_track.count if right_track.valid else 0
        rejected_large_count = 0
        mask_ratio = float(cv2.countNonZero(mask)) / float(mask.size)
        road_ratio = float(cv2.countNonZero(road_mask)) / float(road_mask.size)
        min_road_ratio = float(self.get_parameter("min_road_ratio_for_lane").value)
        reflected_road_ratio = float(
            self.get_parameter("reflected_road_ratio_for_lane").value
        )
        min_road_between_ratio = float(
            self.get_parameter("min_road_between_ratio").value
        )
        reflected_road_between_ratio = float(
            self.get_parameter("reflected_road_between_ratio").value
        )
        min_width_ratio = float(self.get_parameter("min_lane_width_ratio").value)
        max_width_ratio = float(self.get_parameter("max_lane_width_ratio").value)
        max_mask_ratio = float(self.get_parameter("max_lane_mask_ratio").value)
        blank_white_ratio = float(self.get_parameter("blank_white_mask_ratio").value)
        blank_black_road_ratio = float(
            self.get_parameter("blank_black_road_ratio").value
        )
        blank_black_mask_ratio = float(
            self.get_parameter("blank_black_mask_ratio").value
        )
        white_invalid = (
            mask_ratio >= blank_white_ratio
            and road_ratio < min_road_ratio
        )
        black_invalid = (
            blank_black_road_ratio <= 1.0
            and
            road_ratio >= blank_black_road_ratio
            and mask_ratio <= blank_black_mask_ratio
        )
        camera_valid = not (white_invalid or black_invalid)

        # 너무 큰 흰색 배경이 들어오는지
        # 로그로 확인하기 위한 보조 카운트입니다.
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            if area > max_area:
                rejected_large_count += 1

        confidence = 0.0
        geometry_valid = False
        road_between_ratio = 0.0
        lane_width_ratio = 0.0
        if left_x is not None and right_x is not None:
            # 양쪽 차선이 보이면 두 차선의 중간을 주행 중심으로 봅니다.
            road_between_ratio = self.road_between_lanes_ratio(
                road_mask,
                left_x,
                right_x,
            )
            lane_center, lookahead_width_ratio = self.perspective_lane_center(
                left_track,
                right_track,
                image_center,
                width,
            )
            lane_width_ratio = lookahead_width_ratio
            width_valid = min_width_ratio <= lane_width_ratio <= max_width_ratio
            mask_valid = mask_ratio <= max_mask_ratio
            geometry_valid = width_valid and mask_valid
            expected_lane_width = width * self.expected_lane_width_ratio_at(
                float(self.get_parameter("lookahead_ratio").value)
            )
            lane_width = lane_width_ratio * width
            width_score = 1.0 - min(
                abs(lane_width - expected_lane_width) / max(expected_lane_width, 1.0),
                1.0,
            )
            count_score = min((left_count + right_count) / 8.0, 1.0)
            road_score = min(road_ratio / 0.35, 1.0)
            between_score = min(road_between_ratio / min_road_between_ratio, 1.0)
            if geometry_valid:
                confidence = (
                    0.45
                    + 0.35 * width_score
                    + 0.10 * count_score
                    + 0.05 * road_score
                    + 0.05 * between_score
                )
        elif left_x is not None:
            # 한쪽 차선만 보이면 예상 차선 폭으로 반대쪽을 가정합니다.
            expected_width = width * self.expected_lane_width_ratio_at(
                float(self.get_parameter("lookahead_ratio").value)
            )
            lane_center = left_x + expected_width / 2.0
            lane_width_ratio = expected_width / width
            geometry_valid = mask_ratio <= max_mask_ratio
            confidence = 0.38 if geometry_valid else 0.0
        elif right_x is not None:
            expected_width = width * self.expected_lane_width_ratio_at(
                float(self.get_parameter("lookahead_ratio").value)
            )
            lane_center = right_x - expected_width / 2.0
            lane_width_ratio = expected_width / width
            geometry_valid = mask_ratio <= max_mask_ratio
            confidence = 0.38 if geometry_valid else 0.0
        else:
            lane_center = image_center

        road_valid = road_ratio >= min_road_ratio or (
            geometry_valid
            and road_ratio >= reflected_road_ratio
            and road_between_ratio >= reflected_road_between_ratio
        )
        if not camera_valid:
            confidence = 0.0
        elif not road_valid:
            # 검은 도로가 충분히 보이지 않으면 흰 책상/흰 벽을
            # 차선으로 오인했을 가능성이 큽니다.
            confidence = 0.0

        center_error = (image_center - lane_center) / image_center
        center_error = float(np.clip(center_error, -1.0, 1.0))
        return LaneResult(
            center_error,
            confidence,
            left_x,
            right_x,
            mask_ratio,
            left_count,
            right_count,
            rejected_large_count,
            road_ratio,
            camera_obstacle_ratio,
            camera_obstacle,
            road_valid,
            geometry_valid,
            road_between_ratio,
            lane_width_ratio,
            camera_valid,
        )

    def find_lane_pair_tracks(self, mask, road_mask, width: int) -> LanePairTrack:
        band_count = int(self.get_parameter("lane_search_band_count").value)
        band_height = int(self.get_parameter("lane_search_band_height").value)
        min_pixels = int(self.get_parameter("lane_min_band_pixels").value)
        min_band_count = int(self.get_parameter("min_lane_band_count").value)
        max_peak_width = (
            width * float(self.get_parameter("max_lane_peak_width_ratio").value)
        )
        width_tolerance = float(
            self.get_parameter("lane_pair_width_tolerance_ratio").value
        )
        center_prior = width * float(
            self.get_parameter("lane_pair_center_prior_ratio").value
        )

        left_points = []
        right_points = []
        left_widths = []
        right_widths = []
        previous_center = width / 2.0
        height = mask.shape[0]

        for band in range(band_count):
            y_end = height - band * band_height
            y_start = max(0, y_end - band_height)
            if y_end <= y_start:
                continue

            band_mask = mask[y_start:y_end, :]
            histogram = np.sum(band_mask > 0, axis=0)
            segments = self.histogram_segments(histogram, min_pixels, max_peak_width)
            if len(segments) < 2:
                continue

            y_center = (y_start + y_end) / 2.0
            y_ratio = y_center / max(height, 1)
            expected_width = width * self.expected_lane_width_ratio_at(y_ratio)
            best_pair = None
            best_score = float("inf")

            for left_segment in segments:
                for right_segment in segments:
                    if right_segment[0] <= left_segment[0]:
                        continue
                    lane_width = right_segment[0] - left_segment[0]
                    width_error = abs(lane_width - expected_width)
                    if width_error > expected_width * width_tolerance:
                        continue
                    center = (left_segment[0] + right_segment[0]) / 2.0
                    center_error = abs(center - previous_center)
                    if center_error > center_prior:
                        continue
                    road_score = 1.0 - self.road_between_lanes_ratio(
                        road_mask,
                        left_segment[0],
                        right_segment[0],
                    )
                    score = width_error + 0.45 * center_error + 30.0 * road_score
                    if score < best_score:
                        best_score = score
                        best_pair = (left_segment, right_segment, center)

            if best_pair is None:
                continue

            left_segment, right_segment, previous_center = best_pair
            left_points.append((left_segment[0], y_ratio))
            right_points.append((right_segment[0], y_ratio))
            left_widths.append(left_segment[1])
            right_widths.append(right_segment[1])

        if len(left_points) < min_band_count or len(right_points) < min_band_count:
            empty = LaneTrack(None, 0, 1.0, 1.0, False, [])
            return LanePairTrack(empty, empty, False)

        left_track = self.make_track_from_points(left_points, left_widths, width)
        right_track = self.make_track_from_points(right_points, right_widths, width)
        return LanePairTrack(left_track, right_track, left_track.valid and right_track.valid)

    def histogram_segments(self, histogram, min_pixels: int, max_width: float):
        active = histogram >= min_pixels
        segments = []
        index = 0
        while index < len(active):
            if not active[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(active) and active[index + 1]:
                index += 1
            end = index
            segment_width = end - start + 1
            if segment_width <= max_width:
                local_histogram = histogram[start:end + 1]
                peak_offset = int(np.argmax(local_histogram))
                center = start + peak_offset
                segments.append((center, segment_width))
            index += 1
        return segments

    def make_track_from_points(self, points, widths, image_width: int) -> LaneTrack:
        xs = np.array([point[0] for point in points])
        x = float(np.mean(xs))
        std_ratio = float(np.std(xs) / max(image_width, 1))
        width_ratio = float(np.mean(widths) / max(image_width, 1))
        max_std_ratio = float(self.get_parameter("max_lane_x_std_ratio").value)
        valid = len(points) >= int(
            self.get_parameter("min_lane_band_count").value
        ) and std_ratio <= max_std_ratio
        return LaneTrack(x, len(points), std_ratio, width_ratio, valid, points)

    def find_lane_x_from_bands(
        self,
        mask,
        x_start: int,
        x_end: int,
    ) -> LaneTrack:
        band_count = int(self.get_parameter("lane_search_band_count").value)
        band_height = int(self.get_parameter("lane_search_band_height").value)
        min_pixels = int(self.get_parameter("lane_min_band_pixels").value)
        min_band_count = int(self.get_parameter("min_lane_band_count").value)
        max_peak_width = (
            mask.shape[1]
            * float(self.get_parameter("max_lane_peak_width_ratio").value)
        )
        max_std_ratio = float(self.get_parameter("max_lane_x_std_ratio").value)
        margin = int(
            mask.shape[1]
            * float(self.get_parameter("lane_histogram_margin_ratio").value)
        )
        x_start = max(0, x_start + margin)
        x_end = min(mask.shape[1], x_end - margin)
        if x_end <= x_start:
            return LaneTrack(None, 0, 1.0, 1.0, False, [])

        weighted_positions = []
        peak_widths = []
        points = []
        previous_x = None
        max_jump = (
            mask.shape[1]
            * float(self.get_parameter("max_lane_band_jump_ratio").value)
        )
        height = mask.shape[0]
        for band in range(band_count):
            y_end = height - band * band_height
            y_start = max(0, y_end - band_height)
            if y_end <= y_start:
                continue
            band_mask = mask[y_start:y_end, x_start:x_end]
            histogram = np.sum(band_mask > 0, axis=0)
            peak_pixels = int(np.max(histogram)) if histogram.size else 0
            if peak_pixels < min_pixels:
                continue
            peak_indices = self.histogram_peaks(histogram, min_pixels)
            if not peak_indices:
                continue
            if previous_x is None:
                peak_index = max(peak_indices, key=lambda idx: histogram[idx])
            else:
                local_previous = previous_x - x_start
                peak_index = min(
                    peak_indices,
                    key=lambda idx: abs(idx - local_previous),
                )
                if abs((peak_index + x_start) - previous_x) > max_jump:
                    continue
            active_columns = histogram >= max(1, int(peak_pixels * 0.45))
            peak_width = self.contiguous_width(active_columns, peak_index)
            if peak_width > max_peak_width:
                continue
            peak_x = peak_index + x_start
            previous_x = peak_x
            weight = float(band_count - band)
            weighted_positions.append((peak_x, weight))
            peak_widths.append(peak_width)
            y_center = (y_start + y_end) / 2.0
            y_ratio = y_center / max(height, 1)
            points.append((peak_x, y_ratio))

        if not weighted_positions:
            return LaneTrack(None, 0, 1.0, 1.0, False, [])

        weighted_sum = sum(x * weight for x, weight in weighted_positions)
        total_weight = sum(weight for _, weight in weighted_positions)
        x = weighted_sum / total_weight
        positions = np.array([position for position, _ in weighted_positions])
        std_ratio = float(np.std(positions) / max(mask.shape[1], 1))
        width_ratio = float(np.mean(peak_widths) / max(mask.shape[1], 1))
        valid = len(weighted_positions) >= min_band_count and std_ratio <= max_std_ratio
        return LaneTrack(x, len(weighted_positions), std_ratio, width_ratio, valid, points)

    def histogram_peaks(self, histogram, min_pixels: int):
        peaks = []
        if histogram.size < 3:
            return peaks

        for index in range(1, len(histogram) - 1):
            value = histogram[index]
            if value < min_pixels:
                continue
            if value >= histogram[index - 1] and value >= histogram[index + 1]:
                peaks.append(index)

        return peaks

    def perspective_lane_center(
        self,
        left_track: LaneTrack,
        right_track: LaneTrack,
        image_center: float,
        width: int,
    ) -> Tuple[float, float]:
        lookahead_ratio = float(self.get_parameter("lookahead_ratio").value)
        left_x = self.track_x_at(left_track, lookahead_ratio)
        right_x = self.track_x_at(right_track, lookahead_ratio)

        if left_x is not None and right_x is not None:
            lane_width = max(right_x - left_x, 1.0)
            return (left_x + right_x) / 2.0, lane_width / width

        expected_width = width * self.expected_lane_width_ratio_at(lookahead_ratio)
        if left_x is not None:
            return left_x + expected_width / 2.0, expected_width / width
        if right_x is not None:
            return right_x - expected_width / 2.0, expected_width / width
        return image_center, 0.0

    def track_x_at(self, track: LaneTrack, y_ratio: float) -> Optional[float]:
        if not track.points:
            return track.x

        points = sorted(track.points, key=lambda point: point[1])
        if len(points) == 1:
            return points[0][0]

        ys = np.array([point[1] for point in points])
        xs = np.array([point[0] for point in points])
        degree = 2 if len(points) >= 4 else 1
        coefficients = np.polyfit(ys, xs, degree)
        return float(np.polyval(coefficients, y_ratio))

    def expected_lane_width_ratio_at(self, y_ratio: float) -> float:
        top_width = float(self.get_parameter("top_lane_width_ratio").value)
        bottom_width = float(self.get_parameter("bottom_lane_width_ratio").value)
        y_ratio = float(np.clip(y_ratio, 0.0, 1.0))
        return top_width + (bottom_width - top_width) * y_ratio

    def contiguous_width(self, active_columns, peak_index: int) -> int:
        left = peak_index
        while left > 0 and active_columns[left - 1]:
            left -= 1

        right = peak_index
        while right + 1 < len(active_columns) and active_columns[right + 1]:
            right += 1

        return right - left + 1

    def road_between_lanes_ratio(
        self,
        road_mask,
        left_x: float,
        right_x: float,
    ) -> float:
        left = int(max(0, min(left_x, right_x)))
        right = int(min(road_mask.shape[1], max(left_x, right_x)))
        if right <= left:
            return 0.0

        y_start = int(road_mask.shape[0] * 0.45)
        between = road_mask[y_start:, left:right]
        if between.size == 0:
            return 0.0

        return float(cv2.countNonZero(between)) / float(between.size)

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
            (
                f"err={lane.center_error:+.2f} conf={lane.confidence:.2f} "
                f"road={lane.road_ratio * 100.0:.1f}% "
                f"obs={lane.camera_obstacle_ratio * 100.0:.1f}%"
            ),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return debug

    def control_loop(self):
        now = self.get_clock().now()
        image_ok = self.is_recent(self.last_image_time, "image_timeout_sec")
        scan_ok = self.is_recent(self.last_scan_time, "scan_timeout_sec")
        depth_ok = self.is_recent(self.last_depth_time, "depth_timeout_sec")

        msg = Twist()
        # 카메라가 없으면 차선 기반 주행을 할 수 없으므로
        # 즉시 정지합니다.
        if not image_ok:
            self.publish_stop("Waiting for camera image")
            return

        lane = self.latest_lane
        recovery_command = self.aeb_recovery_command(now, scan_ok)
        if recovery_command is not None:
            speed, steering, drive_state = recovery_command
            msg.linear.x = float(speed)
            msg.angular.z = float(
                np.clip(steering, -float(self.get_parameter("max_angular").value),
                        float(self.get_parameter("max_angular").value))
            )
            self.cmd_pub.publish(msg)
            min_confidence = float(self.get_parameter("min_lane_confidence").value)
            lane_valid = lane is not None and lane.confidence >= min_confidence
            self.log_drive_status(msg, lane, lane_valid, drive_state, scan_ok)
            return

        dt = max((now - self.prev_time).nanoseconds / 1e9, 1e-3)
        self.prev_time = now

        min_confidence = float(self.get_parameter("min_lane_confidence").value)
        lane_valid = lane is not None and lane.confidence >= min_confidence
        lane_lost_active = False
        drive_state = "lane_follow"

        if lane_valid:
            raw_steering = self.pid_steering(lane.center_error, dt)
            smoothing = float(self.get_parameter("steering_smoothing").value)
            steering = (
                (1.0 - smoothing) * raw_steering
                + smoothing * self.last_steering
            )
            speed = self.speed_from_lane(lane, steering)
            self.last_steering = steering
        elif (
            lane is not None
            and not lane.camera_valid
        ):
            speed = 0.0
            steering = 0.0
            self.integral = 0.0
            drive_state = "camera_invalid"
            self.get_logger().warn(
                "Camera image is invalid for driving. Stopping.",
                throttle_duration_sec=1.0,
            )
        elif bool(self.get_parameter("enable_lane_lost_drive").value):
            # 차선이 안 보이는 바닥에서는 라이다 fallback을 씁니다.
            # 라이다가 정상일 때는 열린 방향으로 가고,
            # 라이다가 끊겼을 때만 직전 조향을 조금 남긴
            # 저속 전진으로 떨어집니다.
            if (
                scan_ok
                and self.latest_scan is not None
                and bool(self.get_parameter("lane_lost_use_lidar").value)
            ):
                speed, steering = self.lidar_fallback_command(self.latest_scan)
                drive_state = "lane_lost_lidar"
                lane_lost_active = True
            else:
                decay = float(self.get_parameter("lane_lost_steering_decay").value)
                steering = self.last_steering * decay
                speed = float(self.get_parameter("lane_lost_speed").value)
                drive_state = "lane_lost"
                lane_lost_active = True
            self.integral = 0.0
            self.get_logger().warn(
                "Lane not detected. Using lidar fallback.",
                throttle_duration_sec=1.0,
            )
        else:
            self.publish_stop("Lane not detected")
            return

        # 장애물 회피는 차선 추종보다 우선순위가 높습니다.
        # 가까운 장애물은 정지+회전,
        # 조금 먼 장애물은 감속+회피 조향으로 처리합니다.
        if scan_ok and self.latest_scan is not None:
            obstacle_speed, obstacle_steering, mode = self.obstacle_command(
                self.latest_scan
            )
            if mode == "stop_turn":
                if lane_lost_active:
                    speed = float(self.get_parameter("lane_lost_min_speed").value)
                elif lane_valid:
                    speed = float(self.get_parameter("caution_speed").value)
                else:
                    speed = obstacle_speed
                steering = obstacle_steering
                drive_state = mode
            elif mode == "aeb_stop":
                if bool(self.get_parameter("enable_aeb_recovery").value):
                    self.start_aeb_recovery(now, self.latest_scan)
                    recovery_command = self.aeb_recovery_command(now, scan_ok)
                    if recovery_command is not None:
                        speed, steering, drive_state = recovery_command
                    else:
                        speed = obstacle_speed
                        steering = obstacle_steering
                        drive_state = mode
                else:
                    speed = obstacle_speed
                    steering = obstacle_steering
                    drive_state = mode
            elif mode == "slow_avoid":
                if lane_valid:
                    speed = max(
                        min(speed, obstacle_speed),
                        float(self.get_parameter("lane_follow_min_speed").value),
                    )
                else:
                    speed = min(speed, obstacle_speed)
                steering += obstacle_steering
                drive_state = mode
            elif mode == "side_avoid":
                if lane_valid:
                    speed = max(
                        min(speed, obstacle_speed),
                        float(self.get_parameter("lane_follow_min_speed").value),
                    )
                else:
                    speed = min(speed, obstacle_speed)
                steering += obstacle_steering
                drive_state = mode
            elif mode == "tunnel_center":
                steering += obstacle_steering
                drive_state = mode
            elif mode == "slalom_gap":
                if lane_lost_active:
                    speed = max(
                        speed,
                        float(self.get_parameter("lane_lost_min_speed").value),
                    )
                else:
                    speed = min(speed, obstacle_speed)
                    speed = max(
                        speed,
                        float(self.get_parameter("lane_follow_min_speed").value),
                    )
                steering = (
                    float(self.get_parameter("slalom_lane_mix").value) * steering
                    + obstacle_steering
                )
                drive_state = mode
        else:
            # 라이다가 잠시 끊겼을 때 완전 정지 대신
            # 저속 제한을 걸어 회복 여지를 둡니다.
            if lane_valid:
                speed = max(
                    min(speed, float(self.get_parameter("caution_speed").value)),
                    float(self.get_parameter("lane_follow_min_speed").value),
                )
            else:
                speed = min(speed, float(self.get_parameter("caution_speed").value))
            drive_state = "scan_timeout"

        if (
            lane is not None
            and lane.road_valid
            and lane.camera_obstacle
            and lane_valid
            and speed > 0.0
            and bool(self.get_parameter("enable_camera_low_obstacle").value)
        ):
            # 낮은 턱처럼 라이다가 놓칠 수 있는 물체를
            # 검은 트랙 위 카메라 하단 중앙에서 발견하면 우선 감속합니다.
            speed = min(speed, float(self.get_parameter("low_obstacle_speed").value))
            if drive_state == "lane_follow":
                drive_state = "camera_low_obstacle"

        if (
            bool(self.get_parameter("enable_depth_obstacle").value)
            and depth_ok
            and speed > 0.0
        ):
            depth_stop, depth_slow, depth_delta, _depth_ratio = (
                self.latest_depth_obstacle
            )
            if depth_stop:
                if lane_lost_active:
                    speed = float(self.get_parameter("lane_lost_min_speed").value)
                    steering += (
                        float(self.get_parameter("depth_avoid_gain").value)
                        * depth_delta
                    )
                    drive_state = "depth_slow"
                else:
                    speed = 0.0
                    steering = 0.0
                    drive_state = "depth_stop"
            elif depth_slow:
                if lane_lost_active:
                    speed = max(
                        speed,
                        float(self.get_parameter("lane_lost_min_speed").value),
                    )
                else:
                    speed = min(
                        speed,
                        float(self.get_parameter("low_obstacle_speed").value),
                    )
                steering += (
                    float(self.get_parameter("depth_avoid_gain").value)
                    * depth_delta
                )
                drive_state = "depth_slow"

        if (
            lane_lost_active
            and scan_ok
            and drive_state not in (
                "aeb_stop",
                "depth_stop",
                "camera_invalid",
            )
        ):
            speed = float(self.get_parameter("lane_lost_min_speed").value)

        max_angular = float(self.get_parameter("max_angular").value)
        if drive_state.startswith("aeb_recovery"):
            msg.linear.x = float(speed)
        else:
            msg.linear.x = float(max(speed, 0.0))
        msg.angular.z = float(np.clip(steering, -max_angular, max_angular))
        self.cmd_pub.publish(msg)
        self.log_drive_status(msg, lane, lane_valid, drive_state, scan_ok)

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

    def start_aeb_recovery(self, now, scan: Optional[LaserScan]):
        if self.aeb_recovery_active:
            return

        self.aeb_recovery_active = True
        self.aeb_recovery_start_sec = now.nanoseconds / 1e9
        self.aeb_recovery_turn_direction = self.open_side_direction(scan)
        self.integral = 0.0
        self.get_logger().warn(
            "AEB recovery: reverse and search open path.",
            throttle_duration_sec=0.5,
        )

    def aeb_recovery_command(
        self,
        now,
        scan_ok: bool,
    ) -> Optional[Tuple[float, float, str]]:
        if not self.aeb_recovery_active:
            return None

        elapsed = now.nanoseconds / 1e9 - self.aeb_recovery_start_sec
        reverse_sec = float(self.get_parameter("aeb_recovery_reverse_sec").value)
        turn_sec = float(self.get_parameter("aeb_recovery_turn_sec").value)

        if elapsed < reverse_sec:
            # AEB 직후에는 우선 직선 후진으로 20cm 안전거리에서 벗어납니다.
            return (
                float(self.get_parameter("aeb_recovery_reverse_speed").value),
                0.0,
                "aeb_recovery_back",
            )

        if elapsed < reverse_sec + turn_sec:
            turn_speed = float(self.get_parameter("aeb_recovery_turn_speed").value)
            turn_angular = float(
                self.get_parameter("aeb_recovery_turn_angular").value
            )
            return (
                turn_speed,
                self.aeb_recovery_turn_direction * turn_angular,
                "aeb_recovery_turn",
            )

        self.aeb_recovery_active = False

        if scan_ok and self.latest_scan is not None:
            aeb_angle = self.param_rad("aeb_sector_deg")
            front = self.sector_min(self.latest_scan, -aeb_angle, aeb_angle)
            clear_distance = float(
                self.get_parameter("aeb_recovery_clear_distance").value
            )
            if front < clear_distance:
                # 아직 전방이 너무 가까우면 한 번 더 후진 시퀀스를 시작합니다.
                self.start_aeb_recovery(now, self.latest_scan)
                return self.aeb_recovery_command(now, scan_ok)

        return None

    def open_side_direction(self, scan: Optional[LaserScan]) -> float:
        if scan is None:
            return 1.0

        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)
        return 1.0 if left >= right else -1.0

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
        lane_follow_min_speed = float(
            self.get_parameter("lane_follow_min_speed").value
        )
        straight_min_speed = float(self.get_parameter("straight_min_speed").value)
        straight_threshold = float(
            self.get_parameter("straight_steering_threshold").value
        )
        turn_factor = 1.0 - min(abs(steering) / 1.4, 0.75)
        confidence_factor = 0.45 + 0.55 * lane.confidence
        speed = max(
            min_speed,
            lane_follow_min_speed,
            max_speed * turn_factor * confidence_factor,
        )

        if abs(steering) <= straight_threshold:
            speed = max(speed, straight_min_speed)

        return min(speed, max_speed)

    def lidar_fallback_command(self, scan: LaserScan) -> Tuple[float, float]:
        search_angle = self.param_rad("lane_lost_lidar_angle_deg")
        clearance_distance = float(
            self.get_parameter("lane_lost_clearance_distance").value
        )
        open_distance = float(self.get_parameter("lane_lost_open_distance").value)
        gap_gain = float(self.get_parameter("lane_lost_gap_gain").value)
        obstacle_gain = float(self.get_parameter("lane_lost_obstacle_gain").value)
        base_speed = float(self.get_parameter("lane_lost_speed").value)
        min_fallback_speed = float(self.get_parameter("lane_lost_min_speed").value)

        best_score = -float("inf")
        best_angle = 0.0
        repulsion = 0.0
        valid_count = 0

        for i, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                continue
            if value < scan.range_min or value > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > search_angle:
                continue

            valid_count += 1
            distance = min(value, open_distance)
            forward_preference = 0.35 * abs(angle) / max(search_angle, 1e-3)
            score = distance - forward_preference
            if score > best_score:
                best_score = score
                best_angle = angle

            if value < clearance_distance:
                side = 1.0 if angle >= 0.0 else -1.0
                closeness = (clearance_distance - value) / clearance_distance
                repulsion -= side * closeness

        if valid_count == 0:
            return base_speed, 0.0

        gap_steering = gap_gain * (best_angle / max(search_angle, 1e-3))
        obstacle_steering = obstacle_gain * np.clip(repulsion, -1.0, 1.0)
        steering = float(np.clip(gap_steering + obstacle_steering, -1.0, 1.0))

        turn_factor = 1.0 - min(abs(steering), 0.65)
        speed = max(min_fallback_speed, base_speed * turn_factor)
        return speed, steering

    def obstacle_command(self, scan: LaserScan) -> Tuple[float, float, str]:
        aeb_angle = self.param_rad("aeb_sector_deg")
        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")
        # LIMO 라이다 좌표계 기준:
        # 정면 0rad, 왼쪽 양수, 오른쪽 음수로 가정합니다.
        aeb_front = self.sector_min(scan, -aeb_angle, aeb_angle)
        front = self.sector_min(scan, -front_angle, front_angle)
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)
        self.last_obstacle_distances = (aeb_front, front, left, right)

        aeb_distance = float(self.get_parameter("aeb_distance").value)
        stop_distance = float(self.get_parameter("stop_distance").value)
        slow_distance = float(self.get_parameter("slow_distance").value)
        side_obstacle_distance = float(
            self.get_parameter("side_obstacle_distance").value
        )
        tunnel_side_distance = float(self.get_parameter("tunnel_side_distance").value)
        tunnel_balance_tolerance = float(
            self.get_parameter("tunnel_balance_tolerance").value
        )
        tunnel_centering_gain = float(self.get_parameter("tunnel_centering_gain").value)
        avoid_gain = float(self.get_parameter("avoid_gain").value)
        caution_speed = float(self.get_parameter("caution_speed").value)

        if aeb_front < aeb_distance:
            # AEB: 전방 20cm 이내는 조향보다 정지가 우선입니다.
            return 0.0, 0.0, "aeb_stop"

        if front < stop_distance:
            # 너무 가까우면 전진하지 않고 더 넓은 쪽으로 회전합니다.
            turn_direction = 1.0 if left > right else -1.0
            return 0.0, turn_direction * min(1.0, avoid_gain + 0.25), "stop_turn"

        if bool(self.get_parameter("enable_lidar_slalom").value):
            slalom = self.slalom_gap_command(scan)
            if slalom is not None:
                return slalom

        if front < slow_distance:
            # 전방 여유가 작으면 좌우 공간 차이만큼 회피 조향을 더합니다.
            clearance_delta = np.clip((left - right) / slow_distance, -1.0, 1.0)
            return caution_speed, avoid_gain * clearance_delta, "slow_avoid"

        both_sides_close = left < tunnel_side_distance and right < tunnel_side_distance
        sides_balanced = abs(left - right) < tunnel_balance_tolerance
        if both_sides_close and sides_balanced:
            # 터널에서는 양쪽 벽이 모두 가까운 것이 정상입니다.
            # 회피로 튀지 않고 벽 사이 중앙을 약하게 유지합니다.
            clearance_delta = np.clip(
                (left - right) / tunnel_side_distance,
                -1.0,
                1.0,
            )
            return 0.0, tunnel_centering_gain * clearance_delta, "tunnel_center"

        if left < side_obstacle_distance or right < side_obstacle_distance:
            # 책상 다리처럼 정면 바로 앞은 아니지만
            # 옆쪽 가까이에 있는 얇은 장애물도
            # 미리 피하도록 좌우 여유 차이를 조향에 반영합니다.
            clearance_delta = np.clip(
                (left - right) / side_obstacle_distance,
                -1.0,
                1.0,
            )
            return caution_speed, avoid_gain * clearance_delta, "side_avoid"

        return 0.0, 0.0, "clear"

    def slalom_gap_command(
        self,
        scan: LaserScan,
    ) -> Optional[Tuple[float, float, str]]:
        view_angle = self.param_rad("slalom_view_deg")
        obstacle_distance = float(
            self.get_parameter("slalom_obstacle_distance").value
        )
        min_gap = int(self.get_parameter("slalom_min_gap_indices").value)
        kp = float(self.get_parameter("slalom_kp").value)
        speed = float(self.get_parameter("slalom_speed").value)

        candidates = []
        obstacle_indices = []
        for i, value in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > view_angle:
                continue
            if not math.isfinite(value):
                continue
            if value < scan.range_min or value > scan.range_max:
                continue

            candidates.append((i, angle))
            if value < obstacle_distance:
                obstacle_indices.append(i)

        if not obstacle_indices or not candidates:
            return None

        first_index = candidates[0][0]
        last_index = candidates[-1][0]
        gaps = []

        right_space = obstacle_indices[0] - first_index
        if right_space >= min_gap:
            gaps.append((right_space, first_index, obstacle_indices[0]))

        for left_obs, right_obs in zip(obstacle_indices, obstacle_indices[1:]):
            gap = right_obs - left_obs
            if gap >= min_gap:
                gaps.append((gap, left_obs, right_obs))

        left_space = last_index - obstacle_indices[-1]
        if left_space >= min_gap:
            gaps.append((left_space, obstacle_indices[-1], last_index))

        if not gaps:
            return None

        _, gap_start, gap_end = max(gaps, key=lambda item: item[0])
        target_index = (gap_start + gap_end) // 2
        target_angle = scan.angle_min + target_index * scan.angle_increment
        steering = float(np.clip(kp * target_angle, -1.0, 1.0))
        return speed, steering, "slalom_gap"

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
        # 책상 다리처럼 얇은 물체는 한두 개 beam에만 찍힐 수 있습니다.
        # 퍼센타일을 쓰면 넓은 섹터에서 묻히므로
        # 가까운 샘플을 직접 사용합니다.
        closest_count = max(1, int(self.get_parameter("closest_sample_count").value))
        closest = sorted(ranges)[:closest_count]
        return float(np.mean(closest))

    def param_rad(self, name: str) -> float:
        return math.radians(float(self.get_parameter(name).value))

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

    def log_lane_status(
        self,
        lane: LaneResult,
        width: int,
        height: int,
        roi_top: int,
    ):
        if not bool(self.get_parameter("publish_lane_log").value):
            return

        left_text = "none" if lane.left_x is None else f"{lane.left_x:.0f}px"
        right_text = "none" if lane.right_x is None else f"{lane.right_x:.0f}px"
        roi_height = height - roi_top

        self.get_logger().info(
            "lane "
            f"frame={width}x{height} "
            f"roi_y={roi_top}:{height} "
            f"roi_h={roi_height} "
            f"mask={lane.mask_ratio * 100.0:.2f}% "
            f"road={lane.road_ratio * 100.0:.2f}% "
            f"road_ok={lane.road_valid} "
            f"cam_ok={lane.camera_valid} "
            f"geom_ok={lane.geometry_valid} "
            f"between={lane.road_between_ratio * 100.0:.2f}% "
            f"width={lane.lane_width_ratio:.2f} "
            f"cam_obs={lane.camera_obstacle_ratio * 100.0:.2f}% "
            f"cam_hit={lane.camera_obstacle} "
            f"left={left_text} "
            f"right={right_text} "
            f"left_cnt={lane.left_count} "
            f"right_cnt={lane.right_count} "
            f"large_reject={lane.rejected_large_count} "
            f"err={lane.center_error:+.2f} "
            f"conf={lane.confidence:.2f}",
            throttle_duration_sec=float(
                self.get_parameter("lane_log_period_sec").value
            ),
        )

    def log_drive_status(
        self,
        msg: Twist,
        lane: Optional[LaneResult],
        lane_valid: bool,
        drive_state: str,
        scan_ok: bool,
    ):
        if not bool(self.get_parameter("publish_drive_log").value):
            return

        lane_text = "인식" if lane_valid else "미인식"
        aeb_text = "작동" if drive_state.startswith("aeb") else "미작동"

        self.get_logger().info(
            f"차선: {lane_text} | AEB: {aeb_text} | 속도: {msg.linear.x:.2f}m/s",
            throttle_duration_sec=float(
                self.get_parameter("drive_log_period_sec").value
            ),
        )

    def format_distance(self, value: float) -> str:
        if not math.isfinite(value):
            return "inf"
        return f"{value:.2f}"


def main(args=None):
    rclpy.init(args=args)
    node = LimoAutonomousDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C 시점에 rclpy context가 먼저 종료되는 경우가 있습니다.
        # 그 상태에서 publish하면 RCLError가 나므로 context가 살아있을 때만
        # 마지막 정지 명령을 보냅니다.
        if rclpy.ok():
            try:
                node.cmd_pub.publish(Twist())
                rclpy.spin_once(node, timeout_sec=0.05)
            except Exception as exc:
                node.get_logger().warn(f"Failed to publish final stop: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
