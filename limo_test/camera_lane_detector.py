import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneResult:
    """카메라 한 프레임에서 계산한 차선 상태.

    center_error는 -1.0~1.0 범위입니다.
    양수면 차선 중심이 화면 왼쪽에 있으므로
    차량은 우회전 보정이 필요합니다.
    """

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


class OpenCVLaneDetector:
    """OpenCV 4.x 기반 차선 인식 모듈.

    참고한 두 글의 흐름을 LIMO 실시간 주행에 맞게 정리했습니다.
    1. 흰색/노란색 차선 후보를 색공간에서 추출
    2. Gaussian blur, Canny, Sobel/HLS 마스크로 후보 보강
    3. 차량 앞 바닥만 보도록 사다리꼴 ROI 적용
    4. HoughLinesP로 선분 검출
    5. 기울기와 위치로 좌/우 차선을 분리하고 직선 피팅
    6. lookahead 지점의 차선 중심 오차와 신뢰도 계산
    """

    def __init__(self, node):
        self.node = node
        self._declare_parameters()

    def _declare_parameters(self):
        # ROI는 화면 아래쪽 도로 영역만 남기는 사다리꼴입니다.
        self.node.declare_parameter("roi_top_ratio", 0.38)
        self.node.declare_parameter("roi_bottom_width_ratio", 0.92)
        self.node.declare_parameter("roi_top_width_ratio", 0.18)
        self.node.declare_parameter("lookahead_ratio", 0.62)

        # 색상 필터: codingwell 글의 흰색/노란색 필터와
        # velog 글의 HLS 채널 아이디어를 같이 사용합니다.
        self.node.declare_parameter("white_lane_value_min", 150)
        self.node.declare_parameter("white_lane_saturation_max", 95)
        self.node.declare_parameter("yellow_h_min", 15)
        self.node.declare_parameter("yellow_h_max", 45)
        self.node.declare_parameter("yellow_s_min", 70)
        self.node.declare_parameter("yellow_v_min", 80)
        self.node.declare_parameter("hls_l_min", 115)
        self.node.declare_parameter("hls_s_min", 55)

        # Edge/Hough 파라미터. OpenCV 4.12.0의 Python API와 호환됩니다.
        self.node.declare_parameter("gaussian_kernel_size", 5)
        self.node.declare_parameter("canny_low_threshold", 50)
        self.node.declare_parameter("canny_high_threshold", 150)
        self.node.declare_parameter("hough_threshold", 24)
        self.node.declare_parameter("hough_min_line_length", 24)
        self.node.declare_parameter("hough_max_line_gap", 28)
        self.node.declare_parameter("min_lane_slope", 0.35)
        self.node.declare_parameter("max_lane_slope", 4.5)

        # 차선 폭/신뢰도 검증.
        # 기존 YAML 키를 최대한 유지해 호환성을 지킵니다.
        self.node.declare_parameter("expected_lane_width_ratio", 0.48)
        self.node.declare_parameter("top_lane_width_ratio", 0.22)
        self.node.declare_parameter("bottom_lane_width_ratio", 0.62)
        self.node.declare_parameter("single_lane_offset_ratio", 0.24)
        self.node.declare_parameter("min_lane_confidence", 0.25)
        self.node.declare_parameter("min_lane_area", 80)
        self.node.declare_parameter("max_lane_area_ratio", 0.18)
        self.node.declare_parameter("min_lane_width_ratio", 0.18)
        self.node.declare_parameter("max_lane_width_ratio", 0.85)
        self.node.declare_parameter("max_lane_mask_ratio", 0.24)
        self.node.declare_parameter("blank_white_mask_ratio", 0.65)
        self.node.declare_parameter("min_road_ratio_for_lane", 0.05)
        self.node.declare_parameter("black_road_value_max", 120)
        self.node.declare_parameter("black_road_saturation_min", 0)
        self.node.declare_parameter("min_road_between_ratio", 0.04)

        # 카메라 하단 중앙의 낮은 장애물 감지는 기본 비활성입니다.
        self.node.declare_parameter("camera_obstacle_roi_top_ratio", 0.72)
        self.node.declare_parameter("camera_obstacle_center_width_ratio", 0.46)
        self.node.declare_parameter("camera_obstacle_min_ratio", 0.06)
        self.node.declare_parameter("enable_camera_low_obstacle", False)

        self.node.declare_parameter("publish_lane_log", False)
        self.node.declare_parameter("lane_log_period_sec", 0.5)

    def process(self, frame) -> Tuple[LaneResult, np.ndarray]:
        height, width = frame.shape[:2]
        color_mask, road_mask = self._candidate_masks(frame)
        edge_mask = self._edge_mask(frame, color_mask)
        roi_mask, roi_points = self._limit_region(edge_mask)
        lines = self._hough_lines(roi_mask)
        left_line, right_line, left_count, right_count = self._fit_lanes(lines, width)

        lane = self._make_lane_result(
            frame,
            color_mask,
            road_mask,
            left_line,
            right_line,
            left_count,
            right_count,
        )
        self._log_lane_status(lane, width, height)
        debug = self._draw_debug(frame, color_mask, roi_points, left_line, right_line, lane)
        return lane, debug

    def _candidate_masks(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, int(self._p("white_lane_value_min"))]),
            np.array([180, int(self._p("white_lane_saturation_max")), 255]),
        )
        yellow_mask = cv2.inRange(
            hsv,
            np.array([
                int(self._p("yellow_h_min")),
                int(self._p("yellow_s_min")),
                int(self._p("yellow_v_min")),
            ]),
            np.array([int(self._p("yellow_h_max")), 255, 255]),
        )

        # HLS는 밝은 흰 선과 채도가 있는 노란 선을
        # 조명 변화 속에서도 잘 잡습니다.
        hls_l = hls[:, :, 1]
        hls_s = hls[:, :, 2]
        hls_mask = np.zeros_like(hls_l, dtype=np.uint8)
        hls_mask[
            (hls_l >= int(self._p("hls_l_min")))
            | (hls_s >= int(self._p("hls_s_min")))
        ] = 255

        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        lane_mask = cv2.bitwise_and(cv2.bitwise_or(lane_mask, hls_mask), lane_mask)

        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

        road_mask = cv2.inRange(
            hsv,
            np.array([0, int(self._p("black_road_saturation_min")), 0]),
            np.array([180, 255, int(self._p("black_road_value_max"))]),
        )
        return lane_mask, road_mask

    def _edge_mask(self, frame, lane_mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_size = int(self._p("gaussian_kernel_size"))
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(
            blurred,
            int(self._p("canny_low_threshold")),
            int(self._p("canny_high_threshold")),
        )

        # 차선 색상 후보와 edge를 합쳐 Hough 입력을 안정화합니다.
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        max_sobel = float(np.max(sobel_x))
        if max_sobel > 0.0:
            sobel_x = np.uint8(255 * sobel_x / max_sobel)
        sobel_mask = cv2.inRange(sobel_x, 35, 255)
        candidate_edges = cv2.bitwise_or(edges, sobel_mask)
        return cv2.bitwise_and(candidate_edges, lane_mask)

    def _limit_region(self, edge_mask):
        height, width = edge_mask.shape[:2]
        top_y = int(height * float(self._p("roi_top_ratio")))
        bottom_half = width * float(self._p("roi_bottom_width_ratio")) / 2.0
        top_half = width * float(self._p("roi_top_width_ratio")) / 2.0
        center = width / 2.0
        points = np.array(
            [
                [int(center - bottom_half), height - 1],
                [int(center - top_half), top_y],
                [int(center + top_half), top_y],
                [int(center + bottom_half), height - 1],
            ],
            dtype=np.int32,
        )
        mask = np.zeros_like(edge_mask)
        cv2.fillConvexPoly(mask, points, 255)
        return cv2.bitwise_and(edge_mask, mask), points

    def _hough_lines(self, roi_mask):
        lines = cv2.HoughLinesP(
            roi_mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(self._p("hough_threshold")),
            minLineLength=int(self._p("hough_min_line_length")),
            maxLineGap=int(self._p("hough_max_line_gap")),
        )
        if lines is None:
            return []
        return [line[0] for line in lines]

    def _fit_lanes(self, lines, image_width: int):
        left_points = []
        right_points = []
        image_center = image_width / 2.0
        min_slope = float(self._p("min_lane_slope"))
        max_slope = float(self._p("max_lane_slope"))

        for x1, y1, x2, y2 in lines:
            dx = float(x2 - x1)
            if abs(dx) < 1.0:
                continue
            slope = float(y2 - y1) / dx
            if abs(slope) < min_slope or abs(slope) > max_slope:
                continue

            x_mean = (x1 + x2) / 2.0
            if slope < 0.0 and x_mean < image_center:
                left_points.extend([(x1, y1), (x2, y2)])
            elif slope > 0.0 and x_mean > image_center:
                right_points.extend([(x1, y1), (x2, y2)])

        return (
            self._fit_line(left_points),
            self._fit_line(right_points),
            len(left_points) // 2,
            len(right_points) // 2,
        )

    def _fit_line(self, points):
        if len(points) < 4:
            return None
        xs = np.array([point[0] for point in points], dtype=np.float32)
        ys = np.array([point[1] for point in points], dtype=np.float32)
        slope, intercept = np.polyfit(ys, xs, 1)
        return float(slope), float(intercept)

    def _make_lane_result(
        self,
        frame,
        lane_mask,
        road_mask,
        left_line,
        right_line,
        left_count: int,
        right_count: int,
    ) -> LaneResult:
        height, width = frame.shape[:2]
        image_center = width / 2.0
        lookahead_y = height * float(self._p("lookahead_ratio"))
        left_x = self._line_x_at(left_line, lookahead_y)
        right_x = self._line_x_at(right_line, lookahead_y)

        mask_ratio = float(cv2.countNonZero(lane_mask)) / float(lane_mask.size)
        road_ratio = float(cv2.countNonZero(road_mask)) / float(road_mask.size)
        camera_obstacle_ratio, camera_obstacle = self._detect_camera_obstacle(
            frame,
            road_mask,
            lane_mask,
        )

        rejected_large_count = self._large_component_count(lane_mask)
        road_valid = road_ratio >= float(self._p("min_road_ratio_for_lane"))
        camera_valid = (
            mask_ratio < float(self._p("blank_white_mask_ratio"))
            and rejected_large_count == 0
        )

        lane_center = image_center
        confidence = 0.0
        geometry_valid = False
        road_between_ratio = 0.0
        lane_width_ratio = 0.0

        if left_x is not None and right_x is not None and right_x > left_x:
            lane_center = (left_x + right_x) / 2.0
            lane_width_ratio = (right_x - left_x) / float(width)
            road_between_ratio = self._road_between_lanes_ratio(road_mask, left_x, right_x)
            geometry_valid = (
                float(self._p("min_lane_width_ratio"))
                <= lane_width_ratio
                <= float(self._p("max_lane_width_ratio"))
                and mask_ratio <= float(self._p("max_lane_mask_ratio"))
            )
            expected_width = float(self._p("expected_lane_width_ratio"))
            width_score = 1.0 - min(abs(lane_width_ratio - expected_width) / expected_width, 1.0)
            count_score = min((left_count + right_count) / 10.0, 1.0)
            min_between = max(
                float(self._p("min_road_between_ratio")),
                1e-3,
            )
            road_score = min(road_between_ratio / min_between, 1.0)
            if geometry_valid:
                confidence = 0.50 + 0.25 * width_score + 0.15 * count_score + 0.10 * road_score
        elif left_x is not None:
            expected_width = width * float(self._p("expected_lane_width_ratio"))
            lane_center = left_x + expected_width / 2.0
            lane_width_ratio = expected_width / float(width)
            geometry_valid = mask_ratio <= float(self._p("max_lane_mask_ratio"))
            confidence = 0.38 if geometry_valid else 0.0
        elif right_x is not None:
            expected_width = width * float(self._p("expected_lane_width_ratio"))
            lane_center = right_x - expected_width / 2.0
            lane_width_ratio = expected_width / float(width)
            geometry_valid = mask_ratio <= float(self._p("max_lane_mask_ratio"))
            confidence = 0.38 if geometry_valid else 0.0

        if not camera_valid or not road_valid:
            confidence = 0.0

        center_error = (image_center - lane_center) / max(image_center, 1.0)
        return LaneResult(
            center_error=float(np.clip(center_error, -1.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            left_x=left_x,
            right_x=right_x,
            mask_ratio=mask_ratio,
            left_count=left_count,
            right_count=right_count,
            rejected_large_count=rejected_large_count,
            road_ratio=road_ratio,
            camera_obstacle_ratio=camera_obstacle_ratio,
            camera_obstacle=camera_obstacle,
            road_valid=road_valid,
            geometry_valid=geometry_valid,
            road_between_ratio=road_between_ratio,
            lane_width_ratio=lane_width_ratio,
            camera_valid=camera_valid,
        )

    def _line_x_at(self, line, y: float) -> Optional[float]:
        if line is None:
            return None
        slope, intercept = line
        x = slope * y + intercept
        if not math.isfinite(x):
            return None
        return float(x)

    def _large_component_count(self, mask) -> int:
        min_area = int(self._p("min_lane_area"))
        max_area = int(mask.size * float(self._p("max_lane_area_ratio")))
        count = 0
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area and area > max_area:
                count += 1
        return count

    def _road_between_lanes_ratio(self, road_mask, left_x: float, right_x: float) -> float:
        left = int(max(0, min(left_x, right_x)))
        right = int(min(road_mask.shape[1], max(left_x, right_x)))
        top = int(road_mask.shape[0] * 0.45)
        between = road_mask[top:, left:right]
        if between.size == 0:
            return 0.0
        return float(cv2.countNonZero(between)) / float(between.size)

    def _detect_camera_obstacle(self, frame, road_mask, lane_mask):
        height, width = frame.shape[:2]
        top = int(height * float(self._p("camera_obstacle_roi_top_ratio")))
        center_width = int(width * float(self._p("camera_obstacle_center_width_ratio")))
        left = max(0, (width - center_width) // 2)
        right = min(width, left + center_width)
        near = frame[top:, left:right]
        if near.size == 0:
            return 0.0, False

        gray = cv2.cvtColor(near, cv2.COLOR_BGR2GRAY)
        non_road = cv2.bitwise_not(road_mask[top:, left:right])
        non_lane = cv2.bitwise_not(lane_mask[top:, left:right])
        obstacle = cv2.bitwise_and(non_road, non_lane)
        obstacle = cv2.bitwise_and(obstacle, cv2.inRange(gray, 55, 255))
        obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ratio = float(cv2.countNonZero(obstacle)) / float(obstacle.size)
        return ratio, ratio >= float(self._p("camera_obstacle_min_ratio"))

    def _draw_debug(self, frame, lane_mask, roi_points, left_line, right_line, lane):
        debug = frame.copy()
        height, width = frame.shape[:2]
        overlay = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        debug = cv2.addWeighted(debug, 0.75, overlay, 0.25, 0)

        cv2.polylines(debug, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
        image_center = width // 2
        lane_center = int(image_center - lane.center_error * image_center)
        cv2.line(debug, (image_center, 0), (image_center, height), (255, 0, 0), 2)
        cv2.line(debug, (lane_center, 0), (lane_center, height), (0, 255, 0), 2)
        self._draw_fit_line(debug, left_line, (0, 255, 255))
        self._draw_fit_line(debug, right_line, (0, 255, 255))

        cv2.putText(
            debug,
            (
                f"err={lane.center_error:+.2f} conf={lane.confidence:.2f} "
                f"road={lane.road_ratio * 100.0:.1f}% "
                f"mask={lane.mask_ratio * 100.0:.1f}%"
            ),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return debug

    def _draw_fit_line(self, image, line, color):
        if line is None:
            return
        height = image.shape[0]
        y1 = height - 1
        y2 = int(height * float(self._p("roi_top_ratio")))
        x1 = self._line_x_at(line, y1)
        x2 = self._line_x_at(line, y2)
        if x1 is None or x2 is None:
            return
        cv2.line(image, (int(x1), y1), (int(x2), y2), color, 4)

    def _log_lane_status(self, lane: LaneResult, width: int, height: int):
        if not bool(self._p("publish_lane_log")):
            return

        left_text = "none" if lane.left_x is None else f"{lane.left_x:.0f}px"
        right_text = "none" if lane.right_x is None else f"{lane.right_x:.0f}px"
        self.node.get_logger().info(
            "lane "
            f"frame={width}x{height} "
            f"mask={lane.mask_ratio * 100.0:.2f}% "
            f"road={lane.road_ratio * 100.0:.2f}% "
            f"road_ok={lane.road_valid} "
            f"cam_ok={lane.camera_valid} "
            f"geom_ok={lane.geometry_valid} "
            f"between={lane.road_between_ratio * 100.0:.2f}% "
            f"width={lane.lane_width_ratio:.2f} "
            f"cam_obs={lane.camera_obstacle_ratio * 100.0:.2f}% "
            f"cam_hit={lane.camera_obstacle} "
            f"left={left_text} right={right_text} "
            f"left_cnt={lane.left_count} right_cnt={lane.right_count} "
            f"err={lane.center_error:+.2f} conf={lane.confidence:.2f}",
            throttle_duration_sec=float(self._p("lane_log_period_sec")),
        )

    def _p(self, name: str):
        return self.node.get_parameter(name).value
