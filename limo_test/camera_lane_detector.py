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
    camera_obstacle_error: float
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
        """ROS 노드 핸들을 받아 파라미터 접근과 로그 출력을 공유합니다."""

        self.node = node
        self.tracked_lane_width_ratio = None
        self.tracked_left_offset_ratio = None
        self.tracked_right_offset_ratio = None
        self._declare_parameters()

    def _declare_parameters(self):
        """차선 인식에 필요한 OpenCV 튜닝 파라미터를 선언합니다.

        모든 값은 autonomous_params.yaml 또는 launch --ros-args -p로 바꿀 수
        있습니다. 현장에서는 debug GUI를 보면서
        색상/ROI/Hough 값을 조정합니다.
        """

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
        self.node.declare_parameter("enable_clahe", True)
        self.node.declare_parameter("adaptive_block_size", 31)
        self.node.declare_parameter("adaptive_c", -8)
        self.node.declare_parameter("relative_white_margin", 18)
        self.node.declare_parameter("reflection_value_min", 220)
        self.node.declare_parameter("reflection_saturation_max", 45)

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
        self.node.declare_parameter("single_lane_trust", 0.55)
        self.node.declare_parameter("single_lane_confidence", 0.30)
        self.node.declare_parameter("lane_width_memory_alpha", 0.25)
        self.node.declare_parameter("single_lane_memory_trust", 0.85)
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
        """카메라 한 프레임에서 차선 결과와 디버그 이미지를 만듭니다.

        autonomous_drive.py의 image_callback()에서 매 프레임 호출됩니다.
        내부 처리 순서는 색상 후보 마스크, edge 마스크, ROI, Hough 직선,
        좌/우 차선 피팅, LaneResult 생성입니다.
        """

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
        """차선 후보 픽셀과 검은 도로 후보 픽셀을 분리합니다.

        어두운 차선과 햇빛 반사에 대응하기 위해 CLAHE 보정 이미지에서
        HSV/HLS 고정 임계값, adaptive threshold, 상대 밝기 threshold를 함께
        사용합니다. 반환값은 lane_mask, road_mask입니다.
        """

        corrected = self._illumination_corrected(frame)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(corrected, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

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

        block_size = int(self._p("adaptive_block_size"))
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(block_size, 3)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(self._p("adaptive_c")),
        )
        relative_threshold = int(
            np.clip(
                np.mean(gray) + float(self._p("relative_white_margin")),
                60,
                245,
            )
        )
        relative = cv2.inRange(gray, relative_threshold, 255)
        reflection = cv2.inRange(
            hsv,
            np.array([0, 0, int(self._p("reflection_value_min"))]),
            np.array([180, int(self._p("reflection_saturation_max")), 255]),
        )

        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        lane_mask = cv2.bitwise_or(lane_mask, cv2.bitwise_and(adaptive, hls_mask))
        lane_mask = cv2.bitwise_or(lane_mask, cv2.bitwise_and(relative, hls_mask))
        lane_mask = cv2.bitwise_and(lane_mask, cv2.bitwise_not(reflection))

        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

        road_mask = cv2.inRange(
            hsv,
            np.array([0, int(self._p("black_road_saturation_min")), 0]),
            np.array([180, 255, int(self._p("black_road_value_max"))]),
        )
        return lane_mask, road_mask

    def _illumination_corrected(self, frame):
        """CLAHE로 조명 차이를 완화한 BGR 이미지를 반환합니다.

        빛이 약해 차선이 어둡거나 화면 일부만 밝은 환경에서
        색상 threshold가 너무 쉽게 깨지는 것을 줄이기 위한 전처리입니다.
        """

        if not bool(self._p("enable_clahe")):
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        corrected = cv2.merge((lightness, a_channel, b_channel))
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    def _edge_mask(self, frame, lane_mask):
        """색상 후보 영역 안에서 edge 성분만 남겨 Hough 입력을 만듭니다.

        Canny는 선의 경계를 잡고, Sobel-x는 세로 방향 차선 edge를 보강합니다.
        마지막에는 lane_mask와 AND 연산해 도로 외부 edge를 최대한 줄입니다.
        """

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
        """차량 앞 도로 영역만 보도록 사다리꼴 ROI를 적용합니다.

        화면 위쪽의 벽, 사람, 배경 edge가 Hough 검출에 들어가지 않도록
        하단 중심부만 남깁니다.
        디버그 이미지에 그릴 ROI 꼭짓점도 함께 반환합니다.
        """

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
        """ROI edge 이미지에서 HoughLinesP로 짧은 선분들을 검출합니다."""

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
        """Hough 선분을 좌/우 차선 후보로 나누고 각각 직선 피팅합니다.

        기울기가 음수이고 화면 왼쪽에 있으면 왼쪽 차선입니다.
        기울기가 양수이고 화면 오른쪽에 있으면 오른쪽 차선입니다.
        반환하는 line은 x = slope * y + intercept 형태입니다.
        """

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
        """여러 점을 x = slope * y + intercept 직선 하나로 근사합니다."""

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
        """피팅된 좌/우 차선과 마스크 통계를 LaneResult로 변환합니다.

        양쪽 차선이 모두 보이면 두 차선의 중앙을 목표로 삼습니다.
        한쪽만 보이면 최근 양쪽 차선이 보였을 때 저장한 차선 폭/오프셋을
        우선 사용합니다. 저장값이 없으면 expected_lane_width_ratio를 사용하고,
        single_lane_trust로 화면 중심과 섞어
        한쪽 라인을 무는 현상을 줄입니다.
        """

        height, width = frame.shape[:2]
        image_center = width / 2.0
        lookahead_y = height * float(self._p("lookahead_ratio"))
        left_x = self._line_x_at(left_line, lookahead_y)
        right_x = self._line_x_at(right_line, lookahead_y)

        mask_ratio = float(cv2.countNonZero(lane_mask)) / float(lane_mask.size)
        road_ratio = float(cv2.countNonZero(road_mask)) / float(road_mask.size)
        (
            camera_obstacle_ratio,
            camera_obstacle,
            camera_obstacle_error,
        ) = self._detect_camera_obstacle(
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
                self._update_lane_memory(left_x, right_x, lane_center, width)
        elif left_x is not None:
            lane_center, lane_width_ratio = self._center_from_single_lane(
                left_x,
                "left",
                image_center,
                width,
            )
            geometry_valid = mask_ratio <= float(self._p("max_lane_mask_ratio"))
            confidence = float(self._p("single_lane_confidence")) if geometry_valid else 0.0
        elif right_x is not None:
            lane_center, lane_width_ratio = self._center_from_single_lane(
                right_x,
                "right",
                image_center,
                width,
            )
            geometry_valid = mask_ratio <= float(self._p("max_lane_mask_ratio"))
            confidence = float(self._p("single_lane_confidence")) if geometry_valid else 0.0

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
            camera_obstacle_error=camera_obstacle_error,
            road_valid=road_valid,
            geometry_valid=geometry_valid,
            road_between_ratio=road_between_ratio,
            lane_width_ratio=lane_width_ratio,
            camera_valid=camera_valid,
        )

    def _line_x_at(self, line, y: float) -> Optional[float]:
        """피팅된 직선이 특정 y 위치에서 갖는 x 좌표를 계산합니다."""

        if line is None:
            return None
        slope, intercept = line
        x = slope * y + intercept
        if not math.isfinite(x):
            return None
        return float(x)

    def _single_lane_center(self, image_center: float, estimated_center: float) -> float:
        """한쪽 차선만 보일 때 목표 중심을 보수적으로 보정합니다."""

        # 한쪽 차선만 보이는 코너에서는 예상 차선 폭 오차가 커져
        # 차량이 한쪽 라인을 물듯이 붙을 수 있습니다.
        # 추정 중심을 화면 중심과 섞어 과한 한쪽 쏠림을 줄입니다.
        trust = float(np.clip(float(self._p("single_lane_trust")), 0.0, 1.0))
        return image_center + trust * (estimated_center - image_center)

    def _update_lane_memory(
        self,
        left_x: float,
        right_x: float,
        lane_center: float,
        image_width: int,
    ):
        """최근 정상 차선 폭과 좌/우 차선-중심 거리를 저장합니다.

        햇빛 반사 때문에 한쪽 흰 선만 보이는 순간에도,
        직전에 양쪽 차선이 보였을 때의 폭과 오프셋을 이용해
        차량 중심을 복원하기 위한 메모리입니다.
        """

        alpha = float(np.clip(float(self._p("lane_width_memory_alpha")), 0.0, 1.0))
        lane_width_ratio = (right_x - left_x) / max(float(image_width), 1.0)
        left_offset_ratio = (lane_center - left_x) / max(float(image_width), 1.0)
        right_offset_ratio = (right_x - lane_center) / max(float(image_width), 1.0)

        self.tracked_lane_width_ratio = self._ema(
            self.tracked_lane_width_ratio,
            lane_width_ratio,
            alpha,
        )
        self.tracked_left_offset_ratio = self._ema(
            self.tracked_left_offset_ratio,
            left_offset_ratio,
            alpha,
        )
        self.tracked_right_offset_ratio = self._ema(
            self.tracked_right_offset_ratio,
            right_offset_ratio,
            alpha,
        )

    def _center_from_single_lane(
        self,
        lane_x: float,
        side: str,
        image_center: float,
        image_width: int,
    ) -> Tuple[float, float]:
        """한쪽 차선만 보일 때 저장된 폭/오프셋으로 중심을 추정합니다.

        side가 left이면 보이는 왼쪽 차선에서 오른쪽으로 저장된 offset만큼
        떨어진 지점을 차선 중앙으로 봅니다. right이면 반대로 계산합니다.
        메모리가 없을 때만 expected_lane_width_ratio를 fallback으로 씁니다.
        """

        expected_width_ratio = self._remembered_width_ratio()
        if side == "left":
            offset_ratio = self._remembered_offset_ratio(
                self.tracked_left_offset_ratio,
                expected_width_ratio / 2.0,
            )
            estimated_center = lane_x + offset_ratio * image_width
        else:
            offset_ratio = self._remembered_offset_ratio(
                self.tracked_right_offset_ratio,
                expected_width_ratio / 2.0,
            )
            estimated_center = lane_x - offset_ratio * image_width

        memory_trust = float(
            np.clip(float(self._p("single_lane_memory_trust")), 0.0, 1.0)
        )
        conservative_center = self._single_lane_center(image_center, estimated_center)
        lane_center = (
            memory_trust * estimated_center
            + (1.0 - memory_trust) * conservative_center
        )
        return lane_center, expected_width_ratio

    def _remembered_width_ratio(self) -> float:
        """저장된 차선 폭 또는 기본 예상 폭을 반환합니다."""

        if self.tracked_lane_width_ratio is None:
            return float(self._p("expected_lane_width_ratio"))
        return float(self.tracked_lane_width_ratio)

    def _remembered_offset_ratio(
        self,
        remembered_offset: Optional[float],
        fallback_offset: float,
    ) -> float:
        """저장된 차선-중심 offset 또는 fallback offset을 반환합니다."""

        if remembered_offset is None:
            return float(fallback_offset)
        return float(remembered_offset)

    def _ema(self, previous: Optional[float], current: float, alpha: float) -> float:
        """차선 폭/오프셋 메모리에 쓰는 지수 이동 평균입니다."""

        if previous is None:
            return float(current)
        return float((1.0 - alpha) * previous + alpha * current)

    def _large_component_count(self, mask) -> int:
        """너무 큰 흰색 덩어리를 세어 배경/반사 오인을 걸러냅니다."""

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
        """좌/우 차선 사이의 검은 도로 비율을 계산합니다."""

        left = int(max(0, min(left_x, right_x)))
        right = int(min(road_mask.shape[1], max(left_x, right_x)))
        top = int(road_mask.shape[0] * 0.45)
        between = road_mask[top:, left:right]
        if between.size == 0:
            return 0.0
        return float(cv2.countNonZero(between)) / float(between.size)

    def _detect_camera_obstacle(self, frame, road_mask, lane_mask):
        """카메라 하단 중앙의 낮은 장애물 의심 영역을 계산합니다.

        라이다가 고깔 상부만 보고 하부를 늦게 잡는 상황을 보조하기 위한
        간단한 비전 기반 감지입니다.
        반환값은 비율, 감지 여부, 좌우 오차입니다.
        """

        height, width = frame.shape[:2]
        top = int(height * float(self._p("camera_obstacle_roi_top_ratio")))
        center_width = int(width * float(self._p("camera_obstacle_center_width_ratio")))
        left = max(0, (width - center_width) // 2)
        right = min(width, left + center_width)
        near = frame[top:, left:right]
        if near.size == 0:
            return 0.0, False, 0.0

        gray = cv2.cvtColor(near, cv2.COLOR_BGR2GRAY)
        non_road = cv2.bitwise_not(road_mask[top:, left:right])
        non_lane = cv2.bitwise_not(lane_mask[top:, left:right])
        obstacle = cv2.bitwise_and(non_road, non_lane)
        obstacle = cv2.bitwise_and(obstacle, cv2.inRange(gray, 55, 255))
        obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ratio = float(cv2.countNonZero(obstacle)) / float(obstacle.size)
        moments = cv2.moments(obstacle)
        error = 0.0
        if moments["m00"] > 0.0:
            obstacle_x = moments["m10"] / moments["m00"]
            center_x = obstacle.shape[1] / 2.0
            error = (obstacle_x - center_x) / max(center_x, 1.0)
            error = float(np.clip(error, -1.0, 1.0))
        return ratio, ratio >= float(self._p("camera_obstacle_min_ratio")), error

    def _draw_debug(self, frame, lane_mask, roi_points, left_line, right_line, lane):
        """OpenCV GUI와 debug 토픽에 표시할 시각화 이미지를 만듭니다.

        원본 영상 위에 차선 마스크, ROI, 좌/우 피팅 선, 화면 중심선,
        추정 차선 중심선을 겹쳐 그립니다.
        """

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
        """피팅된 차선 직선을 디버그 이미지에 그립니다."""

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
        """publish_lane_log가 켜졌을 때 차선 상세 수치를 로그로 출력합니다."""

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
            f"cam_err={lane.camera_obstacle_error:+.2f} "
            f"left={left_text} right={right_text} "
            f"left_cnt={lane.left_count} right_cnt={lane.right_count} "
            f"err={lane.center_error:+.2f} conf={lane.confidence:.2f}",
            throttle_duration_sec=float(self._p("lane_log_period_sec")),
        )

    def _p(self, name: str):
        """ROS 파라미터 값을 짧게 읽기 위한 헬퍼입니다."""

        return self.node.get_parameter(name).value
