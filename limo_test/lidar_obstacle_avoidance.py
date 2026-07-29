import math
from typing import Optional, Tuple

import numpy as np
from sensor_msgs.msg import LaserScan


class LidarObstacleAvoidance:
    """2D LaserScan 기반 장애물 회피와 차선 손실 fallback 모듈.

    autonomous_drive.py는 이 클래스가 반환하는 속도/조향 보정값만
    사용합니다. 라이다 알고리즘을 별도 파일에 둔 이유는
    차선 인식과 장애물 회피의 책임을
    분리해서 노드 연결 흐름을 눈으로 따라가기 쉽게 하기 위해서입니다.
    """

    def __init__(self, node):
        self.node = node
        self.last_obstacle_distances = (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
        self.aeb_recovery_active = False
        self.aeb_recovery_start_sec = 0.0
        self.aeb_recovery_turn_direction = 1.0
        self._declare_parameters()

    def _declare_parameters(self):
        # 차선이 안 보일 때 열린 라이다 방향을 찾는 fallback입니다.
        self.node.declare_parameter("lane_lost_lidar_angle_deg", 95.0)
        self.node.declare_parameter("lane_lost_clearance_distance", 1.20)
        self.node.declare_parameter("lane_lost_open_distance", 2.50)
        self.node.declare_parameter("lane_lost_gap_gain", 0.85)
        self.node.declare_parameter("lane_lost_obstacle_gain", 0.75)
        self.node.declare_parameter("lane_lost_speed", 0.50)
        self.node.declare_parameter("lane_lost_min_speed", 0.50)

        # 섹터는 LIMO LaserScan 좌표계를 정면 0rad, 왼쪽 +, 오른쪽 -로 봅니다.
        self.node.declare_parameter("aeb_sector_deg", 45.0)
        self.node.declare_parameter("front_sector_deg", 45.0)
        self.node.declare_parameter("side_sector_deg", 100.0)
        self.node.declare_parameter("closest_sample_count", 1)
        self.node.declare_parameter("aeb_distance", 0.15)
        self.node.declare_parameter("stop_distance", 0.50)
        self.node.declare_parameter("slow_distance", 1.20)
        self.node.declare_parameter("side_obstacle_distance", 0.34)
        self.node.declare_parameter("tunnel_side_distance", 0.70)
        self.node.declare_parameter("tunnel_balance_tolerance", 0.28)
        self.node.declare_parameter("tunnel_centering_gain", 0.35)
        self.node.declare_parameter("avoid_gain", 0.85)

        # AEB 후 복구 시퀀스: 짧게 후진하고 열린 쪽으로 선회합니다.
        self.node.declare_parameter("enable_aeb_recovery", True)
        self.node.declare_parameter("aeb_recovery_clear_distance", 0.45)
        self.node.declare_parameter("aeb_recovery_reverse_sec", 0.7)
        self.node.declare_parameter("aeb_recovery_turn_sec", 0.6)
        self.node.declare_parameter("aeb_recovery_reverse_speed", -0.18)
        self.node.declare_parameter("aeb_recovery_turn_speed", 0.05)
        self.node.declare_parameter("aeb_recovery_turn_angular", 0.85)

        # 장애물 사이 빈 공간을 찾아 통과하는 간단한 slalom/gap 선택입니다.
        self.node.declare_parameter("enable_lidar_slalom", True)
        self.node.declare_parameter("slalom_view_deg", 90.0)
        self.node.declare_parameter("slalom_obstacle_distance", 0.75)
        self.node.declare_parameter("slalom_min_gap_indices", 5)
        self.node.declare_parameter("slalom_kp", 1.0)
        self.node.declare_parameter("slalom_speed", 0.50)

    def recovery_command(
        self,
        now,
        scan_ok: bool,
        scan: Optional[LaserScan],
    ) -> Optional[Tuple[float, float, str]]:
        """AEB 복구가 진행 중이면 현재 단계의 명령을 반환합니다."""

        if not self.aeb_recovery_active:
            return None

        elapsed = now.nanoseconds / 1e9 - self.aeb_recovery_start_sec
        reverse_sec = float(self._p("aeb_recovery_reverse_sec"))
        turn_sec = float(self._p("aeb_recovery_turn_sec"))

        if elapsed < reverse_sec:
            return float(self._p("aeb_recovery_reverse_speed")), 0.0, "aeb_recovery_back"

        if elapsed < reverse_sec + turn_sec:
            return (
                float(self._p("aeb_recovery_turn_speed")),
                self.aeb_recovery_turn_direction
                * float(self._p("aeb_recovery_turn_angular")),
                "aeb_recovery_turn",
            )

        self.aeb_recovery_active = False
        if scan_ok and scan is not None:
            front = self.sector_min(
                scan,
                -self.param_rad("aeb_sector_deg"),
                self.param_rad("aeb_sector_deg"),
            )
            if front < float(self._p("aeb_recovery_clear_distance")):
                self.start_recovery(now, scan)
                return self.recovery_command(now, scan_ok, scan)

        return None

    def start_recovery(self, now, scan: Optional[LaserScan]):
        if self.aeb_recovery_active:
            return

        self.aeb_recovery_active = True
        self.aeb_recovery_start_sec = now.nanoseconds / 1e9
        self.aeb_recovery_turn_direction = self.open_side_direction(scan)
        self.node.get_logger().warn(
            "AEB recovery: reverse and search open path.",
            throttle_duration_sec=0.5,
        )

    def lane_lost_command(self, scan: LaserScan) -> Tuple[float, float]:
        """차선 미검출 시 열린 라이다 방향으로 저속 주행합니다."""

        search_angle = self.param_rad("lane_lost_lidar_angle_deg")
        clearance_distance = float(self._p("lane_lost_clearance_distance"))
        open_distance = float(self._p("lane_lost_open_distance"))
        gap_gain = float(self._p("lane_lost_gap_gain"))
        obstacle_gain = float(self._p("lane_lost_obstacle_gain"))
        base_speed = float(self._p("lane_lost_speed"))
        min_speed = float(self._p("lane_lost_min_speed"))

        best_score = -float("inf")
        best_angle = 0.0
        repulsion = 0.0
        valid_count = 0

        for i, value in enumerate(scan.ranges):
            if not self._valid_range(scan, value):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > search_angle:
                continue

            valid_count += 1
            distance = min(value, open_distance)
            forward_penalty = 0.35 * abs(angle) / max(search_angle, 1e-3)
            score = distance - forward_penalty
            if score > best_score:
                best_score = score
                best_angle = angle

            if value < clearance_distance:
                side = 1.0 if angle >= 0.0 else -1.0
                repulsion -= side * (clearance_distance - value) / clearance_distance

        if valid_count == 0:
            return base_speed, 0.0

        gap_steering = gap_gain * (best_angle / max(search_angle, 1e-3))
        obstacle_steering = obstacle_gain * np.clip(repulsion, -1.0, 1.0)
        steering = float(np.clip(gap_steering + obstacle_steering, -1.0, 1.0))
        speed = max(min_speed, base_speed * (1.0 - min(abs(steering), 0.65)))
        return speed, steering

    def obstacle_command(self, scan: LaserScan) -> Tuple[float, float, str]:
        """라이다 장애물 상태를 속도/조향/모드로 변환합니다."""

        aeb_angle = self.param_rad("aeb_sector_deg")
        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")

        aeb_front = self.sector_min(scan, -aeb_angle, aeb_angle)
        front = self.sector_min(scan, -front_angle, front_angle)
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)
        self.last_obstacle_distances = (aeb_front, front, left, right)

        aeb_distance = float(self._p("aeb_distance"))
        stop_distance = float(self._p("stop_distance"))
        slow_distance = float(self._p("slow_distance"))
        side_distance = float(self._p("side_obstacle_distance"))
        tunnel_distance = float(self._p("tunnel_side_distance"))
        tunnel_tolerance = float(self._p("tunnel_balance_tolerance"))
        tunnel_gain = float(self._p("tunnel_centering_gain"))
        avoid_gain = float(self._p("avoid_gain"))
        caution_speed = float(self._p("caution_speed"))

        if aeb_front < aeb_distance:
            return 0.0, 0.0, "aeb_stop"

        if front < stop_distance:
            turn_direction = 1.0 if left > right else -1.0
            return 0.0, turn_direction * min(1.0, avoid_gain + 0.25), "stop_turn"

        if bool(self._p("enable_lidar_slalom")):
            slalom = self.slalom_gap_command(scan)
            if slalom is not None:
                return slalom

        if front < slow_distance:
            clearance_delta = np.clip((left - right) / slow_distance, -1.0, 1.0)
            return caution_speed, avoid_gain * clearance_delta, "slow_avoid"

        both_sides_close = left < tunnel_distance and right < tunnel_distance
        sides_balanced = abs(left - right) < tunnel_tolerance
        if both_sides_close and sides_balanced:
            clearance_delta = np.clip((left - right) / tunnel_distance, -1.0, 1.0)
            return 0.0, tunnel_gain * clearance_delta, "tunnel_center"

        if left < side_distance or right < side_distance:
            clearance_delta = np.clip((left - right) / side_distance, -1.0, 1.0)
            return caution_speed, avoid_gain * clearance_delta, "side_avoid"

        return 0.0, 0.0, "clear"

    def slalom_gap_command(self, scan: LaserScan) -> Optional[Tuple[float, float, str]]:
        view_angle = self.param_rad("slalom_view_deg")
        obstacle_distance = float(self._p("slalom_obstacle_distance"))
        min_gap = int(self._p("slalom_min_gap_indices"))
        kp = float(self._p("slalom_kp"))
        speed = float(self._p("slalom_speed"))

        candidates = []
        obstacle_indices = []
        for i, value in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > view_angle or not self._valid_range(scan, value):
                continue
            candidates.append((i, angle))
            if value < obstacle_distance:
                obstacle_indices.append(i)

        if not obstacle_indices or not candidates:
            return None

        first_index = candidates[0][0]
        last_index = candidates[-1][0]
        gaps = []

        if obstacle_indices[0] - first_index >= min_gap:
            gaps.append((obstacle_indices[0] - first_index, first_index, obstacle_indices[0]))

        for left_obs, right_obs in zip(obstacle_indices, obstacle_indices[1:]):
            gap = right_obs - left_obs
            if gap >= min_gap:
                gaps.append((gap, left_obs, right_obs))

        if last_index - obstacle_indices[-1] >= min_gap:
            gaps.append((last_index - obstacle_indices[-1], obstacle_indices[-1], last_index))

        if not gaps:
            return None

        _, gap_start, gap_end = max(gaps, key=lambda item: item[0])
        target_index = (gap_start + gap_end) // 2
        target_angle = scan.angle_min + target_index * scan.angle_increment
        steering = float(np.clip(kp * target_angle, -1.0, 1.0))
        return speed, steering, "slalom_gap"

    def open_side_direction(self, scan: Optional[LaserScan]) -> float:
        if scan is None:
            return 1.0
        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)
        return 1.0 if left >= right else -1.0

    def sector_min(self, scan: LaserScan, start_angle: float, end_angle: float) -> float:
        ranges = []
        for i, value in enumerate(scan.ranges):
            if not self._valid_range(scan, value):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if start_angle <= angle <= end_angle:
                ranges.append(value)

        if not ranges:
            return float("inf")

        closest_count = max(1, int(self._p("closest_sample_count")))
        return float(np.mean(sorted(ranges)[:closest_count]))

    def param_rad(self, name: str) -> float:
        return math.radians(float(self._p(name)))

    def _valid_range(self, scan: LaserScan, value: float) -> bool:
        return (
            math.isfinite(value)
            and value >= scan.range_min
            and value <= scan.range_max
        )

    def _p(self, name: str):
        return self.node.get_parameter(name).value
