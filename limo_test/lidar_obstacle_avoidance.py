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
        """ROS 노드 핸들을 받아 파라미터와 로그를 공유합니다."""

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
        self.escape_bias_until_sec = 0.0
        self.escape_bias_direction = 0.0
        self.last_decision_log = ""
        self._declare_parameters()

    def _declare_parameters(self):
        """라이다 기반 fallback, 장애물 회피, AEB 파라미터를 선언합니다."""

        # 차선이 안 보일 때 열린 라이다 방향을 찾는 fallback입니다.
        self.node.declare_parameter("lane_lost_lidar_angle_deg", 90.0)
        self.node.declare_parameter("lane_lost_clearance_distance", 1.20)
        self.node.declare_parameter("lane_lost_open_distance", 2.50)
        self.node.declare_parameter("lane_lost_gap_gain", 0.95)
        self.node.declare_parameter("lane_lost_obstacle_gain", 0.95)
        self.node.declare_parameter("lane_lost_wall_gain", 0.50)
        self.node.declare_parameter("wall_follow_target_distance", 0.45)
        self.node.declare_parameter("wall_follow_max_distance", 1.20)
        self.node.declare_parameter("lane_lost_speed", 0.50)
        self.node.declare_parameter("lane_lost_min_speed", 0.50)

        # 섹터는 LIMO LaserScan 좌표계를 정면 0rad, 왼쪽 +, 오른쪽 -로 봅니다.
        self.node.declare_parameter("aeb_sector_deg", 90.0)
        self.node.declare_parameter("aeb_core_sector_deg", 18.0)
        self.node.declare_parameter("front_sector_deg", 90.0)
        self.node.declare_parameter("side_sector_deg", 100.0)
        self.node.declare_parameter("side_check_angle_deg", 90.0)
        self.node.declare_parameter("side_check_width_deg", 15.0)
        self.node.declare_parameter("closest_sample_count", 1)
        self.node.declare_parameter("aeb_distance", 0.30)
        self.node.declare_parameter("wide_aeb_distance", 0.28)
        self.node.declare_parameter("wide_aeb_side_distance", 0.21)
        self.node.declare_parameter("stop_distance", 0.65)
        self.node.declare_parameter("slow_distance", 1.35)
        self.node.declare_parameter("side_obstacle_distance", 0.34)
        self.node.declare_parameter("passable_side_clearance", 0.15)
        self.node.declare_parameter("passable_avoid_speed", 0.18)
        self.node.declare_parameter("corner_side_guard_distance", 0.22)
        self.node.declare_parameter("corner_guard_speed", 0.22)
        self.node.declare_parameter("tunnel_side_distance", 0.70)
        self.node.declare_parameter("tunnel_balance_tolerance", 0.28)
        self.node.declare_parameter("tunnel_centering_gain", 0.35)
        self.node.declare_parameter("avoid_gain", 1.05)

        # AEB 후 복구 시퀀스: 짧게 후진하고 열린 쪽으로 선회합니다.
        self.node.declare_parameter("enable_aeb_recovery", True)
        self.node.declare_parameter("aeb_recovery_clear_distance", 0.45)
        self.node.declare_parameter("aeb_recovery_reverse_sec", 0.7)
        self.node.declare_parameter("aeb_recovery_turn_sec", 0.6)
        self.node.declare_parameter("aeb_recovery_escape_sec", 0.8)
        self.node.declare_parameter("aeb_recovery_reverse_speed", -0.25)
        self.node.declare_parameter("aeb_recovery_turn_speed", 0.05)
        self.node.declare_parameter("aeb_recovery_escape_speed", 0.18)
        self.node.declare_parameter("aeb_recovery_turn_angular", 0.85)
        self.node.declare_parameter("aeb_escape_bias_sec", 1.5)
        self.node.declare_parameter("aeb_escape_bias_angular", 0.45)

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
        """AEB 복구가 진행 중이면 현재 단계의 명령을 반환합니다.

        복구 단계는 후진, 제자리 선회, 전진 원호 순서입니다.
        None을 반환하면 복구가 끝났거나 시작되지 않았다는 뜻입니다.
        """

        if not self.aeb_recovery_active:
            return None

        elapsed = now.nanoseconds / 1e9 - self.aeb_recovery_start_sec
        reverse_sec = float(self._p("aeb_recovery_reverse_sec"))
        turn_sec = float(self._p("aeb_recovery_turn_sec"))
        escape_sec = float(self._p("aeb_recovery_escape_sec"))

        if elapsed < reverse_sec:
            return float(self._p("aeb_recovery_reverse_speed")), 0.0, "aeb_recovery_back"

        if elapsed < reverse_sec + turn_sec:
            return (
                float(self._p("aeb_recovery_turn_speed")),
                self.aeb_recovery_turn_direction
                * float(self._p("aeb_recovery_turn_angular")),
                "aeb_recovery_turn",
            )

        if elapsed < reverse_sec + turn_sec + escape_sec:
            # 후진/제자리 선회 직후 바로 원래 정면으로 재진입하면
            # 같은 장애물에서 AEB가 반복될 수 있습니다.
            # 그래서 짧게 열린 방향으로 전진 원호를 그리며 빠져나갑니다.
            return (
                float(self._p("aeb_recovery_escape_speed")),
                self.aeb_recovery_turn_direction
                * float(self._p("aeb_recovery_turn_angular")),
                "aeb_recovery_escape",
            )

        self.aeb_recovery_active = False
        self.escape_bias_direction = self.aeb_recovery_turn_direction
        self.escape_bias_until_sec = (
            now.nanoseconds / 1e9 + float(self._p("aeb_escape_bias_sec"))
        )
        if scan_ok and scan is not None:
            front = self.sector_min(
                scan,
                -self.param_rad("aeb_core_sector_deg"),
                self.param_rad("aeb_core_sector_deg"),
            )
            if front < float(self._p("aeb_recovery_clear_distance")):
                self.start_recovery(now, scan)
                return self.recovery_command(now, scan_ok, scan)

        return None

    def start_recovery(self, now, scan: Optional[LaserScan]):
        """AEB 이후 후진/선회 복구 시퀀스를 시작합니다.

        열린 쪽을 먼저 판단해 aeb_recovery_turn_direction에 저장하고,
        recovery_command()가 다음 control loop부터 단계별 명령을 반환합니다.
        """

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
        """차선 미검출 시 gap, 장애물 반발, wall follow를 섞어 주행합니다.

        차선이 없거나 카메라가 invalid인 상황의 기본 주행 명령입니다.
        라이다에서 가장 열린 방향을 찾되, 가까운 물체는 반대쪽으로 밀고,
        좌우 벽이 보이면 중앙을 유지하도록 wall_steering을 더합니다.
        """

        search_angle = self.param_rad("lane_lost_lidar_angle_deg")
        clearance_distance = float(self._p("lane_lost_clearance_distance"))
        open_distance = float(self._p("lane_lost_open_distance"))
        gap_gain = float(self._p("lane_lost_gap_gain"))
        obstacle_gain = float(self._p("lane_lost_obstacle_gain"))
        wall_gain = float(self._p("lane_lost_wall_gain"))
        base_speed = float(self._p("lane_lost_speed"))
        min_speed = float(self._p("lane_lost_min_speed"))

        best_score = -float("inf")
        best_angle = 0.0
        repulsion = 0.0
        valid_count = 0
        close_count = 0

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
                close_count += 1
                side = 1.0 if angle >= 0.0 else -1.0
                repulsion -= side * (clearance_distance - value) / clearance_distance

        if valid_count == 0:
            return base_speed, 0.0

        gap_steering = gap_gain * (best_angle / max(search_angle, 1e-3))
        obstacle_steering = obstacle_gain * np.clip(repulsion, -1.0, 1.0)
        wall_steering = wall_gain * self.wall_follow_error(scan)
        steering = float(
            np.clip(gap_steering + obstacle_steering + wall_steering, -1.0, 1.0)
        )
        speed = max(min_speed, base_speed * (1.0 - min(abs(steering), 0.65)))
        mode = "gap_wall" if close_count > 0 else "open_wall"
        self.node.get_logger().info(
            "lane_lost_lidar "
            f"mode={mode} best_angle={math.degrees(best_angle):+.1f}deg "
            f"gap={gap_steering:+.2f} obs={obstacle_steering:+.2f} "
            f"wall={wall_steering:+.2f} steer={steering:+.2f} "
            f"speed={speed:.2f}",
            throttle_duration_sec=0.5,
        )
        return speed, steering

    def wall_follow_error(self, scan: LaserScan) -> float:
        """좌우 벽 거리 차이를 -1.0~1.0 조향 오차로 변환합니다.

        양쪽 벽이 보이면 중앙에 오도록 합니다.
        한쪽 벽만 보이면 target 거리에서 너무 가까워지거나
        멀어지지 않게 보정합니다.
        """

        target = float(self._p("wall_follow_target_distance"))
        max_distance = float(self._p("wall_follow_max_distance"))
        left = self.side_average(scan, 55.0, 105.0, max_distance)
        right = self.side_average(scan, -105.0, -55.0, max_distance)

        if not math.isfinite(left) and not math.isfinite(right):
            return 0.0
        if math.isfinite(left) and math.isfinite(right):
            return float(np.clip((right - left) / max(target, 1e-3), -1.0, 1.0))
        if math.isfinite(left):
            return float(np.clip((target - left) / max(target, 1e-3), -1.0, 1.0))
        return float(np.clip((right - target) / max(target, 1e-3), -1.0, 1.0))

    def side_average(
        self,
        scan: LaserScan,
        start_deg: float,
        end_deg: float,
        max_distance: float,
    ) -> float:
        """지정한 각도 범위의 라이다 거리 중앙값을 반환합니다.

        wall follow에서 55~105도, -105~-55도 같은 측면 벽 거리를 읽는 데
        사용합니다. 유효한 값이 없으면 inf를 반환합니다.
        """

        start = math.radians(start_deg)
        end = math.radians(end_deg)
        values = []
        for i, value in enumerate(scan.ranges):
            if not self._valid_range(scan, value):
                continue
            if value > max_distance:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if start <= angle <= end:
                values.append(value)
        if not values:
            return float("inf")
        return float(np.median(values))

    def obstacle_command(self, scan: LaserScan) -> Tuple[float, float, str]:
        """라이다 장애물 상태를 속도/조향/모드로 변환합니다.

        반환값은 obstacle_speed, obstacle_steering, mode입니다.
        mode는 autonomous_drive.py에서 차선 인식 여부에 따라 다르게 섞입니다.
        AEB는 최우선입니다. 그 다음 passable/tunnel/corner/slow/side
        판단이 이어집니다.
        """

        aeb_angle = self.param_rad("aeb_sector_deg")
        aeb_core_angle = self.param_rad("aeb_core_sector_deg")
        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")

        core_front = self.sector_min(scan, -aeb_core_angle, aeb_core_angle)
        front = self.sector_min(scan, -front_angle, front_angle)
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)
        left_90, right_90 = self.side_clearances(scan)
        self.last_obstacle_distances = (core_front, front, left_90, right_90)

        aeb_distance = float(self._p("aeb_distance"))
        wide_aeb_distance = float(self._p("wide_aeb_distance"))
        stop_distance = float(self._p("stop_distance"))
        slow_distance = float(self._p("slow_distance"))
        side_distance = float(self._p("side_obstacle_distance"))
        tunnel_distance = float(self._p("tunnel_side_distance"))
        tunnel_tolerance = float(self._p("tunnel_balance_tolerance"))
        tunnel_gain = float(self._p("tunnel_centering_gain"))
        avoid_gain = float(self._p("avoid_gain"))
        caution_speed = float(self._p("caution_speed"))
        passable_clearance = float(self._p("passable_side_clearance"))
        passable_speed = float(self._p("passable_avoid_speed"))
        corner_guard = float(self._p("corner_side_guard_distance"))
        corner_speed = float(self._p("corner_guard_speed"))

        passable_width = left_90 >= passable_clearance and right_90 >= passable_clearance
        core_blocked = core_front < aeb_distance
        tunnel_like = left < tunnel_distance and right < tunnel_distance
        sides_balanced = abs(left - right) < tunnel_tolerance

        if core_blocked and not passable_width:
            self.log_decision(
                "aeb_stop",
                core_front,
                front,
                left_90,
                right_90,
                "core front under AEB and side clearance too small",
            )
            return 0.0, 0.0, "aeb_stop"

        wide_aeb_hit = self.wide_aeb_hit(scan, aeb_angle, wide_aeb_distance)
        if wide_aeb_hit:
            # 고깔처럼 정면 core 밖에 걸리는 물체도 실제로는 차체 앞에
            # 걸릴 수 있습니다. 전방 ±90도에서 매우 가까운 beam은
            # 통과 가능 판단보다 우선해서 정지합니다.
            self.log_decision(
                "wide_aeb_stop",
                core_front,
                front,
                left_90,
                right_90,
                "wide front sector under emergency distance",
            )
            return 0.0, 0.0, "aeb_stop"

        if core_blocked and passable_width:
            # 정면 core가 20cm 안으로 들어와도 3시/9시 폭이 살아 있으면
            # 바로 AEB로 멈추지 않고 더 열린 쪽으로 저속 회피합니다.
            # 고깔이나 기둥처럼 통과 가능한 물체에서 반복 AEB를 줄입니다.
            open_delta = np.clip((left_90 - right_90) / passable_clearance, -1.0, 1.0)
            if abs(open_delta) < 0.15:
                open_delta = 1.0 if left >= right else -1.0
            self.log_decision(
                "passable_avoid",
                core_front,
                front,
                left_90,
                right_90,
                "core front under AEB but side clearance is passable",
            )
            return passable_speed, avoid_gain * open_delta, "passable_avoid"

        if tunnel_like and sides_balanced and passable_width and not core_blocked:
            clearance_delta = np.clip((left_90 - right_90) / tunnel_distance, -1.0, 1.0)
            self.log_decision(
                "tunnel_center",
                core_front,
                front,
                left_90,
                right_90,
                "side walls detected but 3/9 o'clock clearance is passable",
            )
            return 0.0, tunnel_gain * clearance_delta, "tunnel_center"

        if front < stop_distance and not passable_width:
            turn_direction = 1.0 if left > right else -1.0
            self.log_decision(
                "stop_turn",
                core_front,
                front,
                left_90,
                right_90,
                "front too close and passage width is not enough",
            )
            return 0.0, turn_direction * min(1.0, avoid_gain + 0.25), "stop_turn"

        if left_90 < corner_guard or right_90 < corner_guard:
            clearance_delta = np.clip((left_90 - right_90) / corner_guard, -1.0, 1.0)
            self.log_decision(
                "corner_guard",
                core_front,
                front,
                left_90,
                right_90,
                "side clearance near 3/9 o'clock is tight",
            )
            return corner_speed, avoid_gain * clearance_delta, "corner_guard"

        if front < slow_distance and bool(self._p("enable_lidar_slalom")):
            slalom = self.slalom_gap_command(scan)
            if slalom is not None:
                self.log_decision(
                    "slalom_gap",
                    core_front,
                    front,
                    left_90,
                    right_90,
                    "gap search found passable path",
                )
                return slalom

        if front < slow_distance:
            clearance_delta = np.clip((left - right) / slow_distance, -1.0, 1.0)
            self.log_decision(
                "slow_avoid",
                core_front,
                front,
                left_90,
                right_90,
                "front obstacle inside slow distance",
            )
            return caution_speed, avoid_gain * clearance_delta, "slow_avoid"

        if tunnel_like and sides_balanced:
            clearance_delta = np.clip((left - right) / tunnel_distance, -1.0, 1.0)
            self.log_decision(
                "tunnel_center",
                core_front,
                front,
                left_90,
                right_90,
                "balanced side walls",
            )
            return 0.0, tunnel_gain * clearance_delta, "tunnel_center"

        if left < side_distance or right < side_distance:
            clearance_delta = np.clip((left - right) / side_distance, -1.0, 1.0)
            self.log_decision(
                "side_avoid",
                core_front,
                front,
                left_90,
                right_90,
                "side obstacle inside guard distance",
            )
            return caution_speed, avoid_gain * clearance_delta, "side_avoid"

        escape_bias = self.escape_bias_command()
        if escape_bias is not None:
            self.log_decision(
                "escape_bias",
                core_front,
                front,
                left_90,
                right_90,
                "post-AEB bias keeps robot from re-entering same path",
            )
            return escape_bias

        self.log_decision("clear", core_front, front, left_90, right_90, "clear")
        return 0.0, 0.0, "clear"

    def slalom_gap_command(self, scan: LaserScan) -> Optional[Tuple[float, float, str]]:
        """전방 장애물 사이의 가장 넓은 gap을 찾아 통과 명령을 만듭니다.

        obstacle_command()에서 front < slow_distance일 때만 호출됩니다.
        차선이 보이는 중에는 autonomous_drive.py가 이 조향을 제한해
        차선을 벗어나지 않도록 합니다.
        """

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

    def wide_aeb_hit(
        self,
        scan: LaserScan,
        max_angle: float,
        front_distance: float,
    ) -> bool:
        """넓은 전방에서 긴급정지할 가까운 beam이 있는지 확인합니다.

        정면은 더 민감하게, ±90도 가장자리로 갈수록 기준 거리를 낮춰
        터널 벽 오탐을 줄입니다. 고깔처럼 정면 core 밖에 걸리는 물체를
        더 빨리 잡기 위한 보조 AEB입니다.
        """

        side_distance = float(self._p("wide_aeb_side_distance"))
        for i, value in enumerate(scan.ranges):
            if not self._valid_range(scan, value):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > max_angle:
                continue

            # 정면은 민감하게, ±90도 가장자리는 터널 벽 오탐을 줄이기 위해
            # 기준 거리를 낮춥니다. 그래도 고깔처럼 차체 앞쪽에 걸리는
            # 물체는 ±90도 시야 안에서 더 빨리 잡힙니다.
            angle_ratio = min(abs(angle) / max(max_angle, 1e-3), 1.0)
            threshold = (
                front_distance
                + (side_distance - front_distance) * angle_ratio
            )
            if value < threshold:
                return True
        return False

    def open_side_direction(self, scan: Optional[LaserScan]) -> float:
        """AEB 복구 시 왼쪽/오른쪽 중 더 열린 방향을 선택합니다."""

        if scan is None:
            return 1.0
        front_angle = self.param_rad("front_sector_deg")
        side_angle = self.param_rad("side_sector_deg")
        left = self.sector_min(scan, front_angle, side_angle)
        right = self.sector_min(scan, -side_angle, -front_angle)
        return 1.0 if left >= right else -1.0

    def side_clearances(self, scan: LaserScan) -> Tuple[float, float]:
        """3시/9시 방향의 통과 가능 폭을 계산합니다."""

        # 3시/9시 방향의 실제 통과 폭을 따로 봅니다.
        # 터널 벽이 ±45도 전방 섹터에 섞여 AEB를 때리는 문제를 줄입니다.
        center = self.param_rad("side_check_angle_deg")
        half_width = self.param_rad("side_check_width_deg")
        left = self.sector_min(scan, center - half_width, center + half_width)
        right = self.sector_min(scan, -center - half_width, -center + half_width)
        return left, right

    def escape_bias_command(self) -> Optional[Tuple[float, float, str]]:
        """AEB 복구 후 같은 경로 재진입을 막는 조향 bias를 반환합니다."""

        now_sec = self.node.get_clock().now().nanoseconds / 1e9
        if now_sec >= self.escape_bias_until_sec:
            return None
        steering = (
            self.escape_bias_direction
            * float(self._p("aeb_escape_bias_angular"))
        )
        return 0.0, steering, "escape_bias"

    def log_decision(
        self,
        mode: str,
        core_front: float,
        front: float,
        left_90: float,
        right_90: float,
        reason: str,
    ):
        """라이다 판단 모드와 핵심 거리값을 로그로 남깁니다."""

        text = (
            f"lidar mode={mode} core={self.format_distance(core_front)}m "
            f"front={self.format_distance(front)}m "
            f"left90={self.format_distance(left_90)}m "
            f"right90={self.format_distance(right_90)}m reason={reason}"
        )
        self.last_decision_log = text
        self.node.get_logger().info(text, throttle_duration_sec=0.5)

    def sector_min(self, scan: LaserScan, start_angle: float, end_angle: float) -> float:
        """지정한 각도 범위에서 가장 가까운 거리 평균을 반환합니다.

        closest_sample_count개만 평균내므로 책상 다리나 고깔처럼 얇은 물체도
        넓은 섹터 평균에 묻히지 않습니다.
        """

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
        """degree 단위 ROS 파라미터를 radian으로 변환합니다."""

        return math.radians(float(self._p(name)))

    def format_distance(self, value: float) -> str:
        """로그용 거리 문자열을 만듭니다."""

        if not math.isfinite(value):
            return "inf"
        return f"{value:.2f}"

    def _valid_range(self, scan: LaserScan, value: float) -> bool:
        """LaserScan range 값이 finite이고 센서 유효 범위 안인지 확인합니다."""

        return (
            math.isfinite(value)
            and value >= scan.range_min
            and value <= scan.range_max
        )

    def _p(self, name: str):
        """ROS 파라미터 값을 짧게 읽기 위한 헬퍼입니다."""

        return self.node.get_parameter(name).value
