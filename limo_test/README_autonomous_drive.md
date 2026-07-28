# LIMO Autonomous Drive

카메라 2D 차선 중심 추종과 2D 라이다 장애물 회피를 합친 실차용 ROS2 노드입니다.
## 빌드

```bash
colcon build --symlink-install --packages-select limo_test
source install/setup.bash
```

## 실행: launch 권장

```bash
ros2 launch limo_test autonomous_drive.launch.py
```

토픽 이름이 다르면 launch argument로 바꿉니다.

```bash
ros2 launch limo_test autonomous_drive.launch.py \
  image_topic:=/camera/color/image_raw \
  scan_topic:=/scan \
  cmd_vel_topic:=/cmd_vel
```

줄바꿈이 헷갈리면 한 줄로 실행하는 것이 가장 안전합니다.

```bash
ros2 launch limo_test autonomous_drive.launch.py image_topic:=/camera/color/image_raw scan_topic:=/scan cmd_vel_topic:=/cmd_vel
```

## 실행: ros2 run

워크스페이스와 ROS 환경을 먼저 source 합니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run limo_test autonomous_drive --ros-args \
  --params-file install/limo_test/share/limo_test/config/autonomous_params.yaml
```

토픽 이름이 다르면 실행 시 파라미터로 바꿉니다.

```bash
ros2 run limo_test autonomous_drive --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p scan_topic:=/scan \
  -p cmd_vel_topic:=/cmd_vel
```

한 줄 실행:

```bash
ros2 run limo_test autonomous_drive --ros-args --params-file ~/wego_ws/src/limo_test/limo_test/autonomous_params.yaml -p image_topic:=/camera/color/image_raw -p scan_topic:=/scan -p cmd_vel_topic:=/cmd_vel
```

여러 줄로 실행할 때는 `\` 뒤에 공백을 넣으면 안 됩니다. `\ `처럼 공백이
붙으면 다음 줄의 `--ros-args`, `-p`, `--params-file`이 별도 bash 명령으로
실행됩니다.

## 사용하는 토픽

- Subscribe: `/camera/color/image_raw` (`sensor_msgs/Image`)
- Subscribe: `/scan` (`sensor_msgs/LaserScan`)
- Publish: `/cmd_vel` (`geometry_msgs/Twist`)
- Publish: `/limo/autonomy/debug_image` (`sensor_msgs/Image`)

## 알고리즘

1. 카메라 하단 ROI에서 검은 도로와 흰색 좌/우 차선을 분리합니다.
2. 중간 시야부터 아래까지 여러 row band를 보고, 아래쪽은 차선 폭을 넓게,
   위쪽은 좁게 기대하는 원근 모델을 적용합니다.
3. 각 band에서 여러 흰색 peak 후보를 뽑고, 아래에서 위로 가장 연속적인 좌/우
   차선 후보를 연결합니다.
4. `lookahead_ratio` 위치의 좌우 차선 중앙을 주행 목표로 잡고, 한쪽만 보일 때는
   원근 기반 예상 차선 폭으로 중앙을 추정합니다.
5. 중심 오차를 PID 제어로 조향값에 반영합니다.
6. 라이다 전방 장애물이 `slow_distance` 안에 들어오면 감속하며 빈 공간 쪽으로
   회피합니다.
7. 라이다 전방 장애물이 `stop_distance` 안에 들어오면 전진을 멈추고 더 넓은
   방향으로 제자리 회전합니다.
8. 라이다 전방 20cm 이내는 `aeb_stop`으로 즉시 정지합니다.
9. 터널에서는 양쪽 라이다 벽을 장애물이 아니라 통로 벽으로 보고 중앙을 유지합니다.
10. 여러 장애물이 있으면 전방 ±90도 라이다 범위에서 좌측 바깥, 장애물 사이 틈,
   우측 바깥 중 가장 넓은 gap을 골라 `slalom_gap`으로 통과합니다.
11. 책상 다리처럼 얇은 장애물은 섹터 퍼센타일 대신 가까운 beam 값을 사용해
   감지합니다.
12. 낮은 턱처럼 라이다가 놓칠 수 있는 장애물은 카메라 하단 중앙의 비검정 영역으로
   보조 감지해 감속합니다.
13. 차선이 안 보이면 `lane_lost_lidar` fallback으로 전환해 라이다 기준 열린
   방향을 찾고, 가까운 물체 반대쪽으로 조향합니다.
14. 카메라 또는 라이다 데이터가 끊기면 속도를 낮추거나 정지합니다.

## 튜닝 순서

1. 바퀴를 띄우거나 낮은 속도에서 `/cmd_vel`이 안전하게 들어가는지 확인합니다.
2. `debug_image_topic`을 보면서 `roi_top_ratio`, `lane_min_band_pixels`를 맞춥니다.
3. 차선 중심이 흔들리면 `kd`를 조금 올리고, 반응이 너무 강하면 `kp`를 낮춥니다.
4. 장애물 회피가 늦으면 `slow_distance`, `stop_distance`를 키웁니다.
5. 터널에서 벽 때문에 튀면 `tunnel_side_distance`와 `tunnel_balance_tolerance`를
   코스 폭에 맞춥니다.
6. 낮은 턱 감지가 너무 민감하면 `camera_obstacle_min_ratio`를 키우고, 너무 늦으면
   낮춥니다.
7. 책상 다리 같은 얇은 장애물을 놓치면 `side_obstacle_distance`를 키우거나
   `closest_sample_count`를 `1`로 유지합니다.
8. AEB가 너무 민감하면 `aeb_sector_deg`를 줄이고, 너무 늦으면 `aeb_distance`를
   `0.25` 정도로 키웁니다.
9. 차선이 없는 바닥에서는 기본 `lane_lost_speed: 0.50`으로 라이다 fallback
   저속 주행을 합니다. 너무 빠르면 `lane_lost_speed`를 낮춥니다. 라이다
   fallback이 너무 크게 꺾으면 `lane_lost_gap_gain`, `lane_lost_obstacle_gain`을
   낮춥니다.
10. 차선 인식 직진 구간은 `straight_min_speed: 1.00` 이상으로 주행합니다.
   불안하면 `straight_min_speed`, `max_speed`를 낮춘 뒤 점진적으로 올립니다.

## 안전 메모

이 노드는 학습/실험용 기본 자율주행 로직입니다. 실제 주행 전에는 반드시 넓은
공간에서 저속으로 테스트하고, 별도 비상정지 수단을 준비하세요.

## 차선 로그 확인

기본 주행 로그는 한 줄에 핵심 3개만 출력합니다.

```text
차선: 인식 | AEB: 미작동 | 속도: 1.00m/s
차선: 미인식 | AEB: 작동 | 속도: 0.00m/s
```

상세 차선 디버깅이 필요하면 `publish_lane_log: true`로 바꾸면 됩니다.

```text
lane frame=640x480 roi_y=264:480 roi_h=216 mask=2.30% road=72.00% cam_obs=1.20% cam_hit=False left=185px right=462px left_cnt=5 right_cnt=5 large_reject=0 err=-0.01 conf=0.88
```

- `mask`: ROI 안에서 흰색 차선으로 잡힌 픽셀 비율입니다. 0에 가까우면 색상
  범위가 안 맞거나 카메라가 차선을 못 보고 있는 상태입니다.
- `road`: ROI 안에서 검은 도로로 잡힌 픽셀 비율입니다. 너무 낮으면 카메라 각도,
  조명, 검은색 임계값을 봐야 합니다.
- `road`가 `min_road_ratio_for_lane`보다 낮으면 흰 선이 보여도 차선으로 인정하지
  않고 라이다 fallback으로 전환합니다.
- 흰 물체 오인을 줄이기 위해 차선 후보는 `min_lane_band_count`개 이상의 row
  band에서 이어져야 하고, `max_lane_peak_width_ratio`보다 넓은 흰 덩어리는
  차선에서 제외합니다.
- 햇빛 반사로 도로가 덜 검게 보일 때는 라인 모양이 충분히 정확하면
  `reflected_road_ratio_for_lane`, `reflected_road_between_ratio` 기준으로
  완화해서 차선을 인정합니다.
- 화면이 거의 전부 흰색이거나 거의 전부 검은색이면 카메라 판단 불가 상태로 보고
  라이다 fallback 속도를 내지 않고 정지합니다.
- `cam_obs/cam_hit`: 카메라 하단 중앙의 낮은 장애물 의심 비율과 감지 여부입니다.
- `left/right`: 좌우 차선 후보의 x 위치입니다. `none`이면 해당 방향 차선을
  못 잡은 것입니다.
- `left_cnt/right_cnt`: 유효한 row band 개수입니다. 계속 0이면 `roi_top_ratio`,
  `lane_min_band_pixels`, 흰색 임계값을 조정해야 합니다.
- `large_reject`: 너무 큰 흰색 영역을 배경으로 판단해 버린 횟수입니다. 바닥이나
  책상을 차선으로 오인할 때 이 값이 올라갑니다.
- `err/conf`: 주행에 실제로 쓰는 중심 오차와 신뢰도입니다.
