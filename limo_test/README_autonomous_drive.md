# LIMO Autonomous Drive

카메라 2D 차선 중심 추종과 2D 라이다 장애물 회피를 합친 실차용 ROS2 노드입니다.
기존 패키지 설정 파일을 수정하지 않기 위해 `ros2 run` 엔트리포인트 대신 Python
모듈로 실행합니다.

## 실행

워크스페이스와 ROS 환경을 먼저 source 합니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m limo_test.autonomous_drive --ros-args \
  --params-file limo_test/autonomous_params.yaml
```

토픽 이름이 다르면 실행 시 파라미터로 바꿉니다.

```bash
python3 -m limo_test.autonomous_drive --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p scan_topic:=/scan \
  -p cmd_vel_topic:=/cmd_vel
```

## 사용하는 토픽

- Subscribe: `/camera/image_raw` (`sensor_msgs/Image`)
- Subscribe: `/scan` (`sensor_msgs/LaserScan`)
- Publish: `/cmd_vel` (`geometry_msgs/Twist`)
- Publish: `/limo/autonomy/debug_image` (`sensor_msgs/Image`)

## 알고리즘

1. 카메라 하단 ROI에서 흰색/노란색 차선 후보를 HSV 마스크로 추출합니다.
2. 좌/우 차선 후보의 중심을 계산하고, 둘 중 하나만 보일 때는 예상 차선 폭으로
   주행 중심을 추정합니다.
3. 중심 오차를 PID 제어로 조향값에 반영합니다.
4. 라이다 전방 장애물이 `slow_distance` 안에 들어오면 감속하며 빈 공간 쪽으로
   회피합니다.
5. 라이다 전방 장애물이 `stop_distance` 안에 들어오면 전진을 멈추고 더 넓은
   방향으로 제자리 회전합니다.
6. 카메라 또는 라이다 데이터가 끊기면 속도를 낮추거나 정지합니다.

## 튜닝 순서

1. 바퀴를 띄우거나 낮은 속도에서 `/cmd_vel`이 안전하게 들어가는지 확인합니다.
2. `debug_image_topic`을 보면서 `roi_top_ratio`, `min_lane_area`를 맞춥니다.
3. 차선 중심이 흔들리면 `kd`를 조금 올리고, 반응이 너무 강하면 `kp`를 낮춥니다.
4. 장애물 회피가 늦으면 `slow_distance`, `stop_distance`를 키웁니다.
5. 실차 첫 주행은 `max_speed: 0.15` 정도로 낮춘 뒤 점진적으로 올립니다.

## 안전 메모

이 노드는 학습/실험용 기본 자율주행 로직입니다. 실제 주행 전에는 반드시 넓은
공간에서 저속으로 테스트하고, 별도 비상정지 수단을 준비하세요.
