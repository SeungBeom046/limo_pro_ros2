# LIMO ROS 2 Humble 개발환경

Mac Apple Silicon + Docker Desktop + VS Code Dev Containers 기준입니다.

## 구성

- Ubuntu 22.04 계열
- ROS 2 Humble Desktop
- AgileX `limo_ros2` Humble 브랜치
- CycloneDDS
- colcon / rosdep / vcstool
- turtlesim / demo_nodes_py
- teleop_twist_keyboard
- rqt_robot_steering
- XQuartz GUI 전달

## 1. XQuartz 설정

Mac에서 XQuartz를 실행하고:

```bash
xhost +localhost
```

기존에 사용했던 설정이 유지되어 있다면 그대로 사용해도 됩니다.

## 2. 컨테이너 빌드 및 실행

이 폴더에서:

```bash
docker compose build
docker compose up -d
docker exec -it limo_dev bash
```

컨테이너 안에서 ROS 환경은 entrypoint가 자동으로 불러옵니다.

## 3. LIMO 소스 설치

컨테이너 안에서:

```bash
bash scripts/bootstrap.sh
source install/setup.bash
```

## 4. VS Code

폴더를 VS Code로 연 뒤:

1. `Cmd + Shift + P`
2. `Dev Containers: Reopen in Container`

## 5. turtlesim 테스트

터미널 1:

```bash
bash scripts/test_turtlesim.sh
```

터미널 2:

```bash
ros2 run turtlesim turtle_teleop_key
```

## 6. chatter 테스트

터미널 1:

```bash
bash scripts/test_chatter_talker.sh
```

터미널 2:

```bash
bash scripts/test_chatter_listener.sh
```

## 7. LIMO 빌드 및 실행

```bash
cd /workspaces/limo_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch limo_base limo_base.launch.py
```

키보드 조작:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Git 분리

이 폴더 자체를 LIMO 전용 Git 저장소로 사용하세요. 기존 F1TENTH 저장소 및
`f1tenth_dev` 컨테이너와 파일·브랜치·의존성이 섞이지 않습니다.

## 제한

XQuartz에서 turtlesim 같은 일반 X11 GUI는 동작하지만, Apple Silicon Docker의
OpenGL/GLX 제약 때문에 RViz2와 Gazebo GUI는 정상 동작하지 않을 수 있습니다.
실제 LIMO에서 RViz를 실행하거나, Linux PC/VM을 사용하는 것이 더 안정적입니다.
