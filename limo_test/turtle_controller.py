import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


class TurtleSpiralController(Node):

    def __init__(self):
        super().__init__("turtle_spiral_controller")

        # turtle1에게 속도 명령을 보내는 Publisher
        self.cmd_pub = self.create_publisher(
            Twist,
            "/turtle1/cmd_vel",
            10
        )

        # turtle1의 현재 위치를 받아오는 Subscriber
        self.pose_sub = self.create_subscription(
            Pose,
            "/turtle1/pose",
            self.pose_callback,
            10
        )

        # 0.1초마다 제어 함수 실행
        self.timer_period = 0.1
        self.timer = self.create_timer(
            self.timer_period,
            self.timer_callback
        )

        # 아직 위치 정보를 받지 않았기 때문에 None으로 시작
        self.current_pose = None

        # 현재 상태
        # expand: 회오리가 점점 커지는 중
        # return: 왔던 경로를 반대로 돌아가는 중
        self.state = "expand"

        # 회오리 크기 계산에 사용하는 카운트
        self.step_count = 0

        # 바깥으로 이동할 때 사용한 명령들을 저장
        self.command_history = []

        # 돌아갈 때 사용할 리스트 인덱스
        self.return_index = -1

        # 전진 속도
        self.linear_speed = 1.0

        # 처음 각속도
        # 클수록 작은 원을 그림
        self.start_angular_speed = 2.0

        # 최소 각속도
        # 너무 작아지면 거의 직선 운동이 되므로 제한
        self.min_angular_speed = 0.25

        # 각속도를 줄이는 비율
        # 값이 클수록 회오리가 빠르게 커짐
        self.angular_decay = 0.006

        # turtlesim 화면 경계 여유값
        self.wall_min = 0.8
        self.wall_max = 10.3

        self.get_logger().info(
            "turtle1 회오리 운동 시작"
        )

    def pose_callback(self, msg):
        """
        turtle1의 현재 위치와 방향을 저장한다.
        """
        self.current_pose = msg

    def is_near_wall(self):
        """
        거북이가 벽 근처에 있는지 검사한다.
        """

        if self.current_pose is None:
            return False

        x = self.current_pose.x
        y = self.current_pose.y

        return (
            x <= self.wall_min
            or x >= self.wall_max
            or y <= self.wall_min
            or y >= self.wall_max
        )

    def create_expand_command(self):
        """
        점점 커지는 회오리 운동 명령을 만든다.

        선속도는 일정하게 유지하고,
        각속도를 조금씩 감소시킨다.

        회전 반경은 대략:

            반경 = 선속도 / 각속도

        따라서 각속도가 작아질수록
        회전 반경이 커진다.
        """

        msg = Twist()

        # 일정한 속도로 앞으로 이동
        msg.linear.x = self.linear_speed

        # 시간이 지날수록 각속도를 감소
        angular_speed = (
            self.start_angular_speed
            - self.angular_decay * self.step_count
        )

        # 각속도가 최소값 아래로 내려가지 않도록 제한
        angular_speed = max(
            angular_speed,
            self.min_angular_speed
        )

        # 양수면 반시계 방향 회전
        msg.angular.z = angular_speed

        return msg

    def create_reverse_command(self, original_command):
        """
        저장된 속도 명령을 반대로 만든다.

        전진 명령은 후진 명령으로,
        왼쪽 회전은 오른쪽 회전으로 바꾼다.
        """

        reverse_msg = Twist()

        reverse_msg.linear.x = -original_command.linear.x
        reverse_msg.angular.z = -original_command.angular.z

        return reverse_msg

    def timer_callback(self):
        """
        0.1초마다 실행되는 메인 제어 함수
        """

        # 아직 위치 정보를 받지 못했다면 대기
        if self.current_pose is None:
            return

        # 회오리가 바깥으로 커지는 상태
        if self.state == "expand":

            # 벽 근처에 도달했다면 복귀 상태로 전환
            if self.is_near_wall():
                self.state = "return"

                # 가장 마지막 명령부터 역순으로 사용
                self.return_index = len(self.command_history) - 1

                self.get_logger().info(
                    f"벽 도달, 복귀 시작 "
                    f"x={self.current_pose.x:.2f}, "
                    f"y={self.current_pose.y:.2f}"
                )

                # 상태가 바뀌는 순간 잠깐 정지
                self.cmd_pub.publish(Twist())
                return

            # 회오리 운동 명령 생성
            msg = self.create_expand_command()

            # 명령 전송
            self.cmd_pub.publish(msg)

            # 나중에 역방향으로 돌아오기 위해 명령 저장
            saved_command = Twist()
            saved_command.linear.x = msg.linear.x
            saved_command.angular.z = msg.angular.z

            self.command_history.append(saved_command)

            # 다음 주기에는 각속도를 조금 더 감소
            self.step_count += 1

        # 왔던 경로를 반대로 돌아가는 상태
        elif self.state == "return":

            # 아직 되돌릴 명령이 남아 있다면
            if self.return_index >= 0:

                original_command = self.command_history[
                    self.return_index
                ]

                reverse_msg = self.create_reverse_command(
                    original_command
                )

                self.cmd_pub.publish(reverse_msg)

                self.return_index -= 1

            else:
                # 저장된 모든 명령을 역순으로 실행 완료
                self.cmd_pub.publish(Twist())

                self.get_logger().info(
                    "복귀 완료, 다시 회오리 확대 시작"
                )

                # 다음 회오리 운동을 위해 초기화
                self.command_history.clear()
                self.step_count = 0
                self.return_index = -1

                self.state = "expand"

    def stop_turtle(self):
        """
        노드 종료 시 거북이를 정지시킨다.
        """
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)

    node = TurtleSpiralController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_turtle()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()