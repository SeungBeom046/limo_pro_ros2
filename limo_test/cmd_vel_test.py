import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelTest(Node):

    def __init__(self):
        super().__init__("cmd_vel_test")

        self.publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.timer = self.create_timer(
            0.1,
            self.timer_callback
        )

    def timer_callback(self):
        msg = Twist()

        msg.linear.x = 0.2
        msg.angular.z = 0.0

        self.publisher.publish(msg)

        self.get_logger().info(
            f"Publishing: linear={msg.linear.x}, angular={msg.angular.z}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = CmdVelTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()