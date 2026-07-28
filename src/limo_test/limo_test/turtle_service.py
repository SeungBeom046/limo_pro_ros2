import rclpy
from rclpy.node import Node

from turtlesim.srv import Spawn, Kill


class TurtleService(Node):

    def __init__(self):
        super().__init__("turtle_service")

        self.spawn_client = self.create_client(Spawn, "/spawn")
        self.kill_client = self.create_client(Kill, "/kill")

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /spawn...")

        while not self.kill_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /kill...")

    def spawn(self):

        req = Spawn.Request()

        req.x = 5.0
        req.y = 5.0
        req.theta = 0.0
        req.name = "1"

        future = self.spawn_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(
            f"Spawned : {future.result().name}"
        )

    def kill(self):

        req = Kill.Request()

        req.name = "1"

        future = self.kill_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info("Killed turtle")


def main(args=None):

    rclpy.init(args=args)

    node = TurtleService()

    node.spawn()

    input("엔터를 누르면 삭제합니다...")

    node.kill()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()