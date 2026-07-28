import rclpy
from rclpy.node import Node

from example_interfaces.srv import AddTwoInts


class AddServer(Node):

    def __init__(self):
        super().__init__("add_server")

        self.srv = self.create_service(
            AddTwoInts,
            "add_two_ints",
            self.callback
        )

        self.get_logger().info("Add Service Ready!")

    def callback(self, request, response):

        response.sum = request.a + request.b

        self.get_logger().info(
            f"{request.a} + {request.b} = {response.sum}"
        )

        return response


def main(args=None):

    rclpy.init(args=args)

    node = AddServer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()