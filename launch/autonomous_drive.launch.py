from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "limo_test"

    default_params = PathJoinSubstitution(
        [
            FindPackageShare(package_name),
            "config",
            "autonomous_params.yaml",
        ]
    )

    params_file = LaunchConfiguration("params_file")
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="YAML parameter file for the autonomous drive node.",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera/color/image_raw",
                description="Camera image topic.",
            ),
            DeclareLaunchArgument(
                "scan_topic",
                default_value="/scan",
                description="2D lidar LaserScan topic.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/depth/image_raw",
                description="Depth image topic.",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic",
                default_value="/cmd_vel",
                description="Velocity command topic.",
            ),
            Node(
                package=package_name,
                executable="autonomous_drive",
                name="limo_autonomous_drive",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "image_topic": image_topic,
                        "depth_topic": depth_topic,
                        "scan_topic": scan_topic,
                        "cmd_vel_topic": cmd_vel_topic,
                    },
                ],
            ),
        ]
    )
