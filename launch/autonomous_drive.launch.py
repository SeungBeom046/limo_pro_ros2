from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    image_qos = LaunchConfiguration("image_qos")
    depth_qos = LaunchConfiguration("depth_qos")
    display_debug_window = LaunchConfiguration("display_debug_window")

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
            DeclareLaunchArgument(
                "image_qos",
                default_value="auto",
                description="Image QoS: auto, reliable, or best_effort.",
            ),
            DeclareLaunchArgument(
                "depth_qos",
                default_value="auto",
                description="Depth QoS: auto, reliable, or best_effort.",
            ),
            DeclareLaunchArgument(
                "display_debug_window",
                default_value="true",
                description="Show OpenCV lane detection GUI window.",
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
                        "image_qos": image_qos,
                        "depth_qos": depth_qos,
                        "display_debug_window": ParameterValue(
                            display_debug_window,
                            value_type=bool,
                        ),
                    },
                ],
            ),
        ]
    )
