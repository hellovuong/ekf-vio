from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("ekf_vio")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([pkg, "config", "euroc.yaml"]),
        description="Path to parameter YAML file",
    )

    vio_node = Node(
        package="ekf_vio",
        executable="vio_node",
        name="ekf_vio_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
        remappings=[
            ("/imu/data",                "/imu0"),
            ("/camera/left/image_raw",   "/cam0/image_raw"),
            ("/camera/right/image_raw",  "/cam1/image_raw"),
        ],
    )

    return LaunchDescription([config_arg, vio_node])
