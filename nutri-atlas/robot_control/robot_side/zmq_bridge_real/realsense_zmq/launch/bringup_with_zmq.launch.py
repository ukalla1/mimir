from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    useRealsenseArg = DeclareLaunchArgument(
        'use_realsense',
        default_value='true',
        description='Whether to launch RealSense camera driver'
    )
    
    useRealsense = LaunchConfiguration('use_realsense')

    # ========= RealSense =========
    realsenseShareDir = get_package_share_directory('realsense2_camera')
    realsenseLaunchPath = os.path.join(
        realsenseShareDir, 'launch', 'rs_launch.py'
    )

    realsenseLaunch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsenseLaunchPath),
        condition=IfCondition(useRealsense),
        launch_arguments={}.items()
    )

    # RealSense → ZMQ (color + depth)
    realsenseZmqNode = Node(
        package='realsense_zmq',
        executable='realsense_zmq_node',
        name='realsense_zmq_node',
        output='screen',
    )

    # ========= Launch Description =========
    return LaunchDescription([
        useRealsenseArg,
        realsenseLaunch,
        realsenseZmqNode,
    ])
