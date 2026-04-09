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

    # ========= Static TF: base → camera_frame =========
    # Translation/rotation measured from robot body (base) to RealSense optical frame.
    # Values extracted from the original C++ static broadcaster:
    #   translation: (0.22162912, -0.36896348, 0.35538645)  metres
    #   rotation:    x=-0.34940  y=0.57682  z=0.61817  w=0.40383  (quaternion)
    cameraStaticTf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_frame',
        arguments=[
            '0.22162912', '-0.36896348', '0.35538645',  # x y z
            '-0.34940', '0.57682', '0.61817', '0.40383',  # qx qy qz qw
            'base', 'camera_frame',                        # parent → child
        ],
        output='screen',
    )

    # ========= Launch Description =========
    return LaunchDescription([
        useRealsenseArg,
        realsenseLaunch,
        realsenseZmqNode,
        cameraStaticTf,
    ])
