"""Launch CYCLOPS against PX4 SITL + Gazebo (gz) with the x500_mono_cam airframe.

Terminal 1:  cd PX4-Autopilot && make px4_sitl gz_x500_mono_cam
Terminal 2:  MicroXRCEAgent udp4 -p 8888
Terminal 3:  ros2 launch cyclops_ros cyclops_px4.launch.py cyclops_root:=/path/to/drones_project
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = LaunchConfiguration('cyclops_root')
    gz_topic = LaunchConfiguration('gz_image_topic')
    return LaunchDescription([
        DeclareLaunchArgument('cyclops_root', default_value='', description='path to the CYCLOPS repo'),
        DeclareLaunchArgument('gz_image_topic',
                              default_value='/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'),
        DeclareLaunchArgument('goal_x', default_value='40.0'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='camera_bridge',
             arguments=[[gz_topic, '@sensor_msgs/msg/Image[gz.msgs.Image']],
             remappings=[(gz_topic, '/camera/image_raw')]),
        Node(package='cyclops_ros', executable='perception_node', name='cyclops_perception',
             parameters=[{'cyclops_root': root}], output='screen'),
        Node(package='cyclops_ros', executable='planner_node', name='cyclops_planner',
             parameters=[{'cyclops_root': root, 'goal_x': LaunchConfiguration('goal_x'),
                          'goal_y': LaunchConfiguration('goal_y')}], output='screen'),
    ])
