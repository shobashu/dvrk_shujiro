#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # Declare console argument (to optionally show viewers)
    console_arg = DeclareLaunchArgument(
        'console',
        default_value='false',
        description='Launch image viewers if true'
    )
    
    # Right camera node
    gscam_right = Node(
        package='gscam',
        executable='gscam_node',
        name='gscam_right',
        output='screen',
        parameters=[{
            'gscam_config': 'decklinkvideosrc device-number=0 ! videoconvert',
            'camera_name': 'camera_right',
            'frame_id': 'camera'
        }],
        remappings=[
            ('camera/image_raw', 'camera_right/image_raw'),
            ('camera/camera_info', 'camera_right/camera_info'),
        ]
    )
    
    # Left camera node
    gscam_left = Node(
        package='gscam',
        executable='gscam_node',
        name='gscam_left',
        output='screen',
        parameters=[{
            'gscam_config': 'decklinkvideosrc device-number=1 ! videoconvert',
            'camera_name': 'camera_left',
            'frame_id': 'camera'
        }],
        remappings=[
            ('camera/image_raw', 'camera_left/image_raw'),
            ('camera/camera_info', 'camera_left/camera_info'),
        ]
    )
    
    # Optional image viewer for left camera
    image_view_left = Node(
        package='image_view',
        executable='image_view',
        name='image_view_left',
        output='screen',
        arguments=['image:=camera_left/image_raw'],
        parameters=[{'autosize': False}],
        condition=IfCondition(LaunchConfiguration('console'))
    )
    
    # Optional image viewer for right camera
    image_view_right = Node(
        package='image_view',
        executable='image_view',
        name='image_view_right',
        output='screen',
        arguments=['image:=camera_right/image_raw'],
        parameters=[{'autosize': False}],
        condition=IfCondition(LaunchConfiguration('console'))
    )
    
    return LaunchDescription([
        console_arg,
        gscam_right,
        gscam_left,
        image_view_left,
        image_view_right,
    ])