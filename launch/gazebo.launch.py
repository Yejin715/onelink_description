#!/usr/bin/env python3
import os
import tempfile
import xacro
import re

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess,
    DeclareLaunchArgument, OpaqueFunction
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    entity = LaunchConfiguration('entity')

    def _make_urdf(context):
        pkg_share = get_package_share_directory('onelink_description')
        xacro_file = os.path.join(pkg_share, 'urdf', 'onelink.xacro')
        urdf_xml_raw = xacro.process_file(xacro_file).toxml()
        urdf_xml = remove_comments(urdf_xml_raw) 
        tmp_urdf = os.path.join(tempfile.gettempdir(), 'onelink_lengths.urdf')
        with open(tmp_urdf, 'w') as f:
            f.write(urdf_xml)
        return urdf_xml, tmp_urdf

    def remove_comments(text):
        pattern = r'<!--(.*?)-->'
        return re.sub(pattern, '', text, flags=re.DOTALL)

    def _start_flow(context, *args, **kwargs):
        """Gazebo 처음 시작 + 스폰"""
        urdf_xml, tmp_urdf = _make_urdf(context)
        gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'pause': 'false'}.items()
        )

        # 4) RSP 미리 기동 (gazebo_ros2_control이 이 파라미터를 읽음)
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[
                    {'robot_description': urdf_xml},
                    {'use_sim_time': True},
            ],
            output='screen'
        )

        # 6) 스폰 (지면 위로 살짝 띄워 확실히 보이게)
        spawn = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', tmp_urdf, '-entity', entity.perform(context)],
            output='screen'
        )

        # 7) controller_manager 서비스 대기 (루트 네임스페이스)
        wait_cm = ExecuteProcess(
            cmd=['bash', '-lc',
                'until ros2 service type /controller_manager/list_controllers >/dev/null 2>&1; '
                'do sleep 0.2; done; echo CM_READY'],
            output='screen'
        )

        # 8) 컨트롤러 스포너 (yaml을 스포너에도 명시 전달)
        sp_jsb = ExecuteProcess(
            cmd=['ros2', 'run', 'controller_manager', 'spawner',
                'joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            output='screen'
        )
        sp_vel = ExecuteProcess(
            cmd=['ros2', 'run', 'controller_manager', 'spawner',
                'prismatic_velocity_controller', '--controller-manager', '/controller_manager'],
            output='screen'
        )

        # (추가) position 컨트롤러는 '비활성'으로 로드만
        sp_pos = ExecuteProcess(
            cmd=['ros2','run','controller_manager','spawner',
                'prismatic_position_controller','--inactive', '--controller-manager','/controller_manager'],
            output='screen'
        )

        # 9) 이벤트 체인: Gazebo → (spawn 서비스 ready) → Spawn → (CM ready) → JSB → Velocity
        ev_after_spawn  = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[wait_cm]))
        ev_after_cm     = RegisterEventHandler(OnProcessExit(target_action=wait_cm, on_exit=[sp_jsb]))
        ev_after_jsb    = RegisterEventHandler(OnProcessExit(target_action=sp_jsb, on_exit=[sp_vel]))
        ev_after_vel    = RegisterEventHandler(OnProcessExit(target_action=sp_vel, on_exit=[sp_pos]))

        return [
            gazebo,
            rsp,
            spawn,
            ev_after_spawn,
            ev_after_cm,
            ev_after_jsb,
            ev_after_vel,
        ]

    return LaunchDescription([
        DeclareLaunchArgument('entity', default_value='yuilrobotics'),
        OpaqueFunction(function=lambda ctx:
            _start_flow(ctx)
        ),
    ])
