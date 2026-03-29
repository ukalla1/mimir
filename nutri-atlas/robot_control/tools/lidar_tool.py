"""
Qwen Agent tool for querying LiDAR spatial data via zmq_bridge_node.

Registers one tool:
  - get_lidar_scan : return 8-sector minimum obstacle distances around the robot
"""
import json
import os

import json5
from qwen_agent.tools.base import BaseTool, register_tool

from .zmq_client import ZMQNavClient

_ROBOT_IP       = os.environ.get('ROBOT_IP',      '10.203.168.250')
_ROBOT_PORT     = int(os.environ.get('ROBOT_PORT', 5555))
_NAV_TIMEOUT_MS = int(os.environ.get('NAV_TIMEOUT_MS', 10000))

_client = ZMQNavClient(robot_ip=_ROBOT_IP, port=_ROBOT_PORT, timeout_ms=_NAV_TIMEOUT_MS)


@register_tool('get_lidar_scan')
class GetLidarScan(BaseTool):
    description = (
        'Return the current LiDAR obstacle distances around the robot, divided into 8 directional sectors: '
        'front, front_left, left, back_left, back, back_right, right, front_right. '
        'Each sector reports the distance in metres to the nearest obstacle. '
        'Also returns safe_directions (sectors with no obstacle closer than min_dist), '
        'the overall min_distance, and its direction. '
        'Use this before moving or navigating to check for nearby obstacles and choose a safe direction.'
    )
    parameters = [
        {
            'name': 'min_dist',
            'type': 'number',
            'description': (
                'Minimum clearance in metres to consider a direction safe. '
                'Default is 1.0 m. Increase for more conservative movement.'
            ),
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params) if params.strip() not in ('', '{}') else {}
        min_dist = float(args.get('min_dist', 1.0))
        print(f'[get_lidar_scan] min_dist={min_dist}m')
        result = _client.send_command('get_scan', min_dist=min_dist)
        return json.dumps(result)
