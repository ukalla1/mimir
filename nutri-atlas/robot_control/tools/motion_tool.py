"""
Qwen Agent tools for direct robot motion primitives.

  - spin_robot : rotate the robot in place by a given angle (degrees)
  - move_robot : move the robot straight forward/backward by a given distance (metres)

Both commands bypass the nav stack and use a P-controller on /cmd_vel with TF feedback
on the robot side. They block until the motion completes or times out.
"""
import json
import os

import json5
from qwen_agent.tools.base import BaseTool, register_tool

from .zmq_client import ZMQNavClient

_ROBOT_IP       = os.environ.get('ROBOT_IP',      '10.203.168.250')
_ROBOT_PORT     = int(os.environ.get('ROBOT_PORT', 5555))
_NAV_TIMEOUT_MS = int(os.environ.get('NAV_TIMEOUT_MS', 60000))

_client = ZMQNavClient(robot_ip=_ROBOT_IP, port=_ROBOT_PORT, timeout_ms=_NAV_TIMEOUT_MS)


@register_tool('spin_robot')
class SpinRobot(BaseTool):
    description = (
        'Rotate the robot in place by a given angle without moving forward or backward. '
        'Positive angle = counter-clockwise (left turn); negative = clockwise (right turn). '
        'Use this when you need the robot to face a different direction or to explore its surroundings with get current detected objects tool.'
    )
    parameters = [
        {
            'name': 'angle_deg',
            'type': 'number',
            'description': (
                'Angle to rotate in degrees. '
                'Positive = counter-clockwise (left), negative = clockwise (right). '
            ),
            'required': True,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        angle_deg = float(args['angle_deg'])
        print(f'[spin_robot] rotating {angle_deg:.1f}°')
        result = _client.send_command('spin', angle_deg=angle_deg)
        return json.dumps(result)


@register_tool('move_robot')
class MoveRobot(BaseTool):
    description = (
        'Move the robot straight forward or backward by a given distance. '
        'Positive distance = forward; negative = backward. '
        'The robot moves along its current heading without turning. '
        'Use this for precise short-range positioning.'
    )
    parameters = [
        {
            'name': 'distance_m',
            'type': 'number',
            'description': (
                'Distance to move in metres. '
                'Positive = forward, negative = backward. '
                'Example: 1.5 moves the robot 1.5 metres forward.'
            ),
            'required': True,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        distance_m = float(args['distance_m'])
        print(f'[move_robot] moving {distance_m:.2f}m')
        result = _client.send_command('move', distance_m=distance_m)
        return json.dumps(result)
