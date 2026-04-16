"""
Qwen Agent tools for robot navigation.

Registers two tools:
  - navigate_to_landmark : send robot to a named location, returns nav result
  - list_landmarks       : list all known landmark names and positions
"""
import json
import os

import json5
from qwen_agent.tools.base import BaseTool, register_tool

from .landmark_loader import LandmarkLoader
from .zmq_client import ZMQNavClient

# --- Config from environment (override via env vars on the server) ---
_ROBOT_IP   = os.environ.get('ROBOT_IP',      '10.203.168.250')
_ROBOT_PORT = int(os.environ.get('ROBOT_PORT', 5555))
_NAV_TIMEOUT_MS = int(os.environ.get('NAV_TIMEOUT_MS', 60000))

_DETECTION_MODE = os.environ.get('DETECTION_MODE', 'sim')

_loader = LandmarkLoader()
_client = ZMQNavClient(robot_ip=_ROBOT_IP, port=_ROBOT_PORT, timeout_ms=_NAV_TIMEOUT_MS)


@register_tool('navigate_to_landmark')
class NavigateToLandmark(BaseTool):
    description = (
        'Navigate the robot to a position in the map — PURE navigation, no observation. '
        'DO NOT use this tool if the user mentions looking, scanning, detecting, checking, '
        'observing, or remembering objects during the trip — use navigate_and_scan instead. '
        'Use this tool ONLY when the user just wants to travel to a place. '
        'Two inputs: (1) landmark_name (looked up via list_landmarks); '
        '(2) x, y coordinates directly (for a previously detected object — '
        'call get_detected_objects first to find (px, py)).'
    )
    parameters = [
        {
            'name': 'landmark_name',
            'type': 'string',
            'description': 'The name of the destination landmark (e.g. "kitchen", "lab"). Omit when navigating by coordinates.',
            'required': False,
        },
        {
            'name': 'x',
            'type': 'number',
            'description': 'Map x coordinate to navigate to. Use when navigating to a detected object position.',
            'required': False,
        },
        {
            'name': 'y',
            'type': 'number',
            'description': 'Map y coordinate to navigate to. Use when navigating to a detected object position.',
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        landmark_name = args.get('landmark_name', '').strip().lower() if args.get('landmark_name') else ''

        if landmark_name:
            try:
                pos = _loader.get(landmark_name)
                x, y = pos['x'], pos['y']
            except KeyError:
                # In real world, also check bridge-stored detected objects
                if _DETECTION_MODE == 'real':
                    reply = _client.send_command('get_detected_objects')
                    obj = reply.get('objects', {}).get(landmark_name)
                    if obj:
                        x, y = obj['px'], obj['py']
                    else:
                        available = list(_loader._landmarks.keys()) + list(reply.get('objects', {}).keys())
                        return json.dumps({'status': 'failed',
                                           'message': f"'{landmark_name}' not found. Available: {', '.join(available)}"})
                else:
                    available = ', '.join(_loader._landmarks.keys())
                    return json.dumps({'status': 'failed',
                                       'message': f"'{landmark_name}' not found. Available: {available}"})
        elif 'x' in args and 'y' in args:
            x, y = float(args['x']), float(args['y'])
        else:
            return json.dumps({'status': 'failed', 'message': 'Provide either landmark_name or both x and y.'})

        print(f"[navigate_to_landmark] navigating to: {landmark_name or f'({x}, {y})'}")
        result = _client.send_goal(x=x, y=y, landmark=landmark_name)
        return json.dumps(result)


@register_tool('list_landmarks')
class ListLandmarks(BaseTool):
    description = (
        'List all locations the robot can navigate to. '
        'Includes both fixed named landmarks AND any objects recently detected by the camera. '
        'Always call this tool fresh — detected objects change every time the camera scans.'
    )
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        landmarks = _loader.list_all()
        if _DETECTION_MODE == 'real':
            reply = _client.send_command('get_detected_objects')
            for name, obj in reply.get('objects', {}).items():
                landmarks.append({
                    'name': name,
                    'x': obj['px'],
                    'y': obj['py'],
                    'description': f"Detected {obj.get('label', name)} (from camera)",
                })
        print(f"[list_landmarks] listing all landmarks")
        return json.dumps(landmarks, ensure_ascii=False)
