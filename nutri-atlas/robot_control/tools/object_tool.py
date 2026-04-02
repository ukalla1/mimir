"""
Qwen Agent tools for querying the detected-objects map.

Registers two tools:
  - get_detected_objects         : full persistent map via zmq_object_server (port 5556)
  - get_current_detected_objects : live detections via zmq_bridge_node (port 5555)
"""
import json
import os

import zmq
from qwen_agent.tools.base import BaseTool, register_tool

from .zmq_client import ZMQNavClient

_OBJECT_SERVER_IP   = os.environ.get('OBJECT_SERVER_IP',  '10.203.168.250')
_OBJECT_SERVER_PORT = int(os.environ.get('OBJECT_SERVER_PORT', 5556))
_OBJECT_TIMEOUT_MS  = int(os.environ.get('OBJECT_TIMEOUT_MS',  3000))

_ROBOT_IP       = os.environ.get('ROBOT_IP',      '10.203.168.250')
_ROBOT_PORT     = int(os.environ.get('ROBOT_PORT', 5555))
_NAV_TIMEOUT_MS = int(os.environ.get('NAV_TIMEOUT_MS', 10000))

# Client for the full persistent map (port 5556)
class _ObjectClient:
    def __init__(self, ip: str, port: int, timeout_ms: int):
        self._addr = f'tcp://{ip}:{port}'
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()

    def get_objects(self) -> dict:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)
        try:
            sock.send_string(json.dumps({'action': 'get_objects'}))
            return json.loads(sock.recv_string())
        except zmq.Again:
            return {'status': 'error', 'message': f'object server timed out after {self._timeout_ms // 1000}s'}
        except Exception as e:
            return {'status': 'error', 'message': f'ZMQ error: {e}'}
        finally:
            sock.close()


_client        = _ObjectClient(ip=_OBJECT_SERVER_IP, port=_OBJECT_SERVER_PORT, timeout_ms=_OBJECT_TIMEOUT_MS)
_bridge_client = ZMQNavClient(robot_ip=_ROBOT_IP, port=_ROBOT_PORT, timeout_ms=_NAV_TIMEOUT_MS)


@register_tool('get_detected_objects')
class GetDetectedObjects(BaseTool):
    description = (
        'Return the full map of objects currently detected by the robot camera. '
        'Each entry contains the object name and its estimated position. '
        'The list updates in real time, but only while the robot is moving — '
        'if the list is not changing, the robot has either stopped or has fully explored the environment. '
    )
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        print("[get_detected_objects] called")
        result = _client.get_objects()
        return json.dumps(result, ensure_ascii=False)


@register_tool('get_current_detected_objects')
class GetCurrentDetectedObjects(BaseTool):
    description = (
        'Return only the objects the robot camera is detecting RIGHT NOW, '
        'Unlike get_detected_objects, this excludes stale entries from earlier exploration. '
        'Use this to confirm what is currently visible and use it with spin to eplore the surroundings. '
    )
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        print('[get_current_detected_objects] called')
        result = _bridge_client.send_command('get_current_objects')
        return json.dumps(result, ensure_ascii=False)
