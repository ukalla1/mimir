"""
ZMQ REQ client — sends a navigation goal to the robot and blocks until feedback.

The robot side runs a REP socket that:
  1. Receives the goal JSON
  2. Publishes it to ROS2 /way_point
  3. Waits for /nav_result from the navigation stack
  4. Sends the result back as JSON reply
"""
import json
import uuid
import zmq


class ZMQNavClient:
    def __init__(self, robot_ip: str = '127.0.0.1', port: int = 5555, timeout_ms: int = 60000):
        """
        Args:
            robot_ip:   IP address of the robot onboard computer.
            port:       ZMQ port (must match zmq_bridge_node on robot side).
            timeout_ms: How long to wait for nav feedback before declaring timeout.
        """
        self._addr = f'tcp://{robot_ip}:{port}'
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()

    def send_goal(self, x: float, y: float, landmark: str = '') -> dict:
        """
        Send a navigation goal and block until the robot replies.

        Returns a dict with keys:
            goal_id  : str
            status   : 'success' | 'failed' | 'timeout'
            message  : str
        """
        goal_id = str(uuid.uuid4())
        payload = {
            'goal_id': goal_id,
            'landmark': landmark,
            'x': x,
            'y': y,
        }

        # Create a fresh REQ socket per call (safe for single-threaded agent).
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)

        try:
            sock.send_string(json.dumps(payload))
            reply_raw = sock.recv_string()
            reply = json.loads(reply_raw)
            return reply
        except zmq.Again:
            return {
                'goal_id': goal_id,
                'status': 'timeout',
                'message': f'No reply from robot within {self._timeout_ms // 1000}s',
            }
        except Exception as e:
            return {
                'goal_id': goal_id,
                'status': 'failed',
                'message': f'ZMQ error: {e}',
            }
        finally:
            sock.close()

    def send_command(self, command_type: str, **params) -> dict:
        """
        Send a generic command (spin, move, etc.) and block until the robot replies.

        Args:
            command_type: 'spin' | 'move'
            **params:     command-specific fields, e.g. angle_deg=90 or distance_m=1.5

        Returns a dict with keys: goal_id, status ('success'|'failed'|'timeout'), message
        """
        goal_id = str(uuid.uuid4())
        payload = {'goal_id': goal_id, 'command_type': command_type, **params}

        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)

        try:
            sock.send_string(json.dumps(payload))
            reply_raw = sock.recv_string()
            return json.loads(reply_raw)
        except zmq.Again:
            return {
                'goal_id': goal_id,
                'status':  'timeout',
                'message': f'No reply from robot within {self._timeout_ms // 1000}s',
            }
        except Exception as e:
            return {
                'goal_id': goal_id,
                'status':  'failed',
                'message': f'ZMQ error: {e}',
            }
        finally:
            sock.close()

    def close(self):
        self._ctx.term()
