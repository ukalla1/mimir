"""
ZMQ REP server for the persistent detected-objects map (robot side).

Maintains a cumulative map of all objects ever detected by the camera.
The robot_assistant queries this via GetDetectedObjects (port 5556).

Message format:
    Request:  {"action": "get_objects"}
    Response: {"status": "ok", "objects": [{"name": "cup", "x": 1.2, "y": -0.5}, ...]}

Usage:
    python zmq_object_server.py                  # default port 5556
    python zmq_object_server.py --port 5556      # explicit port
"""
import argparse
import json
import threading
import time

import zmq


class ObjectMap:
    """Thread-safe persistent object map.

    In production, a background thread would subscribe to camera detections
    (e.g. ROS2 topic) and call `update()` to add/refresh entries.
    """

    def __init__(self):
        self._objects: dict[str, dict] = {}  # name -> {name, x, y, confidence, last_seen}
        self._lock = threading.Lock()

    def update(self, name: str, x: float, y: float, confidence: float = 1.0):
        """Add or update a detected object."""
        with self._lock:
            self._objects[name] = {
                'name': name,
                'x': x,
                'y': y,
                'confidence': confidence,
                'last_seen': time.time(),
            }

    def get_all(self) -> list[dict]:
        """Return all objects as a list."""
        with self._lock:
            return list(self._objects.values())

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._objects.clear()


# Global object map — in production, a ROS2 subscriber thread updates this
_object_map = ObjectMap()


def main():
    parser = argparse.ArgumentParser(description='ZMQ object server (robot side)')
    parser.add_argument('--port', type=int, default=5556, help='ZMQ REP port (default: 5556)')
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    addr = f'tcp://0.0.0.0:{args.port}'
    sock.bind(addr)

    print(f'==> zmq_object_server listening on {addr}')
    print(f'    Serves persistent detected-objects map')
    print(f'    Press Ctrl+C to stop.\n')

    try:
        while True:
            raw = sock.recv_string()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                sock.send_string(json.dumps({
                    'status': 'error',
                    'message': f'Invalid JSON: {raw[:200]}',
                }))
                continue

            action = msg.get('action', '')
            print(f'[recv] action={action}')

            if action == 'get_objects':
                objects = _object_map.get_all()
                reply = {
                    'status': 'ok',
                    'objects': objects,
                    'count': len(objects),
                }
            else:
                reply = {
                    'status': 'error',
                    'message': f'Unknown action: {action}',
                }

            sock.send_string(json.dumps(reply))
            print(f'[sent] {len(reply.get("objects", []))} objects\n')

    except KeyboardInterrupt:
        print('\nShutting down.')
    finally:
        sock.close()
        ctx.term()


if __name__ == '__main__':
    main()
