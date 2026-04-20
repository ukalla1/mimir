#!/usr/bin/env python3
"""
ZMQ object map server — reads the live detected_objects.json written by
detector_dis (go2_yolo) and serves it to remote LLM clients on demand.

No ROS dependency. Run standalone alongside zmq_bridge_node.py.

Usage:
    python3 zmq_object_server.py [--port 5556] [--file /tmp/detected_objects.json]

Protocol (JSON over ZMQ REQ/REP):
    {"action": "get_objects"}
        → {"status": "ok", "objects": { ... full object map ... }}

    {"action": "get_object", "name": "chair_1"}
        → {"status": "ok",    "object": {"px": 3.2, "py": -1.1, ...}}
        → {"status": "error", "message": "not found"}

    {"action": "clear"}
        → {"status": "ok"}    (empties the JSON file)
"""
import argparse
import json
import os
import signal
import sys

import zmq


def load_objects(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f'[zmq_object_server] Warning: failed to read {file_path}: {e}', file=sys.stderr)
        return {}


def clear_objects(file_path: str):
    try:
        with open(file_path, 'w') as f:
            json.dump({}, f)
    except Exception as e:
        print(f'[zmq_object_server] Warning: failed to clear {file_path}: {e}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='ZMQ object map server')
    parser.add_argument('--port', type=int, default=int(os.environ.get('OBJECT_SERVER_PORT', 5556)))
    parser.add_argument('--file', type=str, default=os.environ.get('OBJECTS_FILE',
        os.path.expanduser('~/Go2/autonomy_stack_go2/detected_objects.json')))
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.setsockopt(zmq.LINGER, 0)
    # Clear stale JSON from previous sessions before accepting any requests
    clear_objects(args.file)
    print(f'[zmq_object_server] Cleared object database: {args.file}')

    sock.bind(f'tcp://*:{args.port}')
    print(f'[zmq_object_server] REP socket bound on tcp://*:{args.port}')
    print(f'[zmq_object_server] Serving objects from: {args.file}')

    def _shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue

            try:
                req = json.loads(raw)
            except json.JSONDecodeError as e:
                sock.send_string(json.dumps({'status': 'error', 'message': f'Invalid JSON: {e}'}))
                continue

            action = req.get('action', '')

            if action == 'get_objects':
                objects = load_objects(args.file)
                sock.send_string(json.dumps({'status': 'ok', 'objects': objects}))

            elif action == 'get_object':
                name = req.get('name', '')
                objects = load_objects(args.file)
                if name in objects:
                    sock.send_string(json.dumps({'status': 'ok', 'object': objects[name]}))
                else:
                    sock.send_string(json.dumps({'status': 'error', 'message': f'not found: {name}'}))

            elif action == 'clear':
                clear_objects(args.file)
                sock.send_string(json.dumps({'status': 'ok'}))

            else:
                sock.send_string(json.dumps({'status': 'error', 'message': f'Unknown action: {action}'}))

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        ctx.term()
        print('[zmq_object_server] Shutdown.')


if __name__ == '__main__':
    main()
