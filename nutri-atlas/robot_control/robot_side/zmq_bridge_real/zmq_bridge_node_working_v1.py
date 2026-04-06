"""
ZMQ REP server running on the robot side.

Receives JSON commands from robot_assistant.py (via ZMQNavClient) and dispatches
them to the appropriate ROS2 action / topic. Blocks until the action completes,
then sends back a JSON reply.

Message types handled (identified by presence of fields):

1. Navigation goal  — has 'x' and 'y', no 'command_type'
   {"goal_id": "...", "landmark": "kitchen", "x": 1.5, "y": 2.3}
   → Publish geometry_msgs/PointStamped to /way_point

2. Spin command     — command_type == 'spin'
   {"goal_id": "...", "command_type": "spin", "angle_deg": 120}
   → P-controller on /cmd_vel with TF feedback

3. Move command     — command_type == 'move'
   {"goal_id": "...", "command_type": "move", "distance_m": 1.5}
   → P-controller on /cmd_vel with TF feedback

4. Current objects  — command_type == 'get_current_objects'
   {"goal_id": "...", "command_type": "get_current_objects"}
   → Return live camera detections (filtered by recency)

5. LiDAR scan      — command_type == 'get_scan'
   {"goal_id": "...", "command_type": "get_scan", "min_dist": 1.0}
   → Return 8-sector obstacle distances

Usage:
    python zmq_bridge_node.py                    # default port 5555
    python zmq_bridge_node.py --port 5555        # explicit port
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import zmq


# ---------------------------------------------------------------------------
# ROS2 node (initialized once at startup)
# ---------------------------------------------------------------------------

_ros_node: Node = None
_waypoint_pub = None


def init_ros():
    global _ros_node, _waypoint_pub
    rclpy.init()
    _ros_node = rclpy.create_node('zmq_bridge_node')
    _waypoint_pub = _ros_node.create_publisher(PointStamped, '/way_point', 10)
    print('==> ROS2 node initialized, publishing to /way_point')


def shutdown_ros():
    if _ros_node is not None:
        _ros_node.destroy_node()
    rclpy.shutdown()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_navigate(goal_id: str, x: float, y: float, landmark: str) -> dict:
    """
    Publish a navigation goal to /way_point (geometry_msgs/PointStamped).

    Args:
        goal_id:  Unique ID for this goal (for tracking).
        x, y:     Map coordinates to navigate to.
        landmark: Human-readable name (informational only, coords are canonical).

    Returns:
        {"goal_id": ..., "status": "success"|"failed", "message": ...}
    """
    target = landmark if landmark else f'({x:.2f}, {y:.2f})'
    print(f'  [navigate] goal={goal_id[:8]}  target={target}  x={x:.2f} y={y:.2f}')

    try:
        msg = PointStamped()
        msg.header.stamp = _ros_node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        _waypoint_pub.publish(msg)
        print(f'  [navigate] published /way_point x={x:.3f} y={y:.3f}')
    except Exception as e:
        return {
            'goal_id': goal_id,
            'status': 'failed',
            'message': f'Failed to publish /way_point: {e}',
        }

    return {
        'goal_id': goal_id,
        'status': 'success',
        'message': f'Goal sent to {target} (x={x:.2f}, y={y:.2f})',
    }


def handle_spin(goal_id: str, angle_deg: float) -> dict:
    """
    Handle a spin-in-place command.

    TODO: Publish geometry_msgs/Twist to /cmd_vel with angular.z,
    monitor TF for cumulative rotation, stop when target angle reached.

    Args:
        goal_id:   Unique ID.
        angle_deg: Degrees to rotate. Positive = CCW, negative = CW.

    Returns:
        {"goal_id": ..., "status": "success"|"failed", "message": ...}
    """
    print(f'  [spin] goal={goal_id[:8]}  angle={angle_deg:.1f}°')

    # --- Stub: simulate spin ---
    time.sleep(0.5)

    return {
        'goal_id': goal_id,
        'status': 'success',
        'message': f'Rotated {angle_deg:.1f}°',
    }


def handle_move(goal_id: str, distance_m: float) -> dict:
    """
    Handle a straight-line move command.

    TODO: Publish geometry_msgs/Twist to /cmd_vel with linear.x,
    monitor TF for cumulative displacement, stop when target reached.

    Args:
        goal_id:    Unique ID.
        distance_m: Metres to move. Positive = forward, negative = backward.

    Returns:
        {"goal_id": ..., "status": "success"|"failed", "message": ...}
    """
    print(f'  [move] goal={goal_id[:8]}  distance={distance_m:.2f}m')

    # --- Stub: simulate move ---
    time.sleep(0.5)

    return {
        'goal_id': goal_id,
        'status': 'success',
        'message': f'Moved {distance_m:.2f}m',
    }


def handle_get_current_objects(goal_id: str) -> dict:
    """
    Return objects currently visible to the camera (live, no stale entries).

    TODO: Query the camera detection node for recent detections,
    filter by timestamp (e.g. last 2 seconds).

    Returns:
        {"goal_id": ..., "status": "success", "objects": [...]}
        Each object: {"name": "cup", "x": 1.2, "y": -0.5, "confidence": 0.92}
    """
    print(f'  [get_current_objects] goal={goal_id[:8]}')

    # --- Stub: return empty ---
    return {
        'goal_id': goal_id,
        'status': 'success',
        'objects': [],
        'message': 'No objects currently detected',
    }


def handle_get_scan(goal_id: str, min_dist: float) -> dict:
    """
    Return LiDAR obstacle distances in 8 directional sectors.

    TODO: Subscribe to /scan (sensor_msgs/LaserScan), bin ranges into
    8 sectors, compute min distance per sector.

    Args:
        min_dist: Threshold (metres) — directions with obstacles closer
                  than this are considered unsafe.

    Returns:
        {"goal_id": ..., "status": "success",
         "sectors": {"front": 2.1, "front_left": 1.5, ...},
         "min_distance": 0.8, "min_direction": "right",
         "safe_directions": ["front", "front_left", "left", ...]}
    """
    print(f'  [get_scan] goal={goal_id[:8]}  min_dist={min_dist:.1f}m')

    # --- Stub: return max range for all sectors ---
    sectors = {
        'front': 10.0, 'front_left': 10.0, 'left': 10.0, 'back_left': 10.0,
        'back': 10.0, 'back_right': 10.0, 'right': 10.0, 'front_right': 10.0,
    }
    safe = [name for name, dist in sectors.items() if dist >= min_dist]

    return {
        'goal_id': goal_id,
        'status': 'success',
        'sectors': sectors,
        'min_distance': min(sectors.values()),
        'min_direction': min(sectors, key=sectors.get),
        'safe_directions': safe,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch(msg: dict) -> dict:
    """Route an incoming JSON message to the appropriate handler."""
    goal_id = msg.get('goal_id', 'unknown')
    command_type = msg.get('command_type')

    if command_type == 'spin':
        return handle_spin(goal_id, float(msg.get('angle_deg', 0)))

    elif command_type == 'move':
        return handle_move(goal_id, float(msg.get('distance_m', 0)))

    elif command_type == 'get_current_objects':
        return handle_get_current_objects(goal_id)

    elif command_type == 'get_scan':
        return handle_get_scan(goal_id, float(msg.get('min_dist', 1.0)))

    elif 'x' in msg and 'y' in msg:
        # Navigation goal (no command_type)
        return handle_navigate(
            goal_id,
            float(msg['x']),
            float(msg['y']),
            msg.get('landmark', ''),
        )

    else:
        return {
            'goal_id': goal_id,
            'status': 'failed',
            'message': f'Unknown message format: {list(msg.keys())}',
        }


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ZMQ bridge node (robot side)')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ REP port (default: 5555)')
    args = parser.parse_args()

    init_ros()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    addr = f'tcp://0.0.0.0:{args.port}'
    sock.bind(addr)

    print(f'==> zmq_bridge_node listening on {addr}')
    print(f'    Handles: navigate, spin, move, get_current_objects, get_scan')
    print(f'    Press Ctrl+C to stop.\n')

    try:
        while True:
            raw = sock.recv_string()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                sock.send_string(json.dumps({
                    'status': 'failed',
                    'message': f'Invalid JSON: {raw[:200]}',
                }))
                continue

            cmd = msg.get('command_type', 'navigate')
            print(f'[recv] command_type={cmd}  goal_id={msg.get("goal_id", "?")[:8]}')

            reply = dispatch(msg)
            sock.send_string(json.dumps(reply))

            print(f'[sent] status={reply.get("status")}  message={reply.get("message", "")}\n')

    except KeyboardInterrupt:
        print('\nShutting down.')
    finally:
        sock.close()
        ctx.term()
        shutdown_ros()


if __name__ == '__main__':
    main()
