"""
ZMQ REP server running on the robot side.

Receives JSON commands from robot_assistant.py (via ZMQNavClient) and dispatches
them to the appropriate ROS2 topic. Blocks until the action completes, then
sends back a JSON reply.

Architecture:
  - Main thread : rclpy.spin(node) — keeps TF buffer fresh, handles topic callbacks
  - ZMQ thread  : recv → dispatch → send → recv ... (never calls spin_once)

Message types handled:

1. Navigation goal  — has 'x' and 'y', no 'command_type'
   {"goal_id": "...", "landmark": "kitchen", "x": 1.5, "y": 2.3}
   → Publish geometry_msgs/PoseStamped to /goal_pose (Nav2-compatible)

2. Spin command     — command_type == 'spin'
   {"goal_id": "...", "command_type": "spin", "angle_deg": 120}
   → P-controller on /cmd_vel with TF feedback (blocks until done)

3. Move command     — command_type == 'move'
   {"goal_id": "...", "command_type": "move", "distance_m": 1.5}
   → P-controller on /cmd_vel with TF feedback (blocks until done)

4. Current objects  — command_type == 'get_current_objects'
   {"goal_id": "...", "command_type": "get_current_objects"}
   → Return most recent /detected_objects topic message

5. LiDAR scan      — command_type == 'get_scan'
   {"goal_id": "...", "command_type": "get_scan", "min_dist": 1.0}
   → Return 8-sector obstacle distances from /sensor_scan

Usage:
    python zmq_bridge_node.py [--port 5555]
                              [--spin-kp 1.5] [--move-kp 0.8]
                              [--spin-threshold-deg 3.0] [--move-threshold-m 0.05]

Environment variables:
    ZMQ_PORT   default 5555
"""
import argparse
import json
import math
import os
import signal
import struct
import threading
import time

import zmq
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


_MAP_FRAME = 'map'

# P-controller limits
_MAX_VYAW = 0.8   # rad/s
_MAX_VX   = 0.5   # m/s

# Motion timeouts
_SPIN_TIMEOUT = 30.0   # seconds
_MOVE_TIMEOUT = 30.0   # seconds

# cmd_vel publish rate during spin/move
_CTRL_HZ = 20.0

# LiDAR sector names in CCW order starting from front, each spanning 45°
_SECTOR_NAMES = ['front', 'front_left', 'left', 'back_left',
                 'back', 'back_right', 'right', 'front_right']
_SCAN_MIN_RANGE = 0.15   # metres — ignore points closer than this (robot body)
_SCAN_MAX_RANGE = 10.0   # metres — clip at this distance


class ZMQBridgeNode(Node):
    def __init__(self, zmq_port: int,
                 spin_kp: float, move_kp: float,
                 spin_threshold_deg: float, move_threshold_m: float):
        super().__init__('zmq_bridge_node')

        self._spin_kp        = spin_kp
        self._move_kp        = move_kp
        self._spin_threshold = math.radians(spin_threshold_deg)
        self._move_threshold = move_threshold_m
        # Prefer standard Nav2 base frame; keep fallbacks for compatibility.
        self._base_frame_candidates = ['base_link', 'base', 'vehicle']
        self._active_base_frame = self._base_frame_candidates[0]

        # --- TF2: kept fresh by rclpy.spin in main thread ---
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- ROS2 publishers ---
        self._goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self._cmd_vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- LiDAR scan cache (written by ROS callback, read by ZMQ thread) ---
        self._latest_scan      = None
        self._latest_scan_lock = threading.Lock()
        self.create_subscription(PointCloud2, '/sensor_scan', self._scan_callback, 10)

        # --- Detected objects cache (written by ROS callback, read by ZMQ thread) ---
        # data format published on /detected_objects: "key:px,py\nkey:px,py"
        self._latest_detections      = None
        self._latest_detections_lock = threading.Lock()
        self.create_subscription(String, '/detected_objects', self._detections_callback, 10)

        # --- ZMQ REP socket ---
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.RCVTIMEO, 500)  # unblock every 500ms to check rclpy.ok()
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f'tcp://0.0.0.0:{zmq_port}')
        self.get_logger().info(f'ZMQ REP bound on tcp://0.0.0.0:{zmq_port}')

        # --- ZMQ thread: runs independently from ROS2 spin ---
        self._zmq_thread = threading.Thread(target=self._zmq_loop, daemon=True)
        self._zmq_thread.start()

    # ------------------------------------------------------------------
    # ZMQ thread: strict recv → dispatch → send → recv loop
    # ------------------------------------------------------------------
    def _zmq_loop(self):
        while rclpy.ok():
            try:
                raw = self._sock.recv_string()
            except zmq.Again:
                continue   # 500ms timeout — loop back and check rclpy.ok()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                self._sock.send_string(json.dumps(
                    {'status': 'failed', 'message': f'Invalid JSON: {e}'}
                ))
                continue

            reply = self._dispatch(msg)
            self._sock.send_string(json.dumps(reply))
            self.get_logger().info(f'Reply: status={reply.get("status")} msg={reply.get("message", "")}')

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def _dispatch(self, msg: dict) -> dict:
        goal_id      = msg.get('goal_id', 'unknown')
        command_type = msg.get('command_type')

        if command_type == 'spin':
            self.get_logger().info(f'[{goal_id[:8]}] spin {msg.get("angle_deg")}°')
            return self._handle_spin(goal_id, float(msg.get('angle_deg', 0.0)))

        elif command_type == 'move':
            self.get_logger().info(f'[{goal_id[:8]}] move {msg.get("distance_m")}m')
            return self._handle_move(goal_id, float(msg.get('distance_m', 0.0)))

        elif command_type == 'get_current_objects':
            self.get_logger().info(f'[{goal_id[:8]}] get_current_objects')
            return self._handle_get_current_objects(goal_id)

        elif command_type == 'get_scan':
            min_dist = float(msg.get('min_dist', 1.0))
            self.get_logger().info(f'[{goal_id[:8]}] get_scan (min_dist={min_dist}m)')
            return self._handle_get_scan(goal_id, min_dist)

        elif 'x' in msg and 'y' in msg:
            gx, gy = float(msg['x']), float(msg['y'])
            self.get_logger().info(
                f'[{goal_id[:8]}] navigate landmark={msg.get("landmark")} x={gx:.2f} y={gy:.2f}'
            )
            return self._handle_navigate(goal_id, gx, gy, msg.get('landmark', ''))

        else:
            return {
                'goal_id': goal_id,
                'status':  'failed',
                'message': f'Unknown message format: {list(msg.keys())}',
            }

    # ------------------------------------------------------------------
    # Navigate: publish PoseStamped to /goal_pose (Nav2).
    # No TF required — Nav2 handles arrival.
    # ------------------------------------------------------------------
    def _handle_navigate(self, goal_id: str, x: float, y: float, landmark: str) -> dict:
        target = landmark if landmark else f'({x:.2f}, {y:.2f})'
        try:
            msg = PoseStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = _MAP_FRAME
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = 0.0
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
            self._goal_pose_pub.publish(msg)
            self.get_logger().info(f'Published /goal_pose x={x:.3f} y={y:.3f}')
        except Exception as e:
            return {'goal_id': goal_id, 'status': 'failed',
                    'message': f'Failed to publish /goal_pose: {e}'}

        return {'goal_id': goal_id, 'status': 'success',
                'message': f'Goal sent to {target} (x={x:.2f}, y={y:.2f})'}

    # ------------------------------------------------------------------
    # Spin: P-controller on /cmd_vel using TF yaw feedback
    # ------------------------------------------------------------------
    def _handle_spin(self, goal_id: str, angle_deg: float) -> dict:
        pose = self._get_pose_full()
        if pose is None:
            return {'goal_id': goal_id, 'status': 'failed', 'message': 'TF not available'}

        _, _, _, yaw0 = pose
        target_yaw = yaw0 + math.radians(angle_deg)

        deadline = time.time() + _SPIN_TIMEOUT
        dt = 1.0 / _CTRL_HZ

        try:
            while time.time() < deadline:
                pose = self._get_pose_full()
                if pose is None:
                    time.sleep(dt)
                    continue

                _, _, _, current_yaw = pose
                err = math.atan2(
                    math.sin(target_yaw - current_yaw),
                    math.cos(target_yaw - current_yaw),
                )

                if abs(err) < self._spin_threshold:
                    break

                vyaw = max(-_MAX_VYAW, min(_MAX_VYAW, self._spin_kp * err))
                self._publish_cmd_vel(linear_x=0.0, angular_z=vyaw)
                time.sleep(dt)
        finally:
            self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)  # always stop

        if time.time() >= deadline:
            return {'goal_id': goal_id, 'status': 'timeout',
                    'message': f'Spin not completed within {_SPIN_TIMEOUT}s'}

        return {'goal_id': goal_id, 'status': 'success',
                'message': f'Rotated {angle_deg:.1f}°'}

    # ------------------------------------------------------------------
    # Move: P-controller on /cmd_vel using TF position feedback
    # ------------------------------------------------------------------
    def _handle_move(self, goal_id: str, distance_m: float) -> dict:
        pose = self._get_pose_full()
        if pose is None:
            return {'goal_id': goal_id, 'status': 'failed', 'message': 'TF not available'}

        x0, y0, _, yaw0 = pose
        tx = x0 + distance_m * math.cos(yaw0)
        ty = y0 + distance_m * math.sin(yaw0)

        deadline = time.time() + _MOVE_TIMEOUT
        dt = 1.0 / _CTRL_HZ

        try:
            while time.time() < deadline:
                pose = self._get_pose_full()
                if pose is None:
                    time.sleep(dt)
                    continue

                cx, cy, _, _ = pose
                dx, dy = tx - cx, ty - cy
                # Signed error: project remaining vector onto initial heading
                err = dx * math.cos(yaw0) + dy * math.sin(yaw0)

                if abs(err) < self._move_threshold:
                    break

                vx = max(-_MAX_VX, min(_MAX_VX, self._move_kp * err))
                self._publish_cmd_vel(linear_x=vx, angular_z=0.0)
                time.sleep(dt)
        finally:
            self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)  # always stop

        if time.time() >= deadline:
            return {'goal_id': goal_id, 'status': 'timeout',
                    'message': f'Move not completed within {_MOVE_TIMEOUT}s'}

        return {'goal_id': goal_id, 'status': 'success',
                'message': f'Moved {distance_m:.2f}m'}

    # ------------------------------------------------------------------
    # get_current_objects: return most recent /detected_objects message
    # ------------------------------------------------------------------
    def _handle_get_current_objects(self, goal_id: str) -> dict:
        with self._latest_detections_lock:
            snap = self._latest_detections

        if snap is None:
            return {'goal_id': goal_id, 'status': 'ok', 'objects': {},
                    'message': 'No detections received yet on /detected_objects'}

        _, data = snap

        # Parse "key:px,py\nkey:px,py" format
        objects = {}
        for line in data.strip().split('\n'):
            if ':' not in line:
                continue
            key, coords = line.split(':', 1)
            try:
                px_str, py_str = coords.split(',', 1)
                objects[key.strip()] = {'px': float(px_str), 'py': float(py_str)}
            except ValueError:
                continue

        return {'goal_id': goal_id, 'status': 'ok', 'objects': objects}

    # ------------------------------------------------------------------
    # get_scan: return 8-sector min obstacle distances from /sensor_scan
    # ------------------------------------------------------------------
    def _handle_get_scan(self, goal_id: str, min_dist: float) -> dict:
        with self._latest_scan_lock:
            scan = self._latest_scan

        if scan is None:
            return {'goal_id': goal_id, 'status': 'failed',
                    'message': 'No scan received yet on /sensor_scan'}

        sectors = self._process_scan(scan)
        safe    = [name for name, d in sectors.items() if d > min_dist]
        min_d   = min(sectors.values())
        min_dir = min(sectors, key=sectors.get)

        return {
            'goal_id':         goal_id,
            'status':          'ok',
            'sectors':         sectors,
            'min_distance':    round(min_d, 3),
            'min_direction':   min_dir,
            'safe_directions': safe,
        }

    # ------------------------------------------------------------------
    # ROS topic callbacks (called from rclpy.spin thread)
    # ------------------------------------------------------------------
    def _scan_callback(self, msg: PointCloud2):
        with self._latest_scan_lock:
            self._latest_scan = msg

    def _detections_callback(self, msg: String):
        with self._latest_detections_lock:
            self._latest_detections = (time.time(), msg.data)

    # ------------------------------------------------------------------
    # Process PointCloud2 into 8-sector minimum distances.
    # /sensor_scan is in sensor frame: x=forward, y=left.
    # ------------------------------------------------------------------
    def _process_scan(self, msg: PointCloud2) -> dict:
        sectors = {name: _SCAN_MAX_RANGE for name in _SECTOR_NAMES}

        field_offsets = {f.name: f.offset for f in msg.fields}
        if 'x' not in field_offsets or 'y' not in field_offsets:
            return sectors

        x_off = field_offsets['x']
        y_off = field_offsets['y']
        step  = msg.point_step
        data  = msg.data

        for i in range(len(data) // step):
            base = i * step
            try:
                x = struct.unpack_from('<f', data, base + x_off)[0]
                y = struct.unpack_from('<f', data, base + y_off)[0]
            except struct.error:
                continue

            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            dist = math.hypot(x, y)
            if dist < _SCAN_MIN_RANGE or dist > _SCAN_MAX_RANGE:
                continue

            # azimuth: 0=front(+x), +90°=left(+y)
            azimuth    = math.degrees(math.atan2(y, x))
            sector_idx = int((azimuth % 360.0 + 22.5) % 360.0 / 45.0) % 8
            name       = _SECTOR_NAMES[sector_idx]
            if dist < sectors[name]:
                sectors[name] = dist

        return {k: round(v, 3) for k, v in sectors.items()}

    # ------------------------------------------------------------------
    # TF helpers (called from ZMQ thread — TF buffer is thread-safe)
    # ------------------------------------------------------------------
    def _get_pose_full(self):
        """Returns (x, y, z, yaw) from map->base frame, or None if unavailable."""
        for frame in self._base_frame_candidates:
            try:
                tf = self._tf_buffer.lookup_transform(
                    _MAP_FRAME, frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1),
                )
                self._active_base_frame = frame
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                return t.x, t.y, t.z, yaw
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
        return None

    # ------------------------------------------------------------------
    # Publish Twist to /cmd_vel
    # ------------------------------------------------------------------
    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x  = linear_x
        msg.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)

    def destroy_node(self):
        self._sock.close()
        self._ctx.term()
        super().destroy_node()


# ----------------------------------------------------------------------
def main():
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(description='ZMQ bridge node (robot side)')
    parser.add_argument('--port',               type=int,   default=int(os.environ.get('ZMQ_PORT', 5555)))
    parser.add_argument('--spin-kp',            type=float, default=1.5)
    parser.add_argument('--move-kp',            type=float, default=0.8)
    parser.add_argument('--spin-threshold-deg', type=float, default=3.0)
    parser.add_argument('--move-threshold-m',   type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    node = ZMQBridgeNode(
        zmq_port=args.port,
        spin_kp=args.spin_kp,
        move_kp=args.move_kp,
        spin_threshold_deg=args.spin_threshold_deg,
        move_threshold_m=args.move_threshold_m,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
