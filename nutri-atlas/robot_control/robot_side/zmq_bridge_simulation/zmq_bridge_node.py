"""
ZMQ ↔ ROS2 bridge node (runs on the robot onboard device).

Interfaces with the CMU ARPL autonomy stack (autonomy_stack_go2):
  - Publishes PointStamped  → /goal_point   (long-range goal for far_planner, at 1 Hz)
  - Publishes Joy           → /joy          (axes[2]=-1 keeps autonomyMode=True in stack)
  - Publishes TwistStamped  → /cmd_vel      (direct velocity for spin/move primitives)
  - Subscribes PointCloud2  ← /sensor_scan        (latest LiDAR scan, cached for get_scan queries)
  - Subscribes String       ← /detected_objects   (latest YOLO detections, cached for get_current_objects)
  - Reads current pose      ← TF tree             (transform from 'map' to 'vehicle')

Topic chain inside the stack:
  /goal_point → far_planner → /way_point + /joy → localPlanner → /path → pathFollower → /cmd_vel → robot

Architecture:
  - Main thread : rclpy.spin(node) — keeps TF buffer fresh, handles ROS2 callbacks
  - ZMQ thread  : recv → dispatch → send → recv ... (never calls spin_once)

Commands (JSON over ZMQ):
  navigate  {"command_type": "navigate", "goal_id": "...", "x": 1.0, "y": 2.0}
  spin      {"command_type": "spin",     "goal_id": "...", "angle_deg": 90.0}
  move      {"command_type": "move",     "goal_id": "...", "distance_m": 1.5}
  get_scan            {"command_type": "get_scan",            "goal_id": "...", "min_dist": 1.0}
                      → 8-sector min obstacle distances; safe_directions where dist > min_dist
  get_current_objects {"command_type": "get_current_objects", "goal_id": "...", "max_cache_age": 5.0}
                      → objects seen in the most recent /detected_objects topic message

  command_type defaults to "navigate" for backward compatibility.

Usage:
    python zmq_bridge_node.py [--port 5555] [--nav-timeout 60] [--goal-xy-radius 0.5]
                              [--spin-kp 1.5] [--move-kp 0.8]
                              [--spin-threshold-deg 3.0] [--move-threshold-m 0.05]

Environment variables (alternative to CLI args):
    ZMQ_PORT              default 5555
    NAV_TIMEOUT           default 60   (seconds)
    GOAL_XY_RADIUS        default 0.5  (metres)
"""
import argparse
import json
import math
import os
import signal
import threading
import time

import struct

import zmq
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped, TwistStamped
from sensor_msgs.msg import Joy, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


# axes[2] = -1.0 enables autonomyMode in localPlanner and pathFollower
_AUTONOMY_JOY_AXES    = [0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
_AUTONOMY_JOY_BUTTONS = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

_MAP_FRAME     = 'map'
_VEHICLE_FRAME = 'vehicle'

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
# Points closer than this are ignored (robot body hits)
_SCAN_MIN_RANGE = 0.15   # metres
# Points farther than this are clipped (report as max range)
_SCAN_MAX_RANGE = 10.0   # metres


class ZMQBridgeNode(Node):
    def __init__(self, zmq_port: int, nav_timeout: float, goal_xy_radius: float,
                 spin_kp: float, move_kp: float,
                 spin_threshold_deg: float, move_threshold_m: float,
                 objects_file: str):
        super().__init__('zmq_bridge_node')

        self._nav_timeout       = nav_timeout
        self._goal_xy_radius    = goal_xy_radius
        self._spin_kp           = spin_kp
        self._move_kp           = move_kp
        self._spin_threshold    = math.radians(spin_threshold_deg)
        self._move_threshold    = move_threshold_m

        self._objects_file = objects_file

        # Flag: when True, _publish_autonomy_enable() is a no-op so the nav
        # stack stops fighting our direct cmd_vel commands during spin/move.
        self._autonomy_paused = False

        # --- TF2: kept fresh by rclpy.spin in main thread ---
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- ROS2 publishers ---
        self._goal_pub     = self.create_publisher(PointStamped,  '/goal_point', 10)
        self._joy_pub      = self.create_publisher(Joy,           '/joy',        10)
        self._cmd_vel_pub  = self.create_publisher(TwistStamped,  '/cmd_vel',    10)

        # --- LiDAR scan cache (written by ROS callback, read by ZMQ thread) ---
        self._latest_scan      = None   # most recent PointCloud2 message
        self._latest_scan_lock = threading.Lock()
        self.create_subscription(PointCloud2, '/sensor_scan', self._scan_callback, 10)

        # --- Detected objects cache (written by ROS callback, read by ZMQ thread) ---
        # Stores (recv_time: float, data: str) from the /detected_objects topic.
        # data format: "key:px,py\nkey:px,py" — published only on positive detections.
        self._latest_detections      = None
        self._latest_detections_lock = threading.Lock()
        self.create_subscription(String, '/detected_objects', self._detections_callback, 10)

        # --- ZMQ REP socket ---
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.RCVTIMEO, 500)  # unblock every 500ms to check rclpy.ok()
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f'tcp://*:{zmq_port}')
        self.get_logger().info(f'ZMQ REP bound on tcp://*:{zmq_port}')

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
                continue   # 500ms timeout fired, loop back and check rclpy.ok()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                self.get_logger().error(f'Invalid JSON: {e}')
                self._sock.send_string(json.dumps(
                    {'status': 'failed', 'message': f'Invalid JSON: {e}'}
                ))
                continue

            goal_id      = msg.get('goal_id', 'unknown')
            command_type = msg.get('command_type', 'navigate')

            # get_current_objects does not require TF — dispatch before the TF check
            if command_type == 'get_current_objects':
                self.get_logger().info(f'[{goal_id}] get_current_objects')
                reply = self._handle_get_current_objects(goal_id)
                self._sock.send_string(json.dumps(reply))
                self.get_logger().info(f'Reply sent: {reply}')
                continue

            if not self._wait_for_tf(timeout=5.0):
                self._sock.send_string(json.dumps({
                    'goal_id': goal_id,
                    'status':  'failed',
                    'message': f'TF {_MAP_FRAME}→{_VEHICLE_FRAME} not available',
                }))
                continue

            if command_type == 'get_scan':
                min_dist = float(msg.get('min_dist', 1.0))
                self.get_logger().info(f'[{goal_id}] get_scan (min_dist={min_dist}m)')
                reply = self._handle_get_scan(goal_id, min_dist)

            elif command_type == 'spin':
                angle_deg = float(msg.get('angle_deg', 0.0))
                self.get_logger().info(f'[{goal_id}] spin {angle_deg:.1f}°')
                reply = self._handle_spin(goal_id, angle_deg)

            elif command_type == 'move':
                distance_m = float(msg.get('distance_m', 0.0))
                self.get_logger().info(f'[{goal_id}] move {distance_m:.2f}m')
                reply = self._handle_move(goal_id, distance_m)

            else:  # 'navigate' (default, backward compatible)
                gx = float(msg.get('x', 0.0))
                gy = float(msg.get('y', 0.0))
                self.get_logger().info(
                    f'[{goal_id}] navigate landmark={msg.get("landmark")} x={gx:.2f} y={gy:.2f}'
                )
                reply = self._navigate(goal_id, gx, gy)

            self._sock.send_string(json.dumps(reply))
            self.get_logger().info(f'Reply sent: {reply}')

    # ------------------------------------------------------------------
    # Navigation loop: publishes goal + autonomy enable until arrival
    # ------------------------------------------------------------------
    def _navigate(self, goal_id: str, gx: float, gy: float) -> dict:
        deadline = time.time() + self._nav_timeout
        last_pub = 0.0

        while time.time() < deadline:
            now = time.time()
            # Publish goal at 1 Hz as keepalive (far_planner needs the goal repeated
            # in case it restarted or cleared its internal goal state)
            if now - last_pub >= 1.0:
                self._publish_goal_point(gx, gy)
                self._publish_autonomy_enable()
                last_pub = now

            pose = self._get_pose()
            if pose:
                dist = math.hypot(gx - pose[0], gy - pose[1])
                if dist < self._goal_xy_radius:
                    return {
                        'goal_id': goal_id,
                        'status':  'success',
                        'message': f'Reached goal (dist={dist:.2f}m)',
                    }

            time.sleep(0.05)

        return {
            'goal_id': goal_id,
            'status':  'timeout',
            'message': f'Goal not reached within {self._nav_timeout}s',
        }

    # ------------------------------------------------------------------
    # Spin: pure in-place rotation using P-controller on /cmd_vel
    # ------------------------------------------------------------------
    def _handle_spin(self, goal_id: str, angle_deg: float) -> dict:
        pose = self._get_pose_full()
        if pose is None:
            return {'goal_id': goal_id, 'status': 'failed', 'message': 'TF not available'}

        _, _, _, yaw0 = pose
        target_yaw = yaw0 + math.radians(angle_deg)

        self._autonomy_paused = True
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
            self._autonomy_paused = False

        if time.time() >= deadline:
            return {'goal_id': goal_id, 'status': 'timeout',
                    'message': f'Spin not completed within {_SPIN_TIMEOUT}s'}

        return {'goal_id': goal_id, 'status': 'success',
                'message': f'Rotated {angle_deg:.1f}°'}

    # ------------------------------------------------------------------
    # Move: straight-line motion using P-controller on /cmd_vel
    # ------------------------------------------------------------------
    def _handle_move(self, goal_id: str, distance_m: float) -> dict:
        pose = self._get_pose_full()
        if pose is None:
            return {'goal_id': goal_id, 'status': 'failed', 'message': 'TF not available'}

        x0, y0, _, yaw0 = pose
        tx = x0 + distance_m * math.cos(yaw0)
        ty = y0 + distance_m * math.sin(yaw0)

        self._autonomy_paused = True
        deadline = time.time() + _MOVE_TIMEOUT
        dt = 1.0 / _CTRL_HZ

        try:
            while time.time() < deadline:
                pose = self._get_pose_full()
                if pose is None:
                    time.sleep(dt)
                    continue

                cx, cy, _, _ = pose
                # Signed error: project remaining vector onto initial heading
                dx, dy = tx - cx, ty - cy
                err = dx * math.cos(yaw0) + dy * math.sin(yaw0)

                if abs(err) < self._move_threshold:
                    break

                vx = max(-_MAX_VX, min(_MAX_VX, self._move_kp * err))
                self._publish_cmd_vel(linear_x=vx, angular_z=0.0)
                time.sleep(dt)
        finally:
            self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)  # always stop
            self._autonomy_paused = False

        if time.time() >= deadline:
            return {'goal_id': goal_id, 'status': 'timeout',
                    'message': f'Move not completed within {_MOVE_TIMEOUT}s'}

        return {'goal_id': goal_id, 'status': 'success',
                'message': f'Moved {distance_m:.2f}m'}

    # ------------------------------------------------------------------
    # LiDAR scan callback — caches latest PointCloud2 (called from ROS spin thread)
    # ------------------------------------------------------------------
    def _scan_callback(self, msg: PointCloud2):
        with self._latest_scan_lock:
            self._latest_scan = msg

    # ------------------------------------------------------------------
    # Detected objects callback — caches latest /detected_objects message
    # ------------------------------------------------------------------
    def _detections_callback(self, msg: String):
        with self._latest_detections_lock:
            self._latest_detections = (time.time(), msg.data)

    # ------------------------------------------------------------------
    # get_current_objects handler — returns objects from the most recent
    # /detected_objects topic message, cross-referenced with JSON for
    # conf/label. Does not depend on file timestamps.
    # ------------------------------------------------------------------
    def _handle_get_current_objects(self, goal_id: str) -> dict:
        with self._latest_detections_lock:
            snap = self._latest_detections

        if snap is None:
            return {'goal_id': goal_id, 'status': 'ok', 'objects': {},
                    'message': 'No detections received yet on /detected_objects'}

        _, data = snap

        # Parse "key:px,py\nkey:px,py" format
        current = {}
        for line in data.strip().split('\n'):
            if ':' not in line:
                continue
            key, coords = line.split(':', 1)
            px_str, py_str = coords.split(',', 1)
            try:
                current[key.strip()] = {'px': float(px_str), 'py': float(py_str)}
            except ValueError:
                continue

        # Cross-reference JSON for conf/label
        try:
            with open(self._objects_file) as f:
                full_map = json.load(f)
        except Exception:
            full_map = {}

        objects = {}
        for key, pos in current.items():
            entry = full_map.get(key, {})
            objects[key] = {
                'px':    pos['px'],
                'py':    pos['py'],
                'conf':  entry.get('conf'),
                'label': entry.get('label', key),
            }

        return {'goal_id': goal_id, 'status': 'ok', 'objects': objects}

    # ------------------------------------------------------------------
    # get_scan handler — returns 8-sector min obstacle distances
    # ------------------------------------------------------------------
    def _handle_get_scan(self, goal_id: str, min_dist: float) -> dict:
        with self._latest_scan_lock:
            scan = self._latest_scan

        if scan is None:
            return {
                'goal_id': goal_id,
                'status':  'failed',
                'message': 'No scan received yet on /sensor_scan',
            }

        sectors = self._process_scan(scan)
        safe = [name for name, d in sectors.items() if d > min_dist]
        min_d = min(sectors.values())
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
    # Process PointCloud2 into 8-sector minimum distances.
    # /sensor_scan is in sensor frame: x=forward, y=left.
    # Sectors are 45° wide, centred on cardinal/intercardinal directions.
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

        n_points = len(data) // step
        for i in range(n_points):
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

            # azimuth: 0=front(+x), +90°=left(+y); range [-180, 180]
            azimuth = math.degrees(math.atan2(y, x))
            # normalise to [0, 360) then map to sector index
            sector_idx = int((azimuth % 360.0 + 22.5) % 360.0 / 45.0) % 8
            name = _SECTOR_NAMES[sector_idx]
            if dist < sectors[name]:
                sectors[name] = dist

        return {k: round(v, 3) for k, v in sectors.items()}

    # ------------------------------------------------------------------
    # Read current pose from TF tree (map → vehicle)
    # Returns (x, y, z) — z is map-frame height, used for waypoint z
    # ------------------------------------------------------------------
    def _get_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                _MAP_FRAME, _VEHICLE_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            t = tf.transform.translation
            return t.x, t.y, t.z
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ------------------------------------------------------------------
    # Read current pose + yaw from TF tree (map → vehicle)
    # Returns (x, y, z, yaw)
    # ------------------------------------------------------------------
    def _get_pose_full(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                _MAP_FRAME, _VEHICLE_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return t.x, t.y, t.z, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ------------------------------------------------------------------
    # Publish PointStamped to /goal_point (map frame) for far_planner
    # ------------------------------------------------------------------
    def _publish_goal_point(self, gx: float, gy: float):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = _MAP_FRAME
        msg.point.x = gx
        msg.point.y = gy
        msg.point.z = 0.0
        self._goal_pub.publish(msg)

    # ------------------------------------------------------------------
    # Publish Joy with axes[2]=-1.0 to keep autonomyMode=True in the stack
    # No-op while autonomy is paused (during spin/move).
    # ------------------------------------------------------------------
    def _publish_autonomy_enable(self):
        if self._autonomy_paused:
            return
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes    = _AUTONOMY_JOY_AXES
        msg.buttons = _AUTONOMY_JOY_BUTTONS
        self._joy_pub.publish(msg)

    # ------------------------------------------------------------------
    # Publish TwistStamped to /cmd_vel (used during spin/move)
    # ------------------------------------------------------------------
    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = _VEHICLE_FRAME
        msg.twist.linear.x  = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Wait until map→vehicle TF is available
    # ------------------------------------------------------------------
    def _wait_for_tf(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._get_pose() is not None:
                return True
            time.sleep(0.1)
        return False

    def destroy_node(self):
        self._sock.close()
        self._ctx.term()
        super().destroy_node()


# ----------------------------------------------------------------------
def main():
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(description='ZMQ ↔ ROS2 navigation bridge')
    parser.add_argument('--port',               type=int,   default=int(os.environ.get('ZMQ_PORT',         5555)))
    parser.add_argument('--nav-timeout',        type=float, default=float(os.environ.get('NAV_TIMEOUT',    60)))
    parser.add_argument('--goal-xy-radius',     type=float, default=float(os.environ.get('GOAL_XY_RADIUS', 0.5)))
    parser.add_argument('--spin-kp',            type=float, default=1.5)
    parser.add_argument('--move-kp',            type=float, default=0.8)
    parser.add_argument('--spin-threshold-deg', type=float, default=3.0)
    parser.add_argument('--move-threshold-m',   type=float, default=0.05)
    parser.add_argument('--objects-file',       type=str,
        default=os.path.expanduser('~/Go2/autonomy_stack_go2/detected_objects.json'))
    args = parser.parse_args()

    rclpy.init()
    node = ZMQBridgeNode(
        zmq_port=args.port,
        nav_timeout=args.nav_timeout,
        goal_xy_radius=args.goal_xy_radius,
        spin_kp=args.spin_kp,
        move_kp=args.move_kp,
        spin_threshold_deg=args.spin_threshold_deg,
        move_threshold_m=args.move_threshold_m,
        objects_file=args.objects_file,
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
