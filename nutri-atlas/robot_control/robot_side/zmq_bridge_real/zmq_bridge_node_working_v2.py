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
   → Send nav2_msgs/action/NavigateToPose goal to /navigate_to_pose

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

6. Update objects  — command_type == 'update_objects'
   {"goal_id": "...", "command_type": "update_objects",
    "objects": [{"label": "bottle", "cam_x": 0.1, "cam_y": 0.0, "cam_z": 0.8}, ...]}
   → Look up TF map → camera_link, transform each point to map frame
   → Broadcast detected_{label}_{i} as static TF frames under map
   → Store results in memory for get_detected_objects queries

7. Get detected objects — command_type == 'get_detected_objects'
   {"goal_id": "...", "command_type": "get_detected_objects"}
   → Return persistent detected_objects.json map as {frame: {px, py, label}}

8. Forget object     — command_type == 'forget_object'
   {"goal_id": "...", "command_type": "forget_object", "frame_name": "detected_bottle_0"}
   → Remove a single entry from detected_objects.json

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
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from tf2_ros import (Buffer, TransformListener, StaticTransformBroadcaster,
                     LookupException, ConnectivityException, ExtrapolationException)


_MAP_FRAME = 'map'
_DETECTED_OBJECTS_FILE = os.path.expanduser('~/detected_objects.json')
_SPATIAL_THRESHOLD = 0.5   # metres — objects closer than this with same label are duplicates


def _load_detected_objects(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_detected_objects(path: str, objects: dict):
    with open(path, 'w') as f:
        json.dump(objects, f, indent=2)

# P-controller limits
_MAX_VYAW = 0.8   # rad/s
_MIN_VYAW = 0.15  # rad/s — minimum to overcome friction/motor deadband
_MAX_VX   = 0.5   # m/s
_MIN_VX   = 0.1   # m/s — minimum to overcome friction/motor deadband

# Motion timeouts
_SPIN_TIMEOUT = 30.0   # seconds
_MOVE_TIMEOUT = 30.0   # seconds
_NAV_TIMEOUT = 120.0   # seconds

# Arrival tracking — source of truth is the robot's actual map->base pose,
# not Nav2's BT result. Nav2 in this stack is unreliable (BT SIGSEGVs, false
# successes from stale localization, aborts while the controller tail keeps
# the robot moving). Polling TF survives all three.
_ARRIVAL_THRESHOLD_M = 0.5     # success when within this distance
_NAV_POLL_S          = 0.3     # TF poll period during navigation
_NAV_STALL_S         = 15.0    # max seconds with no measurable progress
_NAV_STALL_PROG_M    = 0.05    # min distance closed to count as progress

# cmd_vel publish rate during spin/move
_CTRL_HZ = 20.0

# LiDAR sector names in CCW order starting from front, each spanning 45°
_SECTOR_NAMES = ['front', 'front_left', 'left', 'back_left',
                 'back', 'back_right', 'right', 'front_right']
_SCAN_MIN_RANGE = 0.15   # metres — ignore points closer than this (robot body)
_SCAN_MAX_RANGE = 10.0   # metres — clip at this distance


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> list:
    """Return a 3×3 rotation matrix (list of rows) from a unit quaternion."""
    return [
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ]


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

        # --- Static TF broadcaster for detected objects ---
        self._obj_tf_broadcaster = StaticTransformBroadcaster(self)

        # --- ROS2 publishers ---
        self._cmd_vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # --- LiDAR scan cache (written by ROS callback, read by ZMQ thread) ---
        self._latest_scan      = None
        self._latest_scan_lock = threading.Lock()
        self.create_subscription(PointCloud2, '/sensor_scan', self._scan_callback, 10)

        # --- Detected objects cache (written by ROS callback, read by ZMQ thread) ---
        # data format published on /detected_objects: "key:px,py\nkey:px,py"
        self._latest_detections      = None
        self._latest_detections_lock = threading.Lock()

        # --- Real-world detected objects store (written by update_objects handler) ---
        # format: {frame_name: {'px': float, 'py': float, 'label': str}}
        self._stored_objects      = {}
        self._stored_objects_lock = threading.Lock()
        self.create_subscription(String, '/detected_objects', self._detections_callback, 10)

        # --- Async navigation state (for start_navigate / check_nav_status) ---
        self._nav_goal_handle   = None   # active Nav2 goal handle
        self._nav_goal_id       = None   # ZMQ goal_id for the active navigation
        self._nav_result_future = None   # async future for Nav2 result
        self._nav_target_desc   = ''     # human-readable target description
        self._nav_target_x      = 0.0    # map x of the active goal (for arrival check)
        self._nav_target_y      = 0.0    # map y of the active goal (for arrival check)
        # Poll-based arrival tracking (used by both sync and async nav paths)
        self._nav_t_start       = 0.0
        self._nav_last_dist     = float('inf')
        self._nav_last_prog_t   = 0.0
        self._nav_lock          = threading.Lock()

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

        elif command_type == 'update_objects':
            objects = msg.get('objects', [])
            self.get_logger().info(f'[{goal_id[:8]}] update_objects ({len(objects)} objects)')
            return self._handle_update_objects(goal_id, objects)

        elif command_type == 'get_detected_objects':
            self.get_logger().info(f'[{goal_id[:8]}] get_detected_objects')
            return self._handle_get_detected_objects(goal_id)

        elif command_type == 'get_camera_pose':
            self.get_logger().info(f'[{goal_id[:8]}] get_camera_pose')
            return self._handle_get_camera_pose(goal_id)

        elif command_type == 'start_navigate':
            gx, gy = float(msg.get('x', 0)), float(msg.get('y', 0))
            self.get_logger().info(
                f'[{goal_id[:8]}] start_navigate landmark={msg.get("landmark")} x={gx:.2f} y={gy:.2f}')
            return self._handle_start_navigate(goal_id, gx, gy, msg.get('landmark', ''))

        elif command_type == 'check_nav_status':
            return self._handle_check_nav_status(goal_id)

        elif command_type == 'cancel_navigate':
            self.get_logger().info(f'[{goal_id[:8]}] cancel_navigate')
            return self._handle_cancel_navigate(goal_id)

        elif command_type == 'update_objects_map':
            objects = msg.get('objects', [])
            self.get_logger().info(f'[{goal_id[:8]}] update_objects_map ({len(objects)} objects)')
            return self._handle_update_objects_map(goal_id, objects)

        elif command_type == 'forget_object':
            frame_name = msg.get('frame_name', '')
            self.get_logger().info(f'[{goal_id[:8]}] forget_object {frame_name}')
            return self._handle_forget_object(goal_id, frame_name)

        elif command_type == 'clear_objects':
            self.get_logger().info(f'[{goal_id[:8]}] clear_objects')
            return self._handle_clear_objects(goal_id)

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
    # Navigate: call Nav2 action /navigate_to_pose and wait for result.
    # ------------------------------------------------------------------
    def _verify_arrival(self, tx, ty):
        """Return (arrived, distance_m). arrived=False if pose unavailable."""
        pose = self._get_pose_full()
        if pose is None:
            return False, float('inf')
        px, py, _pz, _yaw = pose
        dist = math.hypot(px - tx, py - ty)
        return dist <= _ARRIVAL_THRESHOLD_M, dist

    def _cancel_goal_safely(self, goal_handle) -> None:
        """Best-effort cancel of a Nav2 goal handle. Tolerates crashed Nav2."""
        if goal_handle is None:
            return
        try:
            goal_handle.cancel_goal_async()
        except Exception:
            pass

    def _poll_until_arrived(self, goal_handle, x: float, y: float,
                            target: str, goal_id: str) -> dict:
        """Poll TF until the robot reaches (x, y) or stalls.

        Outcomes:
          success — robot pose within _ARRIVAL_THRESHOLD_M of (x, y)
          failed  — no measurable progress for _NAV_STALL_S
        """
        last_prog_t = time.time()
        pose0 = self._get_pose_full()
        last_dist = math.hypot(pose0[0] - x, pose0[1] - y) if pose0 else float('inf')

        while rclpy.ok():
            time.sleep(_NAV_POLL_S)
            pose = self._get_pose_full()
            if pose is None:
                continue  # TF temporarily unavailable — wait, don't reset
            dist = math.hypot(pose[0] - x, pose[1] - y)
            if dist <= _ARRIVAL_THRESHOLD_M:
                self._cancel_goal_safely(goal_handle)
                return {
                    'goal_id': goal_id,
                    'status': 'success',
                    'message': f'Arrived at {target} (x={x:.2f}, y={y:.2f}, dist={dist:.2f} m)',
                }
            if (last_dist - dist) >= _NAV_STALL_PROG_M:
                last_prog_t = time.time()
                last_dist = dist
            if (time.time() - last_prog_t) >= _NAV_STALL_S:
                self._cancel_goal_safely(goal_handle)
                return {
                    'goal_id': goal_id,
                    'status': 'failed',
                    'message': (f'No progress toward {target} for {_NAV_STALL_S:.0f}s '
                                f'(last dist={dist:.2f} m)'),
                }

        # rclpy shutting down
        self._cancel_goal_safely(goal_handle)
        return {
            'goal_id': goal_id,
            'status': 'failed',
            'message': f'Navigation to {target} interrupted by ROS shutdown',
        }

    def _handle_navigate(self, goal_id: str, x: float, y: float, landmark: str) -> dict:
        target = landmark if landmark else f'({x:.2f}, {y:.2f})'
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            return {
                'goal_id': goal_id,
                'status': 'failed',
                'message': 'Nav2 action server /navigate_to_pose is not available',
            }

        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = _MAP_FRAME
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        send_future = self._nav_client.send_goal_async(goal)
        goal_handle, err = self._wait_future(send_future, _NAV_TIMEOUT)
        if err is not None:
            return {
                'goal_id': goal_id,
                'status': 'timeout',
                'message': f'Navigate goal send timed out after {_NAV_TIMEOUT}s',
            }

        if not goal_handle.accepted:
            return {
                'goal_id': goal_id,
                'status': 'failed',
                'message': f'Navigate goal rejected by Nav2 for {target}',
            }

        # Ignore Nav2's BT result and use TF pose as the source of truth.
        # Nav2 here is unreliable: SIGSEGVs, false successes from stale
        # localization, and aborts while the autonomy-stack tail keeps the
        # robot moving. Poll until the body actually arrives, stalls, or
        # times out.
        return self._poll_until_arrived(goal_handle, x, y, target, goal_id)

    # ------------------------------------------------------------------
    # Async navigate: start_navigate / check_nav_status / cancel_navigate
    # ------------------------------------------------------------------
    def _reap_finished_nav(self) -> None:
        """Auto-clear nav state if the stored goal's result future is already done.

        Guards against the case where a client stops polling check_nav_status
        before the nav actually finishes (e.g. ZMQ timeout, client crash) —
        otherwise _nav_goal_handle would leak forever and every subsequent
        start_navigate would be rejected with 'Navigation already in progress'.
        """
        with self._nav_lock:
            if self._nav_goal_handle is None or self._nav_result_future is None:
                return
            if not self._nav_result_future.done():
                return
            # Drain result so rclpy doesn't log "future destroyed with result not taken"
            try:
                self._nav_result_future.result()
            except Exception:
                pass
            self.get_logger().info(
                f'Auto-reaped finished nav (target={self._nav_target_desc!r})')
            self._nav_goal_handle   = None
            self._nav_goal_id       = None
            self._nav_result_future = None
            self._nav_target_desc   = ''
            self._nav_target_x      = 0.0
            self._nav_target_y      = 0.0

    def _handle_start_navigate(self, goal_id: str, x: float, y: float, landmark: str) -> dict:
        """Start Nav2 goal and return immediately. Use check_nav_status to poll."""
        # Self-heal: clear stale state from a previous nav whose client stopped polling.
        self._reap_finished_nav()
        with self._nav_lock:
            if self._nav_goal_handle is not None:
                return {
                    'goal_id': goal_id,
                    'status': 'failed',
                    'message': 'Navigation already in progress. Call check_nav_status or cancel_navigate first.',
                }

        target = landmark if landmark else f'({x:.2f}, {y:.2f})'
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            return {
                'goal_id': goal_id,
                'status': 'failed',
                'message': 'Nav2 action server /navigate_to_pose is not available',
            }

        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = _MAP_FRAME
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        send_future = self._nav_client.send_goal_async(goal)
        # Wait only for goal acceptance (fast, ~1-2s)
        goal_handle, err = self._wait_future(send_future, 5.0)
        if err is not None:
            return {
                'goal_id': goal_id,
                'status': 'failed',
                'message': f'Nav2 goal send timed out',
            }

        if not goal_handle.accepted:
            return {
                'goal_id': goal_id,
                'status': 'failed',
                'message': f'Nav2 rejected goal for {target}',
            }

        # Store state and return immediately
        pose0 = self._get_pose_full()
        d0 = math.hypot(pose0[0] - x, pose0[1] - y) if pose0 else float('inf')
        t0 = time.time()
        with self._nav_lock:
            self._nav_goal_handle   = goal_handle
            self._nav_goal_id       = goal_id
            self._nav_result_future = goal_handle.get_result_async()
            self._nav_target_desc   = target
            self._nav_target_x      = x
            self._nav_target_y      = y
            self._nav_t_start       = t0
            self._nav_last_dist     = d0
            self._nav_last_prog_t   = t0

        self.get_logger().info(f'[{goal_id[:8]}] Navigation started to {target}')
        return {
            'goal_id': goal_id,
            'status': 'started',
            'message': f'Navigation started to {target} (x={x:.2f}, y={y:.2f})',
        }

    def _handle_check_nav_status(self, goal_id: str) -> dict:
        """Poll whether the async navigation has finished.

        TF pose is the source of truth — Nav2's BT result is ignored so we
        survive aborts and crashes while the autonomy-stack tail is still
        driving the robot to the goal.
        """
        with self._nav_lock:
            if self._nav_goal_handle is None:
                return {'goal_id': goal_id, 'status': 'idle', 'message': 'No active navigation'}
            target = self._nav_target_desc
            tx, ty = self._nav_target_x, self._nav_target_y
            t_start = self._nav_t_start

        pose = self._get_pose_full()
        now = time.time()
        if pose is None:
            return {'goal_id': goal_id, 'status': 'navigating',
                    'message': f'Still navigating to {target} (TF temporarily unavailable)'}

        dist = math.hypot(pose[0] - tx, pose[1] - ty)

        def _clear_state():
            self._cancel_goal_safely(self._nav_goal_handle)
            self._nav_goal_handle   = None
            self._nav_goal_id       = None
            self._nav_result_future = None
            self._nav_target_desc   = ''
            self._nav_target_x      = 0.0
            self._nav_target_y      = 0.0

        if dist <= _ARRIVAL_THRESHOLD_M:
            with self._nav_lock:
                _clear_state()
            return {'goal_id': goal_id, 'status': 'success',
                    'message': f'Arrived at {target} (dist={dist:.2f} m)'}

        with self._nav_lock:
            if (self._nav_last_dist - dist) >= _NAV_STALL_PROG_M:
                self._nav_last_prog_t = now
                self._nav_last_dist   = dist
            time_since_progress = now - self._nav_last_prog_t

        if (now - t_start) >= _NAV_TIMEOUT:
            with self._nav_lock:
                _clear_state()
            return {'goal_id': goal_id, 'status': 'timeout',
                    'message': f'Navigation to {target} timed out (last dist={dist:.2f} m)'}

        if time_since_progress >= _NAV_STALL_S:
            with self._nav_lock:
                _clear_state()
            return {'goal_id': goal_id, 'status': 'failed',
                    'message': (f'No progress toward {target} for {_NAV_STALL_S:.0f}s '
                                f'(last dist={dist:.2f} m)')}

        return {'goal_id': goal_id, 'status': 'navigating',
                'message': f'Still navigating to {target} (dist={dist:.2f} m)'}

    def _handle_cancel_navigate(self, goal_id: str) -> dict:
        """Cancel the active async navigation."""
        with self._nav_lock:
            if self._nav_goal_handle is None:
                return {'goal_id': goal_id, 'status': 'ok',
                        'message': 'No active navigation to cancel'}
            self._nav_goal_handle.cancel_goal_async()
            self._nav_goal_handle   = None
            self._nav_goal_id       = None
            self._nav_result_future = None
            self._nav_target_desc   = ''
            self._nav_target_x      = 0.0
            self._nav_target_y      = 0.0
        self.get_logger().info(f'[{goal_id[:8]}] Navigation canceled')
        return {'goal_id': goal_id, 'status': 'ok', 'message': 'Navigation canceled'}

    # ------------------------------------------------------------------
    # update_objects_map: like update_objects but positions are already in map frame
    # ------------------------------------------------------------------
    def _handle_update_objects_map(self, goal_id: str, objects: list) -> dict:
        """
        Register detected objects with map-frame coordinates directly.
        Skips TF camera→map transform (positions already transformed client-side).
        Still does spatial dedup, TF broadcast, and JSON persist.
        """
        stamps         = []
        published      = []
        skipped        = []
        label_counts   = {}

        existing = _load_detected_objects(_DETECTED_OBJECTS_FILE)

        for obj in objects:
            label = str(obj.get('label', 'object')).replace(' ', '_')
            mx    = float(obj.get('map_x', 0.0))
            my    = float(obj.get('map_y', 0.0))

            # Spatial dedup
            too_close = False
            for entry in existing.values():
                if entry.get('label') == label:
                    dx = entry['px'] - mx
                    dy = entry['py'] - my
                    if math.hypot(dx, dy) < _SPATIAL_THRESHOLD:
                        too_close = True
                        break
            if too_close:
                skipped.append(label)
                continue

            idx = label_counts.get(label, 0)
            label_counts[label] = idx + 1
            while f'detected_{label}_{idx}' in existing:
                idx += 1
            label_counts[label] = idx + 1
            frame_name = f'detected_{label}_{idx}'

            ts = TransformStamped()
            ts.header.stamp    = self.get_clock().now().to_msg()
            ts.header.frame_id = _MAP_FRAME
            ts.child_frame_id  = frame_name
            ts.transform.translation.x = mx
            ts.transform.translation.y = my
            ts.transform.translation.z = 0.0
            ts.transform.rotation.w    = 1.0

            stamps.append(ts)
            published.append({
                'frame': frame_name,
                'map_x': round(mx, 3),
                'map_y': round(my, 3),
                'label': label,
            })
            self.get_logger().info(f'  {frame_name} → map ({mx:.3f}, {my:.3f})')

        if skipped:
            self.get_logger().info(f'  skipped (spatial dedup): {", ".join(skipped)}')

        if stamps:
            self._obj_tf_broadcaster.sendTransform(stamps)

        new_store = {
            entry['frame']: {'px': entry['map_x'], 'py': entry['map_y'], 'label': entry['label']}
            for entry in published
        }
        with self._stored_objects_lock:
            self._stored_objects = new_store

        if new_store:
            existing.update(new_store)
            _save_detected_objects(_DETECTED_OBJECTS_FILE, existing)

        return {
            'goal_id': goal_id,
            'status':  'success',
            'frames':  published,
            'skipped': skipped,
            'message': f'Published {len(stamps)} TF frame(s), skipped {len(skipped)} duplicate(s)',
        }

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

                vyaw = self._spin_kp * err
                # Apply minimum velocity to overcome friction/motor deadband
                if vyaw > 0:
                    vyaw = max(_MIN_VYAW, min(_MAX_VYAW, vyaw))
                else:
                    vyaw = max(-_MAX_VYAW, min(-_MIN_VYAW, vyaw))
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

                vx = self._move_kp * err
                # Apply minimum velocity to overcome friction/motor deadband
                if vx > 0:
                    vx = max(_MIN_VX, min(_MAX_VX, vx))
                else:
                    vx = max(-_MAX_VX, min(-_MIN_VX, vx))
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
            # In real-world mode /detected_objects is never published — fall back to
            # the last snapshot stored by update_objects (from detector_node_real_world).
            with self._stored_objects_lock:
                stored = dict(self._stored_objects)
            if stored:
                return {'goal_id': goal_id, 'status': 'ok', 'objects': stored}
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
    # update_objects: transform camera-frame detections to map frame and
    # broadcast each as a static TF frame (detected_{label}_{i} under map).
    # ------------------------------------------------------------------
    def _handle_update_objects(self, goal_id: str, objects: list) -> dict:
        # Look up map → camera_link (full chain via base_link)
        try:
            tf = self._tf_buffer.lookup_transform(
                _MAP_FRAME, 'camera_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return {
                'goal_id': goal_id,
                'status':  'failed',
                'message': f'TF map→camera_link not available: {e}',
            }

        t = tf.transform.translation
        q = tf.transform.rotation
        R = _quat_to_rotation_matrix(q.x, q.y, q.z, q.w)

        stamps         = []
        published      = []
        skipped        = []
        label_counts   = {}

        # Load existing stored objects for spatial dedup
        existing = _load_detected_objects(_DETECTED_OBJECTS_FILE)

        for obj in objects:
            label  = str(obj.get('label', 'object')).replace(' ', '_')
            cam_x  = float(obj.get('cam_x', 0.0))
            cam_y  = float(obj.get('cam_y', 0.0))
            cam_z  = float(obj.get('cam_z', 0.0))

            # Rotate + translate into map frame
            mx = R[0][0]*cam_x + R[0][1]*cam_y + R[0][2]*cam_z + t.x
            my = R[1][0]*cam_x + R[1][1]*cam_y + R[1][2]*cam_z + t.y

            # Spatial dedup: skip if same-label entry exists within threshold
            too_close = False
            for entry in existing.values():
                if entry.get('label') == label:
                    dx = entry['px'] - mx
                    dy = entry['py'] - my
                    if math.hypot(dx, dy) < _SPATIAL_THRESHOLD:
                        too_close = True
                        break
            if too_close:
                skipped.append(label)
                continue

            idx = label_counts.get(label, 0)
            label_counts[label] = idx + 1
            # Find next available frame index not already in existing
            while f'detected_{label}_{idx}' in existing:
                idx += 1
            label_counts[label] = idx + 1
            frame_name = f'detected_{label}_{idx}'

            ts = TransformStamped()
            ts.header.stamp    = self.get_clock().now().to_msg()
            ts.header.frame_id = _MAP_FRAME
            ts.child_frame_id  = frame_name
            ts.transform.translation.x = mx
            ts.transform.translation.y = my
            ts.transform.translation.z = 0.0   # ground plane
            ts.transform.rotation.w    = 1.0   # identity rotation

            stamps.append(ts)
            published.append({
                'frame': frame_name,
                'map_x': round(mx, 3),
                'map_y': round(my, 3),
                'label': label,
            })
            self.get_logger().info(
                f'  {frame_name} → map ({mx:.3f}, {my:.3f})'
            )

        if skipped:
            self.get_logger().info(
                f'  skipped (spatial dedup): {", ".join(skipped)}'
            )

        if stamps:
            self._obj_tf_broadcaster.sendTransform(stamps)

        # Save to in-memory store (current-frame snapshot for get_current_objects)
        new_store = {
            entry['frame']: {'px': entry['map_x'], 'py': entry['map_y'], 'label': entry['label']}
            for entry in published
        }
        with self._stored_objects_lock:
            self._stored_objects = new_store

        # Merge into persistent file (accumulates across sessions)
        if new_store:
            existing.update(new_store)
            _save_detected_objects(_DETECTED_OBJECTS_FILE, existing)

        return {
            'goal_id': goal_id,
            'status':  'success',
            'frames':  published,
            'skipped': skipped,
            'message': f'Published {len(stamps)} TF frame(s), skipped {len(skipped)} duplicate(s)',
        }

    # ------------------------------------------------------------------
    # get_camera_pose: return map→camera_link TF for client-side transform
    # ------------------------------------------------------------------
    def _handle_get_camera_pose(self, goal_id: str) -> dict:
        try:
            tf = self._tf_buffer.lookup_transform(
                _MAP_FRAME, 'camera_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return {
                'goal_id': goal_id,
                'status':  'failed',
                'message': f'TF map→camera_link not available: {e}',
            }

        t = tf.transform.translation
        q = tf.transform.rotation
        R = _quat_to_rotation_matrix(q.x, q.y, q.z, q.w)

        return {
            'goal_id': goal_id,
            'status':  'ok',
            'translation': [t.x, t.y, t.z],
            'rotation':    R,
        }

    # ------------------------------------------------------------------
    # get_detected_objects: return last update_objects snapshot
    # ------------------------------------------------------------------
    def _handle_get_detected_objects(self, goal_id: str) -> dict:
        objects = _load_detected_objects(_DETECTED_OBJECTS_FILE)
        return {'goal_id': goal_id, 'status': 'ok', 'objects': objects}

    def _handle_forget_object(self, goal_id: str, frame_name: str) -> dict:
        objects = _load_detected_objects(_DETECTED_OBJECTS_FILE)
        if frame_name not in objects:
            return {'goal_id': goal_id, 'status': 'failed',
                    'message': f"'{frame_name}' not found in detected objects"}
        del objects[frame_name]
        _save_detected_objects(_DETECTED_OBJECTS_FILE, objects)
        return {'goal_id': goal_id, 'status': 'success',
                'message': f"Removed '{frame_name}' from detected objects"}

    def _handle_clear_objects(self, goal_id: str) -> dict:
        objects = _load_detected_objects(_DETECTED_OBJECTS_FILE)
        count = len(objects)
        _save_detected_objects(_DETECTED_OBJECTS_FILE, {})
        with self._stored_objects_lock:
            self._stored_objects.clear()
        self.get_logger().info(f'[clear_objects] Cleared {count} detected object(s)')
        return {'goal_id': goal_id, 'status': 'success',
                'message': f'Cleared {count} detected object(s)'}

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
    def _wait_future(self, future, timeout_s: float):
        """Wait for a ROS future from ZMQ thread while rclpy spins elsewhere."""
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            if future.done():
                return future.result(), None
            time.sleep(0.02)
        return None, 'timeout'

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
