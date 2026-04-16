"""
Qwen Agent tools for on-demand YOLO object detection.

Two tools:
  - scan_objects      : run YOLO on the current camera frame → temp memory only
                        (non-destructive — does NOT persist to landmark history)
  - register_objects  : multi-frame stability check + interest filter + spatial dedup
                        → persist qualified objects to bridge as landmarks

Heavy resources (YOLO model, ZMQ image subscribers, background frame threads) are
initialised lazily on the first call and reused for all subsequent calls.
"""
import json
import os
import threading
import time
from collections import defaultdict

import cv2
import json5
import numpy as np
from qwen_agent.tools.base import BaseTool, register_tool

from .image_receiver import ImageReceiver
from .zmq_client import ZMQNavClient
from .landmark_loader import LandmarkLoader
from .detector_core import (
    YOLODetector, Detection, FrameCache, fill_depth_3d,
    DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_IOU,
)

_ROBOT_IP       = os.environ.get('ROBOT_IP',      '127.0.0.1')
_ROBOT_PORT     = int(os.environ.get('ROBOT_PORT', 5555))
_COLOR_PORT     = 5557
_DEPTH_PORT     = 5558
_NAV_TIMEOUT_MS = int(os.environ.get('NAV_TIMEOUT_MS', 10000))

# Interest list — only these labels are eligible for registration.
# Empty string = allow all labels.
_INTEREST_OBJECTS = os.environ.get('INTEREST_OBJECTS', '')

# Time to wait for frames on first init (seconds)
_WARMUP_WAIT    = 2.0
# How long to wait for a frame on each scan call (seconds)
_FRAME_WAIT     = 1.0

# Registration stability parameters
_STABILITY_FRAMES   = 1      # number of YOLO scans
_STABILITY_INTERVAL = 0.3    # seconds between scans (unused when _STABILITY_FRAMES == 1)
_STABILITY_MIN_HITS = 1      # must appear in >= this many frames
_DEDUP_DIST_M       = 0.5    # spatial dedup distance (metres)

# navigate_and_scan polling behaviour
_NAV_TERMINAL_STATES = {'success', 'failed', 'canceled', 'aborted', 'idle'}
_NAV_MAX_RUNTIME_S   = 180.0   # safety cap; cancel nav if scan loop exceeds this


# ---------------------------------------------------------------------------
# Lazy singleton scanner
# ---------------------------------------------------------------------------
class _Scanner:
    """Wraps YOLO detector + frame receivers. Created once, reused across calls."""

    def __init__(self):
        print('[scan_objects] Initialising YOLO detector + camera streams...')
        self.detector   = YOLODetector(DEFAULT_MODEL, conf=DEFAULT_CONF, iou=DEFAULT_IOU)
        self.cache      = FrameCache()
        self.zmq_client = ZMQNavClient(robot_ip=_ROBOT_IP, port=_ROBOT_PORT,
                                       timeout_ms=_NAV_TIMEOUT_MS)

        # Temp memory: latest scan results (list of dicts with label, confidence, cam_x/y/z)
        self.temp_objects: list[dict] = []

        # Start background frame receiver threads
        def _color_worker():
            rx = ImageReceiver(robot_ip=_ROBOT_IP, color_port=_COLOR_PORT,
                               depth_port=_DEPTH_PORT, recv_color=True, recv_depth=False)
            try:
                while True:
                    f = rx.recv_color(timeout_ms=2000)
                    if f is not None:
                        self.cache.put_color(f)
            finally:
                rx.close()

        def _depth_worker():
            rx = ImageReceiver(robot_ip=_ROBOT_IP, color_port=_COLOR_PORT,
                               depth_port=_DEPTH_PORT, recv_color=False, recv_depth=True)
            try:
                while True:
                    f = rx.recv_depth(timeout_ms=2000)
                    if f is not None:
                        self.cache.put_depth(f)
            finally:
                rx.close()

        for fn in (_color_worker, _depth_worker):
            threading.Thread(target=fn, daemon=True).start()

        # Wait briefly for first frames to arrive
        print(f'[scan_objects] Waiting {_WARMUP_WAIT}s for camera frames...')
        time.sleep(_WARMUP_WAIT)
        print('[scan_objects] Ready.')

    def _grab_frame(self):
        """Wait up to _FRAME_WAIT for a camera frame. Returns (color, depth) or (None, None)."""
        deadline = time.monotonic() + _FRAME_WAIT
        while time.monotonic() < deadline:
            color_frame, depth_frame = self.cache.get()
            if color_frame is not None:
                return color_frame, depth_frame
            time.sleep(0.05)
        return None, None

    def _detect_once(self, targets: set | None = None):
        """Run YOLO on current frame. Returns (qualified_detections, all_labels) or (None, None) on error."""
        color_frame, depth_frame = self._grab_frame()
        if color_frame is None:
            return None, None

        bgr = (cv2.cvtColor(color_frame.data, cv2.COLOR_RGB2BGR)
               if color_frame.encoding == 'rgb8' else color_frame.data)
        dets = self.detector.detect(bgr)
        fill_depth_3d(dets, color_frame, depth_frame)

        qualified = []
        for d in dets:
            if not d.has_3d:
                continue
            if targets and d.label not in targets:
                continue
            qualified.append(d)

        all_labels = [d.label for d in dets]
        return qualified, all_labels

    # ------------------------------------------------------------------
    # Stage 1: scan (temp memory only, non-destructive)
    # ------------------------------------------------------------------
    def scan(self, targets: set | None = None) -> dict:
        """
        Run one detection cycle. Stores results in temp memory only.
        Does NOT persist to the robot's landmark history.
        """
        qualified, all_labels = self._detect_once(targets)

        if qualified is None:
            self.temp_objects = []
            return {'status': 'error',
                    'message': 'No camera frame available. Is the RealSense camera running?'}

        if not qualified:
            self.temp_objects = []
            return {'status': 'ok', 'objects': [],
                    'message': f'No objects with valid depth detected. '
                               f'Raw detections: {all_labels if all_labels else "none"}'}

        # Store in temp memory
        self.temp_objects = [
            {'label': d.label, 'confidence': round(d.confidence, 2),
             'cam_x': d.cam_x_m, 'cam_y': d.cam_y_m, 'cam_z': d.cam_z_m}
            for d in qualified
        ]

        print(f'[scan_objects] Detected {len(self.temp_objects)} object(s) → temp memory: '
              f'{", ".join(o["label"] for o in self.temp_objects)}')

        return {
            'status': 'ok',
            'objects': [{'label': o['label'], 'confidence': o['confidence'],
                         'cam_z_m': round(o['cam_z'], 2)}
                        for o in self.temp_objects],
            'note': 'Stored in temp memory only. Call register_objects to persist as landmarks.',
        }

    # ------------------------------------------------------------------
    # Stage 2+3: register (stability check → interest filter → dedup → bridge)
    # ------------------------------------------------------------------
    def register(self, targets: set | None = None) -> dict:
        """
        Multi-frame stability check + interest filter + spatial dedup → persist to bridge.

        Args:
            targets: if non-empty, only register objects whose label is in this set.
                     If empty/None, falls back to INTEREST_OBJECTS env var, then all labels.
        """
        # --- Determine interest filter ---
        interest = targets
        if not interest and _INTEREST_OBJECTS:
            interest = {t.strip() for t in _INTEREST_OBJECTS.split(',') if t.strip()}

        # --- Stage 2a: Multi-frame stability ---
        print(f'[register_objects] Running {_STABILITY_FRAMES}-frame stability check...')
        # Track: label → list of (cam_x, cam_y, cam_z) across frames
        hits: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

        for i in range(_STABILITY_FRAMES):
            if i > 0:
                time.sleep(_STABILITY_INTERVAL)
            qualified, _ = self._detect_once(targets=interest)
            if qualified is None:
                continue
            for d in qualified:
                hits[d.label].append((d.cam_x_m, d.cam_y_m, d.cam_z_m))

        # Keep only labels seen in >= _STABILITY_MIN_HITS frames
        stable = {}
        for label, positions in hits.items():
            if len(positions) >= _STABILITY_MIN_HITS:
                # Average positions
                arr = np.array(positions)
                mean = arr.mean(axis=0)
                stable[label] = {
                    'label': label,
                    'cam_x': float(mean[0]),
                    'cam_y': float(mean[1]),
                    'cam_z': float(mean[2]),
                    'hit_count': len(positions),
                }

        unstable = [l for l, p in hits.items() if len(p) < _STABILITY_MIN_HITS]
        if unstable:
            print(f'[register_objects] Filtered out (unstable): {", ".join(unstable)}')

        if not stable:
            return {
                'status': 'ok',
                'registered': [],
                'filtered_unstable': unstable,
                'filtered_interest': [],
                'skipped_dedup': [],
                'message': 'No objects passed stability check.',
            }

        # --- Stage 2b: Interest filter ---
        filtered_interest = []
        if interest:
            for label in list(stable.keys()):
                if label not in interest:
                    filtered_interest.append(label)
                    del stable[label]
                    print(f'[register_objects] Filtered out (not in interest list): {label}')

        if not stable:
            return {
                'status': 'ok',
                'registered': [],
                'filtered_unstable': unstable,
                'filtered_interest': filtered_interest,
                'skipped_dedup': [],
                'message': 'No objects matched the interest list.',
            }

        # --- Stage 2c: Spatial dedup against existing landmarks ---
        # Get camera pose (map → camera_link TF) to transform detections to map frame
        pose_reply = self.zmq_client.send_command('get_camera_pose')
        if pose_reply.get('status') != 'ok':
            # Can't transform — skip dedup, let bridge handle it in Stage 3
            print(f'[register_objects] Warning: camera pose unavailable, skipping pre-dedup')
            to_send = list(stable.values())
            skipped_dedup = []
        else:
            tx, ty, tz = pose_reply['translation']
            R = pose_reply['rotation']

            # Bridge returns {frame_name: {"px": ..., "py": ..., "label": ...}, ...}
            existing_reply = self.zmq_client.send_command('get_detected_objects')
            existing_objects = existing_reply.get('objects', {})

            skipped_dedup = []
            to_send = []
            for label, obj in stable.items():
                # Transform camera-frame → map-frame
                cx, cy, cz = obj['cam_x'], obj['cam_y'], obj['cam_z']
                map_x = R[0][0]*cx + R[0][1]*cy + R[0][2]*cz + tx
                map_y = R[1][0]*cx + R[1][1]*cy + R[1][2]*cz + ty

                # Check distance to existing objects with same label (both in map frame)
                dominated = False
                for frame_name, ex in existing_objects.items():
                    if not isinstance(ex, dict):
                        continue
                    if ex.get('label', '') != label:
                        continue
                    dx = ex.get('px', 0) - map_x
                    dy = ex.get('py', 0) - map_y
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist < _DEDUP_DIST_M:
                        dominated = True
                        break
                if dominated:
                    skipped_dedup.append(label)
                    print(f'[register_objects] Skipped (already nearby): {label}')
                else:
                    to_send.append(obj)

        if not to_send:
            return {
                'status': 'ok',
                'registered': [],
                'filtered_unstable': unstable,
                'filtered_interest': filtered_interest,
                'skipped_dedup': skipped_dedup,
                'message': 'All stable objects already exist nearby.',
            }

        # --- Stage 3: Send to bridge ---
        objects_payload = [{'label': o['label'], 'cam_x': o['cam_x'],
                            'cam_y': o['cam_y'], 'cam_z': o['cam_z']}
                           for o in to_send]

        print(f'[register_objects] Sending {len(objects_payload)} object(s) to bridge: '
              f'{", ".join(o["label"] for o in objects_payload)}')
        reply = self.zmq_client.send_command('update_objects', objects=objects_payload)

        if reply.get('status') == 'success':
            frames  = reply.get('frames', [])
            bridge_skipped = reply.get('skipped', [])
            for f in frames:
                print(f'  → {f["frame"]:30s} map ({f["map_x"]:+.3f}, {f["map_y"]:+.3f})')
            if bridge_skipped:
                print(f'  ↷ bridge skipped (spatial dedup): {", ".join(bridge_skipped)}')
            return {
                'status': 'ok',
                'registered': frames,
                'filtered_unstable': unstable,
                'filtered_interest': filtered_interest,
                'skipped_dedup': skipped_dedup + bridge_skipped,
            }
        else:
            return {'status': 'error', 'message': reply.get('message', 'Bridge error')}

    # ------------------------------------------------------------------
    # Navigate + scan concurrently (async nav on bridge)
    # ------------------------------------------------------------------
    def navigate_and_scan(self, x: float, y: float, landmark: str = '',
                          targets: set | None = None,
                          scan_interval: float = 2.0) -> dict:
        """
        Start async navigation to (x, y) and run periodic YOLO scans while moving.
        Detections are transformed to map frame using get_camera_pose and accumulated
        in temp_objects. Returns nav result + detection summary.
        """
        # 1. Start async navigation
        start_reply = self.zmq_client.send_command(
            'start_navigate', x=x, y=y, landmark=landmark)
        if start_reply.get('status') != 'started':
            return {'status': 'failed',
                    'message': start_reply.get('message', 'Failed to start navigation')}

        print(f'[navigate_and_scan] Navigation started to {landmark or f"({x:.2f}, {y:.2f})"}')

        # 2. Scan loop while navigating
        all_detections = []
        scan_count = 0
        t_start = time.time()
        nav_status = 'navigating'
        nav_reply: dict = {}

        while True:
            time.sleep(scan_interval)

            # Safety cap: cancel nav if loop runs too long (prevents runaway navigation)
            if time.time() - t_start > _NAV_MAX_RUNTIME_S:
                print(f'[navigate_and_scan] runtime cap ({_NAV_MAX_RUNTIME_S}s) exceeded '
                      f'— canceling nav')
                self.zmq_client.send_command('cancel_navigate')
                nav_status = 'canceled'
                nav_reply = {'status': 'canceled',
                             'message': f'Canceled by client after {_NAV_MAX_RUNTIME_S}s'}
                break

            # Poll bridge for nav status
            nav_reply = self.zmq_client.send_command('check_nav_status')
            nav_status = nav_reply.get('status', 'unknown')

            # 'timeout' == ZMQ RCVTIMEO fired (bridge busy). Nav is still running on
            # the robot — retry, don't treat as terminal.
            if nav_status == 'timeout':
                print('[navigate_and_scan] check_nav_status timed out '
                      '(bridge busy) — retrying')
                continue

            if nav_status in _NAV_TERMINAL_STATES:
                break  # nav really finished

            # Otherwise: still navigating → run a scan this iteration
            qualified, _ = self._detect_once(targets=targets)
            if not qualified:
                continue
            scan_count += 1

            # Get camera pose for map-frame transform
            pose_reply = self.zmq_client.send_command('get_camera_pose')
            if pose_reply.get('status') != 'ok':
                print(f'[navigate_and_scan] scan {scan_count}: camera pose unavailable, skipping')
                continue

            tx, ty, _tz = pose_reply['translation']
            R = pose_reply['rotation']

            for d in qualified:
                map_x = R[0][0]*d.cam_x_m + R[0][1]*d.cam_y_m + R[0][2]*d.cam_z_m + tx
                map_y = R[1][0]*d.cam_x_m + R[1][1]*d.cam_y_m + R[1][2]*d.cam_z_m + ty
                all_detections.append({
                    'label': d.label,
                    'confidence': round(d.confidence, 2),
                    'map_x': round(map_x, 3),
                    'map_y': round(map_y, 3),
                    'cam_z': round(d.cam_z_m, 2),
                    'scan_index': scan_count,
                })

            labels_this_scan = [d.label for d in qualified]
            print(f'[navigate_and_scan] scan {scan_count}: {", ".join(labels_this_scan)}')

        # 3. Store in temp memory (map-frame positions)
        self.temp_objects = all_detections

        # Summary by label
        label_counts = {}
        for d in all_detections:
            label_counts[d['label']] = label_counts.get(d['label'], 0) + 1

        print(f'[navigate_and_scan] Done. nav={nav_status}, scans={scan_count}, '
              f'detections={len(all_detections)}')

        return {
            'status': nav_status,
            'nav_message': nav_reply.get('message', ''),
            'scans_performed': scan_count,
            'objects_detected': [{'label': l, 'seen_count': c}
                                 for l, c in label_counts.items()],
            'total_detections': len(all_detections),
            'note': 'Objects in temp memory with map-frame positions. '
                    'Call register_objects to persist as landmarks.',
        }

    # ------------------------------------------------------------------
    # Register from temp memory (map-frame coords, skip re-scan)
    # ------------------------------------------------------------------
    def register_from_temp(self, targets: set | None = None) -> dict:
        """
        Persist objects accumulated in temp_objects (from navigate_and_scan).
        Positions are already in map frame. Performs spatial dedup against existing
        landmarks, then sends to bridge via update_objects_map.
        """
        if not self.temp_objects:
            return {'status': 'ok', 'registered': [], 'skipped_dedup': [],
                    'message': 'No objects in temp memory to register.'}

        # Aggregate: average map positions per label across all scans
        from collections import defaultdict
        label_positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for obj in self.temp_objects:
            label = obj['label']
            if targets and label not in targets:
                continue
            label_positions[label].append((obj['map_x'], obj['map_y']))

        if not label_positions:
            return {'status': 'ok', 'registered': [], 'skipped_dedup': [],
                    'message': 'No matching objects in temp memory.'}

        # Average positions per label
        averaged = {}
        for label, positions in label_positions.items():
            arr = np.array(positions)
            mean = arr.mean(axis=0)
            averaged[label] = {
                'label': label,
                'map_x': float(mean[0]),
                'map_y': float(mean[1]),
                'hit_count': len(positions),
            }

        # Spatial dedup against existing landmarks
        existing_reply = self.zmq_client.send_command('get_detected_objects')
        existing_objects = existing_reply.get('objects', {})

        skipped_dedup = []
        to_send = []
        for label, obj in averaged.items():
            dominated = False
            for _frame, ex in existing_objects.items():
                if not isinstance(ex, dict):
                    continue
                if ex.get('label', '') != label:
                    continue
                dx = ex.get('px', 0) - obj['map_x']
                dy = ex.get('py', 0) - obj['map_y']
                if (dx**2 + dy**2) ** 0.5 < _DEDUP_DIST_M:
                    dominated = True
                    break
            if dominated:
                skipped_dedup.append(label)
                print(f'[register_from_temp] Skipped (already nearby): {label}')
            else:
                to_send.append(obj)

        if not to_send:
            return {'status': 'ok', 'registered': [], 'skipped_dedup': skipped_dedup,
                    'message': 'All objects already exist nearby.'}

        # Send to bridge via update_objects_map (positions already in map frame)
        objects_payload = [{'label': o['label'], 'map_x': o['map_x'], 'map_y': o['map_y']}
                           for o in to_send]

        print(f'[register_from_temp] Sending {len(objects_payload)} object(s) to bridge: '
              f'{", ".join(o["label"] for o in objects_payload)}')
        reply = self.zmq_client.send_command('update_objects_map', objects=objects_payload)

        if reply.get('status') == 'success':
            frames = reply.get('frames', [])
            bridge_skipped = reply.get('skipped', [])
            for f in frames:
                print(f'  → {f["frame"]:30s} map ({f["map_x"]:+.3f}, {f["map_y"]:+.3f})')
            return {
                'status': 'ok',
                'registered': frames,
                'skipped_dedup': skipped_dedup + bridge_skipped,
            }
        else:
            return {'status': 'error', 'message': reply.get('message', 'Bridge error')}


_scanner: _Scanner | None = None
_scanner_lock = threading.Lock()


def _get_scanner() -> _Scanner:
    global _scanner
    if _scanner is None:
        with _scanner_lock:
            if _scanner is None:
                _scanner = _Scanner()
    return _scanner


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------
@register_tool('scan_objects')
class ScanObjects(BaseTool):
    description = (
        'Run YOLO object detection on the current camera frame. '
        'Detects objects and stores them in TEMP MEMORY only — does NOT persist '
        'to landmark history. Use this to look at what is around without side effects. '
        'Call register_objects afterwards to persist detected objects as landmarks.'
    )
    parameters = [
        {
            'name': 'targets',
            'type': 'string',
            'description': (
                'Comma-separated list of object labels to look for '
                '(e.g. "bottle,cup"). Leave empty to detect all objects.'
            ),
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params) if params.strip() else {}
        targets_str = args.get('targets', '').strip()
        targets = {t.strip() for t in targets_str.split(',') if t.strip()} if targets_str else None

        print(f'[scan_objects] called  targets={targets or "all"}')
        scanner = _get_scanner()
        result = scanner.scan(targets=targets)
        return json.dumps(result, ensure_ascii=False)


@register_tool('register_objects')
class RegisterObjects(BaseTool):
    description = (
        'Persist detected objects as permanent landmarks on the robot. '
        'If navigate_and_scan just accumulated detections, this call drains that temp memory '
        '(map-frame positions, deduplicated against existing landmarks). '
        'Otherwise it runs a fresh 1-frame scan at the current camera pose. '
        'Only call this when the user asks to remember/save objects, or when actively '
        'searching for a specific object to navigate to.'
    )
    parameters = [
        {
            'name': 'targets',
            'type': 'string',
            'description': (
                'Comma-separated labels to register (e.g. "bottle,cup"). '
                'Empty = use default interest list or register all.'
            ),
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params) if params.strip() else {}
        targets_str = args.get('targets', '').strip()
        targets = {t.strip() for t in targets_str.split(',') if t.strip()} if targets_str else None

        scanner = _get_scanner()

        # Auto-dispatch: if navigate_and_scan filled temp with map-frame entries,
        # drain those. Otherwise run a fresh scan at the current pose.
        has_map_temp = bool(scanner.temp_objects) and 'map_x' in scanner.temp_objects[0]

        if has_map_temp:
            print(f'[register_objects] auto-drain temp memory '
                  f'({len(scanner.temp_objects)} entries) targets={targets or "all"}')
            result = scanner.register_from_temp(targets=targets)
            # Clear temp on success so a repeat "save them" doesn't replay the same detections.
            if result.get('status') == 'ok':
                scanner.temp_objects = []
        else:
            print(f'[register_objects] fresh scan  targets={targets or "default"}')
            result = scanner.register(targets=targets)

        return json.dumps(result, ensure_ascii=False)


_DETECTION_MODE = os.environ.get('DETECTION_MODE', 'sim')
_landmark_loader = LandmarkLoader()


@register_tool('navigate_and_scan')
class NavigateAndScan(BaseTool):
    description = (
        'Navigate the robot to a landmark while running YOLO detection along the way. '
        'Detections are stored in temp memory with correct map-frame positions. '
        'Call register_objects after arrival to persist interesting objects as landmarks.'
    )
    parameters = [
        {
            'name': 'landmark_name',
            'type': 'string',
            'description': 'Destination landmark name (e.g. "kitchen", "hallway"). Omit when using coordinates.',
            'required': False,
        },
        {
            'name': 'x',
            'type': 'number',
            'description': 'Map x coordinate. Use when navigating by coordinates.',
            'required': False,
        },
        {
            'name': 'y',
            'type': 'number',
            'description': 'Map y coordinate. Use when navigating by coordinates.',
            'required': False,
        },
        {
            'name': 'targets',
            'type': 'string',
            'description': 'Comma-separated labels to detect (e.g. "bottle,cup"). Empty = detect all.',
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params) if params.strip() else {}
        landmark_name = args.get('landmark_name', '').strip().lower() if args.get('landmark_name') else ''
        targets_str = args.get('targets', '').strip()
        targets = {t.strip() for t in targets_str.split(',') if t.strip()} if targets_str else None

        scanner = _get_scanner()

        # Resolve landmark → coordinates
        if landmark_name:
            try:
                pos = _landmark_loader.get(landmark_name)
                x, y = pos['x'], pos['y']
            except KeyError:
                # In real-world mode, also check bridge-stored detected objects
                if _DETECTION_MODE == 'real':
                    reply = scanner.zmq_client.send_command('get_detected_objects')
                    obj = reply.get('objects', {}).get(landmark_name)
                    if obj:
                        x, y = obj['px'], obj['py']
                    else:
                        available = list(_landmark_loader._landmarks.keys()) + list(reply.get('objects', {}).keys())
                        return json.dumps({'status': 'failed',
                                           'message': f"'{landmark_name}' not found. Available: {', '.join(available)}"})
                else:
                    available = ', '.join(_landmark_loader._landmarks.keys())
                    return json.dumps({'status': 'failed',
                                       'message': f"'{landmark_name}' not found. Available: {available}"})
        elif 'x' in args and 'y' in args:
            x, y = float(args['x']), float(args['y'])
        else:
            return json.dumps({'status': 'failed',
                               'message': 'Provide either landmark_name or both x and y.'})

        print(f'[navigate_and_scan] → {landmark_name or f"({x:.2f}, {y:.2f})"}  targets={targets or "all"}')
        result = scanner.navigate_and_scan(x=x, y=y, landmark=landmark_name,
                                           targets=targets)
        return json.dumps(result, ensure_ascii=False)
