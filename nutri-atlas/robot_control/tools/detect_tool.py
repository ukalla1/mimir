"""
Qwen Agent tool for on-demand YOLO object detection.

Registers one tool:
  - scan_objects : run YOLO on the current camera frame, compute 3D positions,
                   and send detected objects to the robot as persistent landmarks.

Heavy resources (YOLO model, ZMQ image subscribers, background frame threads) are
initialised lazily on the first call and reused for all subsequent calls.
"""
import json
import os
import threading
import time

import cv2
import json5
from qwen_agent.tools.base import BaseTool, register_tool

from .image_receiver import ImageReceiver
from .zmq_client import ZMQNavClient
from .detector_core import (
    YOLODetector, FrameCache, fill_depth_3d,
    DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_IOU,
)

_ROBOT_IP       = os.environ.get('ROBOT_IP',      '127.0.0.1')
_ROBOT_PORT     = int(os.environ.get('ROBOT_PORT', 5555))
_COLOR_PORT     = 5557
_DEPTH_PORT     = 5558
_NAV_TIMEOUT_MS = int(os.environ.get('NAV_TIMEOUT_MS', 10000))

# Time to wait for frames on first init (seconds)
_WARMUP_WAIT    = 2.0
# How long to wait for a frame on each scan call (seconds)
_FRAME_WAIT     = 1.0


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

    def scan(self, targets: set | None = None) -> dict:
        """
        Run one detection cycle on the current camera frame.

        Args:
            targets: if non-empty, only keep detections whose label is in this set.

        Returns dict with:
            status  : 'ok' | 'error'
            objects : list of detected objects with labels and positions
            frames  : list of newly registered TF frames (from bridge)
            skipped : list of labels skipped by spatial dedup
        """
        # Wait up to _FRAME_WAIT for a frame
        deadline = time.monotonic() + _FRAME_WAIT
        color_frame, depth_frame = None, None
        while time.monotonic() < deadline:
            color_frame, depth_frame = self.cache.get()
            if color_frame is not None:
                break
            time.sleep(0.05)

        if color_frame is None:
            return {'status': 'error',
                    'message': 'No camera frame available. Is the RealSense camera running?'}

        # Run YOLO
        bgr = (cv2.cvtColor(color_frame.data, cv2.COLOR_RGB2BGR)
               if color_frame.encoding == 'rgb8' else color_frame.data)
        dets = self.detector.detect(bgr)

        # Fill depth + 3D positions
        fill_depth_3d(dets, color_frame, depth_frame)

        # Filter: only keep detections with valid 3D, optionally by target label
        qualified = []
        for d in dets:
            if not d.has_3d:
                continue
            if targets and d.label not in targets:
                continue
            qualified.append(d)

        if not qualified:
            all_labels = [d.label for d in dets]
            return {'status': 'ok', 'objects': [], 'frames': [], 'skipped': [],
                    'message': f'No objects with valid depth detected. '
                               f'Raw detections: {all_labels if all_labels else "none"}'}

        # Build payload and send to bridge
        objects = [{'label': d.label, 'cam_x': d.cam_x_m,
                    'cam_y': d.cam_y_m, 'cam_z': d.cam_z_m}
                   for d in qualified]

        print(f'[scan_objects] Sending {len(objects)} detection(s): '
              f'{", ".join(o["label"] for o in objects)}')
        reply = self.zmq_client.send_command('update_objects', objects=objects)

        if reply.get('status') == 'success':
            frames  = reply.get('frames', [])
            skipped = reply.get('skipped', [])
            for f in frames:
                print(f'  → {f["frame"]:30s} map ({f["map_x"]:+.3f}, {f["map_y"]:+.3f})')
            if skipped:
                print(f'  ↷ skipped (already stored nearby): {", ".join(skipped)}')
            return {
                'status': 'ok',
                'objects': [{'label': d.label, 'confidence': round(d.confidence, 2),
                             'cam_z_m': round(d.cam_z_m, 2)}
                            for d in qualified],
                'frames': frames,
                'skipped': skipped,
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
# Agent tool
# ---------------------------------------------------------------------------
@register_tool('scan_objects')
class ScanObjects(BaseTool):
    description = (
        'Run YOLO object detection on the current camera frame. '
        'Detects objects, computes their 3D positions, and registers them as '
        'persistent landmarks on the robot. Returns newly detected objects and '
        'any that were skipped (already known nearby). '
        'Use this when asked to inspect, scan, or look for objects at the current location.'
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
