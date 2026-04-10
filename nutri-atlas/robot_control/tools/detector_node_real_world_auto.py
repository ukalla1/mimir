"""
Automatic detector node — runs YOLO on RealSense streams and sends stable
detections to the robot without manual input.

Approach D gating:
  1. Stability gate  : detection must appear in N consecutive frames (default 8)
  2. Spatial dedup   : bridge skips objects within 0.5 m of an existing same-label entry

On startup, fetches the current detected_objects.json from the robot so the
spatial dedup starts with an up-to-date picture.

Usage:
    python detector_node_real_world_auto.py --robot-ip 192.168.0.114

    # Tune gating:
    python detector_node_real_world_auto.py --robot-ip 192.168.0.114 --stable-frames 12

Press 'q' to quit.
"""

import argparse
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from image_receiver import ImageReceiver, ImageFrame
from zmq_client import ZMQNavClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_WEIGHTS_DIR       = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')
_DEFAULT_MODEL     = os.path.join(_WEIGHTS_DIR, 'yolo11n.onnx')
_DEFAULT_CONF      = 0.35
_DEFAULT_IOU       = 0.45
_DEFAULT_STABLE_N  = 8      # consecutive frames required before sending
_DEPTH_PATCH_HALF  = 3

COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush',
]


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int
    cx: int; cy: int
    depth_mm: float = 0.0
    cam_x_m:  float = 0.0
    cam_y_m:  float = 0.0
    cam_z_m:  float = 0.0

    @property
    def has_3d(self) -> bool:
        return self.depth_mm > 0


# ---------------------------------------------------------------------------
# Stability tracker
# ---------------------------------------------------------------------------
@dataclass
class _LabelState:
    streak:     int   = 0       # consecutive frames seen
    sent:       bool  = False   # already sent this session
    cam_x_sum:  float = 0.0
    cam_y_sum:  float = 0.0
    cam_z_sum:  float = 0.0


class StabilityTracker:
    """
    Tracks per-label consecutive-frame counts.
    A label is 'stable' when it has been seen for `n_frames` consecutive frames
    without interruption AND has not already been sent this session.

    Filters:
      - min_conf  : per-frame confidence must exceed this to count toward streak
      - targets   : if non-empty, only labels in this set are tracked

    On each frame call `update(detections)`:
      - qualifying detections: increment streak, accumulate averaged 3D position
      - absent/non-qualifying: reset streak (and sent flag if they disappeared)

    Call `pop_ready()` to get labels that just crossed the threshold.
    """
    def __init__(self, n_frames: int, min_conf: float = 0.5, targets: set = None):
        self._n        = n_frames
        self._min_conf = min_conf
        self._targets  = targets or set()   # empty = accept all labels
        self._state: dict[str, _LabelState] = defaultdict(_LabelState)

    def _qualifies(self, det) -> bool:
        if not det.has_3d:
            return False
        if self._targets and det.label not in self._targets:
            return False
        if det.confidence < self._min_conf:
            return False
        return True

    def update(self, detections: list) -> None:
        seen = {d.label for d in detections if self._qualifies(d)}
        for det in detections:
            if not self._qualifies(det):
                continue
            s = self._state[det.label]
            s.streak    += 1
            s.cam_x_sum += det.cam_x_m
            s.cam_y_sum += det.cam_y_m
            s.cam_z_sum += det.cam_z_m

        # Reset labels that disappeared or no longer qualify this frame
        for label, s in self._state.items():
            if label not in seen:
                s.streak    = 0
                s.sent      = False
                s.cam_x_sum = 0.0
                s.cam_y_sum = 0.0
                s.cam_z_sum = 0.0

    def pop_ready(self) -> list[dict]:
        """Return list of {label, cam_x, cam_y, cam_z} for newly stable detections."""
        ready = []
        for label, s in self._state.items():
            if s.streak >= self._n and not s.sent:
                n = s.streak
                ready.append({
                    'label': label,
                    'cam_x': s.cam_x_sum / n,
                    'cam_y': s.cam_y_sum / n,
                    'cam_z': s.cam_z_sum / n,
                })
                s.sent = True   # suppress until it disappears and reappears
        return ready


# ---------------------------------------------------------------------------
# YOLO ONNX runner
# ---------------------------------------------------------------------------
class YOLODetector:
    def __init__(self, model_path: str, conf: float, iou: float):
        self.conf = conf
        self.iou  = iou
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(model_path, providers=providers)
        inp             = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = inp.shape
        print(f'[YOLO] {model_path}  input={inp.shape}'
              f'  provider={self.session.get_providers()[0]}')

    def detect(self, bgr: np.ndarray) -> list:
        ih, iw = bgr.shape[:2]
        resized = cv2.resize(bgr, (self.w, self.h))
        blob    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob    = blob.transpose(2, 0, 1)[np.newaxis]
        raw     = self.session.run(None, {self.input_name: blob})[0]
        pred    = raw[0].T

        boxes_xywh  = pred[:, :4]
        scores      = pred[:, 4:]
        class_ids   = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask        = confidences >= self.conf
        boxes_xywh  = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        sx = iw / self.w;  sy = ih / self.h
        cx_ = boxes_xywh[:, 0] * sx;  cy_ = boxes_xywh[:, 1] * sy
        bw  = boxes_xywh[:, 2] * sx;  bh  = boxes_xywh[:, 3] * sy
        x1  = cx_ - bw / 2;           y1  = cy_ - bh / 2
        x2  = cx_ + bw / 2;           y2  = cy_ + bh / 2

        boxes_cv = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(boxes_cv, confidences.tolist(), self.conf, self.iou)
        if len(keep) == 0:
            return []

        out = []
        for i in keep.flatten():
            ix1, iy1, ix2, iy2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
            label = COCO_NAMES[class_ids[i]] if class_ids[i] < len(COCO_NAMES) else str(class_ids[i])
            out.append(Detection(
                label=label, confidence=float(confidences[i]),
                x1=ix1, y1=iy1, x2=ix2, y2=iy2,
                cx=(ix1 + ix2) // 2, cy=(iy1 + iy2) // 2,
            ))
        return out


# ---------------------------------------------------------------------------
# Frame cache + depth helpers
# ---------------------------------------------------------------------------
class _FrameCache:
    def __init__(self):
        self._color = None; self._depth = None
        self._lock  = threading.Lock()

    def put_color(self, f):
        with self._lock: self._color = f

    def put_depth(self, f):
        with self._lock: self._depth = f

    def get(self):
        with self._lock: return self._color, self._depth


def _sample_depth(depth_frame: ImageFrame, cx: int, cy: int) -> float:
    d = depth_frame.data
    h, w = d.shape[:2]
    n = _DEPTH_PATCH_HALF
    x1 = max(0, cx - n); x2 = min(w, cx + n + 1)
    y1 = max(0, cy - n); y2 = min(h, cy + n + 1)
    patch = d[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0


def _backproject(depth_frame: ImageFrame, dcx: int, dcy: int, depth_mm: float):
    if depth_frame.camera_info is None or len(depth_frame.camera_info.k) < 9:
        return 0.0, 0.0, 0.0
    k = depth_frame.camera_info.k
    fx, fy, ppx, ppy = k[0], k[4], k[2], k[5]
    if fx == 0 or fy == 0:
        return 0.0, 0.0, 0.0
    z_m = depth_mm / 1000.0
    return (dcx - ppx) * z_m / fx, (dcy - ppy) * z_m / fy, z_m


def _annotate(bgr: np.ndarray, detections: list, stable_labels: set) -> np.ndarray:
    out = bgr.copy()
    for det in detections:
        if not det.has_3d:
            color = (0, 120, 220)   # blue — no depth
        elif det.label in stable_labels:
            color = (0, 200, 0)     # green — stable / sent
        else:
            color = (0, 200, 200)   # yellow — accumulating
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        if det.has_3d:
            label_str = (f'{det.label} {det.confidence:.2f} | '
                         f'z={det.cam_z_m:.2f}m x={det.cam_x_m:+.2f}m y={det.cam_y_m:+.2f}m')
        else:
            label_str = f'{det.label} {det.confidence:.2f} | depth?'
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(det.y1 - 6, th + 4)
        cv2.rectangle(out, (det.x1, ty - th - 4), (det.x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label_str, (det.x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (det.cx, det.cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Auto detector — sends stable detections to robot')
    parser.add_argument('--robot-ip',      default=os.environ.get('ROBOT_IP', '127.0.0.1'))
    parser.add_argument('--robot-port',    type=int, default=int(os.environ.get('ROBOT_PORT', 5555)))
    parser.add_argument('--color-port',    type=int, default=5557)
    parser.add_argument('--depth-port',    type=int, default=5558)
    parser.add_argument('--model',         default=_DEFAULT_MODEL)
    parser.add_argument('--conf',          type=float, default=_DEFAULT_CONF)
    parser.add_argument('--iou',           type=float, default=_DEFAULT_IOU)
    parser.add_argument('--stable-frames', type=int,   default=_DEFAULT_STABLE_N,
                        help='Consecutive frames required before sending a detection (default 8)')
    parser.add_argument('--stable-conf',  type=float, default=0.5,
                        help='Min confidence per frame to count toward stability streak (default 0.5)')
    parser.add_argument('--targets',      nargs='*',  default=[], metavar='LABEL',
                        help='Labels to track as landmarks (e.g. --targets person chair). '
                             'Empty = track all detected objects.')
    parser.add_argument('--no-display',    action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'[ERROR] Model not found: {args.model}')
        print('Export: yolo export model=yolo11n.pt format=onnx imgsz=640')
        return

    print('=' * 60)
    print(f'  Robot        : {args.robot_ip}:{args.robot_port}')
    print(f'  Color port   : {args.color_port}')
    print(f'  Depth port   : {args.depth_port}')
    print(f'  Model        : {args.model}')
    print(f'  Conf / IoU   : {args.conf} / {args.iou}')
    print(f'  Stable frames: {args.stable_frames}')
    print(f'  Stable conf  : {args.stable_conf}')
    print(f'  Targets      : {", ".join(sorted(args.targets)) if args.targets else "all"}')
    print('=' * 60)

    detector   = YOLODetector(args.model, conf=args.conf, iou=args.iou)
    zmq_client = ZMQNavClient(robot_ip=args.robot_ip, port=args.robot_port, timeout_ms=5000)
    cache      = _FrameCache()
    tracker    = StabilityTracker(
        n_frames=args.stable_frames,
        min_conf=args.stable_conf,
        targets=set(args.targets),
    )

    # Fetch existing stored objects from robot at startup (for display awareness)
    print('[init] Fetching existing detected objects from robot...')
    init_reply = zmq_client.send_command('get_detected_objects')
    known_labels: set = {v.get('label', '') for v in init_reply.get('objects', {}).values()}
    if known_labels:
        print(f'[init] Already stored: {", ".join(sorted(known_labels))}')
    else:
        print('[init] No existing objects stored.')

    # Background image workers
    def _color_worker():
        rx = ImageReceiver(robot_ip=args.robot_ip, color_port=args.color_port,
                           depth_port=args.depth_port, recv_color=True, recv_depth=False)
        try:
            while True:
                f = rx.recv_color(timeout_ms=2000)
                if f is not None: cache.put_color(f)
        finally:
            rx.close()

    def _depth_worker():
        rx = ImageReceiver(robot_ip=args.robot_ip, color_port=args.color_port,
                           depth_port=args.depth_port, recv_color=False, recv_depth=True)
        try:
            while True:
                f = rx.recv_depth(timeout_ms=2000)
                if f is not None: cache.put_depth(f)
        finally:
            rx.close()

    for fn in (_color_worker, _depth_worker):
        threading.Thread(target=fn, daemon=True).start()

    print('Waiting for first frames... (press q to quit)')
    _intrinsics_printed = False
    stable_sent: set = set()   # labels confirmed sent this session (for display)

    while True:
        color_frame, depth_frame = cache.get()
        if color_frame is None:
            time.sleep(0.01)
            continue

        # Print camera info once
        if not _intrinsics_printed and depth_frame is not None:
            ci = depth_frame.camera_info
            print(f'  Color : {color_frame.width}×{color_frame.height}  enc={color_frame.encoding}')
            print(f'  Depth : {depth_frame.width}×{depth_frame.height}  enc={depth_frame.encoding}')
            if ci and len(ci.k) == 9:
                print(f'  K     : fx={ci.k[0]:.1f}  fy={ci.k[4]:.1f}'
                      f'  ppx={ci.k[2]:.1f}  ppy={ci.k[5]:.1f}')
            print('=' * 60)
            _intrinsics_printed = True

        bgr = (cv2.cvtColor(color_frame.data, cv2.COLOR_RGB2BGR)
               if color_frame.encoding == 'rgb8' else color_frame.data)

        dets = detector.detect(bgr)

        # Fill depth + 3D positions
        if depth_frame is not None:
            dh, dw = depth_frame.data.shape[:2]
            ch, cw = color_frame.data.shape[:2]
            sx, sy = dw / cw, dh / ch
            for det in dets:
                dcx = int(min(max(det.cx * sx, 0), dw - 1))
                dcy = int(min(max(det.cy * sy, 0), dh - 1))
                det.depth_mm = _sample_depth(depth_frame, dcx, dcy)
                if det.depth_mm > 0:
                    det.cam_x_m, det.cam_y_m, det.cam_z_m = _backproject(
                        depth_frame, dcx, dcy, det.depth_mm)

        # Update stability tracker
        tracker.update(dets)
        ready = tracker.pop_ready()

        # Send stable detections to robot
        if ready:
            print(f'[auto] Sending {len(ready)} stable detection(s): '
                  f'{", ".join(o["label"] for o in ready)}')
            reply = zmq_client.send_command('update_objects', objects=ready)
            if reply.get('status') == 'success':
                for f in reply.get('frames', []):
                    print(f'  → {f["frame"]:30s} map ({f["map_x"]:+.3f}, {f["map_y"]:+.3f})')
                    stable_sent.add(f['label'])
                skipped = reply.get('skipped', [])
                if skipped:
                    print(f'  ↷ skipped (already stored nearby): {", ".join(skipped)}')
                    stable_sent.update(skipped)
            else:
                print(f'  [error] {reply.get("message")}')

        # Display
        if not args.no_display:
            annotated = _annotate(bgr, dets, stable_sent)
            n_valid = sum(1 for d in dets if d.has_3d)
            cv2.putText(annotated,
                        f'{len(dets)} det  {n_valid} with depth | auto (stable={args.stable_frames}f)  q=quit',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('detector_node_real_world_auto', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not args.no_display:
        cv2.destroyAllWindows()
    zmq_client.close()
    print('Stopped.')


if __name__ == '__main__':
    main()
