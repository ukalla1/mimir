"""
Interactive detector node — runs YOLO on RealSense streams and sends detections
to the robot on demand (press Enter).

On Enter, the current detections (with valid 3D positions in camera_link frame)
are sent to zmq_bridge_node_working_v2.py via the 'update_objects' command.
The robot transforms them to map frame and broadcasts TF frames:
    detected_{label}_{i}  as children of  map

Usage:
    python detector_node.py --robot-ip 192.168.0.114

Verify on robot:
    ros2 tf echo map detected_bottle_0
    ros2 run tf2_tools view_frames
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from image_reciver_test import ImageReceiver, ImageFrame

# Reuse ZMQNavClient from the operator-side tools
_tools_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'robot_control', 'tools')
sys.path.insert(0, os.path.abspath(_tools_path))
from zmq_client import ZMQNavClient  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_WEIGHTS_DIR      = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'weights')
_DEFAULT_MODEL    = os.path.join(_WEIGHTS_DIR, 'yolo11n.onnx')
_DEFAULT_CONF     = 0.35
_DEFAULT_IOU      = 0.45
_INFER_SIZE       = 640
_DEPTH_PATCH_HALF = 3

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
    depth_mm: float  = 0.0
    cam_x_m:  float  = 0.0
    cam_y_m:  float  = 0.0
    cam_z_m:  float  = 0.0

    @property
    def has_3d(self) -> bool:
        return self.depth_mm > 0


# ---------------------------------------------------------------------------
# YOLO ONNX runner (same as detector_real_image.py)
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
# Frame cache + helpers (same as detector_real_image.py)
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


def _annotate(bgr: np.ndarray, detections: list) -> np.ndarray:
    out = bgr.copy()
    for det in detections:
        color = (0, 200, 0) if det.has_3d else (0, 120, 220)
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        if det.has_3d:
            label = (f'{det.label} {det.confidence:.2f} | '
                     f'z={det.cam_z_m:.2f}m x={det.cam_x_m:+.2f}m y={det.cam_y_m:+.2f}m')
        else:
            label = f'{det.label} {det.confidence:.2f} | depth?'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(det.y1 - 6, th + 4)
        cv2.rectangle(out, (det.x1, ty - th - 4), (det.x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (det.x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (det.cx, det.cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return out


# ---------------------------------------------------------------------------
# Input thread: blocks on Enter, signals main loop
# ---------------------------------------------------------------------------
def _input_worker(send_event: threading.Event, stop_event: threading.Event):
    print('Press Enter to publish current detections as TF frames. Type "q" + Enter to quit.')
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == 'q':
            stop_event.set()
            break
        send_event.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Detector node — send detections to robot as TF frames')
    parser.add_argument('--robot-ip',   default=os.environ.get('ROBOT_IP', '127.0.0.1'))
    parser.add_argument('--robot-port', type=int, default=int(os.environ.get('ROBOT_PORT', 5555)))
    parser.add_argument('--color-port', type=int, default=5557)
    parser.add_argument('--depth-port', type=int, default=5558)
    parser.add_argument('--model',      default=_DEFAULT_MODEL)
    parser.add_argument('--conf',       type=float, default=_DEFAULT_CONF)
    parser.add_argument('--iou',        type=float, default=_DEFAULT_IOU)
    parser.add_argument('--no-display', action='store_true')
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
    print('=' * 60)

    detector   = YOLODetector(args.model, conf=args.conf, iou=args.iou)
    zmq_client = ZMQNavClient(robot_ip=args.robot_ip, port=args.robot_port, timeout_ms=5000)
    cache      = _FrameCache()

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

    # Input thread
    send_event = threading.Event()
    stop_event = threading.Event()
    threading.Thread(target=_input_worker, args=(send_event, stop_event),
                     daemon=True).start()

    print('Waiting for first frames...')
    _intrinsics_printed = False
    detections_lock = threading.Lock()
    latest_detections: list = []

    while not stop_event.is_set():
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

        # Fill depth + 3D
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

        with detections_lock:
            latest_detections = dets

        # Display
        if not args.no_display:
            annotated = _annotate(bgr, dets)
            n_valid = sum(1 for d in dets if d.has_3d)
            cv2.putText(annotated, f'{len(dets)} det  {n_valid} with depth | Enter=send  q=quit',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('detector_node', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Send on Enter
        if send_event.is_set():
            send_event.clear()
            with detections_lock:
                to_send = [d for d in latest_detections if d.has_3d]

            if not to_send:
                print('[send] No detections with valid depth — nothing sent.')
                continue

            objects = [{'label': d.label, 'cam_x': d.cam_x_m,
                        'cam_y': d.cam_y_m, 'cam_z': d.cam_z_m}
                       for d in to_send]

            print(f'[send] Sending {len(objects)} detection(s) to robot...')
            reply = zmq_client.send_command('update_objects', objects=objects)

            if reply.get('status') == 'success':
                for f in reply.get('frames', []):
                    print(f'  → {f["frame"]:30s} map ({f["map_x"]:+.3f}, {f["map_y"]:+.3f})')
            else:
                print(f'  [error] {reply.get("message")}')

    if not args.no_display:
        cv2.destroyAllWindows()
    zmq_client.close()
    print('Stopped.')


if __name__ == '__main__':
    main()
