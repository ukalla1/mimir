"""
Real-time object detection on RealSense color+depth streams.
Uses ONNX Runtime — no PyTorch required.

Pipeline:
  1. Two background threads keep the latest color and depth frames fresh.
  2. Main loop grabs a matched pair, runs YOLO (ONNX) on the color image.
  3. For each detected bounding box, sample the depth value (mm) at the
     bounding-box centre, convert to metres, and annotate the frame.

Requirements:
    pip install onnxruntime-gpu opencv-python numpy pyzmq

Getting the ONNX model (one-time, on any machine with working ultralytics):
    pip install ultralytics
    yolo export model=yolo11n.pt format=onnx imgsz=640
    # copy yolo11n.onnx next to this script

Usage:
    python detector_real_image.py --robot-ip 192.168.0.114
    python detector_real_image.py --model yolo11n.onnx --conf 0.4
    python detector_real_image.py --no-display --save-dir /tmp/det

Environment variables (override CLI defaults):
    ROBOT_IP   — robot IP (default 127.0.0.1)
"""

import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from image_reciver import ImageReceiver, ImageFrame


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_MODEL    = 'yolo11n.onnx'
_DEFAULT_CONF     = 0.35
_DEFAULT_IOU      = 0.45
_INFER_SIZE       = 640            # YOLO input resolution
_DEPTH_PATCH_HALF = 3              # sample a (2N+1)x(2N+1) patch and take median

# COCO 80-class names (index = class id)
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
# Detection result
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int                 # bounding box centre x
    cy: int                 # bounding box centre y
    depth_mm: float         # median depth at centre (0 = invalid)

    @property
    def depth_m(self) -> float:
        return self.depth_mm / 1000.0


# ---------------------------------------------------------------------------
# Frame cache: two threads write, main thread reads
# ---------------------------------------------------------------------------
class _FrameCache:
    def __init__(self):
        self._color: Optional[ImageFrame] = None
        self._depth: Optional[ImageFrame] = None
        self._lock  = threading.Lock()

    def put_color(self, f: ImageFrame):
        with self._lock:
            self._color = f

    def put_depth(self, f: ImageFrame):
        with self._lock:
            self._depth = f

    def get(self):
        """Return (color_frame, depth_frame) snapshot — either may be None."""
        with self._lock:
            return self._color, self._depth


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------
def _sample_depth(depth_frame: ImageFrame, cx: int, cy: int) -> float:
    """
    Return median depth (mm) in a small patch around (cx, cy).
    Returns 0.0 if the patch contains no valid (non-zero) pixels.
    """
    d = depth_frame.data           # (H, W) uint16, mm
    h, w = d.shape[:2]
    n = _DEPTH_PATCH_HALF

    x1 = max(0, cx - n); x2 = min(w, cx + n + 1)
    y1 = max(0, cy - n); y2 = min(h, cy + n + 1)

    patch = d[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# Annotate frame
# ---------------------------------------------------------------------------
def _annotate(bgr: np.ndarray, detections: list) -> np.ndarray:
    out = bgr.copy()
    for det in detections:
        color = (0, 200, 0) if det.depth_mm > 0 else (0, 100, 200)
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        if det.depth_mm > 0:
            label = f'{det.label} {det.confidence:.2f} | {det.depth_m:.2f}m'
        else:
            label = f'{det.label} {det.confidence:.2f} | depth?'

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(det.y1 - 6, th + 4)
        cv2.rectangle(out, (det.x1, ty - th - 4), (det.x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (det.x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.drawMarker(out, (det.cx, det.cy), (255, 255, 255),
                       cv2.MARKER_CROSS, 10, 1)
    return out


# ---------------------------------------------------------------------------
# YOLO ONNX runner
# ---------------------------------------------------------------------------
class YOLODetector:
    """
    Wraps a YOLO v8/v11 ONNX model exported with:
        yolo export model=yolo11n.pt format=onnx imgsz=640

    ONNX output layout: [1, 4+num_classes, num_anchors]
        → cx, cy, w, h (in model-input pixels) + per-class scores
    """

    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.45):
        self.conf = conf
        self.iou  = iou
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session   = ort.InferenceSession(model_path, providers=providers)
        inp            = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = inp.shape   # typically 1,3,640,640
        print(f'[YOLO] {model_path}  input={inp.shape}'
              f'  provider={self.session.get_providers()[0]}')

    def detect(self, bgr: np.ndarray) -> list:
        ih, iw = bgr.shape[:2]

        # Preprocess: resize → RGB → float32 NCHW [0,1]
        resized = cv2.resize(bgr, (self.w, self.h))
        blob    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob    = blob.transpose(2, 0, 1)[np.newaxis]   # (1,3,H,W)

        raw  = self.session.run(None, {self.input_name: blob})[0]  # (1, 4+C, A)
        pred = raw[0].T                    # (A, 4+C)

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

        # Scale boxes from model input size → original image size
        sx = iw / self.w;  sy = ih / self.h
        cx_ = boxes_xywh[:, 0] * sx;  cy_ = boxes_xywh[:, 1] * sy
        bw  = boxes_xywh[:, 2] * sx;  bh  = boxes_xywh[:, 3] * sy
        x1  = cx_ - bw / 2;  y1 = cy_ - bh / 2
        x2  = cx_ + bw / 2;  y2 = cy_ + bh / 2

        # NMS
        boxes_cv = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(boxes_cv, confidences.tolist(), self.conf, self.iou)
        if len(keep) == 0:
            return []

        detections = []
        for i in keep.flatten():
            ix1, iy1 = int(x1[i]), int(y1[i])
            ix2, iy2 = int(x2[i]), int(y2[i])
            label = (COCO_NAMES[class_ids[i]]
                     if class_ids[i] < len(COCO_NAMES) else str(class_ids[i]))
            detections.append(Detection(
                label=label, confidence=float(confidences[i]),
                x1=ix1, y1=iy1, x2=ix2, y2=iy2,
                cx=(ix1 + ix2) // 2, cy=(iy1 + iy2) // 2,
                depth_mm=0.0,
            ))
        return detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='YOLO ONNX + RealSense ZMQ detector')
    parser.add_argument('--robot-ip',   default=os.environ.get('ROBOT_IP', '127.0.0.1'))
    parser.add_argument('--color-port', type=int, default=5557)
    parser.add_argument('--depth-port', type=int, default=5558)
    parser.add_argument('--model',      default=_DEFAULT_MODEL,
                        help='YOLO ONNX model path (default: yolo11n.onnx)')
    parser.add_argument('--conf',       type=float, default=_DEFAULT_CONF)
    parser.add_argument('--iou',        type=float, default=_DEFAULT_IOU)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--save-dir',   default=None,
                        help='Save annotated frames as JPEGs here')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'[ERROR] Model not found: {args.model}')
        print('Export it on any machine with ultralytics:')
        print('  pip install ultralytics')
        print('  yolo export model=yolo11n.pt format=onnx imgsz=640')
        print('  # copy yolo11n.onnx here')
        return

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    detector = YOLODetector(args.model, conf=args.conf, iou=args.iou)
    cache    = _FrameCache()

    def _color_worker():
        rx = ImageReceiver(robot_ip=args.robot_ip,
                           color_port=args.color_port,
                           depth_port=args.depth_port,
                           recv_color=True, recv_depth=False)
        try:
            while True:
                f = rx.recv_color(timeout_ms=2000)
                if f is not None:
                    cache.put_color(f)
        finally:
            rx.close()

    def _depth_worker():
        rx = ImageReceiver(robot_ip=args.robot_ip,
                           color_port=args.color_port,
                           depth_port=args.depth_port,
                           recv_color=False, recv_depth=True)
        try:
            while True:
                f = rx.recv_depth(timeout_ms=2000)
                if f is not None:
                    cache.put_depth(f)
        finally:
            rx.close()

    for fn in (_color_worker, _depth_worker):
        threading.Thread(target=fn, daemon=True).start()

    print(f'Connecting to {args.robot_ip}  color:{args.color_port}  depth:{args.depth_port}')
    print('Waiting for first frames... (press q to quit)')

    saved = 0; fps_t = time.time(); fps_count = 0

    while True:
        color_frame, depth_frame = cache.get()
        if color_frame is None:
            time.sleep(0.01)
            continue

        bgr = (cv2.cvtColor(color_frame.data, cv2.COLOR_RGB2BGR)
               if color_frame.encoding == 'rgb8' else color_frame.data)

        detections = detector.detect(bgr)

        # Fill depth for each detection
        for det in detections:
            if depth_frame is not None:
                dh, dw = depth_frame.data.shape[:2]
                dcx = min(max(det.cx, 0), dw - 1)
                dcy = min(max(det.cy, 0), dh - 1)
                det.depth_mm = _sample_depth(depth_frame, dcx, dcy)

        # Console output once per second
        fps_count += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_t = time.time(); fps_count = 0
            print(f'[fps={fps:.1f}]  {len(detections)} detection(s)')
            for det in detections:
                depth_str = f'{det.depth_m:.2f}m' if det.depth_mm > 0 else 'no depth'
                print(f'  {det.label:20s}  conf={det.confidence:.2f}'
                      f'  centre=({det.cx},{det.cy})  depth={depth_str}')

        annotated = _annotate(bgr, detections)

        if not args.no_display:
            cv2.imshow('detector', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.save_dir:
            ts   = int(color_frame.timestamp * 1000)
            path = os.path.join(args.save_dir, f'det_{ts:016d}.jpg')
            cv2.imwrite(path, annotated)
            saved += 1

    if not args.no_display:
        cv2.destroyAllWindows()
    print(f'Done. saved={saved}')


if __name__ == '__main__':
    main()
