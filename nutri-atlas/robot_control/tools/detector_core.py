"""
Shared YOLO detection primitives used by:
  - detector_node_real_world.py      (manual Enter-to-send)
  - detector_node_real_world_auto.py (automatic stability gate)
  - detect_tool.py                   (agent on-demand scan tool)

Contains: YOLODetector, Detection dataclass, FrameCache, depth helpers, COCO class names.
"""

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

try:
    from .image_receiver import ImageFrame
except ImportError:
    from image_receiver import ImageFrame


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
WEIGHTS_DIR      = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')
DEFAULT_MODEL    = os.path.join(WEIGHTS_DIR, 'yolo11n.onnx')
DEFAULT_CONF     = 0.35
DEFAULT_IOU      = 0.45
DEPTH_PATCH_HALF = 3

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
# YOLO ONNX runner
# ---------------------------------------------------------------------------
class YOLODetector:
    def __init__(self, model_path: str, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU):
        self.conf = conf
        self.iou  = iou
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(model_path, providers=providers)
        inp             = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = inp.shape
        print(f'[YOLO] {model_path}  input={inp.shape}'
              f'  provider={self.session.get_providers()[0]}')

    def detect(self, bgr: np.ndarray) -> list[Detection]:
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
# Thread-safe frame cache
# ---------------------------------------------------------------------------
class FrameCache:
    def __init__(self):
        self._color = None
        self._depth = None
        self._lock  = threading.Lock()

    def put_color(self, f: ImageFrame):
        with self._lock:
            self._color = f

    def put_depth(self, f: ImageFrame):
        with self._lock:
            self._depth = f

    def get(self) -> tuple:
        with self._lock:
            return self._color, self._depth


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------
def sample_depth(depth_frame: ImageFrame, cx: int, cy: int) -> float:
    """Sample median depth in a small patch around (cx, cy). Returns mm."""
    d = depth_frame.data
    h, w = d.shape[:2]
    n = DEPTH_PATCH_HALF
    x1 = max(0, cx - n); x2 = min(w, cx + n + 1)
    y1 = max(0, cy - n); y2 = min(h, cy + n + 1)
    patch = d[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0


def backproject(depth_frame: ImageFrame, dcx: int, dcy: int, depth_mm: float):
    """Back-project a depth pixel to camera-frame 3D coordinates (metres)."""
    if depth_frame.camera_info is None or len(depth_frame.camera_info.k) < 9:
        return 0.0, 0.0, 0.0
    k = depth_frame.camera_info.k
    fx, fy, ppx, ppy = k[0], k[4], k[2], k[5]
    if fx == 0 or fy == 0:
        return 0.0, 0.0, 0.0
    z_m = depth_mm / 1000.0
    return (dcx - ppx) * z_m / fx, (dcy - ppy) * z_m / fy, z_m


def fill_depth_3d(detections: list[Detection], color_frame: ImageFrame, depth_frame: ImageFrame):
    """Fill depth_mm and cam_x/y/z_m for each detection using the depth frame."""
    if depth_frame is None:
        return
    dh, dw = depth_frame.data.shape[:2]
    ch, cw = color_frame.data.shape[:2]
    sx, sy = dw / cw, dh / ch
    for det in detections:
        dcx = int(min(max(det.cx * sx, 0), dw - 1))
        dcy = int(min(max(det.cy * sy, 0), dh - 1))
        det.depth_mm = sample_depth(depth_frame, dcx, dcy)
        if det.depth_mm > 0:
            det.cam_x_m, det.cam_y_m, det.cam_z_m = backproject(
                depth_frame, dcx, dcy, det.depth_mm)
