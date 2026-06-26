"""
Shared YOLO detection primitives used by:
  - detector_node_real_world.py      (manual Enter-to-send)
  - detector_node_real_world_auto.py (automatic stability gate)
  - detect_tool.py                   (agent on-demand scan tool)

Contains: YOLODetector, Detection dataclass, FrameCache, depth helpers, COCO class names.
"""

import base64
import json
import os
import re
import threading
import urllib.request
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
# VLM grounding detector (drop-in replacement for YOLODetector)
# ---------------------------------------------------------------------------
_VLM_GROUNDING_PROMPT = (
    'Detect all food items, dishes, utensils, and household objects in this image. '
    'For each object, output exactly:\n'
    '<ref>specific_label</ref><box>(x1, y1, x2, y2)</box>\n'
    'Use SPECIFIC labels — name the food or object (e.g. "bread", "carrot", '
    '"avocado", "cup", "plate"). Do NOT use generic labels like "toy", "object", '
    'or "item" even if the items look like plastic models. Identify them by what '
    'food/object they represent.\n'
    'Coordinates are in [0, 1000] normalized range.\n'
    'Output ONLY the detection lines, nothing else.'
)


class VLMDetector:
    """VLM-based object detector using Qwen3.5 grounding.

    Two modes:
      detect(bgr)  — same interface as YOLODetector (used by YOLO path, kept for compat)
      detect_from_files(color_path, depth_path) — loads saved frames, runs grounding,
          samples depth, and returns list[Detection] with 3D positions filled in.

    Requires llama-server running with --mmproj.
    """

    # Camera intrinsics (RealSense, from capture session)
    FX, FY, PPX, PPY = 387.4, 387.4, 320.8, 238.4

    def __init__(self, server_url: str = 'http://localhost:8080/v1/chat/completions',
                 max_tokens: int = 2048):
        self.server_url = server_url
        self.max_tokens = max_tokens
        print(f'[VLM] detector ready  server={server_url}')

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Run VLM grounding on a BGR image. Returns detections WITHOUT depth."""
        return self._grounding(bgr)

    def detect_from_files(self, color_path: str, depth_path: str) -> list[Detection]:
        """Run VLM grounding on saved files. Returns detections WITH 3D positions.

        1. Load color + depth from disk
        2. VLM grounding → bounding boxes
        3. Sample depth at each box center → backproject to camera frame 3D
        """
        bgr = cv2.imread(color_path)
        if bgr is None:
            print(f'[VLM] cannot read {color_path}')
            return []
        depth = np.load(depth_path)

        dets = self._grounding(bgr)

        # Fill 3D positions from depth
        dh, dw = depth.shape[:2]
        ch, cw = bgr.shape[:2]
        sx, sy = dw / cw, dh / ch
        for det in dets:
            dcx = int(min(max(det.cx * sx, 0), dw - 1))
            dcy = int(min(max(det.cy * sy, 0), dh - 1))
            det.depth_mm = sample_depth_arr(depth, dcx, dcy)
            if det.depth_mm > 0:
                z = det.depth_mm / 1000.0
                det.cam_x_m = (dcx - self.PPX) * z / self.FX
                det.cam_y_m = (dcy - self.PPY) * z / self.FY
                det.cam_z_m = z

        return dets

    def _grounding(self, bgr: np.ndarray) -> list[Detection]:
        """Send image to VLM, parse <ref><box> response → list[Detection]."""
        h, w = bgr.shape[:2]

        _, buf = cv2.imencode('.jpg', bgr)
        b64 = base64.b64encode(buf).decode()
        print(f'[VLM] sending {len(b64)//1024}KB image ({w}x{h}) to server...')

        response = self._call_vlm(b64)
        print(f'[VLM] response ({len(response)} chars): {response[:300]}')

        results = []
        seen = set()
        for match in re.finditer(r'<ref>(.*?)</ref>\s*<box>(.*?)(?:</box>|$)', response, re.MULTILINE):
            label = match.group(1).strip()
            for coords in re.findall(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', match.group(2)):
                x1_n, y1_n, x2_n, y2_n = [int(v) for v in coords]
                key = (label, x1_n, y1_n, x2_n, y2_n)
                if key in seen:
                    continue
                seen.add(key)
                x1 = int(x1_n / 1000 * w)
                y1 = int(y1_n / 1000 * h)
                x2 = int(x2_n / 1000 * w)
                y2 = int(y2_n / 1000 * h)
                results.append(Detection(
                    label=label, confidence=1.0,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=(x1 + x2) // 2, cy=(y1 + y2) // 2,
                ))

        print(f'[VLM] detected {len(results)} object(s): '
              f'{", ".join(r.label for r in results) or "none"}')
        return results

    def _call_vlm(self, image_b64: str) -> str:
        payload = {
            'model': 'unsloth/Qwen3.5-9B-GGUF',
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                    {'type': 'text', 'text': _VLM_GROUNDING_PROMPT},
                ],
            }],
            'max_tokens': self.max_tokens,
            'temperature': 0.0,
        }
        req = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data['choices'][0]['message']['content']
        except Exception as e:
            import traceback
            print(f'[VLM] request failed: {e}')
            traceback.print_exc()
            return ''


def sample_depth_arr(depth: np.ndarray, cx: int, cy: int) -> float:
    """Median depth (mm) in a small patch around (cx, cy). Standalone version for numpy arrays."""
    h, w = depth.shape[:2]
    n = DEPTH_PATCH_HALF
    x1, x2 = max(0, cx - n), min(w, cx + n + 1)
    y1, y2 = max(0, cy - n), min(h, cy + n + 1)
    patch = depth[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0


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
