"""
ZMQ subscriber for RealSense color and depth streams.

The robot side publishes via realsense_zmq_node on two PUB sockets:
  color port (default 5557): topic='color'  encoding='rgb8'   pixels: HxWx3 uint8
  depth port (default 5558): topic='depth'  encoding='16UC1'  pixels: HxW   uint16 (mm)

Each message is a 3-part ZMQ multipart:
  [0] topic string (b'color' or b'depth')
  [1] JSON header  {"width":W,"height":H,"encoding":"...","stamp":...,
                    "cameraInfo":{"k":[9 floats]}}
  [2] raw pixel bytes (little-endian, row-major)

Usage:
    rx = ImageReceiver(robot_ip='192.168.0.114', color_port=5557, depth_port=5558,
                       recv_color=True, recv_depth=True)
    color_frame = rx.recv_color(timeout_ms=1000)
    depth_frame = rx.recv_depth(timeout_ms=1000)
    rx.close()
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class CameraInfo:
    k: list   # 9-element list [fx, 0, ppx, 0, fy, ppy, 0, 0, 1]


@dataclass
class ImageFrame:
    data:        np.ndarray
    width:       int
    height:      int
    encoding:    str
    timestamp:   float = 0.0
    camera_info: Optional[CameraInfo] = None


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------
class ImageReceiver:
    """
    Subscribes to one or both RealSense ZMQ PUB sockets.

    Use separate instances for color-only vs depth-only to avoid blocking
    one stream while waiting for the other.
    """

    def __init__(self,
                 robot_ip:   str  = '127.0.0.1',
                 color_port: int  = 5557,
                 depth_port: int  = 5558,
                 recv_color: bool = True,
                 recv_depth: bool = True):

        self._ctx        = zmq.Context()
        self._color_sock = None
        self._depth_sock = None

        if recv_color:
            s = self._ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, 2)
            s.setsockopt_string(zmq.SUBSCRIBE, 'realsense/color')
            s.connect(f'tcp://{robot_ip}:{color_port}')
            self._color_sock = s

        if recv_depth:
            s = self._ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, 2)
            s.setsockopt_string(zmq.SUBSCRIBE, 'realsense/depth')
            s.connect(f'tcp://{robot_ip}:{depth_port}')
            self._depth_sock = s

    # ------------------------------------------------------------------
    def recv_color(self, timeout_ms: int = 1000) -> Optional[ImageFrame]:
        if self._color_sock is None:
            return None
        if not self._color_sock.poll(timeout_ms):
            return None
        return self._recv_one(self._color_sock)

    def recv_depth(self, timeout_ms: int = 1000) -> Optional[ImageFrame]:
        if self._depth_sock is None:
            return None
        if not self._depth_sock.poll(timeout_ms):
            return None
        return self._recv_one(self._depth_sock)

    # ------------------------------------------------------------------
    def _recv_one(self, sock: zmq.Socket) -> Optional[ImageFrame]:
        try:
            parts = sock.recv_multipart()
        except zmq.ZMQError:
            return None

        if len(parts) < 3:
            return None

        # parts[0] = topic, parts[1] = JSON header, parts[2] = pixels
        try:
            header = json.loads(parts[1].decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        width    = int(header.get('width',  0))
        height   = int(header.get('height', 0))
        encoding = header.get('encoding', '')
        stamp    = float(header.get('stamp', 0.0))

        # Parse camera intrinsics if present
        ci = None
        ci_raw = header.get('cameraInfo') or header.get('camera_info')
        if ci_raw:
            k = ci_raw.get('k') or ci_raw.get('K')
            if k and len(k) == 9:
                ci = CameraInfo(k=k)

        # Decode pixel buffer
        raw = parts[2]
        try:
            if encoding in ('rgb8', 'bgr8'):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            elif encoding in ('16UC1', 'mono16'):
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(height, width).copy()
            elif encoding in ('8UC1', 'mono8'):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width).copy()
            else:
                # Fallback: try uint8 flat
                arr = np.frombuffer(raw, dtype=np.uint8).copy()
        except ValueError:
            return None

        return ImageFrame(data=arr, width=width, height=height,
                          encoding=encoding, timestamp=stamp,
                          camera_info=ci)

    # ------------------------------------------------------------------
    def close(self):
        for s in (self._color_sock, self._depth_sock):
            if s is not None:
                s.close()
        self._ctx.term()
