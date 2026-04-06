"""
Receiver for RealSense images broadcast by the realsense_zmq node on the robot.

Wire format (ZMQ PUB/SUB, multipart):
    Frame 0 : topic string bytes  — b"realsense/color" or b"realsense/depth"
    Frame 1 : JSON header — {msgType, width, height, encoding, step,
                              stampSec, stampNanosec, frameId,
                              cameraInfo: {...}}   (cameraInfo may be absent)
    Frame 2 : raw pixel bytes (row-major, as defined by step × height)

Ports (defined in zmq.ports.hpp on robot side):
    5557 — color  (encoding typically "rgb8" or "bgr8")
    5558 — depth  (encoding typically "16UC1", millimetres)

Usage (standalone test):
    export ROBOT_IP=192.168.0.114
    python tools/image_reciver.py           # live display with OpenCV
    python tools/image_reciver.py --no-display --save-dir /tmp/frames
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Port constants (mirror zmq.ports.hpp)
# ---------------------------------------------------------------------------
_DEFAULT_COLOR_PORT = 5557
_DEFAULT_DEPTH_PORT = 5558


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CameraInfo:
    width: int
    height: int
    distortion_model: str
    k: list          # 3×3 intrinsic matrix (row-major, 9 floats)
    d: list          # distortion coefficients
    r: list          # rectification matrix (9 floats)
    p: list          # projection matrix (12 floats)


@dataclass
class ImageFrame:
    topic: str
    width: int
    height: int
    encoding: str
    step: int                        # bytes per row
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    data: np.ndarray                 # shape (height, width, C) or (height, width)
    camera_info: Optional[CameraInfo] = None

    @property
    def timestamp(self) -> float:
        return self.stamp_sec + self.stamp_nanosec * 1e-9


# ---------------------------------------------------------------------------
# Encoding → numpy dtype / channels
# ---------------------------------------------------------------------------
_ENCODING_MAP = {
    'rgb8':   (np.uint8,  3),
    'bgr8':   (np.uint8,  3),
    'rgba8':  (np.uint8,  4),
    'bgra8':  (np.uint8,  4),
    'mono8':  (np.uint8,  1),
    'mono16': (np.uint16, 1),
    '16UC1':  (np.uint16, 1),   # RealSense depth (mm)
    '32FC1':  (np.float32, 1),
}


def _decode_image(header: dict, raw: bytes) -> np.ndarray:
    """Convert raw bytes + header into a numpy array."""
    encoding = header['encoding']
    h = header['height']
    w = header['width']

    dtype, channels = _ENCODING_MAP.get(encoding, (np.uint8, 3))
    arr = np.frombuffer(raw, dtype=dtype)

    if channels == 1:
        arr = arr.reshape((h, w))
    else:
        arr = arr.reshape((h, w, channels))

    return arr


def _parse_camera_info(jc: dict) -> CameraInfo:
    return CameraInfo(
        width=jc['width'],
        height=jc['height'],
        distortion_model=jc.get('distortionModel', ''),
        k=jc.get('k', []),
        d=jc.get('d', []),
        r=jc.get('r', []),
        p=jc.get('p', []),
    )


# ---------------------------------------------------------------------------
# ImageReceiver
# ---------------------------------------------------------------------------
class ImageReceiver:
    """
    Subscribes to one ZMQ PUB endpoint and delivers ImageFrame objects.

    Args:
        robot_ip:   IP of the robot running realsense_zmq_node.
        color_port: ZMQ port for color images (default 5557).
        depth_port: ZMQ port for depth images (default 5558).
        recv_color: Whether to subscribe to color stream.
        recv_depth: Whether to subscribe to depth stream.
    """

    def __init__(self,
                 robot_ip: str = '127.0.0.1',
                 color_port: int = _DEFAULT_COLOR_PORT,
                 depth_port: int = _DEFAULT_DEPTH_PORT,
                 recv_color: bool = True,
                 recv_depth: bool = True):
        self._ctx = zmq.Context()
        self._sockets: list[tuple[zmq.Socket, str]] = []  # (socket, topic)

        if recv_color:
            s = self._ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, 2)          # keep only latest frames
            s.setsockopt(zmq.LINGER, 0)
            s.connect(f'tcp://{robot_ip}:{color_port}')
            s.setsockopt_string(zmq.SUBSCRIBE, 'realsense/color')
            self._sockets.append((s, 'realsense/color'))

        if recv_depth:
            s = self._ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, 2)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(f'tcp://{robot_ip}:{depth_port}')
            s.setsockopt_string(zmq.SUBSCRIBE, 'realsense/depth')
            self._sockets.append((s, 'realsense/depth'))

        # poller lets recv_any() wait across both sockets
        self._poller = zmq.Poller()
        for sock, _ in self._sockets:
            self._poller.register(sock, zmq.POLLIN)

    def recv_any(self, timeout_ms: int = 1000) -> Optional[ImageFrame]:
        """
        Block up to timeout_ms and return the next ImageFrame from any stream,
        or None on timeout.
        """
        ready = dict(self._poller.poll(timeout_ms))
        for sock, _ in self._sockets:
            if ready.get(sock) == zmq.POLLIN:
                return self._recv_one(sock)
        return None

    def recv_color(self, timeout_ms: int = 1000) -> Optional[ImageFrame]:
        """Block and return the next color frame, or None on timeout."""
        if not self._sockets:
            return None
        sock = self._sockets[0][0]
        if sock.poll(timeout_ms):
            return self._recv_one(sock)
        return None

    def recv_depth(self, timeout_ms: int = 1000) -> Optional[ImageFrame]:
        """Block and return the next depth frame, or None on timeout."""
        if len(self._sockets) < 2:
            return None
        sock = self._sockets[-1][0]
        if sock.poll(timeout_ms):
            return self._recv_one(sock)
        return None

    def _recv_one(self, sock: zmq.Socket) -> Optional[ImageFrame]:
        try:
            parts = sock.recv_multipart()
        except zmq.ZMQError:
            return None

        if len(parts) != 3:
            return None

        topic_bytes, header_bytes, image_bytes = parts
        topic = topic_bytes.decode('utf-8')

        try:
            header = json.loads(header_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            return None

        data = _decode_image(header, image_bytes)

        cam_info = None
        if 'cameraInfo' in header:
            try:
                cam_info = _parse_camera_info(header['cameraInfo'])
            except (KeyError, TypeError):
                pass

        return ImageFrame(
            topic=topic,
            width=header['width'],
            height=header['height'],
            encoding=header['encoding'],
            step=header['step'],
            stamp_sec=header['stampSec'],
            stamp_nanosec=header['stampNanosec'],
            frame_id=header.get('frameId', ''),
            data=data,
            camera_info=cam_info,
        )

    def close(self):
        for sock, _ in self._sockets:
            sock.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Standalone test / viewer
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='RealSense ZMQ image receiver')
    parser.add_argument('--robot-ip',    default=os.environ.get('ROBOT_IP', '127.0.0.1'))
    parser.add_argument('--color-port',  type=int, default=_DEFAULT_COLOR_PORT)
    parser.add_argument('--depth-port',  type=int, default=_DEFAULT_DEPTH_PORT)
    parser.add_argument('--no-display',  action='store_true', help='Disable OpenCV window')
    parser.add_argument('--save-dir',    default=None, help='Directory to save received frames as PNG')
    parser.add_argument('--max-frames',  type=int, default=0, help='Stop after N frames (0 = unlimited)')
    args = parser.parse_args()

    if not args.no_display:
        import cv2  # noqa: F401 — fail early if not installed

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f'Connecting to {args.robot_ip}  color:{args.color_port}  depth:{args.depth_port}')
    print('Press Ctrl+C to stop.\n')

    saved = 0
    received = 0

    with ImageReceiver(robot_ip=args.robot_ip,
                       color_port=args.color_port,
                       depth_port=args.depth_port) as rx:
        while True:
            frame = rx.recv_any(timeout_ms=2000)
            if frame is None:
                print('[warn] timeout — no frame received')
                continue

            received += 1
            is_color = 'color' in frame.topic
            kind = 'color' if is_color else 'depth'

            print(f'[{kind}] {frame.width}×{frame.height}  enc={frame.encoding}'
                  f'  t={frame.timestamp:.3f}  shape={frame.data.shape}')

            if not args.no_display:
                import cv2
                if is_color and frame.encoding == 'rgb8':
                    disp = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
                elif not is_color:
                    # Normalise depth (16UC1, mm) to 0–255 for display
                    d = frame.data.astype(np.float32)
                    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
                else:
                    disp = frame.data

                cv2.imshow(f'realsense/{kind}', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.save_dir:
                import cv2
                ts = int(frame.timestamp * 1000)
                fname = os.path.join(args.save_dir, f'{kind}_{ts:016d}.png')
                if is_color and frame.encoding == 'rgb8':
                    cv2.imwrite(fname, cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR))
                elif not is_color:
                    cv2.imwrite(fname, frame.data)
                else:
                    cv2.imwrite(fname, frame.data)
                saved += 1
                print(f'  saved → {fname}')

            if args.max_frames and received >= args.max_frames:
                break

    if not args.no_display:
        import cv2
        cv2.destroyAllWindows()

    print(f'\nDone. received={received} saved={saved}')


if __name__ == '__main__':
    main()
