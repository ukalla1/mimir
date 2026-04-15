"""
Interactive detector node — runs YOLO on RealSense streams and sends detections
to the robot on demand (press Enter).

On Enter, the current detections (with valid 3D positions in camera_link frame)
are sent to zmq_bridge_node_working_v2.py via the 'update_objects' command.
The robot transforms them to map frame and broadcasts TF frames:
    detected_{label}_{i}  as children of  map

Usage:
    python detector_node_real_world.py --robot-ip 192.168.0.114 or 164 for realworld or default 127.0.0.1 for simulation

Verify on robot:
    ros2 tf echo map detected_bottle_0
    ros2 run tf2_tools view_frames
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np

from image_receiver import ImageReceiver
from zmq_client import ZMQNavClient

from detector_core import (
    YOLODetector, FrameCache, fill_depth_3d,
    DEFAULT_CONF, DEFAULT_IOU,
)

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
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
    parser.add_argument('--model',      default=os.path.join(_WEIGHTS_DIR, 'yolo11n.onnx'))
    parser.add_argument('--conf',       type=float, default=DEFAULT_CONF)
    parser.add_argument('--iou',        type=float, default=DEFAULT_IOU)
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
    cache      = FrameCache()

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
        fill_depth_3d(dets, color_frame, depth_frame)

        with detections_lock:
            latest_detections = dets

        # Display
        if not args.no_display:
            annotated = _annotate(bgr, dets)
            n_valid = sum(1 for d in dets if d.has_3d)
            cv2.putText(annotated, f'{len(dets)} det  {n_valid} with depth | Enter=send  q=quit',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('detector_node_real_world', annotated)
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
                skipped = reply.get('skipped', [])
                if skipped:
                    print(f'  ↷ skipped (already stored nearby): {", ".join(skipped)}')
                if not reply.get('frames') and not skipped:
                    print('  (nothing new to send)')
            else:
                print(f'  [error] {reply.get("message")}')

    if not args.no_display:
        cv2.destroyAllWindows()
    zmq_client.close()
    print('Stopped.')


if __name__ == '__main__':
    main()
