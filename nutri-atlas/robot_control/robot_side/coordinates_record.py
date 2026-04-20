#!/usr/bin/env python3
"""
Record robot global coordinates from TF into a JSON file.

Usage:
    python coordinates_record.py
    python coordinates_record.py --output landmarks_record.json
    python coordinates_record.py --map-frame map --base-frames base_link base

Behavior:
    - Press Enter to record current (x, y) in map frame.
    - Type "q" (then Enter) to quit.
    - Records are appended to the output JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener


class CoordinateRecorder(Node):
    def __init__(self, map_frame: str, base_frames: list[str]):
        super().__init__("coordinate_recorder")
        self._map_frame = map_frame
        self._base_frames = base_frames
        self._active_base_frame = base_frames[0]

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def get_pose(self):
        for base_frame in self._base_frames:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame,
                    base_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.2),
                )
                self._active_base_frame = base_frame
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                return {
                    "x": float(t.x),
                    "y": float(t.y),
                    "yaw_rad": float(yaw),
                    "map_frame": self._map_frame,
                    "base_frame": self._active_base_frame,
                }
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
        return None


def _load_records(path: Path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_records(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Record robot map coordinates to JSON.")
    parser.add_argument(
        "--output",
        default="recorded_coordinates.json",
        help="Output JSON file path (default: recorded_coordinates.json)",
    )
    parser.add_argument(
        "--map-frame",
        default="map",
        help="Global frame for coordinates (default: map)",
    )
    parser.add_argument(
        "--base-frames",
        nargs="+",
        default=["base_link", "base", "vehicle"],
        help="Robot base frame candidates in priority order.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    records = _load_records(output_path)

    rclpy.init()
    node = CoordinateRecorder(map_frame=args.map_frame, base_frames=args.base_frames)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Coordinate recorder started.")
    print(f"Output file: {output_path}")
    print(f"Map frame: {args.map_frame}")
    print(f"Base frame candidates: {args.base_frames}")
    print('Press Enter to record current coordinate, or type "q" then Enter to quit.\n')

    try:
        while True:
            user_in = input("> ").strip().lower()
            if user_in == "q":
                break

            pose = node.get_pose()
            if pose is None:
                print("TF not available yet. Check localization/TF tree, then try again.")
                continue

            entry = {
                "index": len(records) + 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "x": round(pose["x"], 4),
                "y": round(pose["y"], 4),
                "yaw_rad": round(pose["yaw_rad"], 4),
                "map_frame": pose["map_frame"],
                "base_frame": pose["base_frame"],
            }
            records.append(entry)
            _save_records(output_path, records)
            print(
                f'Recorded #{entry["index"]}: x={entry["x"]}, y={entry["y"]} '
                f'(base={entry["base_frame"]})'
            )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"Saved {len(records)} total records to: {output_path}")


if __name__ == "__main__":
    main()
