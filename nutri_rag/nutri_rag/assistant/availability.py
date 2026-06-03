"""Food availability provider for the robot assistant (Phase B).

Maps "what foods are physically available right now" into a set of fdc_ids
that the recommendation pipeline can use as a hard filter on the candidate
pool. Three sources are supported:

    "json"  — read the same detected_objects.json the robot's
              zmq_object_server already reads, but locally (server-side).
              Default location: nutri_rag/data/detected_objects.json.
              Lets us test the full pipeline without a live robot.

    "zmq"   — query the live robot via the ZMQ object server on port 5556.
              Mirrors what tools/object_tool.py:GetDetectedObjects does.

    "none"  — return None → no availability filter, current behavior.

Selected via the AVAILABILITY_SOURCE env var or the source= argument.
Frame names in the JSON map look like "detected_apple_0" — we strip the
leading "detected_" and trailing "_<num>" to get a clean label, then run a
text-only top-1 lookup over the food index to map label → fdc_id.

The label → fdc_id mapping is cached per process so each label is looked
up once. Labels whose top-1 text score falls below SIM_THRESHOLD are
treated as non-food objects and excluded.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

# Default file the robot's zmq_object_server reads from, mirrored locally
# at the server-side default so tests need no robot.
DEFAULT_AVAILABILITY_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "detected_objects.json",
)

# Min text-cosine to accept a label as a real food. Calibrated against the
# Qwen3-Embedding-0.6B model: real foods score ~0.62-0.70, common YOLO/COCO
# non-food labels score ~0.45-0.56. Threshold 0.58 separates them in practice
# but the blocklist below catches the edge cases where similar prefixes fool
# the embedder (e.g. "chair" ≈ "cheddar", "cat" ≈ "catsup").
SIM_THRESHOLD = 0.58

# Common COCO/YOLO non-food labels that occasionally pass the threshold due
# to text-embedding noise (substring overlap). Override via the
# AVAILABILITY_BLOCKLIST env var (comma-separated).
_NONFOOD_BLOCKLIST = {
    "person", "chair", "couch", "sofa", "bed", "table", "tv", "monitor",
    "laptop", "mouse", "keyboard", "remote", "phone", "cell phone", "book",
    "clock", "vase", "scissors", "toothbrush", "hair drier", "umbrella",
    "handbag", "backpack", "suitcase", "tie", "bench", "toilet",
    "potted plant", "dining table", "refrigerator", "microwave", "oven",
    "toaster", "sink", "car", "truck", "bus", "motorcycle", "bicycle",
    "airplane", "boat", "train", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "bird",
}

# Frame-name pattern: "detected_<label>_<idx>" → label
# Also accepts plain labels like "apple" without the detected_ prefix.
_FRAME_RE = re.compile(r"^(?:detected_)?(.+?)(?:_\d+)?$", re.IGNORECASE)

_label_cache: dict[str, int | None] = {}


def _label_from_frame_name(frame: str) -> str:
    """detected_apple_0 → apple ; bottle_2 → bottle ; apple → apple"""
    m = _FRAME_RE.match(frame.strip())
    if not m:
        return frame.strip()
    return m.group(1).replace("_", " ").lower()


def _read_object_json(path: str) -> set[str]:
    """Parse the same detected_objects.json the robot's object server reads.

    File format (matches robot_side/zmq_bridge_simulation/zmq_object_server.py):
        {"detected_apple_0": {"px": ..., "py": ...}, ...}
    """
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    if not isinstance(data, dict):
        return set()
    return {_label_from_frame_name(k) for k in data.keys()}


def _query_zmq_objects(addr: str | None = None) -> set[str]:
    """Query the robot's ZMQ object server (port 5556 by default).

    Mirrors object_tool._ObjectClient.get_objects but doesn't depend on
    qwen_agent. Returns the set of cleaned object labels.
    """
    try:
        import zmq
    except ImportError:
        return set()
    if addr is None:
        ip = os.environ.get("OBJECT_SERVER_IP", "10.203.168.250")
        port = int(os.environ.get("OBJECT_SERVER_PORT", 5556))
        addr = f"tcp://{ip}:{port}"
    timeout_ms = int(os.environ.get("OBJECT_TIMEOUT_MS", 3000))
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addr)
    try:
        sock.send_string(json.dumps({"action": "get_objects"}))
        reply = json.loads(sock.recv_string())
    except (zmq.Again, OSError):
        return set()
    finally:
        sock.close()
    objects = reply.get("objects") or {}
    if not isinstance(objects, dict):
        return set()
    return {_label_from_frame_name(k) for k in objects.keys()}


def _label_to_fdc_id(label: str, threshold: float = SIM_THRESHOLD) -> int | None:
    """Resolve a label to its best matching fdc_id via text top-1.

    Cached per process. Returns None if either:
    - the label is in the non-food blocklist (chairs, persons, etc.), or
    - the top-1 text cosine is below the threshold.
    """
    if label in _label_cache:
        return _label_cache[label]

    block_env = os.environ.get("AVAILABILITY_BLOCKLIST", "")
    extra_blocklist = {x.strip().lower() for x in block_env.split(",") if x.strip()}
    blocked = _NONFOOD_BLOCKLIST | extra_blocklist
    if label.lower().strip() in blocked:
        _label_cache[label] = None
        return None

    # Import here to avoid loading the text embedder unless availability is
    # actually being looked up. Reuses Phase A's unified primitive.
    from nutri_rag.search import _get_embedder, hybrid_rank
    from nutri_rag.embedding import FOOD_SEARCH_INSTRUCTION

    embedder = _get_embedder()
    q_text = embedder.encode([label], task_instruction=FOOD_SEARCH_INSTRUCTION)[0]
    df = hybrid_rank(q_text=q_text, q_gat=None, alpha=0.0, k=1)

    if df.empty:
        _label_cache[label] = None
        return None
    top_score = float(df.iloc[0]["text_sim"])
    if top_score < threshold:
        _label_cache[label] = None
        return None
    fid = int(df.iloc[0]["fdc_id"])
    _label_cache[label] = fid
    return fid


def get_available_fdc_ids(
    source: str | None = None,
    json_path: str | None = None,
    zmq_addr: str | None = None,
) -> set[int] | None:
    """Return the set of currently-available fdc_ids, or None for no filter.

    source values:
        "json" — read the detected_objects.json mirror (default location:
                 nutri_rag/data/detected_objects.json, override via the
                 AVAILABILITY_PATH env var or the json_path argument).
        "zmq"  — query the live robot's object server.
        "none" — no availability filter (returns None, graceful fallback).

    If source is None, reads the AVAILABILITY_SOURCE env var; default "none".
    """
    if source is None:
        source = os.environ.get("AVAILABILITY_SOURCE", "none")

    if source == "none":
        return None

    if source == "json":
        path = json_path or os.environ.get("AVAILABILITY_PATH", DEFAULT_AVAILABILITY_JSON)
        labels = _read_object_json(path)
    elif source == "zmq":
        labels = _query_zmq_objects(zmq_addr)
    else:
        raise ValueError(f"unknown availability source: {source!r}")

    fdc_ids: set[int] = set()
    for label in labels:
        fid = _label_to_fdc_id(label)
        if fid is not None:
            fdc_ids.add(fid)
    return fdc_ids if fdc_ids else None
