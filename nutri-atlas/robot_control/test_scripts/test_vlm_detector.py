"""
Test VLM-based object detection on a saved color+depth pair.

Two approaches:
  A — Grounding: VLM outputs <ref>label</ref><box>(x1, y1, x2, y2)</box>
      Then we sample depth at the box center and backproject to 3D.
  B — Depth visual: VLM reads a colorized depth map and estimates distances directly.

Usage:
    cd nutri-atlas/robot_control
    python scripts/test_vlm_detector.py --approach A
    python scripts/test_vlm_detector.py --approach B
    python scripts/test_vlm_detector.py                # both
"""
import argparse
import base64
import json
import os
import re
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data')

DEFAULT_COLOR = os.path.join(_DATA_DIR, 'color_0001.png')
DEFAULT_DEPTH = os.path.join(_DATA_DIR, 'depth_0001.npy')

# Camera intrinsics from RealSense (captured during save session)
FX, FY, PPX, PPY = 387.4, 387.4, 320.8, 238.4

LLM_URL = 'http://localhost:8080/v1/chat/completions'
DEPTH_PATCH_HALF = 3


# ---------------------------------------------------------------------------
# Depth helpers (standalone — no import from detector_core to keep this
# self-contained for testing)
# ---------------------------------------------------------------------------
def sample_depth(depth: np.ndarray, cx: int, cy: int) -> float:
    """Median depth (mm) in a small patch around (cx, cy)."""
    h, w = depth.shape[:2]
    n = DEPTH_PATCH_HALF
    x1, x2 = max(0, cx - n), min(w, cx + n + 1)
    y1, y2 = max(0, cy - n), min(h, cy + n + 1)
    patch = depth[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0


def backproject(cx: int, cy: int, depth_mm: float):
    """Back-project pixel + depth to camera-frame 3D (metres)."""
    z = depth_mm / 1000.0
    x = (cx - PPX) * z / FX
    y = (cy - PPY) * z / FY
    return x, y, z


# ---------------------------------------------------------------------------
# VLM call helper
# ---------------------------------------------------------------------------
def _encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def vlm_call(images_b64: list[str], prompt: str, max_tokens: int = 2048) -> str:
    """Send one or more images + text prompt to llama-server. Returns text."""
    content = []
    for b64 in images_b64:
        content.append({'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{b64}'}})
    content.append({'type': 'text', 'text': prompt})

    payload = {
        'model': 'unsloth/Qwen3.5-9B-GGUF',
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': max_tokens,
        'temperature': 0.0,
    }
    req = urllib.request.Request(
        LLM_URL,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data['choices'][0]['message']['content']


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def _parse_json_list(text: str) -> list[dict]:
    """Parse a JSON array or newline-delimited JSON objects from text."""
    text = _strip_markdown_fences(text)
    # Try as a JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Fall back to line-by-line
    results = []
    for line in text.split('\n'):
        line = line.strip().rstrip(',')
        if not line.startswith('{'):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results


# ---------------------------------------------------------------------------
# Approach A — Grounding boxes
# ---------------------------------------------------------------------------
def approach_a(color_path: str, depth: np.ndarray):
    print('\n' + '=' * 60)
    print('Approach A: Grounding (bounding boxes)')
    print('=' * 60)

    b64 = _encode_image(color_path)
    prompt = (
        'Detect every visible object in this image — including food items, drinks, '
        'appliances, furniture, containers, people, signs, fixtures, and any other '
        'distinct items. Be thorough; do not skip an object just because it is in '
        'the background.\n'
        'For each object, output exactly:\n'
        '<ref>specific_label</ref><box>(x1, y1, x2, y2)</box>\n'
        'Use SPECIFIC labels — name what each object actually is. Examples:\n'
        '  - Food: "bread", "carrot", "avocado", "orange", "apple", "peanut butter", '
        '"lettuce", "chicken" (NEVER just "food")\n'
        '  - Drinks: "water bottle", "sprite can", "coca cola", "orange juice", '
        '"milk carton", "wine glass", "coffee mug" (NEVER just "drink" or just "bottle"/"can" — '
        'include the contents or brand when visible)\n'
        '  - Appliances: "microwave", "coffee maker", "toaster", "blender"\n'
        '  - Furniture/fixtures: "drawer", "cupboard", "door", "handle", '
        '"fire extinguisher"\n'
        '  - People: "mannequin", "person"\n'
        'NEVER use generic labels: "food", "drink", "object", "item", "toy", "thing". '
        'If you are uncertain about a specific food or drink, identify it by color, '
        'shape, or container type (e.g. "round red fruit", "leafy green vegetable", '
        '"clear bottle with green label") instead of "food" or "drink".\n'
        'Coordinates are in [0, 1000] normalized range.\n'
        'Output ONLY the detection lines, nothing else.'
    )

    print('[VLM] Sending grounding prompt...')
    response = vlm_call([b64], prompt)
    print(f'[VLM] Response:\n{response}\n')

    # Parse <ref>label</ref><box>(...)</box>
    # Format: one or more (x1, y1, x2, y2) tuples per <box> tag
    h, w = depth.shape[:2]
    results = []
    seen = set()

    for match in re.finditer(r'<ref>(.*?)</ref>\s*<box>(.*?)(?:</box>|$)', response, re.MULTILINE):
        label = match.group(1).strip()
        box_str = match.group(2)
        for coords in re.findall(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', box_str):
            x1_n, y1_n, x2_n, y2_n = [int(v) for v in coords]
            # [0, 1000] → pixel
            x1 = int(x1_n / 1000 * w)
            y1 = int(y1_n / 1000 * h)
            x2 = int(x2_n / 1000 * w)
            y2 = int(y2_n / 1000 * h)

            # Dedup
            key = (label, x1_n, y1_n, x2_n, y2_n)
            if key in seen:
                continue
            seen.add(key)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            d_mm = sample_depth(depth, cx, cy)
            cam_x, cam_y, cam_z = backproject(cx, cy, d_mm) if d_mm > 0 else (0, 0, 0)
            results.append({
                'label': label, 'bbox': [x1, y1, x2, y2],
                'cx': cx, 'cy': cy, 'depth_mm': d_mm,
                'cam_x': cam_x, 'cam_y': cam_y, 'cam_z': cam_z,
            })

    if not results:
        # Fallback: ask for JSON bbox format
        print('[A] No grounding format found, trying JSON fallback...')
        prompt_fb = (
            'Detect all objects in this image (640×480 pixels). '
            'For each object, output one JSON object per line:\n'
            '{"label": "object_name", "bbox_2d": [x1, y1, x2, y2]}\n'
            'where x1,y1,x2,y2 are pixel coordinates in [0, 1000] normalized range. '
            'Output ONLY JSON, nothing else.'
        )
        response = vlm_call([b64], prompt_fb)
        print(f'[VLM] Fallback response:\n{response}\n')
        results = _parse_json_boxes_fallback(response, depth)

    _print_results(results, 'A')
    return results


def _parse_json_boxes_fallback(response: str, depth: np.ndarray):
    """Parse JSON bbox_2d objects from fallback response with dedup."""
    h, w = depth.shape[:2]
    results = []
    seen = set()

    for obj in _parse_json_list(response):
        label = obj.get('label', '?')
        bbox = obj.get('bbox_2d') or obj.get('bbox', [0, 0, 0, 0])
        x1_n, y1_n, x2_n, y2_n = [int(v) for v in bbox[:4]]

        # Dedup
        key = (label, x1_n, y1_n, x2_n, y2_n)
        if key in seen:
            continue
        seen.add(key)

        # [0, 1000] → pixel
        x1 = int(x1_n / 1000 * w)
        y1 = int(y1_n / 1000 * h)
        x2 = int(x2_n / 1000 * w)
        y2 = int(y2_n / 1000 * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        d_mm = sample_depth(depth, cx, cy)
        cam_x, cam_y, cam_z = backproject(cx, cy, d_mm) if d_mm > 0 else (0, 0, 0)
        results.append({
            'label': label, 'bbox': [x1, y1, x2, y2],
            'cx': cx, 'cy': cy, 'depth_mm': d_mm,
            'cam_x': cam_x, 'cam_y': cam_y, 'cam_z': cam_z,
        })
    _print_results(results, 'A-fallback')
    return results


# ---------------------------------------------------------------------------
# Approach B — VLM reads colorized depth
# ---------------------------------------------------------------------------
def approach_b(color_path: str, depth: np.ndarray):
    print('\n' + '=' * 60)
    print('Approach B: VLM reads depth colormap')
    print('=' * 60)

    # Create colorized depth with scale bar
    max_depth_m = 16.0
    depth_clipped = np.clip(depth.astype(np.float32), 0, max_depth_m * 1000)
    depth_norm = (depth_clipped / (max_depth_m * 1000) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Add scale bar on the right side
    bar_w = 40
    bar = np.zeros((depth_color.shape[0], bar_w, 3), dtype=np.uint8)
    for row in range(bar.shape[0]):
        val = int(row / bar.shape[0] * 255)
        bar[row, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                         cv2.COLORMAP_JET)[0, 0]
    # Labels on scale bar
    for i, m in enumerate([0, 4, 8, 12, 16]):
        y = int(i / 4 * (bar.shape[0] - 1))
        cv2.putText(bar, f'{m}m', (2, max(y, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    depth_vis = np.hstack([depth_color, bar])

    # Save composite for debugging
    composite_path = os.path.join(_DATA_DIR, 'depth_colormap_0001.png')
    cv2.imwrite(composite_path, depth_vis)
    print(f'[B] Saved depth colormap → {composite_path}')

    b64_color = _encode_image(color_path)
    b64_depth = base64.b64encode(cv2.imencode('.png', depth_vis)[1]).decode()

    prompt = (
        'You are given two images of the same scene.\n'
        'Image 1: RGB camera image.\n'
        'Image 2: Depth map where blue=close (0m) and red=far (16m). '
        'A scale bar on the right shows the distance mapping.\n\n'
        'For each object you can identify, estimate its distance from the camera '
        'using the depth map colors.\n'
        'Output one JSON object per line:\n'
        '{"label": "object_name", "distance_m": 2.5}\n'
        'Output ONLY JSON lines, nothing else.'
    )

    print('[VLM] Sending color + depth colormap...')
    response = vlm_call([b64_color, b64_depth], prompt)
    print(f'[VLM] Response:\n{response}\n')

    results = []
    for obj in _parse_json_list(response):
        results.append({
            'label': obj.get('label', '?'),
            'distance_m': obj.get('distance_m', 0),
        })
    _print_results_b(results)
    return results


# ---------------------------------------------------------------------------
# Visualization & printing
# ---------------------------------------------------------------------------
def _print_results(results: list, tag: str):
    if not results:
        print(f'[{tag}] No detections.')
        return
    print(f'[{tag}] {len(results)} detection(s):')
    print(f'  {"Label":<20s} {"Center":>12s} {"Depth":>8s} {"3D (cam frame)":>30s}')
    print(f'  {"-"*20} {"-"*12} {"-"*8} {"-"*30}')
    for r in results:
        cx, cy = r.get('cx', 0), r.get('cy', 0)
        d = r['depth_mm']
        if d > 0:
            pos = f'x={r["cam_x"]:+.2f}  y={r["cam_y"]:+.2f}  z={r["cam_z"]:.2f}m'
        else:
            pos = 'no depth'
        print(f'  {r["label"]:<20s} ({cx:3d}, {cy:3d})  {d:7.0f}mm  {pos}')


def _print_results_b(results: list):
    if not results:
        print('[B] No detections.')
        return
    print(f'[B] {len(results)} detection(s):')
    print(f'  {"Label":<20s} {"Distance":>10s}')
    print(f'  {"-"*20} {"-"*10}')
    for r in results:
        print(f'  {r["label"]:<20s} {r["distance_m"]:>8.2f} m')


def draw_results(color_path: str, results_a: list):
    """Draw bboxes on the image and save."""
    bgr = cv2.imread(color_path)
    if bgr is None:
        return

    for r in (results_a or []):
        bbox = r.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if r['depth_mm'] > 0:
                label = f'{r["label"]} z={r["cam_z"]:.1f}m'
            else:
                label = r['label']
            cv2.putText(bgr, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    out_path = os.path.join(_DATA_DIR, 'vlm_result_0001.png')
    cv2.imwrite(out_path, bgr)
    print(f'\nSaved annotated image → {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test VLM-based object detection')
    parser.add_argument('--color', default=DEFAULT_COLOR)
    parser.add_argument('--depth', default=DEFAULT_DEPTH)
    parser.add_argument('--approach', choices=['A', 'B'],
                        help='Run a single approach (default: both)')
    args = parser.parse_args()

    bgr = cv2.imread(args.color)
    if bgr is None:
        print(f'ERROR: cannot read {args.color}')
        return
    depth = np.load(args.depth)
    print(f'Loaded color: {bgr.shape}  depth: {depth.shape} [{depth.min()}–{depth.max()}] mm')

    results_a, results_b = None, None

    if args.approach in (None, 'A'):
        results_a = approach_a(args.color, depth)
    if args.approach in (None, 'B'):
        results_b = approach_b(args.color, depth)

    # Draw visualization for approach A
    if results_a:
        draw_results(args.color, results_a)


if __name__ == '__main__':
    main()
