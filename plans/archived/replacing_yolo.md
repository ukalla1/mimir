# Plan: Replace YOLO with VLM Detection

## Context

YOLO (ONNX, 80 COCO classes, ~50ms) currently handles all object detection in the robot pipeline. We've confirmed Qwen3.5-9B with `--mmproj` produces excellent grounding output (Approach A: `<ref>label</ref><box>(x1, y1, x2, y2)</box>`) with open-vocabulary detection. Now we want to fully replace YOLO with the VLM.

**YOLO is used in 3 places:**
1. `detect_tool.py` → `_Scanner` class (agent tools: `scan_objects`, `register_objects`, `navigate_and_scan`)
2. `detector_node_real_world.py` (standalone manual detector)
3. `detector_node_real_world_auto.py` / `seperate_detector_real_world_auto.py` (standalone auto detector)

The agent tools in `detect_tool.py` are the primary target. The standalone scripts are secondary.

**Current YOLO pipeline in `_Scanner._detect_once()`:**
```
grab frame → bgr = cvtColor(color) → dets = detector.detect(bgr) → fill_depth_3d(dets) → filter by targets + has_3d
```

## Three Approaches to Replace YOLO

### Approach 1: Drop-in Replacement (VLM as detector, same pipeline shape)

Replace `YOLODetector` with a `VLMDetector` class that has the same `detect(bgr) → list[Detection]` interface. Everything downstream (depth sampling, backprojection, stability check, spatial dedup, bridge commands) stays identical.

**`VLMDetector.detect(bgr)` implementation:**
1. Base64-encode the BGR image
2. POST to llama-server with grounding prompt
3. Parse `<ref>label</ref><box>(x1, y1, x2, y2)</box>` response
4. Scale [0,1000] → pixel coordinates
5. Return `list[Detection]` with bbox and center pixel — same dataclass as YOLO

**Changes:**
- New class `VLMDetector` in `detector_core.py` (alongside `YOLODetector`, not replacing it)
- `_Scanner.__init__()` instantiates `VLMDetector` instead of `YOLODetector`
- No changes to `fill_depth_3d`, `scan()`, `register()`, `navigate_and_scan()`, or any bridge code

**Pros:** Minimal changes. All existing logic (stability, dedup, registration) works unchanged. Easy to A/B test — switch one line.
**Cons:** Slow (~3-5s per detection vs ~50ms). The `navigate_and_scan` polling loop currently scans every 2s — with VLM it would scan every ~5s, missing detections during fast movement. Doesn't leverage VLM's ability to understand depth visually.

### Approach 2: VLM with Depth Context (two-image input)

Same as Approach 1 but pass BOTH color + depth colormap to the VLM. The VLM returns bounding boxes AND distance estimates. We still sample actual depth for precision, but can use the VLM's distance estimate as a cross-check or fallback when depth pixels are invalid (0mm).

**Prompt:**
```
Image 1: RGB camera. Image 2: Depth map (blue=0m, red=16m).
Detect all objects. For each: <ref>label</ref><box>(x1, y1, x2, y2)</box>
Also estimate each object's distance from the depth map colors.
```

**Changes:**
- `VLMDetector.detect(bgr, depth=None)` — optionally accepts depth frame
- Generates depth colormap internally, sends both images
- Parses both boxes and distance estimates
- Primary depth from pixel sampling; VLM estimate used as fallback when `depth_mm == 0`

**Pros:** Better depth handling for noisy/missing depth pixels. VLM can reason about occluded objects.
**Cons:** Two images = more tokens = slower inference. Needs to verify llama-server handles 2 images reliably.

### Approach 3: VLM as Scene Analyzer (higher-level abstraction)

Don't replicate the YOLO detect→depth→backproject pipeline at all. Instead, give the VLM both images and ask it to directly output a scene description with object labels and distances. Skip bounding boxes entirely.

**Prompt:**
```
You are a robot camera. Analyze this scene using the RGB image and depth map.
List every object with: label, distance in metres, and position (left/center/right).
Output JSON: [{"label": "person", "distance_m": 2.3, "position": "center"}, ...]
```

The position (left/center/right) maps to rough angular sectors for the robot.

**Changes:**
- `VLMDetector` returns a new format: `list[dict]` with `label`, `distance_m`, `position`
- `_Scanner.scan()` adapts to the new format — no pixel coordinates, no backprojection
- `register()` would need a different approach: use robot's current pose + distance + angular sector to estimate map-frame position
- Approximate but may be sufficient for "go to the person" use cases

**Pros:** Simplest prompt, most natural for VLM. No coordinate parsing headaches. VLM handles depth reasoning end-to-end.
**Cons:** Least precise 3D positions. "Left/center/right" is coarse. Harder to integrate with existing spatial dedup and map-frame registration which expect exact coordinates.

---

## Recommendation

**Start with Approach 1** — it's the safest path and lets us test VLM detection quality in the real pipeline immediately. Once that works, Approach 2 is a straightforward enhancement (add depth colormap as second image).

Approach 3 is interesting for future exploration but requires rethinking the registration pipeline, which is out of scope for now.

## Files to Modify (Approach 1)

### 1. [nutri-atlas/robot_control/tools/detector_core.py](nutri-atlas/robot_control/tools/detector_core.py)

Add `VLMDetector` class alongside `YOLODetector`:

```python
class VLMDetector:
    """Drop-in replacement for YOLODetector using Qwen3.5 VLM grounding."""

    def __init__(self, server_url='http://localhost:8080/v1/chat/completions'):
        self.server_url = server_url
        print(f'[VLM] detector ready  server={server_url}')

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        # 1. Encode image
        _, buf = cv2.imencode('.jpg', bgr)
        b64 = base64.b64encode(buf).decode()

        # 2. Call VLM with grounding prompt
        response = self._call_vlm(b64)

        # 3. Parse <ref>label</ref><box>(x1, y1, x2, y2)</box>
        h, w = bgr.shape[:2]
        results = []
        seen = set()
        for match in re.finditer(r'<ref>(.*?)</ref>\s*<box>(.*?)</box>', response):
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
                    cx=(x1+x2)//2, cy=(y1+y2)//2,
                ))
        return results
```

### 2. [nutri-atlas/robot_control/tools/detect_tool.py](nutri-atlas/robot_control/tools/detect_tool.py)

In `_Scanner.__init__()`, replace:
```python
self.detector = YOLODetector(DEFAULT_MODEL, conf=DEFAULT_CONF, iou=DEFAULT_IOU)
```
with:
```python
self.detector = VLMDetector()
```

Update import accordingly. No other changes needed — `_detect_once()` calls `self.detector.detect(bgr)` which returns the same `list[Detection]`.

### 3. Standalone scripts (optional, later)

`detector_node_real_world.py` and `detector_node_real_world_auto.py` can also switch to `VLMDetector`, but since they run continuously at frame rate, the 3-5s VLM latency makes them less useful. Keep YOLO as an option there.

## Speed Consideration

The `navigate_and_scan` loop currently uses `scan_interval=2.0` (scan every 2 seconds). With VLM inference taking ~3-5s, the effective scan interval becomes ~5-7s. For the initial test this is acceptable — the robot moves slowly. If needed, increase `scan_interval` to 6.0 to avoid overlapping requests.

## Verification

1. Start the server with `--mmproj`:
   ```bash
   cd ~/work/atlas/mimir/nutri_rag && bash scripts/start_server.sh
   ```

2. Run the robot assistant:
   ```bash
   cd ~/work/atlas/mimir/nutri-atlas/robot_control
   python robot_assistant.py --robot-ip 192.168.0.164 --detection-mode real
   ```

3. Test `scan_objects`:
   ```
   User: What do you see?
   ```
   Expected: VLM detects objects with labels + 3D positions (not limited to COCO 80).

4. Test `register_objects`:
   ```
   User: Remember what you see.
   ```
   Expected: Objects persisted to bridge, same as before.

5. Test `navigate_and_scan`:
   ```
   User: Go to the kitchen and look for objects on the way.
   ```
   Expected: Periodic VLM scans during navigation (fewer scans than YOLO due to latency).

## Out of Scope

- Approach 2 (depth colormap as second image) — enhancement after Approach 1 works
- Approach 3 (scene analyzer) — future exploration
- Real-time VLM detection in standalone scripts
- VLM-based spatial dedup (let the VLM decide if it's the same object)
