# Plan: VLM-based Object Detection with Depth Extraction

## Context

Currently, the robot detection pipeline uses YOLO (ONNX) to detect objects and extract bounding boxes, then samples the paired depth image at each detection's center pixel to compute 3D camera-frame positions. YOLO is fast but limited to COCO's 80 classes and doesn't understand scene context.

Qwen3.5-9B is a multimodal model already running on our server with `--mmproj`. We want to test whether it can replace YOLO for detection by:
1. Sending the color image to the VLM with a grounding/detection prompt
2. Parsing bounding boxes from the response
3. Using the same `sample_depth` + `backproject` pipeline from `detector_core.py` to get 3D positions

**Test data:** `nutri-atlas/robot_control/data/color_0001.png` (640x480 BGR) + `depth_0001.npy` (640x480 uint16 mm)

## Approaches: Three Ways to Get 3D Object Info from the VLM

The YOLO pipeline follows a rigid path: bbox → center pixel → sample depth → backproject. A VLM is more flexible. We should test multiple approaches to see which gives the best results.

### Approach A: Grounding (VLM outputs bounding boxes)

Most similar to YOLO. Qwen3 VL supports a grounding format where it returns bounding boxes as normalized coordinates:
```
<ref>person</ref><box>(x1, y1),(x2, y2)</box>
```
Coordinates in [0, 1000] → scale to pixel coords → sample depth at center → backproject.

**Pros:** Structured output, easy to parse, reuses existing `sample_depth` + `backproject` from `detector_core.py`.
**Cons:** May not work via llama-server (grounding tokens might be transformers-only). Need to test.

### Approach B: VLM reads the depth image directly

Pass BOTH the color image AND a colorized depth visualization (JET colormap with distance legend) to the VLM. Ask it to identify objects AND estimate their distances from the color-coded depth map.

```
[color image] [depth colormap with scale bar]
"Here are two views of the same scene. Left: RGB camera. Right: depth map where blue=close (0m), red=far (16m).
For each object you detect, report: label, approximate distance in metres."
```

**Pros:** No coordinate parsing needed. VLM reasons about depth visually — may handle occluded/noisy depth better than naive center-pixel sampling. Leverages the VLM's scene understanding.
**Cons:** Distance estimates are approximate, not precise 3D positions. May be enough for navigation ("person is ~2m away") but not for precise map-frame registration.

### Approach C: VLM outputs center points + we sample depth

Instead of full bounding boxes, ask the VLM to output the approximate center pixel of each object. Simpler output format, easier to parse, and `sample_depth` only needs the center pixel anyway.

```
"Detect all objects in this image (640×480 pixels). For each object, output a JSON line:
{"label": "person", "cx": 320, "cy": 240}
where cx, cy are the approximate center pixel coordinates."
```

Then sample depth at `(cx, cy)` and backproject using camera intrinsics.

**Pros:** Simpler than bboxes. Only needs the center point, which is all we use for depth anyway.
**Cons:** Still requires the VLM to output pixel coordinates accurately.

## Test Plan

Create a single test script that tries all three approaches on the saved data pair. This lets us compare results side-by-side.

### File to create: [nutri-atlas/robot_control/scripts/test_vlm_detector.py](nutri-atlas/robot_control/scripts/test_vlm_detector.py)

```
Usage:
    python test_vlm_detector.py --approach A   # grounding boxes
    python test_vlm_detector.py --approach B   # VLM reads depth colormap
    python test_vlm_detector.py --approach C   # center points + depth sampling
    python test_vlm_detector.py                # runs all three, compares
```

**Shared setup:**
1. Load `color_0001.png` (BGR) and `depth_0001.npy` (uint16 mm)
2. Base64-encode color image for VLM input
3. Camera intrinsics: `fx=387.4, fy=387.4, ppx=320.8, ppy=238.4`

**Approach A implementation:**
- Prompt: `"Detect all objects in this image. For each, output: <ref>label</ref><box>(x1, y1),(x2, y2)</box>"`
- Parse with regex, scale [0,1000] → pixels
- `sample_depth` + `backproject` at box center
- If `<ref><box>` format fails, fall back to asking for JSON bboxes

**Approach B implementation:**
- Generate depth colormap: `cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)` with a scale bar showing 0–16m
- Side-by-side composite: `np.hstack([color, depth_colormap])`
- Prompt: `"Left: RGB camera image. Right: depth map (blue=0m, red=16m). List every object with its label and estimated distance in metres. Output JSON: {\"label\": \"...\", \"distance_m\": ...}"`
- Parse JSON from response — no pixel coordinates or backprojection needed

**Approach C implementation:**
- Prompt: `"Detect all objects in this image (640×480). For each, output JSON: {\"label\": \"...\", \"cx\": ..., \"cy\": ...} where cx,cy are center pixel coordinates."`
- Parse JSON, then `sample_depth(cx, cy)` + `backproject`

**Visualization (all approaches):**
- Print table: `label | distance | 3D position (if available) | approach`
- For A and C: draw boxes/points on color image, save to `data/vlm_result_0001.png`

## Camera Intrinsics

From the earlier capture session output:
```
K: fx=387.4  fy=387.4  ppx=320.8  ppy=238.4
```

These need to be hardcoded in the test script (since we don't save camera_info with the depth npy). For future-proofing, we could save intrinsics alongside, but that's out of scope for this test.

## Verification

1. Run the test script:
   ```bash
   cd nutri-atlas/robot_control
   python scripts/test_vlm_detector.py
   ```
2. Check console output — should list detected objects with labels, bounding boxes, and 3D positions
3. Check `data/vlm_result_0001.png` — should show annotated bounding boxes on the color image
4. Compare with what YOLO found in the same scene (person + chair from earlier)

## Expected Tradeoffs

| | YOLO (current) | VLM Approach A/C | VLM Approach B |
|---|---|---|---|
| Speed | ~50ms | ~2-5s | ~2-5s |
| Object classes | 80 COCO | Open vocabulary | Open vocabulary |
| Depth precision | Exact (pixel sampling) | Exact (pixel sampling) | Approximate (VLM estimate) |
| Scene understanding | None | High | Highest |

## Open Questions

- **Grounding format**: Does llama-server + mmproj support `<ref><box>` grounding tokens? This is the first thing to test.
- **Coordinate accuracy**: How precise are VLM-output pixel coordinates? Even ±20px error at the center is fine for depth sampling (the depth patch is 7×7 pixels).
- **Two-image input**: Can llama-server accept two images in one message (needed for Approach B)?

## Out of Scope

- Integration into `detect_tool.py` or `robot_assistant.py` (wait for test results)
- Saving camera intrinsics alongside depth data
- Real-time VLM detection






# Plan: Fix VLM Detector Test Script — Parsing Bugs

## Context

Ran `test_vlm_detector.py` on `color_0001.png` + `depth_0001.npy`. Results:

- **Approach A**: VLM grounding output was excellent (person, chair, monitor, lamp, window, ceiling fan, bag, desk, cup, box) but TWO parsing bugs prevented depth extraction:
  1. Grounding format is `<box>(x1, y1, x2, y2)</box>` (single 4-value tuple), not `<box>(x1, y1),(x2, y2)</box>` (two tuples). Regex didn't match.
  2. Fallback JSON used key `bbox_2d`, not `bbox`. Parser returned `(0, 0)` for all centers.

- **Approach B**: Worked correctly. Clean JSON output with distance estimates (person 2.5m, chair 2.5m, window 5.0m, etc.).

- **Approach C**: VLM ignored `cx/cy` format, returned `bbox_2d` instead, then hallucinated ~40 duplicate "bottle" entries. Drop this approach.

## Fixes in [nutri-atlas/robot_control/scripts/test_vlm_detector.py](nutri-atlas/robot_control/scripts/test_vlm_detector.py)

### Fix 1: Grounding regex (Approach A)

The actual VLM output format:
```
<ref>person</ref><box>(277, 569, 500, 998)</box>
<ref>chair</ref><box>(384, 701, 564, 998), (644, 748, 804, 998)</box>
```

Note: multiple boxes per `<ref>` tag are comma-separated within one `<box>` tag.

**Old regex:**
```python
r'<ref>(.*?)</ref>\s*<box>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)</box>'
```

**New approach:** Parse `<ref>` and `<box>` separately, then extract all `(x1, y1, x2, y2)` tuples from each box tag:
```python
# Match ref+box pairs
for match in re.finditer(r'<ref>(.*?)</ref>\s*<box>(.*?)</box>', response):
    label = match.group(1)
    box_str = match.group(2)
    # Extract all (x1, y1, x2, y2) tuples
    for coords in re.findall(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', box_str):
        x1, y1, x2, y2 = [int(v) for v in coords]
        # Scale from [0,1000] to pixels
        ...
```

### Fix 2: JSON key (Approach A fallback)

**Old:** `obj.get('bbox', [0, 0, 0, 0])`
**New:** `obj.get('bbox_2d') or obj.get('bbox', [0, 0, 0, 0])`

### Fix 3: Drop Approach C

Remove or skip — it produces hallucinated repetitions and the VLM doesn't follow the `cx/cy` format.

### Fix 4: JSON parsing robustness

The fallback response wraps JSON in ` ```json ... ``` ` markdown fences and uses a JSON array, not one-per-line. Need to:
1. Strip markdown fences
2. Try `json.loads()` on the whole response as an array first
3. Fall back to line-by-line parsing

### Fix 5: Dedup fallback results

The fallback JSON repeated the same detections ~4 times. Add dedup by `(label, bbox)` tuple.

## Verification

```bash
cd ~/work/atlas/mimir/nutri-atlas/robot_control
python scripts/test_vlm_detector.py --approach A
```

Expected: detections with correct pixel coordinates and depth values, not `(0, 0)`.
