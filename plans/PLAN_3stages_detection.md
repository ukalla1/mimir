# Plan: 3-Stage Detection Pipeline (Detect → Check → Store)

## Problem
Currently `scan_objects` detects objects and immediately persists them to the robot's landmark history (`~/detected_objects.json`). This causes:
- Objects detected in passing get permanently stored even if unwanted
- No stability check — single-frame false positives become landmarks
- No way for the agent to "just look" without side effects
- Repeated/overlapping detections of the same physical object

## Goal
Separate detection into 3 stages so the agent has full control:
1. **Detect** — store in temp memory only (no side effects)
2. **Check** — multi-frame stability + interest filter + spatial dedup
3. **Store** — persist to bridge only after all checks pass

---

## Architecture

```
scan_objects  →  YOLO  →  temp memory  →  return to agent
                                              |
                              agent decides to register
                                              |
                                              v
register_objects  →  5-frame stability  →  interest filter  →  spatial dedup  →  bridge
```

---

## Stage 1: Detect (temp memory)

**Tool**: `scan_objects` (modify existing)

Changes:
- Do NOT send `update_objects` to bridge anymore
- Store detections in a temp dict on the operator PC (`_Scanner._temp_objects`)
- Return the list of detected objects with labels, confidence, and 3D positions
- Temp memory is overwritten on each scan (latest frame only)

Used for:
- Answering "what can you see" / "what's around"
- Quick inspection without polluting landmark history

---

## Stage 2: Try to store (check + register)

**Tool**: `register_objects` (new)

When the agent decides detected objects should become persistent landmarks, it calls this tool. The tool performs 3 checks:

### a) Multi-frame stability
- Run 5 quick YOLO scans (~0.3s apart, ~1.5s total)
- Only keep objects detected in >= 3 of 5 frames
- Average the 3D positions across the frames where detected
- Purpose: filter out transient false positives

### b) Interest list filter
- Configurable list of labels to track (e.g. `["bottle", "cup", "chair"]`)
- If the list is non-empty, only objects with matching labels pass
- If the list is empty, all objects pass (default behavior)
- Configured via env var `INTEREST_OBJECTS` (comma-separated) or empty for all

### c) Spatial dedup against landmarks history
- For each stable object, fetch existing landmarks from bridge (`get_detected_objects`)
- Check if any existing landmark with the same label is within 0.5m
- If overlap found → skip (already known)
- If no overlap → pass to Stage 3
- Threshold: 0.5m for all classes (can be made per-class later)

### Parameters
```python
@register_tool('register_objects')
class RegisterObjects(BaseTool):
    parameters = [
        {
            'name': 'targets',
            'type': 'string',
            'description': 'Comma-separated labels to register (e.g. "bottle,cup"). '
                           'Empty = use default interest list.',
            'required': False,
        },
    ]
```

---

## Stage 3: Store

Objects that pass all 3 checks are sent to the bridge via `update_objects`:
- Bridge transforms camera-frame → map-frame
- Bridge broadcasts static TF frames (`detected_{label}_{i}`)
- Bridge saves to `~/detected_objects.json` (persistent across sessions)
- Bridge's own spatial dedup acts as a final safety net

---

## Files to change

| File | Change |
|------|--------|
| `nutri-atlas/robot_control/tools/detect_tool.py` | Modify `scan_objects`: stop calling bridge, store in temp memory. Add new `register_objects` tool with 3-stage check. |
| `nutri-atlas/robot_control/robot_assistant.py` | Add `register_objects` to TOOL_DEFINITIONS + tools dict. Update system prompt with 2-step protocol. |
| `nutri-atlas/robot_control/tools/object_tool.py` | No changes — `get_current_detected_objects` already reads from bridge's in-memory store. |

### Bridge (`zmq_bridge_node_working_v2.py`): NO changes needed
The `update_objects` command already handles TF broadcast + JSON persistence + spatial dedup.

---

## Updated system prompt protocol

```
Scanning rules:
- "what can you see" → call scan_objects. Returns temp detections, does NOT store anything.
- "inspect" / "scan" → call scan_objects. Report findings to user.
- "explore" / "look around" → spin 90° + scan_objects, repeat 4 times. Report findings. Do NOT navigate or register.
- "remember this" / "save these objects" → call register_objects to persist stable detections as landmarks.

IMPORTANT:
- scan_objects is non-destructive — it only looks, never stores to landmark history.
- register_objects is the ONLY way to add detected objects to landmark history.
- Never call register_objects unless the user asks to remember/save objects, or you are explicitly searching for a specific object.

To find a specific object (only when user asks "find X" or "go to X"):
1. Call get_detected_objects — if found, navigate directly.
2. If not found, call register_objects(targets="X") at current location — this scans, validates, and stores in one step.
3. If still not found, do a 360° explore with register_objects after each spin.
4. If still not found, navigate to next landmark and repeat from step 2.
5. If all landmarks exhausted, report not found.
```

---

## Data flow examples

### "What can you see?"
```
User: what can you see?
Agent: calls scan_objects
       → YOLO runs → temp memory updated → returns ["bottle", "chair", "person"]
Agent: "I can see a bottle, a chair, and a person."
       (nothing stored permanently)
```

### "Explore and remember what you find"
```
User: explore and remember what you find
Agent: spin 90° → scan_objects → spin 90° → scan_objects → ... (4 times)
Agent: calls register_objects
       → 5-frame stability check → interest filter → spatial dedup
       → sends qualified objects to bridge → stored in landmarks
Agent: "I found and registered: bottle (new), chair (new). Skipped: person (already known)."
```

### "Find the cup"
```
User: find the cup
Agent: calls get_detected_objects → cup not in history
Agent: calls register_objects(targets="cup")
       → 5-frame scan → cup found in 4/5 frames → passes stability
       → not in history → sends to bridge → stored
Agent: "Found a cup! Navigating to it."
Agent: calls navigate_to_landmark with cup coordinates
```

---

## Verification

1. Run `robot_assistant.py --robot-ip 192.168.0.164 --detection-mode real`
2. "What can you see?" → scan_objects returns objects, nothing persisted
3. Check `~/detected_objects.json` on robot — unchanged
4. "Remember what you see" → register_objects runs, objects persisted
5. Check `~/detected_objects.json` — new entries added
6. "Explore and save" → spin + scan + register cycle
7. Repeat "explore and save" → spatial dedup prevents duplicates
8. "Find the bottle" → register + navigate sequence
