# Plan: Enable Vision Input for Robot Assistant

## Context

Qwen3.5-9B is a multimodal model that supports image input. The current setup only uses text. Enabling vision requires:
1. An mmproj GGUF file (the multimodal projector) — not yet downloaded
2. `--mmproj` flag in `start_server.sh`
3. Image-passing support in the `robot_assistant.py` chat loop

The `model_type: 'qwenvl_oai'` is already set in `robot_assistant.py:407`, so Qwen Agent is already configured for vision — only the server flag and input parsing are missing.

---

## Step 0: Download the mmproj file

The file is available at `unsloth/Qwen3.5-9B-GGUF` on HuggingFace. Three variants exist:
- `mmproj-BF16.gguf` (922 MB) ← recommended (smaller, same quality as F16 on modern GPUs)
- `mmproj-F16.gguf` (918 MB)
- `mmproj-F32.gguf` (1.82 GB)

Download command:
```python
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='unsloth/Qwen3.5-9B-GGUF',
    filename='mmproj-BF16.gguf',
    local_dir='/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF'
)"
```

---

## Files to modify

### 1. [nutri_rag/scripts/start_server.sh](nutri_rag/scripts/start_server.sh)

Add `MMPROJ_PATH` variable and `--mmproj` flag to the llama-server command:

```bash
MODEL_PATH="/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
MMPROJ_PATH="/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
PORT=8080

~/softwares/llama.cpp/llama-server \
    --model "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --port "$PORT" \
    ...
```

### 2. [nutri-atlas/robot_control/robot_assistant.py](nutri-atlas/robot_control/robot_assistant.py)

**Image input syntax**: User prefixes their message with `@/path/to/image.jpg`. Example:
```
User: @/tmp/photo.jpg what objects do you see in this image?
```

**Changes in `main()`** at the input loop (~line 434):

```python
import base64, mimetypes

def _build_user_content(query: str):
    """Return content for the user message — plain string or multimodal list."""
    if query.startswith('@'):
        # Split on first space: "@/path/img.jpg rest of query"
        parts = query[1:].split(' ', 1)
        img_path = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else 'What do you see?'
        img_path = os.path.expanduser(img_path)
        mime = mimetypes.guess_type(img_path)[0] or 'image/jpeg'
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return [
            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}},
            {'type': 'text', 'text': text},
        ]
    return query  # plain text — existing behaviour unchanged

# In the while loop, replace:
#   messages.append({'role': 'user', 'content': query})
# with:
#   content = _build_user_content(query)
#   messages.append({'role': 'user', 'content': content})
```

No changes to `_run_turn`, `TOOL_DEFINITIONS`, or `SYSTEM_MSG` needed.

---

## Verification

1. Restart the server after adding `--mmproj`:
   ```bash
   cd nutri_rag && bash scripts/start_server.sh
   ```
2. In `robot_assistant.py`, type:
   ```
   User: @/path/to/any/photo.jpg describe what you see
   ```
3. Expected: LLM describes the image contents rather than returning a text error.
4. Verify text-only still works: type a plain query with no `@` prefix.

---

## Deferred

- Option 2 (register_objects auto-drain) from the previous plan remains valid and unimplemented.
