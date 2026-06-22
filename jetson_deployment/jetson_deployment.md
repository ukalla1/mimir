# Jetson Orin Deployment: Robot Assistant (nutri-atlas)

## Context

Run the **full embodied robot assistant** (`nutri-atlas/robot_control/robot_assistant.py`) on a Jetson Orin acting as the operator PC for a Unitree Go2. The Jetson hosts:

- The Qwen Agent loop + tool dispatch (`robot_assistant.py`)
- The `nutri_rag` four-stage RAG pipeline (eaten-side ID → gap analysis → food rec → meal composition)
- `llama.cpp` chat server (Qwen3.5-9B) + embedding server (Qwen3-Embedding-0.6B) [+ optional vision adapter]
- ZMQ client to the robot (navigation, detection, etc.)

The robot itself runs the bridge node (`zmq_bridge_node_working_v2.py`) + RealSense ZMQ streams. **The Jetson and robot must be on the same Wi-Fi subnet** (typically `192.168.0.x`).

The GAT model is **not** loaded — only precomputed embedding `.npy` files are shipped.

---

## Two embedding-runtime options

### Option A — PyTorch-based `TextEmbedder` on Jetson

Run the same `TextEmbedder` class as the dev box. Simplest port; just `pip install` and go.

- **Pros**: zero code changes to `embedding.py` / `search.py`
- **Cons**: ~5 GB of PyTorch + transformers + ARM64 wheels; slower cold start

### Option B — Lean port via llama.cpp `/embedding` (recommended)

Replace the `TextEmbedder` with a thin HTTP wrapper that calls llama.cpp's embedding endpoint. No PyTorch on Jetson.

- **Pros**: ~50 MB Python deps; one inference runtime (llama.cpp) for chat + embedding [+ vision]; faster startup
- **Cons**: requires Qwen3-Embedding-0.6B as GGUF; small `embedding.py` patch

**Recommendation: Option B.** All instructions below assume B unless otherwise marked.

---

## Files to transfer to Jetson

| File | Source path on dev | Size | Required for |
|---|---|---|---|
| **Embeddings & KB** | | | |
| `food_embeddings.npy` | `nutri_graph/outputs/embeddings/food_embeddings.npy` | ~2.5 MB | All stages (food GAT) |
| `node_embeddings.npy` | `nutri_graph/outputs/embeddings/node_embeddings.npy` | ~24 MB | Phase C (nutrient-node GAT for target encoding) |
| `nutrient_emb_index.json` | `nutri_graph/data/nutrient_emb_index.json` | small | Phase C (nutrient_id → row mapping) |
| `food_text_embeddings.npy` | `nutri_rag/data/embeddings/food_text_embeddings.npy` | ~40 MB | All stages (food text) |
| `food_fdc_ids.npy` | `nutri_rag/data/embeddings/food_fdc_ids.npy` | small | Alignment for text embeddings |
| `recipe_text_embeddings.npy` | `nutri_rag/data/embeddings/recipe_text_embeddings.npy` | ~330 MB | Phase D meal composition |
| `nutri_kb.duckdb` | `nutri_graph/data/nutri_kb.duckdb` | varies | Foods / nutrients / recipes / edges |
| `meal_categories.json` | `nutri_rag/data/meal_categories.json` | small | Phase C meal-category filter |
| **Python source** | | | |
| `nutri_rag/` | `nutri_rag/nutri_rag/` | small | RAG pipeline |
| `nutri-atlas/robot_control/` | `nutri-atlas/robot_control/` | small | Agent + tools |
| `landmarks.yaml` | `nutri-atlas/robot_control/config/landmarks.yaml` | small | Named map locations |
| **GGUF models** | | | |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | download | ~6 GB | Chat LLM |
| `mmproj-BF16.gguf` | download | ~922 MB | Vision adapter (only if using `--detector vlm`) |
| `qwen3-embedding-0.6b.gguf` | convert from HF, see Step 3 | ~600 MB | Option B embedding runtime |

Total non-GGUF payload: **~400 MB**. GGUFs add ~7.5 GB.

---

## Requirements

### `requirements.jetson.txt` (Option B — recommended)

```
numpy>=1.24
duckdb>=0.9
pandas>=2.0
requests>=2.31
pyzmq>=25.0
pyyaml>=6.0
json5>=0.9
qwen-agent>=0.0.10
```

If you want the YOLO detector path (`--detector yolo`), also:
```
ultralytics>=8.1
opencv-python>=4.9
```

If using Option A (PyTorch on Jetson), add to the base list:
```
torch>=2.1            # install from NVIDIA's Jetson wheel index, not pip default
transformers>=4.40
sentencepiece>=0.1.99
```

For Option A's PyTorch, follow NVIDIA's Jetson PyTorch wheel install instructions for your JetPack version (don't use the default pip URL).

---

## Deployment process

### Step 1 — Set up Python env on Jetson

```bash
ssh jetson
python3 -m venv ~/nutri-venv
source ~/nutri-venv/bin/activate
pip install -r requirements.jetson.txt
```

### Step 2 — Transfer files from dev machine

```bash
# On dev machine
mkdir -p stage/nutri-jetson/{embeddings,data,nutri_rag,nutri-atlas}

# Embeddings + KB
cp nutri_graph/outputs/embeddings/food_embeddings.npy            stage/nutri-jetson/embeddings/
cp nutri_graph/outputs/embeddings/node_embeddings.npy            stage/nutri-jetson/embeddings/
cp nutri_graph/data/nutrient_emb_index.json                      stage/nutri-jetson/embeddings/
cp nutri_rag/data/embeddings/food_text_embeddings.npy            stage/nutri-jetson/embeddings/
cp nutri_rag/data/embeddings/food_fdc_ids.npy                    stage/nutri-jetson/embeddings/
cp nutri_rag/data/embeddings/recipe_text_embeddings.npy          stage/nutri-jetson/embeddings/
cp nutri_graph/data/nutri_kb.duckdb                              stage/nutri-jetson/data/
cp nutri_rag/data/meal_categories.json                           stage/nutri-jetson/data/

# Python source
cp -r nutri_rag/nutri_rag                                        stage/nutri-jetson/
cp -r nutri-atlas/robot_control                                  stage/nutri-jetson/nutri-atlas/

# Ship
rsync -avz stage/nutri-jetson/ jetson:~/nutri/
```

After this the Jetson layout is:
```
~/nutri/
├── embeddings/
│   ├── food_embeddings.npy
│   ├── node_embeddings.npy
│   ├── nutrient_emb_index.json
│   ├── food_text_embeddings.npy
│   ├── food_fdc_ids.npy
│   └── recipe_text_embeddings.npy
├── data/
│   ├── nutri_kb.duckdb
│   └── meal_categories.json
├── nutri_rag/                  # python package
└── nutri-atlas/
    └── robot_control/
        ├── robot_assistant.py
        ├── tools/
        └── config/landmarks.yaml
```

### Step 3 — Set up the embedding GGUF (Option B only)

```bash
# On Jetson — convert Qwen3-Embedding-0.6B to GGUF
cd ~/llama.cpp
python convert_hf_to_gguf.py /path/to/Qwen3-Embedding-0.6B \
    --outfile ~/models/qwen3-embedding-0.6b.gguf

# Optional quantization
./llama-quantize ~/models/qwen3-embedding-0.6b.gguf \
    ~/models/qwen3-embedding-0.6b-q8.gguf q8_0
```

### Step 4 — Download chat + vision adapter GGUFs

```bash
# On Jetson
mkdir -p ~/models
huggingface-cli download unsloth/Qwen3.5-9B-GGUF \
    --include "Qwen3.5-9B-UD-Q4_K_XL.gguf" "mmproj-BF16.gguf" \
    --local-dir ~/models/Qwen3.5-9B-GGUF
```

`mmproj-BF16.gguf` is only needed if you'll use `--detector vlm`. With `--detector yolo`, you can skip it.

### Step 5 — Start llama.cpp servers

```bash
# Chat server on :8080 (loads chat model + vision adapter)
~/llama.cpp/llama-server \
    -m ~/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    --mmproj ~/models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
    --port 8080 --host 0.0.0.0 \
    --n-gpu-layers 999 -c 32768 &

# Embedding server on :8081
~/llama.cpp/llama-server \
    -m ~/models/qwen3-embedding-0.6b.gguf \
    --embedding --pooling last \
    --port 8081 --host 0.0.0.0 \
    --n-gpu-layers 999 &
```

Run as `systemd` services in production — sample units at the bottom.

If you don't need VLM detection, drop the `--mmproj ...` line.

### Step 6 — Patch `nutri_rag/config.py` for Jetson paths

Update absolute paths to match the Jetson layout:

```python
# nutri_rag/config.py — Jetson values
DB_PATH                  = "/home/jetson/nutri/data/nutri_kb.duckdb"
FOOD_EMBEDDINGS_PATH     = "/home/jetson/nutri/embeddings/food_embeddings.npy"
TEXT_EMBEDDINGS_PATH     = "/home/jetson/nutri/embeddings/food_text_embeddings.npy"
TEXT_FDC_IDS_PATH        = "/home/jetson/nutri/embeddings/food_fdc_ids.npy"
RECIPE_TEXT_EMB_PATH     = "/home/jetson/nutri/embeddings/recipe_text_embeddings.npy"
NODE_EMBEDDINGS_PATH     = "/home/jetson/nutri/embeddings/node_embeddings.npy"
NUTRIENT_EMB_INDEX_PATH  = "/home/jetson/nutri/embeddings/nutrient_emb_index.json"
MEAL_CATEGORIES_PATH     = "/home/jetson/nutri/data/meal_categories.json"

LLM_BASE_URL             = "http://localhost:8080/v1"
EMBEDDING_BASE_URL       = "http://localhost:8081/embedding"   # Option B only
```

### Step 7 — Replace `TextEmbedder` (Option B only)

In `nutri_rag/embedding.py`, swap the PyTorch class for an HTTP wrapper that keeps the same interface so `search.py` and the pipeline don't need changes:

```python
import numpy as np
import requests
from nutri_rag.config import EMBEDDING_BASE_URL

class TextEmbedder:
    def __init__(self, model_name=None, device=None, base_url=EMBEDDING_BASE_URL):
        self.base_url = base_url

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            r = requests.post(self.base_url, json={"content": t}, timeout=30)
            r.raise_for_status()
            v = np.array(r.json()["embedding"], dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-10
            vecs.append(v)
        return np.stack(vecs)

    def encode(self, texts, task_instruction=None, batch_size=None):
        if task_instruction:
            texts = [f"Instruct: {task_instruction}\nQuery: {t}" for t in texts]
        return self._embed(texts)

    def encode_queries(self, queries, instruction=None):
        return self.encode(queries, task_instruction=instruction)

    def encode_documents(self, docs):
        return self._embed(docs)
```

### Step 8 — Verify robot connectivity from Jetson

The robot's Wi-Fi IP changes with DHCP. Get it from the robot:

```bash
# On the robot
hostname -I
# Expect three: 192.168.123.x (Unitree internal — don't use)
#               192.168.0.x   (Wi-Fi — use this)
#               172.17.x.x    (Docker — ignore)
```

Then on the Jetson:

```bash
ping -c 3 <robot_wifi_ip>           # must succeed
nc -zv  <robot_wifi_ip> 5555        # bridge port — must be open
nc -zv  <robot_wifi_ip> 5557        # color stream (only if using VLM)
nc -zv  <robot_wifi_ip> 5558        # depth stream (only if using VLM)
```

If `ping` fails or 5555 is closed, the robot's bridge isn't running or the IPs aren't on the same subnet. Fix that first — `robot_assistant.py` will hang for 10s on every nav call otherwise.

### Step 9 — Launch the robot assistant

The new two-phase meal recommendation chain depends on `AVAILABILITY_SOURCE` being set, or Phase 2 (kitchen check) will be decorative. Recommended launch:

```bash
ssh jetson
source ~/nutri-venv/bin/activate
cd ~/nutri/nutri-atlas/robot_control

# Required: tell the recommender to read what the robot has scanned
export AVAILABILITY_SOURCE=zmq

# Optional pipeline toggles (defaults shown — flip if you want the additions)
export EATEN_RETRIEVAL_MODE=hybrid     # Phase A — default
export RECOMMEND_MODE=v1                # Phase C v2 = target-as-query (opt-in)
export MEAL_COMPOSE_MODE=on             # Phase D — adds recipes to recommendation
export RECIPE_SCORE_MODE=hybrid         # Phase D Gap 1 — default

python robot_assistant.py \
    --robot-ip <robot_wifi_ip> --robot-port 5555 \
    --detection-mode real --detector vlm
```

For YOLO detection instead of VLM (faster, COCO 80 classes only):

```bash
python robot_assistant.py \
    --robot-ip <robot_wifi_ip> --robot-port 5555 \
    --detection-mode real --detector yolo

# In a second Jetson terminal — run the YOLO push loop
cd ~/nutri/nutri-atlas/robot_control/tools
python detector_node_real_world.py --robot-ip <robot_wifi_ip>
```

---

## Detector mode reference

| | `--detector yolo` | `--detector vlm` |
|---|---|---|
| Vocabulary | COCO 80 classes | Open — natural language |
| Operator processes | `robot_assistant.py` + `detector_node_real_world.py` | `robot_assistant.py` alone |
| Needs `mmproj-BF16.gguf` loaded into chat server | No | **Yes** |
| Needs `ultralytics` + `opencv-python` | **Yes** | No |
| Per-scan latency | ~50 ms | ~3–8 s (one chat-server call) |
| Best use | Known item types, fast scanning | Semantic queries, unusual targets |

YOLO is the right default unless you need open-vocab detection. VLM is significantly slower and burns chat-server capacity that the agent loop is also using.

---

## Environment variable reference

| Variable | Default | Effect |
|---|---|---|
| `AVAILABILITY_SOURCE` | `none` | `zmq`/`json` enables Phase 2 grounding from robot's detected objects |
| `OBJECT_SERVER_IP` | — | When `AVAILABILITY_SOURCE=zmq`; bridge IP (usually same as `--robot-ip`) |
| `EATEN_RETRIEVAL_MODE` | `hybrid` | `text_top1` reverts Phase A's score-fusion eaten-side ID |
| `FOOD_NEIGHBOR_MODE` | `gat_only` | `hybrid` enables Phase B score-fusion food-neighbor expansion |
| `RECOMMEND_MODE` | `v1` | `v2` enables Phase C target-as-query recommender |
| `RECIPE_SCORE_MODE` | `hybrid` | `pool_centroid` reverts Gap 1 pseudo-anchor recipe scoring |
| `MEAL_COMPOSE_MODE` | `off` | `on` activates Phase D recipe-layer composition |
| `MEAL_MIN_OVERLAP`, `MEAL_MACRO_TOLERANCE`, `MEAL_ALPHA`, `MEAL_GAMMA`, `MEAL_BETA` | see code | Phase E meal-filter tuning |

Recommended baseline for "everything new turned on":
```bash
export AVAILABILITY_SOURCE=zmq
export OBJECT_SERVER_IP=<robot_wifi_ip>
export FOOD_NEIGHBOR_MODE=hybrid
export RECOMMEND_MODE=v2
export MEAL_COMPOSE_MODE=on
```

---

## Verification checklist

Run these in order. Stop at the first failure.

```bash
# 1. Embedding GGUF reachable
curl -s http://localhost:8081/embedding \
    -H "Content-Type: application/json" \
    -d '{"content":"apple"}' | python -c "import sys,json;e=json.load(sys.stdin)['embedding'];print('dim',len(e))"
# expect: dim 1024

# 2. Chat server reachable
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"model":"qwen3.5"}' \
    | head -c 200

# 3. Embeddings + DB load cleanly
python -c "
import numpy as np, duckdb
f = np.load('/home/jetson/nutri/embeddings/food_text_embeddings.npy')
g = np.load('/home/jetson/nutri/embeddings/food_embeddings.npy')
r = np.load('/home/jetson/nutri/embeddings/recipe_text_embeddings.npy', mmap_mode='r')
con = duckdb.connect('/home/jetson/nutri/data/nutri_kb.duckdb', read_only=True)
nrec = con.execute('SELECT COUNT(*) FROM nodes_recipe').fetchone()[0]
print(f'food_text {f.shape}, food_gat {g.shape}, recipe_text {r.shape}, n_recipes {nrec}')
"
# expect: food_text (9991,1024), food_gat (9991,64), recipe_text (82238,1024), n_recipes 82238

# 4. Robot reachable
ping -c 2 <robot_wifi_ip>
nc -zv  <robot_wifi_ip> 5555

# 5. End-to-end (after starting robot_assistant.py)
# Type at the assistant prompt:
#   "I ate an apple, I'm allergic to peanuts, recommend lunch"
# Look for [nutrition] line in stdout. Expected:
#   [nutrition] eaten="an apple", meal_type=breakfast, next=lunch,
#               disliked=['peanuts'], condition=None
#   [nutrition] availability filter: N fdc_ids       ← only if AVAILABILITY_SOURCE=zmq and kitchen scanned
```

End-to-end flow to confirm the two-phase chain works:

```
User: I ate an apple for breakfast, I'm allergic to peanuts. Recommend lunch.
  → get_meal_recommendation with disliked_ingredients=["peanuts"]
  → present ideal recommendation
  → ask: "Want me to check your kitchen?"

User: yes please
  → navigate_to_landmark(landmark_name="kitchen")
  → register_objects(targets="<ingredients>")
  → get_meal_recommendation again (now sees kitchen items)
  → present refined recommendation
```

---

## systemd units (optional)

`/etc/systemd/system/llama-chat.service`:
```ini
[Unit]
Description=llama.cpp chat server (Qwen3.5-9B + mmproj)
After=network-online.target

[Service]
ExecStart=/home/jetson/llama.cpp/llama-server \
    -m /home/jetson/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    --mmproj /home/jetson/models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
    --port 8080 --host 0.0.0.0 --n-gpu-layers 999 -c 32768
Restart=on-failure
User=jetson

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/llama-embed.service` — same pattern, port 8081, no mmproj, `--embedding --pooling last`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-chat llama-embed
```

The assistant itself is interactive (stdin loop), so don't `systemd` that — launch from a terminal.

---

## Performance expectations (Orin AGX 64GB, Q4_K_XL chat + Q8 embedding)

| Stage | Latency |
|---|---|
| Text embed query (Qwen3-Embedding-0.6B) | ~50–150 ms |
| Cosine search over 10k vectors (1024-dim) | <10 ms |
| Cosine search over 10k vectors (64-dim, GAT) | <2 ms |
| Cosine search over 82k recipes (1024-dim, mmap) | ~80 ms |
| Chat-LLM turn (~200 tokens) | ~3–8 s |
| VLM scan (one frame, 640×480, Qwen3.5-9B + mmproj) | ~3–8 s |
| YOLO detect (one frame) | ~50 ms |
| Meal-rec end-to-end (parse → gap → 4-stage RAG → narration) | ~10–20 s |
| Nav to kitchen + VLM register + re-rec | ~30–60 s (mostly travel + VLM) |

On smaller Orin Nano (8GB), drop chat model to IQ2_M (~3 GB) or use Qwen2.5-3B and expect ~2× the per-token latency.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `[startup] clear_objects: No reply from robot within 10s` | Robot bridge not running, wrong IP, or wrong subnet. Run Step 8's checks. |
| `[nutrition] availability filter: N fdc_ids` never appears | `AVAILABILITY_SOURCE` not exported, or kitchen never scanned with `register_objects`. |
| LLM ignores `disliked_ingredients` | User message didn't state a restriction the LLM could extract — try explicit "I'm allergic to X". |
| Phase 2 doesn't trigger | LLM didn't offer kitchen check, OR you said "no". The offer is in the system prompt; check the prompt is up to date (`grep "Phase 1 — Ideal" nutri-atlas/robot_control/robot_assistant.py`). |
| VLM scan returns `lamp, lamp, lamp, ...` | Known open-vocab VLM noise when `targets` is empty/broad. Always pass concrete `targets=` to `register_objects` / `scan_objects`. |
| `nc -zv :5557` fails but `:5555` works | RealSense ZMQ bridge not running on robot — start `bringup_with_zmq.launch.py`. Only needed for VLM. |
| Robot keeps moving after "navigation failed" | Older bridge without Fix A. Pull the latest `zmq_bridge_node_working_v2.py` to the robot. |
