# Jetson Orin Deployment Plan: nutri_rag (GAT + Hybrid RAG)

## Context

The current server-side `nutri_rag` pipeline runs three components:
1. **Text embedding** (Qwen3-Embedding-0.6B) — in-process PyTorch + HuggingFace
2. **LLM** (Qwen3.5-9B) — separate llama.cpp/vLLM-style server on `http://localhost:8080`
3. **Retrieval logic** — numpy cosine search over precomputed `.npy` embeddings + DuckDB lookups

Goal: port this to a Jetson Orin that **already has llama.cpp + Qwen3.5 chat model running**. The GAT model itself doesn't need to be retrained or even loaded — we only need the precomputed embedding files.

---

## Two deployment options

### Option A — Minimal port (keep PyTorch on Jetson)

Run the exact same `TextEmbedder` class on Jetson. Simplest port; just `pip install` and run.

- **Pros**: zero code changes to `search.py` / `embedding.py`
- **Cons**: ~5 GB of PyTorch + transformers + dependencies; slower cold start; Jetson Orin must have working PyTorch ARM64 wheels

### Option B — Lean port (recommended)

Replace the `TextEmbedder` class with a thin HTTP wrapper that calls llama.cpp's `/embedding` endpoint. No PyTorch on Jetson at all.

- **Pros**: ~50 MB total Python deps; faster startup; consistent runtime (llama.cpp for both LLM and embedding)
- **Cons**: requires converting/downloading Qwen3-Embedding-0.6B as GGUF; small code change in `embedding.py`

**Recommendation: Option B** — much leaner, plays to the existing llama.cpp setup, and the `TextEmbedder` class interface stays the same so downstream code (`search.py`) works unchanged.

---

## Files to transfer to Jetson

| File | Source path | Size | Purpose |
|---|---|---|---|
| `food_embeddings.npy` | `nutri_graph/outputs/embeddings/food_embeddings.npy` | ~2.5 MB | GAT vectors (64-dim, ~10k foods) |
| `node_embeddings.pt` | `nutri_graph/outputs/embeddings/node_embeddings.pt` | ~24 MB | Full node embeddings (optional, for GAT expansion) |
| `food_text_embeddings.npy` | `nutri_rag/data/embeddings/food_text_embeddings.npy` | ~40 MB | Text vectors (1024-dim, ~10k foods) |
| `food_fdc_ids.npy` | `nutri_rag/data/embeddings/food_fdc_ids.npy` | small | Food ID ordering for text embeddings |
| `nutri_kb.duckdb` | `nutri_graph/data/nutri_kb.duckdb` | varies | Food metadata (names, categories, nutrient amounts) |
| `nutri_rag/` source | `nutri_rag/nutri_rag/` | small | Python package (search, embedding, llm, config) |
| `qwen3-embedding-0.6b.gguf` | download separately (Option B only) | ~600 MB | GGUF-converted embedding model for llama.cpp |

Total embedding/data payload: **~70 MB** (excluding GGUF model).

---

## Requirements files

### `requirements.jetson.txt` (Option B — lean, recommended)

```
numpy>=1.24
duckdb>=0.9
pandas>=2.0
requests>=2.31
```

### `requirements.jetson.txt` (Option A — full)

```
numpy>=1.24
duckdb>=0.9
pandas>=2.0
requests>=2.31
torch>=2.1            # Jetson-specific ARM64 wheel
transformers>=4.40
sentencepiece>=0.1.99
```

For Option A, install PyTorch from NVIDIA's Jetson wheel index (not pip's default):
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
# OR use NVIDIA's official Jetson container with PyTorch preinstalled
```

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
# From your laptop / workstation
mkdir -p stage/nutri-jetson/embeddings stage/nutri-jetson/data
cp nutri_graph/outputs/embeddings/food_embeddings.npy   stage/nutri-jetson/embeddings/
cp nutri_graph/outputs/embeddings/node_embeddings.pt    stage/nutri-jetson/embeddings/
cp nutri_rag/data/embeddings/food_text_embeddings.npy   stage/nutri-jetson/embeddings/
cp nutri_rag/data/embeddings/food_fdc_ids.npy           stage/nutri-jetson/embeddings/
cp nutri_graph/data/nutri_kb.duckdb                     stage/nutri-jetson/data/
cp -r nutri_rag/nutri_rag                               stage/nutri-jetson/

# Ship to Jetson
rsync -avz stage/nutri-jetson/ jetson:~/nutri/
```

### Step 3 — Set up embedding model (Option B only)

```bash
# On Jetson — download Qwen3-Embedding-0.6B in GGUF format
# Either grab a pre-converted GGUF from HuggingFace mirrors, OR:
cd ~/llama.cpp
python convert_hf_to_gguf.py /path/to/Qwen3-Embedding-0.6B \
    --outfile ~/models/qwen3-embedding-0.6b.gguf

# Quantize for smaller footprint (optional)
./llama-quantize ~/models/qwen3-embedding-0.6b.gguf \
    ~/models/qwen3-embedding-0.6b-q8.gguf q8_0
```

### Step 4 — Start llama.cpp servers

```bash
# Embedding server on port 8081
~/llama.cpp/llama-server \
    -m ~/models/qwen3-embedding-0.6b.gguf \
    --embedding \
    --port 8081 \
    --pooling last \
    --host 0.0.0.0 &

# Chat server (already running, just confirming) on port 8080
~/llama.cpp/llama-server \
    -m ~/models/qwen3.5-9b-chat.gguf \
    --port 8080 \
    --host 0.0.0.0 &
```

Run both as `systemd` services for robustness — sample unit at the bottom of this doc.

### Step 5 — Patch `nutri_rag/config.py` for Jetson

Update paths and add embedding endpoint:

```python
# Paths now relative to ~/nutri/
DB_PATH              = "/home/jetson/nutri/data/nutri_kb.duckdb"
FOOD_EMBEDDINGS_PATH = "/home/jetson/nutri/embeddings/food_embeddings.npy"
TEXT_EMBEDDINGS_PATH = "/home/jetson/nutri/embeddings/food_text_embeddings.npy"
TEXT_FDC_IDS_PATH    = "/home/jetson/nutri/embeddings/food_fdc_ids.npy"

LLM_BASE_URL       = "http://localhost:8080/v1/chat/completions"
EMBEDDING_BASE_URL = "http://localhost:8081/embedding"   # NEW for Option B
```

### Step 6 — Replace `TextEmbedder` (Option B only)

In `nutri_rag/embedding.py`, swap the PyTorch-based class for an HTTP wrapper that keeps the same interface (`encode_queries`, `encode_documents` returning normalized numpy arrays):

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

    def encode_queries(self, queries, instruction=None):
        if instruction:
            queries = [f"Instruct: {instruction}\nQuery: {q}" for q in queries]
        return self._embed(queries)

    def encode_documents(self, docs):
        return self._embed(docs)
```

`search.py`, `llm.py`, and everything downstream stays the same.

### Step 7 — Smoke test

```bash
cd ~/nutri
source ~/nutri-venv/bin/activate
python -c "
from nutri_rag.search import search_food
import duckdb
df = search_food(None, 'whole milk', k=5)
print(df)
"
```

Expected: top 5 milk-related foods returned. If you get a connection refused, check both llama.cpp servers are up (`curl localhost:8080/health`, `curl localhost:8081/health`).

### Step 8 — End-to-end inference script

A minimal `~/nutri/inference.py` driver:

```python
import numpy as np
from nutri_rag.search import search_food
from nutri_rag.llm import chat_completion

def answer(query: str):
    matches = search_food(None, query, k=5)
    context = "\n".join(
        f"- {row.description} (fdc_id {row.fdc_id})"
        for row in matches.itertuples()
    )
    messages = [
        {"role": "system", "content": "You are a nutrition assistant."},
        {"role": "user",   "content": f"Foods relevant to the query:\n{context}\n\nQ: {query}"},
    ]
    return chat_completion(messages)

if __name__ == "__main__":
    print(answer("what can I substitute for whole milk?"))
```

---

## Optional: systemd units for robustness

`/etc/systemd/system/llama-embed.service`:
```ini
[Unit]
Description=llama.cpp embedding server (Qwen3-Embedding-0.6B)
After=network-online.target

[Service]
ExecStart=/home/jetson/llama.cpp/llama-server -m /home/jetson/models/qwen3-embedding-0.6b.gguf --embedding --port 8081 --pooling last --host 0.0.0.0
Restart=on-failure
User=jetson

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/llama-chat.service` — same pattern, port 8080, chat model.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-embed llama-chat
```

---

## Verification checklist

After deployment, confirm each layer works in isolation:

```bash
# 1. Chat server alive
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"model":"qwen3.5"}'

# 2. Embedding server alive
curl -s http://localhost:8081/embedding \
    -H "Content-Type: application/json" \
    -d '{"content":"whole milk"}' | python -c "import sys,json;e=json.load(sys.stdin)['embedding'];print(len(e))"
# expect: 1024

# 3. Embedding files load
python -c "
import numpy as np
g = np.load('/home/jetson/nutri/embeddings/food_embeddings.npy')
t = np.load('/home/jetson/nutri/embeddings/food_text_embeddings.npy')
print('GAT', g.shape, 'Text', t.shape)
"
# expect: GAT (9991, 64) Text (9991, 1024)

# 4. DuckDB lookups work
python -c "
import duckdb
con = duckdb.connect('/home/jetson/nutri/data/nutri_kb.duckdb', read_only=True)
print(con.execute('SELECT COUNT(*) FROM nodes_food').fetchone())
"

# 5. End-to-end retrieval
python ~/nutri/inference.py
```

---

## Performance expectations on Jetson Orin

| Stage | Latency (Orin AGX 64GB, Q4_K_M quantized models) |
|---|---|
| Text embed query (Qwen3-Embedding-0.6B) | ~50–150 ms |
| Cosine search over 10k vectors (1024-dim) | <10 ms (numpy) |
| Cosine search over 10k vectors (64-dim, GAT) | <2 ms |
| LLM generation (Qwen3.5-9B, ~200 tokens) | ~3–8 s |
| **Total per query** | **~4–10 s** |

Smaller Jetson Nano variants (4GB) won't fit Qwen3.5-9B — would need to swap to Qwen2.5-3B or smaller for the chat model.

---

## File summary — what to create / modify

| File | Action |
|---|---|
| `requirements.jetson.txt` | **New** (root of repo or `nutri_rag/`) |
| `nutri_rag/nutri_rag/embedding.py` | **Modify** for Option B — replace `TextEmbedder` body, keep class interface |
| `nutri_rag/nutri_rag/config.py` | **Modify** — add `EMBEDDING_BASE_URL`; update paths if needed |
| `inference.py` | **New** (Jetson side, ~30 lines) |
| `/etc/systemd/system/llama-embed.service` | **New** (Jetson side, optional) |
| `/etc/systemd/system/llama-chat.service` | **New** (Jetson side, optional) |

No GAT training code, no PyTorch Geometric, no DuckDB build script needed — those all stay on the dev machine.
