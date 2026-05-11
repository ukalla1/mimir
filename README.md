# Mimir: Graph-Augmented RAG for Nutritional Intelligence

Mimir combines **Graph Attention Networks (GAT)** with **Retrieval-Augmented Generation (RAG)** to estimate nutritional content from natural-language meal descriptions and provide personalized meal recommendations. The system learns food-nutrient-recipe relationships from USDA databases and FoodKG, and uses them alongside semantic text embeddings and LLM reasoning.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                nutri_graph                    │
                    │    Knowledge Base + GAT Embedding Training    │
                    │                                              │
                    │  USDA FDC + SR Legacy + FoodKG               │
                    │    → Junk filtering (remove lab samples)     │
                    │    → Deduplication (avg nutrients)            │
                    │    → Recipe-ingredient-tag integration        │
                    │    → DuckDB KB                               │
                    │    → Heterogeneous GATv2 Training             │
                    │      (food, nutrient, recipe, tag)            │
                    │    → Food Embeddings (64d)                   │
                    └──────────────┬───────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                           ▼
┌────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  NutriBench        │ │  General Assistant   │ │  PFoodReq Benchmark  │
│  (Nutrient Est.)   │ │  (Meal Recommend.)   │ │  (Recipe Recommend.) │
│                    │ │                      │ │                      │
│ Meal → Parse       │ │ Eaten Foods → Parse  │ │ Tag → All Recipes    │
│ → Embedding Search │ │ → Gap Analysis (LLM) │ │ → Constraint Filter  │
│ → GAT Re-rank      │ │ → DB Query + GAT     │ │ → GAT Re-rank        │
│ → CoT Prompt → LLM│ │ → Preference Re-rank │ │ → Return matches     │
│                    │ │ → Recommend (LLM)    │ │ (no LLM needed)      │
└────────────────────┘ └──────────┬───────────┘ └──────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      nutri-atlas          │
                    │  (Operator PC)            │
                    │                           │
                    │  User ─► Qwen Agent       │
                    │  ─► Tool calls            │
                    │    ├─ Meal recommendation │
                    │    │   (via nutri_rag)    │
                    │    └─ Navigation / Vision │
                    │        (via ZMQ client)   │
                    └────────────┬─────────────┘
                                 │ ZMQ (TCP)
                                 ▼
                    ┌──────────────────────────┐
                    │      robot_side           │
                    │  (Robot Onboard PC)       │
                    │                           │
                    │  zmq_bridge_node :5555    │
                    │    → ROS2 nav / cmd_vel   │
                    │    → TF-based detection   │
                    │  zmq_object_server :5556  │
                    │    → simulation only      │
                    └──────────────────────────┘
```

## Subsystems

### [nutri_graph](nutri_graph/) — Knowledge Base + GAT Training

Builds a cleaned DuckDB KB from three data sources:
- **USDA FoodData Central** (Foundation Foods): ~74K raw entries, cleaned to ~4.6K after removing lab analysis junk
- **USDA SR Legacy**: 7,793 curated entries with cooked/prepared forms
- **FoodKG** (via PFoodReq): ~82K recipes with ingredients, nutrients, and cuisine tags — ingredients matched to USDA fdc_ids via Qwen3-Embedding

Trains a heterogeneous GATv2 on a 4-node-type graph (food, nutrient, recipe, tag) to produce 64-dim embeddings encoding nutritional and relational similarity.

### [nutri_rag](nutri_rag/) — RAG Pipeline

Four retrieval versions for ablation:

| Version | Method | Description |
|---------|--------|-------------|
| V0 | BM25 | Keyword matching with cross-language synonyms |
| V1 | Dense | Qwen3-Embedding semantic search |
| V2 | Dense + GAT | V1 + GAT re-ranking when text is ambiguous |
| V3 | Multi-candidate | Top-5 candidates per food, filtered by similarity threshold (0.60) |

Three modes:
- **NutriBench Benchmark**: Evaluate carb/protein/fat/energy estimation accuracy using lm-evaluation-harness
- **General Assistant**: Interactive meal recommendations with GAT neighbor expansion, gap analysis, and user preference learning
- **PFoodReq Benchmark**: Personalized food recommendation using deterministic constraint filtering + GAT re-ranking

### [nutri-atlas](nutri-atlas/) — Robot Integration (Operator Side)

Connects the nutrition assistant to a Clearpath Go2 robot via Qwen Agent tool-calling. The robot assistant can navigate, detect objects, and provide meal recommendations — all through natural language.

- **Navigation tools**: go to landmarks or detected objects (by name or coordinates), spin
- **Detection tools**: `get_detected_objects` / `get_current_detected_objects` — query the object store
- **Real-world detection**: two switchable backends via `--detector yolo|vlm` (default: `yolo`):
  - **YOLO**: `detector_node_real_world.py` runs YOLO on RealSense streams; press Enter to push detections to the robot as TF frames
  - **VLM**: `scan_objects` saves the current RGB-D frame to disk; the agent LLM (Qwen3.5-9B multimodal) describes the scene inline; `register_objects` runs VLM grounding to get bounding boxes → depth backprojection → 3D camera-frame positions → TF landmarks
- **Inline image input**: prefix a query with `@/path/to/image.jpg` to send an image directly to the LLM (no tool call made)
- **Nutrition tool**: `get_meal_recommendation` wraps `nutri_rag`'s full pipeline (parse → gap analysis → GAT expansion → recommendation)
- **LLM/VLM**: Qwen3.5-9B with mmproj vision adapter as the orchestration agent, deciding when to call navigation vs vision vs nutrition tools
- **Communication**: ZMQ REQ client → sends commands to robot_side
- **Session reset**: detected landmarks are cleared automatically on each `robot_assistant.py` startup

Set `DETECTION_MODE=real` on the operator PC to switch from simulation (port 5556 object server) to real-world mode (detections pushed from YOLO via port 5555).

### [robot_side](robot_side/) — ZMQ Bridge (Robot Side)

Runs on the robot's onboard computer:

- **`zmq_bridge_node_working_v2.py`** (port 5555): handles navigation, spin, move, LiDAR, and real-world object detection (`update_objects` / `get_detected_objects`). Transforms camera-frame YOLO detections to map frame via TF and broadcasts them as static TF frames.
- **`zmq_object_server.py`** (port 5556, simulation only): serves a persistent map from `detected_objects.json` written by the Go2's onboard YOLO.

```
Real world:
  detector_node_real_world.py ──ZMQ REQ──→  zmq_bridge_node.py ──→ TF frames under map
  robot_assistant.py ──────────ZMQ REQ──→  zmq_bridge_node.py ──→ get_detected_objects

Simulation:
  robot_assistant.py ──────────ZMQ REQ──→  zmq_object_server.py ──→ detected_objects.json
```

### [PFoodReq](PFoodReq/) — Benchmark Data

Original dataset and BAMnet baseline from WSDM 2021. Contains test/dev/train splits with ~82K recipes from Recipe1M. Used by `nutri_graph` for recipe knowledge graph construction and by `nutri_rag` for benchmark evaluation.

### [foodkg.github.io](foodkg.github.io/) — FoodKG Construction (External)

External research project for building the FoodKG knowledge graph from USDA + Recipe1M data. Mimir uses its `recipe_kg.json` output as input for `nutri_graph/scripts/build_recipe_kb.py`. Not needed for normal operation — only required if rebuilding the recipe knowledge graph from scratch.

## Results

### NutriBench (Protein, 1000 samples)

| Version | Acc@7.5g | MAE |
|---------|----------|-----|
| Baseline (no RAG) | 0.735 | 7.00 |
| V3 (multi-candidate + GAT) | **0.763** | **6.04** |

### PFoodReq (Full test set, 2244 queries)

| Method | MAP | MAR | F1 |
|--------|-----|-----|-----|
| BAMnet (PFoodReq paper, WSDM 2021) | 62.7 | 61.8 | 63.7 |
| **Ours (Config C: filter + GAT, no LLM)** | **78.7** | **82.9** | **77.5** |

Config C pipeline: tag lookup → deterministic constraint filter (ingredient inclusion/exclusion + nutrient ranges) → GAT re-ranking. The deterministic filter leverages our clean augmented KB with accurate ingredient-nutrient mappings, outperforming BAMnet's learned embedding approach without requiring an LLM.

## Implementation Process (All Submodules)

This is the recommended end-to-end order when deploying on a new machine.

### 0) Environment Setup

**Requirements:**
- Python 3.9+ (tested on 3.10)
- Conda environment recommended
- Kaggle API token at `~/.kaggle/kaggle.json` (for data download)
- NVIDIA GPU recommended for GAT training

#### Step 0a: Install PyTorch first

PyTorch Geometric extensions require PyTorch to already be present at build time — install it before anything else.

```bash
# Check your CUDA version with: nvidia-smi (top-right corner shows CUDA version)
# Adjust the index URL to match (cu118, cu121, cu124, cpu, etc.)
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

#### Step 0b: Install PyTorch Geometric extensions

Use **pre-built wheels** matching your exact torch + CUDA version. Installing mismatched wheels (e.g. cu121 wheels with a cu124 torch) will appear to succeed but crash at runtime with `undefined symbol` errors.

```bash
# Replace torch-2.5.0+cu124 with your actual torch+CUDA combo
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

Always verify the install works at runtime before proceeding:
```bash
python -c "import torch_scatter; print('torch_scatter OK')"
python -c "import torch_sparse; print('torch_sparse OK')"
```

If you see `OSError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs`, the wheels don't match your torch version. Uninstall and reinstall with the correct URL from https://data.pyg.org/whl/

> **Note:** `nutri_graph/requirements.txt` has a line `torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html`. Skip that line — use the versioned wheel URL above for all four PyG packages instead.

#### Step 0c: Install all remaining packages

```bash
cd ~/work/atlas/mimir

# Install local packages
pip install -e nutri_graph
pip install -e nutri_rag

# Install remaining nutri_graph dependencies
pip install torch-geometric
pip install duckdb pandas scikit-learn umap-learn plotly kaleido \
            pyvis networkx tqdm matplotlib kaggle kagglehub pyarrow

# Install nutri-atlas dependencies
pip install qwen-agent pyzmq pyyaml json5 requests transformers

# Install lm-evaluation-harness with API support (required for NutriBench benchmarks)
pip install lm-eval[api]
```

Notes:
- Python `>=3.9` is required by `nutri_graph` and `nutri_rag`.
- `nutri_graph/scripts/download_data.py` needs Kaggle credentials.

### 1) Build `nutri_graph` Knowledge Base

```bash
cd ~/work/atlas/mimir/nutri_graph

# Optional: fetch USDA data
python scripts/download_data.py

# Required: build clean USDA + SR Legacy KB
python scripts/build_kb.py
```

Output:
- `nutri_graph/data/nutri_kb.duckdb`

### 2) Build `nutri_rag` Food Text Embeddings

```bash
cd ~/work/atlas/mimir/nutri_rag
python scripts/build_embeddings.py
```

Why now:
- `build_recipe_kb.py` in `nutri_graph` uses these food embeddings for recipe ingredient matching.

### 3) Integrate FoodKG Recipes into `nutri_graph`

```bash
cd ~/work/atlas/mimir/nutri_graph
python scripts/build_recipe_kb.py
```

Adds recipe/tag tables into the same DuckDB KB.

### 4) Train GAT Embeddings in `nutri_graph`

```bash
cd ~/work/atlas/mimir/nutri_graph
python scripts/train_GAT.py
```

Key outputs:
- `nutri_graph/outputs/embeddings/food_embeddings.npy`
- `nutri_graph/outputs/embeddings/node_embeddings.pt`

### 5) (Optional) Build Recipe Text Embeddings for PFoodReq

```bash
cd ~/work/atlas/mimir/nutri_rag
python scripts/build_recipe_embeddings.py
```

### 6) Build llama.cpp and Download Qwen3.5-9B

#### 6a: Build llama.cpp

The LLM server uses [llama.cpp](https://github.com/ggml-org/llama.cpp). `start_server.sh` expects the binary at `~/softwares/llama.cpp/llama-server`.

```bash
mkdir -p ~/softwares
cd ~/softwares
git clone https://github.com/ggml-org/llama.cpp
```

**Option A — With sudo (simpler):**

```bash
sudo apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y

cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-server

cp llama.cpp/build/bin/llama-server llama.cpp/llama-server
```

**Option B — Without sudo (no root access):**

First, find the nvcc path on your system:
```bash
find /usr/local/cuda* -name nvcc 2>/dev/null
```

Then pass it directly to cmake — no PATH changes needed for the build itself:
```bash
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-server

cp llama.cpp/build/bin/llama-server llama.cpp/llama-server
```

> **Optional — add nvcc to PATH permanently:** If you want `nvcc` available globally in your shell (e.g. for other CUDA tools), add these to `~/.bashrc`. This is NOT required for the cmake build above, which uses the full path directly.
> ```bash
> echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
> echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
> source ~/.bashrc
> ```
> These are user-level changes only — no sudo needed, no system-wide impact.

> **nvcc path:** Do NOT use `/usr/local/cuda-12.1/bin/nvcc`. The latest llama.cpp targets `compute_120` (Blackwell GPUs) which requires CUDA 12.8+. Using CUDA 12.1 will fail with `nvcc fatal : Unsupported gpu architecture 'compute_120'`. Use CUDA 13.0 (or 12.8+) instead. If `nvcc` is not found at all, ask your sysadmin to install the CUDA toolkit.

> **Compile time:** Expect 15–30 minutes. The build compiles many CUDA kernel variants. Progress appears as `[ XX%] Building CUDA object ...` lines — this is normal. The GPU is not used during compilation; `nvidia-smi` will show no compute processes until the server actually runs.

Verify the build succeeded:
```bash
~/softwares/llama.cpp/llama-server --version
```

#### 6b: Download Qwen3.5-9B GGUF

Two files are always required — the base model and the mmproj vision adapter:

```bash
mkdir -p /home/boxun/work/atlas/unsloth

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='unsloth/Qwen3.5-9B-GGUF',
    local_dir='/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF',
    allow_patterns=['*UD-Q4_K_XL*', 'mmproj-BF16.gguf']
)
"
```

This downloads `Qwen3.5-9B-UD-Q4_K_XL.gguf` (~6GB, default model) and `mmproj-BF16.gguf` (~922MB, required for vision/VLM mode).

**Optional — download all quantization variants** (for model sweep benchmarking):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='unsloth/Qwen3.5-9B-GGUF',
    local_dir='/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF',
    allow_patterns=['*.gguf']
)
"
```

Available quantizations (~3–6GB each):

| File | Size | Quality |
|------|------|---------|
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | ~6GB | Best (default) |
| `Qwen3.5-9B-UD-Q3_K_XL.gguf` | ~5GB | Good |
| `Qwen3.5-9B-Q4_K_M.gguf` | ~5.7GB | Good |
| `Qwen3.5-9B-Q4_K_S.gguf` | ~5.4GB | Good |
| `Qwen3.5-9B-Q3_K_M.gguf` | ~4.7GB | Medium |
| `Qwen3.5-9B-Q3_K_S.gguf` | ~4.3GB | Medium |
| `Qwen3.5-9B-UD-Q2_K_XL.gguf` | ~4.1GB | Low |
| `Qwen3.5-9B-UD-IQ2_M.gguf` | ~3.7GB | Low |
| `Qwen3.5-9B-UD-IQ2_XXS.gguf` | ~3.2GB | Lowest |

> **Note:** `huggingface-cli` may not be in PATH even if `huggingface_hub` is installed. Use the Python API above as a reliable alternative.

### 7) Start LLM Server (`nutri_rag`)

Required for:
- `nutri_rag/scripts/run_bench.py`
- `nutri_rag/scripts/demo_assistant.py`
- `nutri-atlas/robot_control/robot_assistant.py`

```bash
cd ~/work/atlas/mimir/nutri_rag
bash scripts/start_server.sh
# serves OpenAI-compatible endpoint on http://0.0.0.0:8080/v1
```

`start_server.sh` automatically loads the mmproj vision adapter (`mmproj-BF16.gguf`) alongside the base model, enabling multimodal input for the VLM detector and inline image queries. Both files must be present in the same GGUF directory.

### 8) Run Each Subsystem

```bash
# A) NutriBench benchmark (single model)
cd ~/work/atlas/mimir/nutri_rag
python scripts/run_bench.py --mode v3 --nutrient protein --limit 200

# A2) NutriBench model sweep — benchmark all downloaded quantizations in one run
# Quick test (20 samples/model, ~2 min/model):
python scripts/run_model_sweep.py --limit 20

# Full run, protein only (~3–6 hours total):
python scripts/run_model_sweep.py

# Full run, all nutrients (~12–24 hours total):
python scripts/run_model_sweep.py --nutrients carb protein fat energy

# Specific models only:
python scripts/run_model_sweep.py --models Qwen3.5-9B-UD-Q4_K_XL.gguf Qwen3.5-9B-Q4_K_M.gguf

# Results saved to nutri_rag/results/model_sweep_{timestamp}/ with a comparison table.

# B) PFoodReq benchmark (LLM server not required)
cd ~/work/atlas/mimir/nutri_rag
python scripts/run_pfoodreq_bench.py

# C) Nutrition assistant
cd ~/work/atlas/mimir/nutri_rag
python scripts/demo_assistant.py

# D) Robot assistant (navigation + nutrition)
cd ~/work/atlas/mimir/nutri-atlas/robot_control
python robot_assistant.py
```

## Two-Computer Deployment (Same Wi-Fi)

The system is designed to run across two machines on the same network:

```
┌─────────────────────────────────┐       ┌─────────────────────────────┐
│  Operator PC (GPU required)     │       │  Robot Onboard PC           │
│                                 │       │                             │
│  LLM server (:8080)             │  TCP  │  zmq_bridge_node (:5555)   │
│  robot_assistant.py ────────────┼──────►│  ROS2 runtime               │
│  detector_node_real_world.py ───┼──────►│  RealSense + TF frames      │
│  nutri_rag (embeddings + DB)    │       │                             │
│  nutri_graph (KB + GAT)         │       │  zmq_object_server (:5556)  │
└─────────────────────────────────┘       │  (simulation only)          │
                                          └─────────────────────────────┘
```

### Robot Onboard PC

```bash
source /opt/ros/humble/setup.bash && source ~/test_ws/install/setup.bash

# --- Real world ---

# Terminal 1 — RealSense camera + ZMQ image bridge + static TF (base_link → camera_link)
ros2 launch realsense_zmq bringup_with_zmq.launch.py

# Terminal 2 — ZMQ bridge (navigation + detection)
cd nutri-atlas/robot_control/robot_side/zmq_bridge_real
python zmq_bridge_node_working_v2.py
# Optional: --port 5555 --spin-kp 1.5 --move-kp 0.8

# Detected objects are stored persistently at ~/detected_objects.json on the robot.

# --- Simulation ---

# Terminal 1 — ZMQ navigation bridge
python zmq_bridge_node_working_v2.py

# Terminal 2 — persistent object map server (reads detected_objects.json from Go2 YOLO)
python zmq_object_server.py   # port 5556
```

### Operator PC

Complete steps 0–6 from [Implementation Process](#implementation-process-all-submodules) first, then:

```bash
# Terminal 1 — LLM server (required for all modes)
cd ~/work/atlas/mimir/nutri_rag
bash scripts/start_server.sh

# --- Real world ---

# Terminal 2 — robot assistant (YOLO detector, default)
cd ~/work/atlas/mimir/nutri-atlas/robot_control
python robot_assistant.py --robot-ip 192.168.0.114 --detection-mode real

# Terminal 2 — robot assistant (VLM detector — open-vocab, uses LLM vision)
python robot_assistant.py --robot-ip 192.168.0.114 --detection-mode real --detector vlm

# Terminal 3a — manual detector: press Enter to push current frame to robot
cd ~/work/atlas/mimir/nutri-atlas/robot_control/tools
python detector_node_real_world.py --robot-ip 192.168.0.114

# Terminal 3b — auto detector (alternative to 3a): sends detections automatically
python detector_node_real_world_auto.py --robot-ip 192.168.0.114
# Filter by label and confidence:
python detector_node_real_world_auto.py --robot-ip 192.168.0.114 \
    --targets person chair --stable-conf 0.6 --stable-frames 10

# --- Simulation ---

# Terminal 2 — robot assistant (default mode)
cd ~/work/atlas/mimir/nutri-atlas/robot_control
python robot_assistant.py --robot-ip 127.0.0.1
```

### What changes between sim and real

| | Simulation | Real World |
|---|---|---|
| Object detection source | Go2 onboard YOLO → `/detected_objects` topic | YOLO on operator PC → `update_objects` ZMQ cmd |
| `get_detected_objects` | port 5556 (`zmq_object_server`) | port 5555 bridge (`get_detected_objects`) |
| `list_landmarks` | YAML only | YAML + detected objects |
| Robot processes | bridge + `zmq_object_server` | bridge + `bringup_with_zmq` |
| Operator processes | `robot_assistant.py` | `robot_assistant.py` + detector |

### Network Checklist

- Both machines on the same subnet (e.g. `192.168.0.x`).
- Verify connectivity: `ping <robot_ip>` from operator PC.
- Firewall allows inbound TCP `5555` on robot (real world); also `5556` for simulation.
- LLM server runs on `localhost:8080` on the operator PC only.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph ML | PyTorch + PyTorch Geometric (GATv2Conv) |
| Text Embeddings | Qwen3-Embedding-0.6B (1024d, last-token pooling) |
| LLM / VLM Inference | Qwen3.5-9B + mmproj vision adapter via llama.cpp / llama-server |
| Agent Framework | Qwen Agent (tool-calling orchestration) |
| Object Detection | YOLO (COCO 80-class) or VLM grounding (open-vocabulary) |
| 3D Localisation | VLM bounding box → RealSense depth backprojection → camera-frame 3D |
| Robot Communication | ZMQ → ROS2 (Clearpath Go2) |
| Database | DuckDB (columnar storage + full-text search) |
| Data Sources | USDA FoodData Central + USDA SR Legacy + FoodKG |
| Evaluation | lm-evaluation-harness |
| Visualization | Plotly, UMAP, scikit-learn |

## Project Structure

```
mimir/
├── nutri_graph/                    # Knowledge base + GAT model
│   ├── nutri_graph/
│   │   ├── kb/builder.py          # KB builder (FDC + SR Legacy + cleaning + dedup)
│   │   ├── kb/recipe_builder.py   # FoodKG recipe integration
│   │   ├── graph/                 # PyG graph construction + negative sampling
│   │   ├── models/gat_model.py    # GATv2 (dual decoders, 4 node types)
│   │   ├── training/trainer.py    # Training loop
│   │   └── visualization/         # UMAP, clustering, training curves
│   ├── scripts/
│   │   ├── build_kb.py            # Build DuckDB KB
│   │   ├── build_recipe_kb.py     # Integrate FoodKG recipes
│   │   ├── train_GAT.py           # Train GAT + save embeddings
│   │   └── export_kb.py           # Export KB to JSON
│   ├── data/                      # USDA CSVs + SR Legacy + DuckDB
│   └── outputs/                   # Embeddings, checkpoints, plots
│
├── nutri_rag/                     # RAG pipeline
│   ├── nutri_rag/
│   │   ├── config.py              # Thresholds, model config
│   │   ├── embedding.py           # TextEmbedder + FoodVectorIndex + GATIndex + RecipeVectorIndex
│   │   ├── search.py              # Semantic + GAT hybrid search
│   │   ├── bench/                 # NutriBench: retriever, prompt, task utils
│   │   ├── assistant/             # Assistant: gap analysis, recommendations
│   │   └── pfoodreq/             # PFoodReq: query parser, retriever, evaluator, prompt
│   ├── tasks/                     # lm-eval task definitions (V0/V1/V2/V3)
│   ├── scripts/                   # build_embeddings, run_bench, run_pfoodreq_bench, demos
│   └── results/                   # Benchmark outputs
│
├── nutri-atlas/                   # Robot integration
│   ├── robot_control/
│   │   ├── robot_assistant.py     # Main chat loop + Qwen Agent (--detector yolo|vlm)
│   │   ├── data/                  # Saved RGB-D frame pairs for VLM grounding
│   │   ├── tools/
│   │   │   ├── navigate_tool.py   # Navigate to landmarks / coordinates
│   │   │   ├── detect_tool.py     # navigate_and_scan, scan_objects, register_objects
│   │   │   ├── detector_core.py   # YOLODetector, VLMDetector, Detection, FrameCache
│   │   │   ├── object_tool.py     # Camera object detection queries
│   │   │   ├── motion_tool.py     # Spin and move primitives
│   │   │   ├── nutrition_tool.py  # Meal recommendation (wraps nutri_rag)
│   │   │   ├── detector_node_real_world.py      # Manual YOLO detector
│   │   │   ├── detector_node_real_world_auto.py # Auto YOLO detector (stability gate)
│   │   │   └── zmq_client.py      # ZMQ transport to robot
│   │   ├── robot_side/
│   │   │   ├── zmq_bridge_real/
│   │   │   │   ├── zmq_bridge_node_working_v2.py  # REP server :5555 → ROS2 nav/detection
│   │   │   │   ├── zmq_object_server.py           # REP server :5556 (simulation only)
│   │   │   │   └── realsense_zmq/launch/          # RealSense + ZMQ image bridge + TF
│   │   │   └── coordinates_record.py              # Record landmark coords via TF
│   │   └── config/landmarks.yaml  # Named room positions
│   └── benchmark_tasks.yaml       # 30+ graded robot benchmark tasks
│
├── PFoodReq/                      # PFoodReq benchmark data (WSDM 2021)
│   └── data/kbqa_data/            # test/dev/train splits + dish_info_map
│
├── foodkg.github.io/              # FoodKG construction (external, optional)
│
└── README.md
```
