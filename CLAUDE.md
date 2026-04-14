# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mimir is a graph-augmented RAG system for nutritional intelligence. It combines GAT embeddings with semantic text embeddings and LLM reasoning to estimate nutrients and recommend meals. The system also integrates with a Clearpath Go2 robot for physical-world interaction.

## Repository Layout

The repo (`mimir/`) contains six subsystems with strict dependency ordering:

```
nutri_graph  →  nutri_rag  →  nutri-atlas  →  robot_side
(KB + GAT)     (RAG pipeline)  (agent + tools)   (ZMQ→ROS2)
```

- **nutri_graph/**: DuckDB knowledge base (USDA + FoodKG) + heterogeneous GATv2 training
- **nutri_rag/**: Four RAG retrieval versions (V0–V3), NutriBench/PFoodReq benchmarks, nutrition assistant
- **nutri-atlas/**: Qwen Agent with tool-calling for robot navigation + nutrition advice
- **nutri-atlas/robot_control/robot_side/**: ZMQ REP servers that run on the robot's onboard PC. Two sub-directories:
  - `zmq_bridge_real/` — **production**: `zmq_bridge_node_working_v2.py` (Nav2 action client) + `zmq_object_server.py`
  - `zmq_bridge_simulation/` — simulation variants (reference only)
  - `coordinates_record.py` — TF-based tool for recording landmark (x, y) coordinates interactively
- **PFoodReq/**: External benchmark data (WSDM 2021), gitignored
- **foodkg.github.io/**: External FoodKG construction project, gitignored

## Build & Setup

Python >=3.9. Install the two local packages:

```bash
pip install -e nutri_graph
pip install -e nutri_rag
pip install -r nutri_graph/requirements.txt
pip install qwen-agent pyzmq pyyaml json5
```

**Data pipeline must run in this exact order** (cross-subsystem dependencies):

```bash
# 1. Build USDA + SR Legacy knowledge base
cd nutri_graph && python scripts/build_kb.py

# 2. Build text embeddings (nutri_graph's recipe builder needs these)
cd nutri_rag && python scripts/build_embeddings.py

# 3. Integrate FoodKG recipes into KB
cd nutri_graph && python scripts/build_recipe_kb.py

# 4. Train GAT on full graph
cd nutri_graph && python scripts/train_GAT.py

# 5. (Optional, for PFoodReq) Build recipe text embeddings
cd nutri_rag && python scripts/build_recipe_embeddings.py
```

## Running

**LLM server** (required for everything except PFoodReq benchmark):

```bash
cd nutri_rag && bash scripts/start_server.sh   # Qwen3.5-9B on port 8080, parallel=1
```

For concurrent benchmark requests use `../../qwen_test/start_server.sh` (parallel=3, otherwise identical).

**Benchmarks:**

```bash
# NutriBench — single mode/nutrient
cd nutri_rag && python scripts/run_bench.py --mode v3 --nutrient protein --limit 200 --concurrent 3

# NutriBench — all combinations
cd nutri_rag && python scripts/run_all_bench.py --modes v0 v1 v3 --nutrients carb protein fat energy --limit 1000

# PFoodReq — no LLM server needed (deterministic filter + GAT)
cd nutri_rag && python scripts/run_pfoodreq_bench.py
```

**Interactive:**

```bash
cd nutri_rag && python scripts/demo_assistant.py          # nutrition assistant only
cd nutri-atlas/robot_control && python robot_assistant.py  # robot + nutrition
```

**Robot-side (run on the robot's onboard PC):**

```bash
# Must source ROS2 first
source /opt/ros/humble/setup.bash
source ~/test_ws/install/setup.bash

# Terminal 1 — RealSense camera + ZMQ image bridge + static TF
ros2 launch realsense_zmq bringup_with_zmq.launch.py

# Terminal 2 — ZMQ navigation bridge (port 5555)
cd nutri-atlas/robot_control/robot_side/zmq_bridge_real
python zmq_bridge_node_working_v2.py          # default port 5555
# Optional flags: --port 5555 --spin-kp 1.5 --move-kp 0.8
#                 --spin-threshold-deg 3.0 --move-threshold-m 0.05
# Also reads env var: ZMQ_PORT

# Terminal 3 — persistent object map server (port 5556, simulation only)
# python zmq_object_server.py                 # not needed in real-world mode

# (Optional) Record landmark coordinates manually for initialization
cd nutri-atlas/robot_control/robot_side
python coordinates_record.py --output landmarks_record.json
```

Run the robot assistant on the operator PC (CLI args take precedence over env vars):

```bash
# Real world
cd nutri-atlas/robot_control
python robot_assistant.py --robot-ip 192.168.0.114 --detection-mode real

# Simulation (default)
python robot_assistant.py --robot-ip 127.0.0.1

# Real-world detectors (separate terminal) — pick one:
cd nutri-atlas/robot_control/tools

# Manual: press Enter to push current frame to robot
python detector_node_real_world.py --robot-ip 192.168.0.114

# Automatic: sends stable detections without manual input
python detector_node_real_world_auto.py --robot-ip 192.168.0.114
python detector_node_real_world_auto.py --robot-ip 192.168.0.114 \
    --targets person chair --stable-conf 0.6 --stable-frames 10
```

## Architecture Details

### Cross-Subsystem Data Flow

`nutri_rag` reads from `nutri_graph` outputs via relative paths in `nutri_rag/nutri_rag/config.py`:
- `nutri_graph/data/nutri_kb.duckdb` — the shared DuckDB knowledge base
- `nutri_graph/outputs/embeddings/food_embeddings.npy` — 64d GAT embeddings
- `nutri_graph/outputs/embeddings/node_embeddings.pt` — all node type embeddings

`nutri-atlas` imports `nutri_rag` by adding it to `sys.path` at runtime (see `tools/nutrition_tool.py`).

### RAG Retrieval Versions

V0 (BM25), V1 (dense text embedding), V2 (dense + GAT re-ranking), V3 (multi-candidate + GAT + similarity threshold). V3 subsumes V2. Each version has a corresponding `tasks/` directory for lm-evaluation-harness.

### PFoodReq Pipeline

Config C: tag lookup → deterministic constraint filter (ingredient inclusion/exclusion + nutrient ranges) → GAT re-ranking. No LLM needed — `no_llm` is the default mode. The `with_llm` ablation exists but hurts performance.

### Robot Communication

Two ZMQ REQ/REP channels between operator PC (nutri-atlas) and robot onboard PC:
- Port 5555: navigation goals, spin, move, current objects, lidar (`zmq_bridge_node_working_v2.py`)
- Port 5556: persistent detected-objects map (`zmq_object_server.py`)

Robot IP configured via env vars: `ROBOT_IP`, `OBJECT_SERVER_IP`.

**`zmq_bridge_node_working_v2.py` ROS2 interface:**
- Sends `nav2_msgs/action/NavigateToPose` → `/navigate_to_pose` (navigate, waits for result)
- Publishes `geometry_msgs/Twist` → `/cmd_vel` (spin and move, P-controller)
- Subscribes `sensor_msgs/PointCloud2` ← `/sensor_scan` (cached for get_scan)
- Subscribes `std_msgs/String` ← `/detected_objects` (cached for get_current_objects, simulation only)
- Reads TF `map → base_link` (falls back to `base`, `vehicle`) for spin/move pose feedback
- Reads TF `map → camera_link` for transforming real-world YOLO detections to map frame
- Requires Nav2 stack running (`/navigate_to_pose` action server)

**Message types dispatched by the bridge:**
| Field | Command | Action |
|-------|---------|--------|
| `'x'` and `'y'` present | navigate | Nav2 action goal to `(x, y)` in map frame |
| `command_type: 'spin'` | spin `angle_deg` | P-controller on `/cmd_vel` |
| `command_type: 'move'` | move `distance_m` | P-controller on `/cmd_vel` |
| `command_type: 'get_current_objects'` | live objects | Latest `/detected_objects` topic (sim); falls back to `_stored_objects` in-memory (real) |
| `command_type: 'get_scan'` | lidar | 8-sector distances from `/sensor_scan` |
| `command_type: 'update_objects'` | ingest detections | Transform camera-frame YOLO detections to map frame via TF, broadcast static TF frames, store in memory + `~/detected_objects.json` (with spatial dedup: skip same-label within 0.5 m) |
| `command_type: 'get_detected_objects'` | persistent store | Read `~/detected_objects.json` — accumulates across sessions |
| `command_type: 'forget_object'` | delete entry | Remove one entry from `~/detected_objects.json` by frame name |

Note: `zmq_bridge_simulation/` is a reference for simulation only.
Use `zmq_bridge_real/zmq_bridge_node_working_v2.py` for real robot deployment.

**Real-world object detection (operator PC, `tools/`):**
- `detector_node_real_world.py` — manual: press Enter to send current YOLO detections to robot
- `detector_node_real_world_auto.py` — automatic: sends detections after N stable frames; supports `--targets` label filter and `--stable-conf` confidence gate
- `image_receiver.py` — ZMQ subscriber for RealSense color+depth streams (topics: `realsense/color`, `realsense/depth`)
- `detector_real_image.py` — standalone YOLO visualizer (no robot connection)

**`DETECTION_MODE` flag** (CLI `--detection-mode` or env var, default `sim`):

| | `sim` | `real` |
|---|---|---|
| `get_detected_objects` | port 5556 (`zmq_object_server`) | port 5555 bridge (`get_detected_objects` cmd) |
| `get_current_detected_objects` | `/detected_objects` ROS topic | `_stored_objects` in-memory fallback |
| `list_landmarks` | YAML only | YAML + detected objects from bridge |
| `navigate_to_landmark("name")` | YAML only | YAML + bridge store fallback |
| Robot processes needed | bridge + `zmq_object_server` | bridge + `bringup_with_zmq` + detector |

**`robot_assistant.py` CLI args** (new — previously env-var only):
```bash
python robot_assistant.py --robot-ip 192.168.0.114 --robot-port 5555 --detection-mode real
```
Args are set as env vars before tools are imported, so tools pick them up correctly.

### Key Config Constants (nutri_rag/config.py)

| Constant | Value | Notes |
|----------|-------|-------|
| `SIMILARITY_THRESHOLD` | 0.60 | Cosine sim filter for RAG candidates |
| `TOP_K_FOODS` | 3 | Max DB matches per food item |
| `GAT_NEIGHBORS_K` | 5 | GAT neighbors per seed |
| `PFOODREQ_LAMBDA` | 1.0 | Pure GAT scoring for PFoodReq |
| `LLM_BASE_URL` | localhost:8080 | llama-server endpoint |

## Key Files

- `nutri_rag/nutri_rag/config.py` — all paths, thresholds, and model settings
- `nutri_rag/nutri_rag/embedding.py` — TextEmbedder, FoodVectorIndex, GATIndex, RecipeVectorIndex
- `nutri_rag/nutri_rag/bench/retriever.py` — NutriBench retrieval (V1/V2/V3)
- `nutri_rag/nutri_rag/pfoodreq/retriever.py` — PFoodReq Config C pipeline
- `nutri_rag/nutri_rag/assistant/pipeline.py` — NutriAssistant end-to-end orchestration
- `nutri_graph/nutri_graph/kb/builder.py` — DuckDB KB construction
- `nutri_graph/nutri_graph/models/gat_model.py` — GATv2 (4 node types, dual decoders)
- `nutri-atlas/robot_control/robot_assistant.py` — Qwen Agent chat loop with tool dispatch
- `nutri-atlas/robot_control/tools/navigate_tool.py` — landmark + detected-object navigation; merges YAML + bridge store when `DETECTION_MODE=real`
- `nutri-atlas/robot_control/tools/object_tool.py` — `get_detected_objects` (sim: port 5556, real: port 5555) + `get_current_detected_objects` + `forget_object`
- `nutri-atlas/robot_control/tools/detector_node_real_world.py` — manual YOLO detector; press Enter to push detections to robot
- `nutri-atlas/robot_control/tools/detector_node_real_world_auto.py` — auto YOLO detector; stability gate + spatial dedup; `--targets` + `--stable-conf` args
- `nutri-atlas/robot_control/tools/image_receiver.py` — ZMQ SUB for RealSense color+depth streams
- `nutri-atlas/robot_control/robot_side/zmq_bridge_real/zmq_bridge_node_working_v2.py` — ZMQ REP server (robot): navigate + spin/move + scan + update_objects + get_detected_objects
- `nutri-atlas/robot_control/robot_side/zmq_bridge_real/zmq_object_server.py` — ZMQ REP server (robot, simulation only): serves persistent detected-objects map on port 5556
- `nutri-atlas/robot_control/robot_side/zmq_bridge_real/realsense_zmq/launch/bringup_with_zmq.launch.py` — launches RealSense driver + ZMQ image bridge + static TF `base_link → camera_link`
- `nutri-atlas/robot_control/robot_side/coordinates_record.py` — TF-based tool to record landmark (x, y) coordinates interactively
- `nutri-atlas/scripts/benchmark_cost.py` — agent cost benchmark (time, tokens, power); use `--mock` to bypass ZMQ

## Git

Single repo on `dev` branch. Large generated files (data/, outputs/, embeddings/, results/) are gitignored.
