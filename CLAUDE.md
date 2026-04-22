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

- **nutri_graph/**: DuckDB knowledge base (USDA + FoodKG) + heterogeneous GATv2 training. Sources: ~74K USDA FDC → ~4.6K after cleaning; 7,793 SR Legacy; ~82K FoodKG recipes matched via Qwen3-Embedding.
- **nutri_rag/**: Four RAG retrieval versions (V0–V3), NutriBench/PFoodReq benchmarks, nutrition assistant
- **nutri-atlas/**: Qwen Agent with tool-calling for robot navigation + nutrition advice
- **nutri-atlas/robot_control/robot_side/**: ZMQ REP servers that run on the robot's onboard PC. Two sub-directories:
  - `zmq_bridge_real/` — **production**: `zmq_bridge_node_working_v2.py` (Nav2 action client) + `zmq_object_server.py`
  - `zmq_bridge_simulation/` — simulation variants (reference only)
  - `coordinates_record.py` — TF-based tool for recording landmark (x, y) coordinates interactively
- **PFoodReq/**: External benchmark data (WSDM 2021), gitignored
- **foodkg.github.io/**: External FoodKG construction project, gitignored


## Git

Single repo on `dev` branch. Large generated files (data/, outputs/, embeddings/, results/) are gitignored.

## Plan

When propose a plan, put the md files under plans folder
