# Qwen Robot Navigation Assistant

A natural language robot control system powered by Qwen 3.5-9B running locally via llama.cpp. The agent interprets user commands and coordinates multi-step navigation and exploration tasks on a Clearpath Go2 robot.

## Overview

The assistant exposes a set of tools to the LLM — navigate to landmarks, detect objects, spin/move the robot, and query a persistent object map — and lets the model reason over them to satisfy complex requests like "find a chair and go stand next to it."

The project also includes a benchmark suite of 30+ graded tasks (easy / medium / challenging) for evaluating agent reasoning and tool-calling accuracy, along with Gantt chart visualizations of tool call sequences.

```
User ──► Qwen 3.5-9B Agent ──► Tool calls ──► ZMQ bridge ──► ROS2 / Robot
```

## Architecture

```
robot_assistant.py       # Main chat loop + LLM agent
robot_control/
  tools/
    navigate_tool.py          # Navigate to named landmarks, coordinates, or detected objects
    object_tool.py            # Query live camera or stored detection map (sim/real switch)
    motion_tool.py            # Spin and move primitives
    lidar_tool.py             # LiDAR scan data
    zmq_client.py             # ZMQ transport layer
    landmark_loader.py        # Loads landmarks from YAML
    detector_node_real_world.py  # YOLO detector (RealSense); press Enter to push to robot
    image_receiver.py         # ZMQ subscriber for RealSense color+depth streams
    detector_real_image.py    # Standalone YOLO visualizer (no robot)
  config/
    landmarks.yaml       # Named room positions (kitchen, bedroom, etc.)
scripts/
  start_server.sh        # Starts the llama.cpp inference server
benchmark_tasks.yaml     # 30+ graded benchmark tasks
figure/                  # Gantt chart generators for tool call analysis
```

## Requirements

**System**

- llama.cpp compiled with GPU support
- Qwen 3.5-9B GGUF model (~5.6 GB)
- Vision encoder `mmproj-F16.gguf` (~876 MB)
- Python 3.10+

**Python packages**

```
qwen-agent
pyzmq
pyyaml
json5
```

Install with:

```bash
pip install qwen-agent pyzmq pyyaml json5
```

## Setup

### 1. Start the LLM server

```bash
conda activate qwen
bash scripts/start_server.sh
# Serves on http://localhost:8001/v1
```

Key server settings (edit `start_server.sh` to change):

| Flag | Default | Description |
|------|---------|-------------|
| `--ctx-size` | 32768 | Context window size |
| `--temp` | 0.6 | Sampling temperature |
| `--top-p` | 0.95 | Top-p sampling |
| `enable_thinking` | true | Extended chain-of-thought reasoning |

### 2. Run the assistant

```bash
export ROBOT_IP=<your-robot-ip>
export DETECTION_MODE=real   # set for real robot; omit for simulation (default: sim)
python robot_control/robot_assistant.py

# Real world only — in a second terminal, run the YOLO detector:
cd robot_control/tools
python detector_node_real_world.py --robot-ip <your-robot-ip>
# Point camera at objects and press Enter to push detections to the robot.
# They will appear in list_landmarks and get_detected_objects automatically.
```

## Configuration

### Landmarks

Edit `robot_control/config/landmarks.yaml` to define named locations:

```yaml
landmarks:
  kitchen:
    x: -4.88
    y: -3.46
    description: "Kitchen area"
  bedroom:
    x: 1.20
    y: 3.10
    description: "Bedroom"
```

## Example Usage

```
User: Go to the kitchen and tell me what you see.
User: Do a full 360-degree scan and list every object you find.
User: Find a chair and navigate to it.
User: Turn left 90 degrees.
User: What objects have been detected in the apartment so far?
```

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `list_landmarks` | List all navigable locations — fixed landmarks + detected objects (real world) |
| `navigate_to_landmark` | Go to a named landmark or detected object by name, or by (x, y) coordinates |
| `get_detected_objects` | Query stored detection snapshot (sim: port 5556; real: port 5555) |
| `get_current_detected_objects` | Query what the camera sees right now; falls back to stored detections in real world |
| `spin_robot` | Rotate the robot by a given angle |
| `move_robot` | Move forward/backward (disabled by default) |
| `get_lidar_scan` | Get LiDAR distance readings (disabled by default) |
| `get_meal_recommendation` | Personalized meal recommendation via nutri_rag |

## Benchmarking

`benchmark_tasks.yaml` contains 30+ tasks organized by difficulty:

- **Easy** — single or two-tool calls (navigate to room, look at scene, rotate)
- **Medium** — 3–5 tool calls (multi-room search, compare map vs. live view)
- **Challenging** — 6+ tool calls using all tools (find object, verify, navigate, report)

Each task includes the natural language prompt, expected tool sequence, and reasoning notes.

### Visualization

Generate Gantt charts of tool call sequences:

```bash
python figure/gantt_multi_task.py
python figure/gantt_three_levels.py
```
