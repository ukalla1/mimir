# Qwen Robot Navigation Assistant

A natural language robot control system powered by Qwen 3.5-9B running locally via llama.cpp. The agent interprets user commands and coordinates multi-step navigation, detection, and exploration tasks on a Clearpath Go2 robot.

## Overview

The assistant exposes a set of tools to the LLM — navigate to landmarks, scan for objects along a route, register detections as permanent landmarks, spin/move the robot, and query a persistent object map — and lets the model reason over them to satisfy complex requests like "go to the kitchen looking for objects on the way, then remember what you found."

The project also includes a benchmark suite of 30+ graded tasks (easy / medium / challenging) for evaluating agent reasoning and tool-calling accuracy, along with Gantt chart visualizations of tool call sequences.

```
User ──► Qwen 3.5-9B Agent ──► Tool calls ──► ZMQ bridge ──► ROS2 / Robot
```

## Architecture

```
robot_assistant.py           # Main chat loop + LLM agent
robot_control/
  tools/
    navigate_tool.py              # Navigate to named landmarks, coordinates, or detected objects
    detect_tool.py                # navigate_and_scan, scan_objects, register_objects
    object_tool.py                # Query live camera or stored detection map (sim/real switch)
    motion_tool.py                # Spin and move primitives
    lidar_tool.py                 # LiDAR scan data
    zmq_client.py                 # ZMQ transport layer
    landmark_loader.py            # Case-insensitive YAML landmark lookup
    detector_node_real_world.py   # YOLO detector (RealSense); press Enter to push to robot
    detector_node_real_world_auto.py  # Auto YOLO detector; stability gate, no manual input
    detector_core.py              # Shared YOLO + depth pipeline
    image_receiver.py             # ZMQ subscriber for RealSense color+depth streams
  config/
    landmarks.yaml           # Named room positions
  robot_side/
    zmq_bridge_real/
      zmq_bridge_node_working_v2.py  # ZMQ REP server on robot (port 5555)
      zmq_object_server.py           # Simulation-only persistent object map (port 5556)
      realsense_zmq/launch/          # RealSense + ZMQ image bridge + static TF
    coordinates_record.py      # Record landmark (x,y) coordinates interactively via TF
scripts/
  start_server.sh           # Starts the llama.cpp inference server
benchmark_tasks.yaml        # 30+ graded benchmark tasks
figure/                     # Gantt chart generators for tool call analysis
```

## Requirements

**System**

- llama.cpp compiled with GPU support (see [root README](../README.md) for build instructions)
- Qwen 3.5-9B GGUF model (UD-Q4_K_XL, ~6 GB)
- Python 3.9+

**Python packages**

```bash
pip install qwen-agent pyzmq pyyaml json5
```

For real-world detection:
```bash
pip install ultralytics opencv-python
```

## Setup

### 1. Start the LLM server

```bash
cd ~/work/atlas/mimir/nutri_rag
bash scripts/start_server.sh
# Serves Qwen3.5-9B on http://localhost:8080/v1
```

### 2. Run the assistant

CLI args take precedence over environment variables:

```bash
# Simulation (default — no real robot required)
cd nutri-atlas/robot_control
python robot_assistant.py --robot-ip 127.0.0.1

# Real world
python robot_assistant.py --robot-ip 192.168.0.114 --detection-mode real
```

### 3. Real-world only: start the YOLO detector (separate terminal)

```bash
cd nutri-atlas/robot_control/tools

# Manual: press Enter to push current frame to robot
python detector_node_real_world.py --robot-ip 192.168.0.114

# Automatic: sends detections after N stable frames without manual input
python detector_node_real_world_auto.py --robot-ip 192.168.0.114
python detector_node_real_world_auto.py --robot-ip 192.168.0.114 \
    --targets person chair --stable-conf 0.6 --stable-frames 10
```

### 4. Robot-side (on the robot's onboard PC)

```bash
source /opt/ros/humble/setup.bash && source ~/test_ws/install/setup.bash

# Terminal 1 — RealSense camera + ZMQ image bridge + static TF (base_link → camera_link)
ros2 launch realsense_zmq bringup_with_zmq.launch.py

# Terminal 2 — ZMQ navigation bridge (port 5555)
cd nutri-atlas/robot_control/robot_side/zmq_bridge_real
python zmq_bridge_node_working_v2.py
# Optional: --port 5555 --spin-kp 1.5 --move-kp 0.8 --spin-threshold-deg 3.0 --move-threshold-m 0.05

# (Optional) Record landmark coordinates
cd nutri-atlas/robot_control/robot_side
python coordinates_record.py --output landmarks_record.json
```

## Configuration

### Landmarks

Edit `robot_control/config/landmarks.yaml` to define named locations:

```yaml
landmarks:
  Reception:
    x: -10.7
    y: 1.8
    description: "Reception area"
  Kitchen:
    x: -4.88
    y: -3.46
    description: "Kitchen area"
```

Landmark names are resolved case-insensitively — the LLM can send "reception" and it will match "Reception".

### Detection modes

| | `sim` (default) | `real` |
|---|---|---|
| `get_detected_objects` | port 5556 (`zmq_object_server`) | port 5555 bridge |
| `list_landmarks` | YAML only | YAML + detected objects from bridge |
| Robot processes needed | bridge + `zmq_object_server` | bridge + `bringup_with_zmq` + detector |

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `list_landmarks` | List all navigable locations — fixed landmarks + detected objects (real world) |
| `navigate_to_landmark` | Go to a named landmark or detected object by name, or by (x, y) coordinates |
| `navigate_and_scan` | Travel to a landmark while scanning for objects at each waypoint; accumulates detections in temp memory |
| `scan_objects` | Take a detection snapshot at the current pose |
| `register_objects` | Persist detections as permanent landmarks. Auto-drains temp memory from `navigate_and_scan` if available; otherwise runs a fresh scan |
| `get_detected_objects` | Query stored detection snapshot (sim: port 5556; real: port 5555) |
| `get_current_detected_objects` | Query what the camera sees right now; falls back to stored detections in real world |
| `forget_object` | Remove a detection from the persistent map by name |
| `spin_robot` | Rotate the robot by a given angle |
| `move_robot` | Move forward/backward (disabled by default) |
| `get_lidar_scan` | Get LiDAR distance readings (disabled by default) |
| `get_meal_recommendation` | Personalized meal recommendation via nutri_rag |

### Common tool sequences

| User intent | Tool sequence |
|---|---|
| "Go to kitchen" | `navigate_to_landmark` |
| "Go to kitchen looking for objects" | `navigate_and_scan` → `register_objects` |
| "What can you see?" | `scan_objects` |
| "Remember what you see" | `register_objects` |
| "Find a chair" | `list_landmarks` → if not found: `navigate_and_scan` across landmarks |
| "Go to the person you detected" | `get_detected_objects` → `navigate_to_landmark(x, y)` |

## Example Usage

```
User: Go to the reception.
User: Go to the kitchen and look for objects on the way.
User: What do you see right now?
User: Remember everything you just found.
User: Find a chair and navigate to it.
User: Turn left 90 degrees.
User: What objects have been detected so far?
```

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
