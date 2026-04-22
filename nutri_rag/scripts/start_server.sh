#!/bin/bash
# Start llama-server for nutri_rag (Qwen3.5-9B on port 8080).
#
# This is the primary server script for all Mimir usage:
#   - NutriBench evaluation
#   - Interactive nutrition assistant (demo_assistant.py)
#   - Robot assistant with nutrition (nutri-atlas/robot_assistant.py)
#   - PFoodReq benchmark does NOT need this server
#
# parallel=1 is suitable for single-user assistant and robot usage.
# For benchmark evaluation with concurrent requests, use:
#   qwen_test/start_server.sh  (identical config but parallel=3)
set -e

MODEL_PATH="/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
MMPROJ_PATH="/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
PORT=8080

echo "==> Starting llama-server for nutri_rag assistant (port $PORT) ..."
echo "    Model: $MODEL_PATH"
echo "    Thinking: enabled"
echo "    Press Ctrl+C to stop."
echo ""

~/softwares/llama.cpp/llama-server \
    --model "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --ctx-size 32768 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'
    
