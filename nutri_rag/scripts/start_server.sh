#!/bin/bash
# Start llama-server for nutri_rag assistant mode.
#
# Differences from qwen_test/start_server.sh:
#   - Thinking enabled (better reasoning for gap analysis)
#   - parallel=1 (assistant is single-user, sequential requests)
set -e

MODEL_PATH="/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
PORT=8080

echo "==> Starting llama-server for nutri_rag assistant (port $PORT) ..."
echo "    Model: $MODEL_PATH"
echo "    Thinking: enabled"
echo "    Press Ctrl+C to stop."
echo ""

~/softwares/llama.cpp/llama-server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --ctx-size 32768 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'
    
