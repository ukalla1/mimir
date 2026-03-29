#!/bin/bash
# Qwen3.5-9B llama.cpp Server Startup Script

MODEL="$HOME/Projects/qwen/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"
LLAMA_SERVER="$HOME/Projects/qwen/llama.cpp/llama-server"
PORT=${PORT:-8001}

echo "Starting Qwen3.5-9B llama-server on port $PORT..."

# Thinking mode (reasoning enabled)
"$LLAMA_SERVER" \
    --model "$MODEL" \
    --ctx-size 32768 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --alias "unsloth/Qwen3.5-9B-GGUF" \
    --port "$PORT" \
    --host 0.0.0.0 \
    -ngl 99 \
    --chat-template-kwargs '{"enable_thinking":true}'
