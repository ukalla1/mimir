#!/bin/bash
# Run NutriBench v2 RAG benchmark against the local llama-server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIMIT="${1:-}"    # pass a number as $1 to limit samples, e.g. "100"

echo "==> Running NutriBench v2 RAG benchmark"
echo ""

python "$SCRIPT_DIR/run_bench.py" ${LIMIT:+--limit $LIMIT}
