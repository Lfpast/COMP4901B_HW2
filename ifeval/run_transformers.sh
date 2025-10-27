#!/bin/bash
# IFEval runner using Transformers (no vLLM required)
# This script can be run on any machine with PyTorch and Transformers installed
#
# Usage:
#   bash run_transformers.sh <MODEL_PATH> <OUTPUT_DIR>
#
# Example:
#   bash run_transformers.sh ../SmolLM2-135M results/base
#   bash run_transformers.sh ../checkpoint-xxx results/finetuned

MODEL_PATH=$1
OUTPUT_DIR=$2

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is required"
    echo "Usage: bash run_transformers.sh <MODEL_PATH> <OUTPUT_DIR>"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR is required"
    echo "Usage: bash run_transformers.sh <MODEL_PATH> <OUTPUT_DIR>"
    exit 1
fi

echo "=========================================="
echo "IFEval with Transformers (no vLLM)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python run_ifeval_transformers.py \
    --mode all \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --device cuda \
    --dtype auto
