#!/bin/bash
# Multi-GPU inference by splitting data across GPUs

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_PATH=$1
TEST_FILE=$2
OUTPUT_FILE=$3
NUM_GPUS=${4:-4}
MAX_SAMPLES=${5:-}
MAX_NEW_TOKENS=${6:-512}
TEMPERATURE=${7:-0.7}
TOP_P=${8:-0.8}
MIN_P=${9:-0}
TOP_K=${10:-20}
IS_IFEVAL_DATA=${12:-false}

# Auto-detect latest checkpoint if directory contains checkpoints
if [ -d "$MODEL_PATH" ] && ls "$MODEL_PATH"/checkpoint-* 1> /dev/null 2>&1; then
    LATEST_CKPT=$(ls -d "$MODEL_PATH"/checkpoint-* | sort -V | tail -n 1)
    echo "Auto-detected latest checkpoint: $LATEST_CKPT"
    MODEL_PATH=$LATEST_CKPT
fi

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
OUTPUT_BASE=$(basename "$OUTPUT_FILE" .jsonl)

echo "Running multi-GPU inference with $NUM_GPUS GPUs"
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Min-p: $MIN_P"
echo "Top-k: $TOP_K"
echo "IFEval data format: $IS_IFEVAL_DATA"

mkdir -p $OUTPUT_DIR

# Split data and run on each GPU in parallel
for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "Starting GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_DIR/batch_inference.py \
        --model_path $MODEL_PATH \
        --test_file $TEST_FILE \
        --output_file $OUTPUT_DIR/${OUTPUT_BASE}_gpu${i}.jsonl \
        --gpu_id $i \
        --num_gpus $NUM_GPUS \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --min_p $MIN_P \
        --top_k $TOP_K \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        $([ "$IS_IFEVAL_DATA" = "true" ] && echo "--is_IFEval_data") &
done

# Wait for all processes to complete
wait

echo "All GPUs completed. Merging results..."

# Merge all prediction files
cat $OUTPUT_DIR/${OUTPUT_BASE}_gpu*.jsonl > $OUTPUT_FILE

echo "Done! Results saved to $OUTPUT_FILE"

# Clean up individual GPU files
rm $OUTPUT_DIR/${OUTPUT_BASE}_gpu*.jsonl
echo "Cleaned up temporary GPU files"
