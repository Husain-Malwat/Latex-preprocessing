#!/bin/bash

# Configuration
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.9
DTYPE="bfloat16"

# Launch GPU 0 Server
echo "Starting vLLM server on GPU 0 (port 8000)..."
CUDA_VISIBLE_DEVICES=0 nohup vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype $DTYPE \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  > serve_gpu0.log 2>&1 &

echo "GPU 0 server PID: $!"

# Wait a bit to avoid race conditions
sleep 5

# Launch GPU 1 Server
echo "Starting vLLM server on GPU 1 (port 8001)..."
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype $DTYPE \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  > serve_gpu1.log 2>&1 &

echo "GPU 1 server PID: $!"

echo ""
echo "✓ Both servers launched!"
echo "  GPU 0: http://localhost:8000/v1"
echo "  GPU 1: http://localhost:8001/v1"
echo ""
echo "Check logs:"
echo "  tail -f serve_gpu0.log"
echo "  tail -f serve_gpu1.log"