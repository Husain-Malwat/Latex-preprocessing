# Multi-GPU vLLM Setup Guide

Complete guide for setting up and using multi-GPU data parallelism with vLLM for batch LaTeX preprocessing.

## Table of Contents
1. [Overview](#overview)
2. [Why Multi-GPU Data Parallelism](#why-multi-gpu-data-parallelism)
3. [Server Setup](#server-setup)
4. [Client Configuration](#client-configuration)
5. [Usage Examples](#usage-examples)
6. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
7. [Performance Optimization](#performance-optimization)

---

## Overview

Multi-GPU data parallelism runs **independent vLLM servers** on different GPUs, each handling different requests. This provides:

- **2x throughput** with 2 GPUs (scales linearly)
- **No NCCL overhead** (no inter-GPU communication)
- **Maximum stability** for batch workloads
- **Simple fault tolerance** (if one server fails, others continue)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│               Your Preprocessing Client             │
│          (load balances across servers)             │
└─────────┬──────────────────────────┬────────────────┘
          │                          │
    ┌─────▼─────┐             ┌──────▼──────┐
    │  vLLM     │             │   vLLM      │
    │  Server   │             │   Server    │
    │  Port     │             │   Port      │
    │  8000     │             │   8001      │
    └─────┬─────┘             └──────┬──────┘
          │                          │
    ┌─────▼─────┐             ┌──────▼──────┐
    │  GPU 0    │             │   GPU 1     │
    │  Model    │             │   Model     │
    │  Copy     │             │   Copy      │
    └───────────┘             └─────────────┘
```

---

## Why Multi-GPU Data Parallelism?

### ✅ Recommended For:
- **Batch preprocessing** (hundreds/thousands of files)
- **High concurrency** workloads (many simultaneous requests)
- **Independent requests** (each file is processed separately)
- **Production deployments** requiring stability

### ❌ NOT Recommended For:
- **Single large requests** (use tensor parallelism instead)
- **Interactive use** (single GPU is simpler)
- **Models that don't fit on one GPU** (use tensor parallelism)



## Server Setup

### Step 1: Check Available GPUs

```bash
# Check GPU availability
nvidia-smi
```

### Step 2: Launch vLLM Servers

Create a script `start_vllm_servers.sh`:

```bash
#!/bin/bash

# Configuration
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_MODEL_LEN=8192
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
```

Make it executable and run:

```bash
chmod +x start_vllm_servers.sh
./start_vllm_servers.sh
```

### Step 3: Verify Servers Are Running

```bash
# Check if servers are listening
netstat -tuln | grep -E '8000|8001'

# Should show:
# tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN
# tcp        0      0 0.0.0.0:8001            0.0.0.0:*               LISTEN

# Test API endpoints
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

### Step 4: Monitor Server Logs

```bash
# Watch both logs simultaneously
tail -f serve_gpu0.log serve_gpu1.log

# Or in separate terminals
terminal1$ tail -f serve_gpu0.log
terminal2$ tail -f serve_gpu1.log
```

---

## Client Configuration

### Updated Command-Line Arguments

The preprocessor now supports multi-GPU configurations:

```bash
python run_llm_preprocessor.py \
    <input> <output> <prompt_template> \
    --backend vllm \
    --vllm-endpoints "http://localhost:8000/v1" "http://localhost:8001/v1" \
    --load-balancing random \
    --concurrency 16
```

### New Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--vllm-endpoints` | List[str] | Single endpoint | List of vLLM server URLs for multi-GPU |
| `--load-balancing` | str | `random` | Strategy: `random` or `round-robin` |

### Load Balancing Strategies

#### 1. Random (Default)
```bash
--load-balancing random
```
- Randomly selects a server for each request
- **Pros:** Simple, good distribution over time
- **Cons:** May have slight imbalance in short runs
- **Best for:** Long-running batch jobs

#### 2. Round-Robin
```bash
--load-balancing round-robin
```
- Cycles through servers in order
- **Pros:** Perfectly balanced distribution
- **Cons:** Predictable pattern
- **Best for:** When you want guaranteed even distribution

---

## Usage Examples

### Example 1: Basic Multi-GPU Setup (2 GPUs)

```bash
# Start servers (in background)
./start_vllm_servers.sh

# Run preprocessing with both GPUs
nohup python run_llm_preprocessor.py \
    /home/husainmalwat/workspace/OCR_Latex/data/less_than_5k \
    /home/husainmalwat/workspace/OCR_Latex/data/less_than_5k_Preprocessed \
    /home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/Latex-preprocessing/sample_prompts/prompt_normalization.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm-endpoints "http://localhost:8000/v1" "http://localhost:8001/v1" \
    --load-balancing random \
    --max-tokens 8192 \
    --concurrency 16 \
    --temperature 0.5 \
    > client_logs/client_log_multi_gpu.txt 2>&1 &
```

### Example 2: 4-GPU Setup

```bash
# Start 4 servers
CUDA_VISIBLE_DEVICES=0 vllm serve ... --port 8000 > gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve ... --port 8001 > gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 vllm serve ... --port 8002 > gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 vllm serve ... --port 8003 > gpu3.log 2>&1 &

# Run preprocessing
python run_llm_preprocessor.py \
    input/ output/ prompt.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm-endpoints \
        "http://localhost:8000/v1" \
        "http://localhost:8001/v1" \
        "http://localhost:8002/v1" \
        "http://localhost:8003/v1" \
    --load-balancing round-robin \
    --concurrency 32
```

### Example 3: Remote Servers

```bash
# If servers are on different machines
python run_llm_preprocessor.py \
    input/ output/ prompt.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm-endpoints \
        "http://192.168.1.100:8000/v1" \
        "http://192.168.1.101:8000/v1" \
    --concurrency 16
```

### Example 4: High Concurrency Processing

```bash
# Maximum throughput configuration
python run_llm_preprocessor.py \
    large_dataset/ processed/ prompt.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm-endpoints "http://localhost:8000/v1" "http://localhost:8001/v1" \
    --load-balancing random \
    --concurrency 32 \
    --max-tokens 8192 \
    --temperature 0.3 \
    --save-raw-responses \
    --stats-file stats_multi_gpu.jsonl \
    --csv-stats-file stats_multi_gpu.csv
```

---

## Monitoring & Troubleshooting

### Monitor GPU Usage

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Expected output (both GPUs should show utilization):
# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A    123456      C   vllm.entrypoints              28000MiB  |
# |    1   N/A  N/A    123457      C   vllm.entrypoints              28000MiB  |
# +-----------------------------------------------------------------------------+
```

### Check Server Health

Create a monitoring script `check_servers.sh`:

```bash
#!/bin/bash

ENDPOINTS=("http://localhost:8000/v1" "http://localhost:8001/v1")

for i in "${!ENDPOINTS[@]}"; do
    endpoint="${ENDPOINTS[$i]}"
    echo "Checking Server $i: $endpoint"
    
    response=$(curl -s "${endpoint}/models" | head -c 100)
    
    if [ -n "$response" ]; then
        echo "  ✓ Server $i is responding"
    else
        echo "  ✗ Server $i is NOT responding"
    fi
    echo ""
done
```

### View Request Distribution

The preprocessing stats will include `server_endpoint` field:

```bash
# Check which servers handled requests
cat processing_stats.csv | cut -d',' -f11 | sort | uniq -c

# Output example:
#    250 http://localhost:8000/v1
#    248 http://localhost:8001/v1
```

## Performance Optimization

### Benchmark Your Setup

Create `benchmark.sh`:

```bash
#!/bin/bash

# Test with different concurrency levels
for concurrency in 8 16 24 32 40; do
    echo "Testing concurrency: $concurrency"
    
    time python run_llm_preprocessor.py \
        test_input/ test_output_${concurrency}/ prompt.md \
        --backend vllm \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --vllm-endpoints "http://localhost:8000/v1" "http://localhost:8001/v1" \
        --concurrency $concurrency \
        --stats-file bench_${concurrency}.jsonl \
        --csv-stats-file bench_${concurrency}.csv
    
    echo "---"
done

# Analyze results
echo "Performance Summary:"
for concurrency in 8 12 16 20 24; do
    echo -n "Concurrency $concurrency: "
    grep "files_per_second" bench_${concurrency}.jsonl | tail -1
done
```

## Shutdown Procedures

### Graceful Shutdown

Create `stop_vllm_servers.sh`:

```bash
#!/bin/bash

echo "Stopping vLLM servers..."

# Find vLLM processes
PIDS=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No vLLM servers running"
    exit 0
fi

echo "Found PIDs: $PIDS"

# Send SIGTERM for graceful shutdown
for pid in $PIDS; do
    echo "Stopping PID $pid..."
    kill -TERM $pid
done

# Wait up to 30 seconds
echo "Waiting for graceful shutdown..."
sleep 30

# Force kill if still running
REMAINING=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}')
if [ -n "$REMAINING" ]; then
    echo "Force killing remaining processes..."
    kill -9 $REMAINING
fi

echo "✓ All servers stopped"
```

### Emergency Stop

```bash
# Kill all vLLM processes immediately
pkill -9 -f "vllm serve"

# Verify
ps aux | grep vllm
```

---

## Quick Reference Commands

```bash
# START SERVERS
./start_vllm_servers.sh

# CHECK STATUS
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models

# RUN PREPROCESSING (2 GPUs)
python run_llm_preprocessor.py \
    input/ output/ prompt.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm-endpoints "http://localhost:8000/v1" "http://localhost:8001/v1" \
    --load-balancing random \
    --concurrency 16

# MONITOR
watch -n 1 nvidia-smi
tail -f serve_gpu0.log serve_gpu1.log

# STOP SERVERS
./stop_vllm_servers.sh
```
