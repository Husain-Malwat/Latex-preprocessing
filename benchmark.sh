#!/bin/bash

# Test with different concurrency levels
for concurrency in 8 16 24 32 40 48; do
    echo "Testing concurrency: $concurrency"
    
    time python run_llm_preprocessor.py \
        /home/husainmalwat/workspace/OCR_Latex/data/demo_test /home/husainmalwat/workspace/OCR_Latex/data/demo_test_output_${concurrency}/ /home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/Latex-preprocessing/sample_prompts/prompt_normalization.md \
        --backend vllm \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --max-tokens 16384 \
        --temperature 0.5 \
        --vllm-endpoints "http://localhost:8000/v1" "http://localhost:8001/v1" \
        --concurrency $concurrency \
        --stats-file bench_${concurrency}.jsonl \
        --csv-stats-file bench_${concurrency}.csv
    
    echo "---"
done

# Analyze results
echo "Performance Summary:"
for concurrency in 8 16 24 32 40 48; do
    echo -n "Concurrency $concurrency: "
    grep "files_per_second" bench_${concurrency}.jsonl | tail -1
done