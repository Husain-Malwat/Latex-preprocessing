#!/bin/bash
# ==========================================================
# Parallel Token Counting Script using vLLM Model
# Author: Husain Malwat
# ==========================================================

# ===== USER CONFIGURATION =====
BASE_DIR="/home/husainmalwat/workspace/OCR_Latex/data/final_merged"
SCRIPT_PATH="/home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/Latex-preprocessing/count_tokens.py"
MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
LOG_DIR="./token_count_logs"
MAX_JOBS=32          # parallel processes
OUTPUT_DIR="./token_count_outputs"
# ==========================================================

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# === STEP 1: Collect all directories ===
echo "📁 Collecting all month-level directories..."
dirs=()

for year in $(seq 2000 2022); do
  for month_dir in "$BASE_DIR/$year/final_merged/"*/; do
    if [ -d "$month_dir" ]; then
      dirs+=("$month_dir")
    fi
  done
done

# Year 2023 — only available months
for month_dir in "$BASE_DIR/2023/final_merged/"*/; do
  if [ -d "$month_dir" ]; then
    dirs+=("$month_dir")
  fi
done

echo "✅ Found ${#dirs[@]} directories to process."
echo "⚙️ Running with up to $MAX_JOBS parallel jobs..."
echo "==========================================================="

# === STEP 2: Define runner function ===
run_token_count() {
    dir="$1"
    year=$(echo "$dir" | grep -oE "20[0-9]{2}")
    month=$(basename "$dir")
    safe_name=$(echo "$dir" | sed 's|/|_|g' | sed 's|:|_|g')

    log_file="$LOG_DIR/${safe_name}.log"
    out_file="$OUTPUT_DIR/token_stats_${year}_${month}.json"

    echo "🚀 [$(date '+%H:%M:%S')] Starting: $dir" | tee -a "$log_file"
    python3 "$SCRIPT_PATH" "$dir" \
        --model-name "$MODEL_NAME" \
        --output "$out_file" \
        --find-outliers >> "$log_file" 2>&1

    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ [$(date '+%H:%M:%S')] Completed: $dir" | tee -a "$log_file"
    else
        echo "❌ [$(date '+%H:%M:%S')] Failed: $dir (exit code $status)" | tee -a "$log_file"
    fi
}

export -f run_token_count
export SCRIPT_PATH MODEL_NAME LOG_DIR OUTPUT_DIR

# === STEP 3: Parallel Execution ===
printf "%s\n" "${dirs[@]}" | xargs -n 1 -P "$MAX_JOBS" bash -c 'run_token_count "$@"' _

echo "🏁 All token counting tasks finished."
echo "Logs saved to: $LOG_DIR"
echo "Outputs saved to: $OUTPUT_DIR"
