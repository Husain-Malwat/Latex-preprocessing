#!/bin/bash
# ==========================================================
# Batch Parallel LaTeX Compilation Script (SLURM Compatible)
# Author: Husain Malwat
# ==========================================================

# ====== USER CONFIGURATION ======
BASE_DIR="/home/husainmalwat/workspace/OCR_Latex/data/final_merged"
SCRIPT_PATH="/home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/Latex-preprocessing/compile_latex.py"  # your Python script
LOG_DIR="./compile_logs"
MAX_JOBS=32   # number of parallel jobs (use all CPUs)
# ==========================================================

mkdir -p "$LOG_DIR"

# === STEP 1: Collect all target subdirectories ===
echo "📁 Collecting all month-level directories..."
dirs=()

# Years 2000–2022 (each with 12 months)
for year in $(seq 2000 2022); do
  for month_dir in "$BASE_DIR/$year/final_merged/"*/; do
    if [ -d "$month_dir" ]; then
      dirs+=("$month_dir")
    fi
  done
done

# Year 2023 (only 8 months)
for month_dir in $(ls -d "$BASE_DIR/2023/final_merged/"*/ | head -n 8); do
  dirs+=("$month_dir")
done

echo "✅ Found ${#dirs[@]} directories to process."

# === STEP 2: Run compilations in parallel ===
run_dir_compile() {
    dir="$1"
    dir_name=$(echo "$dir" | sed 's|/|_|g' | sed 's|:|_|g')
    log_file="$LOG_DIR/${dir_name}.log"

    echo "🚀 [$(date '+%H:%M:%S')] Starting: $dir" | tee -a "$log_file"
    python3 "$SCRIPT_PATH" --mode single --dir "$dir" >> "$log_file" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ [$(date '+%H:%M:%S')] Completed: $dir" | tee -a "$log_file"
    else
        echo "❌ [$(date '+%H:%M:%S')] Failed: $dir (exit code $status)" | tee -a "$log_file"
    fi
}

export -f run_dir_compile
export SCRIPT_PATH LOG_DIR

echo "⚙️ Starting parallel execution with $MAX_JOBS jobs..."
printf "%s\n" "${dirs[@]}" | xargs -n 1 -P "$MAX_JOBS" bash -c 'run_dir_compile "$@"' _

echo "🏁 All directory compilations complete."
