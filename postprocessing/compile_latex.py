import os
import subprocess
import argparse
import logging
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from datetime import datetime

# ================================================================
# Default configurations
# ================================================================
DEFAULT_BASE_DIR = "/home/husainmalwat/workspace/OCR_Latex/data/final_merged"
DEFAULT_WORKERS = 12
DEFAULT_TIMEOUT = 5  # seconds
LOG_DIR = "./logs"

# ================================================================
# Setup Logging
# ================================================================
def setup_logger(log_name):
    """Create and configure a dedicated logger for each run."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Console handler (INFO level only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_path

# ================================================================
# LaTeX Compilation
# ================================================================
def compile_latex_file(tex_path, output_dir, timeout, logger):
    """Compile a single LaTeX file using pdflatex with timeout and logging."""
    tex_name = os.path.basename(tex_path)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{tex_name}.compile.log")

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={output_dir}",
        tex_path
    ]

    try:
        for i in range(2):  # run twice for cross-references
            subprocess.run(
                cmd,
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True
            )
        logger.info(f"✅ SUCCESS: {tex_name}")
        return "SUCCESS"

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ TIMEOUT: {tex_name} (>{timeout}s)")
        return "TIMEOUT"

    except subprocess.CalledProcessError:
        logger.error(f"❌ FAILED: {tex_name}")
        return "FAILED"

    except Exception as e:
        logger.error(f"💥 ERROR compiling {tex_name}: {str(e)}")
        return "ERROR"

# ================================================================
# Folder Processing
# ================================================================
def process_folder(folder_path, timeout, logger):
    """Compile all .tex files inside a directory."""
    compiled_path = folder_path.replace("final_merged", "final_merged_compiled")
    os.makedirs(compiled_path, exist_ok=True)
    
    # Add /final_merged subdirectory if it exists
    check_path = os.path.join(folder_path, "final_merged")
    if os.path.exists(check_path):
        folder_path = check_path

    tex_files = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if f.endswith(".tex"):
                tex_files.append(os.path.join(root, f))

    if not tex_files:
        logger.warning(f"⚠️ No .tex files found in {folder_path}")
        return

    logger.info(f"📁 Found {len(tex_files)} .tex files in {folder_path}")

    results = {}
    for tex_path in tqdm(tex_files, desc=f"Compiling {os.path.basename(folder_path)}"):
        result = compile_latex_file(tex_path, compiled_path, timeout, logger)
        results[tex_path] = result

    # Summary log
    success = sum(1 for r in results.values() if r == "SUCCESS")
    failed = len(results) - success
    logger.info(f"✅ {success} succeeded, ❌ {failed} failed in {folder_path}")

# ================================================================
# Batch Mode: Process all year directories
# ================================================================
def process_all_years(base_dir, workers, timeout, logger):
    """Compile all LaTeX files across all year directories."""
    year_dirs = sorted([
        os.path.join(base_dir, y)
        for y in os.listdir(base_dir)
        if y.isdigit() and os.path.isdir(os.path.join(base_dir, y))
    ])
    logger.info(f"📦 Found {len(year_dirs)} year directories to process.")

    # Process sequentially with progress bar
    for year_dir in tqdm(year_dirs, desc="Processing year directories"):
        logger.info(f"\n🔍 Processing: {year_dir}")
        process_folder(year_dir, timeout, logger)

    logger.info("🎉 All LaTeX documents processed across all years.")

# ================================================================
# CLI Interface
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Batch compile LaTeX documents.")
    parser.add_argument(
        "--mode",
        choices=["all", "single"],
        required=True,
        help="Mode: 'all' to compile all year directories, 'single' for one directory"
    )
    parser.add_argument("--dir", help="Path to directory containing .tex files (for single mode)")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR, help="Base directory path")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per document (seconds)")
    return parser.parse_args()

# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logger(f"latex_compile_{timestamp}")
    logger.info(f"🚀 Starting LaTeX compilation | Mode: {args.mode.upper()} | Log: {log_path}")

    if args.mode == "single":
        if not args.dir:
            logger.error("❌ Please provide --dir for single mode.")
            return
        logger.info(f"Processing single directory: {args.dir}")
        process_folder(args.dir, args.timeout, logger)

    elif args.mode == "all":
        logger.info(f"Processing all directories in: {args.base_dir}")
        process_all_years(args.base_dir, args.workers, args.timeout, logger)

    logger.info("🏁 Compilation process finished.")

if __name__ == "__main__":
    main()
