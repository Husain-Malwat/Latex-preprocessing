import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

# ================================================================
# Default configurations
# ================================================================
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT_FILE = "latex_token_stats.json"
LOG_DIR = "./logs"

# ================================================================
# Setup Logging
# ================================================================
def setup_logger(log_name: str) -> Tuple[logging.Logger, str]:
    """Create and configure a dedicated logger for token counting."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Console handler
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
# Token Counting Functions
# ================================================================
def count_tokens_in_file(filepath: str, tokenizer, logger: logging.Logger) -> int:
    """Read a file and return its token count."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logger.error(f"⚠️ Error processing {filepath}: {e}")
        return 0

def compute_statistics(stats: List[Dict]) -> Dict:
    """Compute summary statistics from token counts."""
    if not stats:
        return {
            "total_files": 0,
            "total_tokens": 0,
            "average_tokens_per_file": 0,
            "max_tokens": 0,
            "min_tokens": 0,
            "median_tokens": 0,
        }
    
    token_counts = [s["tokens"] for s in stats]
    total_tokens = sum(token_counts)
    total_files = len(stats)
    
    sorted_tokens = sorted(token_counts)
    median_idx = total_files // 2
    median_tokens = sorted_tokens[median_idx] if total_files > 0 else 0
    
    return {
        "total_files": total_files,
        "total_tokens": total_tokens,
        "average_tokens_per_file": total_tokens / total_files,
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "median_tokens": median_tokens,
    }

def process_directory(
    latex_dir: str,
    tokenizer,
    logger: logging.Logger,
    recursive: bool = True
) -> List[Dict]:
    """Process all .tex files in a directory and return token statistics."""
    stats = []
    
    logger.info(f"🔍 Scanning directory: {latex_dir}")
    
    if recursive:
        tex_files = []
        for root, _, files in os.walk(latex_dir):
            for file in files:
                if file.endswith(".tex"):
                    tex_files.append(os.path.join(root, file))
    else:
        tex_files = [
            os.path.join(latex_dir, f)
            for f in os.listdir(latex_dir)
            if f.endswith(".tex")
        ]
    
    logger.info(f"📁 Found {len(tex_files)} .tex files")
    
    for filepath in tqdm(tex_files, desc="Counting tokens"):
        token_count = count_tokens_in_file(filepath, tokenizer, logger)
        stats.append({
            "file": filepath,
            "filename": os.path.basename(filepath),
            "tokens": token_count
        })
    
    return stats

def save_statistics(
    stats: List[Dict],
    summary: Dict,
    output_file: str,
    model_name: str,
    logger: logging.Logger
):
    """Save detailed statistics and summary to JSON file."""
    result = {
        "metadata": {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_files_processed": len(stats)
        },
        "summary": summary,
        "details": stats
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✅ Saved detailed stats to: {output_file}")

def print_summary(summary: Dict, logger: logging.Logger):
    """Print summary statistics in a formatted way."""
    logger.info("\n" + "="*60)
    logger.info("📊 TOKEN COUNT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Files:     {summary['total_files']:,}")
    logger.info(f"Total Tokens:    {summary['total_tokens']:,}")
    logger.info(f"Average Tokens:  {summary['average_tokens_per_file']:,.2f}")
    logger.info(f"Median Tokens:   {summary['median_tokens']:,}")
    logger.info(f"Max Tokens:      {summary['max_tokens']:,}")
    logger.info(f"Min Tokens:      {summary['min_tokens']:,}")
    logger.info("="*60)

def find_outliers(stats: List[Dict], threshold_percentile: float = 95) -> Dict:
    """Find files with unusually high token counts."""
    if not stats:
        return {"threshold": 0, "outliers": []}
    
    token_counts = sorted([s["tokens"] for s in stats])
    threshold_idx = int(len(token_counts) * threshold_percentile / 100)
    threshold = token_counts[threshold_idx]
    
    outliers = [s for s in stats if s["tokens"] > threshold]
    outliers.sort(key=lambda x: x["tokens"], reverse=True)
    
    return {
        "threshold": threshold,
        "percentile": threshold_percentile,
        "outliers": outliers[:20]  # Top 20 outliers
    }

# ================================================================
# CLI Interface
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Count tokens in LaTeX files using transformers tokenizer"
    )
    
    # Required arguments
    parser.add_argument(
        "latex_dir",
        type=str,
        help="Path to directory containing .tex files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name for tokenizer (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file for statistics (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively"
    )
    parser.add_argument(
        "--find-outliers",
        action="store_true",
        help="Identify files with unusually high token counts"
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=95,
        help="Percentile threshold for outlier detection (default: 95)"
    )
    
    return parser.parse_args()

# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logger(f"token_count_{timestamp}")
    
    logger.info("🚀 Starting token counting analysis")
    logger.info(f"📝 Log file: {log_path}")
    logger.info(f"🤖 Model: {args.model_name}")
    logger.info(f"📂 Input directory: {args.latex_dir}")
    
    # Validate input directory
    if not os.path.exists(args.latex_dir):
        logger.error(f"❌ Directory not found: {args.latex_dir}")
        return 1
    
    # Load tokenizer
    logger.info("⏳ Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        return 1
    
    # Process files
    stats = process_directory(
        args.latex_dir,
        tokenizer,
        logger,
        recursive=not args.no_recursive
    )
    
    if not stats:
        logger.warning("⚠️ No .tex files found or processed")
        return 1
    
    # Compute summary statistics
    summary = compute_statistics(stats)
    print_summary(summary, logger)
    
    # Find outliers if requested
    if args.find_outliers:
        logger.info("\n🔍 Analyzing outliers...")
        outlier_info = find_outliers(stats, args.outlier_percentile)
        logger.info(f"📊 Outlier threshold ({outlier_info['percentile']}th percentile): {outlier_info['threshold']:,} tokens")
        logger.info(f"📊 Found {len(outlier_info['outliers'])} outlier files")
        
        if outlier_info['outliers']:
            logger.info("\n📋 Top outliers:")
            for i, outlier in enumerate(outlier_info['outliers'][:10], 1):
                logger.info(f"  {i}. {outlier['filename']}: {outlier['tokens']:,} tokens")
    
    # Save results
    save_statistics(stats, summary, args.output, args.model_name, logger)
    
    logger.info("\n🏁 Token counting completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())