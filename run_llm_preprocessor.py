import argparse
from pathlib import Path
from llm_preprocessor import LLMPreprocessor
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LaTeX files using LLM (Gemini or vLLM) with comprehensive logging and error handling"
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory containing .tex files"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output file or directory for processed files"
    )
    parser.add_argument(
        "prompt_template",
        type=Path,
        help="Path to prompt template file (.md or .txt)"
    )
    
    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["gemini", "vllm"],
        default="gemini",
        help="LLM backend to use: 'gemini' for Google Gemini API or 'vllm' for local vLLM server (default: gemini)"
    )
    
    # API/Connection arguments
    parser.add_argument(
        "--api-key",
        help="Google AI API key (required for Gemini backend)"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="Base URL for vLLM server (default: http://localhost:8000/v1)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name to use (default: gemini-2.5-pro for Gemini, or specify model for vLLM like 'Qwen/Qwen2.5-14B-Instruct')"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature 0.0-1.0 (default: 0.3 for more deterministic output)"
    )
    
    # Retry and timeout settings
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per file (default: 3)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each API call (default: 300)"
    )
    
    # Statistics and logging
    parser.add_argument(
        "--stats-file",
        default="processing_stats.jsonl",
        help="JSONL file to save processing statistics (default: processing_stats.jsonl)"
    )
    parser.add_argument(
        "--csv-stats-file",
        default="processing_stats.csv",
        help="CSV file to save processing statistics (default: processing_stats.csv)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save stats after every N files (default: 5)"
    )
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="Save raw LLM responses to separate directory"
    )
    parser.add_argument(
        "--raw-responses-dir",
        type=Path,
        default=Path("raw_llm_responses"),
        help="Directory to save raw LLM responses (default: raw_llm_responses)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    if not args.prompt_template.exists():
        logger.error(f"Prompt template does not exist: {args.prompt_template}")
        return 1
    
    # Validate backend-specific requirements
    if args.backend == "gemini" and not args.api_key:
        logger.error("--api-key is required when using Gemini backend")
        return 1
    
    # Initialize preprocessor
    try:
        preprocessor = LLMPreprocessor(
            model_name=args.model,
            llm_backend=args.backend,
            api_key=args.api_key if args.backend == "gemini" else None,
            vllm_base_url=args.vllm_url,
            max_retries=args.max_retries,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stats_file=args.stats_file,
            csv_stats_file=args.csv_stats_file,
            save_raw_responses=args.save_raw_responses,
            raw_responses_dir=args.raw_responses_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        return 1
    
    # Process single file or directory
    if args.input.is_file():
        logger.info(f"Processing single file: {args.input}")
        result = preprocessor.preprocess_file(
            args.input,
            args.output,
            args.prompt_template
        )
        
        if result.status == "success":
            logger.info(f"✓ File processed successfully")
            logger.info(f"  Backend: {result.llm_backend}")
            logger.info(f"  Model: {result.model_name}")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")
            logger.info(f"  Input tokens: {result.input_tokens}")
            logger.info(f"  Output tokens: {result.output_tokens}")
            return 0
        else:
            logger.error(f"✗ File processing failed: {result.error_message}")
            return 1
    
    elif args.input.is_dir():
        logger.info(f"Processing directory: {args.input}")
        logger.info(f"Backend: {args.backend}")
        logger.info(f"Model: {args.model}")
        
        summary = preprocessor.preprocess_directory(
            args.input,
            args.output,
            args.prompt_template,
            save_interval=args.save_interval
        )
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Backend: {args.backend}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Total files: {summary['total']}")
        logger.info(f"✓ Success: {summary['success']}")
        logger.info(f"⚠ Partial: {summary['partial']}")
        logger.info(f"✗ Failed: {summary['failed']}")
        logger.info(f"Statistics saved to: {args.stats_file} and {args.csv_stats_file}")
        logger.info("="*60)
        
        return 0 if summary["failed"] == 0 else 1
    
    else:
        logger.error(f"Invalid input path: {args.input}")
        return 1


if __name__ == "__main__":
    exit(main())