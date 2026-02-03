import argparse
import asyncio
import csv
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
        "output",
        type=Path,
        help="Output directory for processed files"
    )
    parser.add_argument(
        "prompt_template",
        type=Path,
        help="Path to prompt template file (.md or .txt)"
    )
    
    # Input source (one of these must be provided)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing .tex files"
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Single input .tex file to process"
    )
    input_group.add_argument(
        "--file-list-csv",
        type=Path,
        help="CSV file containing list of .tex files to process (columns: path/file/input)"
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
        help="Base URL for single vLLM server (default: http://localhost:8000/v1). Use --vllm-endpoints for multi-GPU setup"
    )
    parser.add_argument(
        "--vllm-endpoints",
        nargs='+',
        help="List of vLLM server endpoints for multi-GPU setup (e.g., --vllm-endpoints http://localhost:8000/v1 http://localhost:8001/v1)"
    )
    parser.add_argument(
        "--load-balancing",
        choices=["random", "round-robin"],
        default="random",
        help="Load balancing strategy for multi-GPU setup: 'random' or 'round-robin' (default: random)"
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
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent files to process. If >1, uses async processing (default: 1)"
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
    if not args.prompt_template.exists():
        logger.error(f"Prompt template does not exist: {args.prompt_template}")
        return 1
    
    # Validate backend-specific requirements
    if args.backend == "gemini" and not args.api_key:
        logger.error("--api-key is required when using Gemini backend")
        return 1
    
    # Determine vLLM endpoints
    vllm_endpoints = None
    vllm_base_url = args.vllm_url
    
    if args.backend == "vllm":
        if args.vllm_endpoints:
            vllm_endpoints = args.vllm_endpoints
            logger.info(f"Multi-GPU setup detected with {len(vllm_endpoints)} endpoints:")
            for idx, endpoint in enumerate(vllm_endpoints):
                logger.info(f"  Server {idx}: {endpoint}")
            logger.info(f"Load balancing: {args.load_balancing}")
        else:
            logger.info(f"Single GPU setup: {vllm_base_url}")
    
    # Initialize preprocessor
    try:
        preprocessor = LLMPreprocessor(
            model_name=args.model,
            llm_backend=args.backend,
            api_key=args.api_key if args.backend == "gemini" else None,
            vllm_base_url=vllm_base_url,
            vllm_endpoints=vllm_endpoints,
            load_balancing=args.load_balancing,
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
    
    # Determine processing mode
    if args.file_list_csv:
        # Process files from CSV list
        if not args.file_list_csv.exists():
            logger.error(f"CSV file does not exist: {args.file_list_csv}")
            return 1
        
        logger.info(f"Processing files from CSV list: {args.file_list_csv}")
        logger.info(f"Backend: {args.backend}")
        logger.info(f"Model: {args.model}")
        
        if args.concurrency and args.concurrency > 1:
            logger.info(f"Using async concurrent processing with concurrency={args.concurrency}")
            summary = asyncio.run(
                preprocessor.preprocess_from_csv_async(
                    args.file_list_csv,
                    args.output,
                    args.prompt_template,
                    concurrency=args.concurrency
                )
            )
        else:
            summary = preprocessor.preprocess_from_csv(
                args.file_list_csv,
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
        if vllm_endpoints:
            logger.info(f"Multi-GPU: {len(vllm_endpoints)} servers")
            logger.info(f"Load balancing: {args.load_balancing}")
        logger.info(f"Total files: {summary['total']}")
        logger.info(f"✓ Success: {summary['success']}")
        logger.info(f"⚠ Partial: {summary['partial']}")
        logger.info(f"✗ Failed: {summary['failed']}")
        logger.info(f"Statistics saved to: {args.stats_file} and {args.csv_stats_file}")
        logger.info("="*60)
        
        return 0 if summary["failed"] == 0 else 1
    
    elif args.input_file:
        # Process single file
        if not args.input_file.exists():
            logger.error(f"Input file does not exist: {args.input_file}")
            return 1
        
        logger.info(f"Processing single file: {args.input_file}")
        result = preprocessor.preprocess_file(
            args.input_file,
            args.output,
            args.prompt_template
        )
        
        if result.status == "success":
            logger.info(f"✓ File processed successfully")
            logger.info(f"  Backend: {result.llm_backend}")
            logger.info(f"  Model: {result.model_name}")
            if result.server_endpoint:
                logger.info(f"  Server: {result.server_endpoint}")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")
            logger.info(f"  Input tokens: {result.input_tokens}")
            logger.info(f"  Output tokens: {result.output_tokens}")
            return 0
        else:
            logger.error(f"✗ File processing failed: {result.error_message}")
            return 1
    
    elif args.input_dir:
        # Process directory
        if not args.input_dir.exists():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            return 1
        
        logger.info(f"Processing directory: {args.input_dir}")
        logger.info(f"Backend: {args.backend}")
        logger.info(f"Model: {args.model}")
        
        if args.concurrency and args.concurrency > 1:
            logger.info(f"Using async concurrent processing with concurrency={args.concurrency}")
            summary = asyncio.run(
                preprocessor.preprocess_directory_async(
                    args.input_dir,
                    args.output,
                    args.prompt_template,
                    concurrency=args.concurrency
                )
            )
        else:
            summary = preprocessor.preprocess_directory(
                args.input_dir,
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
        if vllm_endpoints:
            logger.info(f"Multi-GPU: {len(vllm_endpoints)} servers")
            logger.info(f"Load balancing: {args.load_balancing}")
        logger.info(f"Total files: {summary['total']}")
        logger.info(f"✓ Success: {summary['success']}")
        logger.info(f"⚠ Partial: {summary['partial']}")
        logger.info(f"✗ Failed: {summary['failed']}")
        logger.info(f"Statistics saved to: {args.stats_file} and {args.csv_stats_file}")
        logger.info("="*60)
        
        return 0 if summary["failed"] == 0 else 1
    
    else:
        logger.error("No input source specified. Use --input-dir, --input-file, or --file-list-csv")
        return 1


if __name__ == "__main__":
    exit(main())