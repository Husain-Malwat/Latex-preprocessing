import os
import asyncio
import re
import json
import time
import csv
import random
from pathlib import Path
from functools import partial
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.generativeai as genai
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Data class to store processing results and metrics."""
    file_id: str
    timestamp: str
    status: str  # success, failed, partial
    input_tokens: int
    output_tokens: int
    processing_time: float
    error_message: Optional[str] = None
    retry_count: int = 0
    model_name: str = "gemini-2.0-flash-exp"
    llm_backend: str = "gemini"  # gemini or vllm
    server_endpoint: Optional[str] = None  # Track which server handled the request


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    total_files: int
    concurrency: int
    total_time: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    input_tps: float
    output_tps: float
    total_tps: float
    avg_time_per_file: float
    files_per_second: float
    success_count: int
    failed_count: int
    partial_count: int
    timestamp: str
    backend: str
    model_name: str
    num_servers: int = 1  # Track number of vLLM servers


class LLMPreprocessor:
    """Handles LLM-based LaTeX preprocessing with retry logic and comprehensive logging."""
    
    def __init__(
        self,
        model_name: str,
        llm_backend: Literal["gemini", "vllm"] = "gemini",
        api_key: Optional[str] = None,
        vllm_base_url: str = "http://localhost:8000/v1",
        vllm_endpoints: Optional[List[str]] = None,  # NEW: Support multiple endpoints
        load_balancing: Literal["random", "round-robin"] = "random",  # NEW: Load balancing strategy
        max_retries: int = 3,
        timeout: int = 300,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        stats_file: str = "processing_stats.jsonl",
        csv_stats_file: str = "processing_stats.csv",
        save_raw_responses: bool = False,
        raw_responses_dir: Path = Path("raw_llm_responses")
    ):
        """
        Initialize the LLM preprocessor.
        
        Args:
            model_name: Model to use for generation
            llm_backend: Which backend to use - "gemini" or "vllm"
            api_key: Google AI API key (required for Gemini)
            vllm_base_url: Base URL for single vLLM server (deprecated, use vllm_endpoints)
            vllm_endpoints: List of vLLM server endpoints for multi-GPU setup
            load_balancing: Strategy for distributing requests: "random" or "round-robin"
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for each API call
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stats_file: JSONL file to save processing statistics
            csv_stats_file: CSV file to save processing statistics
            save_raw_responses: Whether to save raw LLM responses
            raw_responses_dir: Directory to save raw LLM responses
        """
        self.model_name = model_name
        self.llm_backend = llm_backend
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stats_file = stats_file
        self.csv_stats_file = csv_stats_file
        self.save_raw_responses = save_raw_responses
        self.raw_responses_dir = Path(raw_responses_dir)
        self.load_balancing = load_balancing
        self.round_robin_index = 0  # For round-robin load balancing
        
        # Create raw responses directory if needed
        if self.save_raw_responses:
            self.raw_responses_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Raw responses will be saved to: {self.raw_responses_dir}")
        
        # Initialize the appropriate backend
        if self.llm_backend == "gemini":
            if not api_key:
                raise ValueError("api_key is required for Gemini backend")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.vllm_clients = []
            self.vllm_endpoints = []
            logger.info(f"Initialized Gemini backend with model: {model_name}")
            
        elif self.llm_backend == "vllm":
            self.model = None
            
            # Setup multi-GPU endpoints
            if vllm_endpoints:
                self.vllm_endpoints = vllm_endpoints
            else:
                # Fallback to single endpoint for backward compatibility
                self.vllm_endpoints = [vllm_base_url]
            
            # Create a client for each endpoint
            self.vllm_clients = []
            for endpoint in self.vllm_endpoints:
                client = OpenAI(
                    base_url=endpoint,
                    api_key="EMPTY"  # vLLM doesn't require API key
                )
                self.vllm_clients.append(client)
            
            logger.info(f"Initialized vLLM backend with {len(self.vllm_endpoints)} server(s)")
            logger.info(f"Load balancing strategy: {load_balancing}")
            for idx, endpoint in enumerate(self.vllm_endpoints):
                logger.info(f"  Server {idx}: {endpoint}")
            
            # Test connections
            for idx, (client, endpoint) in enumerate(zip(self.vllm_clients, self.vllm_endpoints)):
                try:
                    models = client.models.list()
                    logger.info(f"✓ Server {idx} ({endpoint}): Connected. Available models: {[m.id for m in models.data]}")
                except Exception as e:
                    logger.warning(f"✗ Server {idx} ({endpoint}): Connection failed - {e}")
        else:
            raise ValueError(f"Invalid llm_backend: {llm_backend}. Must be 'gemini' or 'vllm'")
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _get_vllm_client(self) -> tuple[OpenAI, str]:
        """
        Get a vLLM client using the configured load balancing strategy.
        
        Returns:
            Tuple of (client, endpoint_url)
        """
        if not self.vllm_clients:
            raise RuntimeError("No vLLM clients initialized")
        
        if len(self.vllm_clients) == 1:
            return self.vllm_clients[0], self.vllm_endpoints[0]
        
        if self.load_balancing == "random":
            idx = random.randint(0, len(self.vllm_clients) - 1)
        else:  # round-robin
            idx = self.round_robin_index % len(self.vllm_clients)
            self.round_robin_index += 1
        
        return self.vllm_clients[idx], self.vllm_endpoints[idx]
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_stats_file):
            with open(self.csv_stats_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file_id', 'timestamp', 'status', 'input_tokens', 'output_tokens',
                    'processing_time', 'error_message', 'retry_count', 'model_name', 
                    'llm_backend', 'server_endpoint'
                ])
                writer.writeheader()
            logger.info(f"Initialized CSV stats file: {self.csv_stats_file}")
    
    def _save_stats(self, result: ProcessingResult):
        """Save processing statistics to both JSONL and CSV files."""
        # Save to JSONL
        with open(self.stats_file, 'a') as f:
            json.dump(asdict(result), f)
            f.write('\n')
        
        # Save to CSV
        with open(self.csv_stats_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file_id', 'timestamp', 'status', 'input_tokens', 'output_tokens',
                'processing_time', 'error_message', 'retry_count', 'model_name', 
                'llm_backend', 'server_endpoint'
            ])
            writer.writerow(asdict(result))
        
        logger.debug(f"Saved statistics for {result.file_id}")
    
    def _call_gemini(self, prompt: str, file_id: str) -> tuple[str, int, int]:
        """Call Gemini API."""
        logger.info(f"Calling Gemini API for {file_id}")
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=self.max_tokens,
            candidate_count=1,
        )
        
        # Call the API
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={'timeout': self.timeout}
        )
        
        # Extract the generated text
        generated_text = response.text
        
        # Get token counts from usage metadata
        input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        
        logger.info(f"Gemini call successful for {file_id}. Tokens - Input: {input_tokens}, Output: {output_tokens}")
        
        return generated_text, input_tokens, output_tokens
    
    def _call_vllm(self, prompt: str, file_id: str, system_prompt: str = "") -> tuple[str, int, int]:
        """Call vLLM API."""
        logger.info(f"Calling vLLM API for {file_id}")
        
        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt if system_prompt else "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Get vLLM client
        client, endpoint = self._get_vllm_client()
        
        # Call the API
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
        
        # Extract the generated text
        generated_text = response.choices[0].message.content
        
        # Get token counts (vLLM provides these in usage field)
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0

        logger.info(f"vLLM call successful for {file_id} (via {endpoint}). Tokens - Input: {input_tokens}, Output: {output_tokens}")
        
        return generated_text, input_tokens, output_tokens
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _call_llm_with_retry(
        self,
        prompt: str,
        file_id: str,
        retry_count: int = 0,
        system_prompt: str = ""
    ) -> tuple[str, int, int]:
        """
        Call LLM with automatic retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            file_id: Identifier for the file being processed
            retry_count: Current retry attempt number
            system_prompt: System prompt (used for vLLM)
            
        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        try:
            logger.info(f"Calling LLM for {file_id} (attempt {retry_count + 1}/{self.max_retries}) using {self.llm_backend}")
            
            if self.llm_backend == "gemini":
                return self._call_gemini(prompt, file_id)
            elif self.llm_backend == "vllm":
                return self._call_vllm(prompt, file_id, system_prompt)
            else:
                raise ValueError(f"Unknown backend: {self.llm_backend}")
            
        except Exception as e:
            logger.error(f"LLM call failed for {file_id} (attempt {retry_count + 1}): {str(e)}")
            raise
    
    def extract_latex_from_response(self, llm_response: str) -> Optional[str]:
        """
        Extract LaTeX code from markdown-formatted LLM response.
        
        Args:
            llm_response: The raw LLM response
            
        Returns:
            Extracted LaTeX code or None if not found
        """
        # Try to find latex code block
        match = re.search(r"```latex\s*(.*?)\s*```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: try generic code block
        match = re.search(r"```\s*(.*?)\s*```", llm_response, re.DOTALL)
        if match:
            logger.warning("Found generic code block instead of latex-specific block")
            return match.group(1).strip()
        
        logger.error("No code block found in LLM response")
        return None
    
    def _save_raw_response(self, file_id: str, response_text: str, attempt: int = 0):
        """
        Save raw LLM response to file.
        
        Args:
            file_id: Identifier for the file
            response_text: Raw response from LLM
            attempt: Attempt number (for retries)
        """
        if not self.save_raw_responses:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_id}_{timestamp}"
        if attempt > 0:
            filename += f"_retry{attempt}"
        filename += ".txt"
        
        raw_file_path = self.raw_responses_dir / filename
        
        try:
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                f.write(f"File ID: {file_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Attempt: {attempt + 1}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Backend: {self.llm_backend}\n")
                f.write("="*80 + "\n\n")
                f.write(response_text)
            
            logger.debug(f"Saved raw response to: {raw_file_path}")
        except Exception as e:
            logger.error(f"Failed to save raw response for {file_id}: {str(e)}")
    
    def preprocess_file(
        self,
        input_path: Path,
        output_path: Path,
        prompt_template_path: Path
    ) -> ProcessingResult:
        """
        Preprocess a single LaTeX file using LLM.
        
        Args:
            input_path: Path to input LaTeX file
            output_path: Path to save processed file
            prompt_template_path: Path to prompt template
            
        Returns:
            ProcessingResult with metrics and status
        """
        file_id = input_path.stem
        start_time = time.time()
        retry_count = 0
        
        result = ProcessingResult(
            file_id=file_id,
            timestamp=datetime.now().isoformat(),
            status="failed",
            input_tokens=0,
            output_tokens=0,
            processing_time=0.0,
            model_name=self.model_name,
            llm_backend=self.llm_backend
        )
        
        try:
            # Read input file
            logger.info(f"Processing file: {input_path.name}")
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_code = f.read()
            
            # Read prompt template
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            # Construct full prompt
            prompt = prompt_template + f"""

Here is the LaTeX code to process:
```latex
{original_code}
```

Your response MUST contain only the final, preprocessed LaTeX code, enclosed in a single markdown block like this:
```latex
... your generated code here ...
```
"""
            
            # Call LLM with retry
            generated_text = None
            for attempt in range(self.max_retries):
                try:
                    retry_count = attempt
                    generated_text, input_tokens, output_tokens = self._call_llm_with_retry(
                        prompt, file_id, retry_count, system_prompt=prompt_template
                    )
                    
                    # Save raw response
                    self._save_raw_response(file_id, generated_text, attempt)
                    
                    result.input_tokens = input_tokens
                    result.output_tokens = output_tokens
                    result.retry_count = retry_count
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {file_id}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Extract LaTeX code
            if generated_text:
                processed_code = self.extract_latex_from_response(generated_text)
                
                if processed_code:
                    # Save output file
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(processed_code)
                    
                    result.status = "success"
                    logger.info(f"Successfully processed {file_id} -> {output_path}")
                else:
                    result.status = "partial"
                    result.error_message = "Failed to extract LaTeX code from response"
                    logger.error(f"Failed to extract LaTeX code for {file_id}")
                    
                    # Save debug response
                    debug_path = output_path.parent / f"{file_id}_debug_response.txt"
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        f.write(generated_text)
                    logger.info(f"Saved debug response to {debug_path}")
        
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error(f"Error processing {file_id}: {str(e)}", exc_info=True)
        
        finally:
            result.processing_time = time.time() - start_time
            self._save_stats(result)
        
        return result
    
    def preprocess_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        prompt_template_path: Path,
        save_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Preprocess all LaTeX files in a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save processed files
            prompt_template_path: Path to prompt template
            save_interval: Save stats after every N files
            
        Returns:
            Dictionary with summary statistics
        """
        # Get all tex files from input directory
        all_tex_files = sorted(list(input_dir.glob("*.tex")))
        
        if not all_tex_files:
            logger.warning(f"No .tex files found in {input_dir}")
            return {"total": 0, "success": 0, "failed": 0, "partial": 0}
        
        # Get already normalized files from output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_files = sorted(list(output_dir.glob("*.tex")))
        normalized_stems = {f.stem for f in normalized_files}
        
        # Filter out already processed files
        tex_files = [f for f in all_tex_files if f.stem not in normalized_stems]
        
        # Detailed logging
        logger.info(f"\n{'='*80}")
        logger.info("📁 FILE PROCESSING STATUS")
        logger.info(f"{'='*80}")
        logger.info(f"Total files in input directory: {len(all_tex_files)}")
        logger.info(f"Already normalized files in output directory: {len(normalized_files)}")
        logger.info(f"Files yet to process: {len(tex_files)}")
        logger.info(f"{'='*80}\n")
        
        if normalized_files:
            logger.info(f"✓ Previously normalized files ({len(normalized_files)}):")
            for nf in normalized_files[:10]:  # Show first 10
                logger.info(f"  - {nf.name}")
            if len(normalized_files) > 10:
                logger.info(f"  ... and {len(normalized_files) - 10} more")
            logger.info("")
        
        if not tex_files:
            logger.info("✅ All files have already been normalized!")
            return {
                "total": len(all_tex_files),
                "success": len(normalized_files),
                "failed": 0,
                "partial": 0,
                "already_processed": len(normalized_files),
                "newly_processed": 0
            }
        
        logger.info(f"🔄 Files to process in this run ({len(tex_files)}):")
        for tf in tex_files[:10]:  # Show first 10
            logger.info(f"  - {tf.name}")
        if len(tex_files) > 10:
            logger.info(f"  ... and {len(tex_files) - 10} more")
        logger.info("")
        
        input("press enter to continue...")
        # Start timing
        start_time = time.time()
        
        summary = {"total": len(tex_files), "success": 0, "failed": 0, "partial": 0}
        total_input_tokens = 0
        total_output_tokens = 0
        
        for idx, tex_file in enumerate(tex_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {idx}/{len(tex_files)}: {tex_file.name}")
            logger.info(f"{'='*60}")
            
            output_path = output_dir / tex_file.name
            result = self.preprocess_file(tex_file, output_path, prompt_template_path)
            
            summary[result.status] += 1
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            
            # Log progress
            logger.info(f"Progress: {idx}/{len(tex_files)} | Success: {summary['success']} | Failed: {summary['failed']} | Partial: {summary['partial']}")
            
            # Add delay between requests to avoid rate limiting
            if idx < len(tex_files):
                time.sleep(2)
    
        # End timing
        total_time = time.time() - start_time
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate metrics
        input_tps = total_input_tokens / total_time if total_time > 0 else 0
        output_tps = total_output_tokens / total_time if total_time > 0 else 0
        total_tps = total_tokens / total_time if total_time > 0 else 0
        avg_time_per_file = total_time / len(tex_files) if len(tex_files) > 0 else 0
        files_per_second = len(tex_files) / total_time if total_time > 0 else 0
        
        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            total_files=len(tex_files),
            concurrency=1,  # Sequential processing
            total_time=total_time,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            input_tps=input_tps,
            output_tps=output_tps,
            total_tps=total_tps,
            avg_time_per_file=avg_time_per_file,
            files_per_second=files_per_second,
            success_count=summary["success"],
            failed_count=summary["failed"],
            partial_count=summary["partial"],
            timestamp=datetime.now().isoformat(),
            backend=self.llm_backend,
            model_name=self.model_name
        )
        
        # Save performance metrics
        self._save_performance_metrics(perf_metrics)
        
        # Print final summary with metrics
        logger.info(f"\n{'='*80}")
        logger.info("📊 PROCESSING COMPLETE - PERFORMANCE METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"Backend: {self.llm_backend}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total files in input dir: {len(all_tex_files)}")
        logger.info(f"Already normalized: {len(normalized_files)}")
        logger.info(f"Newly processed: {len(tex_files)}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Avg Time/File: {avg_time_per_file:.2f}s")
        logger.info(f"")
        logger.info(f"Total TPS: {total_tps:.2f} tokens/sec")
        logger.info(f"Input TPS: {input_tps:.2f} tokens/sec")
        logger.info(f"Output TPS: {output_tps:.2f} tokens/sec")
        logger.info(f"")
        logger.info(f"✓ Success: {summary['success']}")
        logger.info(f"⚠ Partial: {summary['partial']}")
        logger.info(f"✗ Failed: {summary['failed']}")
        logger.info(f"{'='*80}\n")
        
        summary['performance'] = asdict(perf_metrics)
        summary['already_processed'] = len(normalized_files)
        summary['newly_processed'] = len(tex_files)
        summary['total_in_input_dir'] = len(all_tex_files)
        
        return summary

    async def _process_file_async(self, semaphore: "asyncio.Semaphore", input_path: Path, output_path: Path, prompt_template_path: Path) -> ProcessingResult:
        """Run preprocess_file concurrently with semaphore control in a thread pool."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                partial(self.preprocess_file, input_path, output_path, prompt_template_path)
            )

    async def preprocess_directory_async(
        self,
        input_dir: Path,
        output_dir: Path,
        prompt_template_path: Path,
        concurrency: int = 4
    ) -> Dict[str, Any]:
        """
        Concurrently preprocess LaTeX files in a directory using async tasks.

        Args:
            input_dir: Directory with input .tex files
            output_dir: Directory to save preprocessed files
            prompt_template_path: Path to the prompt template
            concurrency: Number of concurrent requests

        Returns:
            Summary dictionary with performance metrics
        """
        # Get all tex files from input directory
        all_tex_files = sorted(list(input_dir.glob("*.tex")))
        
        if not all_tex_files:
            logger.warning(f"No .tex files found in {input_dir}")
            return {"total": 0, "success": 0, "failed": 0, "partial": 0}
        
        # Get already normalized files from output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_files = sorted(list(output_dir.glob("*.tex")))
        normalized_stems = {f.stem for f in normalized_files}
        
        # Filter out already processed files
        tex_files = [f for f in all_tex_files if f.stem not in normalized_stems]
        
        # Detailed logging
        logger.info(f"\n{'='*80}")
        logger.info("📁 FILE PROCESSING STATUS")
        logger.info(f"{'='*80}")
        logger.info(f"Total files in input directory: {len(all_tex_files)}")
        logger.info(f"Already normalized files in output directory: {len(normalized_files)}")
        logger.info(f"Files yet to process: {len(tex_files)}")
        logger.info(f"{'='*80}\n")
        
        if normalized_files:
            logger.info(f"✓ Previously normalized files ({len(normalized_files)}):")
            for nf in normalized_files[:10]:  # Show first 10
                logger.info(f"  - {nf.name}")
            if len(normalized_files) > 10:
                logger.info(f"  ... and {len(normalized_files) - 10} more")
            logger.info("")
        
        if not tex_files:
            logger.info("✅ All files have already been normalized!")
            return {
                "total": len(all_tex_files),
                "success": len(normalized_files),
                "failed": 0,
                "partial": 0,
                "already_processed": len(normalized_files),
                "newly_processed": 0
            }
        
        logger.info(f"🔄 Files to process in this run ({len(tex_files)}):")
        for tf in tex_files[:10]:  # Show first 10
            logger.info(f"  - {tf.name}")
        if len(tex_files) > 10:
            logger.info(f"  ... and {len(tex_files) - 10} more")
        logger.info("")
        
        logger.info(f"🚀 Starting async preprocessing of {len(tex_files)} files with concurrency={concurrency}")
        
        # Start timing
        start_time = time.time()
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        tasks = []
        for tex_file in tex_files:
            output_path = output_dir / tex_file.name
            tasks.append(self._process_file_async(semaphore, tex_file, output_path, prompt_template_path))

        results = await asyncio.gather(*tasks)
        
        # End timing
        total_time = time.time() - start_time

        # Calculate metrics
        summary = {"total": len(results), "success": 0, "failed": 0, "partial": 0}
        total_input_tokens = 0
        total_output_tokens = 0
        
        for r in results:
            status = getattr(r, "status", "failed")
            if status in summary:
                summary[status] += 1
            else:
                summary["failed"] += 1
            
            total_input_tokens += getattr(r, "input_tokens", 0)
            total_output_tokens += getattr(r, "output_tokens", 0)
        
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate TPS metrics
        input_tps = total_input_tokens / total_time if total_time > 0 else 0
        output_tps = total_output_tokens / total_time if total_time > 0 else 0
        total_tps = total_tokens / total_time if total_time > 0 else 0
        avg_time_per_file = total_time / len(results) if len(results) > 0 else 0
        files_per_second = len(results) / total_time if total_time > 0 else 0
        
        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            total_files=len(results),
            concurrency=concurrency,
            total_time=total_time,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            input_tps=input_tps,
            output_tps=output_tps,
            total_tps=total_tps,
            avg_time_per_file=avg_time_per_file,
            files_per_second=files_per_second,
            success_count=summary["success"],
            failed_count=summary["failed"],
            partial_count=summary["partial"],
            timestamp=datetime.now().isoformat(),
            backend=self.llm_backend,
            model_name=self.model_name
        )
        
        # Save performance metrics
        self._save_performance_metrics(perf_metrics)
        
        # Log detailed metrics
        logger.info(f"\n{'='*80}")
        logger.info("📊 PERFORMANCE METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"Backend: {self.llm_backend}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Concurrency: {concurrency}")
        logger.info(f"Total files in input dir: {len(all_tex_files)}")
        logger.info(f"Already normalized: {len(normalized_files)}")
        logger.info(f"Newly processed: {len(results)}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Avg Time/File: {avg_time_per_file:.2f}s")
        logger.info(f"Files/Second: {files_per_second:.2f}")
        logger.info(f"")
        logger.info(f"Total Input Tokens: {total_input_tokens:,}")
        logger.info(f"Total Output Tokens: {total_output_tokens:,}")
        logger.info(f"Total Tokens: {total_tokens:,}")
        logger.info(f"")
        logger.info(f"Input TPS: {input_tps:.2f} tokens/sec")
        logger.info(f"Output TPS: {output_tps:.2f} tokens/sec")
        logger.info(f"Total TPS: {total_tps:.2f} tokens/sec")
        logger.info(f"")
        logger.info(f"✅ Success: {summary['success']}")
        logger.info(f"⚠️  Partial: {summary['partial']}")
        logger.info(f"❌ Failed: {summary['failed']}")
        logger.info(f"{'='*80}\n")
        
        # Add metrics to summary
        summary['performance'] = asdict(perf_metrics)
        summary['already_processed'] = len(normalized_files)
        summary['newly_processed'] = len(results)
        summary['total_in_input_dir'] = len(all_tex_files)
        
        return summary

    def _save_performance_metrics(self, metrics: PerformanceMetrics):
        """Save performance metrics to JSON and CSV files."""
        perf_json_file = "performance_metrics.jsonl"
        perf_csv_file = "performance_metrics.csv"
        
        # Save to JSONL
        with open(perf_json_file, 'a') as f:
            json.dump(asdict(metrics), f)
            f.write('\n')
        
        # Initialize or append to CSV
        csv_exists = os.path.exists(perf_csv_file)
        with open(perf_csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
            if not csv_exists:
                writer.writeheader()
            writer.writerow(asdict(metrics))
        
        logger.info(f"Saved performance metrics to {perf_json_file} and {perf_csv_file}")