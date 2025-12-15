import os
import re
import json
import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Literal
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


class LLMPreprocessor:
    """Handles LLM-based LaTeX preprocessing with retry logic and comprehensive logging."""
    
    def __init__(
        self,
        model_name: str,
        llm_backend: Literal["gemini", "vllm"] = "gemini",
        api_key: Optional[str] = None,
        vllm_base_url: str = "http://localhost:8000/v1",
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
            model_name: Model to use for generation (e.g., "gemini-2.0-flash-exp" or "Qwen/Qwen3-14B-FP8")
            llm_backend: Which backend to use - "gemini" or "vllm"
            api_key: Google AI API key (required for Gemini, not used for vLLM)
            vllm_base_url: Base URL for vLLM server (default: http://localhost:8000/v1)
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
            self.vllm_client = None
            logger.info(f"Initialized Gemini backend with model: {model_name}")
            
        elif self.llm_backend == "vllm":
            self.model = None
            self.vllm_client = OpenAI(
                base_url=vllm_base_url,
                api_key="EMPTY"  # vLLM doesn't require API key
            )
            logger.info(f"Initialized vLLM backend at {vllm_base_url} with model: {model_name}")
            
            # Test connection
            try:
                models = self.vllm_client.models.list()
                logger.info(f"✓ Connected to vLLM server. Available models: {[m.id for m in models.data]}")
            except Exception as e:
                logger.warning(f"Could not connect to vLLM server: {e}")
        else:
            raise ValueError(f"Invalid llm_backend: {llm_backend}. Must be 'gemini' or 'vllm'")
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_stats_file):
            with open(self.csv_stats_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file_id', 'timestamp', 'status', 'input_tokens',
                    'output_tokens', 'processing_time', 'error_message',
                    'retry_count', 'model_name', 'llm_backend'
                ])
                writer.writeheader()
    
    def _save_stats(self, result: ProcessingResult):
        """Save processing statistics to both JSONL and CSV files."""
        # Save to JSONL
        with open(self.stats_file, 'a') as f:
            json.dump(asdict(result), f)
            f.write('\n')
        
        # Save to CSV
        with open(self.csv_stats_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file_id', 'timestamp', 'status', 'input_tokens',
                'output_tokens', 'processing_time', 'error_message',
                'retry_count', 'model_name', 'llm_backend'
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
        
        # Call the API
        response = self.vllm_client.chat.completions.create(
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

        logger.info(f"vLLM call successful for {file_id}. Tokens - Input: {input_tokens}, Output: {output_tokens}")
        
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
        tex_files = sorted(list(input_dir.glob("*.tex")))
        
        if not tex_files:
            logger.warning(f"No .tex files found in {input_dir}")
            return {"total": 0, "success": 0, "failed": 0, "partial": 0}
        
        logger.info(f"Found {len(tex_files)} LaTeX files to process")
        
        summary = {"total": len(tex_files), "success": 0, "failed": 0, "partial": 0}
        
        for idx, tex_file in enumerate(tex_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {idx}/{len(tex_files)}: {tex_file.name}")
            logger.info(f"{'='*60}")
            
            output_path = output_dir / tex_file.name
            result = self.preprocess_file(tex_file, output_path, prompt_template_path)
            
            summary[result.status] += 1
            
            # Log progress
            logger.info(f"Progress: {idx}/{len(tex_files)} | Success: {summary['success']} | Failed: {summary['failed']} | Partial: {summary['partial']}")
            
            # Add delay between requests to avoid rate limiting
            if idx < len(tex_files):
                time.sleep(2)
        
        # Print final summary
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total files: {summary['total']}")
        logger.info(f"✓ Success: {summary['success']}")
        logger.info(f"⚠ Partial: {summary['partial']}")
        logger.info(f"✗ Failed: {summary['failed']}")
        logger.info(f"{'='*60}\n")
        
        return summary