# LLM-Based LaTeX Preprocessing

This guide explains how to use the LLM-powered LaTeX preprocessor to clean and standardize LaTeX files using **Google Gemini** or **vLLM** (for local models like Qwen, Llama, etc.).

## Features

- **Dual Backend Support**: Choose between Google Gemini API or local vLLM server
- **Automatic Retry Logic**: Handles transient API failures with exponential backoff
- **Comprehensive Logging**: Tracks all processing steps, errors, and metrics
- **Token Usage Tracking**: Monitors input/output tokens for cost analysis
- **Batch Processing**: Process entire directories of LaTeX files
- **Customizable Prompts**: Use custom prompt templates for specific preprocessing needs
- **Statistics Export**: Saves processing metrics to JSONL and CSV formats

## Prerequisites

### For Gemini Backend

You need a Google AI API key. Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### For vLLM Backend

1. Install vLLM:
```bash
pip install vllm
```

2. Start a vLLM server:
```bash
# Example with Qwen model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --max-model-len 32768

# Example with Llama model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

## Creating a Prompt Template

Create a file named `prompt_template.md` with your preprocessing instructions:

```markdown
You are a LaTeX preprocessing assistant. Your task is to clean and standardize LaTeX code.

Please perform the following operations:

1. Remove all comments (% lines and \iffalse...\fi blocks)
2. Replace all figure environments with a standard placeholder
3. Extract and remove the bibliography section
4. Extract only the content between \begin{document} and \end{document}
5. Normalize whitespace (remove multiple consecutive newlines)

Important guidelines:
- Preserve all mathematical equations and formulas
- Keep section structure intact
- Maintain citation references
- Do not modify the actual content or meaning
- Return ONLY the processed LaTeX code in a ```latex code block
```

## Usage

### Using Gemini Backend

#### Process a Single File

```bash
python run_llm_preprocessor.py \
    input.tex \
    output.tex \
    prompt_template.md \
    --backend gemini \
    --api-key "YOUR_GEMINI_API_KEY" \
    --model gemini-2.5-pro
```

#### Process a Directory (Batch Processing)

```bash
python run_llm_preprocessor.py \
    input_folder \
    output_folder \
    prompt_template.md \
    --backend gemini \
    --api-key "YOUR_GEMINI_API_KEY" \
    --model gemini-2.5-pro
```

### Using vLLM Backend

First, start the vLLM server (in a separate terminal):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --max-model-len 32768
```

Then run the preprocessor:

#### Process a Single File

```bash
python run_llm_preprocessor.py \
    input.tex \
    output.tex \
    prompt_template.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --vllm-url http://localhost:8000/v1
```

#### Process a Directory (Batch Processing)

```bash
python run_llm_preprocessor.py \
    input_folder \
    output_folder \
    prompt_template.md \
    --backend vllm \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --vllm-url http://localhost:8000/v1 \
    --max-tokens 16384 \
    --temperature 0.6
```

### Command-Line Arguments

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input file or directory containing .tex files |
| `output` | Output file or directory for processed files |
| `prompt_template` | Path to prompt template file (.md or .txt) |

#### Backend Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `gemini` | LLM backend: `gemini` or `vllm` |

#### API/Connection Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-key` | None | Google AI API key (required for Gemini) |
| `--vllm-url` | `http://localhost:8000/v1` | Base URL for vLLM server |

#### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini-2.5-pro` | Model name (e.g., `gemini-2.5-pro`, `Qwen/Qwen2.5-14B-Instruct`) |
| `--max-tokens` | `16384` | Maximum tokens to generate |
| `--temperature` | `0.3` | Sampling temperature (0.0-1.0) |

#### Retry and Timeout Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-retries` | `3` | Maximum retry attempts per file |
| `--timeout` | `300` | Timeout in seconds for each API call |

#### Statistics and Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--stats-file` | `processing_stats.jsonl` | JSONL file for statistics |
| `--csv-stats-file` | `processing_stats.csv` | CSV file for statistics |
| `--save-interval` | `5` | Save stats after every N files |
| `--save-raw-responses` | `False` | Save raw LLM responses |
| `--raw-responses-dir` | `raw_llm_responses` | Directory for raw responses |

## Advanced Examples

### High-Quality Processing with Gemini

```bash
python run_llm_preprocessor.py \
    papers/ \
    processed_papers/ \
    prompt_template.md \
    --backend gemini \
    --api-key "YOUR_API_KEY" \
    --model gemini-2.5-pro \
    --temperature 0.2 \
    --max-retries 5 \
    --save-raw-responses
```

### Fast Local Processing with vLLM

```bash
# Start vLLM with larger batch size
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9

# Run preprocessor
python run_llm_preprocessor.py \
    papers/ \
    processed_papers/ \
    prompt_template.md \
    --backend vllm \
    --model Qwen/Qwen2.5-14B-Instruct \
    --max-tokens 20000 \
    --temperature 0.1 \
    --timeout 600
```

### Custom vLLM Server URL

```bash
python run_llm_preprocessor.py \
    input_folder \
    output_folder \
    prompt_template.md \
    --backend vllm \
    --model Qwen/Qwen2.5-14B-Instruct \
    --vllm-url http://192.168.1.100:8000/v1
```

## Output Files

### Processed LaTeX Files
- Saved to the specified output directory
- Same filename as input

### Statistics Files

1. **`processing_stats.jsonl`** - Detailed per-file metrics in JSON Lines format
2. **`processing_stats.csv`** - Tabular summary with columns:
   - `file_id`: Filename without extension
   - `timestamp`: Processing timestamp
   - `status`: success, failed, or partial
   - `input_tokens`: Number of input tokens
   - `output_tokens`: Number of output tokens
   - `processing_time`: Time in seconds
   - `error_message`: Error details if failed
   - `retry_count`: Number of retries needed
   - `model_name`: Model used
   - `llm_backend`: Backend used (gemini or vllm)

### Raw Responses (if `--save-raw-responses` enabled)
- Saved in `raw_llm_responses/` directory
- Contains unprocessed LLM outputs for debugging

## Comparison: Gemini vs vLLM

| Feature | Gemini | vLLM |
|---------|--------|------|
| **Setup** | API key only | Requires local GPU + server setup |
| **Cost** | Pay per token | Free (after hardware investment) |
| **Speed** | Fast, cloud-based | Very fast with good GPU |
| **Privacy** | Data sent to Google | All data stays local |
| **Model Options** | Gemini models only | Any supported open-source model |
| **Scalability** | Limited by API rate limits | Limited by local hardware |
| **Best For** | Quick prototyping, small batches | Large-scale processing, privacy-sensitive data |

## Troubleshooting

### Gemini Backend Issues

**"API key is required"**
```bash
# Make sure to provide --api-key
python run_llm_preprocessor.py ... --backend gemini --api-key "YOUR_KEY"
```

**Rate Limiting**
- Increase delay between requests (code has 2s default)
- Reduce batch size
- Use a higher tier API key

### vLLM Backend Issues

**"Could not connect to vLLM server"**
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Start the server
python -m vllm.entrypoints.openai.api_server --model YOUR_MODEL --port 8000
```

**Out of Memory Errors**
```bash
# Reduce max-model-len
python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8
```

**Timeout Errors**
```bash
# Increase timeout for longer documents
python run_llm_preprocessor.py ... --timeout 600
```

## Tips for Best Results

1. **Prompt Engineering**: Customize `prompt_template.md` for your specific needs
2. **Temperature**: Use lower values (0.1-0.3) for more consistent preprocessing
3. **Token Limits**: Adjust `--max-tokens` based on your document sizes
4. **Monitoring**: Check `processing_stats.csv` to identify problematic files
5. **Raw Responses**: Enable `--save-raw-responses` during development to debug issues

## Support

For issues or questions:
1. Check the log file: `llm_preprocessing.log`
2. Review statistics files for patterns in failures
3. Examine raw responses if enabled
4. Verify your prompt template is clear and specific
