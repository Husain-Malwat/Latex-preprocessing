# LLM-Based LaTeX Preprocessing

This guide explains how to use the LLM-powered LaTeX preprocessor to clean and standardize LaTeX files using Google Gemini, and vllm models...

### API Key

You need a Google AI API key. Get one from [Google AI Studio]

### Files 

1. **`llm_preprocessor.py`** - preprocessing module
2. **`run_llm_preprocessor.py`** - CLI interface
3. **`prompt_template.md`** - Prompt template file 

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
- Return ONLY the processed LaTeX code
```

## Usage

#### Process a Single File

```bash
python run_llm_preprocessor.py \
    input.tex \
    output.tex \
    prompt_template.md \
    --api-key "YOUR_API_KEY"
```

#### Process a Directory (Batch Processing)

```bash
python run_llm_preprocessor.py \
    input_folder \
    output_folder \
    prompt_template.md \
    --api-key "YOUR_API_KEY"
```

### Command-Line Arguments

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input file or directory containing .tex files |
| `output` | Output file or directory for processed files |
| `prompt_template` | Path to prompt template file (.md or .txt) |
| `--api-key` | Google AI API key (required) |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini-2.5-pro` | Model name to use |
| `--max-retries` | `3` | Maximum retry attempts per file |
| `--timeout` | `300` | Timeout in seconds for each API call |
| `--stats-file` | `processing_stats.jsonl` | JSONL file for statistics |
| `--csv-stats-file` | `processing_stats.csv` | CSV file for statistics |
| `--save-interval` | `5` | Save stats after every N files |
| `--save-raw-responses` | `False` | Save raw LLM responses |
| `--raw-responses-dir` | `raw_llm_responses` | Directory for raw responses |
