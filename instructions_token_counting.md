# 📊 LaTeX Token Counting - Instructions

## Overview

The `count_tokens.py` script analyzes LaTeX files to count tokens using the same tokenizer as your LLM model. This is essential for:
- **Planning vLLM deployments**: Determine `--max-model-len` requirements
- **Cost estimation**: Calculate API costs for Gemini or other paid services
- **Dataset analysis**: Understand document size distribution
- **Outlier detection**: Find unusually large documents that may need special handling

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Accurate Token Counting** | Uses the exact tokenizer from your target model (e.g., Qwen, Llama, Gemini) |
| **Batch Processing** | Process entire directories recursively |
| **Statistical Analysis** | Compute min, max, average, median token counts |
| **Outlier Detection** | Identify documents with unusually high token counts |
| **Detailed Logging** | Timestamped logs with processing statistics |
| **JSON Export** | Save results in structured JSON format for further analysis |

---

## 🚀 Usage

### Basic Token Counting

Count tokens in a single directory:

```bash
python count_tokens.py /path/to/latex/files
```

**Example:**
```bash
python count_tokens.py /home/husainmalwat/workspace/OCR_Latex/data/final_merged/2023/final_merged
```

---

### Specify Model Tokenizer

Use a specific model's tokenizer (important for accurate counts):

```bash
python count_tokens.py /path/to/latex/files \
    --model-name Qwen/Qwen2.5-Coder-14B-Instruct
```

**Common models:**
```bash
# Qwen models
--model-name Qwen/Qwen2.5-Coder-8B-Instruct
--model-name Qwen/Qwen2.5-Coder-14B-Instruct
--model-name Qwen/Qwen2.5-14B-Instruct

# Llama models
--model-name meta-llama/Llama-3.1-8B-Instruct
--model-name meta-llama/Llama-3.1-70B-Instruct

# Gemini (uses same tokenizer as Gemini API)
--model-name google/gemma-2-9b-it
```

---

### Find Outliers

Identify documents with unusually high token counts:

```bash
python count_tokens.py /path/to/latex/files \
    --find-outliers \
    --outlier-percentile 95
```

This flags documents above the 95th percentile (top 5% largest files).

---

### Custom Output File

Save results to a specific JSON file:

```bash
python count_tokens.py /path/to/latex/files \
    --output my_token_stats.json
```

---

### Non-Recursive Mode

Process only the specified directory (don't search subdirectories):

```bash
python count_tokens.py /path/to/latex/files \
    --no-recursive
```

---

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `latex_dir` | Required | - | Path to directory containing .tex files |
| `--model-name` | Optional | `Qwen/Qwen2.5-Coder-8B-Instruct` | Hugging Face model name for tokenizer |
| `--output` | Optional | `latex_token_stats.json` | Output JSON file path |
| `--no-recursive` | Flag | `False` | Don't search subdirectories |
| `--find-outliers` | Flag | `False` | Identify files with high token counts |
| `--outlier-percentile` | Optional | `95` | Percentile threshold for outliers (0-100) |

---

## 📊 Output Files

### JSON Statistics File (`latex_token_stats.json`)

```json
{
  "metadata": {
    "model_name": "Qwen/Qwen2.5-Coder-8B-Instruct",
    "timestamp": "2024-12-15T14:30:00",
    "total_files_processed": 150
  },
  "summary": {
    "total_files": 150,
    "total_tokens": 2500000,
    "average_tokens_per_file": 16666.67,
    "median_tokens": 12000,
    "max_tokens": 65000,
    "min_tokens": 500
  },
  "details": [
    {
      "file": "/path/to/document1.tex",
      "filename": "document1.tex",
      "tokens": 15000
    },
    ...
  ]
}
```

### Log File (`./logs/token_count_TIMESTAMP.log`)

Contains:
- Processing progress and statistics
- Error messages for problematic files
- Summary of token counts
- Outlier analysis (if enabled)

---

## 💡 Use Cases

### 1. Plan vLLM Server Configuration

```bash
# Count tokens to determine max-model-len
python count_tokens.py /path/to/data \
    --model-name Qwen/Qwen2.5-Coder-14B-Instruct \
    --find-outliers

# Use max_tokens from output to set vLLM parameter:
# --max-model-len <max_tokens + buffer>
```

**Example output:**
```
Max Tokens: 48,000
→ Use: --max-model-len 65536
```

---

### 2. Estimate API Costs (Gemini)

```bash
python count_tokens.py /path/to/data \
    --model-name google/gemma-2-9b-it \
    --output cost_estimate.json

# Calculate costs from total_tokens in JSON:
# Input: $0.075 per 1M tokens (Gemini 2.0 Flash)
# Output: $0.30 per 1M tokens
```

---

### 3. Analyze Dataset Distribution

```bash
python count_tokens.py /path/to/dataset \
    --find-outliers \
    --outlier-percentile 90 \
    --output dataset_analysis.json
```

Check median vs average to understand distribution:
- **Average >> Median**: Few very large documents
- **Average ≈ Median**: Uniform distribution

---

### 4. Split Large Documents

```bash
# Find files > 32K tokens (common model limit)
python count_tokens.py /path/to/data \
    --find-outliers \
    --outlier-percentile 75

# Manually split files listed in outliers
```

---

## 📈 Example Workflows

### Workflow 1: Pre-Processing Analysis

```bash
# Step 1: Count tokens in raw data
python count_tokens.py \
    /home/husainmalwat/workspace/OCR_Latex/data/final_merged/2023/final_merged \
    --model-name Qwen/Qwen2.5-Coder-14B-Instruct \
    --find-outliers \
    --output raw_token_stats.json

# Step 2: Review outliers and split if needed

# Step 3: Run LLM preprocessing
python run_llm_preprocessor.py ...

# Step 4: Count tokens again after preprocessing
python count_tokens.py \
    /path/to/processed/data \
    --model-name Qwen/Qwen2.5-Coder-14B-Instruct \
    --output processed_token_stats.json
```

---

### Workflow 2: Multi-Year Dataset Analysis

```bash
# Analyze each year separately
for year in 2020 2021 2022 2023; do
  python count_tokens.py \
    /home/husainmalwat/workspace/OCR_Latex/data/final_merged/${year}/final_merged \
    --output stats_${year}.json \
    --find-outliers
done

# Compare statistics across years
```

---

## 🔍 Interpreting Results

### Token Count Guidelines

| Token Count | Category | Recommendation |
|-------------|----------|----------------|
| < 2,000 | Small | Safe for all models |
| 2,000 - 8,000 | Medium | Standard processing |
| 8,000 - 32,000 | Large | May need attention |
| 32,000 - 64,000 | Very Large | Check model limits |
| > 64,000 | Extreme | Likely needs splitting |

### vLLM Configuration Recommendations

```bash
# For average_tokens ≤ 16K
--max-model-len 32768

# For average_tokens ≤ 32K
--max-model-len 65536

# For average_tokens > 32K
--max-model-len 131072 \
--rope-scaling '{"type": "yarn", "factor": 4.0, ...}'
```

---

## 🐛 Troubleshooting

### Issue: "Model not found"

**Solution:** Check model name spelling
```bash
# List available models on Hugging Face
huggingface-cli search Qwen
```

---

### Issue: Out of Memory Loading Tokenizer

**Solution:** Use a smaller model's tokenizer (they're often compatible)
```bash
# Instead of 70B model tokenizer, use 8B
--model-name Qwen/Qwen2.5-Coder-8B-Instruct
```

---

### Issue: Very Slow Processing

**Solution:** Process subdirectories separately
```bash
# Instead of entire dataset
python count_tokens.py /path/to/huge/dataset

# Process year by year
for year_dir in /path/to/dataset/*/; do
  python count_tokens.py "$year_dir" --output "stats_$(basename $year_dir).json"
done
```

---

## 📋 Integration with Other Tools

### Use with LLM Preprocessor

```bash
# 1. Count tokens before preprocessing
python count_tokens.py input_data/ --output before.json

# 2. Run preprocessing
python run_llm_preprocessor.py input_data/ output_data/ prompt_template.md

# 3. Count tokens after preprocessing
python count_tokens.py output_data/ --output after.json

# 4. Compare token reduction
```

### Use with LaTeX Compilation

```bash
# Count tokens in source files
python count_tokens.py /path/to/tex/files --output source_stats.json

# Compile LaTeX
python compile_latex.py --mode single --dir /path/to/tex/files

# Analyze which large files failed to compile
```

---

## 📚 Additional Resources

- [Transformers Tokenizers Documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [vLLM Configuration Guide](https://docs.vllm.ai/en/latest/models/engine_args.html)
- [Gemini API Pricing](https://ai.google.dev/pricing)

---