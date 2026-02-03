# 📘 LaTeX Compilation - Instructions

## Overview

The `compile_latex.py` script compiles LaTeX documents in batch with advanced features including timeout handling, structured logging, and progress tracking.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Timeout Handling** | Automatically stops compilation after 5 seconds (configurable) per document |
| **Failure Recovery** | Logs TIMEOUT, FAILED, or ERROR status for each `.tex` file |
| **Per-File Logs** | Each `.tex` file generates a `.compile.log` stored in the output directory |
| **Master Log** | Timestamped master log file saved in `./logs/` directory |
| **Dual Modes** | Process single directory or all year directories |

---

## 🚀 Usage

### Basic Syntax
```bash
python3 compile_latex.py --mode <MODE> [OPTIONS]
```

### Mode 1: Compile All Year Directories
Processes all numeric directories (e.g., 2000, 2001, ..., 2023) in the base directory.

```bash
python3 compile_latex.py --mode all
```

**With custom base directory:**
```bash
python3 compile_latex.py --mode all --base_dir /path/to/your/data/final_merged
```

**With custom timeout and workers:**
```bash
python3 compile_latex.py --mode all --timeout 10 --workers 8
```

---

### Mode 2: Compile Single Directory
Process a specific directory containing `.tex` files.

```bash
python3 compile_latex.py --mode single --dir /path/to/specific/folder
```

**Example:**
```bash
python3 compile_latex.py --mode single --dir /home/husainmalwat/workspace/OCR_Latex/data/final_merged/2023
```

---

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | Required | - | Compilation mode: `all` or `single` |
| `--dir` | Optional | - | Directory path (required for `single` mode) |
| `--base_dir` | Optional | `/home/husainmalwat/workspace/OCR_Latex/data/final_merged` | Base directory for `all` mode |
| `--workers` | Optional | 12 | Number of parallel workers (currently sequential) |
| `--timeout` | Optional | 5 | Timeout per document in seconds |

---

## 📁 Directory Structure

### Input Structure
```
final_merged/
├── 2020/
│   └── final_merged/
│       ├── document1.tex
│       ├── document2.tex
│       └── ...
├── 2021/
│   └── final_merged/
│       └── ...
└── ...
```

### Output Structure
```
final_merged_compiled/
├── document1.pdf
├── document1.tex.compile.log
├── document2.pdf
├── document2.tex.compile.log
└── ...

logs/
└── latex_compile_20241214_153045.log  (master log)
```

---

## 📊 Log Files

### Master Log (`./logs/latex_compile_TIMESTAMP.log`)
Contains:
- Overall progress and statistics
- Success/failure counts per directory
- All errors and warnings
- Compilation timestamps

Example output:
```
2024-12-14 15:30:45 [INFO] 🚀 Starting LaTeX compilation | Mode: ALL | Log: ./logs/latex_compile_20241214_153045.log
2024-12-14 15:30:45 [INFO] 📦 Found 8 year directories to process.
2024-12-14 15:30:46 [INFO] 📁 Found 150 .tex files in /path/to/2020
2024-12-14 15:31:15 [INFO] ✅ 145 succeeded, ❌ 5 failed in /path/to/2020
2024-12-14 15:31:15 [INFO] 🏁 Compilation process finished.
```

### Per-File Logs (`*.tex.compile.log`)
Contains:
- Full pdflatex output
- Error messages and warnings
- Package loading information

---

## 🔍 Status Codes

| Status | Meaning |
|--------|---------|
| ✅ **SUCCESS** | Document compiled successfully (PDF generated) |
| ⏰ **TIMEOUT** | Compilation exceeded timeout limit |
| ❌ **FAILED** | pdflatex returned non-zero exit code |
| 💥 **ERROR** | Unexpected error (file access, permissions, etc.) |

---

## 💡 Best Practices

### 1. Test on Small Dataset First
```bash
python3 compile_latex.py --mode single --dir /path/to/test/folder --timeout 10
```

### 2. Adjust Timeout Based on Document Complexity
- Simple documents: `--timeout 3`
- Complex documents with many packages: `--timeout 10`
- Documents with heavy computations: `--timeout 20`

### 3. Monitor Logs
Check the master log file in `./logs/` for compilation statistics:
```bash
tail -f ./logs/latex_compile_*.log
```

### 4. Handle Failed Documents
After compilation, check failed documents:
```bash
grep "FAILED\|TIMEOUT\|ERROR" ./logs/latex_compile_*.log
```

---

## 🐛 Troubleshooting

### Issue: "pdflatex: command not found"
**Solution:** Install LaTeX distribution
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# Verify installation
pdflatex --version
```

---

