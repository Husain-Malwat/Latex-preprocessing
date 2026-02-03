# LaTeX Rule-Based Preprocessing

how to use the LaTeX preprocessor to clean and standardize LaTeX files.

The LaTeX preprocessor has five main functionalities:

1. **Remove Comments** - Removes inline comments (%), conditional blocks (\iffalse...\fi), and comment environments
2. **Replace Images** - Replaces all figure environments with standardized placeholders
3. **Extract Bibliography** - Extracts bibliography sections and optionally saves them separately
4. **Extract Document Content** - Extracts only the content between \begin{document} and \end{document}
5. **Normalize Whitespace** - Removes multiple consecutive newlines



### Files

- `preprocess_rule_based.py` - Core preprocessing library
- `run_preprocessor.py` - CLI interface for running preprocessing

## Usage

### Basic Commands

#### Process a Single File

```bash
# Process with all operations enabled (default)
python run_preprocessor.py input.tex

# Process and save to different output file
python run_preprocessor.py input.tex -o output.tex
```

#### Process a Folder (Batch Processing)

```bash
# Process all .tex files in a folder
python run_preprocessor.py /path/to/input/folder

# Process folder and save to different output folder
python run_preprocessor.py /path/to/input/folder -o /path/to/output/folder
```

### Other Options

#### Extract Bibliographies to Separate Directory

```bash
python run_preprocessor.py input_folder -o output_folder --bib-output-dir bibliographies/
```

This will:
- Process all .tex files
- Save processed files to `output_folder`
- Save extracted bibliographies to `bibliographies/` with `.bib` extension

#### Disable Specific Operations

All operations are enabled by default. Use flags to disable specific operations:

```bash
# Skip removing comments
python run_preprocessor.py input.tex --no-remove-comments

# Skip replacing images with placeholders
python run_preprocessor.py input.tex --no-replace-images

# Skip extracting bibliography
python run_preprocessor.py input.tex --no-extract-bibliography

# Skip extracting document content
python run_preprocessor.py input.tex --no-extract-document

# Skip normalizing whitespace
python run_preprocessor.py input.tex --no-normalize-whitespace
```

#### Combine Multiple Flags

```bash
# Only remove comments and normalize whitespace
python run_preprocessor.py input.tex \
    --no-replace-images \
    --no-extract-bibliography \
    --no-extract-document
```


##  Usage

You can use the preprocessor in your Python code:

```python
from preprocess_rule_based import LatexPreprocessor

# Initialize preprocessor
preprocessor = LatexPreprocessor()

# Process a single file
success = preprocessor.preprocess(
    file_path="input.tex",
    output_path="output.tex",
    remove_comments=True,
    replace_images=True,
    extract_bibliography=True,
    extract_document=True,
    normalize_whitespace=True,
    bib_output_path="bibliography.bib"
)

# Check result
if success:
    print("Processing completed successfully")
else:
    print(f"Processing failed. Error count: {preprocessor.error_count}")
```

## Command Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input file or folder path |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-o, --output` | Output file or folder path | Overwrites input |
| `--bib-output-dir` | Directory for extracted bibliographies | None |
| `--no-remove-comments` | Skip removing comments | False (enabled) |
| `--no-replace-images` | Skip replacing images | False (enabled) |
| `--no-extract-bibliography` | Skip extracting bibliography | False (enabled) |
| `--no-extract-document` | Skip extracting document content | False (enabled) |
| `--no-normalize-whitespace` | Skip normalizing whitespace | False (enabled) |
