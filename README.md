# LaTeX Source Processing Pipeline

## 1. Objective

This script automates the process of extracting and consolidating a large collection of LaTeX source files,  from arXiv. The primary goal is to transform a nested structure of compressed archives (`.tar.gz`, `.gz`) into a clean, flat directory of <mark style="background-color: yellow; color: red;">single, self-contained</mark>   `.tex` files.

<!-- This is a <mark style="background-color: yellow; color: red;">single</mark> highlighted word. -->

The pipeline performs a three-step process for each month's data:
1.  **Primary Extraction**: Unpacks the main monthly archives (`2301.tar.gz`, `2302.tar.gz`, etc.).
2.  **Individual Paper Extraction**: Extracts each individual paper's source (`.gz`) into its own dedicated folder, handling different archive formats.
3.  **Merging **: Intelligently identifies the main `.tex` file for each paper, merges any `\input` or `\include` dependencies, and inlines the bibliography (`.bbl`) content to produce a single, final `.tex` file.

The entire process is logged to monthly JSON files, providing detailed statistics on successful and failed operations.


-   **Two-Stage Extraction**: Handles both `.tar.gz` archives and simple gzipped files (`.gz`).
-   **Intelligent Root File Detection**: Automatically finds the main `.tex` file in a project by looking for bibliography files (`.bbl`) or the `\begin{document}` command.
-   **Automated LaTeX Merging**: Replaces `\input{}`, `\include{}`, and `\bibliography{}` commands with the actual file content to create a single source file.
-   **Detailed Logging**: For each month, it generates a `stats_{yymm}.json` file with counts of processed, successful, and failed items at each stage.
-   **Modular Execution**: The script is designed to be run in its entirety or step-by-step, allowing for reprocessing or debugging of specific stages.
-   **Automatic Cleanup**: Deletes intermediate extracted files after each month's processing to conserve disk space.

## 3. Input and Output Mapping


### Expected Input Structure

The script expects the `BASE_PROCESSING_DIR` (e.g., `/mnt/NFS/patidarritesh/PDF_2_TEX/2023`) to contain a `src` directory with the initial archives.

```
/mnt/NFS/patidarritesh/PDF_2_TEX/2023/
└── src/
    ├── 2301.tar.gz
    ├── 2302.tar.gz
    ├── ...
    └── 2312.tar.gz
```

### Final Output Structure

After a full run, the base directory will be populated with the final merged TeX files and the processing logs.

```
/mnt/NFS/patidarritesh/PDF_2_TEX/2023/
├── src/
│   ├── 2301.tar.gz
│   └── ...
├── target_src/
│   ├── 2301/
│   │   ├── 2301.0001.gz
│   │   ├── 2301.0002.gz
│   │   └── ...
│   └── 2302/
│       └── ...
├── final_merged/
│   ├── 2301/
│   │   ├── 2301.0001.tex  (fully merged)
│   │   ├── 2301.0002.tex  (fully merged)
│   │   └── ...
│   └── 2302/
│       └── ...
└── processing_logs/
    ├── stats_2301.json
    ├── stats_2302.json
    └── ...
```

-   **`target_src/`**: An intermediate directory created by Step 1, containing the contents of the primary archives.
-   **`final_merged/`**: The final output. Each subfolder contains the fully processed, single-file `.tex` sources for that month.
-   **`processing_logs/`**: Contains a detailed JSON log for each month's operations.


## 4. How to Run the Script

The script is controlled from the `if __name__ == "__main__":` block at the bottom of the file. You can modify this block to run the entire pipeline or only specific parts.

### A. Running the Full Pipeline

To run the entire process for a full year (all three steps), simply execute the script as is. It will first perform the primary extraction and then loop through all 12 months for individual extraction and merging.

```bash
python process_arxiv_sources.py
```

### B. Modular Execution (Running Specific Steps)

You can easily control which steps are executed by commenting out parts of the main execution block.

#### To Run ONLY Step 1: Primary Archive Extraction

If you only want to unpack the main `src/*.tar.gz` files into the `target_src/` directory, comment out the monthly processing loop.

```python
if __name__ == "__main__":
    # --- CONFIGURATION ---
    YEAR = "2023"
    BASE_PROCESSING_DIR = Path(f"/mnt/NFS/patidarritesh/PDF_2_TEX/{YEAR}")
    
    # --- SCRIPT EXECUTION ---
    
    # Step 1: Run this once for the entire year.
    initial_source_dir = BASE_PROCESSING_DIR / "src"
    initial_target_dir = BASE_PROCESSING_DIR / "target_src"
    extract_primary_archives(initial_source_dir, initial_target_dir)

    # Step 2 & 3: Commented out
    # for month_num in range(1, 13):
    #     try:
    #         process_monthly_archives(YEAR, month_num, BASE_PROCESSING_DIR)
    #     except Exception as e:
    #         print(f"A critical error occurred while processing month {month_num}: {e}")

    print("\nPrimary extraction complete.")
```

#### To Run ONLY Steps 2 & 3: Monthly Processing

If you have already completed the primary extraction and the `target_src/` directory is populated, you can skip Step 1 by commenting it out.

```python
if __name__ == "__main__":
    # --- CONFIGURATION ---
    YEAR = "2023"
    BASE_PROCESSING_DIR = Path(f"/mnt/NFS/patidarritesh/PDF_2_TEX/{YEAR}")
    
    # --- SCRIPT EXECUTION ---
    
    # Step 1: Commented out
    # initial_source_dir = BASE_PROCESSING_DIR / "src"
    # initial_target_dir = BASE_PROCESSING_DIR / "target_src"
    # extract_primary_archives(initial_source_dir, initial_target_dir)

    # Step 2 & 3: Loop through each month to process the extracted contents.
    for month_num in range(1, 13):
        try:
            process_monthly_archives(YEAR, month_num, BASE_PROCESSING_DIR)
        except Exception as e:
            print(f"A critical error occurred while processing month {month_num}: {e}")

    print("\nMonthly processing complete.")
```

#### To Run ONLY for a Specific Month

To debug, re-run, or process just a single month (e.g., May, which is `5`), modify the `range` in the loop.

```python
if __name__ == "__main__":
    # --- CONFIGURATION ---
    YEAR = "2023"
    BASE_PROCESSING_DIR = Path(f"/mnt/NFS/patidarritesh/PDF_2_TEX/{YEAR}")

    # Run only for May (month 5)
    for month_num in [5]: # <-- Change is here
        try:
            process_monthly_archives(YEAR, month_num, BASE_PROCESSING_DIR)
        except Exception as e:
            print(f"A critical error occurred while processing month {month_num}: {e}")
```

## 5. Function Utilities

Here is a breakdown of the key functions and their roles in the pipeline:

| Function                        | Utility                                                                                                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `extract_primary_archives`      | **Step 1:** Handles the initial, large-scale extraction. Takes archives like `2301.tar.gz` from `src/` and unpacks their contents (many `.gz` files) into `target_src/2301/`. |
| `extract_individual_papers`     | **Step 2:** Processes each paper's `.gz` file. It first tries `tar` for extraction. If that fails, it assumes the file is a single gzipped TeX file and uses `gunzip` as a fallback. |
| `find_root_tex_file`            | **Core Merging Logic:** This function contains the "intelligence" for merging. It inspects a paper's directory to find the main `.tex` file, which is crucial for projects with multiple source files. |
| `merge_tex_files`               | **Core Merging Engine:** Reads the content of the root TeX file and uses regular expressions to replace `\input`, `\include`, and `\bibliography` commands with the content of the corresponding files. |
| `merge_and_finalize_papers`     | **Step 3 Orchestrator:** Iterates through all extracted paper directories for a month, uses `find_root_tex_file` and `merge_tex_files` to generate the final output, and logs the results. |
| `process_monthly_archives`      | **Main Controller:** A wrapper function that calls the extraction (Step 2) and merging (Step 3) functions for a single month, handles directory creation, and saves the final log file. |

## 6. Understanding the Log Files

At the end of each month's processing, a JSON file like `stats_2301.json` is created in the `processing_logs/` directory. This file provides a detailed summary of the operations.

**Example `stats_2301.json`:**
```json
{
    "year": "2023",
    "month": "01",
    "status": "completed",
    "papers_to_process": 2500,
    "failed_extractions": [
        "2301.1234.gz"
    ],
    "papers_extracted_successfully": 2499,
    "merging_stats": {
        "total_papers": 2499,
        "merged_successfully": 2495,
        "single_tex_file": 2000,
        "failed_to_merge": [
            "2301.5678",
            "2301.9999"
        ],
        "no_tex_files": [
            "2301.0010",
            "2301.0020"
        ]
    }
}
```

-   **`papers_to_process`**: Total `.gz` files found for the month.
-   **`failed_extractions`**: A list of papers that could not be extracted even with the fallback.
-   **`merged_successfully`**: The final count of papers for which a single `.tex` file was generated.
-   **`failed_to_merge`**: Papers that were extracted but where the merging logic failed (e.g., could not find a root file in a multi-file project).
-   **`no_tex_files`**: Papers whose archives contained no `.tex` files after extraction.