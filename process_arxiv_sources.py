import tarfile
import subprocess
import re
import json
from pathlib import Path
from tqdm import tqdm
import shutil

# --- REGULAR EXPRESSIONS FOR LATEX PARSING ---

# Matches \input{...} or \include{...} commands, including commented out ones
# Captures the filename inside the braces.
RE_INCLUDE = re.compile(
    r"%*\s*\\(?:input|include)\s*\{?([\w\/\-\.]+)\}?"
)

# Matches \begin{document} to identify the main .tex file.
RE_BEGIN_DOCUMENT = re.compile(
    r"\\begin\s*\{document\}", re.IGNORECASE
)

# Matches bibliography commands to replace with .bbl content.
RE_BIBLIOGRAPHY = re.compile(
    r"\\bibliography\s*\{?([\w\/\-\.,]+)\}?"
)


def extract_primary_archives(source_dir: Path, target_dir: Path):
    """
    Extracts top-level .tar.gz archives from a source directory.
    This is the first stage of extraction.
    
    Args:
        source_dir: Directory containing the yearly .tar.gz files.
        target_dir: Directory where the contents will be extracted.
    """
    print(f"--- Starting Step 1: Primary Archive Extraction ---")
    target_dir.mkdir(parents=True, exist_ok=True)

    for archive_path in source_dir.glob('*.tar.gz'):
        print(f"Extracting {archive_path.name} to {target_dir}...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=target_dir)
            print(f"Successfully extracted {archive_path.name}.")
        except tarfile.TarError as e:
            print(f"Error extracting {archive_path.name}: {e}")
    print("--- Primary extraction complete. ---\n")


def extract_individual_papers(monthly_source_dir: Path, monthly_final_extract_dir: Path, temp_dir: Path, stats: dict):
    """
    Extracts each individual paper's .gz file into its own dedicated folder.
    Handles cases where the file is a tarball or just a gzipped TeX file.
    
    Args:
        monthly_source_dir: Directory containing .gz files for a specific month.
        monthly_final_extract_dir: The parent directory to store extracted papers.
        temp_dir: A temporary directory for handling extraction fallbacks.
        stats: Dictionary to log processing statistics.
    """
    if not monthly_source_dir.exists():
        print(f"Warning: Source directory {monthly_source_dir} not found. Skipping.")
        return

    gz_files = list(monthly_source_dir.glob('*.gz'))
    stats['papers_to_process'] = len(gz_files)
    stats['failed_extractions'] = []
    
    for gz_path in tqdm(gz_files, desc=f"Extracting papers for {monthly_source_dir.name}"):
        paper_id = gz_path.stem
        paper_output_dir = monthly_final_extract_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Standard extraction using tar
            subprocess.run(
                ["tar", "-xzf", str(gz_path), "-C", str(paper_output_dir)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            # Fallback for non-tar gzipped files (e.g., single .tex file)
            shutil.rmtree(paper_output_dir) # Clean up failed tar attempt
            paper_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                temp_gz_path = temp_dir / gz_path.name
                shutil.copy(gz_path, temp_gz_path)
                
                subprocess.run(["gunzip", str(temp_gz_path)], check=True)
                
                unzipped_file = temp_dir / paper_id
                if unzipped_file.exists():
                    shutil.move(str(unzipped_file), str(paper_output_dir / "main.tex"))
                else:
                    raise FileNotFoundError("Gunzip did not produce expected file.")

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                stats['failed_extractions'].append(gz_path.name)
                print(f"Fallback extraction failed for {gz_path.name}: {e}")
                shutil.rmtree(paper_output_dir, ignore_errors=True)

    stats['papers_extracted_successfully'] = stats['papers_to_process'] - len(stats['failed_extractions'])


def find_root_tex_file(paper_dir: Path) -> Path | None:
    """
    Identifies the main .tex file in a directory.
    
    Strategy:
    1. Look for a .bbl file and a matching .tex file.
    2. If not found, search all .tex files for '\\begin{document}'.
    
    Returns:
        The path to the root .tex file, or None if not found.
    """
    tex_files = list(paper_dir.glob('*.tex'))
    if not tex_files:
        return None
        
    # Strategy 1: Check for a .bbl file and matching .tex
    bbl_files = list(paper_dir.glob('*.bbl'))
    if bbl_files:
        bbl_stem = bbl_files[0].stem
        matching_tex = paper_dir / f"{bbl_stem}.tex"
        if matching_tex.exists():
            return matching_tex

    # Strategy 2: Search for \begin{document}
    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            if RE_BEGIN_DOCUMENT.search(content):
                return tex_file
        except Exception:
            continue
    
    # Fallback: if only one .tex file, assume it's the root
    if len(tex_files) == 1:
        return tex_files[0]
        
    return None


def merge_tex_files(root_file_path: Path, paper_dir: Path) -> str:
    """
    Recursively merges content from \\input and \\include commands into the root file's content.
    Also handles bibliography merging.
    
    Args:
        root_file_path: Path to the main .tex file.
        paper_dir: The directory of the paper being processed.
        
    Returns:
        A string containing the fully merged LaTeX content.
    """
    try:
        content = root_file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Could not read root file {root_file_path}: {e}")
        return ""

    def replace_include(match):
        included_filename = match.group(1)
        # Ensure filename has .tex extension for searching
        if not included_filename.endswith('.tex'):
            included_filename += '.tex'
        
        file_to_include = paper_dir / included_filename
        if file_to_include.exists():
            return file_to_include.read_text(encoding='utf-8', errors='ignore')
        return "" # If file not found, replace command with empty string

    def replace_bibliography(match):
        bbl_files = list(paper_dir.glob('*.bbl'))
        if bbl_files:
            return bbl_files[0].read_text(encoding='utf-8', errors='ignore')
        return ""

    merged_content = RE_INCLUDE.sub(replace_include, content)
    merged_content = RE_BIBLIOGRAPHY.sub(replace_bibliography, merged_content)
    
    return merged_content


def merge_and_finalize_papers(monthly_final_extract_dir: Path, monthly_final_merged_dir: Path, stats: dict):
    """
    Processes each extracted paper directory to produce a single, merged .tex file.
    
    Args:
        monthly_final_extract_dir: Directory containing subfolders for each extracted paper.
        monthly_final_merged_dir: Destination directory for the final .tex files.
        stats: Dictionary to log processing statistics.
    """
    if not monthly_final_extract_dir.exists():
        return

    paper_dirs = [d for d in monthly_final_extract_dir.iterdir() if d.is_dir()]
    stats['merging_stats'] = {
        'total_papers': len(paper_dirs),
        'merged_successfully': 0,
        'single_tex_file': 0,
        'failed_to_merge': [],
        'no_tex_files': []
    }

    for paper_dir in tqdm(paper_dirs, desc=f"Merging TeX files for {monthly_final_extract_dir.name}"):
        tex_files = list(paper_dir.glob('*.tex'))
        
        if not tex_files:
            stats['merging_stats']['no_tex_files'].append(paper_dir.name)
            continue
        
        final_tex_path = monthly_final_merged_dir / f"{paper_dir.name}.tex"

        if len(tex_files) == 1:
            # If only one .tex file, simply copy it after checking for bibliography
            content = merge_tex_files(tex_files[0], paper_dir)
            final_tex_path.write_text(content, encoding='utf-8')
            stats['merging_stats']['single_tex_file'] += 1
            stats['merging_stats']['merged_successfully'] += 1
            continue

        root_file = find_root_tex_file(paper_dir)

        if root_file:
            merged_content = merge_tex_files(root_file, paper_dir)
            if merged_content:
                final_tex_path.write_text(merged_content, encoding='utf-8')
                stats['merging_stats']['merged_successfully'] += 1
            else:
                stats['merging_stats']['failed_to_merge'].append(paper_dir.name)
        else:
            stats['merging_stats']['failed_to_merge'].append(paper_dir.name)


def process_monthly_archives(year: str, month: int, base_dir: Path):
    """
    Main processing pipeline for a single month's worth of archives.
    
    Args:
        year: The year to process (e.g., "2023").
        month: The month to process (1-12).
        base_dir: The root directory for all processing folders (e.g., PDF_2_TEX/2023).
    """
    month_str = f"{month:02d}"
    year_short = year[2:]
    folder_prefix = f"{year_short}{month_str}"

    # --- Define Paths ---
    source_dir = base_dir / "target_src" / folder_prefix
    final_extract_dir = base_dir / "final_extracted" / folder_prefix
    final_merged_dir = base_dir / "final_merged" / folder_prefix
    temp_dir = base_dir / "temp_processing"
    log_dir = base_dir / "processing_logs"
    
    # --- Create Directories ---
    final_extract_dir.mkdir(parents=True, exist_ok=True)
    final_merged_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # --- Initialize Logging ---
    processing_stats = {
        'year': year,
        'month': month_str,
        'status': 'started'
    }

    print(f"\n===== PROCESSING {folder_prefix} =====")
    
    # --- Step 2: Extract Individual Papers ---
    print("\n--- Starting Step 2: Individual Paper Extraction ---")
    extract_individual_papers(source_dir, final_extract_dir, temp_dir, processing_stats)
    print("--- Individual paper extraction complete. ---")

    # --- Step 3: Merge TeX Files ---
    print("\n--- Starting Step 3: Merging TeX Files ---")
    merge_and_finalize_papers(final_extract_dir, final_merged_dir, processing_stats)
    print("--- TeX file merging complete. ---")

    # --- Finalization ---
    processing_stats['status'] = 'completed'
    
    # Save statistics to JSON file
    log_file_path = log_dir / f"stats_{folder_prefix}.json"
    with open(log_file_path, 'w') as f:
        json.dump(processing_stats, f, indent=4)
        
    # Clean up temporary directory and extracted files to save space
    shutil.rmtree(temp_dir)
    shutil.rmtree(final_extract_dir)
    print(f"--- Cleaned up temporary and intermediate extracted files for {folder_prefix}. ---")
    print(f"===== FINISHED PROCESSING {folder_prefix} =====")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    YEAR = "2023"
    BASE_PROCESSING_DIR = Path(f"/mnt/NFS/patidarritesh/PDF_2_TEX/{YEAR}")
    
    # --- SCRIPT EXECUTION ---
    
    # Step 1: Run this once for the entire year.
    # It unpacks the main tarballs (e.g., 2301.tar.gz) into the target_src directory.
    initial_source_dir = BASE_PROCESSING_DIR / "src"
    initial_target_dir = BASE_PROCESSING_DIR / "target_src"
    extract_primary_archives(initial_source_dir, initial_target_dir)

    # Step 2 & 3: Loop through each month to process the extracted contents.
    for month_num in range(1, 13):
        try:
            process_monthly_archives(YEAR, month_num, BASE_PROCESSING_DIR)
        except Exception as e:
            print(f"A critical error occurred while processing month {month_num}: {e}")
            # Optionally log this to a main error file
            error_log_path = BASE_PROCESSING_DIR / "main_error_log.txt"
            with open(error_log_path, "a") as f:
                f.write(f"Error in month {month_num}: {e}\n")

    print("\nAll processing complete for the year.")