import os
import argparse
from pathlib import Path
from typing import List
from merge_arxiv_tex import find_root_tex_file, merge_tex_files

def is_latex_dir(dir_path: Path) -> bool:
    """Check if a directory contains LaTeX files."""
    return any(Path(dir_path).glob('*.tex'))

def find_latex_dirs(parent_dir: Path) -> List[Path]:
    """Find all directories containing LaTeX files within the parent directory."""
    latex_dirs = []
    
    for item in parent_dir.iterdir():
        if item.is_dir():
            if is_latex_dir(item):
                latex_dirs.append(item)
            for subitem in item.iterdir():
                if subitem.is_dir() and is_latex_dir(subitem):
                    latex_dirs.append(subitem)
    
    return latex_dirs

def process_all_latex_dirs(parent_dir: Path, output_dir: Path, merge_bib: bool = False):
    """Process all LaTeX directories found in parent_dir."""
    latex_dirs = find_latex_dirs(parent_dir)
    
    if not latex_dirs:
        print(f"No directories containing LaTeX files found in {parent_dir}")
        return
    
    print(f"Found {len(latex_dirs)} LaTeX directories to process")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for latex_dir in latex_dirs:
        try:
            print(f"\nProcessing: {latex_dir}")
            
            output_name = f"{latex_dir.name}.tex"
            output_file = output_dir / output_name
            
            root_tex = find_root_tex_file(latex_dir)
            if not root_tex:
                print(f"  No root .tex file found in {latex_dir}")
                fail_count += 1
                continue
            
            merged_content = merge_tex_files(root_tex, latex_dir, merge_bib)
            output_file.write_text(merged_content, encoding='utf-8')
            
            print(f"  Success: Merged .tex saved to {output_file}")
            if merge_bib:
                print("  Bibliography content was merged from .bbl file")
            else:
                print("  Bibliography commands were preserved")
                
            success_count += 1
            
        except Exception as e:
            print(f"  Error processing {latex_dir}: {e}")
            fail_count += 1
    
    print(f"\nProcessing complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple LaTeX directories and merge files")
    parser.add_argument("-i", "--input", required=True, help="Parent directory containing LaTeX subdirectories")
    parser.add_argument("-o", "--output", required=True, help="Output directory for merged .tex files")
    parser.add_argument("--merge-bib", action="store_true", 
                        help="If set, replaces \\bibliography commands with .bbl content")
    
    args = parser.parse_args()
    
    parent_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("Configuration:")
    print(f" Parent directory: {parent_dir}")
    print(f" Output directory: {output_dir}")
    print(f" Merge bibliography: {args.merge_bib}")
    
    process_all_latex_dirs(parent_dir, output_dir, args.merge_bib)