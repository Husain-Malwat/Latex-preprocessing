import re
from pathlib import Path
import argparse

# --- REGULAR EXPRESSIONS ---
RE_INCLUDE = re.compile(r"%*\s*\\(?:input|include)\s*\{?([\w\/\-\.]+)\}?")
RE_BEGIN_DOCUMENT = re.compile(r"\\begin\s*\{document\}", re.IGNORECASE)
RE_BIBLIOGRAPHY = re.compile(r"\\bibliography\s*\{?([\w\/\-\.,]+)\}?")

def find_root_tex_file(paper_dir: Path) -> Path | None:
    tex_files = list(paper_dir.glob('*.tex'))

    if not tex_files:
        return None

    # Strategy 1: .bbl match
    bbl_files = list(paper_dir.glob('*.bbl'))
    
    if bbl_files:
        bbl_stem = bbl_files[0].stem
        match_tex = paper_dir / f"{bbl_stem}.tex"
        if match_tex.exists():
            return match_tex
    
    # Strategy 2: look for \begin{document}
    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            if RE_BEGIN_DOCUMENT.search(content):
                return tex_file
        except Exception:
            continue

    if len(tex_files) == 1:
        return tex_files[0]

    return None

def merge_tex_files(root_file_path: Path, paper_dir: Path, merge_bib: bool = True) -> str:
    try:
        content = root_file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Could not read {root_file_path}: {e}")
        return ""

    def replace_include(match):
        fname = match.group(1)
        if not fname.endswith('.tex'):
            fname += '.tex'
        fpath = paper_dir / fname
        if fpath.exists():
            return fpath.read_text(encoding='utf-8', errors='ignore')
        return ""

    def replace_bibliography(match):
        bbl_files = list(paper_dir.glob('*.bbl'))
        if bbl_files:
            return bbl_files[0].read_text(encoding='utf-8', errors='ignore')
        return ""

    merged = RE_INCLUDE.sub(replace_include, content)
    
    if merge_bib:
        merged = RE_BIBLIOGRAPHY.sub(replace_bibliography, merged)
    
    return merged

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="merge LaTeX files")
    parser.add_argument("-i", "--input", required=True, help="Input latex folder")
    parser.add_argument("-o", "--output", required=True, help="Output merged .tex file")
    parser.add_argument("--merge-bib", action="store_true", 
                        help="If set, replaces \\bibliography commands with .bbl content")
    
    args = parser.parse_args()

    paper_dir = Path(args.input)
    output_file = Path(args.output)
    output_dir = output_file.parent  

    output_dir.mkdir(parents=True, exist_ok=True)

    root_tex = find_root_tex_file(paper_dir)
    
    print("Configuration:")
    print(f" Input directory: {paper_dir}")
    print(f" Output file: {output_file}")
    print(f" Merge bibliography: {args.merge_bib}")
    
    if not root_tex:
        print("No root .tex file found.")
    else:
        merged_content = merge_tex_files(root_tex, paper_dir, args.merge_bib)
        output_file.write_text(merged_content, encoding='utf-8')
        print(f"Merged .tex saved to {output_file}")
        if args.merge_bib:
            print("Bibliography content was merged from .bbl file")
        else:
            print("Bibliography commands were preserved")