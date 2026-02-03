import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from preprocess_rule_based import LatexPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_single_file(
    preprocessor: LatexPreprocessor,
    input_path: str,
    output_path: str,
    args: argparse.Namespace
) -> bool:
    """Process a single LaTeX file."""
    bib_output = None
    if args.extract_bibliography and args.bib_output_dir:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        bib_output = os.path.join(args.bib_output_dir, f"{base_name}.bib")
    
    return preprocessor.preprocess(
        file_path=input_path,
        output_path=output_path,
        remove_comments=args.remove_comments,
        replace_images=args.replace_images,
        extract_bibliography=args.extract_bibliography,
        extract_document=args.extract_document,
        normalize_whitespace=args.normalize_whitespace,
        bib_output_path=bib_output
    )


def process_folder(
    preprocessor: LatexPreprocessor,
    input_folder: str,
    output_folder: str,
    args: argparse.Namespace
) -> tuple:
    """Process all LaTeX files in a folder."""
    tex_files = list(Path(input_folder).rglob("*.tex"))
    
    if not tex_files:
        logger.warning(f"No .tex files found in {input_folder}")
        return 0, 0
    
    logger.info(f"Found {len(tex_files)} LaTeX files to process")
    
    success_count = 0
    failure_count = 0
    
    for tex_file in tqdm(tex_files, desc="Processing files"):
        rel_path = tex_file.relative_to(input_folder)
        output_path = os.path.join(output_folder, rel_path)
        
        if process_single_file(preprocessor, str(tex_file), output_path, args):
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LaTeX files with various operations"
    )
    
    parser.add_argument(
        "input",
        help="Input file or folder path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or folder path (default: overwrites input)",
        default=None
    )
    parser.add_argument(
        "--bib-output-dir",
        help="Directory to save extracted bibliographies",
        default=None
    )
    
    parser.add_argument(
        "--no-remove-comments",
        dest="remove_comments",
        action="store_false",
        help="Skip removing comments"
    )
    parser.add_argument(
        "--no-replace-images",
        dest="replace_images",
        action="store_false",
        help="Skip replacing images with placeholders"
    )
    parser.add_argument(
        "--no-extract-bibliography",
        dest="extract_bibliography",
        action="store_false",
        help="Skip extracting bibliography"
    )
    parser.add_argument(
        "--no-extract-document",
        dest="extract_document",
        action="store_false",
        help="Skip extracting document content"
    )
    parser.add_argument(
        "--no-normalize-whitespace",
        dest="normalize_whitespace",
        action="store_false",
        help="Skip normalizing whitespace"
    )
    
    parser.set_defaults(
        remove_comments=True,
        replace_images=True,
        extract_bibliography=True,
        extract_document=True,
        normalize_whitespace=True
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    preprocessor = LatexPreprocessor()
    
    if os.path.isfile(args.input):
        logger.info(f"Processing single file: {args.input}")
        output_path = args.output or args.input
        
        success = process_single_file(preprocessor, args.input, output_path, args)
        
        if success:
            logger.info("✓ Processing completed successfully")
            return 0
        else:
            logger.error("✗ Processing failed")
            return 1
    
    elif os.path.isdir(args.input):
        logger.info(f"Processing folder: {args.input}")
        output_folder = args.output or args.input
        
        success_count, failure_count = process_folder(
            preprocessor, args.input, output_folder, args
        )
        
        total = success_count + failure_count
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Summary:")
        logger.info(f"  Total files: {total}")
        logger.info(f"  ✓ Successful: {success_count}")
        logger.info(f"  ✗ Failed: {failure_count}")
        logger.info(f"{'='*60}")
        
        return 0 if failure_count == 0 else 1
    
    else:
        logger.error(f"Invalid input path: {args.input}")
        return 1


if __name__ == "__main__":
    exit(main())