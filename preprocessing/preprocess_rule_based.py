import os
import re
import logging
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LatexPreprocessor:
    """Handles preprocessing of LaTeX files."""
    
    PATTERN_COMMENT = r"((?<!\\)%.*)"
    PATTERN_IFFALSE = r"\\iffalse"
    PATTERN_FI = r"\\fi"
    PATTERN_BEGIN_COMMENT = r"\\begin{comment}"
    PATTERN_END_COMMENT = r"\\end{comment}"
    
    PATTERN_BIB = [r"\\begin{thebibliography}", r"\\end{thebibliography}"]
    PATTERN_DOCUMENT = [r"\\begin{document}", r"\\end{document}"]
    
    IMAGE_PATTERNS = [
        [r'\\begin{figure}', r'\\end{figure}'],
        [r'\\begin{figure\*}', r'\\end{figure\*}'],
        [r'\\begin{teaserfigure}', r'\\end{teaserfigure}'],
        [r'\\begin{teaserfigure\*}', r'\\end{teaserfigure\*}'],
        [r'\\begin{wrapfigure}', r'\\end{wrapfigure}'],
        [r'\\begin{wrapfigure\*}', r'\\end{wrapfigure\*}'],
    ]
    
    DUMMY_IMAGE = '\n\\begin{figure}[h]\n\\includegraphics[]{image_name}\n\\caption{Placeholder}\n\\label{fig:placeholder}\n\\end{figure}\n\n'
    
    def __init__(self):
        self.error_count = 0
    
    @staticmethod
    def _remove_commands(data: str, start_pattern: str, end_pattern: str) -> str:
        """Remove content between start and end patterns."""
        matches = list(re.finditer(start_pattern, data))
        if not matches:
            return data
        
        new_data = ""
        prev_end = 0
        
        for match in matches:
            start = match.start()
            end_match = re.search(end_pattern, data[start:])
            if end_match is None:
                logger.warning(f"End pattern '{end_pattern}' not found after position {start}")
                continue
            
            end = end_match.end()
            new_data += data[prev_end:start]
            prev_end = start + end
        
        new_data += data[prev_end:]
        return new_data
    
    def remove_comments(self, data: str) -> str:
        """Remove LaTeX comments from the text."""
        data = re.sub(self.PATTERN_COMMENT, "", data)
        
        data = self._remove_commands(data, self.PATTERN_IFFALSE, self.PATTERN_FI)
        
        data = self._remove_commands(data, self.PATTERN_BEGIN_COMMENT, self.PATTERN_END_COMMENT)
        
        return data
    
    def replace_images_with_placeholder(self, data: str) -> str:
        """Replace all figure environments with a placeholder."""
        for pattern in self.IMAGE_PATTERNS:
            start_pattern, end_pattern = pattern
            matches = list(re.finditer(start_pattern, data))
            
            if not matches:
                continue
            
            new_data = ""
            prev_end = 0
            
            for match in matches:
                start = match.start()
                end_match = re.search(end_pattern, data[start:])
                
                if end_match is None:
                    logger.warning(f"End pattern '{end_pattern}' not found after position {start}")
                    continue
                
                end = end_match.end()
                new_data += data[prev_end:start] + self.DUMMY_IMAGE
                prev_end = start + end
            
            new_data += data[prev_end:]
            data = new_data
        
        return data
    
    def extract_bibliography(self, data: str) -> Tuple[str, str]:
        """Extract bibliography section and return modified data and bibliography."""
        start_pattern, end_pattern = self.PATTERN_BIB
        bib_data = ""
        
        matches = list(re.finditer(start_pattern, data))
        if not matches:
            return data, bib_data
        
        new_data = ""
        prev_end = 0
        
        for match in matches:
            start = match.start()
            end_match = re.search(end_pattern, data[start:])
            
            if end_match is None:
                logger.warning(f"Bibliography end tag not found after position {start}")
                continue
            
            end = end_match.end()
            new_data += data[prev_end:start]
            prev_end = start + end
            bib_data += data[start:start + end] + "\n"
        
        new_data += data[prev_end:]
        bib_data = re.sub(r'\n+', '\n', bib_data)
        
        return new_data, bib_data
    
    def extract_document_content(self, data: str) -> Optional[str]:
        """Extract content between \\begin{document} and \\end{document}."""
        start_pattern, end_pattern = self.PATTERN_DOCUMENT
        
        matches = list(re.finditer(start_pattern, data))
        if not matches:
            logger.warning("\\begin{document} not found")
            return data
        
        if len(matches) > 1:
            logger.warning(f"Multiple \\begin{{document}} tags found ({len(matches)})")
        
        match = matches[0]
        start, end_begin = match.start(), match.end()
        
        end_match = re.search(end_pattern, data[start:])
        if end_match is None:
            logger.warning("\\end{document} not found")
            return data
        
        start_end = end_match.start()
        return data[end_begin:start + start_end]
    
    def normalize_whitespace(self, data: str) -> str:
        """Normalize whitespace by removing multiple newlines."""
        return re.sub(r'\n+', '\n', data)
    
    def preprocess(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        remove_comments: bool = True,
        replace_images: bool = True,
        extract_bibliography: bool = True,
        extract_document: bool = True,
        normalize_whitespace: bool = True,
        bib_output_path: Optional[str] = None
    ) -> bool:
        """
        Preprocess a LaTeX file with specified operations.
        
        Args:
            file_path: Path to input LaTeX file
            output_path: Path to save processed file (if None, overwrites input)
            remove_comments: Remove comments and conditional blocks
            replace_images: Replace figure environments with placeholders
            extract_bibliography: Extract and optionally save bibliography
            extract_document: Extract content between document tags
            normalize_whitespace: Remove multiple newlines
            bib_output_path: Path to save extracted bibliography
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                data = f.read()
                try:
                    data = data.decode("utf-8")
                except UnicodeDecodeError:
                    data = data.decode("latin-1", errors="ignore")
                    logger.warning(f"Used latin-1 encoding for {file_path}")
            
            if remove_comments:
                data = self.remove_comments(data)
                logger.debug(f"Removed comments from {file_path}")
            
            if replace_images:
                data = self.replace_images_with_placeholder(data)
                logger.debug(f"Replaced images in {file_path}")
            
            bib_data = ""
            if extract_bibliography:
                data, bib_data = self.extract_bibliography(data)
                logger.debug(f"Extracted bibliography from {file_path}")
                
                if bib_data and bib_output_path:
                    os.makedirs(os.path.dirname(bib_output_path), exist_ok=True)
                    with open(bib_output_path, "w", encoding="utf-8") as f:
                        f.write(bib_data)
                    logger.info(f"Saved bibliography to {bib_output_path}")
            
            # if extract_document:
            #     extracted = self.extract_document_content(data)
            #     if extracted:
            #         data = extracted
            #         logger.debug(f"Extracted document content from {file_path}")
            
            if normalize_whitespace:
                data = self.normalize_whitespace(data)
                logger.debug(f"Normalized whitespace in {file_path}")
            
            output_file = output_path or file_path
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)
            
            logger.info(f"Successfully processed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            self.error_count += 1
            return False