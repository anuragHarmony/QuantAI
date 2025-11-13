"""Document processing for PDF and EPUB files."""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


@dataclass
class DocumentSection:
    """Represents a section of a document."""
    title: str
    content: str
    page_start: int
    page_end: int
    level: int  # 1=chapter, 2=section, 3=subsection
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """Processes documents to extract structured content."""

    def __init__(self):
        """Initialize the document processor."""
        if fitz is None:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

    def process_pdf(self, file_path: Path) -> List[DocumentSection]:
        """
        Process a PDF file and extract structured sections.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of DocumentSection objects
        """
        logger.info(f"Processing PDF: {file_path}")

        try:
            doc = fitz.open(file_path)
            sections = []
            current_section = None
            current_content = []

            # Extract text page by page
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()

                # Try to detect section headers (simple heuristic)
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Detect if line is a header (you may need to adjust this)
                    if self._is_header(line):
                        # Save previous section
                        if current_section and current_content:
                            current_section.content = '\n'.join(current_content)
                            current_section.page_end = page_num - 1
                            sections.append(current_section)

                        # Start new section
                        level = self._detect_header_level(line)
                        current_section = DocumentSection(
                            title=line,
                            content="",
                            page_start=page_num,
                            page_end=page_num,
                            level=level
                        )
                        current_content = []
                    else:
                        current_content.append(line)

            # Add the last section
            if current_section and current_content:
                current_section.content = '\n'.join(current_content)
                current_section.page_end = len(doc)
                sections.append(current_section)

            doc.close()
            logger.info(f"Extracted {len(sections)} sections from {file_path.name}")
            return sections

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise

    def extract_full_text(self, file_path: Path) -> str:
        """
        Extract full text from a PDF without structure.

        Args:
            file_path: Path to the PDF file

        Returns:
            Full text content
        """
        logger.info(f"Extracting full text from: {file_path}")

        try:
            doc = fitz.open(file_path)
            full_text = ""

            for page in doc:
                full_text += page.get_text() + "\n\n"

            doc.close()
            return full_text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    def _is_header(self, line: str) -> bool:
        """
        Detect if a line is likely a header.

        Simple heuristics:
        - All caps
        - Short (< 100 chars)
        - Starts with "Chapter" or numbers
        - No period at the end
        """
        line = line.strip()
        if not line or len(line) > 100:
            return False

        # Check for chapter indicators
        if re.match(r'^(Chapter|CHAPTER|\d+\.?\s+[A-Z])', line):
            return True

        # Check if mostly caps and short
        if line.isupper() and len(line) < 80:
            return True

        return False

    def _detect_header_level(self, line: str) -> int:
        """
        Detect the hierarchy level of a header.

        Returns:
            1 for chapter, 2 for section, 3 for subsection
        """
        if re.match(r'^(Chapter|CHAPTER|\d+\s+[A-Z])', line):
            return 1

        if line.isupper():
            return 2

        return 3

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find the end position
            end = start + chunk_size

            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break != -1 and para_break > start:
                    end = para_break
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - overlap

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
