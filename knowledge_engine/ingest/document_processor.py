"""
Document processing for PDFs, EPUBs, and other formats
"""
from typing import Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from loguru import logger

from shared.models.base import IDocumentProcessor


class PDFProcessor(IDocumentProcessor):
    """PDF document processor using PyMuPDF"""

    def __init__(self):
        """Initialize PDF processor"""
        logger.info("Initialized PDF processor")

    async def extract_text(self, file_path: str) -> str:
        """
        Extract raw text from PDF

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            text = ""

            for page in doc:
                text += page.get_text()

            doc.close()

            logger.info(f"Extracted {len(text)} characters from {file_path}")
            return text

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise

    async def extract_structured(self, file_path: str) -> dict[str, Any]:
        """
        Extract structured data from PDF (chapters, sections, etc.)

        Args:
            file_path: Path to PDF file

        Returns:
            Structured document data
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            structured_data: dict[str, Any] = {
                "title": "",
                "author": "",
                "num_pages": len(doc),
                "chapters": [],
                "sections": [],
                "pages": []
            }

            # Extract metadata
            metadata = doc.metadata
            structured_data["title"] = metadata.get("title", "")
            structured_data["author"] = metadata.get("author", "")

            # Extract Table of Contents
            toc = doc.get_toc()
            current_chapter = None

            for level, title, page_num in toc:
                if level == 1:  # Chapter
                    current_chapter = {
                        "title": title,
                        "page": page_num,
                        "sections": []
                    }
                    structured_data["chapters"].append(current_chapter)
                elif level == 2 and current_chapter:  # Section
                    current_chapter["sections"].append({
                        "title": title,
                        "page": page_num
                    })

            # Extract page-by-page content
            for page_num, page in enumerate(doc, start=1):
                page_data = {
                    "page_number": page_num,
                    "text": page.get_text(),
                    "images": len(page.get_images()),
                    "tables": self._detect_tables(page)
                }
                structured_data["pages"].append(page_data)

            doc.close()

            logger.info(f"Extracted structured data from {file_path}: "
                       f"{len(structured_data['chapters'])} chapters, "
                       f"{len(structured_data['pages'])} pages")

            return structured_data

        except Exception as e:
            logger.error(f"Failed to extract structured data: {e}")
            raise

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract document metadata

        Args:
            file_path: Path to PDF file

        Returns:
            Document metadata
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            metadata = doc.metadata

            result = {
                "format": "PDF",
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "num_pages": len(doc),
                "file_size": Path(file_path).stat().st_size
            }

            doc.close()

            logger.info(f"Extracted metadata from {file_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise

    def _detect_tables(self, page: Any) -> int:
        """
        Detect number of tables in a page (simple heuristic)

        Args:
            page: PyMuPDF page object

        Returns:
            Estimated number of tables
        """
        # Simple heuristic: count number of table-like structures
        # This is a placeholder - more sophisticated methods exist
        text = page.get_text()
        # Count lines with multiple tab or pipe characters
        table_lines = [line for line in text.split('\n')
                      if line.count('\t') > 2 or line.count('|') > 2]
        return len(table_lines) // 3  # Rough estimate


class TextProcessor(IDocumentProcessor):
    """Plain text document processor"""

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize text processor

        Args:
            encoding: Text encoding
        """
        self.encoding = encoding
        logger.info("Initialized text processor")

    async def extract_text(self, file_path: str) -> str:
        """Extract text from file"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                text = f.read()

            logger.info(f"Extracted {len(text)} characters from {file_path}")
            return text

        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            raise

    async def extract_structured(self, file_path: str) -> dict[str, Any]:
        """Extract structured data from text file"""
        text = await self.extract_text(file_path)

        # Simple structure extraction based on markdown-like headers
        lines = text.split('\n')
        sections = []
        current_section = None

        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Header line
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()

                section = {
                    "level": level,
                    "title": title,
                    "line": i + 1,
                    "content": []
                }

                if level == 1:
                    sections.append(section)
                    current_section = section
                elif current_section:
                    current_section["content"].append(section)

        return {
            "format": "text",
            "sections": sections,
            "total_lines": len(lines),
            "total_chars": len(text)
        }

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract file metadata"""
        path = Path(file_path)
        stat = path.stat()

        return {
            "format": "text",
            "file_name": path.name,
            "file_size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime
        }


class DocumentProcessorFactory:
    """Factory for creating document processors"""

    _processors: dict[str, type[IDocumentProcessor]] = {
        ".pdf": PDFProcessor,
        ".txt": TextProcessor,
        ".md": TextProcessor,
    }

    @classmethod
    def create_processor(cls, file_path: str) -> IDocumentProcessor:
        """
        Create appropriate processor for file type

        Args:
            file_path: Path to document file

        Returns:
            Document processor instance

        Raises:
            ValueError: If file type not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        processor_class = cls._processors.get(extension)
        if not processor_class:
            raise ValueError(f"Unsupported file type: {extension}")

        return processor_class()

    @classmethod
    def register_processor(
        cls,
        extension: str,
        processor_class: type[IDocumentProcessor]
    ) -> None:
        """
        Register a custom document processor

        Args:
            extension: File extension (e.g., ".epub")
            processor_class: Processor class
        """
        cls._processors[extension.lower()] = processor_class
        logger.info(f"Registered processor for {extension}")


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies"""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Split text into chunks

        Args:
            text: Text to chunk
            metadata: Optional metadata for chunks

        Returns:
            List of chunk dictionaries
        """
        pass


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap"""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize fixed-size chunking

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

    def chunk(self, text: str, metadata: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Split text into fixed-size chunks"""
        chunks = []

        # Split by separator first
        parts = text.split(self.separator)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) < self.chunk_size:
                current_chunk += part + self.separator
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata or {}
                    })

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + part + self.separator
                else:
                    current_chunk = part + self.separator

        # Add remaining chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata or {}
            })

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Semantic-based chunking using sentence similarity"""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize semantic chunking

        Args:
            max_chunk_size: Maximum chunk size
            similarity_threshold: Similarity threshold for grouping sentences
        """
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Split text into semantically coherent chunks"""
        # This is a simplified version
        # Full implementation would use sentence embeddings for similarity

        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata or {}
                    })
                current_chunk = sentence + " "

        # Add remaining
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata or {}
            })

        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
