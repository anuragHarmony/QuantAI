"""Main knowledge engine orchestrator."""

from pathlib import Path
from typing import List, Optional
from loguru import logger

from knowledge_engine.ingest.document_processor import DocumentProcessor
from knowledge_engine.ingest.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.retrieval.vector_store import VectorStore
from knowledge_engine.retrieval.semantic_search import SemanticSearch
from shared.models.knowledge import KnowledgeChunk, QueryResult, RetrievalContext


class KnowledgeEngine:
    """Main orchestrator for the knowledge system."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        use_llm_extraction: bool = False
    ):
        """
        Initialize the knowledge engine.

        Args:
            openai_api_key: OpenAI API key for LLM extraction
            use_llm_extraction: Whether to use LLM for knowledge extraction
        """
        logger.info("Initializing Knowledge Engine")

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.knowledge_extractor = KnowledgeExtractor(api_key=openai_api_key)
        self.vector_store = VectorStore()
        self.semantic_search = SemanticSearch(vector_store=self.vector_store)

        self.use_llm_extraction = use_llm_extraction and openai_api_key is not None

        logger.info(f"Knowledge Engine initialized (LLM extraction: {self.use_llm_extraction})")

    def ingest_document(
        self,
        file_path: Path,
        book_name: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> int:
        """
        Ingest a document into the knowledge base.

        Args:
            file_path: Path to the document file
            book_name: Name of the book (uses filename if not provided)
            chunk_size: Size of text chunks
            overlap: Overlap between chunks

        Returns:
            Number of chunks added
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        book_name = book_name or file_path.stem
        logger.info(f"Ingesting document: {book_name} from {file_path}")

        # Extract text from PDF
        full_text = self.document_processor.extract_full_text(file_path)

        # Split into chunks
        text_chunks = self.document_processor.chunk_text(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        logger.info(f"Processing {len(text_chunks)} chunks from {book_name}")

        # Extract knowledge from each chunk
        all_knowledge_chunks = []

        for i, text_chunk in enumerate(text_chunks):
            logger.info(f"Extracting knowledge from chunk {i+1}/{len(text_chunks)}")

            # Extract knowledge chunks
            knowledge_chunks = self.knowledge_extractor.extract_knowledge_chunks(
                text=text_chunk,
                source_book=book_name,
                source_chapter=f"Chunk {i+1}"
            )

            all_knowledge_chunks.extend(knowledge_chunks)

        logger.info(f"Extracted {len(all_knowledge_chunks)} knowledge chunks")

        # Add to vector store
        if all_knowledge_chunks:
            self.vector_store.add_chunks(all_knowledge_chunks)

        logger.info(f"Successfully ingested {book_name}: {len(all_knowledge_chunks)} chunks added")

        return len(all_knowledge_chunks)

    def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[dict] = None
    ) -> List[QueryResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            max_results: Maximum number of results
            filters: Optional metadata filters

        Returns:
            List of QueryResult objects
        """
        return self.semantic_search.search(query, max_results, filters)

    def search_strategies(self, query: str, max_results: int = 5) -> List[QueryResult]:
        """Search for trading strategies."""
        return self.semantic_search.search_strategies(query, max_results)

    def search_concepts(self, query: str, max_results: int = 10) -> List[QueryResult]:
        """Search for concepts."""
        return self.semantic_search.search_concepts(query, max_results)

    def search_examples(self, query: str, max_results: int = 5) -> List[QueryResult]:
        """Search for examples."""
        return self.semantic_search.search_examples(query, max_results)

    def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        include_examples: bool = True
    ) -> RetrievalContext:
        """
        Assemble context for AI reasoning.

        Args:
            query: User query
            max_tokens: Maximum token budget
            include_examples: Whether to include examples

        Returns:
            RetrievalContext object
        """
        return self.semantic_search.assemble_context(
            query,
            max_tokens=max_tokens,
            include_examples=include_examples
        )

    def get_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        total_chunks = self.vector_store.count()

        return {
            "total_chunks": total_chunks,
            "vector_store": "ChromaDB",
            "embedding_model": self.vector_store.embedding_model_name
        }

    def reset(self) -> None:
        """Reset the knowledge base (delete all data)."""
        logger.warning("Resetting knowledge base")
        self.vector_store.reset()
        logger.info("Knowledge base reset complete")
