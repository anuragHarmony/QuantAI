"""Script to index sample documents into the knowledge base."""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_engine.knowledge_engine import KnowledgeEngine


def index_text_file(engine: KnowledgeEngine, file_path: Path) -> int:
    """
    Index a text file by converting it to pseudo-PDF format.

    Args:
        engine: KnowledgeEngine instance
        file_path: Path to text file

    Returns:
        Number of chunks added
    """
    logger.info(f"Indexing text file: {file_path}")

    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Process using the document processor and knowledge extractor
    book_name = file_path.stem

    # Chunk the text
    chunks = engine.document_processor.chunk_text(content, chunk_size=1000, overlap=200)

    logger.info(f"Processing {len(chunks)} chunks from {book_name}")

    # Extract knowledge from each chunk
    all_knowledge_chunks = []

    for i, text_chunk in enumerate(chunks):
        knowledge_chunks = engine.knowledge_extractor.extract_knowledge_chunks(
            text=text_chunk,
            source_book=book_name,
            source_chapter=f"Chunk {i+1}"
        )
        all_knowledge_chunks.extend(knowledge_chunks)

    logger.info(f"Extracted {len(all_knowledge_chunks)} knowledge chunks from {book_name}")

    # Add to vector store
    if all_knowledge_chunks:
        engine.vector_store.add_chunks(all_knowledge_chunks)

    return len(all_knowledge_chunks)


def main():
    """Index all sample documents."""
    logger.info("Starting document indexing")

    # Initialize knowledge engine (without OpenAI for now)
    engine = KnowledgeEngine(use_llm_extraction=False)

    # Get sample documents directory
    docs_dir = Path(__file__).parent / "sample_data" / "documents"

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return

    # Find all text files
    text_files = list(docs_dir.glob("*.txt"))

    if not text_files:
        logger.warning("No text files found to index")
        return

    logger.info(f"Found {len(text_files)} documents to index")

    # Index each document
    total_chunks = 0
    for text_file in text_files:
        try:
            num_chunks = index_text_file(engine, text_file)
            total_chunks += num_chunks
            logger.info(f"✓ Indexed {text_file.name}: {num_chunks} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to index {text_file.name}: {e}")

    # Print statistics
    logger.info("=" * 60)
    logger.info("Indexing Complete!")
    logger.info(f"Total documents indexed: {len(text_files)}")
    logger.info(f"Total knowledge chunks: {total_chunks}")

    stats = engine.get_stats()
    logger.info(f"Vector store: {stats['vector_store']}")
    logger.info(f"Embedding model: {stats['embedding_model']}")
    logger.info(f"Total chunks in database: {stats['total_chunks']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    main()
