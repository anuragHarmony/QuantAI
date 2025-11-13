"""Vector store for semantic search using ChromaDB."""

from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from shared.models.knowledge import KnowledgeChunk, QueryResult
from shared.config.settings import settings


class VectorStore:
    """Manages vector embeddings and semantic search using ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection
            embedding_model: Name of the embedding model
        """
        if chromadb is None:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")

        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model_name = embedding_model or settings.embedding_model

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Quant knowledge base"}
        )

        logger.info(f"Vector store initialized with {self.collection.count()} documents")

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """
        Add knowledge chunks to the vector store.

        Args:
            chunks: List of KnowledgeChunk objects
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            # Generate embedding if not present
            if chunk.embedding is None:
                embedding = self.embedding_model.encode(chunk.content).tolist()
            else:
                embedding = chunk.embedding

            ids.append(chunk.id)
            documents.append(chunk.content)
            embeddings.append(embedding)

            # Prepare metadata (ChromaDB requires flat dict)
            metadata = {
                "source_book": chunk.source_book,
                "source_chapter": chunk.source_chapter or "",
                "source_page": chunk.source_page or 0,
                "chunk_type": chunk.chunk_type,
                "hierarchy_level": chunk.hierarchy_level,
                "asset_class": ",".join(chunk.asset_class) if chunk.asset_class else "",
                "strategy_type": ",".join(chunk.strategy_type) if chunk.strategy_type else "",
                "tags": ",".join(chunk.tags) if chunk.tags else "",
            }
            metadatas.append(metadata)

        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """
        Search for relevant knowledge chunks.

        Args:
            query: Search query
            n_results: Number of results to return
            filters: Metadata filters (e.g., {"chunk_type": "strategy"})

        Returns:
            List of QueryResult objects
        """
        logger.info(f"Searching for: '{query}' (n_results={n_results})")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build where clause for filtering
        where_clause = None
        if filters:
            where_clause = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    # Handle list filters (e.g., multiple asset classes)
                    # Note: ChromaDB has limited support for complex queries
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )

            # Convert to QueryResult objects
            query_results = []

            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    # Reconstruct KnowledgeChunk from stored data
                    metadata = results['metadatas'][0][i]

                    chunk = KnowledgeChunk(
                        id=results['ids'][0][i],
                        source_book=metadata.get('source_book', ''),
                        source_chapter=metadata.get('source_chapter') or None,
                        source_page=metadata.get('source_page') or None,
                        chunk_type=metadata.get('chunk_type', 'concept'),
                        hierarchy_level=int(metadata.get('hierarchy_level', 2)),
                        content=results['documents'][0][i],
                        asset_class=metadata.get('asset_class', '').split(',') if metadata.get('asset_class') else [],
                        strategy_type=metadata.get('strategy_type', '').split(',') if metadata.get('strategy_type') else [],
                        tags=metadata.get('tags', '').split(',') if metadata.get('tags') else []
                    )

                    # Distance to similarity score (ChromaDB returns distances)
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    similarity = 1 / (1 + distance)  # Convert distance to similarity

                    query_result = QueryResult(
                        chunk=chunk,
                        relevance_score=similarity,
                        source="vector_search"
                    )
                    query_results.append(query_result)

            logger.info(f"Found {len(query_results)} results")
            return query_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def delete_by_source(self, source_book: str) -> None:
        """
        Delete all chunks from a specific source book.

        Args:
            source_book: Name of the source book
        """
        logger.info(f"Deleting chunks from: {source_book}")

        try:
            self.collection.delete(
                where={"source_book": source_book}
            )
            logger.info(f"Deleted chunks from {source_book}")
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise

    def count(self) -> int:
        """Get the total number of chunks in the vector store."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all data from the vector store."""
        logger.warning("Resetting vector store - all data will be deleted!")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Quant knowledge base"}
        )
        logger.info("Vector store reset complete")
