"""
Vector store implementations for knowledge retrieval
"""
from typing import Any, Optional
from pathlib import Path
import uuid
from loguru import logger

from shared.models.base import IVectorStore, SearchResult, KnowledgeChunk
from shared.config.settings import settings


class ChromaVectorStore(IVectorStore[KnowledgeChunk]):
    """ChromaDB vector store implementation"""

    def __init__(
        self,
        collection_name: str = "knowledge_chunks",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize ChromaDB vector store

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence
        """
        import chromadb
        from chromadb.config import Settings

        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma.chroma_persist_directory

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize client
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized ChromaDB vector store: {collection_name}")

    async def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any]
    ) -> None:
        """
        Add a vector with metadata

        Args:
            id: Unique identifier
            embedding: Embedding vector
            metadata: Metadata dict
        """
        try:
            # Extract text content for storage
            content = metadata.get("content", "")

            self.collection.add(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[content]
            )

            logger.debug(f"Added vector: {id}")

        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            raise

    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        """
        Add multiple vectors

        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
        """
        if not ids or len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError("ids, embeddings, and metadatas must have same length")

        try:
            # Extract documents from metadata
            documents = [m.get("content", "") for m in metadatas]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"Added {len(ids)} vectors in batch")

        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            raise

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Search for similar vectors

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters

        Returns:
            List of search results
        """
        try:
            # Build where clause from filters
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )

            search_results = []

            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    # Convert distance to similarity score (cosine similarity)
                    score = 1.0 - distance

                    # Reconstruct KnowledgeChunk from metadata
                    chunk = self._metadata_to_chunk(chunk_id, metadata)

                    search_results.append(SearchResult(
                        item=chunk,
                        score=score,
                        metadata={"distance": distance}
                    ))

            logger.info(f"Search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def delete(self, id: str) -> None:
        """
        Delete a vector

        Args:
            id: Vector ID to delete
        """
        try:
            self.collection.delete(ids=[id])
            logger.debug(f"Deleted vector: {id}")

        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            raise

    async def get(self, id: str) -> Optional[KnowledgeChunk]:
        """
        Get item by ID

        Args:
            id: Item ID

        Returns:
            KnowledgeChunk if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[id])

            if result["ids"]:
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                return self._metadata_to_chunk(id, metadata)

            return None

        except Exception as e:
            logger.error(f"Failed to get item: {e}")
            raise

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        where: dict[str, Any] = {}

        for key, value in filters.items():
            if isinstance(value, list):
                # Handle list filters (e.g., asset_classes)
                where[key] = {"$in": value}
            else:
                where[key] = value

        return where

    def _metadata_to_chunk(self, chunk_id: str, metadata: dict[str, Any]) -> KnowledgeChunk:
        """Convert metadata dict back to KnowledgeChunk"""
        from datetime import datetime

        # Handle datetime fields
        created_at = metadata.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        last_updated = metadata.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)

        return KnowledgeChunk(
            id=chunk_id,
            source_book=metadata.get("source_book", ""),
            source_chapter=metadata.get("source_chapter", ""),
            source_page=metadata.get("source_page"),
            chunk_type=metadata.get("chunk_type", "concept"),
            hierarchy_level=metadata.get("hierarchy_level", 1),
            content=metadata.get("content", ""),
            embedding=metadata.get("embedding", []),
            code_embedding=metadata.get("code_embedding"),
            related_chunks=metadata.get("related_chunks", []),
            prerequisites=metadata.get("prerequisites", []),
            asset_classes=metadata.get("asset_classes", []),
            strategy_types=metadata.get("strategy_types", []),
            applicable_regimes=metadata.get("applicable_regimes", []),
            tags=metadata.get("tags", []),
            created_at=created_at or datetime.utcnow(),
            last_updated=last_updated or datetime.utcnow(),
            version=metadata.get("version", 1),
            metadata=metadata.get("extra_metadata", {})
        )

    def reset(self) -> None:
        """Reset the collection (delete all data)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning(f"Reset collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise

    def count(self) -> int:
        """Get number of items in collection"""
        return self.collection.count()


class FAISSVectorStore(IVectorStore[KnowledgeChunk]):
    """FAISS vector store implementation (for high-performance search)"""

    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "Flat",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize FAISS vector store

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type (Flat, IVF, HNSW)
            persist_directory: Directory for persistence
        """
        import faiss
        import pickle

        self.dimension = dimension
        self.index_type = index_type
        self.persist_directory = persist_directory or "./data/faiss"

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Storage for metadata and IDs
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}
        self.metadata_store: dict[str, dict[str, Any]] = {}
        self.next_idx = 0

        logger.info(f"Initialized FAISS vector store: {index_type}, dim={dimension}")

    async def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any]
    ) -> None:
        """Add a vector with metadata"""
        import numpy as np

        try:
            # Add to index
            vector = np.array([embedding], dtype=np.float32)
            self.index.add(vector)

            # Store mappings
            idx = self.next_idx
            self.id_to_idx[id] = idx
            self.idx_to_id[idx] = id
            self.metadata_store[id] = metadata

            self.next_idx += 1

            logger.debug(f"Added vector: {id}")

        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            raise

    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        """Add multiple vectors"""
        import numpy as np

        if not ids or len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError("ids, embeddings, and metadatas must have same length")

        try:
            # Add to index
            vectors = np.array(embeddings, dtype=np.float32)
            self.index.add(vectors)

            # Store mappings
            for i, chunk_id in enumerate(ids):
                idx = self.next_idx + i
                self.id_to_idx[chunk_id] = idx
                self.idx_to_id[idx] = chunk_id
                self.metadata_store[chunk_id] = metadatas[i]

            self.next_idx += len(ids)

            logger.info(f"Added {len(ids)} vectors in batch")

        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            raise

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[SearchResult[KnowledgeChunk]]:
        """Search for similar vectors"""
        import numpy as np

        try:
            # Search
            query = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query, top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # No result
                    continue

                chunk_id = self.idx_to_id.get(int(idx))
                if not chunk_id:
                    continue

                metadata = self.metadata_store.get(chunk_id, {})

                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue

                chunk = ChromaVectorStore._metadata_to_chunk(None, chunk_id, metadata)  # type: ignore
                distance = float(distances[0][i])
                score = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity

                results.append(SearchResult(
                    item=chunk,
                    score=score,
                    metadata={"distance": distance}
                ))

            logger.info(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def delete(self, id: str) -> None:
        """Delete a vector (not supported efficiently in FAISS)"""
        logger.warning("FAISS does not support efficient deletion")
        # Remove from metadata
        if id in self.id_to_idx:
            del self.metadata_store[id]
            # Note: Can't actually remove from index without rebuilding

    async def get(self, id: str) -> Optional[KnowledgeChunk]:
        """Get item by ID"""
        metadata = self.metadata_store.get(id)
        if metadata:
            return ChromaVectorStore._metadata_to_chunk(None, id, metadata)  # type: ignore
        return None

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def save(self, path: Optional[str] = None) -> None:
        """Save index and metadata to disk"""
        import faiss
        import pickle

        path = path or self.persist_directory

        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")

        # Save metadata
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump({
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "metadata_store": self.metadata_store,
                "next_idx": self.next_idx
            }, f)

        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load index and metadata from disk"""
        import faiss
        import pickle

        path = path or self.persist_directory

        # Load FAISS index
        self.index = faiss.read_index(f"{path}/index.faiss")

        # Load metadata
        with open(f"{path}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.id_to_idx = data["id_to_idx"]
            self.idx_to_id = data["idx_to_id"]
            self.metadata_store = data["metadata_store"]
            self.next_idx = data["next_idx"]

        logger.info(f"Loaded FAISS index from {path}")
