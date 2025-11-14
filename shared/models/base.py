"""
Base abstract interfaces following SOLID principles
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic, AsyncIterator
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# Type variables for generic interfaces
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class ChunkType(str, Enum):
    """Types of knowledge chunks"""
    CONCEPT = "concept"
    STRATEGY = "strategy"
    FORMULA = "formula"
    EXAMPLE = "example"
    CODE = "code"
    EXPERIENCE = "experience"


class MarketRegime(str, Enum):
    """Market regime types"""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


class AssetClass(str, Enum):
    """Asset class types"""
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"


class KnowledgeChunk(BaseModel):
    """Structured knowledge chunk"""
    id: str = Field(description="Unique chunk identifier")
    source_book: str = Field(description="Source book name")
    source_chapter: str = Field(description="Chapter name")
    source_page: Optional[int] = Field(default=None, description="Page number")
    chunk_type: ChunkType = Field(description="Type of chunk")
    hierarchy_level: int = Field(ge=1, le=4, description="Hierarchy level (1=broad, 4=detail)")
    content: str = Field(description="Chunk content")
    embedding: list[float] = Field(default_factory=list, description="Text embedding")
    code_embedding: Optional[list[float]] = Field(default=None, description="Code embedding if applicable")
    related_chunks: list[str] = Field(default_factory=list, description="Related chunk IDs")
    prerequisites: list[str] = Field(default_factory=list, description="Prerequisite chunk IDs")
    asset_classes: list[AssetClass] = Field(default_factory=list, description="Applicable asset classes")
    strategy_types: list[str] = Field(default_factory=list, description="Strategy types")
    applicable_regimes: list[MarketRegime] = Field(default_factory=list, description="Applicable market regimes")
    tags: list[str] = Field(default_factory=list, description="Free-form tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, description="Version number")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResult(BaseModel, Generic[T]):
    """Generic search result with relevance score"""
    item: T = Field(description="The result item")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalContext(BaseModel):
    """Context assembled from retrieval"""
    chunks: list[KnowledgeChunk] = Field(description="Retrieved knowledge chunks")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    coverage: float = Field(ge=0.0, le=1.0, description="Coverage score")
    sources: list[str] = Field(default_factory=list, description="Unique sources")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ===== Abstract Base Interfaces =====


class IDataFetcher(ABC, Generic[T]):
    """Abstract interface for data fetching"""

    @abstractmethod
    async def fetch(self, source: str, **kwargs: Any) -> T:
        """Fetch data from a source"""
        pass

    @abstractmethod
    async def fetch_batch(self, sources: list[str], **kwargs: Any) -> list[T]:
        """Fetch multiple sources in batch"""
        pass

    @abstractmethod
    async def validate_source(self, source: str) -> bool:
        """Validate if source is accessible"""
        pass


class IDocumentProcessor(ABC):
    """Abstract interface for document processing"""

    @abstractmethod
    async def extract_text(self, file_path: str) -> str:
        """Extract raw text from document"""
        pass

    @abstractmethod
    async def extract_structured(self, file_path: str) -> dict[str, Any]:
        """Extract structured data (chapters, sections, etc.)"""
        pass

    @abstractmethod
    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract document metadata"""
        pass


class IEmbeddingProvider(ABC):
    """Abstract interface for embedding generation"""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass


class IVectorStore(ABC, Generic[T]):
    """Abstract interface for vector storage"""

    @abstractmethod
    async def add(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Add a vector with metadata"""
        pass

    @abstractmethod
    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        """Add multiple vectors"""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[SearchResult[T]]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete a vector"""
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get item by ID"""
        pass


class IKnowledgeGraph(ABC):
    """Abstract interface for knowledge graph operations"""

    @abstractmethod
    async def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any]
    ) -> None:
        """Add a node to the graph"""
        pass

    @abstractmethod
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict[str, Any]] = None
    ) -> None:
        """Add an edge between nodes"""
        pass

    @abstractmethod
    async def find_related(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """Find related nodes"""
        pass

    @abstractmethod
    async def traverse(
        self,
        start_id: str,
        query: str
    ) -> list[dict[str, Any]]:
        """Execute a graph traversal query"""
        pass


class IRetriever(ABC):
    """Abstract interface for semantic retrieval"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[SearchResult[KnowledgeChunk]]:
        """Retrieve relevant knowledge chunks"""
        pass

    @abstractmethod
    async def assemble_context(
        self,
        query: str,
        max_tokens: int = 4000
    ) -> RetrievalContext:
        """Assemble context within token budget"""
        pass


class ILLMProvider(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs: Any
    ) -> str:
        """Generate completion"""
        pass

    @abstractmethod
    async def complete_with_functions(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        **kwargs: Any
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """Generate completion with function calling"""
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream completion chunks"""
        pass


class ICacheProvider(ABC, Generic[K, V]):
    """Abstract interface for caching"""

    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        pass

    @abstractmethod
    async def delete(self, key: K) -> None:
        """Delete from cache"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear entire cache"""
        pass


class IRepository(ABC, Generic[T]):
    """Abstract repository pattern for data access"""

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity"""
        pass

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete entity"""
        pass

    @abstractmethod
    async def find(self, filters: dict[str, Any], limit: int = 100) -> list[T]:
        """Find entities matching filters"""
        pass
