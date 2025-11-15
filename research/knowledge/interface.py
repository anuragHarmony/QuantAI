"""
Knowledge Graph Interface

Abstract interface for knowledge graph operations following SOLID principles.
This allows swapping between Neo4j, in-memory graphs, or other implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ConceptType(Enum):
    """Types of concepts in the knowledge graph"""
    INDICATOR = "indicator"
    STRATEGY = "strategy"
    MARKET_REGIME = "market_regime"
    ASSET = "asset"
    TIMEFRAME = "timeframe"
    RISK_METRIC = "risk_metric"
    PRINCIPLE = "principle"


@dataclass
class Concept:
    """Base concept in the knowledge graph"""
    id: str
    name: str
    concept_type: ConceptType
    description: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    # Metrics
    success_rate: float = 0.0
    tested_count: int = 0
    avg_sharpe: float = 0.0
    last_tested: Optional[datetime] = None


@dataclass
class Relationship:
    """Relationship between concepts"""
    from_concept_id: str
    to_concept_id: str
    relationship_type: str  # USES, WORKS_IN, SIMILAR_TO, etc.
    properties: Dict[str, Any]
    strength: float = 1.0  # Relationship strength (0-1)


class IKnowledgeGraph(ABC):
    """
    Interface for knowledge graph operations

    SOLID Principles:
    - Single Responsibility: Only handles knowledge storage/retrieval
    - Open/Closed: Can extend with new concept types without modifying
    - Liskov Substitution: Any implementation can be swapped
    - Interface Segregation: Focused on knowledge operations only
    - Dependency Inversion: Depends on this abstraction, not concrete DB
    """

    @abstractmethod
    async def add_concept(self, concept: Concept) -> str:
        """Add a concept to the knowledge graph"""
        pass

    @abstractmethod
    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        pass

    @abstractmethod
    async def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between concepts"""
        pass

    @abstractmethod
    async def query_concepts(
        self,
        concept_type: Optional[ConceptType] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Concept]:
        """Query concepts by type and filters"""
        pass

    @abstractmethod
    async def find_related_concepts(
        self,
        concept_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Concept]:
        """Find concepts related to a given concept"""
        pass

    @abstractmethod
    async def find_similar_concepts(
        self,
        concept_id: str,
        top_k: int = 10
    ) -> List[Concept]:
        """Find similar concepts (by SIMILAR_TO relationship)"""
        pass

    @abstractmethod
    async def update_concept_metrics(
        self,
        concept_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Update concept performance metrics"""
        pass

    @abstractmethod
    async def query_by_cypher(self, query: str) -> List[Dict[str, Any]]:
        """Execute raw Cypher query (for Neo4j)"""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        pass


class IConceptEmbedder(ABC):
    """Interface for generating embeddings of concepts"""

    @abstractmethod
    async def embed_concept(self, concept: Concept) -> List[float]:
        """Generate embedding vector for a concept"""
        pass

    @abstractmethod
    async def find_similar_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10
    ) -> List[str]:
        """Find similar concepts by embedding similarity"""
        pass
