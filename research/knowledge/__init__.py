"""
Knowledge Graph Module

Stores and retrieves quantitative trading knowledge including:
- Indicators (RSI, MACD, etc.)
- Strategies (Mean Reversion, Momentum, etc.)
- Market Regimes (Trending, Ranging, etc.)
- Principles (learned facts about what works)
- Relationships between all concepts

Can be backed by Neo4j (production) or in-memory (development).
"""

from .interface import (
    IKnowledgeGraph,
    IConceptEmbedder,
    Concept,
    ConceptType,
    Relationship,
)

from .memory_graph import InMemoryKnowledgeGraph

__all__ = [
    "IKnowledgeGraph",
    "IConceptEmbedder",
    "Concept",
    "ConceptType",
    "Relationship",
    "InMemoryKnowledgeGraph",
]
