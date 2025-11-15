"""
Memory System Interfaces

Multi-level memory for the AI research agent:
1. Episodic Memory: Past experiments (what was tried, results)
2. Semantic Memory: Learned principles and patterns
3. Working Memory: Current research session context
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass
class Experiment:
    """Record of a past experiment"""
    id: str
    timestamp: datetime
    hypothesis: Dict[str, Any]  # The strategy hypothesis tested
    results: Dict[str, Any]  # Backtest results
    insights: Dict[str, Any]  # What was learned
    market_conditions: Dict[str, Any]  # Market regime, volatility, etc.
    strategy_code: str
    success: bool  # Whether it met criteria


@dataclass
class Principle:
    """Learned principle or pattern"""
    id: str
    text: str  # The principle statement
    confidence: float  # 0-1
    evidence: List[str]  # Supporting evidence (experiment IDs)
    discovered_at: datetime
    context: Dict[str, Any]  # When/where this applies


class IEpisodicMemory(ABC):
    """
    Interface for episodic memory

    Stores past experiments with full context.
    Single Responsibility: Remember what was tried
    """

    @abstractmethod
    async def save_experiment(self, experiment: Experiment) -> str:
        """Save an experiment to memory"""
        pass

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID"""
        pass

    @abstractmethod
    async def query_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Experiment]:
        """
        Query experiments

        Filters can include:
        - success: bool
        - min_sharpe: float
        - market_regime: str
        - symbol: str
        - etc.
        """
        pass

    @abstractmethod
    async def find_similar_experiments(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Experiment]:
        """Find experiments similar to query using semantic search"""
        pass

    @abstractmethod
    async def get_successful_experiments(
        self,
        min_sharpe: float = 1.5,
        min_trades: int = 30,
        limit: int = 20
    ) -> List[Experiment]:
        """Get successful experiments meeting criteria"""
        pass


class ISemanticMemory(ABC):
    """
    Interface for semantic memory

    Stores learned principles and patterns.
    Single Responsibility: Remember what works/doesn't work
    """

    @abstractmethod
    async def save_principle(self, principle: Principle) -> str:
        """Save a learned principle"""
        pass

    @abstractmethod
    async def get_principle(self, principle_id: str) -> Optional[Principle]:
        """Get a principle by ID"""
        pass

    @abstractmethod
    async def query_principles(
        self,
        context: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[Principle]:
        """
        Query relevant principles

        Context can include:
        - market_regime: str
        - asset: str
        - strategy_type: str
        """
        pass

    @abstractmethod
    async def update_principle_confidence(
        self,
        principle_id: str,
        new_evidence: str,
        outcome: bool
    ) -> None:
        """Update principle confidence based on new evidence"""
        pass


class IWorkingMemory(ABC):
    """
    Interface for working memory

    Stores current research session context.
    Single Responsibility: Track current research progress
    """

    @abstractmethod
    async def set_context(self, key: str, value: Any) -> None:
        """Set a context variable"""
        pass

    @abstractmethod
    async def get_context(self, key: str) -> Optional[Any]:
        """Get a context variable"""
        pass

    @abstractmethod
    async def add_to_history(self, event: Dict[str, Any]) -> None:
        """Add event to session history"""
        pass

    @abstractmethod
    async def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session events"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear working memory (new session)"""
        pass
