"""
Simple In-Memory Implementation

In-memory implementations of memory systems for development.
Can be replaced with database-backed versions for production.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import uuid

from .interface import (
    IEpisodicMemory,
    ISemanticMemory,
    IWorkingMemory,
    Experiment,
    Principle,
)


class SimpleEpisodicMemory(IEpisodicMemory):
    """
    Simple in-memory episodic memory

    Stores experiments in a list.
    For production, use PostgreSQL + vector DB.
    """

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        logger.info("Initialized SimpleEpisodicMemory")

    async def save_experiment(self, experiment: Experiment) -> str:
        """Save an experiment"""
        self.experiments[experiment.id] = experiment
        logger.debug(f"Saved experiment: {experiment.id}")
        return experiment.id

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)

    async def query_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Experiment]:
        """Query experiments with filters"""
        results = list(self.experiments.values())

        if filters:
            # Filter by success
            if "success" in filters:
                results = [e for e in results if e.success == filters["success"]]

            # Filter by min_sharpe
            if "min_sharpe" in filters:
                results = [
                    e for e in results
                    if e.results.get("sharpe_ratio", 0) >= filters["min_sharpe"]
                ]

            # Filter by market regime
            if "market_regime" in filters:
                results = [
                    e for e in results
                    if e.market_conditions.get("regime") == filters["market_regime"]
                ]

            # Filter by symbol
            if "symbol" in filters:
                results = [
                    e for e in results
                    if e.hypothesis.get("symbol") == filters["symbol"]
                ]

        # Sort by timestamp (most recent first)
        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[:limit]

    async def find_similar_experiments(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Experiment]:
        """
        Find similar experiments using simple text matching

        For production, use vector embeddings and semantic search.
        """
        query_lower = query.lower()
        results = []

        for exp in self.experiments.values():
            # Calculate similarity score (simple keyword matching)
            score = 0

            # Check hypothesis concept
            if "concept" in exp.hypothesis:
                concept = exp.hypothesis["concept"].lower()
                if any(word in concept for word in query_lower.split()):
                    score += 3

            # Check strategy name
            if "name" in exp.hypothesis:
                name = exp.hypothesis["name"].lower()
                if any(word in name for word in query_lower.split()):
                    score += 2

            # Check indicators
            if "indicators" in exp.hypothesis:
                indicators = " ".join(exp.hypothesis["indicators"]).lower()
                if any(word in indicators for word in query_lower.split()):
                    score += 1

            if score > 0:
                results.append((exp, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return [exp for exp, score in results[:top_k]]

    async def get_successful_experiments(
        self,
        min_sharpe: float = 1.5,
        min_trades: int = 30,
        limit: int = 20
    ) -> List[Experiment]:
        """Get successful experiments"""
        return await self.query_experiments(
            filters={
                "success": True,
                "min_sharpe": min_sharpe,
            },
            limit=limit
        )


class SimpleSemanticMemory(ISemanticMemory):
    """
    Simple in-memory semantic memory

    Stores learned principles.
    For production, use knowledge graph.
    """

    def __init__(self):
        self.principles: Dict[str, Principle] = {}
        logger.info("Initialized SimpleSemanticMemory")

    async def save_principle(self, principle: Principle) -> str:
        """Save a principle"""
        self.principles[principle.id] = principle
        logger.debug(f"Saved principle: {principle.text}")
        return principle.id

    async def get_principle(self, principle_id: str) -> Optional[Principle]:
        """Get principle by ID"""
        return self.principles.get(principle_id)

    async def query_principles(
        self,
        context: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[Principle]:
        """Query relevant principles"""
        results = list(self.principles.values())

        # Filter by confidence
        results = [p for p in results if p.confidence >= min_confidence]

        # Filter by context
        if context:
            if "market_regime" in context:
                regime = context["market_regime"]
                results = [
                    p for p in results
                    if p.context.get("market_regime") == regime
                    or p.context.get("market_regime") is None  # General principles
                ]

            if "strategy_type" in context:
                strat_type = context["strategy_type"]
                results = [
                    p for p in results
                    if p.context.get("strategy_type") == strat_type
                    or p.context.get("strategy_type") is None
                ]

        # Sort by confidence
        results.sort(key=lambda p: p.confidence, reverse=True)

        return results[:limit]

    async def update_principle_confidence(
        self,
        principle_id: str,
        new_evidence: str,
        outcome: bool
    ) -> None:
        """Update principle confidence based on new evidence"""
        if principle_id not in self.principles:
            logger.warning(f"Principle {principle_id} not found")
            return

        principle = self.principles[principle_id]

        # Add evidence
        principle.evidence.append(new_evidence)

        # Update confidence using simple Bayesian update
        # Positive outcome increases confidence, negative decreases it
        if outcome:
            principle.confidence = min(1.0, principle.confidence * 1.1)
        else:
            principle.confidence = max(0.0, principle.confidence * 0.9)

        logger.debug(
            f"Updated principle confidence: {principle.text} -> {principle.confidence:.2f}"
        )


class SimpleWorkingMemory(IWorkingMemory):
    """
    Simple in-memory working memory

    Stores current research session state.
    """

    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        logger.info("Initialized SimpleWorkingMemory")

    async def set_context(self, key: str, value: Any) -> None:
        """Set context variable"""
        self.context[key] = value
        logger.debug(f"Set context: {key} = {value}")

    async def get_context(self, key: str) -> Optional[Any]:
        """Get context variable"""
        return self.context.get(key)

    async def add_to_history(self, event: Dict[str, Any]) -> None:
        """Add event to history"""
        event["timestamp"] = datetime.now()
        self.history.append(event)

        # Keep last 1000 events
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    async def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events"""
        return self.history[-limit:]

    async def clear(self) -> None:
        """Clear working memory"""
        self.context.clear()
        self.history.clear()
        logger.info("Cleared working memory")


logger.info("Simple memory implementations loaded")
