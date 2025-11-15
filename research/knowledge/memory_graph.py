"""
In-Memory Knowledge Graph Implementation

Fast in-memory implementation for development and testing.
Can be replaced with Neo4j for production without changing agent code.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import networkx as nx

from .interface import (
    IKnowledgeGraph,
    Concept,
    ConceptType,
    Relationship,
)


class InMemoryKnowledgeGraph(IKnowledgeGraph):
    """
    In-memory knowledge graph using NetworkX

    Advantages:
    - Fast for development/testing
    - No external dependencies
    - Easy to serialize/deserialize

    For production, swap with Neo4jKnowledgeGraph
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.concepts: Dict[str, Concept] = {}
        logger.info("Initialized InMemoryKnowledgeGraph")

    async def add_concept(self, concept: Concept) -> str:
        """Add a concept to the knowledge graph"""
        self.concepts[concept.id] = concept

        # Add node to graph
        self.graph.add_node(
            concept.id,
            name=concept.name,
            concept_type=concept.concept_type.value,
            **concept.properties
        )

        logger.debug(f"Added concept: {concept.name} ({concept.concept_type.value})")
        return concept.id

    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)

    async def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between concepts"""
        self.graph.add_edge(
            relationship.from_concept_id,
            relationship.to_concept_id,
            relationship_type=relationship.relationship_type,
            strength=relationship.strength,
            **relationship.properties
        )

        logger.debug(
            f"Added relationship: {relationship.from_concept_id} "
            f"-[{relationship.relationship_type}]-> "
            f"{relationship.to_concept_id}"
        )

    async def query_concepts(
        self,
        concept_type: Optional[ConceptType] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Concept]:
        """Query concepts by type and filters"""
        results = []

        for concept in self.concepts.values():
            # Filter by type
            if concept_type and concept.concept_type != concept_type:
                continue

            # Filter by properties
            if filters:
                matches = True
                for key, value in filters.items():
                    if key not in concept.properties:
                        matches = False
                        break

                    # Support operators
                    if isinstance(value, dict):
                        # e.g., {"$gt": 0.5}
                        prop_value = concept.properties[key]
                        for op, op_value in value.items():
                            if op == "$gt" and not (prop_value > op_value):
                                matches = False
                            elif op == "$gte" and not (prop_value >= op_value):
                                matches = False
                            elif op == "$lt" and not (prop_value < op_value):
                                matches = False
                            elif op == "$lte" and not (prop_value <= op_value):
                                matches = False
                    else:
                        if concept.properties[key] != value:
                            matches = False
                            break

                if not matches:
                    continue

            results.append(concept)

            if len(results) >= limit:
                break

        return results

    async def find_related_concepts(
        self,
        concept_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Concept]:
        """Find concepts related to a given concept"""
        if concept_id not in self.graph:
            return []

        related_ids = set()

        # BFS to find related concepts
        visited = set()
        queue = [(concept_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if depth > max_depth:
                continue

            if current_id in visited:
                continue

            visited.add(current_id)

            # Get neighbors
            for neighbor_id in self.graph.neighbors(current_id):
                # Filter by relationship type if specified
                if relationship_type:
                    edges = self.graph.get_edge_data(current_id, neighbor_id)
                    if not any(
                        e.get("relationship_type") == relationship_type
                        for e in edges.values()
                    ):
                        continue

                if neighbor_id != concept_id:  # Exclude self
                    related_ids.add(neighbor_id)

                if depth < max_depth:
                    queue.append((neighbor_id, depth + 1))

        # Return Concept objects
        return [
            self.concepts[cid]
            for cid in related_ids
            if cid in self.concepts
        ]

    async def find_similar_concepts(
        self,
        concept_id: str,
        top_k: int = 10
    ) -> List[Concept]:
        """Find similar concepts (by SIMILAR_TO relationship)"""
        return await self.find_related_concepts(
            concept_id,
            relationship_type="SIMILAR_TO",
            max_depth=1
        )

    async def update_concept_metrics(
        self,
        concept_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Update concept performance metrics"""
        if concept_id not in self.concepts:
            logger.warning(f"Concept {concept_id} not found")
            return

        concept = self.concepts[concept_id]

        # Update metrics
        if "success_rate" in metrics:
            concept.success_rate = metrics["success_rate"]
        if "tested_count" in metrics:
            concept.tested_count = metrics["tested_count"]
        if "avg_sharpe" in metrics:
            concept.avg_sharpe = metrics["avg_sharpe"]
        if "last_tested" in metrics:
            concept.last_tested = metrics["last_tested"]

        concept.updated_at = datetime.now()

        logger.debug(f"Updated metrics for concept: {concept.name}")

    async def query_by_cypher(self, query: str) -> List[Dict[str, Any]]:
        """Execute raw Cypher query (simulated for in-memory)"""
        logger.warning("Cypher queries not supported in InMemoryKnowledgeGraph")
        return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        concept_counts = {}
        for concept in self.concepts.values():
            ct = concept.concept_type.value
            concept_counts[ct] = concept_counts.get(ct, 0) + 1

        return {
            "total_concepts": len(self.concepts),
            "total_relationships": self.graph.number_of_edges(),
            "concept_counts": concept_counts,
            "avg_tested_count": sum(c.tested_count for c in self.concepts.values()) / len(self.concepts) if self.concepts else 0,
            "avg_success_rate": sum(c.success_rate for c in self.concepts.values()) / len(self.concepts) if self.concepts else 0,
        }

    # Helper methods for initialization

    async def seed_with_defaults(self) -> None:
        """Seed knowledge graph with default concepts"""
        logger.info("Seeding knowledge graph with default concepts...")

        # Add indicators
        indicators = [
            ("RSI", "Relative Strength Index", {"complexity": "low", "computational_cost": "low"}),
            ("MACD", "Moving Average Convergence Divergence", {"complexity": "medium", "computational_cost": "low"}),
            ("BollingerBands", "Bollinger Bands", {"complexity": "medium", "computational_cost": "low"}),
            ("EMA", "Exponential Moving Average", {"complexity": "low", "computational_cost": "low"}),
            ("SMA", "Simple Moving Average", {"complexity": "low", "computational_cost": "low"}),
            ("ADX", "Average Directional Index", {"complexity": "medium", "computational_cost": "medium"}),
            ("ATR", "Average True Range", {"complexity": "low", "computational_cost": "low"}),
            ("Stochastic", "Stochastic Oscillator", {"complexity": "medium", "computational_cost": "low"}),
            ("VWAP", "Volume Weighted Average Price", {"complexity": "medium", "computational_cost": "medium"}),
            ("OBV", "On Balance Volume", {"complexity": "low", "computational_cost": "low"}),
        ]

        for name, desc, props in indicators:
            concept = Concept(
                id=f"ind_{name.lower()}",
                name=name,
                concept_type=ConceptType.INDICATOR,
                description=desc,
                properties=props,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            await self.add_concept(concept)

        # Add market regimes
        regimes = [
            ("Trending", "Market shows clear directional movement"),
            ("Ranging", "Market oscillates within a range"),
            ("Volatile", "Market shows high volatility"),
            ("LowVolatility", "Market shows low volatility"),
        ]

        for name, desc in regimes:
            concept = Concept(
                id=f"regime_{name.lower()}",
                name=name,
                concept_type=ConceptType.MARKET_REGIME,
                description=desc,
                properties={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            await self.add_concept(concept)

        # Add strategy types
        strategies = [
            ("MeanReversion", "Strategies that exploit price reversions to mean"),
            ("Momentum", "Strategies that follow price trends"),
            ("Arbitrage", "Strategies that exploit price differences"),
            ("MarketMaking", "Strategies that provide liquidity"),
        ]

        for name, desc in strategies:
            concept = Concept(
                id=f"strat_{name.lower()}",
                name=name,
                concept_type=ConceptType.STRATEGY,
                description=desc,
                properties={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            await self.add_concept(concept)

        # Add relationships
        # Mean reversion works in ranging markets
        await self.add_relationship(Relationship(
            from_concept_id="strat_meanreversion",
            to_concept_id="regime_ranging",
            relationship_type="WORKS_IN",
            properties={},
            strength=0.8
        ))

        # Mean reversion uses RSI
        await self.add_relationship(Relationship(
            from_concept_id="strat_meanreversion",
            to_concept_id="ind_rsi",
            relationship_type="USES",
            properties={},
            strength=0.9
        ))

        # Mean reversion uses Bollinger Bands
        await self.add_relationship(Relationship(
            from_concept_id="strat_meanreversion",
            to_concept_id="ind_bollingerbands",
            relationship_type="USES",
            properties={},
            strength=0.9
        ))

        # Momentum works in trending markets
        await self.add_relationship(Relationship(
            from_concept_id="strat_momentum",
            to_concept_id="regime_trending",
            relationship_type="WORKS_IN",
            properties={},
            strength=0.9
        ))

        # Momentum uses EMA
        await self.add_relationship(Relationship(
            from_concept_id="strat_momentum",
            to_concept_id="ind_ema",
            relationship_type="USES",
            properties={},
            strength=0.8
        ))

        # Momentum uses ADX
        await self.add_relationship(Relationship(
            from_concept_id="strat_momentum",
            to_concept_id="ind_adx",
            relationship_type="USES",
            properties={},
            strength=0.9
        ))

        logger.info(f"Seeded knowledge graph with {len(self.concepts)} concepts")


logger.info("In-memory knowledge graph implementation loaded")
