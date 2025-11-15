"""
Integrated Knowledge Retriever

Combines three knowledge sources:
1. RAG System (books, documentation) - textual knowledge
2. Knowledge Graph (structured concepts) - relationships and metrics
3. Memory System (past experiments) - learned from experience

The agent queries all three to get comprehensive knowledge.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from .interface import IKnowledgeGraph, Concept, ConceptType


@dataclass
class IntegratedKnowledge:
    """Combined knowledge from all sources"""

    # From RAG
    textual_knowledge: str
    relevant_documents: List[str]

    # From Knowledge Graph
    related_concepts: List[Concept]
    concept_relationships: List[Dict[str, Any]]

    # From Memory
    past_experiments: List[Dict[str, Any]]
    learned_principles: List[str]

    # Metadata
    sources_used: List[str]
    confidence: float


class IntegratedKnowledgeRetriever:
    """
    Retrieves knowledge from multiple sources

    This is what the agent queries to get comprehensive knowledge.

    Example:
        retriever = IntegratedKnowledgeRetriever(
            knowledge_graph=kg,
            rag_system=rag,
            memory=memory
        )

        knowledge = await retriever.query(
            "What strategies work for mean reversion in BTC?"
        )

        # Returns:
        # - Textual explanation from books (RAG)
        # - Related concepts from graph (Bollinger Bands, RSI)
        # - Past experiments that tried this (Memory)
        # - Learned principles (e.g., "RSI alone has 45% win rate")
    """

    def __init__(
        self,
        knowledge_graph: IKnowledgeGraph,
        rag_system: Optional[Any] = None,  # Your existing RAG system
        memory: Optional[Any] = None  # Memory system (we'll build this)
    ):
        """
        Initialize integrated retriever

        Args:
            knowledge_graph: Knowledge graph implementation
            rag_system: RAG system for querying books/docs (optional)
            memory: Memory system for past experiments (optional)
        """
        self.knowledge_graph = knowledge_graph
        self.rag_system = rag_system
        self.memory = memory

        logger.info("Initialized IntegratedKnowledgeRetriever")

    async def query(
        self,
        query: str,
        include_rag: bool = True,
        include_graph: bool = True,
        include_memory: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> IntegratedKnowledge:
        """
        Query all knowledge sources

        Args:
            query: Natural language query
            include_rag: Include RAG system results
            include_graph: Include knowledge graph results
            include_memory: Include memory results
            context: Additional context (market regime, asset, etc.)

        Returns:
            Integrated knowledge from all sources
        """
        logger.info(f"Querying integrated knowledge: '{query}'")

        sources_used = []
        textual_knowledge = ""
        relevant_documents = []
        related_concepts = []
        concept_relationships = []
        past_experiments = []
        learned_principles = []

        # 1. Query RAG System for textual knowledge
        if include_rag and self.rag_system:
            try:
                rag_result = await self._query_rag(query)
                textual_knowledge = rag_result.get("answer", "")
                relevant_documents = rag_result.get("documents", [])
                sources_used.append("rag")
                logger.debug(f"Retrieved {len(relevant_documents)} documents from RAG")
            except Exception as e:
                logger.error(f"Error querying RAG: {e}")

        # 2. Query Knowledge Graph for structured concepts
        if include_graph:
            try:
                graph_result = await self._query_knowledge_graph(query, context)
                related_concepts = graph_result.get("concepts", [])
                concept_relationships = graph_result.get("relationships", [])
                sources_used.append("knowledge_graph")
                logger.debug(f"Retrieved {len(related_concepts)} concepts from graph")
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")

        # 3. Query Memory for past experiments
        if include_memory and self.memory:
            try:
                memory_result = await self._query_memory(query, context)
                past_experiments = memory_result.get("experiments", [])
                learned_principles = memory_result.get("principles", [])
                sources_used.append("memory")
                logger.debug(f"Retrieved {len(past_experiments)} past experiments from memory")
            except Exception as e:
                logger.error(f"Error querying memory: {e}")

        # Calculate confidence based on sources
        confidence = len(sources_used) / 3.0  # Simple confidence

        return IntegratedKnowledge(
            textual_knowledge=textual_knowledge,
            relevant_documents=relevant_documents,
            related_concepts=related_concepts,
            concept_relationships=concept_relationships,
            past_experiments=past_experiments,
            learned_principles=learned_principles,
            sources_used=sources_used,
            confidence=confidence
        )

    async def _query_rag(self, query: str) -> Dict[str, Any]:
        """
        Query RAG system

        Assumes your existing RAG system has a method like:
        - rag.query(query) -> {"answer": str, "documents": List[str]}
        """
        if not self.rag_system:
            return {"answer": "", "documents": []}

        # Check if RAG system has query method
        if hasattr(self.rag_system, 'query'):
            result = await self.rag_system.query(query)
            return result
        elif hasattr(self.rag_system, 'search'):
            result = await self.rag_system.search(query)
            return result
        else:
            logger.warning("RAG system doesn't have query or search method")
            return {"answer": "", "documents": []}

    async def _query_knowledge_graph(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Query knowledge graph

        Extract concepts from natural language query and find related concepts.
        """
        # Simple keyword extraction (in production, use NER or LLM)
        keywords = self._extract_keywords(query)

        concepts = []
        relationships = []

        # Find concepts matching keywords
        for keyword in keywords:
            # Try to find concepts by name
            all_concepts = await self.knowledge_graph.query_concepts(limit=1000)

            for concept in all_concepts:
                if keyword.lower() in concept.name.lower():
                    concepts.append(concept)

                    # Get related concepts
                    related = await self.knowledge_graph.find_related_concepts(
                        concept.id,
                        max_depth=1
                    )
                    concepts.extend(related)

        # Filter by context if provided
        if context and "market_regime" in context:
            regime = context["market_regime"]

            # Get concepts that work in this regime
            regime_concepts = await self.knowledge_graph.find_related_concepts(
                f"regime_{regime.lower()}",
                relationship_type="WORKS_IN",
                max_depth=1
            )
            concepts.extend(regime_concepts)

        # Remove duplicates
        unique_concepts = []
        seen_ids = set()
        for concept in concepts:
            if concept.id not in seen_ids:
                unique_concepts.append(concept)
                seen_ids.add(concept.id)

        return {
            "concepts": unique_concepts[:20],  # Limit to top 20
            "relationships": relationships
        }

    async def _query_memory(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Query memory system

        Will be implemented when we build the memory system.
        """
        if not self.memory:
            return {"experiments": [], "principles": []}

        # TODO: Implement memory querying
        # For now, placeholder

        return {
            "experiments": [],
            "principles": []
        }

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Simple keyword extraction

        In production, use:
        - Named Entity Recognition (NER)
        - LLM to extract key concepts
        - spaCy or similar NLP library
        """
        # Simple approach: extract capitalized words and common terms
        keywords = []

        # Common trading terms
        trading_terms = [
            "rsi", "macd", "bollinger", "ema", "sma", "adx", "atr",
            "mean reversion", "momentum", "arbitrage", "trend",
            "ranging", "volatile", "support", "resistance"
        ]

        query_lower = query.lower()

        for term in trading_terms:
            if term in query_lower:
                keywords.append(term)

        # Also extract capitalized words (likely concepts)
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                keywords.append(word)

        return list(set(keywords))  # Remove duplicates


logger.info("Integrated knowledge retriever loaded")
