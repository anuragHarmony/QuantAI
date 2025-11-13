"""High-level semantic search with context assembly."""

from typing import List, Dict, Optional, Any
from loguru import logger

from shared.models.knowledge import KnowledgeChunk, QueryResult, RetrievalContext
from .vector_store import VectorStore


class SemanticSearch:
    """High-level semantic search with intelligent context assembly."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize semantic search.

        Args:
            vector_store: VectorStore instance (creates new one if not provided)
        """
        self.vector_store = vector_store or VectorStore()

    def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.5
    ) -> List[QueryResult]:
        """
        Search for relevant knowledge with filtering and ranking.

        Args:
            query: Search query
            max_results: Maximum number of results
            filters: Metadata filters
            min_relevance: Minimum relevance score threshold

        Returns:
            List of QueryResult objects
        """
        logger.info(f"Semantic search for: '{query}'")

        # Perform vector search
        results = self.vector_store.search(
            query=query,
            n_results=max_results * 2,  # Get more initially for filtering
            filters=filters
        )

        # Filter by minimum relevance
        filtered_results = [
            r for r in results
            if r.relevance_score >= min_relevance
        ]

        # Sort by relevance score
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit to max_results
        final_results = filtered_results[:max_results]

        logger.info(f"Returning {len(final_results)} results (filtered from {len(results)})")
        return final_results

    def search_by_type(
        self,
        query: str,
        chunk_type: str,
        max_results: int = 10
    ) -> List[QueryResult]:
        """
        Search for specific type of knowledge.

        Args:
            query: Search query
            chunk_type: Type of chunk (concept, strategy, formula, example, code)
            max_results: Maximum number of results

        Returns:
            List of QueryResult objects
        """
        filters = {"chunk_type": chunk_type}
        return self.search(query, max_results, filters)

    def search_strategies(self, query: str, max_results: int = 5) -> List[QueryResult]:
        """Search for trading strategies."""
        return self.search_by_type(query, "strategy", max_results)

    def search_concepts(self, query: str, max_results: int = 10) -> List[QueryResult]:
        """Search for concepts and theories."""
        return self.search_by_type(query, "concept", max_results)

    def search_examples(self, query: str, max_results: int = 5) -> List[QueryResult]:
        """Search for examples and case studies."""
        return self.search_by_type(query, "example", max_results)

    def assemble_context(
        self,
        query: str,
        max_tokens: int = 4000,
        include_examples: bool = True
    ) -> RetrievalContext:
        """
        Assemble context for AI reasoning.

        Args:
            query: User query
            max_tokens: Maximum token budget for context
            include_examples: Whether to include examples

        Returns:
            RetrievalContext object
        """
        logger.info(f"Assembling context for: '{query}' (max_tokens={max_tokens})")

        all_results = []

        # Search for concepts (high priority)
        concept_results = self.search_concepts(query, max_results=5)
        all_results.extend(concept_results)

        # Search for strategies (medium priority)
        strategy_results = self.search_strategies(query, max_results=3)
        all_results.extend(strategy_results)

        # Search for examples (if requested)
        if include_examples:
            example_results = self.search_examples(query, max_results=2)
            all_results.extend(example_results)

        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.chunk.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.chunk.id)

        # Sort by relevance
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        current_tokens = 0
        selected_results = []

        for result in unique_results:
            chunk_tokens = len(result.chunk.content) // 4
            if current_tokens + chunk_tokens <= max_tokens:
                selected_results.append(result)
                current_tokens += chunk_tokens
            else:
                break

        # Calculate confidence score (average relevance)
        if selected_results:
            confidence = sum(r.relevance_score for r in selected_results) / len(selected_results)
        else:
            confidence = 0.0

        context = RetrievalContext(
            query=query,
            results=selected_results,
            total_tokens=current_tokens,
            confidence_score=confidence,
            metadata={
                "num_concepts": len(concept_results),
                "num_strategies": len(strategy_results),
                "num_examples": len(example_results) if include_examples else 0
            }
        )

        logger.info(
            f"Assembled context: {len(selected_results)} chunks, "
            f"{current_tokens} tokens, confidence={confidence:.2f}"
        )

        return context

    def get_related_chunks(
        self,
        chunk_id: str,
        max_results: int = 5
    ) -> List[QueryResult]:
        """
        Find chunks related to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            max_results: Maximum number of results

        Returns:
            List of related QueryResult objects
        """
        # First, retrieve the chunk content
        # Note: This is a simplified implementation
        # In production, you'd want to store relationships explicitly

        logger.info(f"Finding related chunks for: {chunk_id}")

        # For now, we'll use the chunk's content as the query
        # This is a placeholder - in a full implementation, you'd use
        # the knowledge graph to find explicit relationships

        return []

    def multi_query_search(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> List[QueryResult]:
        """
        Search using multiple queries and combine results.

        Useful for complex or multi-faceted questions.

        Args:
            queries: List of search queries
            max_results_per_query: Max results per individual query

        Returns:
            Combined and deduplicated list of QueryResult objects
        """
        logger.info(f"Multi-query search with {len(queries)} queries")

        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.search(query, max_results=max_results_per_query)

            for result in results:
                if result.chunk.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.chunk.id)

        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"Multi-query search returned {len(all_results)} unique results")
        return all_results
