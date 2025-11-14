"""
Production-grade RAG (Retrieval-Augmented Generation) pipeline
with multi-stage retrieval, re-ranking, and context assembly
"""
from typing import Any, Optional
from enum import Enum
from loguru import logger
from pydantic import BaseModel, Field

from shared.models.base import (
    IRetriever,
    IVectorStore,
    IKnowledgeGraph,
    IEmbeddingProvider,
    ILLMProvider,
    SearchResult,
    KnowledgeChunk,
    RetrievalContext
)
from shared.utils.cache import ICacheProvider


class QueryIntent(str, Enum):
    """Query intent types"""
    EXPLAIN = "explain"  # Explain a concept
    SUGGEST = "suggest"  # Suggest strategies
    COMPARE = "compare"  # Compare approaches
    WHY_FAILED = "why_failed"  # Diagnose failures
    GENERAL = "general"  # General query


class QueryPlan(BaseModel):
    """Query execution plan"""
    original_query: str = Field(description="Original query")
    expanded_query: str = Field(description="Expanded query with synonyms")
    intent: QueryIntent = Field(description="Detected intent")
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    top_k: int = Field(default=100, description="Initial retrieval size")
    final_k: int = Field(default=10, description="Final results size")


class MultiStageRetriever(IRetriever):
    """
    Multi-stage retrieval pipeline:
    1. Query understanding & expansion
    2. Broad retrieval (vector + keyword)
    3. Graph expansion
    4. Metadata filtering
    5. Re-ranking
    6. Deduplication
    """

    def __init__(
        self,
        vector_store: IVectorStore[KnowledgeChunk],
        knowledge_graph: IKnowledgeGraph,
        embedding_provider: IEmbeddingProvider,
        cache: Optional[ICacheProvider[str, Any]] = None
    ):
        """
        Initialize multi-stage retriever

        Args:
            vector_store: Vector store for similarity search
            knowledge_graph: Knowledge graph for relationship traversal
            embedding_provider: Embedding generation
            cache: Optional cache for retrieval results
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.embedding_provider = embedding_provider
        self.cache = cache

        logger.info("Initialized multi-stage retriever")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Retrieve relevant knowledge chunks

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters

        Returns:
            List of search results
        """
        # Stage 1: Query understanding
        query_plan = await self._understand_query(query, filters)

        # Stage 2: Broad retrieval
        candidates = await self._broad_retrieval(query_plan)

        # Stage 3: Graph expansion
        expanded = await self._graph_expansion(candidates)

        # Stage 4: Metadata filtering
        filtered = self._metadata_filtering(expanded, query_plan.filters)

        # Stage 5: Re-ranking
        reranked = await self._rerank(query, filtered)

        # Stage 6: Deduplication
        final = self._deduplicate(reranked)

        # Return top-k
        return final[:top_k]

    async def assemble_context(
        self,
        query: str,
        max_tokens: int = 4000
    ) -> RetrievalContext:
        """
        Assemble context within token budget

        Args:
            query: Search query
            max_tokens: Maximum tokens for context

        Returns:
            Retrieval context with chunks and confidence
        """
        # Retrieve chunks
        results = await self.retrieve(query, top_k=20)

        # Assemble with budget management
        selected_chunks: list[KnowledgeChunk] = []
        total_tokens = 0
        token_per_char = 0.25  # Rough estimate: 4 chars per token

        # Priority order:
        # 1. High relevance chunks (score > 0.7)
        # 2. Prerequisites for selected chunks
        # 3. Related examples

        for result in results:
            chunk = result.item
            estimated_tokens = len(chunk.content) * token_per_char

            if total_tokens + estimated_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += estimated_tokens

                # Add prerequisites if space allows
                for prereq_id in chunk.prerequisites:
                    prereq = await self.vector_store.get(prereq_id)
                    if prereq:
                        prereq_tokens = len(prereq.content) * token_per_char
                        if total_tokens + prereq_tokens <= max_tokens:
                            selected_chunks.append(prereq)
                            total_tokens += prereq_tokens
            else:
                break

        # Calculate confidence
        confidence = self._calculate_confidence(results, selected_chunks)

        # Calculate coverage
        coverage = len(selected_chunks) / len(results) if results else 0.0

        # Extract unique sources
        sources = list(set(chunk.source_book for chunk in selected_chunks))

        return RetrievalContext(
            chunks=selected_chunks,
            confidence=confidence,
            coverage=coverage,
            sources=sources,
            metadata={
                "total_tokens": total_tokens,
                "num_chunks": len(selected_chunks),
                "query": query
            }
        )

    async def _understand_query(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Understand query intent and expand

        Args:
            query: Original query
            filters: Optional filters

        Returns:
            Query execution plan
        """
        # Detect intent from query text
        query_lower = query.lower()

        if any(word in query_lower for word in ["explain", "what is", "how does"]):
            intent = QueryIntent.EXPLAIN
            top_k = 100
            final_k = 15
        elif any(word in query_lower for word in ["suggest", "recommend", "should i"]):
            intent = QueryIntent.SUGGEST
            top_k = 150
            final_k = 20
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            intent = QueryIntent.COMPARE
            top_k = 120
            final_k = 15
        elif any(word in query_lower for word in ["why failed", "didn't work", "problem"]):
            intent = QueryIntent.WHY_FAILED
            top_k = 100
            final_k = 15
        else:
            intent = QueryIntent.GENERAL
            top_k = 100
            final_k = 10

        # Expand query (simple version - in production use LLM)
        expanded_query = query
        synonyms = {
            "moving average": "MA SMA EMA",
            "rsi": "relative strength index",
            "macd": "moving average convergence divergence",
            "stock": "equity share",
            "crypto": "cryptocurrency bitcoin"
        }

        for term, expansion in synonyms.items():
            if term in query_lower:
                expanded_query += f" {expansion}"

        return QueryPlan(
            original_query=query,
            expanded_query=expanded_query,
            intent=intent,
            filters=filters or {},
            top_k=top_k,
            final_k=final_k
        )

    async def _broad_retrieval(
        self,
        query_plan: QueryPlan
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Stage 2: Broad retrieval using vector search

        Args:
            query_plan: Query execution plan

        Returns:
            Candidate results
        """
        # Generate embedding for expanded query
        query_embedding = await self.embedding_provider.embed_text(
            query_plan.expanded_query
        )

        # Vector search
        results = await self.vector_store.search(
            query_embedding,
            top_k=query_plan.top_k,
            filters=query_plan.filters
        )

        logger.info(f"Broad retrieval returned {len(results)} candidates")
        return results

    async def _graph_expansion(
        self,
        candidates: list[SearchResult[KnowledgeChunk]]
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Stage 3: Expand results using knowledge graph

        Args:
            candidates: Initial candidates

        Returns:
            Expanded results
        """
        expanded = list(candidates)
        seen_ids = {result.item.id for result in candidates}

        # For top candidates, find related concepts
        for result in candidates[:10]:  # Only expand top 10
            chunk = result.item

            # Find related chunks from graph
            related_nodes = await self.knowledge_graph.find_related(
                chunk.id,
                edge_types=["RELATES_TO", "PREREQUISITE", "EXAMPLE_OF"],
                max_depth=1
            )

            # Fetch related chunks from vector store
            for node in related_nodes[:5]:  # Limit expansion
                node_id = node.get("id")
                if node_id and node_id not in seen_ids:
                    related_chunk = await self.vector_store.get(node_id)
                    if related_chunk:
                        # Add with reduced score
                        expanded.append(SearchResult(
                            item=related_chunk,
                            score=result.score * 0.8,  # Reduce score for expanded items
                            metadata={"expanded_from": chunk.id}
                        ))
                        seen_ids.add(node_id)

        logger.info(f"Graph expansion added {len(expanded) - len(candidates)} chunks")
        return expanded

    def _metadata_filtering(
        self,
        results: list[SearchResult[KnowledgeChunk]],
        filters: dict[str, Any]
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Stage 4: Apply metadata filters

        Args:
            results: Search results
            filters: Metadata filters

        Returns:
            Filtered results
        """
        if not filters:
            return results

        filtered = []
        for result in results:
            chunk = result.item
            matches = True

            for key, value in filters.items():
                if isinstance(value, list):
                    # Check if any value matches
                    chunk_value = getattr(chunk, key, None)
                    if isinstance(chunk_value, list):
                        if not any(v in chunk_value for v in value):
                            matches = False
                            break
                    elif chunk_value not in value:
                        matches = False
                        break
                else:
                    # Exact match
                    if getattr(chunk, key, None) != value:
                        matches = False
                        break

            if matches:
                filtered.append(result)

        logger.info(f"Metadata filtering: {len(results)} -> {len(filtered)}")
        return filtered

    async def _rerank(
        self,
        query: str,
        results: list[SearchResult[KnowledgeChunk]]
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Stage 5: Re-rank results using cross-encoder
        (Simplified version - in production use a cross-encoder model)

        Args:
            query: Original query
            results: Search results

        Returns:
            Re-ranked results
        """
        # Simple re-ranking based on text overlap and hierarchy
        query_terms = set(query.lower().split())

        for result in results:
            chunk = result.item
            chunk_terms = set(chunk.content.lower().split())

            # Calculate term overlap
            overlap = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0

            # Boost based on hierarchy level (prefer detailed info)
            hierarchy_boost = 1.0 + (chunk.hierarchy_level * 0.1)

            # Boost based on chunk type
            type_boost = {
                "example": 1.2,
                "code": 1.15,
                "strategy": 1.1,
                "concept": 1.0,
                "formula": 1.05
            }.get(chunk.chunk_type.value, 1.0)

            # Combine scores
            new_score = result.score * (1 + overlap) * hierarchy_boost * type_boost
            result.score = min(new_score, 1.0)

        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info("Re-ranked results")
        return results

    def _deduplicate(
        self,
        results: list[SearchResult[KnowledgeChunk]],
        similarity_threshold: float = 0.95
    ) -> list[SearchResult[KnowledgeChunk]]:
        """
        Stage 6: Remove near-duplicate chunks

        Args:
            results: Search results
            similarity_threshold: Similarity threshold for deduplication

        Returns:
            Deduplicated results
        """
        if not results:
            return results

        deduplicated = [results[0]]
        seen_content = {results[0].item.content}

        for result in results[1:]:
            content = result.item.content

            # Simple content-based deduplication
            # In production, use embedding similarity
            is_duplicate = False
            for seen in seen_content:
                # Simple check: if content is very similar
                if len(content) > 0 and len(seen) > 0:
                    shared = len(set(content.split()) & set(seen.split()))
                    similarity = shared / max(len(content.split()), len(seen.split()))

                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(result)
                seen_content.add(content)

        logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)}")
        return deduplicated

    def _calculate_confidence(
        self,
        all_results: list[SearchResult[KnowledgeChunk]],
        selected_chunks: list[KnowledgeChunk]
    ) -> float:
        """
        Calculate retrieval confidence

        Args:
            all_results: All search results
            selected_chunks: Selected chunks for context

        Returns:
            Confidence score (0-1)
        """
        if not all_results:
            return 0.0

        # Average score of top results
        top_scores = [r.score for r in all_results[:10]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Diversity: number of unique sources
        sources = set(chunk.source_book for chunk in selected_chunks)
        diversity = min(len(sources) / 3, 1.0)  # Normalize to max 3 sources

        # Combine
        confidence = (avg_score * 0.7) + (diversity * 0.3)

        return confidence
