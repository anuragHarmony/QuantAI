"""
Agent Factory

Factory for creating configured AI research agents.
"""
from typing import Optional
import os
from loguru import logger

from research.agent.llm.claude_provider import ClaudeProvider
from research.agent.memory.simple_memory import (
    SimpleEpisodicMemory,
    SimpleSemanticMemory,
    SimpleWorkingMemory
)
from research.knowledge.integrated_retriever import IntegratedKnowledgeRetriever
from research.knowledge.memory_graph import InMemoryKnowledgeGraph
from research.data.binance_provider import BinanceDataProvider
from research.data.feature_engineer import TechnicalFeatureEngineer
from research.data.pattern_detector import SimplePatternDetector
from research.agent.core.orchestrator import ResearchOrchestrator, ResearchConfig


class AgentFactory:
    """
    Factory for creating configured AI research agents

    Single Responsibility: Dependency injection and configuration
    """

    @staticmethod
    async def create_research_agent(
        anthropic_api_key: Optional[str] = None,
        config: Optional[ResearchConfig] = None,
        use_neo4j: bool = False,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ) -> ResearchOrchestrator:
        """
        Create a fully configured research agent

        Args:
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            config: Research configuration
            use_neo4j: Use Neo4j for knowledge graph (default: in-memory)
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password

        Returns:
            Configured ResearchOrchestrator
        """
        logger.info("Creating AI research agent...")

        # Get API key
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter"
            )

        # Initialize LLM
        logger.info("Initializing LLM (Claude)...")
        llm = ClaudeProvider(
            api_key=api_key,
            model="claude-3-5-sonnet-20241022"
        )

        # Initialize knowledge graph
        if use_neo4j:
            logger.info("Initializing Neo4j knowledge graph...")
            # TODO: Add Neo4j implementation
            raise NotImplementedError("Neo4j implementation coming soon")
        else:
            logger.info("Initializing in-memory knowledge graph...")
            knowledge_graph = InMemoryKnowledgeGraph()
            await knowledge_graph.seed_with_defaults()

        # Initialize memory systems
        logger.info("Initializing memory systems...")
        episodic_memory = SimpleEpisodicMemory()
        semantic_memory = SimpleSemanticMemory()
        working_memory = SimpleWorkingMemory()

        # Initialize knowledge retriever
        logger.info("Initializing knowledge retriever...")
        # Note: RAG system not implemented yet, will use None
        knowledge_retriever = IntegratedKnowledgeRetriever(
            rag_system=None,  # TODO: Add RAG integration
            knowledge_graph=knowledge_graph,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory
        )

        # Initialize data components
        logger.info("Initializing data providers...")
        data_provider = BinanceDataProvider()
        feature_engineer = TechnicalFeatureEngineer()
        pattern_detector = SimplePatternDetector()

        # Create research config
        research_config = config or ResearchConfig()

        # Create orchestrator
        logger.info("Creating research orchestrator...")
        orchestrator = ResearchOrchestrator(
            llm=llm,
            knowledge_retriever=knowledge_retriever,
            data_provider=data_provider,
            feature_engineer=feature_engineer,
            pattern_detector=pattern_detector,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            working_memory=working_memory,
            config=research_config
        )

        logger.success("AI research agent created successfully!")
        return orchestrator

    @staticmethod
    async def create_simple_agent(
        anthropic_api_key: Optional[str] = None
    ) -> ResearchOrchestrator:
        """
        Create a simple agent with default configuration

        Args:
            anthropic_api_key: Anthropic API key

        Returns:
            Configured ResearchOrchestrator
        """
        config = ResearchConfig(
            max_iterations=5,
            min_sharpe_threshold=1.0,
            parallel_experiments=1
        )

        return await AgentFactory.create_research_agent(
            anthropic_api_key=anthropic_api_key,
            config=config
        )


logger.info("Agent factory loaded")
