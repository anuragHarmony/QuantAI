"""
AI Research Agent

Autonomous AI agent for quantitative trading research.
"""

from .core import (
    HypothesisGenerator,
    CodeGenerator,
    ReflectionEngine,
    ResearchOrchestrator,
    ResearchConfig,
)

from .llm import (
    ILLM,
    ClaudeProvider,
)

from .memory import (
    IEpisodicMemory,
    ISemanticMemory,
    IWorkingMemory,
    SimpleEpisodicMemory,
    SimpleSemanticMemory,
    SimpleWorkingMemory,
    Experiment,
    Principle,
)

__all__ = [
    # Core components
    "HypothesisGenerator",
    "CodeGenerator",
    "ReflectionEngine",
    "ResearchOrchestrator",
    "ResearchConfig",

    # LLM
    "ILLM",
    "ClaudeProvider",

    # Memory
    "IEpisodicMemory",
    "ISemanticMemory",
    "IWorkingMemory",
    "SimpleEpisodicMemory",
    "SimpleSemanticMemory",
    "SimpleWorkingMemory",
    "Experiment",
    "Principle",
]
