"""
AI Agent Core Module

Core components for the autonomous AI research agent.
"""

from .hypothesis import (
    Hypothesis,
    StrategyHypothesis,
    HypothesisGenerator,
)

from .code_generator import (
    GeneratedCode,
    CodeGenerator,
)

from .reflection import (
    Insight,
    ReflectionEngine,
)

from .orchestrator import (
    ResearchPhase,
    ResearchSession,
    ResearchConfig,
    ResearchOrchestrator,
)

__all__ = [
    # Hypothesis
    "Hypothesis",
    "StrategyHypothesis",
    "HypothesisGenerator",

    # Code generation
    "GeneratedCode",
    "CodeGenerator",

    # Reflection
    "Insight",
    "ReflectionEngine",

    # Orchestration
    "ResearchPhase",
    "ResearchSession",
    "ResearchConfig",
    "ResearchOrchestrator",
]
