"""
Memory Module

Multi-level memory system for the AI research agent.
"""

from .interface import (
    IEpisodicMemory,
    ISemanticMemory,
    IWorkingMemory,
    Experiment,
    Principle,
)

from .simple_memory import (
    SimpleEpisodicMemory,
    SimpleSemanticMemory,
    SimpleWorkingMemory,
)

__all__ = [
    # Interfaces
    "IEpisodicMemory",
    "ISemanticMemory",
    "IWorkingMemory",

    # Data types
    "Experiment",
    "Principle",

    # Implementations
    "SimpleEpisodicMemory",
    "SimpleSemanticMemory",
    "SimpleWorkingMemory",
]
