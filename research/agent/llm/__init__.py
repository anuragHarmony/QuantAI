"""
LLM Module

Language model interface and implementations for the AI research agent.
"""

from .interface import ILLM, Message, LLMResponse, LLMProvider
from .claude_provider import ClaudeProvider

__all__ = [
    "ILLM",
    "Message",
    "LLMResponse",
    "LLMProvider",
    "ClaudeProvider",
]
