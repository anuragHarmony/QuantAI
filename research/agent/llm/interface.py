"""
LLM Interface and Implementation

Abstract interface for language models following SOLID principles.
Allows swapping between Claude, GPT-4, or other LLMs.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class Message:
    """Chat message"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """LLM response"""
    content: str
    model: str
    usage: Dict[str, int]  # tokens used
    finish_reason: str


class ILLM(ABC):
    """
    Interface for Language Model providers

    Single Responsibility: Generate text from prompts
    Open/Closed: Can add new providers without modifying existing code
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        response_format: Optional[str] = None,  # "json" or None
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            response_format: "json" if you want JSON response

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """
        Chat completion with message history

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
