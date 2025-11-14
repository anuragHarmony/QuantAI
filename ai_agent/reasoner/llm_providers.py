"""
LLM provider implementations with async support and function calling
"""
from typing import Any, Optional, AsyncIterator
from abc import ABC
import json
from loguru import logger

from shared.models.base import ILLMProvider
from shared.config.settings import settings


class OpenAIProvider(ILLMProvider):
    """OpenAI LLM provider with function calling support"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_tokens: int = 4096,
        temperature: float = 0.1
    ):
        """
        Initialize OpenAI provider

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name
            max_tokens: Maximum tokens per request
            temperature: Sampling temperature
        """
        from openai import AsyncOpenAI

        self.api_key = api_key or settings.llm.openai_api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(f"Initialized OpenAI provider with model: {model}")

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs: Any
    ) -> str:
        """
        Generate completion

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            content = response.choices[0].message.content or ""
            logger.debug(f"OpenAI completion generated: {len(content)} chars")

            return content

        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise

    async def complete_with_functions(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        **kwargs: Any
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """
        Generate completion with function calling

        Args:
            prompt: Input prompt
            functions: Function definitions in OpenAI format
            **kwargs: Additional parameters

        Returns:
            Tuple of (text_response, function_call_dict)
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                functions=functions,
                function_call="auto",
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )

            message = response.choices[0].message
            content = message.content or ""

            # Check if function was called
            function_call = None
            if message.function_call:
                function_call = {
                    "name": message.function_call.name,
                    "arguments": json.loads(message.function_call.arguments)
                }
                logger.info(f"Function called: {function_call['name']}")

            return content, function_call

        except Exception as e:
            logger.error(f"OpenAI function calling failed: {e}")
            raise

    async def stream(
        self,
        prompt: str,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream completion chunks

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise


class AnthropicProvider(ILLMProvider):
    """Anthropic (Claude) LLM provider with tool calling support"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4096,
        temperature: float = 0.1
    ):
        """
        Initialize Anthropic provider

        Args:
            api_key: Anthropic API key (defaults to settings)
            model: Model name
            max_tokens: Maximum tokens per request
            temperature: Sampling temperature
        """
        from anthropic import AsyncAnthropic

        self.api_key = api_key or settings.llm.anthropic_api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncAnthropic(api_key=self.api_key)

        logger.info(f"Initialized Anthropic provider with model: {model}")

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs: Any
    ) -> str:
        """
        Generate completion

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            content = response.content[0].text if response.content else ""
            logger.debug(f"Anthropic completion generated: {len(content)} chars")

            return content

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise

    async def complete_with_functions(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        **kwargs: Any
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """
        Generate completion with tool calling

        Args:
            prompt: Input prompt
            functions: Tool definitions in Anthropic format
            **kwargs: Additional parameters

        Returns:
            Tuple of (text_response, tool_use_dict)
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}],
                tools=functions
            )

            # Extract text and tool use
            text_content = ""
            tool_use = None

            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_use = {
                        "name": block.name,
                        "arguments": block.input
                    }
                    logger.info(f"Tool called: {tool_use['name']}")

            return text_content, tool_use

        except Exception as e:
            logger.error(f"Anthropic tool calling failed: {e}")
            raise

    async def stream(
        self,
        prompt: str,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream completion chunks

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    @staticmethod
    def create_provider(
        provider_name: str = "openai",
        **kwargs: Any
    ) -> ILLMProvider:
        """
        Create LLM provider by name

        Args:
            provider_name: Provider name ("openai" or "anthropic")
            **kwargs: Provider-specific arguments

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider not supported
        """
        provider_name = provider_name.lower()

        if provider_name == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_name == "anthropic":
            return AnthropicProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    @staticmethod
    def create_default() -> ILLMProvider:
        """Create default LLM provider from settings"""
        return LLMProviderFactory.create_provider(
            settings.llm.default_llm_provider,
            model=settings.llm.default_model,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature
        )
