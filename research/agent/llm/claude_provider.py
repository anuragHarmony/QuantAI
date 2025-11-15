"""
Claude LLM Provider

Implementation of ILLM for Anthropic's Claude models.
"""
from typing import List, Dict, Any, Optional
import json
import os
from loguru import logger

from .interface import ILLM, Message, LLMResponse


class ClaudeProvider(ILLM):
    """
    Claude LLM provider using Anthropic API

    Single Responsibility: Interface with Claude API
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        default_temperature: float = 0.7,
    ):
        """
        Initialize Claude provider

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name
            default_temperature: Default temperature
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY env var.")

        self.model = model
        self.default_temperature = default_temperature

        # Initialize client
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info(f"Initialized Claude provider with model: {model}")
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            self.client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        response_format: Optional[str] = None,
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            response_format: "json" if JSON response needed

        Returns:
            Generated text
        """
        if not self.client:
            raise RuntimeError("Claude client not initialized")

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Add JSON instruction if needed
        if response_format == "json":
            prompt += "\n\nReturn your response as valid JSON only, no additional text."

        # Call API
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else "You are a helpful AI assistant specialized in quantitative trading.",
                messages=messages,
            )

            content = response.content[0].text

            # If JSON format requested, try to extract JSON
            if response_format == "json":
                content = self._extract_json(content)

            logger.debug(f"Generated {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise

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
            temperature: Temperature
            max_tokens: Max tokens

        Returns:
            LLM response
        """
        if not self.client:
            raise RuntimeError("Claude client not initialized")

        # Convert to Claude format
        claude_messages = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Call API
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else "You are a helpful AI assistant.",
                messages=claude_messages,
            )

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count

        Note: This is approximate. For exact count, use Anthropic's token counter.
        """
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that might contain markdown or extra text

        Args:
            text: Text potentially containing JSON

        Returns:
            JSON string
        """
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        # Try to find JSON directly
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            potential_json = text[start:end]

            # Validate it's actually JSON
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        # Return as-is if no JSON found
        return text


logger.info("Claude LLM provider loaded")
