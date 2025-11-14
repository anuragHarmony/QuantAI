"""
Embedding provider implementations
"""
from typing import Any, Optional
from loguru import logger

from shared.models.base import IEmbeddingProvider
from shared.config.settings import settings


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """OpenAI embedding provider"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI embedding provider

        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        from openai import AsyncOpenAI

        self.api_key = api_key or settings.llm.openai_api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        logger.info(f"Initialized OpenAI embedding provider with model: {model}")

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimensions.get(self.model, 1536)


class SentenceTransformerEmbeddingProvider(IEmbeddingProvider):
    """Local sentence transformer embedding provider"""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence transformer provider

        Args:
            model: Model name from sentence-transformers
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model)
        self._dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Initialized SentenceTransformer with model: {model}")

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            # Run in thread pool since sentence-transformers is sync
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(text, convert_to_numpy=True)
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            )

            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension


class HybridEmbeddingProvider(IEmbeddingProvider):
    """
    Hybrid embedding provider that uses different models for text and code
    """

    def __init__(
        self,
        text_provider: Optional[IEmbeddingProvider] = None,
        code_provider: Optional[IEmbeddingProvider] = None
    ):
        """
        Initialize hybrid provider

        Args:
            text_provider: Provider for general text
            code_provider: Provider for code/formulas
        """
        self.text_provider = text_provider or OpenAIEmbeddingProvider()
        self.code_provider = code_provider or SentenceTransformerEmbeddingProvider(
            model="sentence-transformers/all-mpnet-base-v2"
        )

        logger.info("Initialized hybrid embedding provider")

    async def embed_text(self, text: str, is_code: bool = False) -> list[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed
            is_code: Whether text is code/formula

        Returns:
            Embedding vector
        """
        provider = self.code_provider if is_code else self.text_provider
        return await provider.embed_text(text)

    async def embed_batch(self, texts: list[str], is_code: bool = False) -> list[list[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            is_code: Whether texts are code/formulas

        Returns:
            List of embedding vectors
        """
        provider = self.code_provider if is_code else self.text_provider
        return await provider.embed_batch(texts)

    def dimension(self) -> int:
        """Get embedding dimension (returns text provider dimension)"""
        return self.text_provider.dimension()


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""

    @staticmethod
    def create_provider(
        provider_name: str = "openai",
        **kwargs: Any
    ) -> IEmbeddingProvider:
        """
        Create embedding provider by name

        Args:
            provider_name: Provider name
            **kwargs: Provider-specific arguments

        Returns:
            Embedding provider instance
        """
        provider_name = provider_name.lower()

        if provider_name == "openai":
            return OpenAIEmbeddingProvider(**kwargs)
        elif provider_name == "sentence-transformer":
            return SentenceTransformerEmbeddingProvider(**kwargs)
        elif provider_name == "hybrid":
            return HybridEmbeddingProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")

    @staticmethod
    def create_default() -> IEmbeddingProvider:
        """Create default embedding provider from settings"""
        return OpenAIEmbeddingProvider(model=settings.llm.embedding_model)
