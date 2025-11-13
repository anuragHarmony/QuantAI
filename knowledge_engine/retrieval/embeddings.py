"""Embedding generation using OpenAI or local models."""

from typing import List, Union
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from shared.config.settings import settings


class EmbeddingGenerator:
    """Unified interface for generating embeddings."""

    def __init__(
        self,
        use_openai: bool = None,
        openai_api_key: str = None,
        model_name: str = None
    ):
        """
        Initialize embedding generator.

        Args:
            use_openai: Whether to use OpenAI (defaults to settings)
            openai_api_key: OpenAI API key (defaults to settings)
            model_name: Model name to use (defaults to settings)
        """
        self.use_openai = use_openai if use_openai is not None else settings.use_openai_embeddings

        if self.use_openai:
            if OpenAI is None:
                logger.warning("OpenAI not installed. Falling back to local embeddings.")
                self.use_openai = False
            else:
                api_key = openai_api_key or settings.openai_api_key
                if not api_key:
                    logger.warning("No OpenAI API key provided. Falling back to local embeddings.")
                    self.use_openai = False
                else:
                    self.client = OpenAI(api_key=api_key)
                    self.model_name = model_name or settings.openai_embedding_model
                    logger.info(f"Using OpenAI embeddings: {self.model_name}")

        if not self.use_openai:
            # Use local sentence transformers
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

            self.model_name = model_name or settings.local_embedding_model
            logger.info(f"Loading local embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Local embedding model loaded")

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        if self.use_openai:
            embeddings = self._encode_openai(texts)
        else:
            embeddings = self._encode_local(texts)

        return embeddings[0] if is_single else embeddings

    def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            logger.debug(f"Generating OpenAI embeddings for {len(texts)} texts")

            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Generated {len(embeddings)} OpenAI embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise

    def _encode_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        try:
            logger.debug(f"Generating local embeddings for {len(texts)} texts")

            embeddings = self.model.encode(texts, show_progress_bar=False)

            # Convert numpy arrays to lists
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings[0], (list, tuple)) is False:
                embeddings = [emb.tolist() for emb in embeddings]

            logger.debug(f"Generated {len(embeddings)} local embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise

    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        if self.use_openai:
            # OpenAI embedding dimensions
            if "text-embedding-3-large" in self.model_name:
                return 3072
            elif "text-embedding-3-small" in self.model_name:
                return 1536
            elif "text-embedding-ada-002" in self.model_name:
                return 1536
            else:
                return 1536  # Default
        else:
            # Local model dimension
            return self.model.get_sentence_embedding_dimension()
