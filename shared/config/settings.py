"""Configuration settings for QuantAI platform."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # OpenAI Configuration
    openai_api_key: str = ""

    # Vector Store Configuration
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "quant_knowledge"

    # Embedding Configuration
    use_openai_embeddings: bool = True  # Use OpenAI for best quality (recommended)
    openai_embedding_model: str = "text-embedding-3-large"  # Latest OpenAI embedding model
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback local model

    # LLM Configuration for RAG
    llm_model: str = "gpt-4o"  # Latest GPT-4 Omni model (best for reasoning)
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000

    # Retrieval Configuration
    max_retrieval_results: int = 20
    context_token_limit: int = 8000

    # Logging
    log_level: str = "INFO"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
