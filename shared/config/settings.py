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

    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.1

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
