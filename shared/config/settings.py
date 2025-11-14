"""
Configuration management using Pydantic Settings
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="quantai", description="Database name")
    postgres_user: str = Field(default="quantai", description="Database user")
    postgres_password: str = Field(default="quantai", description="Database password")

    model_config = SettingsConfigDict(env_prefix="DB_")

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def async_postgres_url(self) -> str:
        """Construct async PostgreSQL connection URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class Neo4jSettings(BaseSettings):
    """Neo4j graph database configuration"""

    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j user")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")

    model_config = SettingsConfigDict(env_prefix="NEO4J_")


class ChromaSettings(BaseSettings):
    """ChromaDB vector store configuration"""

    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    chroma_persist_directory: str = Field(default="./data/chroma", description="Persistence directory")

    model_config = SettingsConfigDict(env_prefix="CHROMA_")


class RedisSettings(BaseSettings):
    """Redis cache and queue configuration"""

    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: Optional[str] = Field(default=None, description="Redis password")

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class LLMSettings(BaseSettings):
    """LLM provider configuration"""

    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    default_llm_provider: str = Field(default="openai", description="Default LLM provider")
    default_model: str = Field(default="gpt-4-turbo-preview", description="Default model")

    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    max_tokens: int = Field(default=4096, description="Max tokens per request")
    temperature: float = Field(default=0.1, description="LLM temperature")

    model_config = SettingsConfigDict(env_prefix="LLM_")


class APISettings(BaseSettings):
    """API server configuration"""

    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Enable auto-reload")
    api_workers: int = Field(default=1, description="Number of workers")

    cors_origins: list[str] = Field(default=["*"], description="CORS origins")

    model_config = SettingsConfigDict(env_prefix="API_")


class Settings(BaseSettings):
    """Main application settings"""

    app_name: str = Field(default="QuantAI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=True, description="Debug mode")

    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="logs/quantai.log", description="Log file path")

    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__"
    )


# Global settings instance
settings = Settings()
