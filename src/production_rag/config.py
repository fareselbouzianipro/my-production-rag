from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, overridable via environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data/raw")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_size: int = 384

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "acme_docs"

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"


settings = Settings()
