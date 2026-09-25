"""
Central app configuration. Values are read from environment variables
(or a .env file in development) via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Postgres ---
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/pageindex_db"

    # --- Ollama (local LLM) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout_seconds: int = 180

    # --- File storage ---
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 50

    # --- CORS ---
    frontend_origin: str = "http://localhost:5173"

    # --- Indexing concurrency ---
    # How many documents can be indexed in parallel. Parsing is CPU-bound;
    # LLM summary calls are I/O-bound. 3-4 is a safe default for a single
    # local Ollama instance without saturating it.
    max_concurrent_indexing: int = 3

    # --- Summary concurrency ---
    # Max parallel LLM summary calls within a single document (semaphore).
    max_concurrent_summaries: int = 4

    # --- Retrieval ---
    max_candidate_documents: int = 15
    max_total_context_nodes: int = 10
    max_hops: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared settings instance, import this everywhere.
settings = Settings()