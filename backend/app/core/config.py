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

    # --- Neo4j (Knowledge Graph) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "pageindex123"
    # Set false to fully disable graph features (tree-only retrieval)
    graph_enabled: bool = True

    # --- File storage ---
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 50

    # --- CORS ---
    frontend_origin: str = "http://localhost:5173"

    # --- Indexing concurrency ---
    max_concurrent_indexing: int = 3
    max_concurrent_summaries: int = 4

    # --- Retrieval ---
    max_candidate_documents: int = 15
    max_total_context_nodes: int = 10
    max_hops: int = 3

    # --- Knowledge Graph retrieval ---
    # Max graph-neighbor nodes to pull when expanding context
    graph_expand_limit: int = 5
    # Min confidence (0-1) for extracted relations to store
    kg_min_confidence: float = 0.4

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()