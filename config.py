from pydantic_settings import BaseSettings
from typing import Optional


def debug_print(message: str, settings_obj=None):
    """Print debug messages if debug mode is enabled"""
    if settings_obj is None:
        # Import here to avoid circular dependency
        from config import settings as default_settings
        settings_obj = default_settings

    if settings_obj.debug:
        print(f"[DEBUG] {message}")


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_https: bool = False  # Set to True for cloud/production Qdrant
    qdrant_prefer_grpc: bool = False  # Set to True to use gRPC instead of HTTP
    
    # LLM Configuration
    default_llm: str = "openai"
    openai_model: str = "gpt-3.5-turbo"
    gemini_model: str = "gemini-pro"
    
    # Cache Configuration
    semantic_similarity_threshold: float = 0.85
    rag_similarity_threshold: float = 0.75
    cache_ttl: int = 3600

    # Chunking Configuration (for Layer 2 RAG)
    enable_chunking: bool = True
    chunk_size: int = 512  # Size in tokens
    chunk_overlap: int = 50  # Overlap between chunks in tokens
    chunking_strategy: str = "recursive"  # Options: "fixed", "recursive", "semantic"

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Debug Configuration
    debug: bool = False  # Set to True to enable debug logging
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

