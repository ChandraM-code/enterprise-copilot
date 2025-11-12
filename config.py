from pydantic_settings import BaseSettings
from typing import Optional


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
    
    # LLM Configuration
    default_llm: str = "openai"
    openai_model: str = "gpt-3.5-turbo"
    gemini_model: str = "gemini-pro"
    
    # Cache Configuration
    semantic_similarity_threshold: float = 0.85
    rag_similarity_threshold: float = 0.75
    cache_ttl: int = 3600
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

