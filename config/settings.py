import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Gemini configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    embedding_mode: str = "text-embedding-004"
    embedding_dimension: int = 768

    # Default Ingestion limits
    default_chunk_size: int = 200
    default_chunk_overlap:int = 40

    # Vector DB / Multitenancy defaults
    default_collection: str = "multitenant_knowledge_base"

settings = Settings()