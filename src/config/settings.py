import os
from pydantic import BaseModel
from typing import Optional
from google.oauth2 import service_account
import json

class Settings(BaseModel):
    # Gemini configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gcs_staging_bucket: str = os.getenv("GCS_STAGING_BUCKET", "")
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768

    # Default Ingestion limits
    default_chunk_size: int = 200
    default_chunk_overlap:int = 40

    # Vector DB / Multitenancy defaults
    default_collection: str = "multitenant_knowledge_base"

    google_application_credentials_json: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")

    @property
    def credentials(self) -> Optional[service_account.Credentials]:
        """Parses the raw JSON string secret into an in-memory Google Credentials object."""
        if self.google_application_credentials_json:
            creds_info = json.loads(self.google_application_credentials_json)
            return service_account.Credentials.from_service_account_info(creds_info)
        return None

settings = Settings()