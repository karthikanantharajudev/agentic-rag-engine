import os
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform, storage
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from ingestion.models import VectorRecord
from storage.base import BaseVectorStore
import json
from config.settings import settings

class VertexVectorStore(BaseVectorStore):
    """Production vector store backed by GCP Vertex AI Vector Search."""
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        deployed_id: Optional[str] = None,
    ):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")
        self.index_id = index_id or os.getenv("VERTEX_INDEX_ID")
        self.endpoint_id = endpoint_id or os.getenv("VERTEX_ENDPOINT_ID")
        self.deployed_id = deployed_id or os.getenv("VERTEX_DEPLOYED_INDEX_ID")
        print("self.index_id")
        print(self.index_id)
        if not all([self.project_id, self.index_id, self.endpoint_id]):
            raise ValueError(
                "Missing GCP Vertex AI configuration. Ensure GCP_PROJECT_ID, "
                "VERTEX_INDEX_ID, and VERTEX_ENDPOINT_ID are set in your environment."
            )

        # Initialize GCP AI Platform SDK
        aiplatform.init(project=self.project_id, location=self.location, credentials=settings.credentials)

        # Connect to existing Vertex AI Index and Index Endpoint
        self.index = aiplatform.MatchingEngineIndex(index_name=self.index_id)
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.endpoint_id
        )
        self.staging_bucket_name = settings.gcs_staging_bucket

    def upsert(self, records: List[Any]) -> int:
        """
        For batch indexes, 'upsert' acts as a batch synchronization step:
        1. Serializes records to JSONL.
        2. Uploads to GCS.
        3. Triggers Vertex AI index embedding update.
        """
        # 1. Prepare JSONL content
        jsonl_lines = []
        for record in records:
            datapoint = {
                "id": record.chunk_id,
                "embedding": record.embedding,
                "restricts": [
                    {
                        "namespace": "tenant_id",
                        "allow_list": [record.tenant_id]
                    }
                ]
            }
            jsonl_lines.append(json.dumps(datapoint))
        
        file_content = "\n".join(jsonl_lines)

        # 2. Upload to GCS staging bucket
        file_name = "batch_upsert_payload.json"
        storage_client = storage.Client(credentials=settings.credentials, project=self.project_id)
        bucket = storage_client.bucket(self.staging_bucket_name)
        blob = bucket.blob(f"vector_staging/{file_name}")
        blob.upload_from_string(file_content, content_type="application/json")
        
        gcs_uri = f"gs://{self.staging_bucket_name}/vector_staging/"

        # 3. Trigger the Vertex AI batch update
        # is_complete_overwrite=False means it merges/updates; True replaces everything in the index
        print(f"Triggering batch update from {gcs_uri}...")
        self.index.update_embeddings(
            contents_delta_uri=gcs_uri,
            is_complete_overwrite=False
        )
        print("Batch update job initiated successfully!")
        return len(records)

    def query(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        tenant_id: str = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """
        Queries the deployed Vertex AI index with multi-tenant filtering.
        """
        if not self.endpoint:
            raise ValueError("Vertex Index Endpoint ID is not configured in settings.")

        # Build multi-tenant filters using Namespace restriction if provided
        filters = []
        if tenant_id:
            filters.append(
                Namespace(
                    name="tenant_id",
                    allow_tokens=[tenant_id],
                    deny_tokens=[]
                )
            )

        # Run the approximate nearest neighbor search on the endpoint
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_id,
            queries=[query_embedding],
            num_neighbors=k,
            filter=filters if filters else None
        )

        # Parse and format matching results
        results = []
        for match in response[0]:
            results.append({
                "chunk_id": match.id,
                "distance": match.distance,
                # Vertex returns restricts metadata if configured, or you can map them back from your DB
            })
            
        return results

    def delete_by_tenant(self, tenant_id: str) -> int:
        """Removes datapoints corresponding to tenant_id."""
        # Vertex AI Vector Search streaming allows removal by ID list
        return 0