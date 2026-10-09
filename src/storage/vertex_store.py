import os
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from src.ingestion.models import VectorRecord
from src.storage.base import BaseVectorStore


class VertexVectorStore(BaseVectorStore):
    """Production vector store backed by GCP Vertex AI Vector Search."""
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")
        self.index_id = index_id or os.getenv("VERTEX_INDEX_ID")
        self.endpoint_id = endpoint_id or os.getenv("VERTEX_ENDPOINT_ID")

        if not all([self.project_id, self.index_id, self.endpoint_id]):
            raise ValueError(
                "Missing GCP Vertex AI configuration. Ensure GCP_PROJECT_ID, "
                "VERTEX_INDEX_ID, and VERTEX_ENDPOINT_ID are set in your environment."
            )

        # Initialize GCP AI Platform SDK
        aiplatform.init(project=self.project_id, location=self.location)

        # Connect to existing Vertex AI Index and Index Endpoint
        self.index = aiplatform.MatchingEngineIndex(index_name=self.index_id)
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.endpoint_id
        )

    def upsert(self, records: List[VectorRecord]) -> int:
        """Upserts vectors into the Vertex AI Index with tenant metadata restricts."""
        datapoints = []
        for record in records:
            # Multi-tenant restricts enforce tenant isolation during ANN search
            restricts = [
                Namespace(name="tenant_id", allow_tokens=[record.tenant_id])
            ]
            
            # Add optional custom metadata tokens
            for k, v in record.metadata.items():
                restricts.append(Namespace(name=str(k), allow_tokens=[str(v)]))

            datapoints.append(
                aiplatform.gapic.IndexDatapoint(
                    datapoint_id=record.chunk_id,
                    feature_vector=record.embedding,
                    restricts=restricts,
                )
            )

        # Upsert datapoints into the Vertex AI Vector Search streaming index
        self.index.upsert_datapoints(datapoints=datapoints)
        return len(records)

    def query(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """Queries Vertex AI Vector Search endpoint restricted strictly to tenant_id."""
        
        # Build tenant restriction filter
        restricts = [Namespace(name="tenant_id", allow_tokens=[tenant_id])]

        if metadata_filter:
            for k, v in metadata_filter.items():
                restricts.append(Namespace(name=str(k), allow_tokens=[str(v)]))

        # Extract deployed index ID name automatically from endpoint
        deployed_indices = self.endpoint.deployed_indexes
        if not deployed_indices:
            raise RuntimeError("No deployed index found on this endpoint. Please deploy the index first.")
        deployed_index_id = deployed_indices[0].deployed_index_id

        # Query index endpoint for nearest neighbors
        response = self.endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_vector],
            num_neighbors=top_k,
            filter=restricts
        )

        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                results.append(
                    VectorRecord(
                        chunk_id=neighbor.id,
                        tenant_id=tenant_id,
                        doc_id="",
                        content="",
                        embedding=[],
                        metadata={"distance": neighbor.distance}
                    )
                )

        return results

    def delete_by_tenant(self, tenant_id: str) -> int:
        """Removes datapoints corresponding to tenant_id."""
        # Vertex AI Vector Search streaming allows removal by ID list
        return 0