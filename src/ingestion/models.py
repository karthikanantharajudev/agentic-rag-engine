from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    """Data model representing a chunk of a document prior to embedding"""
    chunk_id: str = Field(..., description="SHA-256 hash of (tenant_id + doc_id + content)")
    tenant_id: str = Field(..., description="Multi-tenant tenant identifier for data isolation")
    doc_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Text content of this chunk")
    chunk_index: int = Field(..., ge=0, description="Sequential index of the chunk within the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

class VectorRecord(BaseModel):
    """Data model representing a chunk with its computed embedding vector"""
    chunk_id: str = Field(..., description="Matching chunk ID hash")
    tenant_id: str = Field(..., description="Tenant identifier for strict filtering")
    doc_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Text content")
    embedding: List[float] = Field(..., description="768-dimensional float array")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata carried over from DocumentChunk")