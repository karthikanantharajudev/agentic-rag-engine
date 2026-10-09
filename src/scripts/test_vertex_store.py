import os
from ingestion.models import VectorRecord
from storage.vertex_store import VertexVectorStore

def test_vertex_integration():
    print("Initializing VertexVectorStore...")
    store = VertexVectorStore()

    tenant_id = "tenant_acme_corp"
    test_embedding = [0.01] * 768  # 768 dimensions matching gemini-embedding-001

    record = VectorRecord(
        chunk_id="chunk_test_123",
        tenant_id=tenant_id,
        doc_id="doc_999",
        content="Hello Vertex AI Vector Search Multi-Tenant World",
        embedding=test_embedding,
        metadata={"category": "engineering"}
    )

    print(f"Upserting 1 record for tenant: {tenant_id}...")
    count = store.upsert([record])
    print(f"Successfully upserted {count} record.")

    print("Querying index for nearest neighbors...")
    results = store.query(
        tenant_id=tenant_id,
        query_vector=test_embedding,
        top_k=3
    )

    print(f"Query returned {len(results)} matches:")
    for res in results:
        print(f" - Chunk ID: {res.chunk_id}, Distance/Score: {res.metadata.get('distance')}")

if __name__ == "__main__":
    test_vertex_integration()