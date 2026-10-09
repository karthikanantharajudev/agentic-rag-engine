from storage.vertex_store import VertexVectorStore

def test_search():
    store = VertexVectorStore()
    
    # Dummy query vector matching your embedding dimensions (e.g., 768 zeros or sample vector)
    query_vector = [0.02] * 768  
    
    print("Executing similarity search for tenant...")
    results = store.query(
        query_embedding=query_vector,
        k=2,
        tenant_id="tenant_acme_corp"
    )
    
    print("Search results:", results)

if __name__ == "__main__":
    test_search()