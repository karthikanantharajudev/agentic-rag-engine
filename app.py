from src.ingestion.pipeline import IngestionPipeline


def main():
    pipeline = IngestionPipeline()
    sample_text = (
        "Apache Kafka uses consumer groups to allow a pool of processes to divide "
        "the work of processing and consuming data. Partition ownership is rebalanced "
        "dynamically when members join or leave the group. Kafka topic partitions ensure "
        "horizontal scaling and ordered messaging guarantees across consumer instances."
    )

    records = pipeline.process_document(
        tenant_id="tenant_acme",
        doc_id="kafka_guide_v1",
        text=sample_text,
        extra_metadata={"category": "infrastructure", "environment": "production"}
    )

    print("--- Ingestion Pipeline Integration Output ---")
    print(f"✅ Records generated: {len(records)}")
    print(f"🔑 Chunk ID (SHA-256): {records[0].chunk_id}")
    print(f"🏢 Tenant ID: {records[0].tenant_id}")
    print(f"📊 Vector Dimension: {len(records[0].embedding)}")
    print(f"📄 First 3 Floats: {records[0].embedding[:3]}")


if __name__ == "__main__":
    main()