from typing import List, Dict, Any
from ingestion.models import DocumentChunk, VectorRecord
from ingestion.chunker import SlidingWindowChunker
from ingestion.embedder import GeminiEmbedder


class IngestionPipeline:
    """Orchestrates document chunking and vector embedding generation."""

    def __init__(self):
        self.chunker = SlidingWindowChunker()
        self.embedder = GeminiEmbedder()

    def process_document(
        self, 
        tenant_id: str, 
        doc_id: str, 
        text: str, 
        extra_metadata: Dict[str, Any] = None
    ) -> List[VectorRecord]:
        """Splits document into chunks and generates embeddings for each."""
        # 1. Chunk the document
        chunks: List[DocumentChunk] = self.chunker.split_text(
            tenant_id=tenant_id,
            doc_id=doc_id,
            text=text,
            extra_metadata=extra_metadata
        )

        # 2. Embed each chunk
        records: List[VectorRecord] = []
        for chunk in chunks:
            embedding_vector = self.embedder.generate_embedding(chunk.content)

            record = VectorRecord(
                chunk_id=chunk.chunk_id,
                tenant_id=chunk.tenant_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                embedding=embedding_vector,
                metadata=chunk.metadata
            )
            records.append(record)

        return records