import hashlib
from typing import List, Dict, Any
from ingestion.models import DocumentChunk
from config.settings import settings


class SlidingWindowChunker:
    def __init__(
            self, 
            chunk_size: int = settings.default_chunk_size, 
            chunk_overlap: int = settings.default_chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _generate_chunk_id(self, tenant_id: str, doc_id: str, content: str) -> str:
        """Generates a deterministic SHA-256 hash string."""
        raw_key = f"{tenant_id}:{doc_id}:{content}".encode('utf-8')
        return hashlib.sha256(raw_key).hexdigest()

    def split_text(
        self, 
        tenant_id: str, 
        doc_id: str, 
        text: str, 
        extra_metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """Splits document text into overlapping DocumentChunk objects."""
        words = text.split()
        chunks: List[DocumentChunk] = []
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        # Safety check to prevent infinite loops
        if step <= 0:
            raise ValueError("chunk_size must be strictly greater than chunk_overlap.")

        for i in range(0, len(words), step):
            window_words = words[i : i + self.chunk_size]
            chunk_content = " ".join(window_words)

            chunk_id = self._generate_chunk_id(tenant_id, doc_id, chunk_content)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                doc_id=doc_id,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata=extra_metadata or {}
            )
            chunks.append(chunk)
            chunk_index += 1

            # Stop when we reach the end of the words list
            if i + self.chunk_size >= len(words):
                break

        return chunks