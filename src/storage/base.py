from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ingestion.models import VectorRecord


class BaseVectorStore(ABC):
    """Abstract base class defining the standard interface for vector stores."""

    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> int:
        """Upsert vector records into the store. Returns the count of upserted records."""
        pass

    @abstractmethod
    def query(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """Query the store for nearest neighbors restricted by tenant ID."""
        pass

    @abstractmethod
    def delete_by_tenant(self, tenant_id: str) -> int:
        """Delete all records belonging to a specific tenant."""
        pass