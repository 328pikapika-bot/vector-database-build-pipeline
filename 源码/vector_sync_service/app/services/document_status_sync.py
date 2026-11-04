"""Mirror MongoDB overall_status to Milvus sync_status for document completeness visibility."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.config import settings
from app.observability.logger import get_logger
from app.services.milvus_service import get_milvus_service
from app.services.mongo_state_manager import get_mongo_state_manager

logger = get_logger(__name__)

# Terminal statuses that should be mirrored to Milvus
_SYNC_STATUS_MAP = {
    "SUCCESS": "SUCCESS",
    "CORE_SUCCESS": "CORE_SUCCESS",
    "PARTIAL_FAILURE": "PARTIAL_FAILURE",
}
_DEFAULT_SYNC_STATUS = "PROCESSING"


class DocumentStatusSync:
    """Reads MongoDB overall_status and mirrors it to Milvus sync_status."""

    def __init__(self) -> None:
        self._mongo = get_mongo_state_manager()
        self._milvus = get_milvus_service()
        self._collection = settings.MILVUS_COLLECTION

    async def sync_one(self, doc_id: str) -> bool:
        """Sync sync_status for a single document from MongoDB to Milvus."""
        try:
            status_doc = await self._mongo.get_status(doc_id)
        except Exception:
            logger.warning("document_status_sync_mongo_read_failed", doc_id=doc_id)
            return False

        if status_doc is None:
            return False

        overall = status_doc.get("overall_status", "")
        sync_status = self._to_sync_status(overall)
        if sync_status is None:
            return False

        try:
            return await asyncio.to_thread(
                self._milvus.update_sync_status,
                self._collection,
                doc_id,
                sync_status,
            )
        except Exception:
            logger.warning("document_status_sync_milvus_write_failed", doc_id=doc_id)
            return False

    async def sync_batch(self, doc_ids: list[str]) -> dict[str, int]:
        """Batch sync sync_status for multiple documents.

        Queries MongoDB for each doc's overall_status, groups by status,
        then batch-updates Milvus.
        """
        if not doc_ids:
            return {"synced": 0, "skipped": 0}

        updates: list[tuple[str, str]] = []
        for doc_id in doc_ids:
            try:
                status_doc = await self._mongo.get_status(doc_id)
            except Exception:
                continue
            if status_doc is None:
                continue
            overall = status_doc.get("overall_status", "")
            sync_status = self._to_sync_status(overall)
            if sync_status is None:
                continue
            updates.append((doc_id, sync_status))

        if not updates:
            return {"synced": 0, "skipped": len(doc_ids)}

        try:
            result = await asyncio.to_thread(
                self._milvus.update_sync_status_batch,
                self._collection,
                updates,
            )
            return {"synced": result.get("updated", 0), "skipped": result.get("skipped", 0)}
        except Exception:
            logger.warning("document_status_sync_batch_failed", count=len(updates))
            return {"synced": 0, "skipped": len(doc_ids)}

    @staticmethod
    def _to_sync_status(overall_status: str) -> Optional[str]:
        if not overall_status:
            return None
        return _SYNC_STATUS_MAP.get(overall_status, _DEFAULT_SYNC_STATUS)


_document_status_sync: Optional[DocumentStatusSync] = None


def get_document_status_sync() -> DocumentStatusSync:
    global _document_status_sync
    if _document_status_sync is None:
        _document_status_sync = DocumentStatusSync()
    return _document_status_sync
