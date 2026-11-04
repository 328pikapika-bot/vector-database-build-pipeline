"""Mongo-backed repository for full-text task state."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Optional

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.collection import Collection

from app.config import settings
from app.models.sync_runtime import parse_record_update_time, utcnow
from app.services.fulltext_task_manager import FullTextTask, FullTextTaskStatus
from app.services.mongo_state_manager import MongoStateManager, get_mongo_state_manager


class FullTextTaskRepository:
    def __init__(
        self,
        *,
        collection: Optional[Collection] = None,
        manager: Optional[MongoStateManager] = None,
    ) -> None:
        self._collection = collection
        self._manager = manager

    @property
    def collection(self) -> Collection:
        if self._collection is not None:
            return self._collection
        manager = self._manager or get_mongo_state_manager()
        return manager.fulltext_tasks_collection

    async def ensure_indexes(self) -> None:
        await asyncio.to_thread(self._ensure_indexes_sync)

    def _ensure_indexes_sync(self) -> None:
        collection = self.collection
        collection.create_index([("task_id", ASCENDING)], unique=True, name="ux_task_id")
        collection.create_index(
            [
                ("claim_rank", ASCENDING),
                ("status", ASCENDING),
                ("next_retry_at", ASCENDING),
                ("lease_expires_at", ASCENDING),
                ("priority", DESCENDING),
                ("updated_at", ASCENDING),
            ],
            name="idx_claim",
        )
        collection.create_index(
            [("doc_id", ASCENDING), ("record_update_time", DESCENDING)],
            name="idx_doc_time",
        )
        collection.create_index(
            [("status", ASCENDING)],
            name="idx_status",
        )
        ttl_seconds = int(settings.MONGO_FULLTEXT_TASK_TTL_DAYS) * 24 * 3600
        if ttl_seconds > 0:
            collection.create_index(
                [("finished_at", ASCENDING)],
                expireAfterSeconds=ttl_seconds,
                name="ttl_finished_at",
            )

    async def upsert_tasks(self, tasks: list[FullTextTask | dict[str, Any]]) -> dict[str, int]:
        return await asyncio.to_thread(self._upsert_tasks_sync, tasks)

    def _upsert_tasks_sync(self, tasks: list[FullTextTask | dict[str, Any]]) -> dict[str, int]:
        matched = 0
        upserted = 0
        for task in tasks:
            document = _task_to_document(task)
            task_id = document["task_id"]
            now = utcnow()
            insert_doc = _build_insert_document(document, now)
            update_doc = _build_upsert_set(document, now)

            result = self.collection.update_one(
                {"_id": task_id},
                {
                    "$setOnInsert": insert_doc,
                    "$set": update_doc,
                },
                upsert=True,
            )
            matched += int(getattr(result, "matched_count", 0))
            if getattr(result, "upserted_id", None) is not None:
                upserted += 1
        return {"matched": matched, "upserted": upserted}

    async def claim_next_task(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        exclude_task_ids: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(
            self._claim_next_task_sync,
            worker_id,
            lease_seconds,
            exclude_task_ids or set(),
        )

    def _claim_next_task_sync(
        self,
        worker_id: str,
        lease_seconds: int,
        exclude_task_ids: set[str],
    ) -> Optional[dict[str, Any]]:
        now = utcnow()
        status_filter: dict[str, Any] = {
            "$or": [
                {"status": FullTextTaskStatus.PENDING.value},
                {
                    "status": FullTextTaskStatus.FAILURE.value,
                    "retryable": True,
                    "next_retry_at": {"$lte": now},
                },
                {
                    "status": FullTextTaskStatus.LEASED.value,
                    "lease_expires_at": {"$lte": now},
                },
            ]
        }
        claim_created_after = _fulltext_claim_created_after()
        if claim_created_after is not None:
            status_filter = {
                "$and": [
                    status_filter,
                    {"created_at": {"$gt": claim_created_after}},
                ]
            }
        claim_filter: dict[str, Any]
        if exclude_task_ids:
            claim_filter = {
                "$and": [
                    status_filter,
                    {"_id": {"$nin": sorted(exclude_task_ids)}},
                ]
            }
        else:
            claim_filter = status_filter
        claimed = self.collection.find_one_and_update(
            claim_filter,
            {
                "$set": {
                    "status": FullTextTaskStatus.LEASED.value,
                    "claim_rank": 1,
                    "lease_owner": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "last_claimed_at": now,
                    "updated_at": now,
                },
                "$inc": {"attempt_count": 1},
            },
            sort=[
                ("claim_rank", ASCENDING),
                ("priority", DESCENDING),
                ("updated_at", ASCENDING),
            ],
            return_document=ReturnDocument.AFTER,
        )
        return claimed

    async def mark_task_success(
        self,
        task_id: str,
        *,
        artifact_refs: list[str],
        chunk_count: int,
    ) -> None:
        await asyncio.to_thread(self._delete_task_sync, task_id)

    def _delete_task_sync(self, task_id: str) -> None:
        self.collection.delete_one({"_id": task_id})

    def _mark_task_success_sync(
        self,
        task_id: str,
        artifact_refs: list[str],
        chunk_count: int,
    ) -> None:
        now = utcnow()
        self.collection.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": FullTextTaskStatus.SUCCESS.value,
                    "claim_rank": 9,
                    "artifact_refs": artifact_refs,
                    "chunk_count": chunk_count,
                    "finished_at": now,
                    "updated_at": now,
                    "retryable": False,
                    "next_retry_at": None,
                    "error_code": None,
                    "error_message": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            },
        )

    async def mark_task_failure(
        self,
        task_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> None:
        await asyncio.to_thread(
            self._mark_task_failure_sync,
            task_id,
            error_code,
            error_message,
            retryable,
        )

    def _mark_task_failure_sync(
        self,
        task_id: str,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> None:
        now = utcnow()
        existing = self.collection.find_one({"_id": task_id}) or {}
        retry_count = int(existing.get("retry_count", 0)) + 1
        next_retry_at = None
        if retryable and retry_count < settings.MAX_RETRY:
            next_retry_at = now + timedelta(seconds=_compute_retry_delay(retry_count))
        will_retry = bool(retryable and next_retry_at is not None)

        self.collection.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": FullTextTaskStatus.FAILURE.value,
                    "claim_rank": 2 if will_retry else 9,
                    "retryable": bool(retryable),
                    "error_code": error_code,
                    "error_message": error_message,
                    "retry_count": retry_count,
                    "max_retries": settings.MAX_RETRY,
                    "next_retry_at": next_retry_at,
                    "finished_at": now,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            },
        )

    async def mark_task_skipped(self, task_id: str, *, reason: str) -> None:
        await asyncio.to_thread(self._mark_task_skipped_sync, task_id, reason)

    def _mark_task_skipped_sync(self, task_id: str, reason: str) -> None:
        now = utcnow()
        self.collection.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": FullTextTaskStatus.SKIPPED.value,
                    "claim_rank": 9,
                    "skip_reason": reason,
                    "finished_at": now,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            },
        )

    async def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self.collection.find_one, {"_id": task_id})

    async def get_summary(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_summary_sync)

    def _get_summary_sync(self) -> dict[str, Any]:
        counts = {
            status.value: int(self.collection.count_documents({"status": status.value}))
            for status in FullTextTaskStatus
        }
        retryable_failures = int(
            self.collection.count_documents(
                {
                    "status": FullTextTaskStatus.FAILURE.value,
                    "retryable": True,
                }
            )
        )
        total = sum(counts.values())
        return {
            "collection": getattr(
                self.collection,
                "name",
                settings.MONGO_FULLTEXT_TASK_COLLECTION,
            ),
            "total": total,
            "counts": counts,
            "retryable_failures": retryable_failures,
        }


def _task_to_document(task: FullTextTask | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, FullTextTask):
        return task.to_mongo_document()
    return dict(task)


def _build_insert_document(document: dict[str, Any], now) -> dict[str, Any]:
    task_id = document["task_id"]
    return {
        "_id": task_id,
        "task_id": task_id,
        "status": document.get("status", FullTextTaskStatus.PENDING.value),
        "claim_rank": _claim_rank_for_status(
            str(document.get("status", FullTextTaskStatus.PENDING.value)),
            retryable=bool(document.get("retryable", False)),
        ),
        "priority": document.get("priority", 0),
        "retry_count": document.get("retry_count", 0),
        "attempt_count": document.get("attempt_count", 0),
        "artifact_refs": list(document.get("artifact_refs", [])),
        "chunk_count": document.get("chunk_count", 0),
        "lease_owner": document.get("lease_owner"),
        "lease_expires_at": document.get("lease_expires_at"),
        "next_retry_at": document.get("next_retry_at"),
        "created_at": document.get("created_at", now),
    }


def _build_upsert_set(document: dict[str, Any], now) -> dict[str, Any]:
    fields = [
        "doc_id",
        "record_update_time",
        "relative_url",
        "title",
        "keyword",
        "accessType",
        "classificationCode",
        "year",
        "hasAuthorname",
        "hasCreateOrganizationname",
        "language",
        "stage_group",
        "source_item_index",
    ]
    update = {field: document.get(field) for field in fields if field in document}
    update["doc_id"] = document.get("doc_id", "")
    update["record_update_time"] = document.get("record_update_time", "")
    update["relative_url"] = document.get("relative_url", "")
    update["stage_group"] = document.get("stage_group", "fulltext")
    update["source_item_index"] = document.get("source_item_index", 0)
    update["updated_at"] = now
    return update


def _claim_rank_for_status(status: str, *, retryable: bool = False) -> int:
    if status == FullTextTaskStatus.PENDING.value:
        return 0
    if status == FullTextTaskStatus.LEASED.value:
        return 1
    if status == FullTextTaskStatus.FAILURE.value and retryable:
        return 2
    return 9


def _fulltext_claim_created_after():
    raw_value = str(getattr(settings, "FULLTEXT_WORKER_CLAIM_CREATED_AFTER", "") or "").strip()
    if not raw_value:
        return None
    parsed = parse_record_update_time(raw_value)
    if parsed is None:
        raise ValueError(
            "FULLTEXT_WORKER_CLAIM_CREATED_AFTER must be an ISO datetime or epoch timestamp"
        )
    return parsed


def _compute_retry_delay(retry_count: int) -> float:
    attempt = max(retry_count - 1, 0)
    return min(
        settings.RETRY_BASE_DELAY_SEC * (2 ** attempt),
        settings.RETRY_MAX_DELAY_SEC,
    )


_fulltext_task_repository: Optional[FullTextTaskRepository] = None


def get_fulltext_task_repository() -> FullTextTaskRepository:
    global _fulltext_task_repository
    if _fulltext_task_repository is None:
        _fulltext_task_repository = FullTextTaskRepository()
    return _fulltext_task_repository
