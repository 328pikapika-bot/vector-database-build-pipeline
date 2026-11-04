"""Mongo-backed repository for image task state."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Optional

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.collection import Collection

from app.config import settings
from app.models.sync_runtime import utcnow
from app.services.image_task_manager import ImageTask, ImageTaskStatus
from app.services.mongo_state_manager import MongoStateManager, get_mongo_state_manager


class ImageTaskRepository:
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
        return manager.image_tasks_collection

    async def ensure_indexes(self) -> None:
        await asyncio.to_thread(self._ensure_indexes_sync)

    def _ensure_indexes_sync(self) -> None:
        collection = self.collection
        collection.create_index([("task_id", ASCENDING)], unique=True, name="ux_task_id")
        collection.create_index(
            [
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
        collection.create_index([("image_id", ASCENDING)], name="idx_image_id")
        collection.create_index(
            [("status", ASCENDING)],
            name="idx_status",
        )
        collection.create_index(
            [
                ("status", ASCENDING),
                ("doc_id", ASCENDING),
                ("record_update_time", DESCENDING),
                ("updated_at", ASCENDING),
            ],
            name="idx_success_doc_time_updated",
        )
        ttl_seconds = int(settings.MONGO_IMAGE_TASK_TTL_DAYS) * 24 * 3600
        if ttl_seconds > 0:
            collection.create_index(
                [("finished_at", ASCENDING)],
                expireAfterSeconds=ttl_seconds,
                name="ttl_finished_at",
            )

    async def upsert_tasks(self, tasks: list[ImageTask | dict[str, Any]]) -> dict[str, int]:
        return await asyncio.to_thread(self._upsert_tasks_sync, tasks)

    def _upsert_tasks_sync(self, tasks: list[ImageTask | dict[str, Any]]) -> dict[str, int]:
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
        download_lane: Optional[str] = "fast",
    ) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(
            self._claim_next_task_sync,
            worker_id,
            lease_seconds,
            exclude_task_ids or set(),
            download_lane,
        )

    async def claim_next_tasks(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        limit: int,
        exclude_task_ids: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        resolved_limit = max(int(limit), 1)
        tasks: list[dict[str, Any]] = []
        excluded = set(exclude_task_ids or set())
        for _ in range(resolved_limit):
            task = await self.claim_next_task(
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                exclude_task_ids=excluded,
            )
            if task is None:
                break
            tasks.append(task)
            excluded.add(str(task.get("_id") or task.get("task_id") or ""))
        return tasks

    def _claim_next_task_sync(
        self,
        worker_id: str,
        lease_seconds: int,
        exclude_task_ids: set[str],
        download_lane: Optional[str],
    ) -> Optional[dict[str, Any]]:
        now = utcnow()
        max_attempts = max(int(settings.IMAGE_TASK_MAX_ATTEMPTS or 1), 1)
        status_filter: dict[str, Any] = {
            "$or": [
                {"status": ImageTaskStatus.PENDING.value},
                {
                    "status": ImageTaskStatus.FAILURE.value,
                    "retryable": True,
                    "retry_count": {"$lt": max_attempts},
                    "next_retry_at": {"$lte": now},
                },
                {
                    "status": ImageTaskStatus.LEASED.value,
                    "lease_expires_at": {"$lte": now},
                    "attempt_count": {"$lt": max_attempts},
                },
            ]
        }
        if download_lane == "slow":
            status_filter["download_lane"] = "slow"
        elif download_lane == "fast":
            status_filter["download_lane"] = {"$ne": "slow"}
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
        return self.collection.find_one_and_update(
            claim_filter,
            {
                "$set": {
                    "status": ImageTaskStatus.LEASED.value,
                    "lease_owner": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "last_claimed_at": now,
                    "updated_at": now,
                },
                "$inc": {"attempt_count": 1},
            },
            sort=[("priority", DESCENDING), ("updated_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    async def mark_task_success(
        self,
        task_id: str,
        *,
        artifact_refs: list[str],
        result_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        await asyncio.to_thread(self._delete_task_sync, task_id)

    async def mark_task_success_minimal(
        self,
        task_id: str,
        *,
        image_id: str,
        success_stage: str,
        milvus_write_mode: str,
        artifact_refs: Optional[list[str]] = None,
    ) -> None:
        await asyncio.to_thread(self._delete_task_sync, task_id)

    def _delete_task_sync(self, task_id: str) -> None:
        self.collection.delete_one({"_id": task_id})

    def _mark_task_success_sync(
        self,
        task_id: str,
        artifact_refs: list[str],
        result_summary: dict[str, Any],
    ) -> None:
        now = utcnow()
        self.collection.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": ImageTaskStatus.SUCCESS.value,
                    "artifact_refs": artifact_refs,
                    "result_summary": result_summary,
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
        max_attempts = max(int(settings.IMAGE_TASK_MAX_ATTEMPTS or 1), 1)
        will_retry = bool(retryable and retry_count < max_attempts)
        next_retry_at = None
        if will_retry:
            next_retry_at = now + timedelta(
                seconds=_compute_retry_delay(
                    retry_count,
                    slow_lane=error_code == "IMAGE_DOWNLOAD_RETRYABLE",
                )
            )

        update_fields = {
            "status": ImageTaskStatus.FAILURE.value,
            "retryable": will_retry,
            "error_code": error_code,
            "error_message": error_message,
            "retry_count": retry_count,
            "max_retries": max_attempts,
            "max_attempts": max_attempts,
            "next_retry_at": next_retry_at,
            "finished_at": now,
            "updated_at": now,
            "lease_owner": None,
            "lease_expires_at": None,
        }
        if error_code == "IMAGE_DOWNLOAD_RETRYABLE":
            update_fields["download_lane"] = "slow"

        self.collection.update_one(
            {"_id": task_id},
            {
                "$set": {
                    **update_fields,
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
                    "status": ImageTaskStatus.SKIPPED.value,
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
            for status in ImageTaskStatus
        }
        retryable_failures = int(
            self.collection.count_documents(
                {
                    "status": ImageTaskStatus.FAILURE.value,
                    "retryable": True,
                }
            )
        )
        total = sum(counts.values())
        return {
            "collection": getattr(
                self.collection,
                "name",
                settings.MONGO_IMAGE_TASK_COLLECTION,
            ),
            "total": total,
            "counts": counts,
            "retryable_failures": retryable_failures,
        }


def _task_to_document(task: ImageTask | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, ImageTask):
        return task.to_mongo_document()
    return dict(task)


def _build_insert_document(document: dict[str, Any], now) -> dict[str, Any]:
    task_id = document["task_id"]
    return {
        "_id": task_id,
        "task_id": task_id,
        "status": document.get("status", ImageTaskStatus.PENDING.value),
        "priority": document.get("priority", 0),
        "retry_count": document.get("retry_count", 0),
        "attempt_count": document.get("attempt_count", 0),
        "artifact_refs": list(document.get("artifact_refs", [])),
        "lease_owner": document.get("lease_owner"),
        "lease_expires_at": document.get("lease_expires_at"),
        "next_retry_at": document.get("next_retry_at"),
        "download_lane": document.get("download_lane", "fast"),
        "created_at": document.get("created_at", now),
    }


def _build_upsert_set(document: dict[str, Any], now) -> dict[str, Any]:
    fields = [
        "doc_id",
        "record_update_time",
        "relative_url",
        "mmd_relative_url",
        "image_id",
        "sequence",
        "doc_title",
        "keyword",
        "accessType",
        "classificationCode",
        "year",
        "hasAuthorname",
        "hasCreateOrganizationname",
        "language",
        "sysDataStatus",
        "hasSourcesysSourceId",
        "hasSourceidSet",
        "title",
        "name",
        "width",
        "height",
        "file_type",
        "size",
        "raw_img_meta",
        "img_meta",
        "context_before",
        "context_after",
        "stage_group",
        "source_item_index",
        "image_index",
    ]
    update = {field: document.get(field) for field in fields if field in document}
    update["stage_group"] = document.get("stage_group", "image")
    update["updated_at"] = now
    return update


def _compute_retry_delay(retry_count: int, *, slow_lane: bool = False) -> float:
    attempt = max(retry_count - 1, 0)
    delay = min(
        settings.RETRY_BASE_DELAY_SEC * (2 ** attempt),
        settings.RETRY_MAX_DELAY_SEC,
    )
    if slow_lane:
        delay = max(delay, int(settings.IMAGE_SLOW_RETRY_DELAY_SEC or 0))
    return delay


_image_task_repository: Optional[ImageTaskRepository] = None


def get_image_task_repository() -> ImageTaskRepository:
    global _image_task_repository
    if _image_task_repository is None:
        _image_task_repository = ImageTaskRepository()
    return _image_task_repository
