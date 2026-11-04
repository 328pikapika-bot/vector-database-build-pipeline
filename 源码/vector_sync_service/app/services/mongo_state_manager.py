"""MongoDB 状态管理与失败补偿存储。"""
from __future__ import annotations

import asyncio
import gzip
import json
import random
import threading
import traceback
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any, Optional

from bson import Binary, ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError

from app.config import settings
from app.models.sync_runtime import (
    ArtifactType,
    ErrorType,
    FailureRef,
    FailureStatus,
    ReplayMode,
    ScopeType,
    SourceEventMeta,
    StageExecutionStatus,
    SyncOverallStatus,
    SyncStage,
    normalize_record_update_time,
    parse_record_update_time,
    sha256_hex,
    utcnow,
)
from app.observability.logger import get_logger
from app.observability.metrics import failure_pending_count, gave_up_total

logger = get_logger(__name__)

_dump_lock = threading.Lock()

PIPELINE_STATUS_NOT_PRESENT = "NOT_PRESENT"
PIPELINE_STATUS_PENDING = "PENDING"
PIPELINE_STATUS_PROCESSING = "PROCESSING"
PIPELINE_STATUS_SUCCESS = "SUCCESS"
PIPELINE_STATUS_PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
PIPELINE_STATUS_FAILURE = "FAILURE"
PIPELINE_STATUS_SKIPPED = "SKIPPED"


class MongoPersistenceError(RuntimeError):
    """Mongo 写入失败，当前消息不允许安全提交。"""


class MongoStateManager:
    """负责 Mongo 连接、状态、失败队列和 artifact 持久化。"""

    def __init__(self) -> None:
        self._client: Optional[MongoClient] = None
        self._db = None
        self._connected = False
        self._worker_id = f"{settings.KAFKA_GROUP_ID}:{random.randint(1000, 9999)}"

    @property
    def connected(self) -> bool:
        return self._connected and self._client is not None and self._db is not None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def db(self):
        """Expose the selected database for monitoring and legacy helpers."""
        self._require_sync()
        return self._db

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect_sync)

    def _connect_sync(self) -> None:
        client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=settings.MONGO_SERVER_SELECTION_TIMEOUT_MS,
            connectTimeoutMS=settings.MONGO_CONNECT_TIMEOUT_MS,
            socketTimeoutMS=settings.MONGO_SOCKET_TIMEOUT_MS,
        )
        client.admin.command("ping")
        self._client = client
        self._db = client[settings.MONGO_DB]
        self._ensure_indexes_sync()
        self._connected = True
        self._refresh_pending_count_sync()
        logger.info("mongo_state_manager_connected", database=settings.MONGO_DB)

    async def ping(self) -> bool:
        try:
            await asyncio.to_thread(self._ping_sync)
            self._connected = True
            return True
        except Exception as error:
            self._connected = False
            logger.warning("mongo_ping_failed", error=str(error))
            return False

    def _ping_sync(self) -> None:
        self._require_sync()
        self._client.admin.command("ping")

    async def close(self) -> None:
        if self._client is None:
            return
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._db = None
        self._connected = False
        logger.info("mongo_state_manager_closed")

    async def get_status(self, doc_id: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._get_status_sync, doc_id)

    def _get_status_sync(self, doc_id: str) -> Optional[dict[str, Any]]:
        self._require_sync()
        return self.status_collection.find_one({"_id": doc_id})

    async def find_terminal_status_docs(
        self, *, since_minutes: int = 10, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Find recently updated documents with terminal overall_status."""
        return await asyncio.to_thread(
            self._find_terminal_status_docs_sync, since_minutes, limit
        )

    def _find_terminal_status_docs_sync(
        self, since_minutes: int, limit: int
    ) -> list[dict[str, Any]]:
        self._require_sync()
        cutoff = utcnow() - timedelta(minutes=since_minutes)
        terminal_statuses = ["SUCCESS", "CORE_SUCCESS", "PARTIAL_FAILURE"]
        cursor = self.status_collection.find(
            {
                "overall_status": {"$in": terminal_statuses},
                "updated_at": {"$gte": cutoff},
            },
            {"_id": 1, "overall_status": 1},
        ).limit(limit)
        return list(cursor)

    async def initialize_status(
        self,
        *,
        doc_id: str,
        record_update_time: str,
        sub_doc_type: str,
        event_type: str,
    ) -> None:
        await asyncio.to_thread(
            self._initialize_status_sync,
            doc_id,
            record_update_time,
            sub_doc_type,
            event_type,
        )

    def _initialize_status_sync(
        self,
        doc_id: str,
        record_update_time: str,
        sub_doc_type: str,
        event_type: str,
    ) -> None:
        self._require_sync()
        now = utcnow()
        self.status_collection.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "doc_id": doc_id,
                    "sub_doc_type": sub_doc_type,
                    "latest_record_update_time": record_update_time,
                    "latest_event_type": event_type,
                    "overall_status": SyncOverallStatus.PROCESSING.value,
                    "core_status": PIPELINE_STATUS_PROCESSING,
                    "document_status": PIPELINE_STATUS_PROCESSING,
                    "fulltext_status": PIPELINE_STATUS_PENDING,
                    "image_status": PIPELINE_STATUS_PENDING,
                    "image_required": None,
                    "fulltext_required": None,
                    "current_stage": None,
                    "last_success_stage": None,
                    "stage_statuses": {},
                    "progress": {
                        "article_count": 0,
                        "chunk_count_expected": 0,
                        "chunk_count_written": 0,
                        "image_count_expected": 0,
                        "image_count_written": 0,
                        "document_vectors_written": 0,
                    },
                    "retry_summary": {
                        "has_pending_failure": False,
                        "pending_stage_count": 0,
                        "next_retry_at": None,
                        "gave_up_count": 0,
                    },
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )

    async def mark_stage_started(
        self,
        *,
        doc_id: str,
        stage: SyncStage,
    ) -> None:
        await asyncio.to_thread(self._mark_stage_started_sync, doc_id, stage)

    def _mark_stage_started_sync(self, doc_id: str, stage: SyncStage) -> None:
        self._require_sync()
        now = utcnow()
        prefix = f"stage_statuses.{stage.value}"
        self.status_collection.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    f"{prefix}.status": StageExecutionStatus.PROCESSING.value,
                    f"{prefix}.started_at": now,
                    f"{prefix}.updated_at": now,
                    "current_stage": stage.value,
                    "updated_at": now,
                },
                "$inc": {
                    f"{prefix}.attempt_count": 1,
                },
            },
            upsert=False,
        )

    async def mark_stage_finished(
        self,
        *,
        doc_id: str,
        stage: SyncStage,
        status: StageExecutionStatus,
        artifact_ref: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        await asyncio.to_thread(
            self._mark_stage_finished_sync,
            doc_id,
            stage,
            status,
            artifact_ref,
            error_code,
            error_message,
            duration_ms,
        )

    def _mark_stage_finished_sync(
        self,
        doc_id: str,
        stage: SyncStage,
        status: StageExecutionStatus,
        artifact_ref: Optional[str],
        error_code: Optional[str],
        error_message: Optional[str],
        duration_ms: Optional[int],
    ) -> None:
        self._require_sync()
        now = utcnow()
        prefix = f"stage_statuses.{stage.value}"
        update_set: dict[str, Any] = {
            f"{prefix}.status": status.value,
            f"{prefix}.finished_at": now,
            f"{prefix}.updated_at": now,
            "current_stage": stage.value,
            "updated_at": now,
        }
        if duration_ms is not None:
            update_set[f"{prefix}.duration_ms"] = duration_ms
        if artifact_ref is not None:
            update_set[f"{prefix}.artifact_ref"] = artifact_ref
        if error_code is not None:
            update_set[f"{prefix}.last_error_code"] = error_code
        if error_message is not None:
            update_set[f"{prefix}.last_error_message"] = error_message
        if status in {
            StageExecutionStatus.SUCCESS,
            StageExecutionStatus.SKIPPED_EMPTY,
            StageExecutionStatus.SKIPPED_NOT_PRESENT,
            StageExecutionStatus.SKIPPED_STALE,
        }:
            update_set["last_success_stage"] = stage.value
        self.status_collection.update_one(
            {"_id": doc_id},
            {"$set": update_set},
            upsert=False,
        )

    async def record_attempt(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        event_type: Optional[str],
        source: str,
        stage: SyncStage,
        stage_status: str,
        retry_count_snapshot: int = 0,
        artifact_ref: Optional[str] = None,
        error_summary: Optional[dict[str, Any]] = None,
        source_event_meta: Optional[SourceEventMeta] = None,
        started_at=None,
        finished_at=None,
        duration_ms: Optional[int] = None,
    ) -> None:
        await asyncio.to_thread(
            self._record_attempt_sync,
            doc_id,
            record_update_time,
            sub_doc_type,
            event_type,
            source,
            stage,
            stage_status,
            retry_count_snapshot,
            artifact_ref,
            error_summary,
            source_event_meta,
            started_at,
            finished_at,
            duration_ms,
        )

    def _record_attempt_sync(
        self,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        event_type: Optional[str],
        source: str,
        stage: SyncStage,
        stage_status: str,
        retry_count_snapshot: int,
        artifact_ref: Optional[str],
        error_summary: Optional[dict[str, Any]],
        source_event_meta: Optional[SourceEventMeta],
        started_at,
        finished_at,
        duration_ms: Optional[int],
    ) -> None:
        self._require_sync()
        start_time = started_at or utcnow()
        finish_time = finished_at or utcnow()
        resolved_duration = duration_ms
        if resolved_duration is None:
            resolved_duration = max(
                0,
                int((finish_time - start_time).total_seconds() * 1000),
            )
        self.attempts_collection.insert_one(
            {
                "run_id": ObjectId(),
                "doc_id": doc_id,
                "record_update_time": record_update_time,
                "sub_doc_type": sub_doc_type,
                "event_type": event_type,
                "source": source,
                "stage": stage.value,
                "stage_status": stage_status,
                "started_at": start_time,
                "finished_at": finish_time,
                "duration_ms": resolved_duration,
                "retry_count_snapshot": retry_count_snapshot,
                "artifact_ref": artifact_ref,
                "error_summary": error_summary,
                "source_event_meta": self._event_meta_to_dict(source_event_meta),
            }
        )

    async def save_artifact(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        artifact_type: ArtifactType,
        content: Any,
        success_path: bool,
        content_type: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        expire_days: Optional[int] = None,
    ) -> str:
        return await asyncio.to_thread(
            self._save_artifact_sync,
            doc_id,
            record_update_time,
            sub_doc_type,
            stage,
            scope_type,
            scope_id,
            artifact_type,
            content,
            success_path,
            content_type,
            meta or {},
            expire_days,
        )

    def _save_artifact_sync(
        self,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        artifact_type: ArtifactType,
        content: Any,
        success_path: bool,
        content_type: Optional[str],
        meta: dict[str, Any],
        expire_days: Optional[int],
    ) -> str:
        self._require_sync()
        now = utcnow()
        payload_doc = self._serialize_artifact_payload(content, content_type)
        keep_days = expire_days
        if keep_days is None:
            keep_days = (
                settings.MONGO_ARTIFACT_SUCCESS_RETENTION_DAYS
                if success_path
                else settings.MONGO_ARTIFACT_FAILURE_RETENTION_DAYS
            )
        artifact_doc = {
            "doc_id": doc_id,
            "record_update_time": record_update_time,
            "sub_doc_type": sub_doc_type,
            "stage": stage.value,
            "scope_type": scope_type.value,
            "scope_id": scope_id,
            "artifact_type": artifact_type.value,
            "created_at": now,
            "expire_at": now + timedelta(days=keep_days),
            "meta": meta,
            **payload_doc,
        }
        result = self.artifacts_collection.insert_one(artifact_doc)
        return str(result.inserted_id)

    async def read_artifact(self, artifact_ref: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._read_artifact_sync, artifact_ref)

    def _read_artifact_sync(self, artifact_ref: str) -> Optional[dict[str, Any]]:
        self._require_sync()
        document = self.artifacts_collection.find_one({"_id": ObjectId(artifact_ref)})
        if document is None:
            return None
        return self._deserialize_artifact_document(document)

    async def persist_failure(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        event_type: Optional[str],
        failure_stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        error_type: ErrorType,
        error_code: str,
        error_message: str,
        exception_class: str,
        stacktrace: str,
        replay_mode: ReplayMode,
        artifact_refs: list[str],
        source_event_meta: SourceEventMeta,
        pipeline: str = "",
        source_version: str = "",
        item_id: str = "",
        logical_collection: str = "",
        resolved_collection: str = "",
        target_signature: str = "",
        replay_payload: Optional[dict[str, Any]] = None,
        retry_increment: int = 0,
        max_retries: Optional[int] = None,
    ) -> FailureRef:
        return await asyncio.to_thread(
            self._persist_failure_sync,
            doc_id,
            record_update_time,
            sub_doc_type,
            event_type,
            failure_stage,
            scope_type,
            scope_id,
            error_type,
            error_code,
            error_message,
            exception_class,
            stacktrace,
            replay_mode,
            artifact_refs,
            source_event_meta,
            pipeline,
            source_version,
            item_id,
            logical_collection,
            resolved_collection,
            target_signature,
            replay_payload,
            retry_increment,
            max_retries,
        )

    def _persist_failure_sync(
        self,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        event_type: Optional[str],
        failure_stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        error_type: ErrorType,
        error_code: str,
        error_message: str,
        exception_class: str,
        stacktrace: str,
        replay_mode: ReplayMode,
        artifact_refs: list[str],
        source_event_meta: SourceEventMeta,
        pipeline: str,
        source_version: str,
        item_id: str,
        logical_collection: str,
        resolved_collection: str,
        target_signature: str,
        replay_payload: Optional[dict[str, Any]],
        retry_increment: int,
        max_retries: Optional[int],
    ) -> FailureRef:
        self._require_sync()
        failure_key = self.build_failure_key(
            doc_id=doc_id,
            record_update_time=record_update_time,
            failure_stage=failure_stage,
            scope_type=scope_type,
            scope_id=scope_id,
            pipeline=pipeline,
            target_signature=target_signature,
        )
        now = utcnow()
        max_retry_limit = max_retries if max_retries is not None else settings.MAX_RETRY
        existing = self.failures_collection.find_one({"failure_key": failure_key})
        retry_count = int(existing.get("retry_count", 0)) if existing else 0
        retry_count += retry_increment

        should_auto_retry = error_type == ErrorType.RETRYABLE and retry_count < max_retry_limit
        if should_auto_retry:
            status = FailureStatus.PENDING
            next_retry_at = now + timedelta(
                seconds=self._compute_backoff_delay(retry_count)
            )
        else:
            status = FailureStatus.GAVE_UP if error_type == ErrorType.NON_RETRYABLE or retry_count >= max_retry_limit else FailureStatus.PENDING
            next_retry_at = None
            if status == FailureStatus.GAVE_UP:
                gave_up_total.inc()

        document = {
            "failure_key": failure_key,
            "doc_id": doc_id,
            "record_update_time": record_update_time,
            "sub_doc_type": sub_doc_type,
            "event_type": event_type,
            "failure_stage": failure_stage.value,
            "scope_type": scope_type.value,
            "scope_id": scope_id,
            "status": status.value,
            "error_type": error_type.value,
            "error_code": error_code,
            "error_message": error_message,
            "exception_class": exception_class,
            "stacktrace": stacktrace,
            "retry_count": retry_count,
            "max_retries": max_retry_limit,
            "next_retry_at": next_retry_at,
            "last_retry_at": now if retry_increment > 0 else existing.get("last_retry_at") if existing else None,
            "replay_mode": replay_mode.value,
            "artifact_refs": artifact_refs,
            "source_event_meta": self._event_meta_to_dict(source_event_meta),
            "pipeline": str(pipeline or "").strip().lower(),
            "source_version": str(source_version or record_update_time or "").strip(),
            "item_id": str(item_id or scope_id or "").strip(),
            "logical_collection": str(logical_collection or "").strip(),
            "resolved_collection": str(resolved_collection or "").strip(),
            "target_signature": str(target_signature or "").strip(),
            # Direct Kafka workers keep the minimal task needed to retry a
            # failure. Successful messages never create this Mongo record.
            "replay_payload": dict(replay_payload or {}),
            "updated_at": now,
        }
        if status == FailureStatus.GAVE_UP:
            document["resolved_at"] = now
        else:
            document["resolved_at"] = None
        update = {
            "$set": document,
            "$setOnInsert": {
                "created_at": now,
            },
        }
        self.failures_collection.update_one(
            {"failure_key": failure_key},
            update,
            upsert=True,
        )
        if doc_id:
            self._refresh_retry_summary_sync(doc_id)
        self._refresh_pending_count_sync()
        return FailureRef(
            failure_key=failure_key,
            failure_stage=failure_stage,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    async def finalize_document_status(
        self,
        *,
        doc_id: str,
        overall_status: SyncOverallStatus,
        current_stage: Optional[SyncStage],
        last_success_stage: Optional[SyncStage],
        progress: Optional[dict[str, Any]] = None,
    ) -> None:
        await asyncio.to_thread(
            self._finalize_document_status_sync,
            doc_id,
            overall_status,
            current_stage,
            last_success_stage,
            progress or {},
        )

    def _finalize_document_status_sync(
        self,
        doc_id: str,
        overall_status: SyncOverallStatus,
        current_stage: Optional[SyncStage],
        last_success_stage: Optional[SyncStage],
        progress: dict[str, Any],
    ) -> None:
        self._require_sync()
        update_set: dict[str, Any] = {
            "overall_status": overall_status.value,
            "updated_at": utcnow(),
        }
        if current_stage is not None:
            update_set["current_stage"] = current_stage.value
        if last_success_stage is not None:
            update_set["last_success_stage"] = last_success_stage.value
        for key, value in progress.items():
            update_set[f"progress.{key}"] = value
        self.status_collection.update_one({"_id": doc_id}, {"$set": update_set}, upsert=False)

    async def update_pipeline_status(
        self,
        *,
        doc_id: str,
        document_status: Optional[str] = None,
        fulltext_status: Optional[str] = None,
        image_status: Optional[str] = None,
        core_status: Optional[str] = None,
        overall_status: Optional[str | SyncOverallStatus] = None,
        fulltext_required: Optional[bool] = None,
        image_required: Optional[bool] = None,
        progress: Optional[dict[str, Any]] = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_pipeline_status_sync,
            doc_id,
            document_status,
            fulltext_status,
            image_status,
            core_status,
            overall_status,
            fulltext_required,
            image_required,
            progress or {},
        )

    def _update_pipeline_status_sync(
        self,
        doc_id: str,
        document_status: Optional[str],
        fulltext_status: Optional[str],
        image_status: Optional[str],
        core_status: Optional[str],
        overall_status: Optional[str | SyncOverallStatus],
        fulltext_required: Optional[bool],
        image_required: Optional[bool],
        progress: dict[str, Any],
    ) -> None:
        self._require_sync()
        update_set: dict[str, Any] = {"updated_at": utcnow()}
        if document_status is not None:
            update_set["document_status"] = document_status
        if fulltext_status is not None:
            update_set["fulltext_status"] = fulltext_status
        if image_status is not None:
            update_set["image_status"] = image_status
        if core_status is not None:
            update_set["core_status"] = core_status
        if overall_status is not None:
            update_set["overall_status"] = (
                overall_status.value
                if isinstance(overall_status, SyncOverallStatus)
                else str(overall_status)
            )
        if fulltext_required is not None:
            update_set["fulltext_required"] = bool(fulltext_required)
        if image_required is not None:
            update_set["image_required"] = bool(image_required)
        for key, value in progress.items():
            update_set[f"progress.{key}"] = value
        self.status_collection.update_one({"_id": doc_id}, {"$set": update_set}, upsert=False)

    async def refresh_async_pipeline_status(
        self,
        *,
        doc_id: str,
        record_update_time: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self._refresh_async_pipeline_status_sync,
            doc_id,
            record_update_time,
        )

    def _refresh_async_pipeline_status_sync(
        self,
        doc_id: str,
        record_update_time: Optional[str],
    ) -> None:
        self._require_sync()
        status_doc = self.status_collection.find_one({"_id": doc_id}) or {}
        task_filter: dict[str, Any] = {"doc_id": doc_id}
        if record_update_time:
            task_filter["record_update_time"] = record_update_time

        fulltext_counts = _task_status_counts(self.fulltext_tasks_collection, task_filter)
        image_counts = _task_status_counts(self.image_tasks_collection, task_filter)
        fulltext_status = _derive_task_group_status(fulltext_counts, partial_success=False)
        image_status = _derive_task_group_status(image_counts, partial_success=True)
        document_status = str(
            status_doc.get("document_status")
            or (
                PIPELINE_STATUS_SUCCESS
                if status_doc.get("last_success_stage") == SyncStage.MILVUS_WRITE_DOCUMENT.value
                else PIPELINE_STATUS_PROCESSING
            )
        )
        core_status = _derive_core_status(document_status, fulltext_status)
        overall_status = _derive_overall_status(core_status, image_status)

        fulltext_total = sum(fulltext_counts.values())
        image_total = sum(image_counts.values())
        progress = {
            "fulltext_task_count": fulltext_total,
            "fulltext_task_success": int(fulltext_counts.get("SUCCESS", 0)),
            "fulltext_task_failure": int(fulltext_counts.get("FAILURE", 0)),
            "image_count_expected": image_total,
            "image_count_written": int(image_counts.get("SUCCESS", 0)),
            "image_task_failure": int(image_counts.get("FAILURE", 0)),
            "image_task_pending": int(image_counts.get("PENDING", 0)),
            "image_task_leased": int(image_counts.get("LEASED", 0)),
        }
        self._update_pipeline_status_sync(
            doc_id,
            document_status,
            fulltext_status,
            image_status,
            core_status,
            overall_status,
            fulltext_total > 0,
            image_total > 0,
            progress,
        )

    async def resolve_failures_for_document(
        self,
        doc_id: str,
        record_update_time: str,
    ) -> None:
        await asyncio.to_thread(
            self._resolve_failures_for_document_sync,
            doc_id,
            record_update_time,
        )

    def _resolve_failures_for_document_sync(self, doc_id: str, record_update_time: str) -> None:
        self._require_sync()
        now = utcnow()
        self.failures_collection.update_many(
            {
                "doc_id": doc_id,
                "record_update_time": record_update_time,
                "status": {
                    "$in": [
                        FailureStatus.PENDING.value,
                        FailureStatus.RETRYING.value,
                        FailureStatus.GAVE_UP.value,
                    ]
                },
            },
            {
                "$set": {
                    "status": FailureStatus.DONE.value,
                    "resolved_at": now,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            },
        )
        self._refresh_retry_summary_sync(doc_id)
        self._refresh_pending_count_sync()

    async def mark_failure_skipped_stale(self, failure_key: str, reason: str) -> None:
        await asyncio.to_thread(self._mark_failure_skipped_stale_sync, failure_key, reason)

    def _mark_failure_skipped_stale_sync(self, failure_key: str, reason: str) -> None:
        self._require_sync()
        now = utcnow()
        doc = self.failures_collection.find_one_and_update(
            {"failure_key": failure_key},
            {
                "$set": {
                    "status": FailureStatus.SKIPPED_STALE.value,
                    "error_message": reason,
                    "resolved_at": now,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if doc and doc.get("doc_id"):
            self._refresh_retry_summary_sync(doc["doc_id"])
        self._refresh_pending_count_sync()

    async def mark_failure_done(self, failure_key: str, note: str = "") -> None:
        await asyncio.to_thread(self._mark_failure_done_sync, failure_key, note)

    def _mark_failure_done_sync(self, failure_key: str, note: str) -> None:
        self._require_sync()
        now = utcnow()
        doc = self.failures_collection.find_one_and_update(
            {"failure_key": failure_key},
            {
                "$set": {
                    "status": FailureStatus.DONE.value,
                    "resolved_at": now,
                    "updated_at": now,
                    "resolution_note": note or None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if doc and doc.get("doc_id"):
            self._refresh_retry_summary_sync(doc["doc_id"])
        self._refresh_pending_count_sync()

    async def claim_failure_for_retry(self) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._claim_failure_for_retry_sync)

    async def claim_failure_for_retry_filtered(
        self,
        *,
        pipelines: set[str],
        targets: dict[str, tuple[str, str]],
    ) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(
            self._claim_failure_for_retry_filtered_sync,
            {str(value).strip().lower() for value in pipelines if str(value).strip()},
            dict(targets),
        )

    def _claim_failure_for_retry_sync(self) -> Optional[dict[str, Any]]:
        return self._claim_failure_for_retry_filtered_sync(set(), {})

    def _claim_failure_for_retry_filtered_sync(
        self,
        pipelines: set[str],
        targets: dict[str, tuple[str, str]],
    ) -> Optional[dict[str, Any]]:
        self._require_sync()
        now = utcnow()
        lease_expiry = now + timedelta(seconds=settings.COMPENSATOR_LEASE_SEC)
        eligibility: dict[str, Any] = {
            "$or": [
                {
                    "status": FailureStatus.PENDING.value,
                    "next_retry_at": {"$lte": now},
                },
                {
                    "status": FailureStatus.RETRYING.value,
                    "lease_expires_at": {"$lte": now},
                },
            ]
        }
        if pipelines:
            self._hold_legacy_failures_without_target_binding_sync(now)
            target_filters = []
            for pipeline in sorted(pipelines):
                target = targets.get(pipeline)
                if target is None:
                    continue
                resolved_collection, target_signature = target
                target_filters.append(
                    {
                        "pipeline": pipeline,
                        "resolved_collection": resolved_collection,
                        "target_signature": target_signature,
                    }
                )
            if not target_filters:
                return None
            eligibility = {"$and": [eligibility, {"$or": target_filters}]}
        claimed = self.failures_collection.find_one_and_update(
            eligibility,
            {
                "$set": {
                    "status": FailureStatus.RETRYING.value,
                    "lease_owner": self.worker_id,
                    "lease_expires_at": lease_expiry,
                    "last_retry_at": now,
                    "updated_at": now,
                }
            },
            sort=[("next_retry_at", ASCENDING), ("updated_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        return claimed

    def _hold_legacy_failures_without_target_binding_sync(self, now) -> int:
        """Freeze historical failures that cannot be tied to a safe target.

        Older records do not carry a pipeline, resolved collection, or target
        signature. Retrying them after a collection switch could write stale
        data into a new collection, so keep them visible in Mongo as history
        rather than treating them as retry candidates.
        """
        legacy_filter = {
            "$and": [
                {
                    "$or": [
                        {"status": FailureStatus.PENDING.value},
                        {"status": FailureStatus.RETRYING.value},
                    ]
                },
                {
                    "$or": [
                        {"pipeline": {"$exists": False}},
                        {"pipeline": None},
                        {"pipeline": ""},
                        {"resolved_collection": {"$exists": False}},
                        {"resolved_collection": None},
                        {"resolved_collection": ""},
                        {"target_signature": {"$exists": False}},
                        {"target_signature": None},
                        {"target_signature": ""},
                    ]
                },
            ]
        }
        result = self.failures_collection.update_many(
            legacy_filter,
            {
                "$set": {
                    "status": FailureStatus.HELD_HISTORY.value,
                    "resolution_note": "skipped: legacy failure has no pipeline target binding",
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            },
        )
        held_count = int(getattr(result, "matched_count", 0))
        if held_count:
            self._refresh_pending_count_sync()
        return held_count

    async def release_claimed_failure(
        self,
        failure_key: str,
        *,
        error_type: ErrorType,
        error_code: str,
        error_message: str,
        exception_class: str,
        stacktrace: str,
        artifact_refs: list[str],
    ) -> FailureRef:
        existing = await self.get_failure(failure_key)
        if existing is None:
            raise MongoPersistenceError(f"failure task missing after claim: {failure_key}")
        return await self.persist_failure(
            doc_id=existing.get("doc_id"),
            record_update_time=existing.get("record_update_time"),
            sub_doc_type=existing.get("sub_doc_type"),
            event_type=existing.get("event_type"),
            failure_stage=SyncStage(existing["failure_stage"]),
            scope_type=ScopeType(existing["scope_type"]),
            scope_id=existing["scope_id"],
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            exception_class=exception_class,
            stacktrace=stacktrace,
            replay_mode=ReplayMode(existing.get("replay_mode", ReplayMode.FROM_RAW_EVENT.value)),
            artifact_refs=artifact_refs,
            source_event_meta=self._dict_to_event_meta(existing.get("source_event_meta") or {}),
            pipeline=str(existing.get("pipeline") or ""),
            source_version=str(existing.get("source_version") or ""),
            item_id=str(existing.get("item_id") or ""),
            logical_collection=str(existing.get("logical_collection") or ""),
            resolved_collection=str(existing.get("resolved_collection") or ""),
            target_signature=str(existing.get("target_signature") or ""),
            replay_payload=dict(existing.get("replay_payload") or {}),
            retry_increment=1,
            max_retries=existing.get("max_retries"),
        )

    async def get_failure(self, failure_key: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._get_failure_sync, failure_key)

    def _get_failure_sync(self, failure_key: str) -> Optional[dict[str, Any]]:
        self._require_sync()
        return self.failures_collection.find_one({"failure_key": failure_key})

    async def get_failure_artifacts(self, artifact_refs: list[str]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_failure_artifacts_sync, artifact_refs)

    def _get_failure_artifacts_sync(self, artifact_refs: list[str]) -> list[dict[str, Any]]:
        self._require_sync()
        object_ids = [ObjectId(ref) for ref in artifact_refs if ObjectId.is_valid(ref)]
        if not object_ids:
            return []
        docs = list(self.artifacts_collection.find({"_id": {"$in": object_ids}}))
        return [self._deserialize_artifact_document(item) for item in docs]

    async def get_document_diagnostics_metadata(
        self,
        doc_id: str,
        *,
        recent_limit: int = 20,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._get_document_diagnostics_metadata_sync,
            doc_id,
            recent_limit,
        )

    def _get_document_diagnostics_metadata_sync(
        self,
        doc_id: str,
        recent_limit: int,
    ) -> dict[str, Any]:
        self._require_sync()
        bounded_limit = max(1, min(int(recent_limit), 20))
        failure_filter = {"doc_id": doc_id}
        artifact_filter = {"doc_id": doc_id}
        recent_failures = list(
            self.failures_collection.find(
                failure_filter,
                projection={
                    "error_message": 0,
                    "stacktrace": 0,
                },
            )
            .sort("updated_at", DESCENDING)
            .limit(bounded_limit)
        )
        recent_artifacts = list(
            self.artifacts_collection.find(
                artifact_filter,
                projection={
                    "meta": 0,
                    "payload_json": 0,
                    "payload_text": 0,
                    "payload_gzip": 0,
                },
            )
            .sort("created_at", DESCENDING)
            .limit(bounded_limit)
        )
        return {
            "status": self.status_collection.find_one({"_id": doc_id}),
            "failure_counts": self._count_document_failures_sync(doc_id),
            "recent_failures": recent_failures,
            "artifact_counts": self._count_document_artifacts_sync(doc_id),
            "recent_artifacts": recent_artifacts,
        }

    def _count_document_failures_sync(self, doc_id: str) -> dict[str, Any]:
        base_filter = {"doc_id": doc_id}
        return {
            "total": int(self.failures_collection.count_documents(base_filter)),
            "by_status": {
                status.value: int(
                    self.failures_collection.count_documents(
                        {**base_filter, "status": status.value}
                    )
                )
                for status in FailureStatus
            },
            "by_stage": {
                stage.value: int(
                    self.failures_collection.count_documents(
                        {**base_filter, "failure_stage": stage.value}
                    )
                )
                for stage in SyncStage
            },
            "by_error_type": {
                error_type.value: int(
                    self.failures_collection.count_documents(
                        {**base_filter, "error_type": error_type.value}
                    )
                )
                for error_type in ErrorType
            },
        }

    def _count_document_artifacts_sync(self, doc_id: str) -> dict[str, Any]:
        base_filter = {"doc_id": doc_id}
        return {
            "total": int(self.artifacts_collection.count_documents(base_filter)),
            "by_type": {
                artifact_type.value: int(
                    self.artifacts_collection.count_documents(
                        {**base_filter, "artifact_type": artifact_type.value}
                    )
                )
                for artifact_type in ArtifactType
            },
            "by_stage": {
                stage.value: int(
                    self.artifacts_collection.count_documents(
                        {**base_filter, "stage": stage.value}
                    )
                )
                for stage in SyncStage
            },
        }

    async def mark_mongo_unavailable(self, context: dict[str, Any]) -> None:
        await asyncio.to_thread(self._dump_mongo_fallback_sync, context)

    def _dump_mongo_fallback_sync(self, context: dict[str, Any]) -> None:
        with _dump_lock:
            with open(settings.KAFKA_ERROR_DUMP_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(context, ensure_ascii=False) + "\n")

    async def get_pipeline_source_version(
        self,
        *,
        pipeline: str,
        logical_collection: str,
        target_signature: str,
        doc_id: str,
    ) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_pipeline_source_version_sync,
            pipeline,
            logical_collection,
            target_signature,
            doc_id,
        )

    def _get_pipeline_source_version_sync(
        self,
        pipeline: str,
        logical_collection: str,
        target_signature: str,
        doc_id: str,
    ) -> Optional[dict[str, Any]]:
        self._require_sync()
        return self.pipeline_versions_collection.find_one(
            {
                "pipeline": str(pipeline or "").strip().lower(),
                "logical_collection": str(logical_collection or "").strip(),
                "target_signature": str(target_signature or "").strip(),
                "doc_id": str(doc_id or "").strip(),
            },
            {"source_version": 1, "source_hash": 1, "updated_at": 1},
        )

    async def upsert_pipeline_source_version(
        self,
        *,
        pipeline: str,
        logical_collection: str,
        target_signature: str,
        doc_id: str,
        source_version: str,
        source_hash: str,
    ) -> None:
        await asyncio.to_thread(
            self._upsert_pipeline_source_version_sync,
            pipeline,
            logical_collection,
            target_signature,
            doc_id,
            source_version,
            source_hash,
        )

    def _upsert_pipeline_source_version_sync(
        self,
        pipeline: str,
        logical_collection: str,
        target_signature: str,
        doc_id: str,
        source_version: str,
        source_hash: str,
    ) -> None:
        self._require_sync()
        now = utcnow()
        identity = {
            "pipeline": str(pipeline or "").strip().lower(),
            "logical_collection": str(logical_collection or "").strip(),
            "target_signature": str(target_signature or "").strip(),
            "doc_id": str(doc_id or "").strip(),
        }
        update = {
            "$set": {
                **identity,
                "source_version": str(source_version or "").strip(),
                "source_hash": str(source_hash or "").strip(),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        }
        try:
            self.pipeline_versions_collection.update_one(identity, update, upsert=True)
        except DuplicateKeyError:
            # Two in-flight batches can finish the same repeated Kafka ID at
            # once. The winner inserted the watermark; update it without a
            # second upsert instead of logging a spurious pipeline failure.
            self.pipeline_versions_collection.update_one(
                identity,
                {"$set": update["$set"]},
                upsert=False,
            )

    @property
    def status_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_STATUS_COLLECTION]

    @property
    def failures_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_FAILURE_COLLECTION]

    @property
    def attempts_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_ATTEMPT_COLLECTION]

    @property
    def artifacts_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_ARTIFACT_COLLECTION]

    @property
    def pipeline_versions_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_PIPELINE_VERSION_COLLECTION]

    @property
    def fulltext_tasks_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_FULLTEXT_TASK_COLLECTION]

    @property
    def image_tasks_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.MONGO_IMAGE_TASK_COLLECTION]

    @property
    def image_priority_windows_collection(self) -> Collection:
        self._require_sync()
        return self._db[settings.IMAGE_PRIORITY_WINDOW_COLLECTION]

    async def ensure_image_priority_window_indexes(self) -> None:
        await asyncio.to_thread(self._ensure_image_priority_window_indexes_sync)

    def _ensure_image_priority_window_indexes_sync(self) -> None:
        collection = self.image_priority_windows_collection
        collection.create_index(
            [("doc_id", ASCENDING), ("record_update_time", DESCENDING)],
            name="idx_doc_time",
        )
        collection.create_index([("window_id", ASCENDING)], name="idx_window_id")
        collection.create_index(
            [("window_id", ASCENDING), ("metadata_backfill_status", ASCENDING)],
            name="idx_window_metadata_backfill_status",
        )
        collection.create_index(
            [
                ("window_id", ASCENDING),
                ("metadata_backfill_status", ASCENDING),
                ("image_task_count", ASCENDING),
                ("updated_at", ASCENDING),
            ],
            name="idx_window_metadata_backfill_ready",
        )
        collection.create_index(
            [("window_id", ASCENDING), ("fulltext_backfill_status", ASCENDING)],
            name="idx_window_fulltext_backfill_status",
        )
        collection.create_index(
            [
                ("window_id", ASCENDING),
                ("fulltext_backfill_status", ASCENDING),
                ("image_task_count", ASCENDING),
                ("updated_at", ASCENDING),
            ],
            name="idx_window_fulltext_backfill_ready",
        )
        collection.create_index(
            [("image_ready_at", ASCENDING), ("updated_at", ASCENDING)],
            name="idx_image_ready_updated",
        )
        collection.create_index([("status", ASCENDING)], name="idx_status")
        collection.create_index([("updated_at", DESCENDING)], name="idx_updated")

    async def bulk_write_image_priority_windows(self, operations: list[Any]) -> None:
        await asyncio.to_thread(self._bulk_write_image_priority_windows_sync, operations)

    def _bulk_write_image_priority_windows_sync(self, operations: list[Any]) -> None:
        if not operations:
            return
        self.image_priority_windows_collection.bulk_write(operations, ordered=False)

    async def find_image_priority_windows_for_backfill(
        self,
        *,
        window_id: Optional[str] = None,
        statuses: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._find_image_priority_windows_for_backfill_sync,
            window_id,
            statuses,
            limit,
        )

    def _find_image_priority_windows_for_backfill_sync(
        self,
        window_id: Optional[str],
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self._require_sync()
        query: dict[str, Any] = {}
        if window_id:
            query["window_id"] = window_id
        query["backfill_status"] = {"$in": statuses or ["PENDING", "FAILED"]}
        cursor = self.image_priority_windows_collection.find(query).sort(
            [("updated_at", ASCENDING), ("_id", ASCENDING)]
        )
        if limit > 0:
            cursor = cursor.limit(int(limit))
        return list(cursor)

    async def find_image_success_windows_for_metadata_backfill(
        self,
        *,
        window_id: Optional[str] = None,
        statuses: Optional[list[str]] = None,
        limit: int = 100,
        target: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._find_image_success_windows_for_metadata_backfill_sync,
            window_id,
            statuses,
            limit,
            target,
        )

    def _find_image_success_windows_for_metadata_backfill_sync(
        self,
        window_id: Optional[str],
        statuses: Optional[list[str]],
        limit: int,
        target: Optional[str],
    ) -> list[dict[str, Any]]:
        self._require_sync()
        allowed_statuses = statuses or ["PENDING", "FAILED"]
        window_query: dict[str, Any] = {
            "image_ready_at": {"$ne": None},
            "$or": [
                {"metadata_backfill_target": {"$ne": target}},
                {"metadata_backfill_status": {"$in": allowed_statuses}},
            ],
        }
        if window_id:
            window_query["window_id"] = window_id
        cursor = self.image_priority_windows_collection.find(window_query).sort(
            [("updated_at", ASCENDING), ("_id", ASCENDING)]
        )
        if limit > 0:
            cursor = cursor.limit(int(limit))
        windows = list(cursor)
        if not windows:
            return []
        return windows

    async def find_image_success_windows_for_fulltext_backfill(
        self,
        *,
        window_id: Optional[str] = None,
        statuses: Optional[list[str]] = None,
        limit: int = 100,
        target: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._find_image_success_windows_for_fulltext_backfill_sync,
            window_id,
            statuses,
            limit,
            target,
        )

    def _find_image_success_windows_for_fulltext_backfill_sync(
        self,
        window_id: Optional[str],
        statuses: Optional[list[str]],
        limit: int,
        target: Optional[str],
    ) -> list[dict[str, Any]]:
        self._require_sync()
        allowed_statuses = statuses or ["PENDING", "FAILED"]
        window_query: dict[str, Any] = {
            "image_ready_at": {"$ne": None},
            "$or": [
                {"fulltext_backfill_target": {"$ne": target}},
                {"fulltext_backfill_status": {"$in": allowed_statuses}},
            ],
        }
        if window_id:
            window_query["window_id"] = window_id
        cursor = self.image_priority_windows_collection.find(window_query).sort(
            [("updated_at", ASCENDING), ("_id", ASCENDING)]
        )
        if limit > 0:
            cursor = cursor.limit(int(limit))
        return list(cursor)

    async def mark_metadata_backfill_result(
        self,
        *,
        ids: list[str],
        target: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self._mark_backfill_result_sync,
            ids,
            "metadata",
            target,
            status,
            error,
        )

    async def mark_fulltext_backfill_result(
        self,
        *,
        ids: list[str],
        target: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self._mark_backfill_result_sync,
            ids,
            "fulltext",
            target,
            status,
            error,
        )

    def _mark_backfill_result_sync(
        self,
        ids: list[str],
        lane: str,
        target: str,
        status: str,
        error: Optional[str],
    ) -> None:
        self._require_sync()
        if not ids:
            return
        now = utcnow()
        prefix = "metadata" if lane == "metadata" else "fulltext"
        set_payload: dict[str, Any] = {
            f"{prefix}_backfill_target": target,
            f"{prefix}_backfill_status": status,
            f"{prefix}_backfill_error": error,
            "updated_at": now,
        }
        if status == "DONE":
            set_payload[f"{prefix}_backfilled_at"] = now
        self.image_priority_windows_collection.update_many(
            {"_id": {"$in": ids}},
            {"$set": set_payload},
        )

    async def mark_image_priority_window_ready(
        self,
        *,
        doc_id: str,
        record_update_time: str,
        image_ready_at: Optional[datetime] = None,
    ) -> None:
        await asyncio.to_thread(
            self._mark_image_priority_window_ready_sync,
            doc_id,
            record_update_time,
            image_ready_at or utcnow(),
        )

    def _mark_image_priority_window_ready_sync(
        self,
        doc_id: str,
        record_update_time: str,
        image_ready_at: datetime,
    ) -> None:
        self._require_sync()
        self.image_priority_windows_collection.update_one(
            {"_id": f"{doc_id}|{record_update_time}"},
            {
                "$set": {
                    "image_ready_at": image_ready_at,
                    "updated_at": utcnow(),
                }
            },
            upsert=False,
        )

    def build_failure_key(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        failure_stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        pipeline: str = "",
        target_signature: str = "",
    ) -> str:
        return "|".join(
            [
                doc_id or "",
                record_update_time or "",
                failure_stage.value,
                scope_type.value,
                scope_id,
                str(pipeline or "").strip().lower(),
                str(target_signature or "").strip(),
            ]
        )

    def _ensure_indexes_sync(self) -> None:
        self.status_collection.create_index(
            [("overall_status", ASCENDING), ("updated_at", DESCENDING)],
            name="idx_status_updated",
        )
        self.status_collection.create_index(
            [("retry_summary.next_retry_at", ASCENDING)],
            name="idx_next_retry",
        )
        self.status_collection.create_index(
            [("core_status", ASCENDING), ("updated_at", DESCENDING)],
            name="idx_core_status_updated",
        )
        self.status_collection.create_index(
            [("image_status", ASCENDING), ("updated_at", DESCENDING)],
            name="idx_image_status_updated",
        )

        self.failures_collection.create_index(
            [("failure_key", ASCENDING)],
            unique=True,
            name="ux_failure_key",
        )
        self.failures_collection.create_index(
            [("status", ASCENDING), ("next_retry_at", ASCENDING), ("retry_count", ASCENDING)],
            name="idx_failure_retry",
        )
        self.failures_collection.create_index(
            [
                ("pipeline", ASCENDING),
                ("resolved_collection", ASCENDING),
                ("target_signature", ASCENDING),
                ("status", ASCENDING),
                ("next_retry_at", ASCENDING),
            ],
            name="idx_failure_target_retry",
        )
        self.failures_collection.create_index(
            [("doc_id", ASCENDING), ("record_update_time", DESCENDING)],
            name="idx_failure_doc_time",
        )
        _ensure_ttl_index(
            self.failures_collection,
            [("resolved_at", ASCENDING)],
            expire_after_seconds=(
                settings.MONGO_FAILURE_TTL_DAYS * 24 * 3600
                if settings.MONGO_FAILURE_RETENTION_ENABLED
                else None
            ),
            name="ttl_failure_resolved",
        )

        self.attempts_collection.create_index(
            [("doc_id", ASCENDING), ("record_update_time", DESCENDING), ("started_at", DESCENDING)],
            name="idx_attempt_doc_time_start",
        )
        _ensure_ttl_index(
            self.attempts_collection,
            [("finished_at", ASCENDING)],
            expire_after_seconds=settings.MONGO_ATTEMPT_TTL_DAYS * 24 * 3600,
            name="ttl_attempt_finished",
        )

        self.artifacts_collection.create_index(
            [("doc_id", ASCENDING), ("record_update_time", DESCENDING), ("artifact_type", ASCENDING), ("stage", ASCENDING)],
            name="idx_artifact_doc_type_stage",
        )
        self.pipeline_versions_collection.create_index(
            [
                ("pipeline", ASCENDING),
                ("logical_collection", ASCENDING),
                ("target_signature", ASCENDING),
                ("doc_id", ASCENDING),
            ],
            unique=True,
            name="ux_pipeline_target_doc",
        )
        self.pipeline_versions_collection.create_index(
            [("updated_at", DESCENDING)],
            name="idx_pipeline_version_updated",
        )
        _ensure_ttl_index(
            self.artifacts_collection,
            [("expire_at", ASCENDING)],
            expire_after_seconds=0,
            name="ttl_artifact_expire",
        )

    def _require_sync(self) -> None:
        if self._client is None or self._db is None:
            raise MongoPersistenceError("mongo state manager is not connected")

    def _serialize_artifact_payload(self, content: Any, content_type: Optional[str]) -> dict[str, Any]:
        if isinstance(content, (dict, list)):
            serialized = json.dumps(content, ensure_ascii=False, default=_json_default)
            resolved_content_type = content_type or "application/json"
        elif isinstance(content, bytes):
            serialized = content.decode("utf-8", errors="replace")
            resolved_content_type = content_type or "text/plain"
        else:
            serialized = str(content)
            resolved_content_type = content_type or "text/plain"

        raw_bytes = serialized.encode("utf-8")
        document: dict[str, Any] = {
            "content_type": resolved_content_type,
            "sha256": sha256_hex(raw_bytes),
            "size_bytes": len(raw_bytes),
            "compression": None,
        }
        if len(raw_bytes) > settings.MONGO_ARTIFACT_COMPRESS_THRESHOLD_BYTES:
            document["compression"] = "gzip"
            document["payload_gzip"] = Binary(gzip.compress(raw_bytes))
        elif resolved_content_type == "application/json":
            document["payload_json"] = json.loads(serialized)
        else:
            document["payload_text"] = serialized
        return document

    def _deserialize_artifact_document(self, document: dict[str, Any]) -> dict[str, Any]:
        payload: Any = None
        if document.get("compression") == "gzip" and document.get("payload_gzip") is not None:
            raw_bytes = gzip.decompress(document["payload_gzip"])
            if document.get("content_type") == "application/json":
                payload = json.loads(raw_bytes.decode("utf-8"))
            else:
                payload = raw_bytes.decode("utf-8")
        elif "payload_json" in document:
            payload = document["payload_json"]
        elif "payload_text" in document:
            payload = document["payload_text"]
        return {
            **document,
            "_id": str(document["_id"]),
            "payload": payload,
        }

    def _compute_backoff_delay(self, retry_count: int) -> float:
        base_delay = settings.RETRY_BASE_DELAY_SEC
        max_delay = settings.RETRY_MAX_DELAY_SEC
        delay = min(base_delay * (2 ** retry_count), max_delay)
        jitter = random.uniform(0.75, 1.25)
        return delay * jitter

    def _refresh_retry_summary_sync(self, doc_id: str) -> None:
        self._require_sync()
        pending_filter = {
            "doc_id": doc_id,
            "status": {
                "$in": [
                    FailureStatus.PENDING.value,
                    FailureStatus.RETRYING.value,
                ]
            },
        }
        pending_stage_count = self.failures_collection.count_documents(pending_filter)
        next_retry_doc = self.failures_collection.find_one(
            pending_filter,
            sort=[("next_retry_at", ASCENDING), ("updated_at", ASCENDING)],
        )
        gave_up_count = self.failures_collection.count_documents(
            {
                "doc_id": doc_id,
                "status": FailureStatus.GAVE_UP.value,
            }
        )
        retry_summary = {
            "has_pending_failure": pending_stage_count > 0,
            "pending_stage_count": int(pending_stage_count),
            "next_retry_at": next_retry_doc.get("next_retry_at") if next_retry_doc else None,
            "gave_up_count": int(gave_up_count),
        }
        self.status_collection.update_one(
            {"_id": doc_id},
            {"$set": {"retry_summary": retry_summary, "updated_at": utcnow()}},
            upsert=False,
        )

    def _refresh_pending_count_sync(self) -> None:
        self._require_sync()
        pending = self.failures_collection.count_documents(
            {"status": {"$in": [FailureStatus.PENDING.value, FailureStatus.RETRYING.value]}}
        )
        failure_pending_count.set(float(pending))

    def _event_meta_to_dict(self, meta: Optional[SourceEventMeta]) -> Optional[dict[str, Any]]:
        if meta is None:
            return None
        return {
            "topic": meta.topic,
            "partition": meta.partition,
            "offset": meta.offset,
            "event_fingerprint": meta.event_fingerprint,
        }

    def _dict_to_event_meta(self, value: dict[str, Any]) -> SourceEventMeta:
        return SourceEventMeta(
            topic=str(value.get("topic", settings.KAFKA_TOPIC)),
            partition=int(value.get("partition", -1)),
            offset=int(value.get("offset", -1)),
            event_fingerprint=str(value.get("event_fingerprint", "")),
        )


def _task_status_counts(collection: Collection, task_filter: dict[str, Any]) -> dict[str, int]:
    rows = collection.aggregate(
        [
            {"$match": dict(task_filter)},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
    )
    return {str(row.get("_id") or ""): int(row.get("count") or 0) for row in rows}


def _derive_task_group_status(
    counts: dict[str, int],
    *,
    partial_success: bool,
) -> str:
    total = sum(int(value or 0) for value in counts.values())
    if total <= 0:
        return PIPELINE_STATUS_NOT_PRESENT
    success = int(counts.get("SUCCESS", 0))
    failure = int(counts.get("FAILURE", 0))
    pending = int(counts.get("PENDING", 0))
    leased = int(counts.get("LEASED", 0))
    skipped = int(counts.get("SKIPPED", 0))
    if failure:
        if partial_success and success:
            return PIPELINE_STATUS_PARTIAL_SUCCESS
        return PIPELINE_STATUS_FAILURE
    if leased:
        return PIPELINE_STATUS_PROCESSING
    if pending:
        return PIPELINE_STATUS_PROCESSING if success else PIPELINE_STATUS_PENDING
    if success == total:
        return PIPELINE_STATUS_SUCCESS
    if skipped == total:
        return PIPELINE_STATUS_SKIPPED
    if success:
        return PIPELINE_STATUS_PARTIAL_SUCCESS if partial_success else PIPELINE_STATUS_PROCESSING
    return PIPELINE_STATUS_PROCESSING


def _derive_core_status(document_status: str, fulltext_status: str) -> str:
    if document_status == PIPELINE_STATUS_FAILURE:
        return PIPELINE_STATUS_FAILURE
    if document_status != PIPELINE_STATUS_SUCCESS:
        return PIPELINE_STATUS_PROCESSING
    if fulltext_status in {
        PIPELINE_STATUS_SUCCESS,
        PIPELINE_STATUS_NOT_PRESENT,
        PIPELINE_STATUS_SKIPPED,
    }:
        return PIPELINE_STATUS_SUCCESS
    if fulltext_status in {
        PIPELINE_STATUS_PENDING,
        PIPELINE_STATUS_PROCESSING,
    }:
        return PIPELINE_STATUS_PROCESSING
    return PIPELINE_STATUS_FAILURE


def _derive_overall_status(core_status: str, image_status: str) -> str:
    if core_status == PIPELINE_STATUS_FAILURE:
        return SyncOverallStatus.PARTIAL_FAILURE.value
    if core_status != PIPELINE_STATUS_SUCCESS:
        return SyncOverallStatus.PROCESSING.value
    if image_status in {
        PIPELINE_STATUS_SUCCESS,
        PIPELINE_STATUS_NOT_PRESENT,
        PIPELINE_STATUS_SKIPPED,
    }:
        return SyncOverallStatus.SUCCESS.value
    return SyncOverallStatus.CORE_SUCCESS.value


_mongo_state_manager: Optional[MongoStateManager] = None


def get_mongo_state_manager() -> MongoStateManager:
    global _mongo_state_manager
    if _mongo_state_manager is None:
        _mongo_state_manager = MongoStateManager()
    return _mongo_state_manager


def get_existing_mongo_state_manager() -> Optional[MongoStateManager]:
    return _mongo_state_manager


async def close_mongo_state_manager() -> None:
    global _mongo_state_manager
    if _mongo_state_manager is not None:
        await _mongo_state_manager.close()
        _mongo_state_manager = None


def _ensure_ttl_index(
    collection: Collection,
    keys: list[tuple[str, int]],
    *,
    expire_after_seconds: Optional[int],
    name: str,
) -> None:
    if expire_after_seconds is None:
        try:
            collection.drop_index(name)
        except Exception:
            pass
        return
    try:
        collection.create_index(keys, expireAfterSeconds=expire_after_seconds, name=name)
    except OperationFailure as error:
        if getattr(error, "code", None) != 85:
            raise
        logger.warning(
            "mongo_ttl_index_recreate",
            collection=getattr(collection, "name", ""),
            index=name,
            expire_after_seconds=expire_after_seconds,
        )
        try:
            collection.drop_index(name)
        except Exception:
            pass
        collection.create_index(keys, expireAfterSeconds=expire_after_seconds, name=name)


def build_stacktrace(error: Exception) -> str:
    """生成适合存 Mongo 的异常栈。"""
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _json_default(value: Any) -> Any:
    """将 Milvus / protobuf 等返回对象转成可 JSON 序列化的基础类型。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_default(item) for item in value]
    if hasattr(value, "items"):
        try:
            return {str(key): _json_default(item) for key, item in value.items()}
        except Exception:
            pass
    if isinstance(value, Iterable):
        try:
            return [_json_default(item) for item in value]
        except Exception:
            pass
    return str(value)
