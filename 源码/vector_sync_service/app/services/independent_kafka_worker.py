"""Independent Kafka data-plane processors for metadata, fulltext, and images.

Every processor consumes the same Kafka topic through a different consumer
group.  It never writes a successful task to Mongo: Mongo only receives a
target-bound replay record when processing fails.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, AsyncIterator, Literal, Optional

from app.config import settings
from app.models.sync_runtime import (
    BatchProcessingResult,
    ConsumedRecord,
    ErrorType,
    ProcessOutcome,
    ReplayMode,
    ScopeType,
    SourceEventMeta,
    SyncOverallStatus,
    SyncStage,
)
from app.observability.logger import get_logger
from app.services.batch_processor_existing import process_batch_existing_arch, retry_failure_task
from app.services.collection_targets import (
    build_document_collection_signature,
    build_document_index_spec,
    build_fulltext_chunk_index_spec,
    build_fulltext_collection_signature,
    build_image_collection_signature,
    build_image_dense_index_spec,
    collection_target_binding,
    resolve_fulltext_collection_target,
)
from app.services.fulltext_task_manager import build_fulltext_task_documents
from app.services.fulltext_worker import FullTextWorker
from app.services.image_task_manager import build_image_task_documents
from app.services.image_worker import ImageWorker
from app.services.message_utils import infer_data_status
from app.services.milvus_chunk_service import MilvusChunkService
from app.services.milvus_image_service import MilvusImageService
from app.services.mongo_state_manager import MongoStateManager, get_mongo_state_manager
from app.services.solr_source import SolrSourceError, SolrSourceResolver, is_solr_doc_id_payload

logger = get_logger(__name__)
WorkerRole = Literal["metadata", "fulltext", "image"]
_VALID_ROLES = {"metadata", "fulltext", "image"}


class _NoopFullTextTaskRepository:
    """Direct Kafka workers do not retain successful fulltext queue tasks."""

    async def mark_task_success(self, *_args, **_kwargs) -> None:
        return None

    async def mark_task_failure(self, *_args, **_kwargs) -> None:
        return None


class _NoopImageTaskRepository:
    """Direct Kafka workers do not retain successful image queue tasks."""

    async def mark_task_success(self, *_args, **_kwargs) -> None:
        return None

    async def mark_task_success_minimal(self, *_args, **_kwargs) -> None:
        return None

    async def mark_task_failure(self, *_args, **_kwargs) -> None:
        return None


class _NoopImageStateManager:
    """Keep direct image execution out of Mongo until a Kafka event fails."""

    async def save_artifact(self, *_args, **_kwargs) -> str:
        return ""

    async def persist_failure(self, *_args, **_kwargs) -> None:
        return None

    async def refresh_async_pipeline_status(self, *_args, **_kwargs) -> None:
        return None


def kafka_group_id_for_role(role: str) -> str:
    normalized = _normalize_role(role)
    configured = {
        "metadata": settings.KAFKA_METADATA_GROUP_ID,
        "fulltext": settings.KAFKA_FULLTEXT_GROUP_ID,
        "image": settings.KAFKA_IMAGE_GROUP_ID,
    }[normalized]
    value = str(configured or "").strip()
    if not value:
        raise ValueError(f"Kafka group id is required for role={normalized}")
    if not bool(getattr(settings, "KAFKA_TARGET_VERSIONED_GROUP_ENABLED", True)):
        return value
    # Kafka stores offsets by group id.  Folding the target identity into the
    # group makes a collection rename or build-index change a fresh replay,
    # while search-only nprobe/ef changes leave the group unchanged.
    target_version = sha256(_target_group_identity(normalized).encode("utf-8")).hexdigest()[:12]
    return f"{value}--target-{target_version}"


class IndependentKafkaWorker:
    """Process complete Kafka events for exactly one target collection lane."""

    def __init__(
        self,
        role: WorkerRole | str,
        *,
        state_manager: Optional[MongoStateManager] = None,
        fulltext_worker: Optional[FullTextWorker] = None,
        image_worker: Optional[ImageWorker] = None,
        source_resolver: Optional[SolrSourceResolver] = None,
    ) -> None:
        self.role = _normalize_role(role)
        self.state_manager = state_manager or get_mongo_state_manager()
        self.fulltext_worker = fulltext_worker
        self.image_worker = image_worker
        self.source_resolver = source_resolver
        self._chunk_service: Optional[MilvusChunkService] = None
        self._image_service: Optional[MilvusImageService] = None
        self._image_task_semaphore = asyncio.Semaphore(
            max(int(settings.IMAGE_WORKER_CONCURRENCY or 1), 1)
        )
        self._source_resolve_semaphore = asyncio.Semaphore(
            max(int(settings.SOLR_RESOLVE_CONCURRENCY or 1), 1)
        )
        self._source_version_cache: dict[tuple[str, str, str, str], tuple[str, str]] = {}
        self._source_version_binding: Optional[dict[str, str]] = None
        # Parallel image batches are safe for different documents, but two
        # batches must never replace the same document snapshot concurrently.
        self._image_document_locks: dict[str, asyncio.Lock] = {}
        self._image_unknown_document_lock = asyncio.Lock()

    async def process_batch(self, batch: list[ConsumedRecord]) -> BatchProcessingResult:
        if self.role == "metadata":
            return await self._process_metadata_batch(batch)
        if self.role == "image":
            async with self._image_batch_locks(batch):
                return await self._process_image_batch(batch)

        outcomes = await asyncio.gather(
            *(self._process_record(record) for record in batch),
            return_exceptions=False,
        )
        return BatchProcessingResult(outcomes=outcomes)

    @asynccontextmanager
    async def _image_batch_locks(self, batch: list[ConsumedRecord]) -> AsyncIterator[None]:
        """Acquire document locks in stable order for safe batch overlap."""
        document_ids: set[str] = set()
        unknown_document = False
        for record in batch:
            if not record.is_valid:
                continue
            ids = _document_ids(record.parsed_payload or {})
            if ids:
                document_ids.update(ids)
            else:
                unknown_document = True

        async with AsyncExitStack() as stack:
            if unknown_document:
                await stack.enter_async_context(self._image_unknown_document_lock)
            for doc_id in sorted(document_ids):
                lock = self._image_document_locks.setdefault(doc_id, asyncio.Lock())
                await stack.enter_async_context(lock)
            yield

    async def _process_image_batch(
        self,
        batch: list[ConsumedRecord],
    ) -> BatchProcessingResult:
        """Process Kafka image events in memory and write all complete docs together."""
        outcomes: list[Optional[ProcessOutcome]] = [None] * len(batch)
        document_units: list[dict[str, Any]] = []

        hydrated_records = await self._hydrate_valid_records(batch, include_images=True)
        for record_index, record in enumerate(batch):
            if not record.is_valid:
                outcomes[record_index] = await self._persist_record_failure(
                    record,
                    error_code="KAFKA_EVENT_INVALID",
                    error_message=record.decode_error or record.validation_error or "invalid Kafka event",
                    error_type=ErrorType.NON_RETRYABLE,
                    stage=SyncStage.KAFKA_VALIDATE,
                    replay_mode=ReplayMode.FROM_RAW_EVENT,
                    replay_payload={"raw_text": record.raw_text},
                )
                continue

            hydrated = hydrated_records[record_index]
            if isinstance(hydrated, SolrSourceError):
                outcomes[record_index] = await self._persist_record_failure(
                    record,
                    error_code="SOLR_SOURCE_RESOLVE_ERROR",
                    error_message=str(hydrated),
                    error_type=ErrorType.RETRYABLE,
                    stage=SyncStage.SOURCE_RESOLVE,
                    replay_mode=ReplayMode.FROM_RAW_EVENT,
                    replay_payload={"kind": "solr_doc_id", "doc_id": _first_document_id(record.parsed_payload or {})},
                )
                continue
            record = hydrated

            payload = record.parsed_payload or {}
            if _event_type(payload) == "DELETE":
                outcomes[record_index] = await self._process_delete_record(record, payload)
                continue

            source_snapshot = self._source_snapshot(payload)
            if await self._is_unchanged_source(source_snapshot):
                outcomes[record_index] = _source_unchanged_outcome(source_snapshot)
                continue

            try:
                tasks = build_image_task_documents(payload)
            except Exception as error:
                outcomes[record_index] = await self._persist_record_failure(
                    record,
                    error_code="IMAGE_TASK_BUILD_ERROR",
                    error_message=str(error),
                    error_type=ErrorType.NON_RETRYABLE if isinstance(error, ValueError) else ErrorType.RETRYABLE,
                    stage=SyncStage.IMAGE_DISCOVER,
                    replay_mode=ReplayMode.FROM_RAW_EVENT,
                    replay_payload={"kind": "build_from_event", "payload": payload},
                )
                continue

            if not tasks:
                await self._mark_source_processed(source_snapshot)
                outcomes[record_index] = _outcome(
                    final_status=SyncOverallStatus.SUCCESS.value,
                    commit_safe=True,
                    message="image_not_present",
                )
                continue

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for task in tasks:
                task["event_type"] = "UPSERT"
                task["source_event_meta"] = _source_meta_dict(record.source_event_meta)
                task["_source_timings_ms"] = _source_timings(payload)
                grouped[str(task.get("doc_id") or "")].append(task)
            for doc_id, document_tasks in grouped.items():
                document_units.append(
                    {
                        "record_index": record_index,
                        "record": record,
                        "doc_id": doc_id,
                        "tasks": document_tasks,
                        "source_snapshot": source_snapshot,
                    }
                )

        if not document_units:
            return BatchProcessingResult(outcomes=[item for item in outcomes if item is not None])

        worker = self._get_image_worker()
        all_tasks = [task for unit in document_units for task in unit["tasks"]]
        logger.info(
            "image_direct_batch_processing_started",
            record_count=len(batch),
            document_count=len(document_units),
            task_count=len(all_tasks),
            task_timeout_seconds=float(
                getattr(settings, "IMAGE_WORKER_TASK_TIMEOUT_SEC", 0) or 0
            ),
        )
        all_results = await _run_limited(
            all_tasks,
            lambda task: self._process_image_task_with_timeout(worker, task),
            concurrency=settings.IMAGE_WORKER_CONCURRENCY,
            semaphore=self._image_task_semaphore,
        )
        logger.info(
            "image_direct_batch_processing_finished",
            record_count=len(batch),
            document_count=len(document_units),
            task_count=len(all_tasks),
            failed_task_count=sum(1 for item in all_results if not item.success),
        )

        failures_by_record: dict[int, list[ProcessOutcome]] = defaultdict(list)
        complete_documents: list[dict[str, Any]] = []
        result_offset = 0
        for unit in document_units:
            task_count = len(unit["tasks"])
            results = all_results[result_offset:result_offset + task_count]
            result_offset += task_count
            failed = [item for item in results if not item.success]
            records = [item.image_record for item in results if item.image_record is not None]
            if failed:
                failure = await self._persist_image_document_failure(
                    unit,
                    failed=failed,
                    error_code="IMAGE_TASK_PROCESS_ERROR",
                    error_message=(
                        str(getattr(failed[0], "error_message", "") or "image task did not produce a record")
                    ),
                    stage=SyncStage.IMAGE_EMBED_CAPTION,
                )
                failures_by_record[int(unit["record_index"])].append(failure)
                continue
            # A quality-quarantined image completes successfully without a
            # Milvus row.  Other indexable images from the same document still
            # form the document snapshot and are written together.
            if records:
                complete_documents.append({**unit, "records": records})

        for write_batch in _unique_document_write_batches(complete_documents):
            documents = {
                str(unit["doc_id"]): list(unit["records"])
                for unit in write_batch
            }
            configured_write_mode = str(
                getattr(settings, "IMAGE_MILVUS_WRITE_MODE", "upsert") or "upsert"
            ).strip().lower()
            use_insert_or_skip = (
                not bool(settings.IMAGE_DIRECT_STALE_CLEANUP_ENABLED)
                and configured_write_mode in {
                    "insert_or_skip_if_exists",
                    "insert_only",
                }
            )
            flat_records = [
                record
                for unit in write_batch
                for record in unit["records"]
            ]
            write_timeout_seconds = max(
                float(getattr(settings, "IMAGE_WORKER_TASK_TIMEOUT_SEC", 0) or 0),
                0.0,
            )
            write_started = asyncio.get_running_loop().time()
            try:
                image_service = self._get_image_service()
                if use_insert_or_skip:
                    logger.info(
                        "image_milvus_batch_write_mode_selected",
                        mode="insert_or_skip_if_exists",
                        document_count=len(documents),
                        image_count=len(flat_records),
                    )
                    write_operation = asyncio.to_thread(
                        image_service.insert_or_skip_images_by_img_id_batch,
                        settings.IMAGE_COLLECTION_NAME,
                        flat_records,
                    )
                else:
                    logger.info(
                        "image_milvus_batch_write_mode_selected",
                        mode="replace_images_for_documents",
                        configured_mode=configured_write_mode,
                        document_count=len(documents),
                        image_count=len(flat_records),
                    )
                    write_operation = asyncio.to_thread(
                        image_service.replace_images_for_documents,
                        settings.IMAGE_COLLECTION_NAME,
                        documents,
                        complete_snapshot=settings.IMAGE_DIRECT_STALE_CLEANUP_ENABLED,
                    )
                if write_timeout_seconds > 0:
                    await asyncio.wait_for(write_operation, timeout=write_timeout_seconds)
                else:
                    await write_operation
            except Exception as error:
                if isinstance(error, asyncio.TimeoutError):
                    logger.error(
                        "image_direct_milvus_write_timeout",
                        document_count=len(documents),
                        image_count=sum(len(items) for items in documents.values()),
                        timeout_seconds=write_timeout_seconds,
                        elapsed_seconds=round(
                            asyncio.get_running_loop().time() - write_started,
                            3,
                        ),
                    )
                for unit in write_batch:
                    failure = await self._persist_image_document_failure(
                        unit,
                        failed=[],
                        error_code="IMAGE_MILVUS_WRITE_ERROR",
                        error_message=str(error),
                        stage=SyncStage.MILVUS_WRITE_IMAGES,
                    )
                    failures_by_record[int(unit["record_index"])].append(failure)

        documents_by_record: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for unit in document_units:
            documents_by_record[int(unit["record_index"])].append(unit)
        for record_index, document_list in documents_by_record.items():
            if outcomes[record_index] is not None:
                continue
            failures = failures_by_record.get(record_index, [])
            if failures:
                outcomes[record_index] = _outcome(
                    final_status=SyncOverallStatus.FAILURE.value,
                    commit_safe=all(item.commit_safe for item in failures),
                    failure_persisted=all(item.failure_persisted for item in failures),
                    message="image_document_processing_failed",
                    doc_id=str(document_list[0].get("doc_id") or ""),
                )
            else:
                await self._mark_source_processed(document_list[0].get("source_snapshot"))
                outcomes[record_index] = _outcome(
                    final_status=SyncOverallStatus.SUCCESS.value,
                    commit_safe=True,
                    message="image_document_batch_written",
                    doc_id=str(document_list[0].get("doc_id") or ""),
                )

        return BatchProcessingResult(outcomes=[item for item in outcomes if item is not None])

    async def _process_image_task_with_timeout(
        self,
        worker: ImageWorker,
        task: dict[str, Any],
    ) -> Any:
        """Bound direct-Kafka image work so one task cannot freeze offset commits."""
        timeout_seconds = max(
            float(getattr(settings, "IMAGE_WORKER_TASK_TIMEOUT_SEC", 0) or 0),
            0.0,
        )
        task_id = str(task.get("task_id") or task.get("_id") or "")
        doc_id = str(task.get("doc_id") or "")
        started = asyncio.get_running_loop().time()
        try:
            operation = worker.process_task(task)
            if timeout_seconds > 0:
                return await asyncio.wait_for(operation, timeout=timeout_seconds)
            return await operation
        except asyncio.TimeoutError:
            elapsed_seconds = asyncio.get_running_loop().time() - started
            logger.error(
                "image_direct_task_timeout",
                task_id=task_id,
                doc_id=doc_id,
                timeout_seconds=timeout_seconds,
                elapsed_seconds=round(elapsed_seconds, 3),
            )
            from app.services.image_worker import ImageWorkerResult

            return ImageWorkerResult(
                task_id=task_id,
                success=False,
                artifact_refs=[],
                error_code="IMAGE_WORKER_TASK_TIMEOUT",
                error_message=(
                    f"direct image task exceeded {timeout_seconds:g}s timeout"
                ),
                retryable=True,
            )

    async def replay_failure(self, failure_task: dict[str, Any]) -> ProcessOutcome:
        """Replay one already claimed, target-bound failure from Mongo."""
        if self.role == "metadata":
            return await retry_failure_task(failure_task)

        payload = failure_task.get("replay_payload")
        if not isinstance(payload, dict) or not payload:
            await self.state_manager.mark_failure_skipped_stale(
                str(failure_task.get("failure_key") or ""),
                "direct worker failure has no replay payload",
            )
            return _outcome(
                final_status=SyncOverallStatus.SKIPPED_STALE.value,
                commit_safe=True,
                message="retry_skipped_missing_replay_payload",
                doc_id=str(failure_task.get("doc_id") or ""),
                record_update_time=str(failure_task.get("record_update_time") or ""),
            )

        if self.role == "fulltext":
            return await self._replay_fulltext_failure(failure_task, payload)
        return await self._replay_image_failure(failure_task, payload)

    async def close(self) -> None:
        if self.source_resolver is not None:
            await self.source_resolver.close()
        if self.image_worker is not None:
            await self.image_worker.close()
        if self._chunk_service is not None:
            self._chunk_service.close()
        if self._image_service is not None:
            self._image_service.close()

    async def _process_metadata_batch(self, batch: list[ConsumedRecord]) -> BatchProcessingResult:
        """Keep the high-throughput bulk lane for UPSERT and safe DELETE semantics."""
        upserts: list[ConsumedRecord] = []
        deletes: list[ConsumedRecord] = []
        result = BatchProcessingResult()
        hydrated_records = await self._hydrate_valid_records(batch, include_images=False)
        source_snapshots: dict[str, dict[str, str]] = {}
        for record_index, record in enumerate(batch):
            if not record.is_valid:
                upserts.append(record)
                continue
            hydrated = hydrated_records[record_index]
            if isinstance(hydrated, SolrSourceError):
                result.outcomes.append(
                    await self._persist_record_failure(
                        record,
                        error_code="SOLR_SOURCE_RESOLVE_ERROR",
                        error_message=str(hydrated),
                        error_type=ErrorType.RETRYABLE,
                        stage=SyncStage.SOURCE_RESOLVE,
                        replay_mode=ReplayMode.FROM_RAW_EVENT,
                        replay_payload={"kind": "solr_doc_id", "doc_id": _first_document_id(record.parsed_payload or {})},
                    )
                )
                continue
            if _event_type(hydrated.parsed_payload or {}) == "DELETE":
                deletes.append(hydrated)
            else:
                snapshot = self._source_snapshot(hydrated.parsed_payload or {})
                if await self._is_unchanged_source(snapshot):
                    result.outcomes.append(_source_unchanged_outcome(snapshot))
                else:
                    upserts.append(hydrated)
                    if snapshot is not None:
                        source_snapshots[snapshot["doc_id"]] = snapshot

        if upserts:
            metadata_result = await process_batch_existing_arch(upserts, mode="metadata_bulk")
            result.outcomes.extend(metadata_result.outcomes)
            for outcome in metadata_result.outcomes:
                if outcome.is_success:
                    await self._mark_source_processed(source_snapshots.get(str(outcome.doc_id or "")))
        if deletes:
            result.outcomes.extend(
                (
                    await process_batch_existing_arch(deletes, mode="metadata_only")
                ).outcomes
            )
        return result

    async def _process_record(self, record: ConsumedRecord) -> ProcessOutcome:
        if not record.is_valid:
            return await self._persist_record_failure(
                record,
                error_code="KAFKA_EVENT_INVALID",
                error_message=record.decode_error or record.validation_error or "invalid Kafka event",
                error_type=ErrorType.NON_RETRYABLE,
                stage=SyncStage.KAFKA_VALIDATE,
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                replay_payload={"raw_text": record.raw_text},
            )

        try:
            record = await self._hydrate_record_from_solr(
                record,
                include_images=self.role == "image",
            )
        except SolrSourceError as error:
            return await self._persist_record_failure(
                record,
                error_code="SOLR_SOURCE_RESOLVE_ERROR",
                error_message=str(error),
                error_type=ErrorType.RETRYABLE,
                stage=SyncStage.SOURCE_RESOLVE,
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                replay_payload={"kind": "solr_doc_id", "doc_id": _first_document_id(record.parsed_payload or {})},
            )

        payload = record.parsed_payload or {}
        if _event_type(payload) == "DELETE":
            return await self._process_delete_record(record, payload)

        source_snapshot = self._source_snapshot(payload)
        if await self._is_unchanged_source(source_snapshot):
            return _source_unchanged_outcome(source_snapshot)

        try:
            task_documents = (
                build_fulltext_task_documents(payload)
                if self.role == "fulltext"
                else build_image_task_documents(payload)
            )
        except Exception as error:
            return await self._persist_record_failure(
                record,
                error_code=("FULLTEXT_TASK_BUILD_ERROR" if self.role == "fulltext" else "IMAGE_TASK_BUILD_ERROR"),
                error_message=str(error),
                error_type=ErrorType.NON_RETRYABLE if isinstance(error, ValueError) else ErrorType.RETRYABLE,
                stage=SyncStage.FULLTEXT_DISCOVER if self.role == "fulltext" else SyncStage.IMAGE_DISCOVER,
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                replay_payload={"kind": "build_from_event", "payload": payload},
            )

        if not task_documents:
            await self._mark_source_processed(source_snapshot)
            return _outcome(
                final_status=SyncOverallStatus.SUCCESS.value,
                commit_safe=True,
                message=f"{self.role}_not_present",
            )

        for task in task_documents:
            task["event_type"] = "UPSERT"
            task["source_event_meta"] = _source_meta_dict(record.source_event_meta)
            task["_source_timings_ms"] = _source_timings(payload)

        if self.role == "fulltext":
            outcome = await self._process_fulltext_tasks(task_documents)
        else:
            outcome = await self._process_image_tasks(task_documents, source_record=record)
        if outcome.is_success:
            await self._mark_source_processed(source_snapshot)
        return outcome

    async def _process_delete_record(
        self,
        record: ConsumedRecord,
        payload: dict[str, Any],
    ) -> ProcessOutcome:
        doc_ids = _document_ids(payload)
        # The production full-text lane runs in append-only mode.  A DELETE
        # event is acknowledged after being recorded as skipped so Kafka can
        # continue, but it must never reach Milvus.  This also covers the
        # synthetic DELETE generated when Solr no longer returns a document.
        if self.role == "fulltext" and bool(getattr(settings, "LIVE_E2E_NO_DELETE", False)):
            return _outcome(
                final_status=SyncOverallStatus.SKIPPED_STALE.value,
                commit_safe=True,
                message="fulltext_delete_skipped_no_delete",
                doc_id=doc_ids[0] if doc_ids else "",
            )
        if not doc_ids:
            return await self._persist_record_failure(
                record,
                error_code="DELETE_DOCUMENT_ID_MISSING",
                error_message="DELETE event contains no document id",
                error_type=ErrorType.NON_RETRYABLE,
                stage=SyncStage.DELETE_CHUNKS if self.role == "fulltext" else SyncStage.DELETE_IMAGES,
                replay_mode=ReplayMode.FROM_DELETE_CONTEXT,
                replay_payload={"kind": "delete_document", "doc_ids": []},
            )

        try:
            if self.role == "fulltext":
                service = self._get_chunk_service()
                for doc_id in doc_ids:
                    await asyncio.to_thread(
                        service.delete_chunks_by_doc_id,
                        settings.resolved_fulltext_chunk_collection_name,
                        doc_id,
                    )
            else:
                service = self._get_image_service()
                for doc_id in doc_ids:
                    await asyncio.to_thread(
                        service.delete_images_by_doc_id,
                        settings.IMAGE_COLLECTION_NAME,
                        doc_id,
                    )
        except Exception as error:
            return await self._persist_record_failure(
                record,
                error_code="MILVUS_DELETE_ERROR",
                error_message=str(error),
                error_type=ErrorType.RETRYABLE,
                stage=SyncStage.DELETE_CHUNKS if self.role == "fulltext" else SyncStage.DELETE_IMAGES,
                replay_mode=ReplayMode.FROM_DELETE_CONTEXT,
                replay_payload={"kind": "delete_document", "doc_ids": doc_ids},
            )

        return _outcome(
            final_status=SyncOverallStatus.DELETED.value,
            commit_safe=True,
            message=f"{self.role}_delete_completed",
            doc_id=doc_ids[0],
        )

    async def _hydrate_record_from_solr(
        self,
        record: ConsumedRecord,
        *,
        include_images: bool,
    ) -> ConsumedRecord:
        """Turn the bare Kafka document ID contract into one source snapshot."""
        payload = record.parsed_payload or {}
        if not is_solr_doc_id_payload(payload):
            return record
        doc_id = _first_document_id(payload)
        if not doc_id:
            raise SolrSourceError("Kafka document-id message has no doc_id")
        if self.source_resolver is None:
            self.source_resolver = SolrSourceResolver()
        async with self._source_resolve_semaphore:
            started_at = asyncio.get_running_loop().time()
            resolved = await self.source_resolver.resolve_document(doc_id, include_images=include_images)
        if resolved.found:
            payload = dict(resolved.payload)
            payload["_pipeline_timings_ms"] = {
                "solr": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)
            }
            return replace(record, parsed_payload=payload)

        # A bare-ID topic carries the desired current state.  Missing source
        # data therefore means this lane must remove any prior V2 projection.
        deleted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return replace(
            record,
            parsed_payload={
                "data": [{"id": doc_id, "sysId": doc_id, "sysDataUpdateTime": deleted_at}],
                "operation": "DELETE",
                "source_mode": "solr_not_found",
            },
        )

    async def _hydrate_valid_records(
        self,
        records: list[ConsumedRecord],
        *,
        include_images: bool,
    ) -> list[ConsumedRecord | SolrSourceError]:
        """Resolve the bare-ID portion of one Kafka batch with bounded I/O."""
        results: list[ConsumedRecord | SolrSourceError] = list(records)
        candidates = [
            (index, record)
            for index, record in enumerate(records)
            if record.is_valid and is_solr_doc_id_payload(record.parsed_payload or {})
        ]
        if not candidates:
            return results
        resolved = await asyncio.gather(
            *(
                self._hydrate_record_from_solr(record, include_images=include_images)
                for _, record in candidates
            ),
            return_exceptions=True,
        )
        for (index, _record), value in zip(candidates, resolved):
            if isinstance(value, SolrSourceError):
                results[index] = value
            elif isinstance(value, Exception):
                results[index] = SolrSourceError(str(value))
            else:
                results[index] = value
        return results

    async def _process_fulltext_tasks(self, tasks: list[dict[str, Any]]) -> ProcessOutcome:
        worker = self._get_fulltext_worker()
        results = await _run_limited(
            tasks,
            worker.process_task,
            concurrency=settings.FULLTEXT_WORKER_CONCURRENCY,
        )
        failed = [item for item in results if not item.success]
        return _task_results_outcome("fulltext", tasks, failed)

    async def _process_image_tasks(
        self,
        tasks: list[dict[str, Any]],
        *,
        source_record: Optional[ConsumedRecord] = None,
    ) -> ProcessOutcome:
        worker = self._get_image_worker()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in tasks:
            grouped[str(task.get("doc_id") or "")].append(task)

        outcomes: list[ProcessOutcome] = []
        for doc_id, document_tasks in grouped.items():
            results = await _run_limited(
                document_tasks,
                worker.process_task,
                concurrency=settings.IMAGE_WORKER_CONCURRENCY,
                semaphore=self._image_task_semaphore,
            )
            failed = [item for item in results if not item.success]
            if failed:
                outcomes.append(_task_results_outcome("image", document_tasks, failed))
                continue

            records = [item.image_record for item in results if item.image_record is not None]
            if len(records) != len(document_tasks):
                # No stale cleanup is safe unless every image produced a Milvus record.
                outcomes.append(
                    _outcome(
                        final_status=SyncOverallStatus.SUCCESS.value,
                        commit_safe=True,
                        message="image_written_without_complete_snapshot_cleanup",
                        doc_id=doc_id,
                    )
                )
                continue

            try:
                await asyncio.to_thread(
                    self._get_image_service().replace_images_for_document,
                    settings.IMAGE_COLLECTION_NAME,
                    doc_id,
                    records,
                    complete_snapshot=settings.IMAGE_DIRECT_STALE_CLEANUP_ENABLED,
                )
                outcomes.append(
                    _outcome(
                        final_status=SyncOverallStatus.SUCCESS.value,
                        commit_safe=True,
                        message="image_snapshot_written_and_cleaned",
                        doc_id=doc_id,
                    )
                )
            except Exception as error:
                if source_record is None:
                    outcomes.append(
                        _outcome(
                            final_status=SyncOverallStatus.FAILURE.value,
                            commit_safe=False,
                            message=f"IMAGE_SNAPSHOT_CLEANUP_ERROR:{error}",
                            doc_id=doc_id,
                        )
                    )
                else:
                    outcomes.append(
                        await self._persist_record_failure(
                            source_record,
                            error_code="IMAGE_SNAPSHOT_CLEANUP_ERROR",
                            error_message=str(error),
                            error_type=ErrorType.RETRYABLE,
                            stage=SyncStage.MILVUS_WRITE_IMAGES,
                            replay_mode=ReplayMode.FROM_IMAGE_MANIFEST,
                            replay_payload={
                                "kind": "image_snapshot",
                                "doc_id": doc_id,
                                "tasks": document_tasks,
                            },
                        )
                    )

        failed_outcomes = [item for item in outcomes if item.is_failure]
        if failed_outcomes:
            return _outcome(
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=all(item.commit_safe for item in failed_outcomes),
                message="image_document_processing_failed",
                doc_id=failed_outcomes[0].doc_id,
            )
        return _outcome(
            final_status=SyncOverallStatus.SUCCESS.value,
            commit_safe=True,
            message="image_document_processing_completed",
            doc_id=outcomes[0].doc_id if outcomes else "",
        )

    async def _persist_image_document_failure(
        self,
        unit: dict[str, Any],
        *,
        failed: list[Any],
        error_code: str,
        error_message: str,
        stage: SyncStage,
    ) -> ProcessOutcome:
        record = unit["record"]
        tasks = list(unit.get("tasks") or [])
        retryable = any(bool(getattr(item, "retryable", False)) for item in failed)
        return await self._persist_record_failure(
            record,
            error_code=error_code,
            error_message=error_message,
            error_type=ErrorType.RETRYABLE if retryable or not failed else ErrorType.NON_RETRYABLE,
            stage=stage,
            replay_mode=ReplayMode.FROM_IMAGE_MANIFEST,
            replay_payload={
                "kind": "image_snapshot",
                "doc_id": str(unit.get("doc_id") or ""),
                "tasks": tasks,
                "download_lane": (
                    "slow"
                    if any(
                        str(getattr(item, "error_code", "") or "").startswith("IMAGE_DOWNLOAD")
                        for item in failed
                    )
                    else "fast"
                ),
            },
            doc_id=str(unit.get("doc_id") or ""),
            record_update_time=str((tasks[0] if tasks else {}).get("record_update_time") or ""),
        )

    async def _persist_record_failure(
        self,
        record: ConsumedRecord,
        *,
        error_code: str,
        error_message: str,
        error_type: ErrorType,
        stage: SyncStage,
        replay_mode: ReplayMode,
        replay_payload: dict[str, Any],
        doc_id: str = "",
        record_update_time: str = "",
    ) -> ProcessOutcome:
        resolved_doc_id = str(doc_id or "").strip()
        if not resolved_doc_id:
            doc_ids = _document_ids(record.parsed_payload or {})
            resolved_doc_id = doc_ids[0] if doc_ids else ""
        resolved_record_update_time = str(record_update_time or _record_update_time(record.parsed_payload or {}) or "")
        try:
            target_binding = self._target_binding()
            target_binding["source_version"] = str(
                resolved_record_update_time
            )
            target_binding["item_id"] = resolved_doc_id or record.event_fingerprint
            await self.state_manager.persist_failure(
                doc_id=resolved_doc_id or None,
                record_update_time=resolved_record_update_time or None,
                sub_doc_type=None,
                event_type="DELETE" if replay_mode == ReplayMode.FROM_DELETE_CONTEXT else "UPSERT",
                failure_stage=stage,
                scope_type=ScopeType.DOCUMENT,
                scope_id=resolved_doc_id or record.event_fingerprint,
                error_type=error_type,
                error_code=error_code,
                error_message=error_message,
                exception_class="IndependentKafkaWorkerError",
                stacktrace=error_message,
                replay_mode=replay_mode,
                artifact_refs=[],
                source_event_meta=record.source_event_meta,
                replay_payload=replay_payload,
                **target_binding,
            )
            persisted = True
        except Exception as error:
            logger.error(
                "independent_worker_failure_persist_failed",
                role=self.role,
                event_fingerprint=record.event_fingerprint,
                error=str(error),
            )
            persisted = False
        return _outcome(
            final_status=SyncOverallStatus.FAILURE.value,
            commit_safe=persisted,
            message=error_code,
            doc_id=resolved_doc_id,
            failure_persisted=persisted,
        )

    async def _replay_fulltext_failure(
        self,
        failure_task: dict[str, Any],
        payload: dict[str, Any],
    ) -> ProcessOutcome:
        if payload.get("kind") == "solr_doc_id":
            return await self._replay_source_document_failure(failure_task, payload, include_images=False)
        if payload.get("kind") == "delete_document":
            return await self._replay_delete_failure(failure_task, payload)
        if payload.get("kind") == "build_from_event":
            event_payload = payload.get("payload")
            if not isinstance(event_payload, dict):
                return await self._skip_missing_replay_payload(failure_task)
            tasks = build_fulltext_task_documents(event_payload)
            for task in tasks:
                task["event_type"] = "UPSERT"
                task["source_event_meta"] = failure_task.get("source_event_meta") or {}
            outcome = await self._process_fulltext_tasks(tasks)
            return await self._finish_replay(failure_task, outcome)

        result = await self._get_fulltext_worker().process_task(payload)
        outcome = _outcome(
            final_status=(SyncOverallStatus.SUCCESS.value if result.success else SyncOverallStatus.FAILURE.value),
            commit_safe=bool(result.success or result.failure_persisted),
            message=result.error_code or "fulltext_replayed",
            doc_id=str(payload.get("doc_id") or failure_task.get("doc_id") or ""),
            failure_persisted=result.failure_persisted,
        )
        return await self._finish_replay(failure_task, outcome)

    async def _replay_image_failure(
        self,
        failure_task: dict[str, Any],
        payload: dict[str, Any],
    ) -> ProcessOutcome:
        if payload.get("kind") == "solr_doc_id":
            return await self._replay_source_document_failure(failure_task, payload, include_images=True)
        if payload.get("kind") == "delete_document":
            return await self._replay_delete_failure(failure_task, payload)
        if payload.get("kind") == "build_from_event":
            event_payload = payload.get("payload")
            if not isinstance(event_payload, dict):
                return await self._skip_missing_replay_payload(failure_task)
            tasks = build_image_task_documents(event_payload)
        elif payload.get("kind") == "image_snapshot":
            tasks = payload.get("tasks")
            if not isinstance(tasks, list):
                return await self._skip_missing_replay_payload(failure_task)
        else:
            tasks = [payload]
        for task in tasks:
            if not isinstance(task, dict):
                return await self._skip_missing_replay_payload(failure_task)
            task["event_type"] = "UPSERT"
            task["source_event_meta"] = failure_task.get("source_event_meta") or {}
            if str(payload.get("download_lane") or "").strip().lower() == "slow":
                task["_download_lane"] = "slow"
        outcome = await self._process_image_tasks(tasks)
        return await self._finish_replay(failure_task, outcome)

    async def _replay_source_document_failure(
        self,
        failure_task: dict[str, Any],
        payload: dict[str, Any],
        *,
        include_images: bool,
    ) -> ProcessOutcome:
        doc_id = str(payload.get("doc_id") or failure_task.get("doc_id") or "").strip()
        if not doc_id:
            return await self._skip_missing_replay_payload(failure_task)
        try:
            if self.source_resolver is None:
                self.source_resolver = SolrSourceResolver()
            resolved = await self.source_resolver.resolve_document(doc_id, include_images=include_images)
        except SolrSourceError as error:
            return _outcome(
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=False,
                message=f"SOLR_SOURCE_RESOLVE_ERROR:{error}",
                doc_id=doc_id,
            )

        if not resolved.found:
            synthetic = {"data": [{"id": doc_id}], "operation": "DELETE"}
            return await self._replay_delete_failure(
                failure_task,
                {"kind": "delete_document", "doc_ids": [doc_id], "payload": synthetic},
            )
        tasks = (
            build_fulltext_task_documents(resolved.payload)
            if self.role == "fulltext"
            else build_image_task_documents(resolved.payload)
        )
        for task in tasks:
            task["event_type"] = "UPSERT"
            task["source_event_meta"] = failure_task.get("source_event_meta") or {}
        outcome = (
            await self._process_fulltext_tasks(tasks)
            if self.role == "fulltext"
            else await self._process_image_tasks(tasks)
        )
        return await self._finish_replay(failure_task, outcome)

    async def _replay_delete_failure(
        self,
        failure_task: dict[str, Any],
        payload: dict[str, Any],
    ) -> ProcessOutcome:
        doc_ids = [str(value).strip() for value in payload.get("doc_ids") or [] if str(value).strip()]
        if not doc_ids:
            return await self._skip_missing_replay_payload(failure_task)
        if self.role == "fulltext" and bool(getattr(settings, "LIVE_E2E_NO_DELETE", False)):
            outcome = _outcome(
                final_status=SyncOverallStatus.SKIPPED_STALE.value,
                commit_safe=True,
                message="fulltext_delete_replay_skipped_no_delete",
                doc_id=doc_ids[0],
            )
            await self.state_manager.mark_failure_skipped_stale(
                str(failure_task.get("failure_key") or ""),
                "fulltext delete replay skipped because LIVE_E2E_NO_DELETE is enabled",
            )
            return outcome
        try:
            service = self._get_chunk_service() if self.role == "fulltext" else self._get_image_service()
            for doc_id in doc_ids:
                operation = service.delete_chunks_by_doc_id if self.role == "fulltext" else service.delete_images_by_doc_id
                collection = (
                    settings.resolved_fulltext_chunk_collection_name
                    if self.role == "fulltext"
                    else settings.IMAGE_COLLECTION_NAME
                )
                await asyncio.to_thread(operation, collection, doc_id)
            outcome = _outcome(
                final_status=SyncOverallStatus.DELETED.value,
                commit_safe=True,
                message=f"{self.role}_delete_replayed",
                doc_id=doc_ids[0],
            )
        except Exception as error:
            outcome = _outcome(
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=False,
                message=f"MILVUS_DELETE_ERROR:{error}",
                doc_id=doc_ids[0],
            )
        return await self._finish_replay(failure_task, outcome)

    async def _finish_replay(
        self,
        failure_task: dict[str, Any],
        outcome: ProcessOutcome,
    ) -> ProcessOutcome:
        failure_key = str(failure_task.get("failure_key") or "")
        if outcome.is_success:
            await self.state_manager.mark_failure_done(failure_key, "replayed by independent worker")
            return outcome
        if outcome.failure_persisted:
            # Worker persistence refreshed the exact target-bound failure. Add
            # one retry attempt to the claimed record and preserve its payload.
            await self.state_manager.release_claimed_failure(
                failure_key,
                error_type=_failure_error_type(failure_task),
                error_code=outcome.message or "DIRECT_REPLAY_FAILED",
                error_message=outcome.message or "direct replay failed",
                exception_class="IndependentKafkaWorkerError",
                stacktrace=outcome.message or "direct replay failed",
                artifact_refs=list(failure_task.get("artifact_refs") or []),
            )
            return _outcome(
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=True,
                message=outcome.message,
                doc_id=outcome.doc_id,
                failure_persisted=True,
            )
        return outcome

    async def _skip_missing_replay_payload(self, failure_task: dict[str, Any]) -> ProcessOutcome:
        await self.state_manager.mark_failure_skipped_stale(
            str(failure_task.get("failure_key") or ""),
            "direct worker replay payload is incomplete",
        )
        return _outcome(
            final_status=SyncOverallStatus.SKIPPED_STALE.value,
            commit_safe=True,
            message="retry_skipped_incomplete_replay_payload",
            doc_id=str(failure_task.get("doc_id") or ""),
        )

    async def _is_unchanged_source(self, snapshot: Optional[dict[str, str]]) -> bool:
        if not snapshot or not bool(getattr(settings, "PIPELINE_VERSIONED_DEDUP_ENABLED", True)):
            return False
        key = self._source_version_key(snapshot)
        current = (snapshot["source_version"], snapshot["source_hash"])
        if self._source_version_cache.get(key) == current:
            return True

        getter = getattr(self.state_manager, "get_pipeline_source_version", None)
        if not callable(getter):
            return False
        try:
            stored = await getter(
                pipeline=snapshot["pipeline"],
                logical_collection=snapshot["logical_collection"],
                target_signature=snapshot["target_signature"],
                doc_id=snapshot["doc_id"],
            )
        except Exception as error:
            # Deduplication must fail open. A completed write is still valid
            # when the optional optimization state cannot be read.
            logger.warning(
                "pipeline_source_version_read_failed",
                role=self.role,
                doc_id=snapshot["doc_id"],
                error=str(error),
            )
            return False
        if not isinstance(stored, dict):
            return False
        stored_value = (
            str(stored.get("source_version") or ""),
            str(stored.get("source_hash") or ""),
        )
        if stored_value != current:
            return False
        self._source_version_cache[key] = current
        return True

    async def _mark_source_processed(self, snapshot: Optional[dict[str, str]]) -> None:
        if not snapshot or not bool(getattr(settings, "PIPELINE_VERSIONED_DEDUP_ENABLED", True)):
            return
        key = self._source_version_key(snapshot)
        current = (snapshot["source_version"], snapshot["source_hash"])
        self._source_version_cache[key] = current
        upsert = getattr(self.state_manager, "upsert_pipeline_source_version", None)
        if not callable(upsert):
            return
        try:
            await upsert(
                pipeline=snapshot["pipeline"],
                logical_collection=snapshot["logical_collection"],
                target_signature=snapshot["target_signature"],
                doc_id=snapshot["doc_id"],
                source_version=snapshot["source_version"],
                source_hash=snapshot["source_hash"],
            )
        except Exception as error:
            logger.warning(
                "pipeline_source_version_write_failed",
                role=self.role,
                doc_id=snapshot["doc_id"],
                error=str(error),
            )

    def _source_version_key(self, snapshot: dict[str, str]) -> tuple[str, str, str, str]:
        return (
            snapshot["pipeline"],
            snapshot["logical_collection"],
            snapshot["target_signature"],
            snapshot["doc_id"],
        )

    def _source_target_binding(self) -> dict[str, str]:
        if self._source_version_binding is None:
            self._source_version_binding = self._target_binding()
        return self._source_version_binding

    def _source_snapshot(self, payload: dict[str, Any]) -> Optional[dict[str, str]]:
        if not bool(getattr(settings, "PIPELINE_VERSIONED_DEDUP_ENABLED", True)):
            return None
        if not callable(getattr(self.state_manager, "get_pipeline_source_version", None)):
            return None
        doc_ids = _document_ids(payload)
        if len(doc_ids) != 1:
            # A multi-document legacy payload is handled by its normal
            # idempotency path. Bare-ID topics resolve one current snapshot.
            return None
        doc_id = doc_ids[0]
        binding = self._source_target_binding()
        projection = _source_projection(self.role, payload, doc_id)
        source_hash = sha256(
            json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return {
            "pipeline": self.role,
            "logical_collection": str(binding.get("logical_collection") or ""),
            "target_signature": str(binding.get("target_signature") or ""),
            "doc_id": doc_id,
            "source_version": _record_update_time(payload),
            "source_hash": source_hash,
        }

    def _get_fulltext_worker(self) -> FullTextWorker:
        if self.fulltext_worker is None:
            self.fulltext_worker = FullTextWorker(
                repository=_NoopFullTextTaskRepository(),
                state_manager=self.state_manager,
                chunk_writer=self._get_chunk_service(),
                chunk_collection_name=settings.resolved_fulltext_chunk_collection_name,
            )
        return self.fulltext_worker

    def _get_image_worker(self) -> ImageWorker:
        if self.image_worker is None:
            self.image_worker = ImageWorker(
                repository=_NoopImageTaskRepository(),
                state_manager=_NoopImageStateManager(),
                # Direct Kafka mode collects all records of one document in
                # memory, then writes that document once in the batch writer.
                image_write_enabled=False,
                image_collection_name=settings.IMAGE_COLLECTION_NAME,
            )
        return self.image_worker

    def _get_chunk_service(self) -> MilvusChunkService:
        if self._chunk_service is None:
            self._chunk_service = MilvusChunkService()
        return self._chunk_service

    def _get_image_service(self) -> MilvusImageService:
        if self._image_service is None:
            self._image_service = MilvusImageService()
        return self._image_service

    def _target_binding(self) -> dict[str, str]:
        if self.role == "metadata":
            logical = settings.resolved_document_collection_name
            return {
                "pipeline": "metadata",
                "source_version": "",
                "item_id": "",
                "logical_collection": logical,
                "resolved_collection": logical,
                "target_signature": (
                    f"{build_document_collection_signature(vector_dim=settings.VECTOR_DIM)}|"
                    f"INDEX={build_document_index_spec().signature}"
                ),
            }
        if self.role == "fulltext":
            logical = settings.resolved_fulltext_chunk_collection_name
            service = self._get_chunk_service()
            try:
                target = resolve_fulltext_collection_target(
                    service.client,
                    logical_name=logical,
                    vector_dim=settings.VECTOR_DIM,
                    ensure_collection=None,
                )
                return collection_target_binding(target, pipeline="fulltext", source_version="", item_id="")
            except Exception:
                return {
                    "pipeline": "fulltext",
                    "source_version": "",
                    "item_id": "",
                    "logical_collection": logical,
                    "resolved_collection": logical,
                    "target_signature": (
                        f"{build_fulltext_collection_signature(vector_dim=settings.VECTOR_DIM)}|"
                        f"INDEX={build_fulltext_chunk_index_spec().signature}"
                    ),
                }

        logical = str(settings.IMAGE_COLLECTION_NAME or "scientific_images").strip()
        service = self._get_image_service()
        try:
            target = service.resolve_collection_target(collection_name=logical)
            return collection_target_binding(target, pipeline="image", source_version="", item_id="")
        except Exception:
            return {
                "pipeline": "image",
                "source_version": "",
                "item_id": "",
                "logical_collection": logical,
                "resolved_collection": logical,
                "target_signature": (
                    f"{build_image_collection_signature(schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE)}|"
                    f"INDEX={build_image_dense_index_spec().signature}"
                ),
            }


async def _run_limited(
    tasks: list[dict[str, Any]],
    func,
    *,
    concurrency: int,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> list[Any]:
    semaphore = semaphore or asyncio.Semaphore(max(int(concurrency or 1), 1))

    async def _run(task: dict[str, Any]):
        async with semaphore:
            return await func(task)

    return list(await asyncio.gather(*(_run(task) for task in tasks)))


def _unique_document_write_batches(
    documents: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Do not reorder two updates for the same document inside one Kafka batch."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_doc_ids: set[str] = set()
    for unit in documents:
        doc_id = str(unit.get("doc_id") or "")
        if current and doc_id in current_doc_ids:
            batches.append(current)
            current = []
            current_doc_ids = set()
        current.append(unit)
        current_doc_ids.add(doc_id)
    if current:
        batches.append(current)
    return batches


def _task_results_outcome(role: str, tasks: list[dict[str, Any]], failed: list[Any]) -> ProcessOutcome:
    if not failed:
        return _outcome(
            final_status=SyncOverallStatus.SUCCESS.value,
            commit_safe=True,
            message=f"{role}_tasks_completed",
            doc_id=str(tasks[0].get("doc_id") or "") if tasks else "",
        )
    return _outcome(
        final_status=SyncOverallStatus.FAILURE.value,
        commit_safe=all(bool(item.failure_persisted) for item in failed),
        message=f"{role}_task_failure",
        doc_id=str(tasks[0].get("doc_id") or "") if tasks else "",
        failure_persisted=all(bool(item.failure_persisted) for item in failed),
    )


def _outcome(
    *,
    final_status: str,
    commit_safe: bool,
    message: str,
    doc_id: str = "",
    record_update_time: str = "",
    failure_persisted: bool = False,
) -> ProcessOutcome:
    return ProcessOutcome(
        doc_id=doc_id or None,
        record_update_time=record_update_time or None,
        final_status=final_status,
        commit_safe=commit_safe,
        failure_persisted=failure_persisted,
        message=message,
    )


def _source_unchanged_outcome(snapshot: Optional[dict[str, str]]) -> ProcessOutcome:
    snapshot = snapshot or {}
    return _outcome(
        final_status=SyncOverallStatus.SKIPPED_STALE.value,
        commit_safe=True,
        message="source_version_unchanged",
        doc_id=str(snapshot.get("doc_id") or ""),
        record_update_time=str(snapshot.get("source_version") or ""),
    )


def _normalize_role(role: str) -> WorkerRole:
    value = str(role or "").strip().lower()
    if value not in _VALID_ROLES:
        raise ValueError(f"SYNC_WORKER_ROLE must be one of {sorted(_VALID_ROLES)}")
    return value  # type: ignore[return-value]


def _event_type(payload: dict[str, Any]) -> str:
    return str(infer_data_status(payload) or "UPSERT").strip().upper()


def _source_meta_dict(meta: SourceEventMeta) -> dict[str, Any]:
    return {
        "topic": meta.topic,
        "partition": meta.partition,
        "offset": meta.offset,
        "event_fingerprint": meta.event_fingerprint,
    }


def _document_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    values: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        nested = item.get("dp") if isinstance(item.get("dp"), dict) else {}
        raw = item.get("sysId") or item.get("id") or item.get("docId") or nested.get("sysId") or nested.get("id") or nested.get("docId")
        value = str(raw or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _first_document_id(payload: dict[str, Any]) -> str:
    values = _document_ids(payload)
    return values[0] if values else ""


def _record_update_time(payload: dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return ""
    item = data[0]
    nested = item.get("dp") if isinstance(item.get("dp"), dict) else {}
    return str(
        item.get("sysDataUpdateTime")
        or item.get("recordUpdateTime")
        or item.get("record_update_time")
        or nested.get("sysDataUpdateTime")
        or nested.get("recordUpdateTime")
        or ""
    ).strip()


def _source_timings(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("_pipeline_timings_ms")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = round(max(float(value), 0.0), 3)
        except (TypeError, ValueError):
            continue
    return result


def _source_projection(role: WorkerRole, payload: dict[str, Any], doc_id: str) -> dict[str, Any]:
    """Hash only source fields that can change the lane's written projection."""
    item: dict[str, Any] = {}
    for candidate in payload.get("data") or []:
        if not isinstance(candidate, dict):
            continue
        if doc_id in {
            str(candidate.get("id") or "").strip(),
            str(candidate.get("sysId") or "").strip(),
            str(candidate.get("docId") or "").strip(),
        }:
            item = candidate
            break
    if not item:
        item = {"id": doc_id}
    common = {
        "id": doc_id,
        "sysDataUpdateTime": item.get("sysDataUpdateTime"),
        "cls": item.get("cls"),
        "dp": item.get("dp"),
    }
    operations = item.get("op") if isinstance(item.get("op"), dict) else {}
    if role == "metadata":
        return {**common, "op": operations}
    if role == "fulltext":
        return {**common, "hasFullTextFile": operations.get("hasFullTextFile")}
    return {
        **common,
        "hasFullTextFile": operations.get("hasFullTextFile"),
        "hasPictureFile": operations.get("hasPictureFile"),
    }


def _target_group_identity(role: WorkerRole) -> str:
    if role == "metadata":
        return "|".join(
            [
                role,
                settings.resolved_document_collection_name,
                build_document_collection_signature(vector_dim=settings.VECTOR_DIM),
                build_document_index_spec().signature,
            ]
        )
    if role == "fulltext":
        return "|".join(
            [
                role,
                settings.resolved_fulltext_chunk_collection_name,
                build_fulltext_collection_signature(vector_dim=settings.VECTOR_DIM),
                build_fulltext_chunk_index_spec().signature,
            ]
        )
    return "|".join(
        [
            role,
            str(settings.IMAGE_COLLECTION_NAME or "scientific_images").strip(),
            build_image_collection_signature(schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE),
            build_image_dense_index_spec().signature,
        ]
    )


def _failure_error_type(failure_task: dict[str, Any]) -> ErrorType:
    try:
        return ErrorType(str(failure_task.get("error_type") or ErrorType.RETRYABLE.value))
    except ValueError:
        return ErrorType.RETRYABLE
