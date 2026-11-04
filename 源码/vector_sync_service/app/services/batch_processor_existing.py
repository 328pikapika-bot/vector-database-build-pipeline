"""当前标量链路批处理与重试编排。"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.config import settings
from app.models.document import Document
from app.models.sync_runtime import (
    ArtifactType,
    BatchProcessingResult,
    ConsumedRecord,
    DocumentContext,
    EmbeddingItemResult,
    ErrorType,
    FailureRef,
    ProcessOutcome,
    ReplayMode,
    ScopeType,
    StageExecutionStatus,
    SyncOverallStatus,
    SyncStage,
    normalize_record_update_time,
    parse_record_update_time,
)
from app.observability.logger import get_logger
from app.services.embedder import get_embedding_client
from app.services.fulltext_chunk_pipeline import delete_chunks_for_document
from app.services.fulltext_task_manager import extract_fulltext_tasks
from app.services.fulltext_task_repository import get_fulltext_task_repository
from app.services.image_delete_pipeline import delete_images_for_document
from app.services.image_task_manager import extract_image_tasks
from app.services.image_task_repository import get_image_task_repository
from app.services.message_utils import extract_item_value, infer_data_status
from app.services.collection_targets import (
    collection_target_binding,
    resolve_document_collection_target,
)
from app.services.milvus_chunk_service import MilvusChunkService
from app.services.milvus_service import get_milvus_service
from app.services.mongo_state_manager import (
    MongoPersistenceError,
    PIPELINE_STATUS_FAILURE,
    PIPELINE_STATUS_NOT_PRESENT,
    PIPELINE_STATUS_PENDING,
    PIPELINE_STATUS_PROCESSING,
    PIPELINE_STATUS_SUCCESS,
    build_stacktrace,
    get_mongo_state_manager,
)
from app.utils.parser import map_to_document

logger = get_logger(__name__)

# These calls only exist to retain successful-document audit history.  Failure
# persistence has separate operation names and is deliberately never skipped.
_SUCCESS_AUDIT_MONGO_OPERATIONS = {
    "initialize_status",
    "save_raw_event_artifact",
    "save_delete_artifact",
    "save_delete_chunks_artifact",
    "save_delete_images_artifact",
    "save_parsed_document",
    "save_parsed_article",
    "save_embed_input",
    "save_milvus_summary",
    "save_fulltext_file_meta",
    "save_image_manifest",
    "finalize_document_status_duplicate",
    "finalize_delete_status",
    "finalize_success_status",
    "update_pipeline_status_document_success",
    "update_pipeline_status_fulltext_not_present",
    "update_pipeline_status_fulltext_pending",
    "update_pipeline_status_image_not_present",
    "update_pipeline_status_image_pending",
    "refresh_pipeline_status_after_image_not_present",
    "refresh_pipeline_status_after_image_pending",
}
_METADATA_BULK_WRITE_LOCKS: dict[int, asyncio.Lock] = {}
_METADATA_BULK_TIMING_SPANS: dict[int, list[tuple[str, float, float]]] = {}
_EMPTY_METADATA_VECTOR_SENTINEL = 1e-6


def _empty_metadata_vector() -> list[float]:
    """Keep required document vector fields writable when a source field is absent."""
    vector_dim = max(int(settings.VECTOR_DIM or 0), 1)
    return [_EMPTY_METADATA_VECTOR_SENTINEL, *([0.0] * (vector_dim - 1))]


def _metadata_embedding_text(value: str) -> str:
    """Bound an individual field before it reaches the embedding model."""
    max_chars = max(int(getattr(settings, "METADATA_EMBEDDING_MAX_CHARS", 30_000) or 0), 1)
    return value[:max_chars]


def _is_skipped_empty_embedding(result: EmbeddingItemResult) -> bool:
    return bool(result.skipped_empty and result.error_code == "EMBEDDING_INPUT_EMPTY")


def _metadata_bulk_write_lock() -> asyncio.Lock:
    """Serialize Milvus bulk writes while allowing the next batch to embed."""
    loop = asyncio.get_running_loop()
    return _METADATA_BULK_WRITE_LOCKS.setdefault(id(loop), asyncio.Lock())


def _record_metadata_bulk_timing(stage: str, started_at: float) -> None:
    if not bool(getattr(settings, "METADATA_BULK_TIMING_ENABLED", False)):
        return
    loop = asyncio.get_running_loop()
    _METADATA_BULK_TIMING_SPANS.setdefault(id(loop), []).append(
        (stage, started_at, time.perf_counter())
    )


def reset_metadata_bulk_timing() -> None:
    """Clear optional per-event-loop timing spans before a benchmark run."""
    loop = asyncio.get_running_loop()
    _METADATA_BULK_TIMING_SPANS[id(loop)] = []


def get_metadata_bulk_timing_snapshot() -> dict[str, Any]:
    """Return per-stage total and calendar time for an optional benchmark."""
    loop = asyncio.get_running_loop()
    spans = _METADATA_BULK_TIMING_SPANS.get(id(loop), [])
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for stage, started_at, finished_at in spans:
        grouped[stage].append((started_at, finished_at))

    stages: dict[str, dict[str, float | int]] = {}
    for stage, intervals in grouped.items():
        ordered = sorted(intervals)
        merged: list[list[float]] = []
        for started_at, finished_at in ordered:
            if not merged or started_at > merged[-1][1]:
                merged.append([started_at, finished_at])
            else:
                merged[-1][1] = max(merged[-1][1], finished_at)
        stages[stage] = {
            "occurrences": len(intervals),
            "total_seconds": round(sum(end - start for start, end in intervals), 6),
            "calendar_seconds": round(sum(end - start for start, end in merged), 6),
        }
    return {"span_count": len(spans), "stages": stages}


class ScalarSyncPipeline:
    """当前文档标量链路的统一处理入口。"""

    def __init__(
        self,
        *,
        mode: str = "standard",
        document_collection_name: Optional[str] = None,
    ) -> None:
        self.manager = get_mongo_state_manager()
        self.milvus = get_milvus_service()
        self.embedder = get_embedding_client()
        self.mode = str(mode or "standard").strip().lower()
        self.success_audit_enabled = bool(
            getattr(settings, "MONGO_SUCCESS_AUDIT_ENABLED", True)
        )
        self.document_collection_name = str(
            document_collection_name or settings.resolved_document_collection_name
        ).strip()

    def _document_target(self):
        # Unit-test writers do not expose a Milvus client because they only
        # assert payloads. Production writers always resolve the target.
        if not hasattr(self.milvus, "client"):
            return None
        return resolve_document_collection_target(
            self.milvus.client,
            logical_name=self.document_collection_name,
            vector_dim=settings.VECTOR_DIM,
            ensure_collection=lambda name: self.milvus._ensure_document_collection_created(  # noqa: SLF001
                name,
                vector_dim=settings.VECTOR_DIM,
            ),
        )
    def _resolved_document_collection_name(self) -> str:
        target = self._document_target()
        return target.resolved_name if target is not None else self.document_collection_name

    def _failure_target_binding(self, *, context: Optional[DocumentContext], item_id: str) -> dict[str, str]:
        target = self._document_target()
        if target is None:
            return {
                "pipeline": "metadata",
                "source_version": str(context.record_update_time if context else ""),
                "item_id": str(item_id or ""),
                "logical_collection": self.document_collection_name,
                "resolved_collection": self.document_collection_name,
                "target_signature": "legacy-test-target",
            }
        return collection_target_binding(
            target,
            pipeline="metadata",
            source_version=str(context.record_update_time if context else ""),
            item_id=str(item_id or ""),
        )

    async def process_batch(self, batch: list[ConsumedRecord]) -> BatchProcessingResult:
        result = BatchProcessingResult()
        contexts: list[DocumentContext] = []

        expand_started_at = time.perf_counter()
        for record in batch:
            if not record.is_valid:
                result.outcomes.append(await self._process_invalid_record(record))
                continue

            expanded_contexts, immediate_outcomes = await self._expand_record(record)
            result.outcomes.extend(immediate_outcomes)
            contexts.extend(expanded_contexts)
        _record_metadata_bulk_timing("kafka_payload_expand", expand_started_at)

        dedup_started_at = time.perf_counter()
        deduped_contexts, duplicate_count = self._dedup_contexts(contexts)
        result.skip_duplicates = duplicate_count
        _record_metadata_bulk_timing("deduplicate", dedup_started_at)

        if self.mode == "metadata_bulk" and not self.success_audit_enabled:
            indexed_outcomes = await self._process_metadata_bulk(deduped_contexts)
        else:
            if self.mode == "metadata_bulk" and self.success_audit_enabled:
                logger.warning(
                    "metadata_bulk_requires_failure_only_mongo",
                    fallback_mode="standard",
                )
            indexed_outcomes = await self._process_contexts_for_batch(deduped_contexts)
        result.outcomes.extend(outcome for _, outcome in indexed_outcomes)

        return result

    async def _process_metadata_bulk(
        self,
        contexts: list[DocumentContext],
    ) -> list[tuple[int, ProcessOutcome]]:
        """Run a metadata-only batch with cross-document embedding and Milvus I/O.

        This path intentionally has no successful-document Mongo traffic.  It
        keeps a per-document failure boundary so a malformed document or a
        failed embedding is retriable without discarding the rest of the batch.
        """
        indexed_outcomes: list[tuple[int, ProcessOutcome]] = []
        prepared: list[tuple[int, DocumentContext, Any]] = []

        normalize_started_at = time.perf_counter()
        for index, context in enumerate(contexts):
            started_at = self._utcnow()
            try:
                document = map_to_document(Document, context.payload)
                articles = document.data or []
                if not articles:
                    raise ValueError("No parsed articles after normalization")
                article = articles[0]
                if not article.sysId:
                    article.sysId = context.doc_id
                prepared.append((index, context, article))
            except Exception as error:
                error_type, error_code = self._classify_processing_error(
                    error,
                    default_retryable=False,
                    default_code="ARTICLE_NORMALIZE_ERROR",
                )
                outcome = await self._persist_stage_failure(
                    context=context,
                    source="kafka",
                    stage=SyncStage.ARTICLE_NORMALIZE,
                    scope_type=ScopeType.ARTICLE,
                    scope_id=context.article_scope_id,
                    error_type=error_type,
                    error_code=error_code,
                    error_message=str(error),
                    exception_class=type(error).__name__,
                    stacktrace=build_stacktrace(error),
                    replay_mode=ReplayMode.FROM_RAW_EVENT,
                    artifact_refs=[],
                    retry_count_snapshot=0,
                    overall_status=SyncOverallStatus.FAILURE,
                    stage_started_at=started_at,
                )
                indexed_outcomes.append((index, outcome))
        _record_metadata_bulk_timing("document_normalize", normalize_started_at)

        field_plan = (
            ("title", "title_embed", SyncStage.DOC_EMBED_TITLE),
            ("abstract", "abstract_embed", SyncStage.DOC_EMBED_ABSTRACT),
            ("keyword", "keyword_embed", SyncStage.DOC_EMBED_KEYWORD),
        )
        embedding_inputs: list[str] = []
        embedding_targets: list[tuple[int, str, SyncStage]] = []
        embedding_input_started_at = time.perf_counter()
        for prepared_index, (_, _, article) in enumerate(prepared):
            values = {
                "title": "\n".join(article.title) if article.title else "",
                "abstract": "\n".join(article.abstract) if article.abstract else "",
                "keyword": "; ".join(article.keyword) if article.keyword else "",
            }
            for field_name, vector_attr, stage in field_plan:
                embedding_inputs.append(_metadata_embedding_text(values[field_name]))
                embedding_targets.append((prepared_index, vector_attr, stage))
        _record_metadata_bulk_timing("embedding_input_build", embedding_input_started_at)

        embedding_started_at = time.perf_counter()
        results = await self.embedder.embed_texts_detailed(embedding_inputs)
        _record_metadata_bulk_timing("embedding_api", embedding_started_at)
        failures_by_index: dict[int, tuple[SyncStage, EmbeddingItemResult]] = {}
        embedding_result_started_at = time.perf_counter()
        for (prepared_index, vector_attr, stage), embedding_result in zip(
            embedding_targets,
            results,
        ):
            if embedding_result.success and embedding_result.vector is not None:
                setattr(prepared[prepared_index][2], vector_attr, embedding_result.vector)
                continue
            if _is_skipped_empty_embedding(embedding_result):
                setattr(prepared[prepared_index][2], vector_attr, _empty_metadata_vector())
                continue
            failures_by_index.setdefault(prepared_index, (stage, embedding_result))

        ready: list[tuple[int, DocumentContext, Any]] = []
        for prepared_index, item in enumerate(prepared):
            failure = failures_by_index.get(prepared_index)
            if failure is None:
                ready.append(item)
                continue
            index, context, _ = item
            stage, embedding_result = failure
            outcome = await self._persist_stage_failure(
                context=context,
                source="kafka",
                stage=stage,
                scope_type=ScopeType.ARTICLE,
                scope_id=context.article_scope_id,
                error_type=embedding_result.error_type or ErrorType.NON_RETRYABLE,
                error_code=embedding_result.error_code or "EMBEDDING_FAILED",
                error_message=embedding_result.error_message or "embedding failed",
                exception_class="EmbeddingFailure",
                stacktrace=embedding_result.error_message or "embedding failed",
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                artifact_refs=[],
                retry_count_snapshot=0,
                overall_status=SyncOverallStatus.FAILURE,
                stage_started_at=self._utcnow(),
            )
            indexed_outcomes.append((index, outcome))
        _record_metadata_bulk_timing("embedding_result_map", embedding_result_started_at)

        write_batch_size = max(
            int(getattr(settings, "DOCUMENT_BULK_MILVUS_BATCH_SIZE", 512) or 512),
            1,
        )
        for offset in range(0, len(ready), write_batch_size):
            write_items = ready[offset : offset + write_batch_size]
            articles = [article for _, _, article in write_items]
            for article in articles:
                # This lane is final metadata ingestion, not a multi-stage task fan-out.
                article.sync_status = "SUCCESS"
            try:
                # Only one large Milvus write runs at a time.  The next batch
                # is free to use the embedding service while this write blocks.
                milvus_lock_wait_started_at = time.perf_counter()
                async with _metadata_bulk_write_lock():
                    _record_metadata_bulk_timing("milvus_write_lock_wait", milvus_lock_wait_started_at)
                    milvus_write_started_at = time.perf_counter()
                    await asyncio.to_thread(
                        self.milvus.process_and_insert,
                        self.document_collection_name,
                        articles,
                        settings.VECTOR_DIM,
                    )
                    _record_metadata_bulk_timing("milvus_write", milvus_write_started_at)
            except Exception as error:
                error_type, error_code = self._classify_milvus_error(error)
                for index, context, _ in write_items:
                    outcome = await self._persist_stage_failure(
                        context=context,
                        source="kafka",
                        stage=SyncStage.MILVUS_WRITE_DOCUMENT,
                        scope_type=ScopeType.DOCUMENT,
                        scope_id=context.doc_id,
                        error_type=error_type,
                        error_code=error_code,
                        error_message=str(error),
                        exception_class=type(error).__name__,
                        stacktrace=build_stacktrace(error),
                        replay_mode=ReplayMode.FROM_RAW_EVENT,
                        artifact_refs=[],
                        retry_count_snapshot=0,
                        overall_status=SyncOverallStatus.FAILURE,
                        stage_started_at=self._utcnow(),
                    )
                    indexed_outcomes.append((index, outcome))
                continue

            for index, context, _ in write_items:
                indexed_outcomes.append(
                    (
                        index,
                        ProcessOutcome(
                            doc_id=context.doc_id,
                            record_update_time=context.record_update_time,
                            final_status=SyncOverallStatus.SUCCESS.value,
                            commit_safe=True,
                            sub_doc_type=context.sub_doc_type,
                            event_type=context.event_type,
                            message="metadata_bulk_written",
                        ),
                    )
                )

        indexed_outcomes.sort(key=lambda item: item[0])
        return indexed_outcomes

    async def _process_contexts_for_batch(
        self,
        contexts: list[DocumentContext],
    ) -> list[tuple[int, ProcessOutcome]]:
        if not contexts:
            return []

        concurrency = max(int(getattr(settings, "DOCUMENT_PIPELINE_CONCURRENCY", 1) or 1), 1)
        if concurrency <= 1 or len(contexts) <= 1:
            return [
                (index, await self._process_context(context, source="kafka"))
                for index, context in enumerate(contexts)
            ]

        groups: dict[str, list[tuple[int, DocumentContext]]] = defaultdict(list)
        group_order: list[str] = []
        for index, context in enumerate(contexts):
            if context.doc_id not in groups:
                group_order.append(context.doc_id)
            groups[context.doc_id].append((index, context))

        semaphore = asyncio.Semaphore(concurrency)

        async def run_group(items: list[tuple[int, DocumentContext]]) -> list[tuple[int, ProcessOutcome]]:
            async with semaphore:
                group_results: list[tuple[int, ProcessOutcome]] = []
                for item_index, item_context in items:
                    outcome = await self._process_context(item_context, source="kafka")
                    group_results.append((item_index, outcome))
                return group_results

        grouped_results = await asyncio.gather(
            *(run_group(groups[doc_id]) for doc_id in group_order)
        )
        ordered_results = [item for group in grouped_results for item in group]
        ordered_results.sort(key=lambda item: item[0])
        return ordered_results

    async def retry_failure(self, failure_task: dict[str, Any]) -> ProcessOutcome:
        doc_id = failure_task.get("doc_id")
        record_update_time = failure_task.get("record_update_time")
        if not doc_id or not record_update_time:
            await self._mongo_call(
                "mark_failure_skipped_stale",
                self.manager.mark_failure_skipped_stale(
                    failure_task["failure_key"],
                    "missing doc_id or record_update_time for retry",
                ),
                {
                    "failure_key": failure_task.get("failure_key"),
                    "doc_id": doc_id,
                    "record_update_time": record_update_time,
                },
            )
            return ProcessOutcome(
                doc_id=doc_id,
                record_update_time=record_update_time,
                final_status=SyncOverallStatus.SKIPPED_STALE.value,
                commit_safe=True,
                message="retry_skipped_missing_identity",
            )

        status_doc = await self._mongo_call(
            "get_status",
            self.manager.get_status(doc_id),
            {"doc_id": doc_id},
        )
        stale_reason = self._check_retry_stale(status_doc, failure_task)
        if stale_reason is not None:
            await self._mongo_call(
                "mark_failure_skipped_stale",
                self.manager.mark_failure_skipped_stale(
                    failure_task["failure_key"],
                    stale_reason,
                ),
                {
                    "failure_key": failure_task["failure_key"],
                    "doc_id": doc_id,
                    "record_update_time": record_update_time,
                },
            )
            return ProcessOutcome(
                doc_id=doc_id,
                record_update_time=record_update_time,
                final_status=SyncOverallStatus.SKIPPED_STALE.value,
                commit_safe=True,
                sub_doc_type=failure_task.get("sub_doc_type"),
                event_type=failure_task.get("event_type"),
                message=stale_reason,
            )

        context = await self._rebuild_context_from_failure(failure_task)
        if context is None:
            await self._mongo_call(
                "mark_failure_skipped_stale",
                self.manager.mark_failure_skipped_stale(
                    failure_task["failure_key"],
                    "retry_artifact_missing_or_invalid",
                ),
                {
                    "failure_key": failure_task["failure_key"],
                    "doc_id": doc_id,
                    "record_update_time": record_update_time,
                },
            )
            return ProcessOutcome(
                doc_id=doc_id,
                record_update_time=record_update_time,
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=True,
                sub_doc_type=failure_task.get("sub_doc_type"),
                event_type=failure_task.get("event_type"),
                pending_failure_refs=[failure_task["failure_key"]],
                failure_persisted=True,
                message="retry_artifact_missing_or_invalid",
            )

        outcome = await self._process_context(
            context,
            source="retry",
            retry_count_snapshot=int(failure_task.get("retry_count", 0)),
        )
        if (
            outcome.commit_safe
            and outcome.final_status in {
                SyncOverallStatus.FAILURE.value,
                SyncOverallStatus.PARTIAL_FAILURE.value,
            }
            and failure_task["failure_key"] not in outcome.pending_failure_refs
        ):
            await self._mongo_call(
                "mark_failure_done_superseded",
                self.manager.mark_failure_done(
                    failure_task["failure_key"],
                    "superseded_by_new_failure_stage",
                ),
                {
                    "failure_key": failure_task["failure_key"],
                    "doc_id": doc_id,
                    "record_update_time": record_update_time,
                },
            )
        return outcome

    async def _process_invalid_record(self, record: ConsumedRecord) -> ProcessOutcome:
        stage = SyncStage.KAFKA_DECODE if record.decode_error else SyncStage.KAFKA_VALIDATE
        error_code = "KAFKA_JSON_DECODE_ERROR" if record.decode_error else "KAFKA_SCHEMA_VALIDATE_ERROR"
        error_message = record.decode_error or record.validation_error or "unknown kafka error"

        try:
            artifact_refs: list[str] = []
            raw_meta_ref = await self._mongo_call(
                "save_artifact_invalid_raw",
                self.manager.save_artifact(
                    doc_id=None,
                    record_update_time=None,
                    sub_doc_type=None,
                    stage=stage,
                    scope_type=ScopeType.EVENT,
                    scope_id=record.event_fingerprint,
                    artifact_type=ArtifactType.RAW_EVENT_BYTES_META,
                    content={
                        "topic": record.topic,
                        "partition": record.partition,
                        "offset": record.offset,
                        "event_fingerprint": record.event_fingerprint,
                        "size_bytes": len(record.raw_bytes),
                        "raw_text_preview": record.raw_text[:4000],
                    },
                    success_path=False,
                    content_type="application/json",
                ),
                {"event_fingerprint": record.event_fingerprint},
            )
            artifact_refs.append(raw_meta_ref)

            if record.parsed_payload is not None:
                raw_event_ref = await self._mongo_call(
                    "save_artifact_invalid_json",
                    self.manager.save_artifact(
                        doc_id=None,
                        record_update_time=None,
                        sub_doc_type=None,
                        stage=stage,
                        scope_type=ScopeType.EVENT,
                        scope_id=record.event_fingerprint,
                        artifact_type=ArtifactType.RAW_EVENT_JSON,
                        content=record.parsed_payload,
                        success_path=False,
                        content_type="application/json",
                    ),
                    {"event_fingerprint": record.event_fingerprint},
                )
                artifact_refs.append(raw_event_ref)

            error_ref = await self._save_error_context(
                doc_id=None,
                record_update_time=None,
                sub_doc_type=None,
                stage=stage,
                scope_type=ScopeType.EVENT,
                scope_id=record.event_fingerprint,
                error_type=ErrorType.NON_RETRYABLE,
                error_code=error_code,
                error_message=error_message,
                extra={
                    "topic": record.topic,
                    "partition": record.partition,
                    "offset": record.offset,
                    "event_fingerprint": record.event_fingerprint,
                },
            )
            artifact_refs.append(error_ref)

            failure_ref = await self._mongo_call(
                "persist_failure_invalid",
                self.manager.persist_failure(
                    doc_id=None,
                    record_update_time=None,
                    sub_doc_type=None,
                    event_type=None,
                    failure_stage=stage,
                    scope_type=ScopeType.EVENT,
                    scope_id=record.event_fingerprint,
                    error_type=ErrorType.NON_RETRYABLE,
                    error_code=error_code,
                    error_message=error_message,
                    exception_class="KafkaRecordError",
                    stacktrace=error_message,
                    replay_mode=ReplayMode.FROM_RAW_EVENT,
                    artifact_refs=artifact_refs,
                    source_event_meta=record.source_event_meta,
                ),
                {"event_fingerprint": record.event_fingerprint, "error_code": error_code},
            )

            await self._best_effort_attempt(
                doc_id=None,
                record_update_time=None,
                sub_doc_type=None,
                event_type=None,
                source="kafka",
                stage=stage,
                stage_status=StageExecutionStatus.FAILURE.value,
                artifact_ref=error_ref,
                error_summary={"error_code": error_code, "error_message": error_message},
                source_event_meta=record.source_event_meta,
            )
            return ProcessOutcome(
                doc_id=None,
                record_update_time=None,
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=True,
                pending_failure_refs=[failure_ref.failure_key],
                failure_persisted=True,
                message=error_code,
            )
        except MongoPersistenceError as error:
            logger.error(
                "invalid_record_mongo_persist_failed",
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                event_fingerprint=record.event_fingerprint,
                error=str(error),
            )
            return ProcessOutcome(
                doc_id=None,
                record_update_time=None,
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=False,
                message=error_code,
            )

    async def _expand_record(
        self,
        record: ConsumedRecord,
    ) -> tuple[list[DocumentContext], list[ProcessOutcome]]:
        payload = record.parsed_payload or {}
        data = payload.get("data") or []
        event_type = infer_data_status(payload)
        sub_doc_type = str(payload.get("subDocType") or "")
        contexts: list[DocumentContext] = []
        outcomes: list[ProcessOutcome] = []

        if not isinstance(data, list) or not data:
            failure_outcome = await self._persist_event_validation_failure(
                record=record,
                stage=SyncStage.KAFKA_VALIDATE,
                error_code="EMPTY_DATA_LIST",
                error_message="Kafka payload does not contain any data item",
                scope_id=record.event_fingerprint,
                doc_id=None,
                record_update_time=None,
                sub_doc_type=sub_doc_type or None,
            )
            outcomes.append(failure_outcome)
            return contexts, outcomes

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                outcomes.append(
                    await self._persist_event_validation_failure(
                        record=record,
                        stage=SyncStage.KAFKA_VALIDATE,
                        error_code="DATA_ITEM_NOT_OBJECT",
                        error_message=f"Kafka data[{index}] is not an object",
                        scope_id=f"{record.event_fingerprint}:{index}",
                        doc_id=None,
                        record_update_time=None,
                        sub_doc_type=sub_doc_type or None,
                    )
                )
                continue

            doc_id = item.get("id") or item.get("sysId") or extract_item_value(item, "sysId")
            record_update_time = normalize_record_update_time(
                extract_item_value(item, "sysDataUpdateTime")
            )
            item_sub_doc_type = str(
                extract_item_value(item, "sysSubType") or payload.get("subDocType") or ""
            )
            if not doc_id:
                outcomes.append(
                    await self._persist_event_validation_failure(
                        record=record,
                        stage=SyncStage.KAFKA_VALIDATE,
                        error_code="MISSING_SYS_ID",
                        error_message=f"Kafka data[{index}] missing sysId/id",
                        scope_id=f"{record.event_fingerprint}:{index}",
                        doc_id=None,
                        record_update_time=record_update_time or None,
                        sub_doc_type=item_sub_doc_type or None,
                    )
                )
                continue

            if not record_update_time:
                outcomes.append(
                    await self._persist_event_validation_failure(
                        record=record,
                        stage=SyncStage.KAFKA_VALIDATE,
                        error_code="MISSING_RECORD_UPDATE_TIME",
                        error_message=f"Kafka data[{index}] missing sysDataUpdateTime",
                        scope_id=str(doc_id),
                        doc_id=str(doc_id),
                        record_update_time=None,
                        sub_doc_type=item_sub_doc_type or None,
                    )
                )
                continue

            contexts.append(
                DocumentContext(
                    doc_id=str(doc_id),
                    record_update_time=record_update_time,
                    sub_doc_type=item_sub_doc_type,
                    event_type=event_type,
                    payload={
                        **payload,
                        "data": [item],
                    },
                    article_raw=item,
                    article_index=index,
                    source_event_meta=record.source_event_meta,
                    event_fingerprint=record.event_fingerprint,
                )
            )

        return contexts, outcomes

    async def _process_context(
        self,
        context: DocumentContext,
        *,
        source: str,
        retry_count_snapshot: int = 0,
    ) -> ProcessOutcome:
        try:
            # A fresh Kafka document needs no Mongo lookup in failure-only mode.
            # Retry processing still reads failure state so stale retries remain safe.
            existing_status = None
            if self.success_audit_enabled or source == "retry":
                existing_status = await self._mongo_call(
                    "get_status",
                    self.manager.get_status(context.doc_id),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                    },
                )
            skip_reason = self._check_skip_reason(existing_status, context, source=source)
            if skip_reason is not None:
                if (
                    existing_status
                    and existing_status.get("overall_status") == SyncOverallStatus.PROCESSING.value
                    and existing_status.get("last_success_stage")
                    in {
                        SyncStage.MILVUS_WRITE_DOCUMENT.value,
                        SyncStage.DELETE_DOCUMENT.value,
                    }
                    and not (existing_status.get("retry_summary") or {}).get("has_pending_failure")
                ):
                    await self._mongo_call(
                        "finalize_document_status_duplicate",
                        self.manager.finalize_document_status(
                            doc_id=context.doc_id,
                            overall_status=(
                                SyncOverallStatus.DELETED
                                if context.event_type == "DELETE"
                                else SyncOverallStatus.SUCCESS
                            ),
                            current_stage=(
                                SyncStage.DELETE_DOCUMENT
                                if context.event_type == "DELETE"
                                else SyncStage.MILVUS_WRITE_DOCUMENT
                            ),
                            last_success_stage=(
                                SyncStage.DELETE_DOCUMENT
                                if context.event_type == "DELETE"
                                else SyncStage.MILVUS_WRITE_DOCUMENT
                            ),
                        ),
                        {
                            "doc_id": context.doc_id,
                            "record_update_time": context.record_update_time,
                        },
                    )

                return ProcessOutcome(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    final_status=SyncOverallStatus.SKIPPED_STALE.value,
                    commit_safe=True,
                    sub_doc_type=context.sub_doc_type,
                    event_type=context.event_type,
                    message=skip_reason,
                )

            await self._mongo_call(
                "initialize_status",
                self.manager.initialize_status(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    sub_doc_type=context.sub_doc_type,
                    event_type=context.event_type,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "event_type": context.event_type,
                },
            )

            raw_event_ref = await self._mongo_call(
                "save_raw_event_artifact",
                self.manager.save_artifact(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    sub_doc_type=context.sub_doc_type,
                    stage=SyncStage.KAFKA_VALIDATE,
                    scope_type=ScopeType.DOCUMENT,
                    scope_id=context.doc_id,
                    artifact_type=ArtifactType.RAW_EVENT_JSON,
                    content=context.payload,
                    success_path=True,
                    content_type="application/json",
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                },
            )

            await self._record_stage_success(
                context=context,
                stage=SyncStage.KAFKA_VALIDATE,
                artifact_ref=raw_event_ref,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
            )
            await self._record_stage_success(
                context=context,
                stage=SyncStage.IDEMPOTENCY_CHECK,
                artifact_ref=raw_event_ref,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
            )

            if context.event_type == "DELETE":
                return await self._process_delete_context(
                    context,
                    raw_event_ref=raw_event_ref,
                    source=source,
                    retry_count_snapshot=retry_count_snapshot,
                )

            return await self._process_upsert_context(
                context,
                raw_event_ref=raw_event_ref,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
            )
        except MongoPersistenceError as error:
            logger.error(
                "document_process_mongo_failed",
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                source=source,
                error=str(error),
            )
            return ProcessOutcome(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=False,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                message="mongo_persist_failed",
            )

    async def _process_delete_context(
        self,
        context: DocumentContext,
        *,
        raw_event_ref: str,
        source: str,
        retry_count_snapshot: int,
    ) -> ProcessOutcome:
        stage_started_at = self._utcnow()
        await self._mongo_call(
            "mark_stage_started_delete",
            self.manager.mark_stage_started(doc_id=context.doc_id, stage=SyncStage.DELETE_DOCUMENT),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.DELETE_DOCUMENT.value,
            },
        )
        if settings.LIVE_E2E_NO_DELETE:
            delete_result = _live_e2e_no_delete_result(
                operation="delete_document",
                collection_name=self._resolved_document_collection_name(),
                doc_id=context.doc_id,
            )
        else:
            try:
                delete_result = await asyncio.to_thread(
                    self.milvus.delete_by_sys_id,
                    self.document_collection_name,
                    context.doc_id,
                )
            except Exception as error:
                return await self._persist_stage_failure(
                    context=context,
                    source=source,
                    stage=SyncStage.DELETE_DOCUMENT,
                    scope_type=ScopeType.DOCUMENT,
                    scope_id=context.doc_id,
                    error_type=ErrorType.RETRYABLE,
                    error_code="MILVUS_DELETE_ERROR",
                    error_message=str(error),
                    exception_class=type(error).__name__,
                    stacktrace=build_stacktrace(error),
                    replay_mode=ReplayMode.FROM_DELETE_CONTEXT,
                    artifact_refs=[raw_event_ref],
                    retry_count_snapshot=retry_count_snapshot,
                    overall_status=SyncOverallStatus.FAILURE,
                    stage_started_at=stage_started_at,
                )

        write_ref = await self._mongo_call(
            "save_delete_artifact",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.DELETE_DOCUMENT,
                scope_type=ScopeType.DOCUMENT,
                scope_id=context.doc_id,
                artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
                content=delete_result,
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.DELETE_DOCUMENT.value,
            },
        )
        await self._record_stage_success(
            context=context,
            stage=SyncStage.DELETE_DOCUMENT,
            artifact_ref=write_ref,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            started_at=stage_started_at,
        )
        return await self._finalize_after_document_delete(
            context=context,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            artifact_refs=[raw_event_ref, write_ref],
        )

    async def _finalize_after_document_delete(
        self,
        *,
        context: DocumentContext,
        source: str,
        retry_count_snapshot: int,
        artifact_refs: list[str],
    ) -> ProcessOutcome:
        cleanup_artifact_refs = list(artifact_refs)
        last_success_stage = SyncStage.DELETE_DOCUMENT
        progress: dict[str, Any] = {
            "article_count": 1,
            "document_vectors_written": 0,
        }
        if settings.LIVE_E2E_NO_DELETE:
            progress["delete_noop_recorded"] = True

        if settings.FULLTEXT_DELETE_CHUNKS_ENABLED and not settings.LIVE_E2E_NO_DELETE:
            delete_chunks_outcome, chunks_ref = await self._handle_delete_chunks_after_document_delete(
                context=context,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
                artifact_refs=cleanup_artifact_refs,
                last_success_stage=last_success_stage,
                progress=progress,
            )
            if delete_chunks_outcome is not None:
                return delete_chunks_outcome
            if chunks_ref is not None:
                cleanup_artifact_refs.append(chunks_ref)
                last_success_stage = SyncStage.DELETE_CHUNKS
                progress["chunk_vectors_deleted"] = 1

        if settings.IMAGE_DELETE_IMAGES_ENABLED and not settings.LIVE_E2E_NO_DELETE:
            delete_images_outcome, images_ref = await self._handle_delete_images_after_document_delete(
                context=context,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
                artifact_refs=cleanup_artifact_refs,
                last_success_stage=last_success_stage,
                progress=progress,
            )
            if delete_images_outcome is not None:
                return delete_images_outcome
            if images_ref is not None:
                cleanup_artifact_refs.append(images_ref)
                last_success_stage = SyncStage.DELETE_IMAGES
                progress["image_vectors_deleted"] = 1

        await self._mongo_call(
            "finalize_delete_status",
            self.manager.finalize_document_status(
                doc_id=context.doc_id,
                overall_status=SyncOverallStatus.DELETED,
                current_stage=last_success_stage,
                last_success_stage=last_success_stage,
                progress=progress,
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "overall_status": SyncOverallStatus.DELETED.value,
            },
        )
        if self.success_audit_enabled or source == "retry":
            await self._mongo_call(
                "resolve_delete_failures",
                self.manager.resolve_failures_for_document(
                    context.doc_id,
                    context.record_update_time,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                },
            )
        return ProcessOutcome(
            doc_id=context.doc_id,
            record_update_time=context.record_update_time,
            final_status=SyncOverallStatus.DELETED.value,
            commit_safe=True,
            sub_doc_type=context.sub_doc_type,
            event_type=context.event_type,
        )

    async def _handle_delete_chunks_after_document_delete(
        self,
        *,
        context: DocumentContext,
        source: str,
        retry_count_snapshot: int,
        artifact_refs: list[str],
        last_success_stage: SyncStage,
        progress: dict[str, Any],
    ) -> tuple[Optional[ProcessOutcome], Optional[str]]:
        stage_started_at = self._utcnow()
        chunk_writer: Optional[MilvusChunkService] = None
        try:
            await self._mongo_call(
                "mark_stage_started_delete_chunks",
                self.manager.mark_stage_started(
                    doc_id=context.doc_id,
                    stage=SyncStage.DELETE_CHUNKS,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": SyncStage.DELETE_CHUNKS.value,
                },
            )
            chunk_writer = MilvusChunkService()
            delete_summary = delete_chunks_for_document(
                context.doc_id,
                chunk_writer=chunk_writer,
            )
        except Exception as error:
            failure_outcome = await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.DELETE_CHUNKS,
                scope_type=ScopeType.CHUNK,
                scope_id=context.doc_id,
                error_type=ErrorType.RETRYABLE,
                error_code="FULLTEXT_DELETE_CHUNKS_ERROR",
                error_message=str(error),
                exception_class=type(error).__name__,
                stacktrace=build_stacktrace(error),
                replay_mode=ReplayMode.FROM_DELETE_CONTEXT,
                artifact_refs=artifact_refs,
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.PARTIAL_FAILURE,
                stage_started_at=stage_started_at,
            )
            if failure_outcome.commit_safe:
                failure_progress = {**progress, "chunk_vectors_deleted": 0}
                await self._mongo_call(
                    "finalize_delete_chunks_failure_status",
                    self.manager.finalize_document_status(
                        doc_id=context.doc_id,
                        overall_status=SyncOverallStatus.PARTIAL_FAILURE,
                        current_stage=SyncStage.DELETE_CHUNKS,
                        last_success_stage=last_success_stage,
                        progress=failure_progress,
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "overall_status": SyncOverallStatus.PARTIAL_FAILURE.value,
                    },
                )
            return failure_outcome, None
        finally:
            if chunk_writer is not None:
                try:
                    chunk_writer.close()
                except Exception as close_error:
                    logger.warning(
                        "milvus_chunk_service_close_failed",
                        doc_id=context.doc_id,
                        error=str(close_error),
                    )

        chunks_ref = await self._mongo_call(
            "save_delete_chunks_artifact",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.DELETE_CHUNKS,
                scope_type=ScopeType.CHUNK,
                scope_id=context.doc_id,
                artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
                content=delete_summary,
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.DELETE_CHUNKS.value,
            },
        )
        await self._record_stage_success(
            context=context,
            stage=SyncStage.DELETE_CHUNKS,
            artifact_ref=chunks_ref,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            started_at=stage_started_at,
        )
        return None, chunks_ref

    async def _handle_delete_images_after_document_delete(
        self,
        *,
        context: DocumentContext,
        source: str,
        retry_count_snapshot: int,
        artifact_refs: list[str],
        last_success_stage: SyncStage,
        progress: dict[str, Any],
    ) -> tuple[Optional[ProcessOutcome], Optional[str]]:
        stage_started_at = self._utcnow()
        await self._mongo_call(
            "mark_stage_started_delete_images",
            self.manager.mark_stage_started(
                doc_id=context.doc_id,
                stage=SyncStage.DELETE_IMAGES,
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.DELETE_IMAGES.value,
            },
        )
        delete_summary = delete_images_for_document(context.doc_id)
        if not delete_summary.get("success"):
            error_code = str(delete_summary.get("error_code") or "IMAGE_DELETE_IMAGES_ERROR")
            error_message = str(delete_summary.get("error_message") or "image delete failed")
            retryable = bool(delete_summary.get("retryable", True))
            error = RuntimeError(error_message)
            failure_outcome = await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.DELETE_IMAGES,
                scope_type=ScopeType.IMAGE,
                scope_id=context.doc_id,
                error_type=ErrorType.RETRYABLE if retryable else ErrorType.NON_RETRYABLE,
                error_code=error_code,
                error_message=error_message,
                exception_class=type(error).__name__,
                stacktrace=build_stacktrace(error),
                replay_mode=ReplayMode.FROM_DELETE_CONTEXT,
                artifact_refs=artifact_refs,
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.PARTIAL_FAILURE,
                stage_started_at=stage_started_at,
            )
            if failure_outcome.commit_safe:
                failure_progress = {**progress, "image_vectors_deleted": 0}
                await self._mongo_call(
                    "finalize_delete_images_failure_status",
                    self.manager.finalize_document_status(
                        doc_id=context.doc_id,
                        overall_status=SyncOverallStatus.PARTIAL_FAILURE,
                        current_stage=SyncStage.DELETE_IMAGES,
                        last_success_stage=last_success_stage,
                        progress=failure_progress,
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "overall_status": SyncOverallStatus.PARTIAL_FAILURE.value,
                    },
                )
            return failure_outcome, None

        images_ref = await self._mongo_call(
            "save_delete_images_artifact",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.DELETE_IMAGES,
                scope_type=ScopeType.IMAGE,
                scope_id=context.doc_id,
                artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
                content=delete_summary,
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.DELETE_IMAGES.value,
            },
        )
        await self._record_stage_success(
            context=context,
            stage=SyncStage.DELETE_IMAGES,
            artifact_ref=images_ref,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            started_at=stage_started_at,
        )
        return None, images_ref

    async def _process_upsert_context(
        self,
        context: DocumentContext,
        *,
        raw_event_ref: str,
        source: str,
        retry_count_snapshot: int,
    ) -> ProcessOutcome:
        normalize_started_at = self._utcnow()
        await self._mongo_call(
            "mark_stage_started_normalize",
            self.manager.mark_stage_started(doc_id=context.doc_id, stage=SyncStage.ARTICLE_NORMALIZE),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.ARTICLE_NORMALIZE.value,
            },
        )
        try:
            document = map_to_document(Document, context.payload)
        except Exception as error:
            error_type, error_code = self._classify_processing_error(error, default_retryable=False, default_code="ARTICLE_NORMALIZE_ERROR")
            return await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.ARTICLE_NORMALIZE,
                scope_type=ScopeType.ARTICLE,
                scope_id=context.article_scope_id,
                error_type=error_type,
                error_code=error_code,
                error_message=str(error),
                exception_class=type(error).__name__,
                stacktrace=build_stacktrace(error),
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                artifact_refs=[raw_event_ref],
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.FAILURE,
                stage_started_at=normalize_started_at,
            )

        articles = document.data or []
        if not articles:
            return await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.ARTICLE_NORMALIZE,
                scope_type=ScopeType.ARTICLE,
                scope_id=context.article_scope_id,
                error_type=ErrorType.NON_RETRYABLE,
                error_code="EMPTY_ARTICLES",
                error_message="No parsed articles after normalization",
                exception_class="ValueError",
                stacktrace="No parsed articles after normalization",
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                artifact_refs=[raw_event_ref],
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.FAILURE,
                stage_started_at=normalize_started_at,
            )

        article = articles[0]
        if not article.sysId:
            article.sysId = context.doc_id

        parsed_document_ref = await self._mongo_call(
            "save_parsed_document",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.ARTICLE_NORMALIZE,
                scope_type=ScopeType.DOCUMENT,
                scope_id=context.doc_id,
                artifact_type=ArtifactType.PARSED_DOCUMENT_JSON,
                content=document.model_dump(mode="json"),
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.ARTICLE_NORMALIZE.value,
            },
        )
        parsed_article_ref = await self._mongo_call(
            "save_parsed_article",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.ARTICLE_NORMALIZE,
                scope_type=ScopeType.ARTICLE,
                scope_id=context.article_scope_id,
                artifact_type=ArtifactType.PARSED_ARTICLE_JSON,
                content=article.model_dump(mode="json"),
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.ARTICLE_NORMALIZE.value,
            },
        )
        await self._record_stage_success(
            context=context,
            stage=SyncStage.ARTICLE_NORMALIZE,
            artifact_ref=parsed_article_ref,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            started_at=normalize_started_at,
        )

        embed_inputs = {
            "title": "\n".join(article.title) if article.title else "",
            "abstract": "\n".join(article.abstract) if article.abstract else "",
            "keyword": "; ".join(article.keyword) if article.keyword else "",
        }
        embed_input_ref = await self._mongo_call(
            "save_embed_input",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.DOC_EMBED_TITLE,
                scope_type=ScopeType.ARTICLE,
                scope_id=context.article_scope_id,
                artifact_type=ArtifactType.EMBED_INPUT_JSON,
                content=embed_inputs,
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
            },
        )

        field_plan = [
            ("title", "title_embed", SyncStage.DOC_EMBED_TITLE),
            ("abstract", "abstract_embed", SyncStage.DOC_EMBED_ABSTRACT),
            ("keyword", "keyword_embed", SyncStage.DOC_EMBED_KEYWORD),
        ]

        stage_start_times: dict[SyncStage, datetime] = {}
        for _, _, stage in field_plan:
            stage_start_times[stage] = self._utcnow()
            await self._mongo_call(
                f"mark_stage_started_{stage.value}",
                self.manager.mark_stage_started(doc_id=context.doc_id, stage=stage),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": stage.value,
                },
            )

        results = await self.embedder.embed_texts_detailed([embed_inputs[key] for key, _, _ in field_plan])
        failures: list[tuple[str, SyncStage, EmbeddingItemResult]] = []
        succeeded_stage: Optional[SyncStage] = SyncStage.ARTICLE_NORMALIZE

        for (field_name, vector_attr, stage), result in zip(field_plan, results):
            if result.success and result.vector is not None:
                setattr(article, vector_attr, result.vector)
                await self._record_stage_success(
                    context=context,
                    stage=stage,
                    artifact_ref=embed_input_ref,
                    source=source,
                    retry_count_snapshot=retry_count_snapshot,
                    started_at=stage_start_times[stage],
                    skipped_empty=result.skipped_empty,
                )
                succeeded_stage = stage
                continue
            if _is_skipped_empty_embedding(result):
                setattr(article, vector_attr, _empty_metadata_vector())
                await self._record_stage_success(
                    context=context,
                    stage=stage,
                    artifact_ref=embed_input_ref,
                    source=source,
                    retry_count_snapshot=retry_count_snapshot,
                    started_at=stage_start_times[stage],
                    skipped_empty=True,
                )
                succeeded_stage = stage
                continue

            failures.append((field_name, stage, result))

        if failures:
            final_status = (
                SyncOverallStatus.PARTIAL_FAILURE
                if len(failures) < len(field_plan)
                else SyncOverallStatus.FAILURE
            )
            pending_failure_refs: list[str] = []
            for field_name, stage, result in failures:
                failure_outcome = await self._persist_stage_failure(
                    context=context,
                    source=source,
                    stage=stage,
                    scope_type=ScopeType.ARTICLE,
                    scope_id=context.article_scope_id,
                    error_type=result.error_type or ErrorType.NON_RETRYABLE,
                    error_code=result.error_code or f"{field_name.upper()}_EMBED_FAILED",
                    error_message=result.error_message or f"{field_name} embedding failed",
                    exception_class="EmbeddingFailure",
                    stacktrace=result.error_message or f"{field_name} embedding failed",
                    replay_mode=ReplayMode.FROM_PARSED_DOCUMENT,
                    artifact_refs=[
                        raw_event_ref,
                        parsed_document_ref,
                        parsed_article_ref,
                        embed_input_ref,
                    ],
                    retry_count_snapshot=retry_count_snapshot,
                    overall_status=final_status,
                    stage_started_at=stage_start_times[stage],
                    raise_on_mongo_failure=True,
                )
                pending_failure_refs.extend(failure_outcome.pending_failure_refs)

            if succeeded_stage is None:
                succeeded_stage = SyncStage.ARTICLE_NORMALIZE

            await self._mongo_call(
                "finalize_partial_failure_status",
                self.manager.finalize_document_status(
                    doc_id=context.doc_id,
                    overall_status=final_status,
                    current_stage=SyncStage(failures[-1][1].value),
                    last_success_stage=succeeded_stage,
                    progress={"article_count": len(articles)},
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "overall_status": final_status.value,
                },
            )
            return ProcessOutcome(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                final_status=final_status.value,
                commit_safe=True,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                pending_failure_refs=pending_failure_refs,
                failure_persisted=True,
                message="embedding_failure_persisted",
            )

        milvus_stage_started_at = self._utcnow()
        await self._mongo_call(
            "mark_stage_started_milvus",
            self.manager.mark_stage_started(doc_id=context.doc_id, stage=SyncStage.MILVUS_WRITE_DOCUMENT),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": SyncStage.MILVUS_WRITE_DOCUMENT.value,
            },
        )
        # Set initial sync_status for Milvus completeness tracking.
        # The Compensator will reconcile to terminal statuses (SUCCESS / CORE_SUCCESS / PARTIAL_FAILURE).
        if hasattr(article, "sync_status"):
            article.sync_status = "PROCESSING"
        else:
            setattr(article, "sync_status", "PROCESSING")

        try:
            insert_result = await asyncio.to_thread(
                self.milvus.process_and_insert,
                self.document_collection_name,
                [article],
                settings.VECTOR_DIM,
            )
        except Exception as error:
            error_type, error_code = self._classify_milvus_error(error)
            return await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.MILVUS_WRITE_DOCUMENT,
                scope_type=ScopeType.DOCUMENT,
                scope_id=context.doc_id,
                error_type=error_type,
                error_code=error_code,
                error_message=str(error),
                exception_class=type(error).__name__,
                stacktrace=build_stacktrace(error),
                replay_mode=ReplayMode.FROM_PARSED_DOCUMENT,
                artifact_refs=[
                    raw_event_ref,
                    parsed_document_ref,
                    parsed_article_ref,
                    embed_input_ref,
                ],
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.FAILURE,
                stage_started_at=milvus_stage_started_at,
            )

        milvus_summary = {
            "collection_name": self._resolved_document_collection_name(),
            "sys_id": context.doc_id,
            "write_mode": "primary_key_upsert",
            "insert_result": insert_result,
        }
        milvus_ref = await self._mongo_call(
            "save_milvus_summary",
            self.manager.save_artifact(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=SyncStage.MILVUS_WRITE_DOCUMENT,
                scope_type=ScopeType.DOCUMENT,
                scope_id=context.doc_id,
                artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
                content=milvus_summary,
                success_path=True,
                content_type="application/json",
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
            },
        )
        await self._record_stage_success(
            context=context,
            stage=SyncStage.MILVUS_WRITE_DOCUMENT,
            artifact_ref=milvus_ref,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            started_at=milvus_stage_started_at,
        )
        await self._mongo_call(
            "finalize_success_status",
            self.manager.finalize_document_status(
                doc_id=context.doc_id,
                overall_status=SyncOverallStatus.SUCCESS,
                current_stage=SyncStage.MILVUS_WRITE_DOCUMENT,
                last_success_stage=SyncStage.MILVUS_WRITE_DOCUMENT,
                progress={
                    "article_count": len(articles),
                    "document_vectors_written": 1,
                },
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "overall_status": SyncOverallStatus.SUCCESS.value,
            },
        )
        initial_fulltext_status = (
            PIPELINE_STATUS_PENDING
            if settings.FULLTEXT_TASK_UPSERT_ENABLED
            else PIPELINE_STATUS_NOT_PRESENT
        )
        initial_image_status = (
            PIPELINE_STATUS_PENDING
            if settings.IMAGE_TASK_UPSERT_ENABLED
            else PIPELINE_STATUS_NOT_PRESENT
        )
        initial_core_status = (
            PIPELINE_STATUS_PROCESSING
            if initial_fulltext_status == PIPELINE_STATUS_PENDING
            else PIPELINE_STATUS_SUCCESS
        )
        initial_overall_status = (
            SyncOverallStatus.PROCESSING
            if initial_core_status == PIPELINE_STATUS_PROCESSING
            else (
                SyncOverallStatus.CORE_SUCCESS
                if initial_image_status == PIPELINE_STATUS_PENDING
                else SyncOverallStatus.SUCCESS
            )
        )
        await self._mongo_call(
            "update_pipeline_status_document_success",
            self.manager.update_pipeline_status(
                doc_id=context.doc_id,
                document_status=PIPELINE_STATUS_SUCCESS,
                fulltext_status=initial_fulltext_status,
                image_status=initial_image_status,
                core_status=initial_core_status,
                overall_status=initial_overall_status,
                fulltext_required=(
                    initial_fulltext_status == PIPELINE_STATUS_PENDING
                ),
                image_required=(initial_image_status == PIPELINE_STATUS_PENDING),
                progress={
                    "document_vectors_written": 1,
                },
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "core_status": initial_core_status,
                "image_status": initial_image_status,
            },
        )
        if self.success_audit_enabled or source == "retry":
            await self._mongo_call(
                "resolve_success_failures",
                self.manager.resolve_failures_for_document(
                    context.doc_id,
                    context.record_update_time,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                },
            )
        fulltext_outcome = await self._handle_fulltext_discover_after_document_success(
            context=context,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            artifact_refs=[
                raw_event_ref,
                parsed_document_ref,
                parsed_article_ref,
                embed_input_ref,
                milvus_ref,
            ],
        )
        final_outcome = ProcessOutcome(
            doc_id=context.doc_id,
            record_update_time=context.record_update_time,
            final_status=SyncOverallStatus.SUCCESS.value,
            commit_safe=True,
            sub_doc_type=context.sub_doc_type,
            event_type=context.event_type,
        )
        if fulltext_outcome is not None:
            final_outcome = self._merge_post_success_outcomes(
                final_outcome,
                fulltext_outcome,
            )
            if not final_outcome.commit_safe:
                return final_outcome

        image_outcome = await self._handle_image_discover_after_document_success(
            context=context,
            source=source,
            retry_count_snapshot=retry_count_snapshot,
            artifact_refs=[
                raw_event_ref,
                parsed_document_ref,
                parsed_article_ref,
                embed_input_ref,
                milvus_ref,
            ],
        )
        if image_outcome is not None:
            final_outcome = self._merge_post_success_outcomes(
                final_outcome,
                image_outcome,
            )
        return final_outcome

    async def _handle_fulltext_discover_after_document_success(
        self,
        *,
        context: DocumentContext,
        source: str,
        retry_count_snapshot: int,
        artifact_refs: list[str],
    ) -> Optional[ProcessOutcome]:
        if self.mode == "metadata_only":
            return None
        if not settings.FULLTEXT_TASK_UPSERT_ENABLED or context.event_type == "DELETE":
            return None

        stage_started_at = self._utcnow()
        failure_artifact_refs = list(artifact_refs)
        try:
            await self._mongo_call(
                "mark_stage_started_fulltext_discover",
                self.manager.mark_stage_started(
                    doc_id=context.doc_id,
                    stage=SyncStage.FULLTEXT_DISCOVER,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": SyncStage.FULLTEXT_DISCOVER.value,
                },
            )
            tasks = extract_fulltext_tasks(context.payload)
            if not tasks:
                await self._record_stage_terminal_status(
                    context=context,
                    stage=SyncStage.FULLTEXT_DISCOVER,
                    status=StageExecutionStatus.SKIPPED_NOT_PRESENT,
                    source=source,
                    retry_count_snapshot=retry_count_snapshot,
                    started_at=stage_started_at,
                )
                await self._mongo_call(
                    "update_pipeline_status_fulltext_not_present",
                    self.manager.update_pipeline_status(
                        doc_id=context.doc_id,
                        fulltext_status=PIPELINE_STATUS_NOT_PRESENT,
                        core_status=PIPELINE_STATUS_SUCCESS,
                        overall_status=(
                            SyncOverallStatus.CORE_SUCCESS
                            if settings.IMAGE_TASK_UPSERT_ENABLED
                            else SyncOverallStatus.SUCCESS
                        ),
                        fulltext_required=False,
                        progress={
                            "fulltext_task_count": 0,
                            "fulltext_task_success": 0,
                            "fulltext_task_failure": 0,
                        },
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "fulltext_status": PIPELINE_STATUS_NOT_PRESENT,
                    },
                )
                return ProcessOutcome(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    final_status=SyncOverallStatus.SUCCESS.value,
                    commit_safe=True,
                    sub_doc_type=context.sub_doc_type,
                    event_type=context.event_type,
                    message="fulltext_not_present",
                )

            upsert_result = await get_fulltext_task_repository().upsert_tasks(tasks)
            meta_payload = {
                "task_ids": [task.task_id for task in tasks],
                "relative_urls": [task.relative_url for task in tasks],
                "upsert_result": upsert_result,
            }
            meta_ref = await self._mongo_call(
                "save_fulltext_file_meta",
                self.manager.save_artifact(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    sub_doc_type=context.sub_doc_type,
                    stage=SyncStage.FULLTEXT_DISCOVER,
                    scope_type=ScopeType.DOCUMENT,
                    scope_id=context.doc_id,
                    artifact_type=ArtifactType.FULLTEXT_FILE_META,
                    content=meta_payload,
                    success_path=True,
                    content_type="application/json",
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": SyncStage.FULLTEXT_DISCOVER.value,
                },
            )
            failure_artifact_refs.append(meta_ref)
            await self._record_stage_success(
                context=context,
                stage=SyncStage.FULLTEXT_DISCOVER,
                artifact_ref=meta_ref,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
                started_at=stage_started_at,
            )
            await self._mongo_call(
                "update_pipeline_status_fulltext_pending",
                self.manager.update_pipeline_status(
                    doc_id=context.doc_id,
                    fulltext_status=PIPELINE_STATUS_PENDING,
                    core_status=PIPELINE_STATUS_PROCESSING,
                    overall_status=SyncOverallStatus.PROCESSING,
                    fulltext_required=True,
                    progress={
                        "fulltext_task_count": len(tasks),
                        "fulltext_task_success": 0,
                        "fulltext_task_failure": 0,
                    },
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "fulltext_task_count": len(tasks),
                },
            )
            return ProcessOutcome(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                final_status=SyncOverallStatus.SUCCESS.value,
                commit_safe=True,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                message="fulltext_tasks_upserted",
            )
        except Exception as error:
            outcome = await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.FULLTEXT_DISCOVER,
                scope_type=ScopeType.DOCUMENT,
                scope_id=context.doc_id,
                error_type=ErrorType.RETRYABLE,
                error_code="FULLTEXT_TASK_UPSERT_ERROR",
                error_message=str(error),
                exception_class=type(error).__name__,
                stacktrace=build_stacktrace(error),
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                artifact_refs=failure_artifact_refs,
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.SUCCESS,
                stage_started_at=stage_started_at,
            )
            if outcome.commit_safe:
                await self._mongo_call(
                    "update_pipeline_status_fulltext_discover_failure",
                    self.manager.update_pipeline_status(
                        doc_id=context.doc_id,
                        fulltext_status=PIPELINE_STATUS_FAILURE,
                        core_status=PIPELINE_STATUS_FAILURE,
                        overall_status=SyncOverallStatus.PARTIAL_FAILURE,
                        fulltext_required=True,
                        progress={
                            "fulltext_task_failure": 1,
                        },
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "fulltext_status": PIPELINE_STATUS_FAILURE,
                    },
                )
            return outcome

    async def _handle_image_discover_after_document_success(
        self,
        *,
        context: DocumentContext,
        source: str,
        retry_count_snapshot: int,
        artifact_refs: list[str],
    ) -> Optional[ProcessOutcome]:
        if not settings.IMAGE_TASK_UPSERT_ENABLED or context.event_type == "DELETE":
            return None

        stage_started_at = self._utcnow()
        failure_artifact_refs = list(artifact_refs)
        try:
            await self._mongo_call(
                "mark_stage_started_image_discover",
                self.manager.mark_stage_started(
                    doc_id=context.doc_id,
                    stage=SyncStage.IMAGE_DISCOVER,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": SyncStage.IMAGE_DISCOVER.value,
                },
            )
            tasks = extract_image_tasks(context.payload)
            if not tasks:
                await self._record_stage_terminal_status(
                    context=context,
                    stage=SyncStage.IMAGE_DISCOVER,
                    status=StageExecutionStatus.SKIPPED_NOT_PRESENT,
                    source=source,
                    retry_count_snapshot=retry_count_snapshot,
                    started_at=stage_started_at,
                )
                await self._mongo_call(
                    "update_pipeline_status_image_not_present",
                    self.manager.update_pipeline_status(
                        doc_id=context.doc_id,
                        image_status=PIPELINE_STATUS_NOT_PRESENT,
                        image_required=False,
                        progress={
                            "image_count_expected": 0,
                            "image_count_written": 0,
                            "image_task_failure": 0,
                            "image_task_pending": 0,
                            "image_task_leased": 0,
                        },
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "image_status": PIPELINE_STATUS_NOT_PRESENT,
                    },
                )
                await self._mongo_call(
                    "refresh_pipeline_status_after_image_not_present",
                    self.manager.refresh_async_pipeline_status(
                        doc_id=context.doc_id,
                        record_update_time=context.record_update_time,
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                    },
                )
                return ProcessOutcome(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    final_status=SyncOverallStatus.SUCCESS.value,
                    commit_safe=True,
                    sub_doc_type=context.sub_doc_type,
                    event_type=context.event_type,
                    message="image_not_present",
                )

            upsert_result = await get_image_task_repository().upsert_tasks(tasks)
            manifest_payload = {
                "task_ids": [task.task_id for task in tasks],
                "image_ids": [task.image_id for task in tasks],
                "relative_urls": [task.relative_url for task in tasks],
                "mmd_relative_urls": [task.mmd_relative_url for task in tasks],
                "upsert_result": upsert_result,
                "tasks": [_image_task_manifest_item(task) for task in tasks],
            }
            manifest_ref = await self._mongo_call(
                "save_image_manifest",
                self.manager.save_artifact(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    sub_doc_type=context.sub_doc_type,
                    stage=SyncStage.IMAGE_DISCOVER,
                    scope_type=ScopeType.DOCUMENT,
                    scope_id=context.doc_id,
                    artifact_type=ArtifactType.IMAGE_MANIFEST_JSON,
                    content=manifest_payload,
                    success_path=True,
                    content_type="application/json",
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": SyncStage.IMAGE_DISCOVER.value,
                },
            )
            failure_artifact_refs.append(manifest_ref)
            await self._record_stage_success(
                context=context,
                stage=SyncStage.IMAGE_DISCOVER,
                artifact_ref=manifest_ref,
                source=source,
                retry_count_snapshot=retry_count_snapshot,
                started_at=stage_started_at,
            )
            await self._mongo_call(
                "update_pipeline_status_image_pending",
                self.manager.update_pipeline_status(
                    doc_id=context.doc_id,
                    image_status=PIPELINE_STATUS_PENDING,
                    image_required=True,
                    progress={
                        "image_count_expected": len(tasks),
                        "image_count_written": 0,
                        "image_task_failure": 0,
                        "image_task_pending": len(tasks),
                        "image_task_leased": 0,
                    },
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "image_task_count": len(tasks),
                },
            )
            await self._mongo_call(
                "refresh_pipeline_status_after_image_pending",
                self.manager.refresh_async_pipeline_status(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                },
            )
            return ProcessOutcome(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                final_status=SyncOverallStatus.SUCCESS.value,
                commit_safe=True,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                message="image_tasks_upserted",
            )
        except Exception as error:
            outcome = await self._persist_stage_failure(
                context=context,
                source=source,
                stage=SyncStage.IMAGE_DISCOVER,
                scope_type=ScopeType.DOCUMENT,
                scope_id=context.doc_id,
                error_type=ErrorType.RETRYABLE,
                error_code="IMAGE_TASK_UPSERT_ERROR",
                error_message=str(error),
                exception_class=type(error).__name__,
                stacktrace=build_stacktrace(error),
                replay_mode=ReplayMode.FROM_RAW_EVENT,
                artifact_refs=failure_artifact_refs,
                retry_count_snapshot=retry_count_snapshot,
                overall_status=SyncOverallStatus.SUCCESS,
                stage_started_at=stage_started_at,
            )
            if outcome.commit_safe:
                await self._mongo_call(
                    "update_pipeline_status_image_discover_failure",
                    self.manager.update_pipeline_status(
                        doc_id=context.doc_id,
                        image_status=PIPELINE_STATUS_FAILURE,
                        image_required=True,
                        progress={
                            "image_task_failure": 1,
                        },
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "image_status": PIPELINE_STATUS_FAILURE,
                    },
                )
            return outcome

    async def _persist_event_validation_failure(
        self,
        *,
        record: ConsumedRecord,
        stage: SyncStage,
        error_code: str,
        error_message: str,
        scope_id: str,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
    ) -> ProcessOutcome:
        try:
            artifact_refs: list[str] = []
            raw_event_ref = await self._mongo_call(
                "save_event_validation_raw",
                self.manager.save_artifact(
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    sub_doc_type=sub_doc_type,
                    stage=stage,
                    scope_type=ScopeType.EVENT if doc_id is None else ScopeType.DOCUMENT,
                    scope_id=scope_id,
                    artifact_type=ArtifactType.RAW_EVENT_JSON,
                    content=record.parsed_payload,
                    success_path=False,
                    content_type="application/json",
                ),
                {
                    "event_fingerprint": record.event_fingerprint,
                    "doc_id": doc_id,
                    "record_update_time": record_update_time,
                },
            )
            artifact_refs.append(raw_event_ref)
            error_ref = await self._save_error_context(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=sub_doc_type,
                stage=stage,
                scope_type=ScopeType.EVENT if doc_id is None else ScopeType.DOCUMENT,
                scope_id=scope_id,
                error_type=ErrorType.NON_RETRYABLE,
                error_code=error_code,
                error_message=error_message,
                extra={
                    "event_fingerprint": record.event_fingerprint,
                    "topic": record.topic,
                    "partition": record.partition,
                    "offset": record.offset,
                },
            )
            artifact_refs.append(error_ref)
            failure_ref = await self._mongo_call(
                "persist_event_validation_failure",
                self.manager.persist_failure(
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    sub_doc_type=sub_doc_type,
                    event_type=infer_data_status(record.parsed_payload or {}),
                    failure_stage=stage,
                    scope_type=ScopeType.EVENT if doc_id is None else ScopeType.DOCUMENT,
                    scope_id=scope_id,
                    error_type=ErrorType.NON_RETRYABLE,
                    error_code=error_code,
                    error_message=error_message,
                    exception_class="ValidationError",
                    stacktrace=error_message,
                    replay_mode=ReplayMode.FROM_RAW_EVENT,
                    artifact_refs=artifact_refs,
                    source_event_meta=record.source_event_meta,
                ),
                {
                    "event_fingerprint": record.event_fingerprint,
                    "doc_id": doc_id,
                    "record_update_time": record_update_time,
                },
            )
            await self._best_effort_attempt(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=sub_doc_type,
                event_type=infer_data_status(record.parsed_payload or {}),
                source="kafka",
                stage=stage,
                stage_status=StageExecutionStatus.FAILURE.value,
                artifact_ref=error_ref,
                error_summary={"error_code": error_code, "error_message": error_message},
                source_event_meta=record.source_event_meta,
            )
            return ProcessOutcome(
                doc_id=doc_id,
                record_update_time=record_update_time,
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=True,
                sub_doc_type=sub_doc_type,
                pending_failure_refs=[failure_ref.failure_key],
                failure_persisted=True,
                message=error_code,
            )
        except MongoPersistenceError as error:
            logger.error(
                "event_validation_failure_persist_failed",
                event_fingerprint=record.event_fingerprint,
                doc_id=doc_id,
                record_update_time=record_update_time,
                error=str(error),
            )
            return ProcessOutcome(
                doc_id=doc_id,
                record_update_time=record_update_time,
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=False,
                sub_doc_type=sub_doc_type,
                message=error_code,
            )

    async def _persist_stage_failure(
        self,
        *,
        context: DocumentContext,
        source: str,
        stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        error_type: ErrorType,
        error_code: str,
        error_message: str,
        exception_class: str,
        stacktrace: str,
        replay_mode: ReplayMode,
        artifact_refs: list[str],
        retry_count_snapshot: int,
        overall_status: SyncOverallStatus,
        stage_started_at: datetime,
        raise_on_mongo_failure: bool = False,
    ) -> ProcessOutcome:
        try:
            # Successful runs do not retain raw payloads in failure-only mode.
            # Persist one replayable payload at the moment the document fails.
            failure_raw_ref = ""
            if not self.success_audit_enabled:
                failure_raw_ref = await self._mongo_call(
                    "save_failure_raw_event",
                    self.manager.save_artifact(
                        doc_id=context.doc_id,
                        record_update_time=context.record_update_time,
                        sub_doc_type=context.sub_doc_type,
                        stage=stage,
                        scope_type=ScopeType.DOCUMENT,
                        scope_id=context.doc_id,
                        artifact_type=ArtifactType.RAW_EVENT_JSON,
                        content=context.payload,
                        success_path=False,
                        content_type="application/json",
                    ),
                    {
                        "doc_id": context.doc_id,
                        "record_update_time": context.record_update_time,
                        "stage": stage.value,
                    },
                )
            error_ref = await self._save_error_context(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                stage=stage,
                scope_type=scope_type,
                scope_id=scope_id,
                error_type=error_type,
                error_code=error_code,
                error_message=error_message,
                extra={
                    "source": source,
                    "scope_type": scope_type.value,
                    "scope_id": scope_id,
                },
            )
            all_artifact_refs = list(
                dict.fromkeys(
                    ref for ref in [failure_raw_ref, *artifact_refs, error_ref] if ref
                )
            )
            failure_ref = await self._mongo_call(
                f"persist_failure_{stage.value}",
                self.manager.persist_failure(
                    doc_id=context.doc_id,
                    record_update_time=context.record_update_time,
                    sub_doc_type=context.sub_doc_type,
                    event_type=context.event_type,
                    failure_stage=stage,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    error_type=error_type,
                    error_code=error_code,
                    error_message=error_message,
                    exception_class=exception_class,
                    stacktrace=stacktrace,
                    replay_mode=replay_mode,
                    artifact_refs=all_artifact_refs,
                    source_event_meta=context.source_event_meta,
                    **self._failure_target_binding(context=context, item_id=scope_id),
                    retry_increment=1 if source == "retry" else 0,
                ),
                {
                    "doc_id": context.doc_id,
                    "record_update_time": context.record_update_time,
                    "stage": stage.value,
                    "error_code": error_code,
                },
            )
        except MongoPersistenceError:
            if raise_on_mongo_failure:
                raise
            return ProcessOutcome(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                final_status=overall_status.value,
                commit_safe=False,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                message=error_code,
            )

        await self._best_effort_stage_failure_status(
            context=context,
            stage=stage,
            artifact_ref=error_ref,
            error_code=error_code,
            error_message=error_message,
        )
        await self._best_effort_attempt(
            doc_id=context.doc_id,
            record_update_time=context.record_update_time,
            sub_doc_type=context.sub_doc_type,
            event_type=context.event_type,
            source=source,
            stage=stage,
            stage_status=StageExecutionStatus.FAILURE.value,
            retry_count_snapshot=retry_count_snapshot,
            artifact_ref=error_ref,
            error_summary={
                "error_code": error_code,
                "error_message": error_message,
                "exception_class": exception_class,
            },
            source_event_meta=context.source_event_meta,
            started_at=stage_started_at,
            finished_at=self._utcnow(),
        )
        return ProcessOutcome(
            doc_id=context.doc_id,
            record_update_time=context.record_update_time,
            final_status=overall_status.value,
            commit_safe=True,
            sub_doc_type=context.sub_doc_type,
            event_type=context.event_type,
            pending_failure_refs=[failure_ref.failure_key],
            failure_persisted=True,
            message=error_code,
        )

    async def _record_stage_success(
        self,
        *,
        context: DocumentContext,
        stage: SyncStage,
        artifact_ref: str,
        source: str,
        retry_count_snapshot: int,
        started_at: Optional[datetime] = None,
        skipped_empty: bool = False,
    ) -> None:
        if not self.success_audit_enabled:
            return
        stage_started_at = started_at or self._utcnow()
        finished_at = self._utcnow()
        status = (
            StageExecutionStatus.SKIPPED_EMPTY
            if skipped_empty
            else StageExecutionStatus.SUCCESS
        )
        await self._mongo_call(
            f"mark_stage_finished_{stage.value}",
            self.manager.mark_stage_finished(
                doc_id=context.doc_id,
                stage=stage,
                status=status,
                artifact_ref=artifact_ref,
                duration_ms=self._duration_ms(stage_started_at, finished_at),
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": stage.value,
            },
        )
        await self._mongo_call(
            f"record_attempt_{stage.value}",
            self.manager.record_attempt(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                source=source,
                stage=stage,
                stage_status=status.value,
                retry_count_snapshot=retry_count_snapshot,
                artifact_ref=artifact_ref,
                source_event_meta=context.source_event_meta,
                started_at=stage_started_at,
                finished_at=finished_at,
                duration_ms=self._duration_ms(stage_started_at, finished_at),
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": stage.value,
            },
        )

    async def _record_stage_terminal_status(
        self,
        *,
        context: DocumentContext,
        stage: SyncStage,
        status: StageExecutionStatus,
        source: str,
        retry_count_snapshot: int,
        artifact_ref: Optional[str] = None,
        started_at: Optional[datetime] = None,
        error_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        if (
            not self.success_audit_enabled
            and status is not StageExecutionStatus.FAILURE
        ):
            return
        stage_started_at = started_at or self._utcnow()
        finished_at = self._utcnow()
        await self._mongo_call(
            f"mark_stage_finished_{stage.value}",
            self.manager.mark_stage_finished(
                doc_id=context.doc_id,
                stage=stage,
                status=status,
                artifact_ref=artifact_ref,
                duration_ms=self._duration_ms(stage_started_at, finished_at),
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": stage.value,
            },
        )
        await self._mongo_call(
            f"record_attempt_{stage.value}",
            self.manager.record_attempt(
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                sub_doc_type=context.sub_doc_type,
                event_type=context.event_type,
                source=source,
                stage=stage,
                stage_status=status.value,
                retry_count_snapshot=retry_count_snapshot,
                artifact_ref=artifact_ref,
                error_summary=error_summary,
                source_event_meta=context.source_event_meta,
                started_at=stage_started_at,
                finished_at=finished_at,
                duration_ms=self._duration_ms(stage_started_at, finished_at),
            ),
            {
                "doc_id": context.doc_id,
                "record_update_time": context.record_update_time,
                "stage": stage.value,
            },
        )

    async def _save_error_context(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        stage: SyncStage,
        scope_type: ScopeType,
        scope_id: str,
        error_type: ErrorType,
        error_code: str,
        error_message: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        payload = {
            "error_type": error_type.value,
            "error_code": error_code,
            "error_message": error_message,
            "extra": extra or {},
        }
        return await self._mongo_call(
            f"save_error_context_{stage.value}",
            self.manager.save_artifact(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=sub_doc_type,
                stage=stage,
                scope_type=scope_type,
                scope_id=scope_id,
                artifact_type=ArtifactType.ERROR_CONTEXT_JSON,
                content=payload,
                success_path=False,
                content_type="application/json",
            ),
            {
                "doc_id": doc_id,
                "record_update_time": record_update_time,
                "stage": stage.value,
                "error_code": error_code,
            },
        )

    async def _rebuild_context_from_failure(
        self,
        failure_task: dict[str, Any],
    ) -> Optional[DocumentContext]:
        artifacts = await self._mongo_call(
            "get_failure_artifacts",
            self.manager.get_failure_artifacts(failure_task.get("artifact_refs", [])),
            {
                "failure_key": failure_task.get("failure_key"),
                "doc_id": failure_task.get("doc_id"),
            },
        )
        artifact_map = {item.get("artifact_type"): item for item in artifacts}
        raw_event = artifact_map.get(ArtifactType.RAW_EVENT_JSON.value)
        if raw_event is None or not isinstance(raw_event.get("payload"), dict):
            return None

        payload = raw_event["payload"]
        data = payload.get("data") or []
        if not data or not isinstance(data[0], dict):
            return None
        item = data[0]
        article_index = 0
        scope_id = str(failure_task.get("scope_id") or "")
        if ":" in scope_id:
            maybe_index = scope_id.rsplit(":", 1)[-1]
            if maybe_index.isdigit():
                article_index = int(maybe_index)
        return DocumentContext(
            doc_id=str(failure_task["doc_id"]),
            record_update_time=str(failure_task["record_update_time"]),
            sub_doc_type=str(failure_task.get("sub_doc_type") or payload.get("subDocType") or ""),
            event_type=str(failure_task.get("event_type") or infer_data_status(payload)),
            payload=payload,
            article_raw=item,
            article_index=article_index,
            source_event_meta=self.manager._dict_to_event_meta(failure_task.get("source_event_meta") or {}),
            event_fingerprint=str((failure_task.get("source_event_meta") or {}).get("event_fingerprint", "")),
        )

    async def _mongo_call(self, operation: str, awaitable, fallback_context: dict[str, Any]):
        if not self.success_audit_enabled and (
            operation in _SUCCESS_AUDIT_MONGO_OPERATIONS
            or operation.startswith("mark_stage_started_")
        ):
            # The coroutine is intentionally not awaited: no Mongo I/O is issued.
            awaitable.close()
            return None
        try:
            return await awaitable
        except Exception as error:
            if isinstance(error, MongoPersistenceError):
                self.manager._connected = False
            elif "pymongo" in type(error).__module__.lower():
                self.manager._connected = False
            fallback_payload = {
                "mongo_operation": operation,
                "error": str(error),
                "fallback_context": fallback_context,
                "timestamp": self._utcnow().isoformat(),
            }
            try:
                await self.manager.mark_mongo_unavailable(fallback_payload)
            except Exception as fallback_error:
                logger.error(
                    "mongo_fallback_dump_failed",
                    operation=operation,
                    error=str(fallback_error),
                )
            raise MongoPersistenceError(f"{operation} failed: {error}") from error

    async def _best_effort_stage_failure_status(
        self,
        *,
        context: DocumentContext,
        stage: SyncStage,
        artifact_ref: str,
        error_code: str,
        error_message: str,
    ) -> None:
        try:
            await self.manager.mark_stage_finished(
                doc_id=context.doc_id,
                stage=stage,
                status=StageExecutionStatus.FAILURE,
                artifact_ref=artifact_ref,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception as error:
            logger.warning(
                "best_effort_mark_stage_failure_failed",
                doc_id=context.doc_id,
                record_update_time=context.record_update_time,
                stage=stage.value,
                error=str(error),
            )

    async def _best_effort_attempt(
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
        source_event_meta=None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ) -> None:
        try:
            await self.manager.record_attempt(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=sub_doc_type,
                event_type=event_type,
                source=source,
                stage=stage,
                stage_status=stage_status,
                retry_count_snapshot=retry_count_snapshot,
                artifact_ref=artifact_ref,
                error_summary=error_summary,
                source_event_meta=source_event_meta,
                started_at=started_at,
                finished_at=finished_at,
            )
        except Exception as error:
            logger.warning(
                "best_effort_record_attempt_failed",
                doc_id=doc_id,
                record_update_time=record_update_time,
                stage=stage.value,
                error=str(error),
            )

    def _dedup_contexts(
        self,
        contexts: list[DocumentContext],
    ) -> tuple[list[DocumentContext], int]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[DocumentContext] = []
        for context in contexts:
            key = (context.doc_id, context.record_update_time, context.sub_doc_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(context)
        return deduped, len(contexts) - len(deduped)

    def _check_skip_reason(
        self,
        existing_status: Optional[dict[str, Any]],
        context: DocumentContext,
        *,
        source: str,
    ) -> Optional[str]:
        if existing_status is None:
            return None

        latest_time = str(existing_status.get("latest_record_update_time") or "")
        compare = self._compare_record_times(latest_time, context.record_update_time)
        retry_summary = existing_status.get("retry_summary") or {}

        if compare > 0:
            return "older_than_latest_processed_version"

        if compare == 0:
            if source == "retry":
                if existing_status.get("overall_status") in {
                    SyncOverallStatus.SUCCESS.value,
                    SyncOverallStatus.DELETED.value,
                } and not retry_summary.get("has_pending_failure"):
                    return "retry_target_already_resolved"
                return None

            if retry_summary.get("has_pending_failure"):
                return "same_version_pending_failure_already_recorded"

            if existing_status.get("overall_status") in {
                SyncOverallStatus.SUCCESS.value,
                SyncOverallStatus.DELETED.value,
                SyncOverallStatus.PARTIAL_FAILURE.value,
                SyncOverallStatus.FAILURE.value,
            }:
                return "same_version_already_processed"

            if existing_status.get("last_success_stage") in {
                SyncStage.MILVUS_WRITE_DOCUMENT.value,
                SyncStage.DELETE_DOCUMENT.value,
            }:
                return "same_version_write_already_completed"

        return None

    def _check_retry_stale(
        self,
        status_doc: Optional[dict[str, Any]],
        failure_task: dict[str, Any],
    ) -> Optional[str]:
        if status_doc is None:
            return None
        latest_time = str(status_doc.get("latest_record_update_time") or "")
        failure_time = str(failure_task.get("record_update_time") or "")
        compare = self._compare_record_times(latest_time, failure_time)
        if compare > 0:
            return "newer_version_already_processed"
        if (
            status_doc.get("overall_status") == SyncOverallStatus.DELETED.value
            and failure_task.get("failure_stage") != SyncStage.DELETE_DOCUMENT.value
        ):
            return "document_already_deleted"
        return None

    def _compare_record_times(self, left: str, right: str) -> int:
        left_dt = parse_record_update_time(left)
        right_dt = parse_record_update_time(right)
        if left_dt is not None and right_dt is not None:
            if left_dt > right_dt:
                return 1
            if left_dt < right_dt:
                return -1
            return 0
        if left > right:
            return 1
        if left < right:
            return -1
        return 0

    def _classify_processing_error(
        self,
        error: Exception,
        *,
        default_retryable: bool,
        default_code: str,
    ) -> tuple[ErrorType, str]:
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            if status == 429 or status >= 500:
                return ErrorType.RETRYABLE, f"HTTP_{status}"
            return ErrorType.NON_RETRYABLE, f"HTTP_{status}"
        if isinstance(
            error,
            (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            return ErrorType.RETRYABLE, default_code
        if isinstance(error, ValueError):
            return ErrorType.NON_RETRYABLE, default_code
        return (
            ErrorType.RETRYABLE if default_retryable else ErrorType.NON_RETRYABLE,
            default_code,
        )

    def _classify_milvus_error(self, error: Exception) -> tuple[ErrorType, str]:
        message = str(error).lower()
        if "dimension" in message or "missing required vector" in message:
            return ErrorType.NON_RETRYABLE, "MILVUS_VECTOR_DIM_MISMATCH"
        if "does not exist" in message:
            return ErrorType.RETRYABLE, "MILVUS_COLLECTION_UNAVAILABLE"
        return ErrorType.RETRYABLE, "MILVUS_WRITE_ERROR"

    def _merge_post_success_outcomes(
        self,
        base: ProcessOutcome,
        extra: ProcessOutcome,
    ) -> ProcessOutcome:
        messages = [item for item in (base.message, extra.message) if item]
        pending_failure_refs = list(
            dict.fromkeys([*base.pending_failure_refs, *extra.pending_failure_refs])
        )
        return ProcessOutcome(
            doc_id=base.doc_id or extra.doc_id,
            record_update_time=base.record_update_time or extra.record_update_time,
            final_status=extra.final_status or base.final_status,
            commit_safe=base.commit_safe and extra.commit_safe,
            sub_doc_type=base.sub_doc_type or extra.sub_doc_type,
            event_type=base.event_type or extra.event_type,
            pending_failure_refs=pending_failure_refs,
            failure_persisted=base.failure_persisted or extra.failure_persisted,
            message=";".join(messages),
        )

    def _duration_ms(self, start: datetime, end: datetime) -> int:
        return max(0, int((end - start).total_seconds() * 1000))

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)


def _live_e2e_no_delete_result(
    *,
    operation: str,
    collection_name: str,
    doc_id: str,
) -> dict[str, Any]:
    return {
        "skipped": True,
        "status": "DELETE_NOOP_RECORDED",
        "reason": "LIVE_E2E_NO_DELETE",
        "operation": operation,
        "collection_name": collection_name,
        "doc_id": doc_id,
    }


def _image_task_manifest_item(task: Any) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "image_id": task.image_id,
        "doc_id": task.doc_id,
        "record_update_time": task.record_update_time,
        "relative_url": task.relative_url,
        "mmd_relative_url": task.mmd_relative_url,
        "sequence": task.sequence,
        "doc_title": task.doc_title,
        "keyword": task.keyword,
        "accessType": task.accessType,
        "classificationCode": task.classificationCode,
        "year": task.year,
        "hasAuthorname": task.hasAuthorname,
        "hasCreateOrganizationname": task.hasCreateOrganizationname,
        "language": task.language,
        "sysDataStatus": task.sysDataStatus,
        "hasSourcesysSourceId": task.hasSourcesysSourceId,
        "hasSourceidSet": task.hasSourceidSet,
        "title": task.title,
        "name": task.name,
        "width": task.width,
        "height": task.height,
        "file_type": task.file_type,
        "size": task.size,
        "raw_img_meta": task.raw_img_meta,
        "img_meta": task.img_meta,
        "context_before": task.context_before,
        "context_after": task.context_after,
        "source_item_index": task.source_item_index,
        "image_index": task.image_index,
    }


async def process_batch_existing_arch(
    batch: list[ConsumedRecord],
    *,
    mode: str = "standard",
    document_collection_name: Optional[str] = None,
) -> BatchProcessingResult:
    pipeline = ScalarSyncPipeline(
        mode=mode,
        document_collection_name=document_collection_name,
    )
    return await pipeline.process_batch(batch)


async def retry_failure_task(failure_task: dict[str, Any]) -> ProcessOutcome:
    pipeline = ScalarSyncPipeline()
    return await pipeline.retry_failure(failure_task)
