"""Full-text worker skeleton for download/artifact/chunk-plan processing."""
from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.config import settings
from app.models.chunk import Chunk
from app.models.sync_runtime import (
    ArtifactType,
    ErrorType,
    ReplayMode,
    ScopeType,
    SourceEventMeta,
    SyncStage,
)
from app.observability.logger import get_logger
from app.services.chunker import build_document_chunks
from app.services.fulltext_downloader import (
    FullTextConfigurationError,
    FullTextDocumentTooLarge,
    FullTextDownloader,
    FullTextNonRetryableDownloadError,
    FullTextRetryableDownloadError,
    build_fulltext_readiness_status,
)
from app.services.fulltext_parser import ParsedFullText, parse_mmd
from app.services.fulltext_chunk_pipeline import process_fulltext_chunk_plan
from app.services.fulltext_task_repository import (
    FullTextTaskRepository,
    get_fulltext_task_repository,
)
from app.services.embedder import get_embedding_client
from app.services.collection_targets import (
    build_fulltext_chunk_index_spec,
    build_fulltext_collection_signature,
    collection_target_binding,
    resolve_fulltext_collection_target,
)
from app.services.milvus_chunk_service import MilvusChunkService
from app.services.mongo_state_manager import MongoStateManager, get_mongo_state_manager


class FullTextArtifactError(RuntimeError):
    """Raised when artifact persistence fails."""


class FullTextParseError(RuntimeError):
    """Raised when .mmd parsing fails."""


class FullTextChunkPlanError(RuntimeError):
    """Raised when chunk plan construction fails."""


class FullTextChunkPipelineError(RuntimeError):
    """Raised when chunk embedding/write pipeline fails."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


logger = get_logger(__name__)


@dataclass(slots=True)
class FullTextWorkerResult:
    task_id: str
    success: bool
    artifact_refs: list[str]
    chunk_count: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    failure_persisted: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


class FullTextWorker:
    def __init__(
        self,
        *,
        repository: Optional[FullTextTaskRepository] = None,
        downloader: Optional[FullTextDownloader] = None,
        state_manager: Optional[MongoStateManager] = None,
        parse_func: Callable[[str], ParsedFullText] = parse_mmd,
        chunk_builder: Callable[..., list[Chunk]] = build_document_chunks,
        embedder: Optional[Any] = None,
        chunk_writer: Optional[Any] = None,
        chunk_pipeline: Callable[..., Any] = process_fulltext_chunk_plan,
        chunk_collection_name: Optional[str] = None,
    ) -> None:
        self.repository = repository or get_fulltext_task_repository()
        self.downloader = downloader or FullTextDownloader()
        self.state_manager = state_manager or get_mongo_state_manager()
        self.parse_func = parse_func
        self.chunk_builder = chunk_builder
        self.embedder = embedder
        self.chunk_writer = chunk_writer
        self.chunk_pipeline = chunk_pipeline
        self.chunk_collection_name = str(
            chunk_collection_name or settings.resolved_fulltext_chunk_collection_name
        ).strip()

    def _resolved_chunk_collection_name(self, chunk_writer: Any) -> str:
        target = resolve_fulltext_collection_target(
            chunk_writer.client,
            logical_name=self.chunk_collection_name,
            ensure_collection=lambda name: chunk_writer._ensure_chunk_collection_created(  # noqa: SLF001
                name,
                vector_dim=settings.VECTOR_DIM,
            ),
        )
        return target.resolved_name

    async def process_task(self, task: dict[str, Any]) -> FullTextWorkerResult:
        task_id = str(task.get("task_id") or task.get("_id") or "")
        doc_id = str(task.get("doc_id") or "")
        record_update_time = str(task.get("record_update_time") or "")
        relative_url = str(task.get("relative_url") or "")
        artifact_refs: list[str] = []
        timings_ms: dict[str, float] = _source_timings(task)
        task_started_at = time.perf_counter()
        failure_persisted = False

        try:
            if not task_id or not doc_id or not relative_url:
                raise FullTextParseError("task_id, doc_id, and relative_url are required")

            download_started_at = time.perf_counter()
            download = await self.downloader.download_text(relative_url)
            timings_ms["mmd_download"] = round(
                (time.perf_counter() - download_started_at) * 1000.0,
                3,
            )
            _enforce_fulltext_token_limit(download.text)
            parse_started_at = time.perf_counter()
            parsed = self._parse(download.text)
            timings_ms["parse"] = round((time.perf_counter() - parse_started_at) * 1000.0, 3)
            chunk_build_started_at = time.perf_counter()
            chunks = self._build_chunks(doc_id=doc_id, parsed=parsed)
            timings_ms["chunk_build"] = round(
                (time.perf_counter() - chunk_build_started_at) * 1000.0,
                3,
            )
            chunk_plan = {
                "doc_id": doc_id,
                "task_id": task_id,
                "relative_url": relative_url,
                "chunk_count": len(chunks),
                "doc_meta": _task_doc_meta(task),
                "chunks": [_chunk_to_dict(chunk) for chunk in chunks],
            }
            chunk_pipeline_started_at = time.perf_counter()
            pipeline_result = await self._run_chunk_pipeline(chunk_plan)
            timings_ms["chunk_pipeline"] = round(
                (time.perf_counter() - chunk_pipeline_started_at) * 1000.0,
                3,
            )
            pipeline_summary = _pipeline_result_to_dict(pipeline_result)
            pipeline_timings = (pipeline_summary.get("metadata") or {}).get("pipeline_timings_ms")
            if isinstance(pipeline_timings, dict):
                for stage, value in pipeline_timings.items():
                    try:
                        timings_ms[f"pipeline_{stage}"] = round(float(value), 3)
                    except (TypeError, ValueError):
                        continue
            if not bool(pipeline_summary.get("success")):
                pipeline_summary_ref = await self._save_artifact(
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    task_id=task_id,
                    stage=SyncStage.MILVUS_WRITE_CHUNKS,
                    artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
                    content=pipeline_summary,
                    content_type="application/json",
                    meta={
                        "relative_url": relative_url,
                        "chunk_count": pipeline_summary.get("chunk_count", len(chunks)),
                        "success": False,
                    },
                    success_path=False,
                )
                artifact_refs.append(pipeline_summary_ref)
                raise FullTextChunkPipelineError(
                    _format_pipeline_failure_message(pipeline_summary),
                    retryable=_pipeline_retryable(pipeline_summary),
                )

            final_chunk_count = int(pipeline_summary.get("chunk_count") or len(chunks))
            timings_ms["total"] = round((time.perf_counter() - task_started_at) * 1000.0, 3)
            await self.repository.mark_task_success(
                task_id,
                artifact_refs=artifact_refs,
                chunk_count=final_chunk_count,
            )
            return FullTextWorkerResult(
                task_id=task_id,
                success=True,
                artifact_refs=artifact_refs,
                chunk_count=final_chunk_count,
                timings_ms={
                    **timings_ms,
                    "total": round((time.perf_counter() - task_started_at) * 1000.0, 3),
                },
            )
        except Exception as error:
            error_code, retryable = _classify_worker_error(error)
            error_message = str(error)
            if task_id:
                await self.repository.mark_task_failure(
                    task_id,
                    error_code=error_code,
                    error_message=error_message,
                    retryable=retryable,
                )
                failure_persisted = await self._persist_failure_record(
                    task=task,
                    task_id=task_id,
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    artifact_refs=artifact_refs,
                    error=error,
                    error_code=error_code,
                    error_message=error_message,
                    retryable=retryable,
                )
                await _refresh_pipeline_status_best_effort(
                    self.state_manager,
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                )
            return FullTextWorkerResult(
                task_id=task_id,
                success=False,
                artifact_refs=artifact_refs,
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
                failure_persisted=failure_persisted,
                timings_ms={
                    **timings_ms,
                    "total": round((time.perf_counter() - task_started_at) * 1000.0, 3),
                },
            )

    async def _persist_failure_record(
        self,
        *,
        task: dict[str, Any],
        task_id: str,
        doc_id: str,
        record_update_time: str,
        artifact_refs: list[str],
        error: Exception,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> bool:
        """Keep a retry record bound to the exact fulltext write target."""
        persist = getattr(self.state_manager, "persist_failure", None)
        if not callable(persist):
            return False
        try:
            await persist(
                doc_id=doc_id or None,
                record_update_time=record_update_time or None,
                sub_doc_type=None,
                event_type=str(task.get("event_type") or "UPSERT"),
                failure_stage=_fulltext_failure_stage(error_code),
                scope_type=ScopeType.FULLTEXT_FILE,
                scope_id=task_id,
                error_type=ErrorType.RETRYABLE if retryable else ErrorType.NON_RETRYABLE,
                error_code=error_code,
                error_message=error_message,
                exception_class=type(error).__name__,
                stacktrace=error_message,
                replay_mode=ReplayMode.FROM_MMD_TEXT,
                artifact_refs=list(artifact_refs),
                source_event_meta=_task_source_event_meta(task, task_id, "fulltext"),
                replay_payload=_replay_task_payload(task),
                **self._failure_target_binding(
                    source_version=record_update_time,
                    item_id=task_id,
                ),
            )
            return True
        except Exception as persistence_error:
            # The failed task remains in the Mongo task queue. Do not hide the
            # original processing error when the secondary retry index is down.
            logger.error(
                "fulltext_failure_record_persist_failed",
                task_id=task_id,
                doc_id=doc_id,
                error=str(persistence_error),
            )
            return False

    def _failure_target_binding(
        self,
        *,
        source_version: str,
        item_id: str,
    ) -> dict[str, str]:
        chunk_writer = self.chunk_writer
        if chunk_writer is None and isinstance(self.state_manager, MongoStateManager):
            # Download and parse failures can happen before the normal chunk
            # writer is needed. Resolve the production target read-only so the
            # resulting failure record is still tied to its physical target.
            try:
                chunk_writer = MilvusChunkService()
                self.chunk_writer = chunk_writer
            except Exception as error:
                logger.warning(
                    "fulltext_failure_target_writer_init_failed",
                    collection_name=self.chunk_collection_name,
                    error=str(error),
                )
        client = getattr(chunk_writer, "client", None)
        if client is not None:
            try:
                target = resolve_fulltext_collection_target(
                    client,
                    logical_name=self.chunk_collection_name,
                    vector_dim=settings.VECTOR_DIM,
                    ensure_collection=None,
                )
                return collection_target_binding(
                    target,
                    pipeline="fulltext",
                    source_version=source_version,
                    item_id=item_id,
                )
            except Exception as error:
                logger.warning(
                    "fulltext_failure_target_resolution_failed",
                    collection_name=self.chunk_collection_name,
                    error=str(error),
                )

        # An early download/parse failure has no Milvus writer yet. Bind it to
        # the configured target signature; a later target mismatch is skipped
        # by the compensator instead of being written into a different target.
        schema_signature = build_fulltext_collection_signature(vector_dim=settings.VECTOR_DIM)
        index_signature = build_fulltext_chunk_index_spec().signature
        return {
            "pipeline": "fulltext",
            "source_version": str(source_version or ""),
            "item_id": str(item_id or ""),
            "logical_collection": self.chunk_collection_name,
            "resolved_collection": self.chunk_collection_name,
            "target_signature": f"{schema_signature}|INDEX={index_signature}",
        }

    async def _save_artifact(
        self,
        *,
        doc_id: str,
        record_update_time: str,
        task_id: str,
        stage: SyncStage,
        artifact_type: ArtifactType,
        content: Any,
        content_type: str,
        meta: dict[str, Any],
        success_path: bool = True,
    ) -> str:
        try:
            return await self.state_manager.save_artifact(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=None,
                stage=stage,
                scope_type=ScopeType.FULLTEXT_FILE,
                scope_id=task_id,
                artifact_type=artifact_type,
                content=content,
                success_path=success_path,
                content_type=content_type,
                meta=meta,
            )
        except Exception as error:
            raise FullTextArtifactError(str(error)) from error

    def _parse(self, mmd_text: str) -> ParsedFullText:
        try:
            return self.parse_func(mmd_text)
        except Exception as error:
            raise FullTextParseError(str(error)) from error

    def _build_chunks(self, *, doc_id: str, parsed: ParsedFullText) -> list[Chunk]:
        try:
            return self.chunk_builder(doc_id=doc_id, parsed=parsed)
        except Exception as error:
            raise FullTextChunkPlanError(str(error)) from error

    async def _run_chunk_pipeline(self, chunk_plan: dict[str, Any]):
        embedder = self.embedder or get_embedding_client()
        chunk_writer = self.chunk_writer or MilvusChunkService()
        try:
            kwargs = {
                "embedder": embedder,
                "chunk_writer": chunk_writer,
            }
            if _callable_accepts_keyword(self.chunk_pipeline, "collection_name"):
                kwargs["collection_name"] = self._resolved_chunk_collection_name(chunk_writer)
            maybe_result = self.chunk_pipeline(chunk_plan, **kwargs)
            if inspect.isawaitable(maybe_result):
                return await maybe_result
            return maybe_result
        except FullTextChunkPipelineError:
            raise
        except Exception as error:
            raise FullTextChunkPipelineError(
                str(error),
                retryable=not isinstance(error, ValueError),
            ) from error


def build_fulltext_runtime_status(
    *,
    worker_running: bool = False,
    worker_status: Optional[dict[str, Any]] = None,
    task_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    status = {
        **build_fulltext_readiness_status(),
        "worker_enabled": settings.FULLTEXT_WORKER_ENABLED,
        "worker_running": worker_running,
        "lease_seconds": settings.FULLTEXT_WORKER_LEASE_SEC,
        "batch_size": settings.FULLTEXT_WORKER_BATCH_SIZE,
        "concurrency": settings.FULLTEXT_WORKER_CONCURRENCY,
        "claim_created_after": getattr(settings, "FULLTEXT_WORKER_CLAIM_CREATED_AFTER", ""),
        "poll_interval_seconds": settings.FULLTEXT_WORKER_POLL_INTERVAL_SEC,
        "task_timeout_seconds": settings.FULLTEXT_WORKER_TASK_TIMEOUT_SEC,
        "last_run_at": None,
        "last_task_id": None,
        "last_processed_count": 0,
        "last_error": None,
        "paused": False,
        "draining": False,
        "pause_reason": None,
        "inflight_runs": 0,
        "task_summary": task_summary,
    }
    if worker_status:
        status.update(
            {
                "worker_enabled": bool(worker_status.get("enabled", status["worker_enabled"])),
                "worker_running": bool(worker_status.get("running", status["worker_running"])),
                "batch_size": worker_status.get("batch_size", status["batch_size"]),
                "concurrency": worker_status.get("concurrency", status["concurrency"]),
                "claim_created_after": worker_status.get(
                    "claim_created_after",
                    status["claim_created_after"],
                ),
                "poll_interval_seconds": worker_status.get(
                    "poll_interval_seconds",
                    status["poll_interval_seconds"],
                ),
                "task_timeout_seconds": worker_status.get(
                    "task_timeout_seconds",
                    status["task_timeout_seconds"],
                ),
                "last_run_at": worker_status.get("last_run_at"),
                "last_task_id": worker_status.get("last_task_id"),
                "last_processed_count": worker_status.get("last_processed_count", 0),
                "last_error": worker_status.get("last_error"),
                "paused": bool(worker_status.get("paused", status["paused"])),
                "draining": bool(
                    worker_status.get("draining", status["draining"])
                ),
                "pause_reason": worker_status.get("pause_reason"),
                "inflight_runs": worker_status.get(
                    "inflight_runs",
                    status["inflight_runs"],
                ),
            }
        )
    return status


def _callable_accepts_keyword(func: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == keyword
        for parameter in signature.parameters.values()
    )


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump(exclude_none=True)
    return chunk.dict()


def _task_doc_meta(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "title",
        "keyword",
        "accessType",
        "classificationCode",
        "year",
        "hasAuthorname",
        "hasCreateOrganizationname",
        "language",
    )
    return {key: task.get(key) for key in keys if task.get(key) is not None}


def _source_timings(task: dict[str, Any]) -> dict[str, float]:
    raw = task.get("_source_timings_ms")
    if not isinstance(raw, dict):
        return {}
    timings: dict[str, float] = {}
    for key, value in raw.items():
        try:
            timings[str(key)] = round(max(float(value), 0.0), 3)
        except (TypeError, ValueError):
            continue
    return timings


async def _refresh_pipeline_status_best_effort(
    state_manager: Any,
    *,
    doc_id: str,
    record_update_time: str,
) -> None:
    refresh = getattr(state_manager, "refresh_async_pipeline_status", None)
    if refresh is None or not doc_id:
        return
    try:
        await refresh(doc_id=doc_id, record_update_time=record_update_time)
    except Exception:
        return


def _enforce_fulltext_token_limit(text: str) -> None:
    """Reject an oversized document before parse/chunk/embed consumes memory."""
    configured_limit = max(int(getattr(settings, "FULLTEXT_MAX_TOKENS", 0) or 0), 0)
    if not configured_limit:
        return
    estimated_tokens = _estimate_text_tokens(text)
    if estimated_tokens > configured_limit:
        raise FullTextDocumentTooLarge(
            "MMD text estimated token count "
            f"{estimated_tokens} exceeds FULLTEXT_MAX_TOKENS={configured_limit}"
        )


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text))
    symbols = len(re.findall(r"[^\s\w\u4e00-\u9fff]", text))
    return cjk_chars + latin_words + symbols


def _classify_worker_error(error: Exception) -> tuple[str, bool]:
    if isinstance(error, FullTextConfigurationError):
        return "FULLTEXT_CONFIG_ERROR", False
    if isinstance(error, FullTextDocumentTooLarge):
        return "FULLTEXT_DOCUMENT_TOO_LARGE", False
    if isinstance(error, FullTextRetryableDownloadError):
        return "FULLTEXT_DOWNLOAD_RETRYABLE", True
    if isinstance(error, FullTextNonRetryableDownloadError):
        return "FULLTEXT_DOWNLOAD_NON_RETRYABLE", False
    if isinstance(error, FullTextArtifactError):
        return "FULLTEXT_ARTIFACT_ERROR", True
    if isinstance(error, FullTextParseError):
        return "FULLTEXT_PARSE_ERROR", False
    if isinstance(error, FullTextChunkPlanError):
        return "FULLTEXT_CHUNK_PLAN_ERROR", False
    if isinstance(error, FullTextChunkPipelineError):
        return "FULLTEXT_CHUNK_PIPELINE_ERROR", error.retryable
    return "FULLTEXT_WORKER_ERROR", False


def _fulltext_failure_stage(error_code: str) -> SyncStage:
    if error_code.startswith("FULLTEXT_DOWNLOAD") or error_code == "FULLTEXT_DOCUMENT_TOO_LARGE":
        return SyncStage.FULLTEXT_DOWNLOAD
    if error_code == "FULLTEXT_PARSE_ERROR":
        return SyncStage.FULLTEXT_PARSE_MMD
    if error_code == "FULLTEXT_CHUNK_PLAN_ERROR":
        return SyncStage.FULLTEXT_BUILD_CHUNKS
    if error_code == "FULLTEXT_CHUNK_PIPELINE_ERROR":
        return SyncStage.MILVUS_WRITE_CHUNKS
    return SyncStage.FULLTEXT_DISCOVER


def _task_source_event_meta(task: dict[str, Any], task_id: str, pipeline: str) -> SourceEventMeta:
    raw_meta = task.get("source_event_meta")
    if isinstance(raw_meta, SourceEventMeta):
        return raw_meta
    if isinstance(raw_meta, dict):
        return SourceEventMeta(
            topic=str(raw_meta.get("topic") or f"mongo-{pipeline}-task"),
            partition=_as_int(raw_meta.get("partition"), -1),
            offset=_as_int(raw_meta.get("offset"), -1),
            event_fingerprint=str(raw_meta.get("event_fingerprint") or f"{pipeline}-task:{task_id}"),
        )
    return SourceEventMeta(
        topic=str(task.get("source_topic") or f"mongo-{pipeline}-task"),
        partition=_as_int(task.get("source_partition"), -1),
        offset=_as_int(task.get("source_offset"), -1),
        event_fingerprint=str(task.get("event_fingerprint") or f"{pipeline}-task:{task_id}"),
    )


def _replay_task_payload(task: dict[str, Any]) -> dict[str, Any]:
    """Keep only JSON-compatible task data needed by the direct retry path."""
    return {
        key: value
        for key, value in task.items()
        if key not in {"_id", "created_at", "updated_at", "lease_expires_at"}
    }


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pipeline_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return dict(result)
    raise FullTextChunkPipelineError(
        f"chunk pipeline returned unsupported result type: {type(result).__name__}",
        retryable=False,
    )


def _pipeline_retryable(summary: dict[str, Any]) -> bool:
    return any(bool(item.get("retryable")) for item in _pipeline_failure_items(summary))


def _format_pipeline_failure_message(summary: dict[str, Any]) -> str:
    payload = {"failure_items": _pipeline_failure_items(summary)}
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= 4000:
        return text
    return text[:3997] + "..."


def _pipeline_failure_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    items = summary.get("failure_items") or []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if isinstance(item, dict):
            normalized.append(
                {
                    "chunkId": item.get("chunkId"),
                    "error_code": item.get("error_code"),
                    "error_message": item.get("error_message"),
                    "retryable": bool(item.get("retryable")),
                }
            )
        else:
            normalized.append(
                {
                    "chunkId": None,
                    "error_code": "FULLTEXT_CHUNK_PIPELINE_ERROR",
                    "error_message": str(item),
                    "retryable": False,
                }
            )
    return normalized
