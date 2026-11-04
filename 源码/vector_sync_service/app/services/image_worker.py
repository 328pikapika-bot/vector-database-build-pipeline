"""Image worker skeleton for download and durable manifest persistence."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from hashlib import sha256
import inspect
import time
from typing import Any, Awaitable, Callable, Optional

from app.config import settings
from app.models.sync_runtime import (
    ArtifactType,
    ErrorType,
    ReplayMode,
    ScopeType,
    SourceEventMeta,
    SyncStage,
)
from app.observability.logger import get_logger
from app.observability.metrics import (
    image_mmd_cache_total,
    image_stage_latency_seconds,
)
from app.services.fulltext_downloader import (
    FullTextConfigurationError,
    FullTextDocumentTooLarge,
    FullTextDownloader,
    FullTextDownloadResult,
    FullTextNonRetryableDownloadError,
    FullTextRetryableDownloadError,
)
from app.services.image_context_extractor import (
    compose_context_after,
    extract_mmd_image_contexts,
    find_mmd_context_for_image,
)
from app.services.image_index_quality import (
    apply_image_index_quality,
    evaluate_image_indexability,
)
from app.services.image_context_text import canonicalize_context_text, first_canonical_text, join_context_parts
from app.services.image_downloader import (
    ImageConfigurationError,
    ImageDownloader,
    ImageDownloadResult,
    ImageNonRetryableDownloadError,
    ImageRetryableDownloadError,
    build_image_readiness_status,
)
from app.services.image_task_repository import (
    ImageTaskRepository,
    get_image_task_repository,
)
from app.services.image_worker_resources import (
    ImageWorkerResourcePool,
    create_image_worker_resource_pool,
)
from app.services.image_worker_utils import sanitize_for_mongo
from app.services.image_features import (
    ImageDimensionValidationResult,
    ImageManifestValidationFailure,
    ImageManifestValidationResult,
    IMAGE_PIXEL_LIMIT_EXCEEDED,
    build_image_record_draft_from_manifest,
    validate_image_bytes_dimensions,
    validate_image_manifest,
)
from app.services.mongo_state_manager import MongoStateManager, get_mongo_state_manager
from app.services.runtime_attachment_store import RuntimeAttachmentBatch
from app.services.collection_targets import (
    build_image_collection_signature,
    build_image_dense_index_spec,
    collection_target_binding,
    resolve_image_collection_target,
)


_REFRESH_INFLIGHT: dict[tuple[int, str, str], asyncio.Task[None]] = {}
_REFRESH_INFLIGHT_LOCK = asyncio.Lock()
logger = get_logger(__name__)


class ImageArtifactError(RuntimeError):
    """Raised when image manifest artifact persistence fails."""


class ImageTaskInvalidError(ValueError):
    """Raised when an image task is missing required fields."""


class ImageMmdContextError(RuntimeError):
    """Raised when a required MMD resource cannot provide image context."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        self.retryable = retryable
        super().__init__(message)


class ImageDimensionInvalidError(ValueError):
    """Raised when downloaded image bytes fail local dimension validation."""

    def __init__(self, result: ImageDimensionValidationResult) -> None:
        self.result = result
        error_code = result.error_code or "IMAGE_DIMENSION_DECODE_ERROR"
        message = result.error_message or "image dimension validation failed"
        super().__init__(f"{error_code}: {message}")


class ImageManifestInvalidError(ValueError):
    """Raised when a worker-produced image manifest fails validation."""

    def __init__(self, result: ImageManifestValidationResult) -> None:
        self.result = result
        super().__init__(_format_manifest_failures(result.failures))


class ImageFeaturePipelineError(RuntimeError):
    """Raised when feature pipeline execution fails before returning a result."""

    def __init__(
        self,
        message: str,
        *,
        failure_items: list[dict[str, Any]],
        retryable: bool,
    ) -> None:
        self.failure_items = failure_items
        self.retryable = retryable
        super().__init__(message)


@dataclass(slots=True)
class ImageWorkerResult:
    task_id: str
    success: bool
    artifact_refs: list[str]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    failure_persisted: bool = False
    image_record: Any = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _MmdCacheEntry:
    text: str
    url: str
    relative_url: str
    char_count: int
    contexts: list[Any]
    download_ms: float = 0.0
    parse_ms: float = 0.0
    cached_at: float = 0.0


class _MmdContextCache:
    def __init__(self) -> None:
        self._items: OrderedDict[str, _MmdCacheEntry] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[_MmdCacheEntry]] = {}
        self._lock = asyncio.Lock()

    def get(self, key: str) -> Optional[_MmdCacheEntry]:
        if not key:
            return None
        entry = self._items.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return entry

    def put(self, key: str, entry: _MmdCacheEntry) -> None:
        if not key:
            return
        entry.cached_at = time.monotonic()
        self._items[key] = entry
        self._items.move_to_end(key)
        max_size = max(int(settings.IMAGE_MMD_CONTEXT_CACHE_SIZE or 0), 0)
        while max_size and len(self._items) > max_size:
            self._items.popitem(last=False)
        if max_size == 0:
            self._items.clear()

    def _is_expired(self, entry: _MmdCacheEntry) -> bool:
        ttl = int(settings.IMAGE_MMD_CONTEXT_CACHE_TTL_SEC or 0)
        return bool(ttl > 0 and time.monotonic() - entry.cached_at > ttl)

    async def get_or_create(
        self,
        key: str,
        factory: Callable[[], Awaitable[_MmdCacheEntry]],
    ) -> tuple[_MmdCacheEntry, bool]:
        if not key:
            return await factory(), False
        cached = self.get(key)
        if cached is not None:
            return cached, True

        async with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached, True
            task = self._inflight.get(key)
            created = False
            if task is None:
                task = asyncio.create_task(factory())
                self._inflight[key] = task
                created = True

        try:
            entry = await task
        finally:
            if created:
                async with self._lock:
                    self._inflight.pop(key, None)

        if created:
            self.put(key, entry)
        return entry, not created


class ImageWorker:
    def __init__(
        self,
        *,
        repository: Optional[ImageTaskRepository] = None,
        downloader: Optional[ImageDownloader] = None,
        fulltext_downloader: Optional[FullTextDownloader] = None,
        state_manager: Optional[MongoStateManager] = None,
        feature_providers: Optional[Any] = None,
        feature_pipeline: Optional[Callable[..., Any]] = None,
        feature_pipeline_enabled: Optional[bool] = None,
        image_writer: Optional[Any] = None,
        image_write_enabled: Optional[bool] = None,
        image_collection_name: Optional[str] = None,
        resource_pool: Optional[ImageWorkerResourcePool] = None,
        download_timeout: Optional[int] = None,
    ) -> None:
        self.repository = repository or get_image_task_repository()
        self.state_manager = state_manager or get_mongo_state_manager()
        self.feature_pipeline_enabled = (
            settings.IMAGE_FEATURE_PIPELINE_ENABLED
            if feature_pipeline_enabled is None
            else bool(feature_pipeline_enabled)
        )
        self.image_write_enabled = (
            settings.IMAGE_MILVUS_WRITE_ENABLED
            if image_write_enabled is None
            else bool(image_write_enabled)
        )
        self._owns_resource_pool = resource_pool is None and (
            downloader is None
            or fulltext_downloader is None
            or (
                feature_providers is None
                and feature_pipeline is None
                and self.feature_pipeline_enabled
            )
            or (image_writer is None and self.image_write_enabled)
        )
        self._resource_pool = (
            create_image_worker_resource_pool(
                include_writer=self.image_write_enabled,
                download_timeout=download_timeout,
                download_concurrency=settings.IMAGE_DOWNLOAD_CONCURRENCY,
            )
            if self._owns_resource_pool
            else resource_pool
        )
        self.downloader = (
            downloader
            or (
                self._resource_pool.downloader
                if self._resource_pool is not None
                else ImageDownloader(timeout=download_timeout)
            )
        )
        self.fulltext_downloader = (
            fulltext_downloader
            or (
                self._resource_pool.fulltext_downloader
                if self._resource_pool is not None
                else FullTextDownloader()
            )
        )
        self.feature_providers = feature_providers
        self.feature_pipeline = feature_pipeline
        self.image_writer = (
            image_writer
            or (
                self._resource_pool.writer
                if self._resource_pool is not None
                else None
            )
        )
        self.image_collection_name = image_collection_name
        self._mmd_context_cache = _MmdContextCache()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def close(self) -> None:
        pending = [task for task in self._background_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._owns_resource_pool and self._resource_pool is not None:
            await self._resource_pool.close()

    def stage_executor_status(self) -> dict[str, Any]:
        return {}

    async def process_task(self, task: dict[str, Any]) -> ImageWorkerResult:
        task_started_at = time.perf_counter()
        task_id = str(task.get("task_id") or task.get("_id") or "")
        artifact_refs: list[str] = []
        runtime_batch: Optional[RuntimeAttachmentBatch] = None
        draft_record = None
        failure_persisted = False

        try:
            doc_id = str(task.get("doc_id") or "")
            record_update_time = str(task.get("record_update_time") or "")
            relative_url = str(task.get("relative_url") or "")
            mmd_relative_url = str(task.get("mmd_relative_url") or "").strip()
            image_id = str(task.get("image_id") or "")
            if not task_id or not doc_id or not relative_url or not image_id:
                raise ImageTaskInvalidError(
                    "task_id, doc_id, relative_url, and image_id are required"
                )

            runtime_batch = RuntimeAttachmentBatch.create(
                doc_id=doc_id,
                batch_id=task_id,
                task_id=task_id,
                image_relative_url=relative_url,
                mmd_relative_url=mmd_relative_url or None,
            )
            staged_state: dict[str, Any] = {}

            async def _start_visual_stage(image_result: ImageDownloadResult) -> None:
                """Start image preparation and visual embedding as soon as bytes arrive."""
                staged_state["visual_stage_task"] = asyncio.create_task(
                    self._prepare_and_run_visual_stage(
                        task=task,
                        download=image_result,
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        task_id=task_id,
                        image_id=image_id,
                    )
                )

            use_staged_pipeline = self._uses_staged_default_feature_pipeline()
            download, mmd_download, task, context_status = await self._download_task_resources(
                task=task,
                relative_url=relative_url,
                mmd_relative_url=mmd_relative_url,
                runtime_batch=runtime_batch,
                image_ready_callback=(
                    _start_visual_stage if use_staged_pipeline else None
                ),
            )
            context_status = _ensure_hot_path_context(task, context_status)
            runtime_batch.record_context(**context_status)

            if use_staged_pipeline:
                visual_stage_task = staged_state.get("visual_stage_task")
                if visual_stage_task is None:
                    raise ImageFeaturePipelineError(
                        "visual stage was not started",
                        failure_items=[{
                            "feature_name": "visual_embed",
                            "error_code": "IMAGE_VISUAL_STAGE_NOT_STARTED",
                            "error_message": "image download callback did not start visual stage",
                            "retryable": True,
                        }],
                        retryable=True,
                    )
                staged = await visual_stage_task
                manifest = staged["manifest"]
                manifest_result = staged["manifest_result"]
                dimension_result = staged["dimension_result"]
                oversize_for_model = staged["oversize_for_model"]
                draft_record = staged["draft_record"]
                staged_execution = staged["execution"]
                staged_feature_error = staged.get("visual_error")
            else:
                (
                    manifest,
                    manifest_result,
                    dimension_result,
                    oversize_for_model,
                    draft_record,
                ) = (
                    self._prepare_image_artifacts(
                        task=task,
                        download=download,
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        task_id=task_id,
                    )
                )
                staged_execution = None
                staged_feature_error = None

            # A replayable manifest needs the validated manifest produced by
            # preparation.  The staged and legacy paths both retain it here.
            manifest = _build_replayable_manifest(
                manifest=manifest,
                dimension_result=dimension_result,
                validated_manifest=manifest_result.manifest,
                draft_record=draft_record,
            )
            if mmd_download is not None:
                manifest["mmd_download_url"] = mmd_download.url
            if staged_execution is not None:
                # The visual stage ran against the initial task while MMD was
                # downloading.  Replace only the textual metadata before the
                # text stage so context vectors always match stored fields.
                draft_record.context_before = task.get("context_before")
                draft_record.context_after = task.get("context_after")
                draft_record.img_meta.update(_dict_or_empty(task.get("img_meta")))
                staged_execution.context.draft_record = draft_record
                staged_execution.context.context_text = _feature_context_text(task)

            image_quality = evaluate_image_indexability(
                task,
                context_status,
                policy=settings.IMAGE_TEXT_INDEX_POLICY,
                require_mmd_context=settings.IMAGE_MMD_CONTEXT_REQUIRED_FOR_TEXT_INDEX,
            )
            apply_image_index_quality(
                draft_record,
                image_quality,
                pipeline_version=settings.IMAGE_QUALITY_PIPELINE_VERSION,
            )
            context_status.update(
                {
                    "quality_status": image_quality.status,
                    "quality_reason": image_quality.reason_code,
                    "caption_input_hash": image_quality.caption_input_hash,
                    "context_input_hash": image_quality.context_input_hash,
                }
            )
            manifest_ref: Optional[str] = None
            if _should_save_success_artifacts():
                manifest_ref = await self._save_manifest(
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    task_id=task_id,
                    image_id=image_id,
                    manifest=manifest,
                    success_path=True,
                )
                artifact_refs.append(manifest_ref)

            # Visual retrieval is image-first.  Missing caption or adjacent
            # MMD context only suppresses the corresponding text vectors; it
            # must never prevent visual_embed from being generated and stored.
            draft_record.img_meta["visual_first_processing"] = True
            draft_record.img_meta["text_vectorization_skipped"] = not image_quality.indexable

            feature_summary = None
            feature_result = None
            feature_summary_ref: Optional[str] = None
            if self._feature_pipeline_should_run():
                try:
                    if staged_feature_error is not None:
                        raise staged_feature_error
                    if staged_execution is not None:
                        from app.services.image_feature_pipeline import (
                            finalize_image_feature_execution,
                            run_image_feature_text_stage,
                        )

                        providers = staged_execution.context.scratch.get("providers")
                        if providers is None:
                            raise ImageFeaturePipelineError(
                                "staged visual providers are unavailable",
                                failure_items=[{
                                    "feature_name": "image_provider_config",
                                    "error_code": "IMAGE_PROVIDER_CONFIG_ERROR",
                                    "error_message": "staged visual stage did not retain providers",
                                    "retryable": True,
                                }],
                                retryable=True,
                            )
                        text_timeout = float(settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC or 0)
                        if text_timeout > 0:
                            await asyncio.wait_for(
                                run_image_feature_text_stage(
                                    staged_execution,
                                    providers=providers,
                                ),
                                timeout=text_timeout,
                            )
                        else:
                            await run_image_feature_text_stage(
                                staged_execution,
                                providers=providers,
                            )
                        feature_result = finalize_image_feature_execution(
                            staged_execution
                        )
                    else:
                        feature_result = await self._run_feature_pipeline_with_timeout(
                            image_bytes=download.content,
                            draft_record=draft_record,
                            context_text=(
                                image_quality.context_text
                                if image_quality.status != "UNVERIFIED_LEGACY"
                                else _feature_context_text(task)
                            ),
                        )
                    feature_success = bool(getattr(feature_result, "success", False))
                    failure_items = _feature_failure_items(feature_result)
                except asyncio.TimeoutError:
                    failure_items = [{
                        "feature_name": "context_embed",
                        "error_code": "IMAGE_TEXT_STAGE_TIMEOUT",
                        "error_message": "text embedding stage timed out",
                        "retryable": True,
                    }]
                    feature_result = None
                    feature_success = False
                except ImageFeaturePipelineError as feature_error:
                    feature_result = None
                    feature_success = False
                    failure_items = feature_error.failure_items
                if not feature_success:
                    draft_record.img_meta["failure_stage"] = "feature"
                    if manifest_ref is None:
                        manifest_ref = await self._save_manifest(
                            doc_id=doc_id,
                            record_update_time=record_update_time,
                            task_id=task_id,
                            image_id=image_id,
                            manifest=manifest,
                            success_path=False,
                        )
                        artifact_refs.append(manifest_ref)
                    feature_summary = _build_feature_summary_content(
                        task_id=task_id,
                        image_id=image_id,
                        doc_id=doc_id,
                        feature_success=feature_success,
                        feature_result=feature_result,
                        failure_items=failure_items,
                        artifact_refs=artifact_refs,
                        image_diagnostics=_image_diagnostics(draft_record),
                    )
                    feature_summary_ref = await self._save_feature_summary(
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        task_id=task_id,
                        image_id=image_id,
                        summary=feature_summary,
                        success_path=False,
                    )
                    artifact_refs.append(feature_summary_ref)
                    retryable = _task_failure_retryable(
                        task,
                        _feature_retryable(failure_items),
                    )
                    error_message = _format_feature_failures(failure_items)
                    _finalize_runtime_failure(
                        runtime_batch,
                        error_code="IMAGE_FEATURE_PIPELINE_ERROR",
                        error_message=error_message,
                    )
                    await self.repository.mark_task_failure(
                        task_id,
                        error_code="IMAGE_FEATURE_PIPELINE_ERROR",
                        error_message=error_message,
                        retryable=retryable,
                    )
                    failure_persisted = await self._persist_failure_record(
                        task=task,
                        task_id=task_id,
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        image_id=image_id,
                        artifact_refs=artifact_refs,
                        error=ImageFeaturePipelineError(
                            error_message,
                            failure_items=failure_items,
                            retryable=retryable,
                        ),
                        error_code="IMAGE_FEATURE_PIPELINE_ERROR",
                        error_message=error_message,
                        retryable=retryable,
                    )
                    _schedule_pipeline_status_refresh_best_effort(
                        self.state_manager,
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                    )
                    return ImageWorkerResult(
                        task_id=task_id,
                        success=False,
                        artifact_refs=artifact_refs,
                        error_code="IMAGE_FEATURE_PIPELINE_ERROR",
                        error_message=error_message,
                        retryable=retryable,
                        failure_persisted=failure_persisted,
                    )

                feature_summary = _build_feature_summary_content(
                    task_id=task_id,
                    image_id=image_id,
                    doc_id=doc_id,
                    feature_success=feature_success,
                    feature_result=feature_result,
                    failure_items=failure_items,
                    artifact_refs=artifact_refs,
                    image_diagnostics=_image_diagnostics(draft_record),
                )
                if _should_save_success_artifacts():
                    feature_summary_ref = await self._save_feature_summary(
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        task_id=task_id,
                        image_id=image_id,
                        summary=feature_summary,
                        success_path=True,
                    )
                    artifact_refs.append(feature_summary_ref)

            write_summary = None
            if self._image_write_should_run(feature_result):
                image_record = getattr(feature_result, "image_record", None)
                collection_name = (
                    self.image_collection_name
                    or settings.IMAGE_COLLECTION_NAME
                    or "scientific_images"
                )
                try:
                    write_result = await self._replace_image_by_img_id(
                        collection_name=collection_name,
                        image_record=image_record,
                        draft_record=draft_record,
                    )
                except Exception as write_error:
                    draft_record.img_meta["failure_stage"] = "milvus_write"
                    if manifest_ref is None:
                        manifest_ref = await self._save_manifest(
                            doc_id=doc_id,
                            record_update_time=record_update_time,
                            task_id=task_id,
                            image_id=image_id,
                            manifest=manifest,
                            success_path=False,
                        )
                        artifact_refs.append(manifest_ref)
                    if feature_summary is not None and feature_summary_ref is None:
                        feature_summary_ref = await self._save_feature_summary(
                            doc_id=doc_id,
                            record_update_time=record_update_time,
                            task_id=task_id,
                            image_id=image_id,
                            summary=feature_summary,
                            success_path=False,
                        )
                        artifact_refs.append(feature_summary_ref)
                    retryable = _task_failure_retryable(
                        task,
                        not isinstance(write_error, ValueError),
                    )
                    error_message = str(write_error)
                    _finalize_runtime_failure(
                        runtime_batch,
                        error_code="IMAGE_MILVUS_WRITE_ERROR",
                        error_message=error_message,
                    )
                    write_summary = _build_milvus_write_summary_content(
                        task_id=task_id,
                        image_id=image_id,
                        doc_id=doc_id,
                        collection_name=collection_name,
                        write_success=False,
                        write_result=None,
                        artifact_refs=artifact_refs,
                        failure_details={
                            "error_code": "IMAGE_MILVUS_WRITE_ERROR",
                            "error_message": error_message,
                            "retryable": retryable,
                            "exception_type": type(write_error).__name__,
                        },
                    )
                    write_summary_ref = await self._save_milvus_write_summary(
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        task_id=task_id,
                        image_id=image_id,
                        summary=write_summary,
                        success_path=False,
                    )
                    artifact_refs.append(write_summary_ref)
                    await self.repository.mark_task_failure(
                        task_id,
                        error_code="IMAGE_MILVUS_WRITE_ERROR",
                        error_message=error_message,
                        retryable=retryable,
                    )
                    failure_persisted = await self._persist_failure_record(
                        task=task,
                        task_id=task_id,
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        image_id=image_id,
                        artifact_refs=artifact_refs,
                        error=write_error,
                        error_code="IMAGE_MILVUS_WRITE_ERROR",
                        error_message=error_message,
                        retryable=retryable,
                    )
                    _schedule_pipeline_status_refresh_best_effort(
                        self.state_manager,
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                    )
                    return ImageWorkerResult(
                        task_id=task_id,
                        success=False,
                        artifact_refs=artifact_refs,
                        error_code="IMAGE_MILVUS_WRITE_ERROR",
                        error_message=error_message,
                        retryable=retryable,
                        failure_persisted=failure_persisted,
                    )

                draft_record.img_meta["milvus_batch_id"] = write_result.get("batch_id")
                draft_record.img_meta["milvus_batch_size"] = write_result.get("batch_size")
                write_summary = _build_milvus_write_summary_content(
                    task_id=task_id,
                    image_id=image_id,
                    doc_id=doc_id,
                    collection_name=collection_name,
                    write_success=True,
                    write_result=write_result,
                    artifact_refs=artifact_refs,
                    failure_details=None,
                )
                if _should_save_success_artifacts():
                    write_summary_ref = await self._save_milvus_write_summary(
                        doc_id=doc_id,
                        record_update_time=record_update_time,
                        task_id=task_id,
                        image_id=image_id,
                        summary=write_summary,
                        success_path=True,
                    )
                    artifact_refs.append(write_summary_ref)

            draft_record.img_meta["success_stage"] = (
                "milvus_write"
                if write_summary is not None
                else ("feature" if feature_summary is not None else "download")
            )
            result_summary = _build_success_result_summary(
                task_id=task_id,
                image_id=image_id,
                relative_url=relative_url,
                mmd_relative_url=mmd_relative_url,
                download=download,
                mmd_download=mmd_download,
                manifest=manifest,
                dimension_result=dimension_result,
                oversize_for_model=oversize_for_model,
                artifact_refs=artifact_refs,
                feature_pipeline_enabled=self._feature_pipeline_should_run(),
                context_status=context_status,
                feature_summary=feature_summary,
                write_summary=write_summary,
                draft_record=draft_record,
            )
            if write_summary is not None:
                result_summary["milvus_write_enabled"] = True
                result_summary["milvus_write_success"] = write_summary["write_success"]
                if _should_include_success_write_summary():
                    result_summary["milvus_write_summary"] = write_summary

            await self._mark_task_success(
                task_id=task_id,
                image_id=image_id,
                success_stage=str(draft_record.img_meta.get("success_stage") or ""),
                milvus_write_mode=str(result_summary.get("milvus_write_mode") or ""),
                artifact_refs=artifact_refs,
                result_summary=result_summary,
            )
            _cleanup_runtime_success_best_effort(runtime_batch)
            return ImageWorkerResult(
                task_id=task_id,
                success=True,
                artifact_refs=artifact_refs,
                image_record=(
                    getattr(feature_result, "image_record", None)
                    if feature_result is not None
                    else None
                ),
                timings_ms=_result_timings_ms(draft_record, task_started_at),
            )
        except asyncio.CancelledError:
            _finalize_runtime_failure(
                runtime_batch,
                error_code="IMAGE_WORKER_TASK_CANCELLED",
                error_message="image worker task was cancelled",
            )
            raise
        except Exception as error:
            visual_stage_task = locals().get("staged_state", {}).get("visual_stage_task")
            if visual_stage_task is not None and not visual_stage_task.done():
                visual_stage_task.cancel()
                await asyncio.gather(visual_stage_task, return_exceptions=True)
            error_code, retryable = _classify_worker_error(error)
            retryable = _task_failure_retryable(task, retryable)
            error_message = str(error)
            _finalize_runtime_failure(
                runtime_batch,
                error_code=error_code,
                error_message=error_message,
            )
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
                    doc_id=str(task.get("doc_id") or ""),
                    record_update_time=str(task.get("record_update_time") or ""),
                    image_id=str(task.get("image_id") or ""),
                    artifact_refs=artifact_refs,
                    error=error,
                    error_code=error_code,
                    error_message=error_message,
                    retryable=retryable,
                )
                _schedule_pipeline_status_refresh_best_effort(
                    self.state_manager,
                    doc_id=str(task.get("doc_id") or ""),
                    record_update_time=str(task.get("record_update_time") or ""),
                )
            return ImageWorkerResult(
                task_id=task_id,
                success=False,
                artifact_refs=artifact_refs,
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
                failure_persisted=failure_persisted if task_id else False,
            )

    async def _save_manifest(
        self,
        *,
        doc_id: str,
        record_update_time: str,
        task_id: str,
        image_id: str,
        manifest: dict[str, Any],
        success_path: bool = True,
    ) -> str:
        try:
            return await self.state_manager.save_artifact(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=None,
                stage=SyncStage.IMAGE_DOWNLOAD,
                scope_type=ScopeType.IMAGE,
                scope_id=image_id or task_id,
                artifact_type=ArtifactType.IMAGE_MANIFEST_JSON,
                content=manifest,
                success_path=success_path,
                content_type="application/json",
                meta={
                    "task_id": task_id,
                    "image_id": image_id,
                    "relative_url": manifest.get("relative_url"),
                    "size_bytes": manifest.get("size_bytes"),
                },
            )
        except Exception as error:
            raise ImageArtifactError(str(error)) from error

    def _schedule_success_finalization(
        self,
        *,
        task_id: str,
        image_id: str,
        success_stage: str,
        milvus_write_mode: str,
        artifact_refs: list[str],
        result_summary: dict[str, Any],
        runtime_batch: Optional[RuntimeAttachmentBatch],
        doc_id: str,
        record_update_time: str,
    ) -> bool:
        async def _finalize() -> None:
            try:
                await self._mark_task_success(
                    task_id=task_id,
                    image_id=image_id,
                    success_stage=success_stage,
                    milvus_write_mode=milvus_write_mode,
                    artifact_refs=artifact_refs,
                    result_summary=result_summary,
                )
            except Exception as error:
                logger.error(
                    "image_worker_success_finalization_failed",
                    task_id=task_id,
                    image_id=image_id,
                    error=str(error),
                )
            finally:
                _cleanup_runtime_success_best_effort(runtime_batch)

        task = asyncio.create_task(_finalize())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _mark_task_success(
        self,
        *,
        task_id: str,
        image_id: str,
        success_stage: str,
        milvus_write_mode: str,
        artifact_refs: list[str],
        result_summary: dict[str, Any],
    ) -> bool:
        minimal_write = getattr(self.repository, "mark_task_success_minimal", None)
        if _success_result_mode() == "status_only" and callable(minimal_write):
            await minimal_write(
                task_id,
                image_id=image_id,
                success_stage=success_stage,
                milvus_write_mode=milvus_write_mode,
                artifact_refs=artifact_refs,
            )
            return
        await self.repository.mark_task_success(
            task_id,
            artifact_refs=artifact_refs,
            result_summary=result_summary,
        )

    async def _persist_failure_record(
        self,
        *,
        task: dict[str, Any],
        task_id: str,
        doc_id: str,
        record_update_time: str,
        image_id: str,
        artifact_refs: list[str],
        error: Exception,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> None:
        """Store one replayable failure record for the image target in use."""
        persist = getattr(self.state_manager, "persist_failure", None)
        if not callable(persist):
            return False
        try:
            await persist(
                doc_id=doc_id or None,
                record_update_time=record_update_time or None,
                sub_doc_type=None,
                event_type=str(task.get("event_type") or "UPSERT"),
                failure_stage=_image_failure_stage(error_code),
                scope_type=ScopeType.IMAGE,
                scope_id=image_id or task_id,
                error_type=ErrorType.RETRYABLE if retryable else ErrorType.NON_RETRYABLE,
                error_code=error_code,
                error_message=error_message,
                exception_class=type(error).__name__,
                stacktrace=error_message,
                replay_mode=ReplayMode.FROM_IMAGE_MANIFEST,
                artifact_refs=list(artifact_refs),
                source_event_meta=_image_task_source_event_meta(task, task_id),
                replay_payload=_image_replay_task_payload(task),
                **self._failure_target_binding(
                    source_version=record_update_time,
                    item_id=image_id or task_id,
                ),
            )
            return True
        except Exception as persistence_error:
            # The failed Mongo task is kept even when the retry index cannot be
            # updated, so the primary image failure is never hidden.
            logger.error(
                "image_failure_record_persist_failed",
                task_id=task_id,
                image_id=image_id,
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
        collection_name = str(
            self.image_collection_name or settings.IMAGE_COLLECTION_NAME or "scientific_images"
        ).strip()
        writer = self.image_writer
        service = getattr(writer, "service", writer)
        if service is None and isinstance(self.state_manager, MongoStateManager):
            # Bind an early download/feature failure to the same target the
            # writer would use, without creating or mutating a collection.
            try:
                from app.services.milvus_image_service import MilvusImageService

                service = MilvusImageService()
            except Exception as error:
                logger.warning(
                    "image_failure_target_writer_init_failed",
                    collection_name=collection_name,
                    error=str(error),
                )
        client = getattr(service, "client", None)
        if client is not None:
            try:
                resolve_target = getattr(service, "resolve_collection_target", None)
                if callable(resolve_target):
                    target = resolve_target(collection_name=collection_name)
                else:
                    target = resolve_image_collection_target(
                        client,
                        logical_name=collection_name,
                        schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE,
                        ensure_collection=None,
                    )
                return collection_target_binding(
                    target,
                    pipeline="image",
                    source_version=source_version,
                    item_id=item_id,
                )
            except Exception as error:
                logger.warning(
                    "image_failure_target_resolution_failed",
                    collection_name=collection_name,
                    error=str(error),
                )

        schema_signature = build_image_collection_signature(
            schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE,
        )
        index_signature = build_image_dense_index_spec().signature
        return {
            "pipeline": "image",
            "source_version": str(source_version or ""),
            "item_id": str(item_id or ""),
            "logical_collection": collection_name,
            "resolved_collection": collection_name,
            "target_signature": f"{schema_signature}|INDEX={index_signature}",
        }

    async def _save_feature_summary(
        self,
        *,
        doc_id: str,
        record_update_time: str,
        task_id: str,
        image_id: str,
        summary: dict[str, Any],
        success_path: bool,
    ) -> str:
        try:
            return await self.state_manager.save_artifact(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=None,
                stage=SyncStage.IMAGE_EMBED_VISUAL,
                scope_type=ScopeType.IMAGE,
                scope_id=image_id or task_id,
                artifact_type=ArtifactType.IMAGE_FEATURE_SUMMARY_JSON,
                content=summary,
                success_path=success_path,
                content_type="application/json",
                meta={
                    "task_id": task_id,
                    "image_id": image_id,
                    "feature_success": success_path,
                },
            )
        except Exception as error:
            raise ImageArtifactError(str(error)) from error

    async def _save_milvus_write_summary(
        self,
        *,
        doc_id: str,
        record_update_time: str,
        task_id: str,
        image_id: str,
        summary: dict[str, Any],
        success_path: bool,
    ) -> str:
        try:
            return await self.state_manager.save_artifact(
                doc_id=doc_id,
                record_update_time=record_update_time,
                sub_doc_type=None,
                stage=SyncStage.MILVUS_WRITE_IMAGES,
                scope_type=ScopeType.IMAGE,
                scope_id=image_id or task_id,
                artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
                content=summary,
                success_path=success_path,
                content_type="application/json",
                meta={
                    "task_id": task_id,
                    "image_id": image_id,
                    "collection_name": summary.get("collection_name"),
                    "write_success": success_path,
                },
            )
        except Exception as error:
            raise ImageArtifactError(str(error)) from error

    def _feature_pipeline_should_run(self) -> bool:
        return bool(
            self.feature_pipeline_enabled
            or self.feature_pipeline is not None
            or self.feature_providers is not None
        )

    def _image_write_should_run(self, feature_result: Any) -> bool:
        return bool(
            feature_result is not None
            and getattr(feature_result, "success", False)
            and getattr(feature_result, "image_record", None) is not None
            and (self.image_write_enabled or self.image_writer is not None)
        )

    def _image_milvus_write_mode(self) -> str:
        mode = str(settings.IMAGE_MILVUS_WRITE_MODE or "upsert").strip().lower()
        if settings.LIVE_E2E_NO_DELETE:
            return "insert_only"
        if mode in {"insert", "insert_only"}:
            return "insert_only"
        if mode in {"insert_or_skip_if_exists", "insert_or_skip", "skip_if_exists"}:
            return "insert_or_skip_if_exists"
        if mode in {"upsert", "upsert_if_exists", "replace", "replace_if_exists"}:
            return "upsert_if_exists"
        raise ValueError(
            "IMAGE_MILVUS_WRITE_MODE must be one of insert_only, insert_or_skip_if_exists, or upsert"
        )

    def _mmd_hot_path_mode(self) -> str:
        mode = str(getattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background") or "")
        mode = mode.strip().lower()
        if mode in {"required", "sync", "synchronous"}:
            return "required"
        if mode in {"disabled", "off", "none"}:
            return "disabled"
        return "background"

    async def _replace_image_by_img_id(
        self,
        *,
        collection_name: str,
        image_record: Any,
        draft_record: Any | None = None,
    ) -> dict[str, Any]:
        writer = self.image_writer
        owns_writer = False
        if writer is None:
            from app.services.milvus_image_service import MilvusImageService

            writer = MilvusImageService()
            owns_writer = True
        started = time.perf_counter()
        try:
            write_mode = self._image_milvus_write_mode()
            if write_mode == "insert_only":
                insert_result = writer.insert_images(
                    collection_name,
                    [image_record],
                )
                if inspect.isawaitable(insert_result):
                    insert_result = await insert_result
                result = {
                    "delete_result": {
                        "skipped": True,
                        "status": "DELETE_NOOP_RECORDED",
                        "reason": (
                            "LIVE_E2E_NO_DELETE"
                            if settings.LIVE_E2E_NO_DELETE
                            else "IMAGE_MILVUS_WRITE_MODE_INSERT_ONLY"
                        ),
                        "operation": "replace_image_predelete",
                        "collection_name": collection_name,
                        "img_id": getattr(image_record, "img_id", None),
                    },
                    "insert_result": insert_result,
                    "write_mode": write_mode,
                }
            elif write_mode == "insert_or_skip_if_exists":
                insert_or_skip = getattr(writer, "insert_or_skip_images_by_img_id_batch", None)
                if insert_or_skip is not None:
                    result = insert_or_skip(
                        collection_name,
                        [image_record],
                    )
                else:
                    result = writer.upsert_images_by_img_id_batch(
                        collection_name,
                        [image_record],
                    )
            elif write_mode == "upsert_if_exists":
                result = writer.upsert_images_by_img_id_batch(
                    collection_name,
                    [image_record],
                )
            else:
                result = writer.replace_image_by_img_id(collection_name, image_record)
            if inspect.isawaitable(result):
                result = await result
            target_record = draft_record or image_record
            if target_record is not None and hasattr(target_record, "img_meta"):
                img_meta = getattr(target_record, "img_meta", None)
                if isinstance(img_meta, dict):
                    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                    img_meta["milvus_write_ms"] = elapsed_ms
            return _write_result_dict(result)
        finally:
            if owns_writer:
                await _close_writer_best_effort(writer)

    async def _run_feature_pipeline(
        self,
        *,
        image_bytes: bytes,
        draft_record,
        context_text: str,
    ):
        pipeline = self.feature_pipeline
        if pipeline is None:
            from app.services.image_feature_pipeline import process_image_features

            providers = self._resolve_feature_providers()
            return await process_image_features(
                image_bytes=image_bytes,
                manifest_or_draft=draft_record,
                providers=providers,
                context_text=context_text,
                schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE,
            )

        try:
            providers = self.feature_providers
            result = pipeline(
                image_bytes,
                draft_record,
                providers=providers,
                context_text=context_text,
            )
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as error:
            failure_items = [
                {
                    "feature_name": "image_feature_pipeline",
                    "error_code": "IMAGE_FEATURE_PIPELINE_ERROR",
                    "error_message": str(error),
                    "retryable": True,
                }
            ]
            raise ImageFeaturePipelineError(
                _format_feature_failures(failure_items),
                failure_items=failure_items,
                retryable=True,
            ) from error

    async def _run_feature_pipeline_with_timeout(
        self,
        *,
        image_bytes: bytes,
        draft_record,
        context_text: str,
    ):
        timeout_seconds = float(settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC or 0)
        started = time.perf_counter()
        try:
            if timeout_seconds > 0:
                return await asyncio.wait_for(
                    self._run_feature_pipeline(
                        image_bytes=image_bytes,
                        draft_record=draft_record,
                        context_text=context_text,
                    ),
                    timeout=timeout_seconds,
                )
            return await self._run_feature_pipeline(
                image_bytes=image_bytes,
                draft_record=draft_record,
                context_text=context_text,
            )
        except asyncio.TimeoutError as error:
            failure_items = [
                {
                    "feature_name": "image_feature_pipeline",
                    "error_code": "IMAGE_FEATURE_PIPELINE_TIMEOUT",
                    "error_message": (
                        f"image feature pipeline exceeded {timeout_seconds:g}s timeout"
                    ),
                    "retryable": True,
                }
            ]
            raise ImageFeaturePipelineError(
                _format_feature_failures(failure_items),
                failure_items=failure_items,
                retryable=True,
            ) from error
        finally:
            if draft_record is not None and hasattr(draft_record, "img_meta"):
                img_meta = getattr(draft_record, "img_meta", None)
                if isinstance(img_meta, dict):
                    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                    timings = img_meta.setdefault("feature_stage_timings_ms", {})
                    if isinstance(timings, dict):
                        timings["feature_total"] = elapsed_ms
            image_stage_latency_seconds.labels(stage="feature_total").observe(
                time.perf_counter() - started
            )

    def _uses_staged_default_feature_pipeline(self) -> bool:
        # Only the built-in provider path can use the staged API.  Injected
        # test/extension pipelines retain the historical single-call contract.
        return self.feature_pipeline is None and self._feature_pipeline_should_run()

    def _resolve_feature_providers(self):
        providers = self.feature_providers
        if providers is not None:
            return providers
        try:
            if self._resource_pool is not None:
                return self._resource_pool.get_feature_providers()
            from app.services.image_offline_providers import (
                build_offline_image_feature_providers,
            )

            return build_offline_image_feature_providers()
        except Exception as error:
            failure_items = [
                {
                    "feature_name": "image_provider_config",
                    "error_code": str(
                        getattr(error, "error_code", "IMAGE_PROVIDER_CONFIG_ERROR")
                    ),
                    "error_message": str(error),
                    "retryable": bool(getattr(error, "retryable", False)),
                }
            ]
            raise ImageFeaturePipelineError(
                _format_feature_failures(failure_items),
                failure_items=failure_items,
                retryable=bool(getattr(error, "retryable", False)),
            ) from error

    async def _run_staged_default_feature_pipeline(
        self,
        *,
        image_bytes: bytes,
        draft_record,
        context_text: str,
        providers,
    ):
        from app.services.image_feature_pipeline import process_image_features

        return await process_image_features(
            image_bytes=image_bytes,
            manifest_or_draft=draft_record,
            providers=providers,
            context_text=context_text,
            schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE,
        )

    def _prepare_image_artifacts(
        self,
        *,
        task: dict[str, Any],
        download: ImageDownloadResult,
        doc_id: str,
        record_update_time: str,
        task_id: str,
    ) -> tuple[Any, Any, ImageDimensionValidationResult, bool, Any]:
        """Validate downloaded bytes and build the draft used by all stages."""
        bytes_sha256 = sha256(download.content).hexdigest()
        manifest = _build_manifest(
            task=task,
            download_url=download.url,
            mmd_download_url=None,
            content_type=download.content_type,
            size_bytes=download.size_bytes,
            bytes_sha256=bytes_sha256,
        )
        dimension_result = validate_image_bytes_dimensions(
            download.content,
            max_pixels=settings.IMAGE_MAX_PIXELS,
        )
        oversize_for_model = (
            not dimension_result.success
            and dimension_result.error_code == IMAGE_PIXEL_LIMIT_EXCEEDED
        )
        if not dimension_result.success and not oversize_for_model:
            raise ImageDimensionInvalidError(dimension_result)
        manifest_result = validate_image_manifest(manifest, image_bytes=download.content)
        if not manifest_result.success or manifest_result.manifest is None:
            raise ImageManifestInvalidError(manifest_result)
        draft_record = build_image_record_draft_from_manifest(
            manifest_result,
            dimension_result=dimension_result,
            feature_metadata={
                "name": task.get("name") or "",
                "image_title": task.get("title") or "",
                "doc_meta": _task_doc_meta(
                    task,
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                ),
                "raw_img_meta": task.get("raw_img_meta") or {},
                "context_before": task.get("context_before"),
                "context_after": task.get("context_after"),
                "img_meta": {
                    **_dict_or_empty(task.get("img_meta")),
                    "task_id": task_id,
                    "source_metadata": _source_metadata(task),
                    "source_stage_timings_ms": _source_stage_timings(task),
                    "oversize_for_model": oversize_for_model,
                    "original_pixel_count": dimension_result.pixel_count,
                },
            },
        )
        return manifest, manifest_result, dimension_result, oversize_for_model, draft_record

    async def _prepare_and_run_visual_stage(
        self,
        *,
        task: dict[str, Any],
        download: ImageDownloadResult,
        doc_id: str,
        record_update_time: str,
        task_id: str,
        image_id: str,
    ) -> dict[str, Any]:
        from app.services.image_feature_pipeline import (
            prepare_image_feature_execution,
            run_image_feature_visual_stage,
        )

        (
            manifest,
            manifest_result,
            dimension_result,
            oversize_for_model,
            draft_record,
        ) = self._prepare_image_artifacts(
            task=task,
            download=download,
            doc_id=doc_id,
            record_update_time=record_update_time,
            task_id=task_id,
        )
        draft_record.img_meta["visual_first_processing"] = True
        providers = None
        try:
            providers = self._resolve_feature_providers()
            prepared = prepare_image_feature_execution(
                download.content,
                draft_record,
                providers=providers,
                context_text=_feature_context_text(task),
                schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE,
            )
            if not hasattr(prepared, "context"):
                failure_items = _feature_failure_items(prepared)
                raise ImageFeaturePipelineError(
                    _format_feature_failures(failure_items),
                    failure_items=failure_items,
                    retryable=_feature_retryable(failure_items),
                )
            prepared.context.scratch["providers"] = providers
            visual_timeout = float(settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC or 0)
            if visual_timeout > 0:
                await asyncio.wait_for(
                    run_image_feature_visual_stage(prepared, providers=providers),
                    timeout=visual_timeout,
                )
            else:
                await run_image_feature_visual_stage(prepared, providers=providers)
        except asyncio.TimeoutError as error:
            failure_items = [{
                "feature_name": "visual_embed",
                "error_code": "IMAGE_VISUAL_STAGE_TIMEOUT",
                "error_message": "visual embedding stage timed out",
                "retryable": True,
            }]
            error = ImageFeaturePipelineError(
                _format_feature_failures(failure_items),
                failure_items=failure_items,
                retryable=True,
            )
            return {
                "manifest": manifest,
                "manifest_result": manifest_result,
                "dimension_result": dimension_result,
                "oversize_for_model": oversize_for_model,
                "draft_record": draft_record,
                "execution": None,
                "visual_error": error,
                "image_id": image_id,
            }
        except ImageFeaturePipelineError as error:
            return {
                "manifest": manifest,
                "manifest_result": manifest_result,
                "dimension_result": dimension_result,
                "oversize_for_model": oversize_for_model,
                "draft_record": draft_record,
            "execution": None,
                "visual_error": error,
                "image_id": image_id,
            }
        return {
            "manifest": manifest,
            "manifest_result": manifest_result,
            "dimension_result": dimension_result,
            "oversize_for_model": oversize_for_model,
            "draft_record": draft_record,
            "execution": prepared,
            "visual_error": None,
            "image_id": image_id,
        }

    async def _download_task_resources(
        self,
        *,
        task: dict[str, Any],
        relative_url: str,
        mmd_relative_url: str,
        runtime_batch: RuntimeAttachmentBatch,
        image_ready_callback: Optional[Callable[[ImageDownloadResult], Awaitable[None]]] = None,
    ) -> tuple[
        ImageDownloadResult,
        Optional[FullTextDownloadResult],
        dict[str, Any],
        dict[str, Any],
    ]:
        download_started = time.perf_counter()
        download_lane = _task_download_lane(task)
        image_download_task = asyncio.create_task(
            self._download_image_bytes(relative_url, lane=download_lane)
        )
        mmd_hot_path_mode = self._mmd_hot_path_mode()
        mmd_download_required = bool(mmd_relative_url) and mmd_hot_path_mode == "required"
        mmd_cache_entry: Optional[_MmdCacheEntry] = None
        mmd_cache_shared = False
        mmd_timing_owner = False
        updated_task = dict(task)
        context_status: Optional[dict[str, Any]] = None
        cache_key = ""

        if _task_has_located_mmd_context(updated_task):
            mmd_download_required = False
            image_mmd_cache_total.labels(result="skipped").inc()
            updated_task, context_status = _task_with_existing_mmd_context(updated_task)
        elif mmd_relative_url and mmd_hot_path_mode != "disabled":
            cache_key = _mmd_cache_key(
                mmd_relative_url,
                str(task.get("record_update_time") or ""),
            )
            mmd_cache_entry = self._mmd_context_cache.get(cache_key)
            if mmd_cache_entry is not None:
                mmd_download_required = False
                image_mmd_cache_total.labels(result="hit").inc()
            else:
                image_mmd_cache_total.labels(result="miss").inc()

        mmd_entry_task: Optional[asyncio.Task[Any]] = None
        if mmd_download_required:
            mmd_entry_task = asyncio.create_task(
                self._mmd_context_cache.get_or_create(
                    cache_key,
                    lambda: self._download_and_parse_mmd_contexts(mmd_relative_url),
                )
            )

        # Await the image independently.  The visual callback runs at this
        # point while MMD download/parse continues in mmd_entry_task.
        try:
            image_result = await image_download_task
        except BaseException as image_error:
            if mmd_entry_task is not None and not mmd_entry_task.done():
                mmd_entry_task.cancel()
                await asyncio.gather(mmd_entry_task, return_exceptions=True)
            results: list[Any] = [image_error]
        else:
            if image_ready_callback is not None:
                await image_ready_callback(image_result)
            results = [image_result]
            if mmd_entry_task is not None:
                try:
                    results.append(await mmd_entry_task)
                except BaseException as mmd_error:
                    results.append(mmd_error)
        image_stage_latency_seconds.labels(stage="download").observe(
            time.perf_counter() - download_started
        )
        image_result = results[0]
        mmd_result: Optional[FullTextDownloadResult] = None
        if len(results) > 1 and not isinstance(results[1], BaseException):
            mmd_cache_entry, mmd_cache_shared = results[1]
            # Charge MMD network/parse time once to the task which populated
            # the shared document cache, not once for every figure using it.
            mmd_timing_owner = not mmd_cache_shared
            mmd_result = FullTextDownloadResult(
                relative_url=mmd_cache_entry.relative_url,
                url=mmd_cache_entry.url,
                text=mmd_cache_entry.text,
                char_count=mmd_cache_entry.char_count,
            )

        if isinstance(image_result, ImageDownloadResult):
            runtime_batch.record_image_download(
                relative_url=image_result.relative_url,
                url=image_result.url,
                content=image_result.content,
                content_type=image_result.content_type,
                size_bytes=image_result.size_bytes,
            )
        if mmd_result is not None:
            runtime_batch.record_mmd_download(
                relative_url=mmd_result.relative_url,
                url=mmd_result.url,
                text=mmd_result.text,
                char_count=mmd_result.char_count,
            )

        if isinstance(image_result, BaseException):
            if isinstance(image_result, asyncio.CancelledError):
                raise image_result
            raise image_result

        # In required mode an MMD error is a task failure: no incomplete image
        # record may reach Milvus. Background mode retains its legacy behavior.
        mmd_error: Optional[BaseException] = None
        if len(results) > 1 and isinstance(results[1], BaseException):
            if isinstance(results[1], asyncio.CancelledError):
                raise results[1]
            mmd_error = results[1]
            image_mmd_cache_total.labels(result="error").inc()
        elapsed_ms = round((time.perf_counter() - download_started) * 1000.0, 3)
        img_meta = _dict_or_empty(updated_task.get("img_meta"))
        timings = _dict_or_empty(img_meta.get("feature_stage_timings_ms"))
        timings["download"] = elapsed_ms
        timings[f"download_{download_lane}"] = elapsed_ms
        if mmd_timing_owner and mmd_cache_entry is not None:
            timings["mmd_download"] = round(mmd_cache_entry.download_ms, 3)
            timings["mmd_parse"] = round(mmd_cache_entry.parse_ms, 3)
        img_meta["feature_stage_timings_ms"] = timings
        img_meta["download_lane"] = download_lane
        updated_task["img_meta"] = img_meta

        if mmd_result is not None and mmd_cache_entry is not None:
            if mmd_cache_shared:
                image_mmd_cache_total.labels(result="hit").inc()
            updated_task, context_status = _task_with_mmd_contexts(
                updated_task,
                contexts=mmd_cache_entry.contexts,
                context_available=True,
            )
        elif mmd_cache_entry is not None:
            runtime_batch.record_mmd_download(
                relative_url=mmd_cache_entry.relative_url,
                url=mmd_cache_entry.url,
                text=mmd_cache_entry.text,
                char_count=mmd_cache_entry.char_count,
            )
            updated_task, context_status = _task_with_mmd_contexts(
                updated_task,
                contexts=mmd_cache_entry.contexts,
                context_available=True,
            )
            mmd_result = FullTextDownloadResult(
                relative_url=mmd_cache_entry.relative_url,
                url=mmd_cache_entry.url,
                text=mmd_cache_entry.text,
                char_count=mmd_cache_entry.char_count,
            )
        elif context_status is None:
            updated_task, context_status = _task_with_downloaded_mmd_context(
                updated_task,
                mmd_text="",
            )
            if mmd_relative_url and mmd_hot_path_mode == "background":
                context_status.update(
                    {
                        "mmd_context_available": None,
                        "mmd_context_located": None,
                        "mmd_context_source": "background",
                        "mmd_backfill_status": "pending",
                        "mmd_download_skipped": True,
                    }
                )
                img_meta = _dict_or_empty(updated_task.get("img_meta"))
                img_meta.update(
                    {
                        "mmd_context_located": None,
                        "mmd_context_source": "background",
                        "mmd_backfill_status": "pending",
                        "mmd_relative_url": mmd_relative_url,
                    }
                )
                updated_task["img_meta"] = img_meta

        if mmd_error is not None:
            context_status = dict(context_status or {})
            context_status.update(
                {
                    "mmd_context_available": False,
                    "mmd_context_located": False,
                    "mmd_context_source": "download_error",
                    "mmd_backfill_status": "pending",
                    "mmd_download_error": str(mmd_error),
                }
            )
            img_meta = _dict_or_empty(updated_task.get("img_meta"))
            img_meta.update(
                {
                    "mmd_context_available": False,
                    "mmd_context_located": False,
                    "mmd_context_source": "download_error",
                    "mmd_backfill_status": "pending",
                    "mmd_download_error": str(mmd_error),
                }
            )
            updated_task["img_meta"] = img_meta

            if mmd_hot_path_mode == "required":
                raise ImageMmdContextError(
                    f"required MMD download failed: {mmd_error}",
                    retryable=True,
                ) from mmd_error

        if (
            mmd_hot_path_mode == "required"
            and mmd_relative_url
            and not bool((context_status or {}).get("mmd_context_located"))
        ):
            raise ImageMmdContextError(
                "required MMD was downloaded but the image could not be matched",
                retryable=True,
            )

        return image_result, mmd_result, updated_task, context_status or {}

    async def _download_image_bytes(self, relative_url: str, *, lane: str) -> ImageDownloadResult:
        """Use the slow lane only for replayed fast-path download failures.

        Test and extension downloaders commonly implement the historical
        one-argument protocol, so keep that protocol working for them.
        """
        if isinstance(self.downloader, ImageDownloader):
            return await self.downloader.download_bytes(relative_url, lane=lane)
        return await self.downloader.download_bytes(relative_url)

    async def _download_and_parse_mmd_contexts(
        self,
        mmd_relative_url: str,
    ) -> _MmdCacheEntry:
        download_started = time.perf_counter()
        mmd_result = await self.fulltext_downloader.download_text(mmd_relative_url)
        download_ms = (time.perf_counter() - download_started) * 1000.0
        parse_started = time.perf_counter()
        contexts = extract_mmd_image_contexts(
            mmd_result.text,
            before_window=settings.IMAGE_CONTEXT_BEFORE_PARAGRAPHS,
            after_window=settings.IMAGE_CONTEXT_AFTER_PARAGRAPHS,
        )
        image_stage_latency_seconds.labels(stage="mmd_cache").observe(
            time.perf_counter() - parse_started
        )
        parse_ms = (time.perf_counter() - parse_started) * 1000.0
        return _MmdCacheEntry(
            text=mmd_result.text,
            url=mmd_result.url,
            relative_url=mmd_result.relative_url,
            char_count=mmd_result.char_count,
            contexts=contexts,
            download_ms=download_ms,
            parse_ms=parse_ms,
        )


def build_image_runtime_status(
    *,
    worker_running: bool = False,
    worker_status: Optional[dict[str, Any]] = None,
    task_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    status = {
        **build_image_readiness_status(),
        "worker_enabled": settings.IMAGE_WORKER_ENABLED,
        "worker_running": worker_running,
        "lease_seconds": settings.IMAGE_WORKER_LEASE_SEC,
        "batch_size": settings.IMAGE_WORKER_BATCH_SIZE,
        "concurrency": settings.IMAGE_WORKER_CONCURRENCY,
        "poll_interval_seconds": settings.IMAGE_WORKER_POLL_INTERVAL_SEC,
        "task_timeout_seconds": settings.IMAGE_WORKER_TASK_TIMEOUT_SEC,
        "task_max_attempts": settings.IMAGE_TASK_MAX_ATTEMPTS,
        "download_lanes": {
            "enabled": settings.IMAGE_DOWNLOAD_LANE_ENABLED,
            "fast_timeout_seconds": settings.IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC,
            "slow_timeout_seconds": settings.IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC,
            "slow_concurrency": settings.IMAGE_SLOW_DOWNLOAD_CONCURRENCY,
            "slow_retry_delay_seconds": settings.IMAGE_SLOW_RETRY_DELAY_SEC,
        },
        "milvus_write_enabled": settings.IMAGE_MILVUS_WRITE_ENABLED,
        "vector_schema_mode": settings.IMAGE_VECTOR_SCHEMA_MODE,
        "context_clean_enabled": settings.IMAGE_CONTEXT_CLEAN_ENABLED,
        "feature_timeout": settings.IMAGE_FEATURE_TIMEOUT,
        "feature_total_timeout": settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC,
        "feature_max_retries": settings.IMAGE_FEATURE_MAX_RETRIES,
        "download_max_retries": settings.IMAGE_DOWNLOAD_MAX_RETRIES,
        "mmd_download_timeout": settings.IMAGE_MMD_DOWNLOAD_TIMEOUT,
        "mmd_download_max_retries": settings.IMAGE_MMD_DOWNLOAD_MAX_RETRIES,
        "model_concurrency": {
            "vl_chat": settings.IMAGE_VL_CHAT_CONCURRENCY,
            "vl_embed": settings.IMAGE_VL_EMBED_CONCURRENCY,
            "text_embed": settings.IMAGE_TEXT_EMBED_CONCURRENCY,
        },
        "http_pool": {
            "max_connections": settings.IMAGE_HTTP_POOL_MAX_CONNECTIONS,
            "max_keepalive": settings.IMAGE_HTTP_POOL_MAX_KEEPALIVE,
            "model_max_connections": settings.IMAGE_MODEL_HTTP_POOL_MAX_CONNECTIONS,
            "model_max_keepalive": settings.IMAGE_MODEL_HTTP_POOL_MAX_KEEPALIVE,
        },
        "model_input": {
            "format": settings.IMAGE_MODEL_INPUT_FORMAT,
            "max_pixels": settings.IMAGE_MODEL_INPUT_MAX_PIXELS,
            "max_side": settings.IMAGE_MODEL_INPUT_MAX_SIDE,
            "jpeg_quality": settings.IMAGE_MODEL_INPUT_JPEG_QUALITY,
            "min_jpeg_quality": settings.IMAGE_MODEL_INPUT_MIN_JPEG_QUALITY,
            "max_bytes": settings.IMAGE_MODEL_INPUT_MAX_BYTES,
        },
        "milvus_write": {
            "concurrency": settings.IMAGE_MILVUS_WRITE_CONCURRENCY,
            "batch_size": settings.IMAGE_MILVUS_WRITE_BATCH_SIZE,
            "flush_interval_ms": settings.IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS,
            "mode": settings.IMAGE_MILVUS_WRITE_MODE,
        },
        "mmd_context_cache": {
            "size": settings.IMAGE_MMD_CONTEXT_CACHE_SIZE,
            "ttl_seconds": settings.IMAGE_MMD_CONTEXT_CACHE_TTL_SEC,
        },
        "runtime_attachment_mode": settings.IMAGE_RUNTIME_ATTACHMENT_MODE,
        "success_artifact_mode": settings.IMAGE_SUCCESS_ARTIFACT_MODE,
        "success_result_mode": settings.IMAGE_SUCCESS_RESULT_MODE,
        "stage_executors": {},
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
        if "slow_lane" in worker_status:
            status["slow_lane"] = worker_status["slow_lane"]
    stage_executors = worker_status.get("stage_executors") if worker_status else None
    if isinstance(stage_executors, dict):
        status["stage_executors"] = stage_executors
    return status


def _build_manifest(
    *,
    task: dict[str, Any],
    download_url: str,
    mmd_download_url: Optional[str],
    content_type: str,
    size_bytes: int,
    bytes_sha256: str,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or task.get("_id") or ""),
        "image_id": str(task.get("image_id") or ""),
        "doc_id": str(task.get("doc_id") or ""),
        "record_update_time": str(task.get("record_update_time") or ""),
        "relative_url": str(task.get("relative_url") or ""),
        "mmd_relative_url": task.get("mmd_relative_url") or None,
        "download_url": download_url,
        "mmd_download_url": mmd_download_url,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "bytes_sha256": bytes_sha256,
        "sequence": task.get("sequence"),
        "title": task.get("title") or "",
        "name": task.get("name") or "",
        "width": task.get("width"),
        "height": task.get("height"),
        "file_type": task.get("file_type") or "",
        "doc_metadata": _task_doc_meta(
            task,
            doc_id=str(task.get("doc_id") or ""),
            record_update_time=str(task.get("record_update_time") or ""),
        ),
        "raw_img_meta": task.get("raw_img_meta") or {},
        "img_meta": task.get("img_meta") or {},
        "context_before": task.get("context_before"),
        "context_after": task.get("context_after"),
        "source_metadata": _source_metadata(task),
    }


def _build_replayable_manifest(
    *,
    manifest: dict[str, Any],
    dimension_result: ImageDimensionValidationResult,
    validated_manifest,
    draft_record,
) -> dict[str, Any]:
    replayable = dict(manifest)
    replayable["dimension_validation"] = asdict(dimension_result)
    replayable["validated_manifest"] = asdict(validated_manifest)
    replayable["draft_record_metadata"] = _draft_record_metadata(draft_record)
    return replayable


def _draft_record_metadata(draft_record) -> dict[str, Any]:
    return {
        "img_id": draft_record.img_id,
        "image_id": draft_record.imageId,
        "sysId": draft_record.sysId,
        "doc_id": draft_record.docId,
        "url": draft_record.url,
        "title": draft_record.title,
        "keyword": draft_record.keyword,
        "accessType": draft_record.accessType,
        "classificationCode": draft_record.classificationCode,
        "year": draft_record.year,
        "hasAuthorname": draft_record.hasAuthorname,
        "hasCreateOrganizationname": draft_record.hasCreateOrganizationname,
        "language": draft_record.language,
        "sysDataStatus": draft_record.sysDataStatus,
        "hasSourcesysSourceId": draft_record.hasSourcesysSourceId,
        "hasSourceidSet": draft_record.hasSourceidSet,
        "name": draft_record.name,
        "width": draft_record.width,
        "height": draft_record.height,
        "sequence": draft_record.sequence,
        "file_type": draft_record.fileType,
        "size": draft_record.size,
        "raw_img_meta": draft_record.raw_img_meta,
        "text": draft_record.text,
        "context_before": draft_record.context_before,
        "context_after": draft_record.context_after,
        "doc_meta": draft_record.doc_meta,
        "img_meta": draft_record.img_meta,
    }


def _source_metadata(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "sequence",
        "title",
        "name",
        "mmd_relative_url",
        "width",
        "height",
        "file_type",
        "size",
        "source_item_index",
        "image_index",
    )
    return {key: task.get(key) for key in keys if key in task}


def _task_doc_meta(
    task: dict[str, Any],
    *,
    doc_id: str,
    record_update_time: str,
) -> dict[str, Any]:
    doc_meta = {
        **_dict_or_empty(task.get("doc_meta")),
        "sysId": doc_id,
        "doc_id": doc_id,
        "record_update_time": record_update_time,
    }
    for key, task_key in (
        ("title", "doc_title"),
        ("keyword", "keyword"),
        ("accessType", "accessType"),
        ("classificationCode", "classificationCode"),
        ("year", "year"),
        ("hasAuthorname", "hasAuthorname"),
        ("hasCreateOrganizationname", "hasCreateOrganizationname"),
        ("language", "language"),
        ("sysDataStatus", "sysDataStatus"),
        ("hasSourcesysSourceId", "hasSourcesysSourceId"),
        ("hasSourceidSet", "hasSourceidSet"),
    ):
        value = task.get(task_key)
        if value not in (None, "", [], {}):
            doc_meta[key] = value
    return {
        key: value
        for key, value in doc_meta.items()
        if value not in (None, "", [], {})
    }


def _task_with_downloaded_mmd_context(
    task: dict[str, Any],
    *,
    mmd_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(task)
    caption = _caption_from_task(updated)
    img_meta = _dict_or_empty(updated.get("img_meta"))
    if caption:
        img_meta["caption"] = caption
    updated["img_meta"] = img_meta

    if not str(mmd_text or "").strip():
        return updated, {
            "mmd_context_available": False,
            "mmd_context_located": None,
            "caption_chars": len(caption),
            "context_before_chars": len(str(updated.get("context_before") or "")),
            "context_after_chars": len(str(updated.get("context_after") or "")),
        }

    contexts = extract_mmd_image_contexts(
        mmd_text,
        before_window=settings.IMAGE_CONTEXT_BEFORE_PARAGRAPHS,
        after_window=settings.IMAGE_CONTEXT_AFTER_PARAGRAPHS,
    )
    return _task_with_mmd_contexts(
        updated,
        contexts=contexts,
        context_available=True,
    )


def _task_with_mmd_contexts(
    task: dict[str, Any],
    *,
    contexts: list[Any],
    context_available: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(task)
    caption = _caption_from_task(updated)
    img_meta = _dict_or_empty(updated.get("img_meta"))
    if caption:
        img_meta["caption"] = caption
    updated["img_meta"] = img_meta

    context = find_mmd_context_for_image(
        contexts,
        image_name=str(updated.get("name") or ""),
        relative_url=str(updated.get("relative_url") or ""),
    )
    if context is None:
        if settings.IMAGE_ALLOW_DOCUMENT_TITLE_CONTEXT_FALLBACK:
            updated["context_before"] = _preferred_context_before(
                updated.get("context_before"),
                None,
                _hot_path_context_before(updated),
            )
            updated["context_after"] = _preferred_context_after(
                updated.get("context_after"),
                None,
                caption,
                _hot_path_context_after(updated, caption),
            )
        img_meta["mmd_context_located"] = False
        return updated, {
            "mmd_context_available": context_available,
            "mmd_context_located": False,
            "caption_chars": len(caption),
            "context_before_chars": len(str(updated.get("context_before") or "")),
            "context_after_chars": len(str(updated.get("context_after") or "")),
            "candidate_contexts": len(contexts),
        }

    if str(getattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background") or "").strip().lower() == "required":
        # Required mode must persist the actual MMD boundaries.  In
        # particular, do not turn the document title into fake before-context.
        updated["context_before"] = context.context_before or None
        updated["context_after"] = compose_context_after(
            caption,
            context.context_after or "",
        )
    else:
        updated["context_before"] = _preferred_context_before(
            context.context_before,
            updated.get("context_before"),
            _hot_path_context_before(updated),
        )
        updated["context_after"] = _preferred_context_after(
            join_context_parts(caption, context.context_after or ""),
            updated.get("context_after"),
            caption,
            _hot_path_context_after(updated, caption),
        )
    img_meta.update(context.img_meta)
    img_meta["mmd_context_located"] = True
    if caption:
        img_meta["caption"] = caption
    return updated, {
        "mmd_context_available": context_available,
        "mmd_context_located": True,
        "caption_chars": len(caption),
        "context_before_chars": len(str(updated.get("context_before") or "")),
        "context_after_chars": len(str(updated.get("context_after") or "")),
        "mmd_image_path": context.image_path,
        "mmd_paragraph_index": context.paragraph_index,
    }


def _task_has_located_mmd_context(task: dict[str, Any]) -> bool:
    img_meta = _dict_or_empty(task.get("img_meta"))
    has_context = bool(
        str(task.get("context_before") or "").strip()
        or str(task.get("context_after") or "").strip()
    )
    return bool(img_meta.get("mmd_context_located") is True and has_context)


def _task_with_existing_mmd_context(
    task: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(task)
    caption = _caption_from_task(updated)
    img_meta = _dict_or_empty(updated.get("img_meta"))
    if caption:
        img_meta["caption"] = caption
    img_meta["mmd_context_located"] = True
    updated["img_meta"] = img_meta
    return updated, {
        "mmd_context_available": True,
        "mmd_context_located": True,
        "mmd_context_source": "task",
        "mmd_download_skipped": True,
        "caption_chars": len(caption),
        "context_before_chars": len(str(updated.get("context_before") or "")),
        "context_after_chars": len(str(updated.get("context_after") or "")),
    }


def _ensure_hot_path_context(
    task: dict[str, Any],
    context_status: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(task)
    img_meta = _dict_or_empty(updated.get("img_meta"))
    caption = _caption_from_task(updated)
    if caption:
        img_meta["caption"] = caption
    hot_mode = str(getattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background") or "").strip().lower()
    if hot_mode != "required" and settings.IMAGE_ALLOW_DOCUMENT_TITLE_CONTEXT_FALLBACK:
        updated["context_before"] = _preferred_context_before(
            updated.get("context_before"),
            None,
            _hot_path_context_before(updated),
        )
        updated["context_after"] = _preferred_context_after(
            updated.get("context_after"),
            None,
            caption,
            _hot_path_context_after(updated, caption),
        )

    if hot_mode == "required":
        context_status = dict(context_status)
        context_status.setdefault("mmd_context_available", True)
        context_status.setdefault("mmd_context_located", bool(caption))
        context_status.setdefault("mmd_context_source", "required")
    else:
        context_status = dict(context_status)
        context_status.setdefault("mmd_context_available", False)
        context_status.setdefault("mmd_context_located", False)
        context_status.setdefault("mmd_context_source", "hot_path_fallback")
        context_status.setdefault("mmd_backfill_status", "pending")
    context_status["caption_chars"] = len(caption)
    context_status["context_before_chars"] = len(str(updated.get("context_before") or ""))
    context_status["context_after_chars"] = len(str(updated.get("context_after") or ""))
    img_meta.update(
        {
            "mmd_context_located": context_status.get("mmd_context_located"),
            "mmd_context_source": (
                "hot_path_fallback"
                if hot_mode != "required" and context_status.get("mmd_backfill_status") == "pending"
                else context_status.get("mmd_context_source")
            ),
            "mmd_backfill_status": context_status.get("mmd_backfill_status"),
        }
    )
    updated["img_meta"] = img_meta
    task.update(updated)
    return context_status


def _hot_path_context_before(task: dict[str, Any]) -> str:
    return first_canonical_text(
        task.get("context_before"),
        task.get("doc_title"),
    ) or ""


def _hot_path_context_after(task: dict[str, Any], caption: str) -> str:
    return first_canonical_text(
        task.get("context_after"),
        caption,
    ) or ""


def _mmd_cache_key(mmd_relative_url: str, record_update_time: str) -> str:
    return f"{str(mmd_relative_url or '').strip()}::{str(record_update_time or '').strip()}"


def _caption_from_task(task: dict[str, Any]) -> str:
    title = canonicalize_context_text(task.get("title"))
    if title:
        return title
    img_meta = task.get("img_meta")
    if isinstance(img_meta, dict):
        caption = canonicalize_context_text(img_meta.get("caption"))
        if caption:
            return caption
    raw_img_meta = task.get("raw_img_meta")
    if isinstance(raw_img_meta, dict):
        for key in ("title", "caption", "figureTitle", "imageTitle"):
            caption = canonicalize_context_text(raw_img_meta.get(key))
            if caption:
                return caption
    return _default_caption()


def _default_caption() -> str:
    return str(settings.IMAGE_DEFAULT_CAPTION or "No caption").strip() or "No caption"


def _task_download_lane(task: dict[str, Any]) -> str:
    if not bool(getattr(settings, "IMAGE_DOWNLOAD_LANE_ENABLED", False)):
        return "fast"
    lane = str(task.get("_download_lane") or "fast").strip().lower()
    return "slow" if lane == "slow" else "fast"


def _source_stage_timings(task: dict[str, Any]) -> dict[str, float]:
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


def _classify_worker_error(error: Exception) -> tuple[str, bool]:
    if isinstance(error, ImageConfigurationError):
        return "IMAGE_CONFIG_ERROR", False
    if isinstance(error, FullTextConfigurationError):
        return "IMAGE_MMD_CONFIG_ERROR", False
    if isinstance(error, FullTextDocumentTooLarge):
        return "IMAGE_MMD_TOO_LARGE", False
    if isinstance(
        error,
        (
            FullTextRetryableDownloadError,
            FullTextNonRetryableDownloadError,
        ),
    ):
        return "IMAGE_MMD_DOWNLOAD_FAILED", True
    if isinstance(error, ImageRetryableDownloadError):
        return "IMAGE_DOWNLOAD_RETRYABLE", True
    if isinstance(error, ImageNonRetryableDownloadError):
        return "IMAGE_DOWNLOAD_NON_RETRYABLE", False
    if isinstance(error, ImageArtifactError):
        return "IMAGE_ARTIFACT_ERROR", True
    if isinstance(error, ImageTaskInvalidError):
        return "IMAGE_TASK_INVALID", False
    if isinstance(error, ImageMmdContextError):
        return "IMAGE_MMD_CONTEXT_REQUIRED", bool(error.retryable)
    if isinstance(error, ImageDimensionInvalidError):
        return error.result.error_code or "IMAGE_DIMENSION_DECODE_ERROR", False
    if isinstance(error, ImageManifestInvalidError):
        return "IMAGE_MANIFEST_VALIDATION_FAILED", False
    if isinstance(error, ImageFeaturePipelineError):
        return "IMAGE_FEATURE_PIPELINE_ERROR", error.retryable
    return "IMAGE_WORKER_ERROR", False


def _image_failure_stage(error_code: str) -> SyncStage:
    if error_code.startswith("IMAGE_DOWNLOAD") or error_code.startswith("IMAGE_MMD"):
        return SyncStage.IMAGE_DOWNLOAD
    if error_code in {
        "IMAGE_TASK_INVALID",
        "IMAGE_MANIFEST_VALIDATION_FAILED",
        "IMAGE_DIMENSION_DECODE_ERROR",
        IMAGE_PIXEL_LIMIT_EXCEEDED,
    }:
        return SyncStage.IMAGE_DISCOVER
    if error_code == "IMAGE_FEATURE_PIPELINE_ERROR":
        return SyncStage.IMAGE_EMBED_VISUAL
    if error_code == "IMAGE_MILVUS_WRITE_ERROR":
        return SyncStage.MILVUS_WRITE_IMAGES
    return SyncStage.IMAGE_DISCOVER


def _image_task_source_event_meta(task: dict[str, Any], task_id: str) -> SourceEventMeta:
    raw_meta = task.get("source_event_meta")
    if isinstance(raw_meta, SourceEventMeta):
        return raw_meta
    if isinstance(raw_meta, dict):
        return SourceEventMeta(
            topic=str(raw_meta.get("topic") or "mongo-image-task"),
            partition=_as_int(raw_meta.get("partition"), -1),
            offset=_as_int(raw_meta.get("offset"), -1),
            event_fingerprint=str(raw_meta.get("event_fingerprint") or f"image-task:{task_id}"),
        )
    return SourceEventMeta(
        topic=str(task.get("source_topic") or "mongo-image-task"),
        partition=_as_int(task.get("source_partition"), -1),
        offset=_as_int(task.get("source_offset"), -1),
        event_fingerprint=str(task.get("event_fingerprint") or f"image-task:{task_id}"),
    )


def _image_replay_task_payload(task: dict[str, Any]) -> dict[str, Any]:
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


def _format_manifest_failures(
    failures: list[ImageManifestValidationFailure],
) -> str:
    if not failures:
        return "image manifest validation failed"
    return "; ".join(
        f"{failure.field_name}:{failure.error_code}:{failure.error_message}"
        for failure in failures
    )


def _feature_context_text(task: dict[str, Any]) -> str:
    return join_context_parts(
        task.get("context_before"),
        task.get("context_after"),
        separator="\n",
    ) or ""


def _preferred_context_before(*values: Any) -> Optional[str]:
    return first_canonical_text(*values)


def _preferred_context_after(*values: Any) -> Optional[str]:
    return first_canonical_text(*values)


def _build_feature_summary_content(
    *,
    task_id: str,
    image_id: str,
    doc_id: str,
    feature_success: bool,
    feature_result,
    failure_items: list[dict[str, Any]],
    artifact_refs: list[str],
    image_diagnostics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "image_id": image_id,
        "doc_id": doc_id,
        "feature_success": feature_success,
        "feature_summary": _feature_summary_dict(feature_result),
        "failure_items": failure_items,
        "image_diagnostics": image_diagnostics or {},
        "artifact_refs": list(artifact_refs),
    }


def _feature_summary_dict(feature_result) -> dict[str, Any]:
    if feature_result is None:
        return {}
    summary = getattr(feature_result, "feature_summary", None)
    if summary is None:
        return {}
    if is_dataclass(summary):
        return asdict(summary)
    if isinstance(summary, dict):
        return dict(summary)
    return {
        key: value
        for key, value in vars(summary).items()
        if not key.startswith("_")
    }


def _feature_failure_items(feature_result) -> list[dict[str, Any]]:
    if feature_result is None:
        return []
    items = getattr(feature_result, "failure_items", None) or []
    return [_feature_failure_item_dict(item) for item in items]


def _feature_failure_item_dict(item) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return {
        "feature_name": str(getattr(item, "feature_name", "image_feature_pipeline")),
        "error_code": str(getattr(item, "error_code", "IMAGE_FEATURE_PIPELINE_ERROR")),
        "error_message": str(getattr(item, "error_message", "")),
        "retryable": bool(getattr(item, "retryable", False)),
    }


def _image_diagnostics(draft_record) -> dict[str, Any]:
    img_meta = getattr(draft_record, "img_meta", None)
    if not isinstance(img_meta, dict):
        return {}
    keys = (
        "context_cleaning_fallback",
        "context_cleaning_error",
        "context_cleaning_preview",
        "context_cleaning_raw_length",
        "draft_chat_ms",
        "embed_text_ms",
        "vl_context_chars",
        "vl_context_truncated",
        "caption_chars",
        "context_before_chars",
        "context_after_chars",
        "analysis_error_cached",
        "analysis_error_code",
        "analysis_error_message",
        "data_url_cache_hit",
        "data_url_chars",
        "model_input_format",
        "model_input_max_pixels",
        "feature_stage_timings_ms",
        "milvus_write_ms",
        "text_embedding_batch_fallback",
        "text_embedding_batch_fallback_features",
        "text_embedding_batch_fallback_error",
        "text_vectorization_failures",
        "text_vectorization_skipped",
        "visual_first_processing",
        "milvus_batch_id",
        "milvus_batch_size",
        "mmd_backfill_status",
        "mmd_context_source",
        "mmd_download_error",
        "success_stage",
        "failure_stage",
        "oversize_for_model",
        "original_pixel_count",
    )
    diagnostics = {
        key: img_meta.get(key)
        for key in keys
        if key in img_meta and img_meta.get(key) not in (None, "", [], {})
    }
    if "caption_chars" not in diagnostics:
        caption = str(img_meta.get("caption") or "").strip()
        if caption:
            diagnostics["caption_chars"] = len(caption)
    if "context_before_chars" not in diagnostics:
        diagnostics["context_before_chars"] = len(
            str(getattr(draft_record, "context_before", None) or "")
        )
    if "context_after_chars" not in diagnostics:
        diagnostics["context_after_chars"] = len(
            str(getattr(draft_record, "context_after", None) or "")
        )
    return {
        key: value
        for key, value in diagnostics.items()
        if value not in (None, "", [], {})
    }


def _result_timings_ms(draft_record, task_started_at: float) -> dict[str, float]:
    diagnostics = _image_diagnostics(draft_record)
    source = diagnostics.get("feature_stage_timings_ms")
    timings: dict[str, float] = {}
    if isinstance(source, dict):
        for key, value in source.items():
            try:
                timings[str(key)] = round(float(value), 3)
            except (TypeError, ValueError):
                continue
    source_timings = diagnostics.get("source_stage_timings_ms")
    if isinstance(source_timings, dict):
        for key, value in source_timings.items():
            try:
                timings[str(key)] = round(max(float(value), 0.0), 3)
            except (TypeError, ValueError):
                continue
    try:
        milvus_write_ms = float(diagnostics.get("milvus_write_ms"))
    except (TypeError, ValueError):
        milvus_write_ms = 0.0
    if milvus_write_ms > 0:
        timings["milvus_write"] = round(milvus_write_ms, 3)
    timings["total"] = round((time.perf_counter() - task_started_at) * 1000.0, 3)
    return timings


def _success_result_mode() -> str:
    return str(settings.IMAGE_SUCCESS_RESULT_MODE or "full").strip().lower()


def _should_include_success_write_summary() -> bool:
    return _success_result_mode() not in {"minimal", "min", "status_only"}


def _build_success_result_summary(
    *,
    task_id: str,
    image_id: str,
    relative_url: str,
    mmd_relative_url: Optional[str],
    download,
    mmd_download,
    manifest: dict[str, Any],
    dimension_result,
    oversize_for_model: bool,
    artifact_refs: list[str],
    feature_pipeline_enabled: bool,
    context_status: dict[str, Any],
    feature_summary: Optional[dict[str, Any]],
    write_summary: Optional[dict[str, Any]],
    draft_record,
) -> dict[str, Any]:
    diagnostics = _image_diagnostics(draft_record)
    success_stage = diagnostics.get("success_stage")
    milvus_write_mode = (
        (write_summary or {}).get("write_mode")
        or str(settings.IMAGE_MILVUS_WRITE_MODE or "replace").strip().lower()
    )
    success_mode = _success_result_mode()
    if success_mode in {"status_only"}:
        summary = {
            "task_id": task_id,
            "image_id": image_id,
            "success_stage": success_stage,
            "milvus_write_mode": milvus_write_mode,
        }
        if write_summary is not None:
            summary["milvus_write_success"] = write_summary["write_success"]
        return {
            key: value
            for key, value in summary.items()
            if value not in (None, "", [], {})
        }
    if success_mode in {"minimal", "min"}:
        summary = {
            "task_id": task_id,
            "image_id": image_id,
            "relative_url": relative_url,
            "mmd_relative_url": mmd_relative_url or None,
            "bytes_sha256": manifest["bytes_sha256"],
            "feature_pipeline_enabled": feature_pipeline_enabled,
            "feature_success": (
                None if feature_summary is None else feature_summary["feature_success"]
            ),
            "success_stage": success_stage,
            "milvus_write_mode": milvus_write_mode,
        }
        if write_summary is not None:
            summary["milvus_write_success"] = write_summary["write_success"]
        return {
            key: value
            for key, value in summary.items()
            if value not in (None, "", [], {})
        }

    return {
        "image_id": image_id,
        "relative_url": relative_url,
        "mmd_relative_url": mmd_relative_url or None,
        "download_url": download.url,
        "mmd_download_url": (
            mmd_download.url if mmd_download is not None else None
        ),
        "content_type": download.content_type,
        "size_bytes": download.size_bytes,
        "bytes_sha256": manifest["bytes_sha256"],
        "width": dimension_result.width,
        "height": dimension_result.height,
        "pixel_count": dimension_result.pixel_count,
        "image_format": dimension_result.image_format,
        "oversize_for_model": oversize_for_model,
        "artifact_refs": artifact_refs,
        "feature_pipeline_enabled": feature_pipeline_enabled,
        "mmd_context_status": context_status,
        "feature_success": (
            None if feature_summary is None else feature_summary["feature_success"]
        ),
        "feature_summary": (
            None if feature_summary is None else feature_summary["feature_summary"]
        ),
        "success_stage": success_stage,
        "milvus_write_mode": milvus_write_mode,
        "image_diagnostics": diagnostics,
    }


def _feature_retryable(failure_items: list[dict[str, Any]]) -> bool:
    return any(bool(item.get("retryable")) for item in failure_items)


def _task_failure_retryable(task: dict[str, Any], retryable: bool) -> bool:
    if not retryable:
        return False
    max_attempts = max(int(settings.IMAGE_TASK_MAX_ATTEMPTS or 1), 1)
    attempt_count = int(task.get("attempt_count") or 1)
    return attempt_count < max_attempts


def _should_save_success_artifacts() -> bool:
    # MongoDB is a failure store. Successful image artifacts are intentionally
    # omitted even if older deployments retain a historical mode setting.
    return False


def _format_feature_failures(failure_items: list[dict[str, Any]]) -> str:
    if not failure_items:
        return "image feature pipeline failed"
    return "; ".join(
        f"{item.get('feature_name', 'image_feature_pipeline')}:"
        f"{item.get('error_code', 'IMAGE_FEATURE_PIPELINE_ERROR')}:"
        f"{item.get('error_message', '')}"
        for item in failure_items
    )


def _sanitize_for_mongo(obj: Any) -> Any:
    return sanitize_for_mongo(obj)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return _first_text(value.get("value"))
        for key in ("title", "caption", "name"):
            text = _first_text(value.get(key))
            if text:
                return text
    return ""


async def _refresh_pipeline_status_best_effort(
    state_manager: Any,
    *,
    doc_id: str,
    record_update_time: str,
) -> None:
    refresh = getattr(state_manager, "refresh_async_pipeline_status", None)
    if refresh is None or not doc_id:
        return
    key = (id(state_manager), doc_id, record_update_time)
    current_task = asyncio.current_task()
    if current_task is None:
        return
    wait_for_task: Optional[asyncio.Task[Any]] = None
    try:
        async with _REFRESH_INFLIGHT_LOCK:
            task = _REFRESH_INFLIGHT.get(key)
            if task is None or task.done():
                _REFRESH_INFLIGHT[key] = current_task
            else:
                wait_for_task = task
        if wait_for_task is not None:
            await wait_for_task
            return
        await _run_pipeline_status_refresh(
            refresh,
            doc_id=doc_id,
            record_update_time=record_update_time,
        )
    except Exception:
        return
    finally:
        async with _REFRESH_INFLIGHT_LOCK:
            if _REFRESH_INFLIGHT.get(key) is current_task:
                _REFRESH_INFLIGHT.pop(key, None)


def _schedule_pipeline_status_refresh_best_effort(
    state_manager: Any,
    *,
    doc_id: str,
    record_update_time: str,
) -> None:
    if not doc_id:
        return
    try:
        asyncio.create_task(
            _refresh_pipeline_status_best_effort(
                state_manager,
                doc_id=doc_id,
                record_update_time=record_update_time,
            )
        )
    except Exception:
        return


async def _run_pipeline_status_refresh(
    refresh: Any,
    *,
    doc_id: str,
    record_update_time: str,
) -> None:
    started = time.perf_counter()
    try:
        await refresh(doc_id=doc_id, record_update_time=record_update_time)
    finally:
        image_stage_latency_seconds.labels(stage="mongo_status_refresh").observe(
            time.perf_counter() - started
        )


def _finalize_runtime_failure(
    runtime_batch: Optional[RuntimeAttachmentBatch],
    *,
    error_code: str,
    error_message: str,
) -> None:
    if runtime_batch is None:
        return
    try:
        runtime_batch.mark_failure(
            error_code=error_code,
            error_message=error_message,
        )
    except Exception:
        return


def _cleanup_runtime_success_best_effort(
    runtime_batch: Optional[RuntimeAttachmentBatch],
) -> None:
    return


def _build_milvus_write_summary_content(
    *,
    task_id: str,
    image_id: str,
    doc_id: str,
    collection_name: str,
    write_success: bool,
    write_result: Optional[dict[str, Any]],
    artifact_refs: list[str],
    failure_details: Optional[dict[str, Any]],
) -> dict[str, Any]:
    result = write_result or {}
    return {
        "task_id": task_id,
        "image_id": image_id,
        "doc_id": doc_id,
        "collection_name": collection_name,
        "write_success": write_success,
        "delete_result": _sanitize_for_mongo(result.get("delete_result")),
        "insert_result": _sanitize_for_mongo(result.get("insert_result")),
        "artifact_refs": list(artifact_refs),
        "failure_details": failure_details,
    }


def _write_result_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return dict(result)
    return {"result": str(result)}


async def _close_writer_best_effort(writer: Any) -> None:
    close = getattr(writer, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        return
