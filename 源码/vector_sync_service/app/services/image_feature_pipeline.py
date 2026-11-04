"""Injectable image feature pipeline for fake-provider testing.

This module only orchestrates caller-injected providers. It does not construct
live model clients and does not write to Milvus.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field as dataclass_field
import inspect
import math
from typing import Any, Awaitable, Callable, Optional

from app.config import settings
from app.models.image import ImageRecord
from app.services.image_features import (
    ImageManifestValidationResult,
    ValidatedImageManifest,
    source_image_id,
    build_image_record_draft_from_manifest,
)
from app.services.milvus_image_service import (
    IMAGE_SCHEMA_MODE_FOUR_VECTOR,
    IMAGE_SCHEMA_MODE_CAPTION_CONTEXT_2,
    IMAGE_SCHEMA_MODE_HISTORY_V2,
    IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    TEXT_VECTOR_DIM,
    VISUAL_TEXT_VECTOR_DIM,
    VISUAL_VECTOR_DIM,
    normalize_image_schema_mode,
    validate_writable_image_record,
)


IMAGE_FEATURE_PROVIDER_MISSING = "IMAGE_FEATURE_PROVIDER_MISSING"
IMAGE_FEATURE_PROVIDER_ERROR = "IMAGE_FEATURE_PROVIDER_ERROR"
IMAGE_FEATURE_PHASH_INVALID = "IMAGE_FEATURE_PHASH_INVALID"
IMAGE_FEATURE_VECTOR_INVALID = "IMAGE_FEATURE_VECTOR_INVALID"
IMAGE_FEATURE_SPARSE_INVALID = "IMAGE_FEATURE_SPARSE_INVALID"
IMAGE_FEATURE_TEXT_MISSING = "IMAGE_FEATURE_TEXT_MISSING"
IMAGE_RECORD_NOT_WRITABLE = "IMAGE_RECORD_NOT_WRITABLE"
IMAGE_MANIFEST_INVALID = "IMAGE_MANIFEST_INVALID"
RESERVED_IMAGE_TEXT_PLACEHOLDER = "__reserved_image_text__"

_PHASH_BYTES = 8
_RESERVED_VECTOR_SENTINEL = 1e-6

ProviderCallable = Callable[["ImageFeaturePipelineContext"], Any | Awaitable[Any]]


@dataclass(slots=True)
class ImageFeatureProviders:
    p_hash_provider: Optional[ProviderCallable] = None
    visual_embed_provider: Optional[ProviderCallable] = None
    caption_embed_provider: Optional[ProviderCallable] = None
    context_embed_provider: Optional[ProviderCallable] = None
    text_embed_provider: Optional[ProviderCallable] = None
    visual_text_embed_provider: Optional[ProviderCallable] = None
    text_sparse_vec_provider: Optional[ProviderCallable] = None
    bm25_text_provider: Optional[ProviderCallable] = None


@dataclass(slots=True)
class ImageFeaturePipelineContext:
    image_bytes: bytes
    draft_record: ImageRecord
    manifest: Optional[ValidatedImageManifest] = None
    context_text: str = ""
    scratch: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(slots=True)
class ImageFeatureFailure:
    feature_name: str
    error_code: str
    error_message: str
    retryable: bool = False


@dataclass(slots=True)
class ImageFeatureSummary:
    p_hash_present: bool = False
    visual_dim: Optional[int] = None
    caption_dim: Optional[int] = None
    context_dim: Optional[int] = None
    text_dim: Optional[int] = None
    visual_text_dim: Optional[int] = None
    sparse_terms: Optional[int] = None
    bm25_text_chars: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ImageFeatureFieldPlan:
    schema_mode: str
    include_visual_embed: bool
    include_caption_context: bool
    include_sparse_vector: bool
    include_reserved_text_placeholders: bool
    include_reserved_visual_placeholder: bool


@dataclass(slots=True)
class ImageFeaturePipelineResult:
    success: bool
    image_id: Optional[str]
    doc_id: Optional[str]
    feature_summary: ImageFeatureSummary = dataclass_field(default_factory=ImageFeatureSummary)
    failure_items: list[ImageFeatureFailure] = dataclass_field(default_factory=list)
    image_record: Optional[ImageRecord] = None


@dataclass(slots=True)
class ImageFeatureExecution:
    context: ImageFeaturePipelineContext
    field_plan: ImageFeatureFieldPlan
    summary: ImageFeatureSummary = dataclass_field(default_factory=ImageFeatureSummary)
    failures: list[ImageFeatureFailure] = dataclass_field(default_factory=list)
    values: dict[str, Any] = dataclass_field(default_factory=dict)


def prepare_image_feature_execution(
    image_bytes: bytes,
    manifest_or_draft: dict[str, Any] | ValidatedImageManifest | ImageManifestValidationResult | ImageRecord,
    *,
    providers: ImageFeatureProviders,
    context_text: str = "",
    schema_mode: str = "",
) -> ImageFeatureExecution | ImageFeaturePipelineResult:
    draft, manifest, input_failures = _coerce_pipeline_input(manifest_or_draft)
    if input_failures:
        return ImageFeaturePipelineResult(
            success=False,
            image_id=getattr(draft, "img_id", None),
            doc_id=getattr(draft, "docId", None),
            failure_items=input_failures,
        )

    field_plan = _feature_field_plan(schema_mode)
    missing = _missing_provider_failures(providers, field_plan)
    if missing:
        return ImageFeaturePipelineResult(
            success=False,
            image_id=draft.img_id,
            doc_id=draft.docId,
            failure_items=missing,
        )

    return ImageFeatureExecution(
        context=ImageFeaturePipelineContext(
            image_bytes=image_bytes,
            draft_record=draft,
            manifest=manifest,
            context_text=context_text,
        ),
        field_plan=field_plan,
    )


async def run_image_feature_visual_stage(
    execution: ImageFeatureExecution,
    *,
    providers: ImageFeatureProviders,
) -> ImageFeatureExecution:
    execution.values.update(
        await _collect_visual_features_concurrently(
            providers,
            execution.context,
            execution.field_plan,
            execution.summary,
            execution.failures,
        )
    )
    return execution


async def run_image_feature_text_stage(
    execution: ImageFeatureExecution,
    *,
    providers: ImageFeatureProviders,
) -> ImageFeatureExecution:
    if execution.failures:
        return execution
    execution.values.update(
        await _collect_text_features_concurrently(
            providers,
            execution.context,
            execution.field_plan,
            execution.summary,
            execution.failures,
        )
    )
    return execution


def finalize_image_feature_execution(
    execution: ImageFeatureExecution,
) -> ImageFeaturePipelineResult:
    draft = execution.context.draft_record
    summary = execution.summary
    failures = _drop_nonfatal_text_failures(draft, execution.failures)

    if failures:
        return ImageFeaturePipelineResult(
            success=False,
            image_id=draft.img_id,
            doc_id=draft.docId,
            feature_summary=summary,
            failure_items=failures,
        )

    text_embed = execution.values.get("text_embed")
    visual_text_embed = execution.values.get("visual_text_embed")
    text_sparse_vec = execution.values.get("text_sparse_vec")
    bm25_text = execution.values.get("bm25_text")
    if execution.field_plan.include_reserved_text_placeholders:
        text_embed = _reserved_dense_vector(TEXT_VECTOR_DIM)
        visual_text_embed = _reserved_dense_vector(VISUAL_TEXT_VECTOR_DIM)
        bm25_text = RESERVED_IMAGE_TEXT_PLACEHOLDER
        summary.text_dim = TEXT_VECTOR_DIM
        summary.visual_text_dim = VISUAL_TEXT_VECTOR_DIM
        summary.bm25_text_chars = len(RESERVED_IMAGE_TEXT_PLACEHOLDER)
        draft.img_meta["reserved_text_fields_placeholder"] = True
    if execution.field_plan.include_reserved_visual_placeholder:
        execution.values["visual_embed"] = _reserved_dense_vector(VISUAL_VECTOR_DIM)
        summary.visual_dim = VISUAL_VECTOR_DIM
        draft.img_meta["reserved_visual_embed_placeholder"] = True

    record = _copy_record_with_features(
        draft,
        p_hash=execution.values.get("p_hash"),
        visual_embed=execution.values.get("visual_embed"),
        caption_embed=execution.values.get("caption_embed"),
        context_embed=execution.values.get("context_embed"),
        text_embed=text_embed,
        visual_text_embed=visual_text_embed,
        text_sparse_vec=text_sparse_vec,
        bm25_text=bm25_text,
    )
    try:
        validate_writable_image_record(
            record,
            expected_doc_id=draft.docId,
            schema_mode=execution.field_plan.schema_mode,
        )
    except ValueError as exc:
        return ImageFeaturePipelineResult(
            success=False,
            image_id=draft.img_id,
            doc_id=draft.docId,
            feature_summary=summary,
            failure_items=[
                ImageFeatureFailure(
                    feature_name="image_record",
                    error_code=IMAGE_RECORD_NOT_WRITABLE,
                    error_message=str(exc),
                    retryable=False,
                )
            ],
        )

    return ImageFeaturePipelineResult(
        success=True,
        image_id=record.img_id,
        doc_id=record.docId,
        feature_summary=summary,
        image_record=record,
    )


async def process_image_features(
    image_bytes: bytes,
    manifest_or_draft: dict[str, Any] | ValidatedImageManifest | ImageManifestValidationResult | ImageRecord,
    *,
    providers: ImageFeatureProviders,
    context_text: str = "",
    schema_mode: str = "",
) -> ImageFeaturePipelineResult:
    """Build a writer-ready ImageRecord only when all injected features are valid."""
    prepared = prepare_image_feature_execution(
        image_bytes,
        manifest_or_draft,
        providers=providers,
        context_text=context_text,
        schema_mode=schema_mode,
    )
    if isinstance(prepared, ImageFeaturePipelineResult):
        return prepared

    context = prepared.context
    summary = prepared.summary
    failures = prepared.failures
    field_plan = prepared.field_plan
    draft = context.draft_record
    feature_values = await _collect_features_concurrently(
        providers,
        context,
        field_plan,
        summary,
        failures,
    )
    p_hash = feature_values.get("p_hash")
    visual_embed = feature_values.get("visual_embed")
    caption_embed = feature_values.get("caption_embed")
    context_embed = feature_values.get("context_embed")
    text_embed = feature_values.get("text_embed")
    visual_text_embed = feature_values.get("visual_text_embed")
    text_sparse_vec = feature_values.get("text_sparse_vec")
    bm25_text = feature_values.get("bm25_text")

    failures = _drop_nonfatal_text_failures(draft, failures)
    if failures:
        return ImageFeaturePipelineResult(
            success=False,
            image_id=draft.img_id,
            doc_id=draft.docId,
            feature_summary=summary,
            failure_items=failures,
        )

    if field_plan.include_reserved_text_placeholders:
        text_embed = _reserved_dense_vector(TEXT_VECTOR_DIM)
        visual_text_embed = _reserved_dense_vector(VISUAL_TEXT_VECTOR_DIM)
        bm25_text = RESERVED_IMAGE_TEXT_PLACEHOLDER
        summary.text_dim = TEXT_VECTOR_DIM
        summary.visual_text_dim = VISUAL_TEXT_VECTOR_DIM
        summary.bm25_text_chars = len(RESERVED_IMAGE_TEXT_PLACEHOLDER)
        draft.img_meta["reserved_text_fields_placeholder"] = True
    if field_plan.include_reserved_visual_placeholder:
        visual_embed = _reserved_dense_vector(VISUAL_VECTOR_DIM)
        summary.visual_dim = VISUAL_VECTOR_DIM
        draft.img_meta["reserved_visual_embed_placeholder"] = True

    record = _copy_record_with_features(
        draft,
        p_hash=p_hash,
        visual_embed=visual_embed,
        caption_embed=caption_embed,
        context_embed=context_embed,
        text_embed=text_embed,
        visual_text_embed=visual_text_embed,
        text_sparse_vec=text_sparse_vec,
        bm25_text=bm25_text,
    )
    try:
        validate_writable_image_record(
            record,
            expected_doc_id=draft.docId,
            schema_mode=field_plan.schema_mode,
        )
    except ValueError as exc:
        return ImageFeaturePipelineResult(
            success=False,
            image_id=draft.img_id,
            doc_id=draft.docId,
            feature_summary=summary,
            failure_items=[
                ImageFeatureFailure(
                    feature_name="image_record",
                    error_code=IMAGE_RECORD_NOT_WRITABLE,
                    error_message=str(exc),
                    retryable=False,
                )
            ],
        )

    return ImageFeaturePipelineResult(
        success=True,
        image_id=record.img_id,
        doc_id=record.docId,
        feature_summary=summary,
        image_record=record,
    )


def _drop_nonfatal_text_failures(
    draft: ImageRecord,
    failures: list[ImageFeatureFailure],
) -> list[ImageFeatureFailure]:
    """Keep visual ingestion alive when an optional text field cannot run."""
    required: list[ImageFeatureFailure] = []
    optional: list[dict[str, Any]] = []
    for failure in failures:
        if failure.feature_name in {"caption_embed", "context_embed"}:
            optional.append(
                {
                    "feature_name": failure.feature_name,
                    "error_code": failure.error_code,
                    "error_message": failure.error_message,
                    "retryable": failure.retryable,
                }
            )
        else:
            required.append(failure)
    if optional:
        draft.img_meta["text_vectorization_failures"] = optional
        draft.img_meta["text_vectorization_skipped"] = True
    return required


async def _collect_features_concurrently(
    providers: ImageFeatureProviders,
    context: ImageFeaturePipelineContext,
    field_plan: ImageFeatureFieldPlan,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    collectors: list[tuple[str, Awaitable[Any]]] = [
        ("p_hash", _collect_p_hash(providers.p_hash_provider, context, summary, failures)),
    ]
    if field_plan.include_visual_embed:
        collectors.append(
            (
                "visual_embed",
                _collect_dense_vector(
                    "visual_embed",
                    providers.visual_embed_provider,
                    context,
                    VISUAL_VECTOR_DIM,
                    summary,
                    failures,
                ),
            )
        )
    if field_plan.include_caption_context:
        collectors.extend(
            [
                (
                    "caption_embed",
                    _collect_dense_vector(
                        "caption_embed",
                        providers.caption_embed_provider,
                        context,
                        TEXT_VECTOR_DIM,
                        summary,
                        failures,
                    ),
                ),
                (
                    "context_embed",
                    _collect_dense_vector(
                        "context_embed",
                        providers.context_embed_provider,
                        context,
                        TEXT_VECTOR_DIM,
                        summary,
                        failures,
                    ),
                ),
            ]
        )
    if field_plan.include_sparse_vector and providers.text_sparse_vec_provider is not None:
        collectors.append(
            (
                "text_sparse_vec",
                _collect_sparse_vector(
                    providers.text_sparse_vec_provider,
                    context,
                    summary,
                    failures,
                ),
            )
        )
    results = await asyncio.gather(*(collector for _, collector in collectors))
    for (feature_name, _), value in zip(collectors, results):
        values[feature_name] = value
    return values


async def _collect_visual_features_concurrently(
    providers: ImageFeatureProviders,
    context: ImageFeaturePipelineContext,
    field_plan: ImageFeatureFieldPlan,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    collectors: list[tuple[str, Awaitable[Any]]] = [
        ("p_hash", _collect_p_hash(providers.p_hash_provider, context, summary, failures)),
    ]
    if field_plan.include_visual_embed:
        collectors.append(
            (
                "visual_embed",
                _collect_dense_vector(
                    "visual_embed",
                    providers.visual_embed_provider,
                    context,
                    VISUAL_VECTOR_DIM,
                    summary,
                    failures,
                ),
            )
        )
    results = await asyncio.gather(*(collector for _, collector in collectors))
    for (feature_name, _), value in zip(collectors, results):
        values[feature_name] = value
    return values


async def _collect_text_features_concurrently(
    providers: ImageFeatureProviders,
    context: ImageFeaturePipelineContext,
    field_plan: ImageFeatureFieldPlan,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    collectors: list[tuple[str, Awaitable[Any]]] = []
    if field_plan.include_caption_context:
        collectors.extend(
            [
                (
                    "caption_embed",
                    _collect_dense_vector(
                        "caption_embed",
                        providers.caption_embed_provider,
                        context,
                        TEXT_VECTOR_DIM,
                        summary,
                        failures,
                    ),
                ),
                (
                    "context_embed",
                    _collect_dense_vector(
                        "context_embed",
                        providers.context_embed_provider,
                        context,
                        TEXT_VECTOR_DIM,
                        summary,
                        failures,
                    ),
                ),
            ]
        )
    if field_plan.include_sparse_vector and providers.text_sparse_vec_provider is not None:
        collectors.append(
            (
                "text_sparse_vec",
                _collect_sparse_vector(
                    providers.text_sparse_vec_provider,
                    context,
                    summary,
                    failures,
                ),
            )
        )
    if not collectors:
        return values
    results = await asyncio.gather(*(collector for _, collector in collectors))
    for (feature_name, _), value in zip(collectors, results):
        values[feature_name] = value
    return values


def _coerce_pipeline_input(
    value: dict[str, Any] | ValidatedImageManifest | ImageManifestValidationResult | ImageRecord,
) -> tuple[Optional[ImageRecord], Optional[ValidatedImageManifest], list[ImageFeatureFailure]]:
    if isinstance(value, ImageRecord):
        failures = _validate_draft_identity(value)
        return value, None, failures

    manifest: Optional[ValidatedImageManifest] = None
    if isinstance(value, ValidatedImageManifest):
        manifest = value
    elif isinstance(value, ImageManifestValidationResult):
        if value.success and value.manifest is not None:
            manifest = value.manifest
        else:
            return None, None, [_manifest_failure(str(value.failures))]

    try:
        draft = build_image_record_draft_from_manifest(manifest or value)
    except Exception as exc:
        return None, None, [_manifest_failure(str(exc))]
    return draft, manifest, []


def _validate_draft_identity(draft: ImageRecord) -> list[ImageFeatureFailure]:
    if not draft.docId or not draft.img_id:
        return [_manifest_failure("draft record must include docId and img_id")]
    upstream_id = source_image_id(draft.raw_img_meta)
    if not upstream_id:
        return [_manifest_failure("draft raw_img_meta.id is required")]
    if draft.img_id != upstream_id:
        return [_manifest_failure(f"draft image id must match upstream id {upstream_id}")]
    return []


def _manifest_failure(message: str) -> ImageFeatureFailure:
    return ImageFeatureFailure(
        feature_name="manifest",
        error_code=IMAGE_MANIFEST_INVALID,
        error_message=message,
        retryable=False,
    )


def _feature_field_plan(schema_mode: str = "") -> ImageFeatureFieldPlan:
    resolved = normalize_image_schema_mode(schema_mode or settings.IMAGE_VECTOR_SCHEMA_MODE)
    include_legacy_placeholders = resolved == IMAGE_SCHEMA_MODE_HISTORY_V2
    include_visual_embed = bool(getattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", False))
    return ImageFeatureFieldPlan(
        schema_mode=resolved,
        include_visual_embed=include_visual_embed,
        include_caption_context=True,
        include_sparse_vector=False,
        include_reserved_text_placeholders=include_legacy_placeholders,
        include_reserved_visual_placeholder=not include_visual_embed,
    )


def _missing_provider_failures(
    providers: ImageFeatureProviders,
    field_plan: ImageFeatureFieldPlan,
) -> list[ImageFeatureFailure]:
    required = {"p_hash": providers.p_hash_provider}
    if field_plan.include_visual_embed:
        required["visual_embed"] = providers.visual_embed_provider
    # Caption/context vectors are optional per image.  Their providers may be
    # absent (or return None when the source has no corresponding text) while
    # visual_embed remains mandatory for the image-first ingestion contract.
    return [
        ImageFeatureFailure(
            feature_name=feature_name,
            error_code=IMAGE_FEATURE_PROVIDER_MISSING,
            error_message=f"{feature_name} provider is required",
            retryable=False,
        )
        for feature_name, provider in required.items()
        if provider is None
    ]


async def _collect_p_hash(
    provider: ProviderCallable,
    context: ImageFeaturePipelineContext,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> Optional[bytes]:
    value = await _call_provider("p_hash", provider, context, failures)
    if value is None:
        return None
    if hasattr(value, "success") and hasattr(value, "p_hash"):
        if not value.success:
            failures.append(
                ImageFeatureFailure(
                    feature_name="p_hash",
                    error_code=getattr(value, "error_code", None) or IMAGE_FEATURE_PHASH_INVALID,
                    error_message=getattr(value, "error_message", None) or "p_hash provider failed",
                    retryable=bool(getattr(value, "retryable", False)),
                )
            )
            return None
        value = value.p_hash
    try:
        p_hash = _normalize_p_hash(value)
    except ValueError as exc:
        failures.append(
            ImageFeatureFailure(
                feature_name="p_hash",
                error_code=IMAGE_FEATURE_PHASH_INVALID,
                error_message=str(exc),
                retryable=False,
            )
        )
        return None
    summary.p_hash_present = True
    return p_hash


async def _collect_dense_vector(
    feature_name: str,
    provider: Optional[ProviderCallable],
    context: ImageFeaturePipelineContext,
    expected_dim: int,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> Optional[list[float]]:
    if provider is None:
        return None
    value = await _call_provider(feature_name, provider, context, failures)
    if value is None:
        return None
    try:
        vector = _validate_dense_vector(value, expected_dim, feature_name)
    except ValueError as exc:
        failures.append(
            ImageFeatureFailure(
                feature_name=feature_name,
                error_code=IMAGE_FEATURE_VECTOR_INVALID,
                error_message=str(exc),
                retryable=False,
            )
        )
        return None
    if feature_name == "visual_embed":
        summary.visual_dim = len(vector)
    elif feature_name == "text_embed":
        summary.text_dim = len(vector)
    elif feature_name == "caption_embed":
        summary.caption_dim = len(vector)
    elif feature_name == "context_embed":
        summary.context_dim = len(vector)
    elif feature_name == "visual_text_embed":
        summary.visual_text_dim = len(vector)
    return vector


async def _collect_sparse_vector(
    provider: Optional[ProviderCallable],
    context: ImageFeaturePipelineContext,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> Optional[dict[Any, float]]:
    if provider is None:
        return None
    value = await _call_provider("text_sparse_vec", provider, context, failures)
    if value is None:
        return None
    try:
        sparse = _validate_sparse_vector(value)
    except ValueError as exc:
        failures.append(
            ImageFeatureFailure(
                feature_name="text_sparse_vec",
                error_code=IMAGE_FEATURE_SPARSE_INVALID,
                error_message=str(exc),
                retryable=False,
            )
        )
        return None
    summary.sparse_terms = len(sparse)
    return sparse


async def _collect_bm25_text(
    provider: Optional[ProviderCallable],
    context: ImageFeaturePipelineContext,
    summary: ImageFeatureSummary,
    failures: list[ImageFeatureFailure],
) -> Optional[str]:
    if provider is None:
        return None
    value = await _call_provider("bm25_text", provider, context, failures)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        failures.append(
            ImageFeatureFailure(
                feature_name="bm25_text",
                error_code=IMAGE_FEATURE_TEXT_MISSING,
                error_message="bm25_text must not be empty",
                retryable=False,
            )
        )
        return None
    summary.bm25_text_chars = len(text)
    return text


async def _call_provider(
    feature_name: str,
    provider: ProviderCallable,
    context: ImageFeaturePipelineContext,
    failures: list[ImageFeatureFailure],
) -> Any:
    try:
        value = provider(context)
        if inspect.isawaitable(value):
            value = await value
        return value
    except Exception as exc:
        failures.append(
            ImageFeatureFailure(
                feature_name=feature_name,
                error_code=str(getattr(exc, "error_code", IMAGE_FEATURE_PROVIDER_ERROR)),
                error_message=str(exc),
                retryable=bool(getattr(exc, "retryable", True)),
            )
        )
        return None


def _copy_record_with_features(
    draft: ImageRecord,
    *,
    p_hash: bytes,
    visual_embed: Optional[list[float]],
    caption_embed: Optional[list[float]],
    context_embed: Optional[list[float]],
    text_embed: Optional[list[float]],
    visual_text_embed: Optional[list[float]],
    text_sparse_vec: Optional[dict[Any, float]],
    bm25_text: Optional[str],
) -> ImageRecord:
    updates = {
        "p_hash": p_hash,
        "visual_embed": visual_embed,
        "caption_embed": caption_embed,
        "context_embed": context_embed,
        "text_embed": text_embed,
        "visual_text_embed": visual_text_embed,
        "text_sparse_vec": text_sparse_vec,
        "text": bm25_text,
    }
    if hasattr(draft, "model_copy"):
        return draft.model_copy(update=updates)
    return draft.copy(update=updates)


def _normalize_p_hash(value: Any) -> bytes:
    if isinstance(value, bytes):
        result = value
    elif isinstance(value, bytearray):
        result = bytes(value)
    elif isinstance(value, str):
        compact = value.strip().lower().removeprefix("0x")
        if len(compact) != _PHASH_BYTES * 2:
            raise ValueError("p_hash must be 16 hex characters")
        try:
            result = bytes.fromhex(compact)
        except ValueError as exc:
            raise ValueError("p_hash must be hex encoded") from exc
    else:
        raise ValueError("p_hash must be bytes or 16-character hex")
    if len(result) != _PHASH_BYTES:
        raise ValueError("p_hash must be 8 bytes")
    if all(byte == 0 for byte in result):
        raise ValueError("p_hash appears to be a placeholder")
    return result


def _validate_dense_vector(value: Any, expected_dim: int, feature_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{feature_name} must be a list")
    if len(value) != expected_dim:
        raise ValueError(f"{feature_name} dimension mismatch: expected={expected_dim}, actual={len(value)}")
    floats: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"{feature_name} must contain numbers")
        candidate = float(item)
        if not math.isfinite(candidate):
            raise ValueError(f"{feature_name} must contain finite numbers")
        floats.append(candidate)
    if all(item == 0.0 for item in floats):
        raise ValueError(f"{feature_name} appears to be a placeholder vector")
    return floats


def _reserved_dense_vector(dim: int) -> list[float]:
    if dim <= 0:
        return []
    return [_RESERVED_VECTOR_SENTINEL, *([0.0] * (dim - 1))]


def _validate_sparse_vector(value: Any) -> dict[Any, float]:
    if not isinstance(value, dict):
        raise ValueError("text_sparse_vec must be a non-empty dict")
    if not value:
        raise ValueError("text_sparse_vec must not be empty")
    sparse: dict[Any, float] = {}
    has_signal = False
    for key, raw_value in value.items():
        if key is None:
            raise ValueError("text_sparse_vec keys must not be null")
        if not isinstance(raw_value, (int, float)):
            raise ValueError("text_sparse_vec values must be numbers")
        candidate = float(raw_value)
        if not math.isfinite(candidate):
            raise ValueError("text_sparse_vec values must be finite")
        sparse[key] = candidate
        has_signal = has_signal or candidate != 0.0
    if not has_signal:
        raise ValueError("text_sparse_vec appears to be a placeholder vector")
    return sparse
