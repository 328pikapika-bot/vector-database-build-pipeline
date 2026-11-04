"""Full-text chunk plan orchestration data plane."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from app.config import settings
from app.contracts.docx_contract import FULLTEXT_DOC_FIELDS, extract_docx_doc_meta
from app.models.chunk import Chunk
from app.services.chunk_embedding import ChunkEmbeddingFailure, embed_chunks
from app.services.milvus_chunk_service import (
    DEFAULT_CHUNK_COLLECTION,
    DEFAULT_CHUNKER_VERSION,
    DEFAULT_PARSER_VERSION,
    DEFAULT_SCHEMA_VERSION,
    MilvusChunkService,
)
from app.observability.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class FullTextChunkPipelineFailure:
    chunkId: Optional[str]
    error_code: str
    error_message: str
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunkId,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class FullTextChunkPipelineResult:
    success: bool
    doc_id: str
    chunk_count: int
    embedded_count: int
    failed_count: int
    failure_items: list[FullTextChunkPipelineFailure]
    write_summary: Optional[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "embedded_count": self.embedded_count,
            "failed_count": self.failed_count,
            "failure_items": [failure.to_dict() for failure in self.failure_items],
            "write_summary": self.write_summary,
            "metadata": self.metadata,
        }


async def process_fulltext_chunk_plan(
    chunk_plan_or_artifact: Mapping[str, Any],
    *,
    embedder: Any,
    chunk_writer: MilvusChunkService,
    collection_name: str = DEFAULT_CHUNK_COLLECTION,
    vector_dim: int = settings.VECTOR_DIM,
    no_delete: Optional[bool] = None,
) -> FullTextChunkPipelineResult:
    """Embed a chunk plan and write the document's Milvus chunks."""

    timings_ms: dict[str, float] = {}

    try:
        chunk_plan = _unwrap_chunk_plan(chunk_plan_or_artifact)
        chunks = _restore_chunks(chunk_plan)
        doc_id = _extract_doc_id(chunk_plan, chunks)
        doc_meta = _extract_doc_metadata(chunk_plan)
        _apply_doc_metadata(chunks, doc_meta=doc_meta)
        metadata = _extract_writer_metadata(chunk_plan)
    except Exception as error:
        failure = FullTextChunkPipelineFailure(
            chunkId=None,
            error_code="CHUNK_PLAN_INVALID",
            error_message=str(error),
            retryable=False,
        )
        return FullTextChunkPipelineResult(
            success=False,
            doc_id="",
            chunk_count=0,
            embedded_count=0,
            failed_count=1,
            failure_items=[failure],
            write_summary=None,
            metadata={},
        )

    embedding_started_at = time.perf_counter()
    embedding_result = await embed_chunks(chunks, embedder, vector_dim=vector_dim)
    timings_ms["embedding"] = round((time.perf_counter() - embedding_started_at) * 1000.0, 3)
    timings_ms["embedding_queue"] = round(embedding_result.average_queue_wait_ms, 3)
    failures = [_from_embedding_failure(failure) for failure in embedding_result.failures]
    if failures:
        return FullTextChunkPipelineResult(
            success=False,
            doc_id=doc_id,
            chunk_count=len(chunks),
            embedded_count=len(embedding_result.chunks),
            failed_count=len(failures),
            failure_items=failures,
            write_summary=None,
            metadata=metadata,
        )

    write_mode = _resolve_write_mode(no_delete)
    write_started_at = time.perf_counter()
    try:
        if not embedding_result.chunks:
            # An authoritative empty/parseable MMD means this document has no
            # current chunks. Treat it as a successful empty snapshot: remove
            # any stale chunks instead of turning the event into a retry loop.
            if write_mode == "safe_replace":
                delete_result = await asyncio.to_thread(
                    chunk_writer.delete_chunks_by_doc_id,
                    collection_name,
                    doc_id,
                )
            else:
                delete_result = {
                    "skipped": True,
                    "status": "DELETE_NOOP_RECORDED",
                    "reason": "FULLTEXT_EMPTY_CHUNK_INSERT_MODE",
                    "collection_name": collection_name,
                    "doc_id": doc_id,
                }
            write_summary = {
                "delete_result": delete_result,
                "insert_result": {"insert_count": 0},
                "empty_snapshot": True,
            }
        elif write_mode == "insert":
            insert_result = await asyncio.to_thread(
                chunk_writer.insert_chunks,
                collection_name,
                embedding_result.chunks,
                vector_dim=vector_dim,
                source_record_update_time=metadata["source_record_update_time"],
                parser_version=metadata["parser_version"],
                chunker_version=metadata["chunker_version"],
                embedding_model=metadata["embedding_model"],
                embedding_version=metadata["embedding_version"],
                schema_version=metadata["schema_version"],
                ingested_at=metadata["ingested_at"],
            )
            write_summary = {
                "delete_result": {
                    "skipped": True,
                    "status": "DELETE_NOOP_RECORDED",
                    "reason": (
                        "LIVE_E2E_NO_DELETE"
                        if _no_delete_enabled(no_delete)
                        else "FULLTEXT_CHUNK_WRITE_MODE_INSERT"
                    ),
                    "operation": "replace_chunks_predelete",
                    "collection_name": collection_name,
                    "doc_id": doc_id,
                },
                "insert_result": insert_result,
            }
        else:
            write_summary = await asyncio.to_thread(
                chunk_writer.replace_chunks_for_document,
                collection_name,
                doc_id,
                embedding_result.chunks,
                vector_dim=vector_dim,
                source_record_update_time=metadata["source_record_update_time"],
                parser_version=metadata["parser_version"],
                chunker_version=metadata["chunker_version"],
                embedding_model=metadata["embedding_model"],
                embedding_version=metadata["embedding_version"],
                schema_version=metadata["schema_version"],
                ingested_at=metadata["ingested_at"],
            )
    except Exception as error:
        failure = FullTextChunkPipelineFailure(
            chunkId=None,
            error_code="MILVUS_CHUNK_WRITE_ERROR",
            error_message=str(error),
            retryable=not isinstance(error, ValueError),
        )
        return FullTextChunkPipelineResult(
            success=False,
            doc_id=doc_id,
            chunk_count=len(chunks),
            embedded_count=len(embedding_result.chunks),
            failed_count=1,
            failure_items=[failure],
            write_summary=None,
            metadata={**metadata, "pipeline_timings_ms": timings_ms},
        )

    timings_ms["milvus_write"] = round((time.perf_counter() - write_started_at) * 1000.0, 3)
    visibility_started_at = time.perf_counter()
    visibility_failure = await _verify_written_chunks_visible(
        chunk_writer=chunk_writer,
        collection_name=collection_name,
        chunks=embedding_result.chunks,
    )
    if visibility_failure is not None:
        if not getattr(settings, "FULLTEXT_CHUNK_VISIBILITY_STRICT", False):
            logger.warning(
                "fulltext_chunk_visibility_probe_failed_non_strict",
                collection_name=collection_name,
                doc_id=doc_id,
                error_code=visibility_failure.error_code,
                error_message=visibility_failure.error_message,
            )
            write_summary = {
                **(write_summary or {}),
                "visibility_check": {
                    "success": False,
                    "strict": False,
                    "failure": visibility_failure.to_dict(),
                },
            }
        else:
            write_summary = {
                **(write_summary or {}),
                "visibility_check": {
                    "success": False,
                    "strict": True,
                    "failure": visibility_failure.to_dict(),
                },
            }
            return FullTextChunkPipelineResult(
                success=False,
                doc_id=doc_id,
                chunk_count=len(chunks),
                embedded_count=len(embedding_result.chunks),
                failed_count=1,
                failure_items=[visibility_failure],
                write_summary=write_summary,
                metadata={**metadata, "pipeline_timings_ms": timings_ms},
            )

    timings_ms["visibility_check"] = round(
        (time.perf_counter() - visibility_started_at) * 1000.0,
        3,
    )

    return FullTextChunkPipelineResult(
        success=True,
        doc_id=doc_id,
        chunk_count=len(chunks),
        embedded_count=len(embedding_result.chunks),
        failed_count=0,
        failure_items=[],
        write_summary=write_summary,
        metadata={**metadata, "pipeline_timings_ms": timings_ms},
    )


def delete_chunks_for_document(
    doc_id: str,
    *,
    chunk_writer: MilvusChunkService,
    collection_name: str = DEFAULT_CHUNK_COLLECTION,
) -> dict[str, Any]:
    """Delete full-text chunks for a document and return the writer summary."""

    normalized_doc_id = str(doc_id or "").strip()
    if not normalized_doc_id:
        raise ValueError("doc_id is required for DELETE_CHUNKS")

    # Keep the production append-only guard at this lower-level helper too.
    # Callers outside the normal pipeline cannot accidentally delete rows
    # while LIVE_E2E_NO_DELETE is enabled.
    if _no_delete_enabled(None):
        return {
            "skipped": True,
            "status": "DELETE_NOOP_RECORDED",
            "reason": "LIVE_E2E_NO_DELETE",
            "collection_name": collection_name,
            "doc_id": normalized_doc_id,
        }

    delete_result = chunk_writer.delete_chunks_by_doc_id(collection_name, normalized_doc_id)
    return {
        "doc_id": normalized_doc_id,
        "collection_name": collection_name,
        "delete_result": delete_result,
    }


def _unwrap_chunk_plan(chunk_plan_or_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(chunk_plan_or_artifact, Mapping):
        raise ValueError("chunk plan must be a mapping")

    for wrapper_key in ("content", "payload"):
        nested = chunk_plan_or_artifact.get(wrapper_key)
        if isinstance(nested, Mapping) and "chunks" in nested:
            return nested
    return chunk_plan_or_artifact


def _restore_chunks(chunk_plan: Mapping[str, Any]) -> list[Chunk]:
    raw_chunks = chunk_plan.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("chunk plan must contain a chunks list")

    chunks: list[Chunk] = []
    for index, raw_chunk in enumerate(raw_chunks):
        if isinstance(raw_chunk, Chunk):
            chunks.append(raw_chunk)
            continue
        if isinstance(raw_chunk, Mapping):
            chunks.append(_chunk_from_mapping(raw_chunk))
            continue
        raise ValueError(f"chunk at index {index} must be a mapping or Chunk")
    return chunks


def _chunk_from_mapping(raw_chunk: Mapping[str, Any]) -> Chunk:
    data = dict(raw_chunk)
    if hasattr(Chunk, "model_validate"):
        return Chunk.model_validate(data)
    return Chunk(**data)


def _extract_doc_id(chunk_plan: Mapping[str, Any], chunks: Sequence[Chunk]) -> str:
    doc_id = _first_present(chunk_plan, ("doc_id", "docId"))
    if doc_id is None and chunks:
        doc_id = chunks[0].doc_id

    normalized_doc_id = str(doc_id or "").strip()
    if not normalized_doc_id:
        raise ValueError("doc_id is required in chunk plan")
    return normalized_doc_id


def _apply_doc_metadata(
    chunks: Sequence[Chunk],
    *,
    doc_meta: Mapping[str, Any],
) -> None:
    for chunk in chunks:
        for field_name in FULLTEXT_DOC_FIELDS:
            if field_name == "doc_id":
                continue
            value = doc_meta.get(field_name)
            if value is not None and getattr(chunk, field_name, None) is None:
                setattr(chunk, field_name, value)
        if not chunk.language:
            chunk.language = ["unknown"]
        if not chunk.sectionPath:
            chunk.sectionPath = "body"


def _extract_doc_metadata(chunk_plan: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in _metadata_sources(chunk_plan):
        merged.update({key: value for key, value in source.items() if key != "chunks"})
    for key in ("doc_meta", "document_meta", "article_meta"):
        nested = chunk_plan.get(key)
        if isinstance(nested, Mapping):
            merged.update(nested)
    return extract_docx_doc_meta(merged)


def _extract_writer_metadata(chunk_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_record_update_time": _metadata_value(
            chunk_plan,
            ("source_record_update_time", "sourceRecordUpdateTime", "record_update_time", "recordUpdateTime"),
            default=None,
        ),
        "parser_version": _metadata_value(
            chunk_plan,
            ("parser_version", "parserVersion"),
            default=DEFAULT_PARSER_VERSION,
        ),
        "chunker_version": _metadata_value(
            chunk_plan,
            ("chunker_version", "chunkerVersion"),
            default=DEFAULT_CHUNKER_VERSION,
        ),
        "embedding_model": _metadata_value(
            chunk_plan,
            ("embedding_model", "embeddingModel"),
            default=settings.EMBEDDING_MODEL,
        ),
        "embedding_version": _metadata_value(
            chunk_plan,
            ("embedding_version", "embeddingVersion"),
            default="",
        ),
        "schema_version": _metadata_value(
            chunk_plan,
            ("schema_version", "schemaVersion"),
            default=DEFAULT_SCHEMA_VERSION,
        ),
        "ingested_at": _metadata_value(
            chunk_plan,
            ("ingested_at", "ingestedAt"),
            default=None,
        ),
    }


def _metadata_value(chunk_plan: Mapping[str, Any], aliases: Sequence[str], *, default: Any) -> Any:
    for source in _metadata_sources(chunk_plan):
        value = _first_present(source, aliases)
        if value is not None:
            return value
    return default


def _metadata_sources(chunk_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = [chunk_plan]
    for key in ("metadata", "meta"):
        nested = chunk_plan.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    return sources


def _first_present(source: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in source and source[key] not in ("", None):
            return source[key]
    return None


def _from_embedding_failure(failure: ChunkEmbeddingFailure) -> FullTextChunkPipelineFailure:
    return FullTextChunkPipelineFailure(
        chunkId=failure.chunkId,
        error_code=failure.error_code,
        error_message=failure.error_message,
        retryable=failure.retryable,
    )


async def _verify_written_chunks_visible(
    *,
    chunk_writer: MilvusChunkService,
    collection_name: str,
    chunks: Sequence[Chunk],
) -> Optional[FullTextChunkPipelineFailure]:
    if not getattr(settings, "FULLTEXT_CHUNK_VISIBILITY_CHECK_ENABLED", True):
        return None
    if not chunks:
        return None
    visible_counter = getattr(chunk_writer, "count_visible_chunk_ids", None)
    if visible_counter is None:
        return None

    limit = max(int(getattr(settings, "FULLTEXT_CHUNK_VISIBILITY_CHECK_LIMIT", 3) or 0), 0)
    if limit <= 0:
        return None
    sample_ids = [chunk.child_chunk_id for chunk in chunks[:limit]]
    timeout = float(getattr(settings, "FULLTEXT_CHUNK_VISIBILITY_CHECK_TIMEOUT_SEC", 10) or 10)
    attempts = max(int(getattr(settings, "FULLTEXT_CHUNK_VISIBILITY_RETRIES", 3) or 1), 1)
    delay = max(float(getattr(settings, "FULLTEXT_CHUNK_VISIBILITY_RETRY_DELAY_SEC", 1.0) or 0), 0)
    last_error: Optional[Exception] = None
    visible_count = 0

    for attempt in range(attempts):
        try:
            visible_count = int(
                await asyncio.to_thread(
                    visible_counter,
                    collection_name,
                    sample_ids,
                    timeout=timeout,
                )
                or 0
            )
        except Exception as error:
            last_error = error
        else:
            last_error = None
            if visible_count >= len(sample_ids):
                return None

        if attempt < attempts - 1 and delay > 0:
            await asyncio.sleep(delay)

    if last_error is not None:
        return FullTextChunkPipelineFailure(
            chunkId=sample_ids[0] if sample_ids else None,
            error_code="MILVUS_CHUNK_VISIBILITY_ERROR",
            error_message=f"Milvus chunk visibility check failed: {last_error}",
            retryable=True,
        )
    return FullTextChunkPipelineFailure(
        chunkId=sample_ids[0] if sample_ids else None,
        error_code="MILVUS_CHUNK_VISIBILITY_ERROR",
        error_message=(
            "Milvus chunk write returned success but sampled rows are not visible: "
            f"visible={int(visible_count or 0)}/{len(sample_ids)}"
        ),
        retryable=True,
    )


def _resolve_write_mode(no_delete: Optional[bool]) -> str:
    if _no_delete_enabled(no_delete):
        return "insert"
    mode = str(
        getattr(settings, "FULLTEXT_CHUNK_WRITE_MODE", "safe_replace") or "safe_replace"
    ).strip().lower()
    if mode in {"insert", "insert_only"}:
        return "insert"
    if mode in {"safe_replace", "replace", "upsert"}:
        return "safe_replace"
    raise ValueError(
        "FULLTEXT_CHUNK_WRITE_MODE must be one of safe_replace, insert, replace, or upsert"
    )


def _no_delete_enabled(no_delete: Optional[bool]) -> bool:
    return settings.LIVE_E2E_NO_DELETE if no_delete is None else bool(no_delete)
