"""Chunk embedding helper for full-text chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from app.config import settings
from app.models.chunk import Chunk


@dataclass(frozen=True)
class ChunkEmbeddingFailure:
    chunkId: str
    error_code: str
    error_message: str
    retryable: bool


@dataclass(frozen=True)
class ChunkEmbeddingBatchResult:
    chunks: List[Chunk]
    failures: List[ChunkEmbeddingFailure]
    average_queue_wait_ms: float = 0.0


async def embed_chunks(
    chunks: Sequence[Chunk],
    embedder: Any,
    vector_dim: int = settings.VECTOR_DIM,
) -> ChunkEmbeddingBatchResult:
    """Embed chunks and return only chunks that are safe to write."""

    successes: List[Chunk] = []
    failures: List[ChunkEmbeddingFailure] = []
    eligible: List[tuple[int, Chunk]] = []

    for index, chunk in enumerate(chunks):
        if not chunk.text or not chunk.text.strip():
            failures.append(
                _failure(
                    chunk,
                    "CHUNK_EMPTY_TEXT",
                    "chunk text is empty; embedding skipped",
                    retryable=False,
                )
            )
            continue
        eligible.append((index, chunk))

    if not eligible:
        return ChunkEmbeddingBatchResult(chunks=successes, failures=failures)

    texts = [chunk.text for _, chunk in eligible]
    try:
        results = await embedder.embed_texts_detailed(texts)
    except Exception as error:
        for _, chunk in eligible:
            failures.append(
                _failure(
                    chunk,
                    "CHUNK_EMBEDDING_BATCH_ERROR",
                    str(error),
                    retryable=True,
                )
            )
        return ChunkEmbeddingBatchResult(chunks=successes, failures=failures)

    if len(results) != len(eligible):
        message = f"embedding result count mismatch: expected={len(eligible)}, actual={len(results)}"
        for _, chunk in eligible:
            failures.append(
                _failure(
                    chunk,
                    "CHUNK_EMBEDDING_RESULT_COUNT_MISMATCH",
                    message,
                    retryable=False,
                )
            )
        return ChunkEmbeddingBatchResult(chunks=successes, failures=failures)

    queue_wait_samples: list[float] = []
    for (_, chunk), result in zip(eligible, results):
        try:
            queue_wait_samples.append(max(float(getattr(result, "queue_wait_ms", 0.0) or 0.0), 0.0))
        except (TypeError, ValueError):
            pass
        if not getattr(result, "success", False):
            failures.append(
                _failure(
                    chunk,
                    getattr(result, "error_code", None) or "CHUNK_EMBEDDING_FAILED",
                    getattr(result, "error_message", None) or "chunk embedding failed",
                    retryable=bool(getattr(result, "retryable", False)),
                )
            )
            continue

        if getattr(result, "skipped_empty", False):
            failures.append(
                _failure(
                    chunk,
                    "CHUNK_EMPTY_TEXT",
                    "embedder skipped empty text; chunk is not writable",
                    retryable=False,
                )
            )
            continue

        vector = getattr(result, "vector", None)
        if not isinstance(vector, list) or not vector:
            failures.append(
                _failure(
                    chunk,
                    "CHUNK_EMBEDDING_MISSING_VECTOR",
                    "embedding result did not include a vector",
                    retryable=False,
                )
            )
            continue

        if len(vector) != vector_dim:
            failures.append(
                _failure(
                    chunk,
                    "CHUNK_EMBEDDING_DIMENSION_MISMATCH",
                    f"embedding dimension mismatch: expected={vector_dim}, actual={len(vector)}",
                    retryable=False,
                )
            )
            continue

        successes.append(_copy_chunk_with_embedding(chunk, vector))

    return ChunkEmbeddingBatchResult(
        chunks=successes,
        failures=failures,
        average_queue_wait_ms=(sum(queue_wait_samples) / len(queue_wait_samples)) if queue_wait_samples else 0.0,
    )


def _failure(chunk: Chunk, error_code: str, error_message: str, *, retryable: bool) -> ChunkEmbeddingFailure:
    return ChunkEmbeddingFailure(
        chunkId=chunk.chunkId,
        error_code=error_code,
        error_message=error_message,
        retryable=retryable,
    )


def _copy_chunk_with_embedding(chunk: Chunk, vector: list[float]) -> Chunk:
    if hasattr(chunk, "model_copy"):
        return chunk.model_copy(update={"text_embed": vector})
    return chunk.copy(update={"text_embed": vector})
