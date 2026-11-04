"""Embedding API 客户端。"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import httpx

from app.config import settings
from app.models.sync_runtime import EmbeddingItemResult, ErrorType
from app.observability.logger import get_logger
from app.observability.metrics import embedding_latency_seconds, embedding_requests_total
from app.services.embedding_rate_limiter import build_text_rate_limiter

logger = get_logger(__name__)


@dataclass(slots=True)
class _QueuedEmbedding:
    text: str
    future: asyncio.Future[EmbeddingItemResult]
    enqueued_at: float


class EmbeddingClient:
    """Embedding API 异步客户端。"""

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
        max_concurrency: int = 0,
        timeout: int = 0,
        max_retries: int = 0,
        batch_size: int = 0,
    ) -> None:
        self.base_url = base_url or settings.EMBEDDING_API_URL
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.max_concurrency = max_concurrency or settings.EMBEDDING_MAX_CONCURRENCY
        self.timeout = timeout or settings.EMBEDDING_TIMEOUT
        self.max_retries = max_retries or settings.EMBEDDING_MAX_RETRIES
        self.batch_size = max(int(batch_size or settings.EMBEDDING_BATCH_SIZE or 1), 1)
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._text_rate_limiter = build_text_rate_limiter(
            enabled=bool(getattr(settings, "EMBEDDING_RATE_LIMIT_ENABLED", False)),
            rate_per_second=float(getattr(settings, "EMBEDDING_TEXT_RATE", 0.0) or 0.0),
            burst=int(getattr(settings, "EMBEDDING_TEXT_BURST", 32) or 32),
        )
        self._coalesce_lock = asyncio.Lock()
        self._coalesce_pending: list[_QueuedEmbedding] = []
        self._coalesce_flush_task: Optional[asyncio.Task[None]] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                limits=httpx.Limits(
                    max_connections=200,
                    max_keepalive_connections=50,
                ),
            )
        return self._client

    async def close(self) -> None:
        await self._flush_coalesced(force=True)
        if self._coalesce_flush_task is not None:
            self._coalesce_flush_task.cancel()
            self._coalesce_flush_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_once(self, text: str) -> List[float]:
        vectors = await self._request_batch_once([text])
        return vectors[0]

    async def _request_batch_once(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        client = await self._get_client()
        resolved_texts = [str(text or "") for text in texts]
        if self._text_rate_limiter is not None:
            await self._text_rate_limiter.acquire(len(resolved_texts))
        payload_input = resolved_texts[0] if len(resolved_texts) == 1 else resolved_texts
        payload = {"model": self.model, "input": payload_input}
        start = time.monotonic()
        response = await client.post(self.base_url, json=payload)
        elapsed = time.monotonic() - start

        if response.status_code != 200:
            if response.status_code == 429:
                embedding_requests_total.labels(status="429").inc()
            elif response.status_code >= 500:
                embedding_requests_total.labels(status="5xx").inc()
            else:
                embedding_requests_total.labels(status="4xx").inc()
            raise httpx.HTTPStatusError(
                f"Embedding API status={response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        data = response.json()
        embedding_data = data.get("data", [])
        if not embedding_data:
            raise ValueError(f"Empty embedding data in response: {data}")
        if len(embedding_data) != len(resolved_texts):
            raise ValueError(
                "Embedding result count mismatch: "
                f"expected={len(resolved_texts)}, actual={len(embedding_data)}"
            )
        vectors: List[List[float]] = []
        for index, item in enumerate(embedding_data):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid embedding item at index={index}: {data}")
            vector = item.get("embedding", [])
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Invalid embedding vector at index={index}: {data}")
            vectors.append(vector)

        embedding_requests_total.labels(status="success").inc()
        embedding_latency_seconds.observe(elapsed)
        return vectors

    async def embed_single_detailed(self, text: str) -> EmbeddingItemResult:
        """单文本向量化，返回显式成功/失败结果。"""
        if not text or not text.strip():
            return EmbeddingItemResult(
                success=False,
                vector=None,
                error_type=ErrorType.NON_RETRYABLE,
                error_code="EMBEDDING_INPUT_EMPTY",
                error_message="embedding input is empty",
                retryable=False,
                skipped_empty=True,
            )

        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    vector = await self._request_once(text)
                    return EmbeddingItemResult(success=True, vector=vector)
                except httpx.HTTPStatusError as error:
                    result = self._classify_http_error(error)
                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.RemoteProtocolError,
                    ConnectionError,
                    TimeoutError,
                ) as error:
                    embedding_requests_total.labels(status="network_error").inc()
                    result = EmbeddingItemResult(
                        success=False,
                        vector=None,
                        error_type=ErrorType.RETRYABLE,
                        error_code="EMBEDDING_NETWORK_ERROR",
                        error_message=str(error),
                        retryable=True,
                    )
                except ValueError as error:
                    result = EmbeddingItemResult(
                        success=False,
                        vector=None,
                        error_type=ErrorType.NON_RETRYABLE,
                        error_code="EMBEDDING_EMPTY_RESPONSE",
                        error_message=str(error),
                        retryable=False,
                    )
                except Exception as error:
                    result = EmbeddingItemResult(
                        success=False,
                        vector=None,
                        error_type=ErrorType.NON_RETRYABLE,
                        error_code="EMBEDDING_UNKNOWN_ERROR",
                        error_message=str(error),
                        retryable=False,
                    )

                if not result.retryable or attempt >= self.max_retries:
                    logger.error(
                        "embed_single_failed",
                        attempt=attempt,
                        error_code=result.error_code,
                        retryable=result.retryable,
                        text_preview=text[:100],
                        error=result.error_message,
                    )
                    return result

                delay = self._compute_backoff_delay(attempt)
                logger.warning(
                    "embed_single_retrying",
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                    error_code=result.error_code,
                    text_preview=text[:100],
                    error=result.error_message,
                )
                await asyncio.sleep(delay)

        return EmbeddingItemResult(
            success=False,
            vector=None,
            error_type=ErrorType.NON_RETRYABLE,
            error_code="EMBEDDING_UNEXPECTED_EXIT",
            error_message="embedding request exited unexpectedly",
            retryable=False,
        )

    async def embed_single(self, text: str) -> Optional[List[float]]:
        """兼容旧接口：成功返回向量，失败返回 None。"""
        result = await self.embed_single_detailed(text)
        return result.vector

    async def embed_texts_detailed(self, texts: List[str]) -> List[EmbeddingItemResult]:
        """批量文本向量化，返回逐项显式结果。"""
        if not texts:
            return []
        results: List[Optional[EmbeddingItemResult]] = [None] * len(texts)
        eligible: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            if not text or not text.strip():
                results[index] = EmbeddingItemResult(
                    success=False,
                    vector=None,
                    error_type=ErrorType.NON_RETRYABLE,
                    error_code="EMBEDDING_INPUT_EMPTY",
                    error_message="embedding input is empty",
                    retryable=False,
                    skipped_empty=True,
                )
                continue
            eligible.append((index, text))

        if eligible:
            if self._should_coalesce_short_request(len(eligible)):
                batch_results = [await self._embed_texts_coalesced(eligible)]
            else:
                tasks = [
                    self._embed_text_batch_detailed(eligible[start : start + self.batch_size])
                    for start in range(0, len(eligible), self.batch_size)
                ]
                batch_results = await asyncio.gather(*tasks)
            for batch in batch_results:
                for index, result in batch:
                    results[index] = result

        return [
            item
            if item is not None
            else EmbeddingItemResult(
                success=False,
                vector=None,
                error_type=ErrorType.NON_RETRYABLE,
                error_code="EMBEDDING_UNEXPECTED_MISSING_RESULT",
                error_message="embedding result was not populated",
                retryable=False,
            )
            for item in results
        ]

    async def _embed_text_batch_detailed(
        self,
        batch: Sequence[tuple[int, str]],
    ) -> list[tuple[int, EmbeddingItemResult]]:
        texts = [text for _, text in batch]
        for attempt in range(self.max_retries + 1):
            async with self._semaphore:
                try:
                    vectors = await self._request_batch_once(texts)
                    return [
                        (index, EmbeddingItemResult(success=True, vector=vector))
                        for (index, _), vector in zip(batch, vectors)
                    ]
                except httpx.HTTPStatusError as error:
                    result = self._classify_http_error(error)
                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.RemoteProtocolError,
                    ConnectionError,
                    TimeoutError,
                ) as error:
                    embedding_requests_total.labels(status="network_error").inc()
                    result = EmbeddingItemResult(
                        success=False,
                        vector=None,
                        error_type=ErrorType.RETRYABLE,
                        error_code="EMBEDDING_NETWORK_ERROR",
                        error_message=str(error),
                        retryable=True,
                    )
                except ValueError as error:
                    result = EmbeddingItemResult(
                        success=False,
                        vector=None,
                        error_type=ErrorType.NON_RETRYABLE,
                        error_code="EMBEDDING_EMPTY_RESPONSE",
                        error_message=str(error),
                        retryable=False,
                    )
                except Exception as error:
                    result = EmbeddingItemResult(
                        success=False,
                        vector=None,
                        error_type=ErrorType.NON_RETRYABLE,
                        error_code="EMBEDDING_UNKNOWN_ERROR",
                        error_message=str(error),
                        retryable=False,
                    )

            if _should_split_failed_batch(result, len(batch)):
                logger.warning(
                    "embed_batch_splitting_after_failure",
                    attempt=attempt,
                    batch_size=len(batch),
                    error_code=result.error_code,
                    retryable=result.retryable,
                    text_preview=texts[0][:100] if texts else "",
                    error=result.error_message,
                )
                return await self._split_embed_text_batch_detailed(batch)

            if not result.retryable or attempt >= self.max_retries:
                logger.error(
                    "embed_batch_failed",
                    attempt=attempt,
                    batch_size=len(batch),
                    error_code=result.error_code,
                    retryable=result.retryable,
                    text_preview=texts[0][:100] if texts else "",
                    error=result.error_message,
                )
                return [(index, result) for index, _ in batch]

            delay = self._compute_backoff_delay(attempt)
            logger.warning(
                "embed_batch_retrying",
                attempt=attempt + 1,
                delay_seconds=round(delay, 2),
                batch_size=len(batch),
                error_code=result.error_code,
                text_preview=texts[0][:100] if texts else "",
                error=result.error_message,
            )
            await asyncio.sleep(delay)

        result = EmbeddingItemResult(
            success=False,
            vector=None,
            error_type=ErrorType.NON_RETRYABLE,
            error_code="EMBEDDING_UNEXPECTED_EXIT",
            error_message="embedding request exited unexpectedly",
            retryable=False,
        )
        return [(index, result) for index, _ in batch]

    def _should_coalesce_short_request(self, eligible_count: int) -> bool:
        return bool(
            getattr(settings, "EMBEDDING_CROSS_DOCUMENT_BATCH_ENABLED", False)
            and self.batch_size > 1
            and 0 < eligible_count < self.batch_size
        )

    async def _embed_texts_coalesced(
        self,
        eligible: Sequence[tuple[int, str]],
    ) -> list[tuple[int, EmbeddingItemResult]]:
        """Queue a short caller so concurrent documents share one API batch."""
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        queued = [
            _QueuedEmbedding(
                text=text,
                future=loop.create_future(),
                enqueued_at=now,
            )
            for _, text in eligible
        ]
        async with self._coalesce_lock:
            self._coalesce_pending.extend(queued)
            if len(self._coalesce_pending) >= self.batch_size:
                asyncio.create_task(self._flush_coalesced(force=False))
            elif self._coalesce_flush_task is None or self._coalesce_flush_task.done():
                self._coalesce_flush_task = asyncio.create_task(self._flush_after_wait())

        item_results = await asyncio.gather(*(item.future for item in queued))
        return list(zip((index for index, _ in eligible), item_results))

    async def _flush_after_wait(self) -> None:
        wait_seconds = max(
            float(getattr(settings, "EMBEDDING_CROSS_DOCUMENT_BATCH_WAIT_MS", 25) or 0) / 1000.0,
            0.0,
        )
        if wait_seconds:
            await asyncio.sleep(wait_seconds)
        await self._flush_coalesced(force=True)

    async def _flush_coalesced(self, *, force: bool) -> None:
        batches: list[list[_QueuedEmbedding]] = []
        async with self._coalesce_lock:
            while len(self._coalesce_pending) >= self.batch_size:
                batches.append(self._coalesce_pending[: self.batch_size])
                del self._coalesce_pending[: self.batch_size]
            if force and self._coalesce_pending:
                batches.append(list(self._coalesce_pending))
                self._coalesce_pending.clear()
            if not self._coalesce_pending:
                self._coalesce_flush_task = None

        if batches:
            await asyncio.gather(*(self._dispatch_coalesced_batch(batch) for batch in batches))

    async def _dispatch_coalesced_batch(self, batch: Sequence[_QueuedEmbedding]) -> None:
        try:
            # The local offsets are only used by the existing batch API to map
            # results; every queued caller receives its own future below.
            results = await self._embed_text_batch_detailed(
                [(index, item.text) for index, item in enumerate(batch)]
            )
            resolved_at = time.perf_counter()
            for queued, (_, result) in zip(batch, results):
                if not queued.future.done():
                    queued.future.set_result(
                        replace(
                            result,
                            queue_wait_ms=round((resolved_at - queued.enqueued_at) * 1000.0, 3),
                        )
                    )
        except Exception as error:  # defensive: callers must never hang on a flush task
            for queued in batch:
                if not queued.future.done():
                    queued.future.set_result(
                        EmbeddingItemResult(
                            success=False,
                            vector=None,
                            error_type=ErrorType.RETRYABLE,
                            error_code="EMBEDDING_COALESCE_DISPATCH_ERROR",
                            error_message=str(error),
                            retryable=True,
                            queue_wait_ms=round((time.perf_counter() - queued.enqueued_at) * 1000.0, 3),
                        )
                    )

    async def _split_embed_text_batch_detailed(
        self,
        batch: Sequence[tuple[int, str]],
    ) -> list[tuple[int, EmbeddingItemResult]]:
        if len(batch) <= 1:
            return await self._embed_text_batch_detailed(batch)
        midpoint = len(batch) // 2
        left, right = await asyncio.gather(
            self._embed_text_batch_detailed(batch[:midpoint]),
            self._embed_text_batch_detailed(batch[midpoint:]),
        )
        return [*left, *right]

    async def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """兼容旧接口：失败项返回 None。"""
        results = await self.embed_texts_detailed(texts)
        return [item.vector for item in results]

    async def health_check(self) -> bool:
        try:
            result = await self.embed_single_detailed("ping")
            return bool(result.success and result.vector)
        except Exception:
            return False

    def _classify_http_error(self, error: httpx.HTTPStatusError) -> EmbeddingItemResult:
        status = error.response.status_code
        if status == 429:
            return EmbeddingItemResult(
                success=False,
                vector=None,
                error_type=ErrorType.RETRYABLE,
                error_code="EMBEDDING_RATE_LIMIT",
                error_message=str(error),
                retryable=True,
            )
        if status >= 500:
            return EmbeddingItemResult(
                success=False,
                vector=None,
                error_type=ErrorType.RETRYABLE,
                error_code=f"EMBEDDING_HTTP_{status}",
                error_message=str(error),
                retryable=True,
            )
        return EmbeddingItemResult(
            success=False,
            vector=None,
            error_type=ErrorType.NON_RETRYABLE,
            error_code=f"EMBEDDING_HTTP_{status}",
            error_message=str(error),
            retryable=False,
        )

    def _compute_backoff_delay(self, attempt: int) -> float:
        delay = min(
            settings.RETRY_BASE_DELAY_SEC * (2 ** attempt),
            settings.RETRY_MAX_DELAY_SEC,
        )
        return delay * random.uniform(0.75, 1.25)


_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def _should_split_failed_batch(result: EmbeddingItemResult, batch_size: int) -> bool:
    if batch_size <= 1 or result.retryable:
        return False
    code = str(result.error_code or "")
    return code.startswith("EMBEDDING_HTTP_4") or code == "EMBEDDING_EMPTY_RESPONSE"


async def close_embedding_client() -> None:
    global _embedding_client
    if _embedding_client is not None:
        await _embedding_client.close()
        _embedding_client = None
