"""Downloader foundation for image control-plane tasks."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import httpx

from app.config import settings
from app.services.static_resource_url import (
    StaticResourceConfigurationError,
    build_static_resource_url,
    is_static_resource_configured,
)


class ImageDownloadError(RuntimeError):
    """Base class for image download failures."""


class ImageConfigurationError(ImageDownloadError):
    """Raised when image download settings are incomplete."""


class ImageRetryableDownloadError(ImageDownloadError):
    """Raised after retryable download failures are exhausted."""


class ImageNonRetryableDownloadError(ImageDownloadError):
    """Raised for deterministic download failures such as 4xx responses."""


@dataclass(slots=True)
class ImageDownloadResult:
    relative_url: str
    url: str
    content: bytes
    content_type: str
    size_bytes: int


class ImageDownloader:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        concurrency: Optional[int] = None,
        slow_timeout: Optional[int] = None,
        slow_concurrency: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.base_url = settings.ATTACHMENT_PREFIX if base_url is None else base_url
        # The direct Kafka lane must not let a few slow static-resource reads
        # occupy the whole document batch. Retried records explicitly use the
        # bounded slow lane from the compensation path.
        fast_timeout = (
            settings.IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC
            if timeout is None and settings.IMAGE_DOWNLOAD_LANE_ENABLED
            else settings.IMAGE_DOWNLOAD_TIMEOUT if timeout is None else timeout
        )
        self.timeout = max(int(fast_timeout or settings.IMAGE_DOWNLOAD_TIMEOUT), 1)
        self.slow_timeout = max(
            int(
                settings.IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC
                if slow_timeout is None
                else slow_timeout
            ),
            self.timeout,
        )
        resolved_max_retries = (
            settings.IMAGE_DOWNLOAD_MAX_RETRIES
            if max_retries is None
            else max_retries
        )
        self.max_retries = max(int(resolved_max_retries or 0), 0)
        resolved_concurrency = (
            settings.IMAGE_DOWNLOAD_CONCURRENCY
            if concurrency is None
            else concurrency
        )
        self.concurrency = max(int(resolved_concurrency or 1), 1)
        self._semaphore = asyncio.Semaphore(self.concurrency)
        resolved_slow_concurrency = (
            settings.IMAGE_SLOW_DOWNLOAD_CONCURRENCY
            if slow_concurrency is None
            else slow_concurrency
        )
        self.slow_concurrency = max(int(resolved_slow_concurrency or 1), 1)
        self._slow_semaphore = asyncio.Semaphore(self.slow_concurrency)
        self._client = client
        self._sleep = sleep

    async def download_bytes(self, relative_url: str, *, lane: str = "fast") -> ImageDownloadResult:
        normalized_lane = str(lane or "fast").strip().lower()
        if normalized_lane not in {"fast", "slow"}:
            raise ImageConfigurationError(f"unsupported image download lane: {lane}")
        semaphore = self._slow_semaphore if normalized_lane == "slow" else self._semaphore
        timeout = self.slow_timeout if normalized_lane == "slow" else self.timeout
        async with semaphore:
            return await self._download_bytes(relative_url, timeout=timeout)

    async def _download_bytes(self, relative_url: str, *, timeout: int) -> ImageDownloadResult:
        url = build_image_url(relative_url, base_url=self.base_url)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._request_once(url, timeout=timeout)
                if response.status_code != 200:
                    response.raise_for_status()

                content = response.content
                return ImageDownloadResult(
                    relative_url=relative_url,
                    url=url,
                    content=content,
                    content_type=response.headers.get("content-type", ""),
                    size_bytes=len(content),
                )
            except httpx.HTTPStatusError as error:
                if not _is_retryable_status(error.response.status_code):
                    raise ImageNonRetryableDownloadError(
                        _format_download_error(error)
                    ) from error
                last_error = error
            except _RETRYABLE_NETWORK_ERRORS as error:
                last_error = error

            if attempt >= self.max_retries:
                break
            await self._sleep(_compute_backoff_delay(attempt))

        raise ImageRetryableDownloadError(_format_download_error(last_error)) from last_error

    async def _request_once(self, url: str, *, timeout: int) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, timeout=httpx.Timeout(timeout))

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            return await client.get(url)


def build_image_url(relative_url: str, *, base_url: Optional[str] = None) -> str:
    try:
        return build_static_resource_url(relative_url, base_url=base_url)
    except StaticResourceConfigurationError as exc:
        raise ImageConfigurationError("ATTACHMENT_PREFIX is required") from exc


def is_image_download_configured() -> bool:
    return is_static_resource_configured()


def build_image_readiness_status() -> dict[str, object]:
    configured = is_image_download_configured()
    visual_embed_enabled = bool(getattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", False))
    vl_embed_key = settings.IMAGE_VL_EMBED_API_KEY or settings.IMAGE_MODEL_API_KEY
    text_embed_key = (
        settings.IMAGE_TEXT_EMBED_API_KEY
        or settings.IMAGE_MODEL_API_KEY
        or settings.EMBEDDING_API_KEY
    )
    required_model_values = [
        text_embed_key,
        settings.IMAGE_TEXT_EMBED_URL,
        settings.IMAGE_TEXT_EMBED_MODEL,
    ]
    if visual_embed_enabled:
        required_model_values.extend(
            [
                vl_embed_key,
                settings.IMAGE_VL_EMBED_URL,
                settings.IMAGE_VL_EMBED_MODEL,
            ]
        )
    model_configured = all(
        str(value or "").strip()
        for value in required_model_values
    )
    return {
        "configured": configured,
        "base_url_configured": configured,
        "task_upsert_enabled": settings.IMAGE_TASK_UPSERT_ENABLED,
        "worker_enabled": settings.IMAGE_WORKER_ENABLED,
        "worker_running": False,
        "feature_pipeline_enabled": settings.IMAGE_FEATURE_PIPELINE_ENABLED,
        "visual_embed_enabled": visual_embed_enabled,
        "milvus_write_enabled": settings.IMAGE_MILVUS_WRITE_ENABLED,
        "timeout": settings.IMAGE_DOWNLOAD_TIMEOUT,
        "max_retries": settings.IMAGE_DOWNLOAD_MAX_RETRIES,
        "download_concurrency": settings.IMAGE_DOWNLOAD_CONCURRENCY,
        "preprocess_concurrency": settings.IMAGE_PREPROCESS_CONCURRENCY,
        "max_pixels": settings.IMAGE_MAX_PIXELS,
        "feature_model_configured": model_configured,
        "feature_timeout": settings.IMAGE_FEATURE_TIMEOUT,
        "feature_total_timeout": settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC,
        "feature_max_retries": settings.IMAGE_FEATURE_MAX_RETRIES,
        "mmd_hot_path_mode": settings.IMAGE_MMD_HOT_PATH_MODE,
        "model_total_concurrency": settings.IMAGE_MODEL_TOTAL_CONCURRENCY,
        "visual_model_priority": True,
        "vl_context_max_chars": settings.IMAGE_VL_CONTEXT_MAX_CHARS,
        "context_before_paragraphs": settings.IMAGE_CONTEXT_BEFORE_PARAGRAPHS,
        "context_after_paragraphs": settings.IMAGE_CONTEXT_AFTER_PARAGRAPHS,
        "feature_auth_sources": {
            "vl_embed_api_key": (
                "IMAGE_VL_EMBED_API_KEY"
                if str(settings.IMAGE_VL_EMBED_API_KEY or "").strip()
                else "IMAGE_MODEL_API_KEY"
            ),
            "text_embed_api_key": (
                "IMAGE_TEXT_EMBED_API_KEY"
                if str(settings.IMAGE_TEXT_EMBED_API_KEY or "").strip()
                else (
                    "IMAGE_MODEL_API_KEY"
                    if str(settings.IMAGE_MODEL_API_KEY or "").strip()
                    else "EMBEDDING_API_KEY"
                )
            ),
        },
        "pixel_validation_available": True,
        "pixel_validation_note": "png_jpeg_header_dimension_validation",
    }


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _compute_backoff_delay(attempt: int) -> float:
    delay = min(
        settings.RETRY_BASE_DELAY_SEC * (2 ** attempt),
        settings.RETRY_MAX_DELAY_SEC,
    )
    return delay * random.uniform(0.75, 1.25)


def _format_download_error(error: Optional[Exception]) -> str:
    if error is None:
        return "download failed without captured exception"
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        return (
            f"{type(error).__name__}:status={response.status_code};"
            f"url={response.url};body={_safe_preview(response.text)}"
        )
    request = getattr(error, "request", None)
    url = getattr(request, "url", "")
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}:{message};url={url}"
    return f"{type(error).__name__};url={url}"


def _safe_preview(text: str, max_chars: int = 200) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


_RETRYABLE_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)
