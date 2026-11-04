"""Downloader for .mmd full-text files."""
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


class FullTextDownloadError(RuntimeError):
    """Base class for full-text download failures."""


class FullTextConfigurationError(FullTextDownloadError):
    """Raised when full-text download settings are incomplete."""


class FullTextRetryableDownloadError(FullTextDownloadError):
    """Raised after retryable download failures are exhausted."""


class FullTextNonRetryableDownloadError(FullTextDownloadError):
    """Raised for deterministic download failures such as 4xx responses."""


class FullTextDocumentTooLarge(FullTextDownloadError):
    """Raised when downloaded text exceeds MAX_DOC_CHARS."""


@dataclass(slots=True)
class FullTextDownloadResult:
    relative_url: str
    url: str
    text: str
    char_count: int


class FullTextDownloader:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        max_doc_chars: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.base_url = settings.ATTACHMENT_PREFIX if base_url is None else base_url
        self.timeout = (
            settings.FULLTEXT_DOWNLOAD_TIMEOUT if timeout is None else timeout
        )
        self.max_retries = (
            settings.FULLTEXT_DOWNLOAD_MAX_RETRIES
            if max_retries is None
            else max_retries
        )
        self.max_doc_chars = (
            settings.MAX_DOC_CHARS if max_doc_chars is None else max_doc_chars
        )
        self._client = client
        self._sleep = sleep

    async def download_text(self, relative_url: str) -> FullTextDownloadResult:
        url = build_fulltext_url(relative_url, base_url=self.base_url)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._request_once(url)
                if response.status_code != 200:
                    response.raise_for_status()

                text = response.text
                self._ensure_within_size_limit(text)
                return FullTextDownloadResult(
                    relative_url=relative_url,
                    url=url,
                    text=text,
                    char_count=len(text),
                )
            except FullTextDocumentTooLarge:
                raise
            except httpx.HTTPStatusError as error:
                if not _is_retryable_status(error.response.status_code):
                    raise FullTextNonRetryableDownloadError(str(error)) from error
                last_error = error
            except _RETRYABLE_NETWORK_ERRORS as error:
                last_error = error

            if attempt >= self.max_retries:
                break
            await self._sleep(_compute_backoff_delay(attempt))

        raise FullTextRetryableDownloadError(str(last_error)) from last_error

    async def _request_once(self, url: str) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, timeout=httpx.Timeout(self.timeout))

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            return await client.get(url)

    def _ensure_within_size_limit(self, text: str) -> None:
        if self.max_doc_chars is None:
            return
        if self.max_doc_chars >= 0 and len(text) > self.max_doc_chars:
            raise FullTextDocumentTooLarge(
                f".mmd text has {len(text)} chars, exceeds MAX_DOC_CHARS={self.max_doc_chars}"
            )


def build_fulltext_url(relative_url: str, *, base_url: Optional[str] = None) -> str:
    try:
        return build_static_resource_url(relative_url, base_url=base_url)
    except StaticResourceConfigurationError as exc:
        raise FullTextConfigurationError("ATTACHMENT_PREFIX is required") from exc


def is_fulltext_download_configured() -> bool:
    return is_static_resource_configured()


def build_fulltext_readiness_status() -> dict[str, object]:
    configured = is_fulltext_download_configured()
    return {
        "configured": configured,
        "base_url_configured": configured,
        "timeout": settings.FULLTEXT_DOWNLOAD_TIMEOUT,
        "max_retries": settings.FULLTEXT_DOWNLOAD_MAX_RETRIES,
        "max_doc_chars": settings.MAX_DOC_CHARS,
    }


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _compute_backoff_delay(attempt: int) -> float:
    delay = min(
        settings.RETRY_BASE_DELAY_SEC * (2 ** attempt),
        settings.RETRY_MAX_DELAY_SEC,
    )
    return delay * random.uniform(0.75, 1.25)


_RETRYABLE_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)
