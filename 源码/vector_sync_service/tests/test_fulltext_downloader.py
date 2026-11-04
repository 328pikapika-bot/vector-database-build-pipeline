from __future__ import annotations

import httpx
import pytest

from app.config import settings
from app.services.fulltext_downloader import (
    FullTextConfigurationError,
    FullTextDocumentTooLarge,
    FullTextDownloader,
    FullTextNonRetryableDownloadError,
    FullTextRetryableDownloadError,
    build_fulltext_url,
)


def test_fulltext_download_defaults_are_locked():
    assert settings.ATTACHMENT_PREFIX == "http://127.0.0.1:12001/"
    assert settings.FULLTEXT_DOWNLOAD_TIMEOUT == 30
    assert settings.FULLTEXT_DOWNLOAD_MAX_RETRIES == 3
    assert settings.FULLTEXT_TASK_UPSERT_ENABLED is True
    assert settings.FULLTEXT_WORKER_LEASE_SEC == 120
    assert settings.FULLTEXT_WORKER_BATCH_SIZE == 50
    assert settings.FULLTEXT_WORKER_ENABLED is True
    assert settings.FULLTEXT_WORKER_POLL_INTERVAL_SEC == 3
    assert settings.MAX_DOC_CHARS is None


def test_build_fulltext_url_joins_base_and_relative_url():
    assert (
        build_fulltext_url("/fulltext/paper.mmd", base_url="http://files.test/")
        == "http://files.test/fulltext/paper.mmd"
    )
    assert (
        build_fulltext_url("fulltext/paper.mmd", base_url="http://files.test")
        == "http://files.test/fulltext/paper.mmd"
    )


def test_build_fulltext_url_requires_base_url():
    with pytest.raises(FullTextConfigurationError):
        build_fulltext_url("/fulltext/paper.mmd", base_url="")


@pytest.mark.asyncio
async def test_download_text_success_returns_mmd_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://files.test/fulltext/paper.mmd"
        return httpx.Response(200, text="# Title\nBody", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FullTextDownloader(
            base_url="http://files.test",
            client=client,
            sleep=_no_sleep,
        ).download_text("/fulltext/paper.mmd")

    assert result.url == "http://files.test/fulltext/paper.mmd"
    assert result.text == "# Title\nBody"
    assert result.char_count == len("# Title\nBody")


@pytest.mark.asyncio
async def test_download_text_retries_network_failures_then_raises():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("network down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = FullTextDownloader(
            base_url="http://files.test",
            max_retries=2,
            client=client,
            sleep=_no_sleep,
        )
        with pytest.raises(FullTextRetryableDownloadError):
            await downloader.download_text("/fulltext/paper.mmd")

    assert attempts == 3


@pytest.mark.asyncio
async def test_download_text_4xx_is_non_retryable():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="missing", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = FullTextDownloader(
            base_url="http://files.test",
            max_retries=2,
            client=client,
            sleep=_no_sleep,
        )
        with pytest.raises(FullTextNonRetryableDownloadError):
            await downloader.download_text("/fulltext/missing.mmd")

    assert attempts == 1


@pytest.mark.asyncio
async def test_download_text_retries_5xx_then_succeeds():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="busy", request=request)
        return httpx.Response(200, text="ready", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await FullTextDownloader(
            base_url="http://files.test",
            max_retries=2,
            client=client,
            sleep=_no_sleep,
        ).download_text("/fulltext/paper.mmd")

    assert attempts == 2
    assert result.text == "ready"


@pytest.mark.asyncio
async def test_download_text_rejects_text_over_max_doc_chars():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="abcdef", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = FullTextDownloader(
            base_url="http://files.test",
            max_doc_chars=3,
            client=client,
            sleep=_no_sleep,
        )
        with pytest.raises(FullTextDocumentTooLarge):
            await downloader.download_text("/fulltext/paper.mmd")


async def _no_sleep(_: float) -> None:
    return None
