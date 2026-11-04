from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import settings
from app.services.image_downloader import (
    ImageConfigurationError,
    ImageDownloader,
    ImageNonRetryableDownloadError,
    ImageRetryableDownloadError,
    build_image_readiness_status,
    build_image_url,
)


def test_image_control_defaults_are_locked():
    assert settings.IMAGE_TASK_UPSERT_ENABLED is True
    assert settings.IMAGE_WORKER_ENABLED is True
    assert settings.IMAGE_WORKER_LEASE_SEC == 180
    assert settings.IMAGE_WORKER_BATCH_SIZE == 64
    assert settings.IMAGE_WORKER_POLL_INTERVAL_SEC == 1
    assert settings.IMAGE_DOWNLOAD_TIMEOUT == 180
    assert settings.IMAGE_DOWNLOAD_MAX_RETRIES == 0
    assert settings.IMAGE_DOWNLOAD_CONCURRENCY > 0
    assert settings.IMAGE_MAX_PIXELS > 0
    assert settings.IMAGE_FEATURE_PIPELINE_ENABLED is True
    assert settings.IMAGE_FEATURE_TIMEOUT == 45
    assert settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC == 170
    assert settings.IMAGE_FEATURE_MAX_RETRIES == 0
    assert settings.IMAGE_VL_CONTEXT_MAX_CHARS == 500
    assert settings.IMAGE_CONTEXT_BEFORE_PARAGRAPHS == 1
    assert settings.IMAGE_CONTEXT_AFTER_PARAGRAPHS == 2
    assert settings.IMAGE_MILVUS_WRITE_ENABLED is True
    assert settings.MONGO_IMAGE_TASK_COLLECTION == "doc_image_tasks"


def test_build_image_url_joins_base_and_relative_url():
    assert (
        build_image_url("/images/fig1.png", base_url="http://files.test/")
        == "http://files.test/images/fig1.png"
    )
    assert (
        build_image_url("images/fig1.png", base_url="http://files.test")
        == "http://files.test/images/fig1.png"
    )


def test_build_image_url_requires_base_url():
    with pytest.raises(ImageConfigurationError):
        build_image_url("/images/fig1.png", base_url="")


def test_image_readiness_exposes_pixel_and_image_write_readiness():
    status = build_image_readiness_status()
    expected_vl_key = settings.IMAGE_VL_EMBED_API_KEY or settings.IMAGE_MODEL_API_KEY
    expected_text_key = (
        settings.IMAGE_TEXT_EMBED_API_KEY
        or settings.IMAGE_MODEL_API_KEY
        or settings.EMBEDDING_API_KEY
    )
    expected_model_configured = all(
        str(value or "").strip()
        for value in (
            expected_vl_key,
            settings.IMAGE_VL_EMBED_URL,
            settings.IMAGE_VL_EMBED_MODEL,
            expected_text_key,
            settings.IMAGE_TEXT_EMBED_URL,
            settings.IMAGE_TEXT_EMBED_MODEL,
        )
    )

    assert status["configured"] is True
    assert status["base_url_configured"] is True
    assert status["task_upsert_enabled"] is True
    assert status["worker_enabled"] is True
    assert status["worker_running"] is False
    assert status["feature_pipeline_enabled"] is True
    assert status["milvus_write_enabled"] is True
    assert status["timeout"] == 180
    assert status["max_retries"] == 0
    assert status["download_concurrency"] == settings.IMAGE_DOWNLOAD_CONCURRENCY
    assert status["preprocess_concurrency"] == settings.IMAGE_PREPROCESS_CONCURRENCY
    assert status["max_pixels"] == settings.IMAGE_MAX_PIXELS
    assert status["feature_model_configured"] is expected_model_configured
    assert status["feature_timeout"] == 45
    assert status["feature_total_timeout"] == 170
    assert status["feature_max_retries"] == 0
    assert status["vl_context_max_chars"] == 500
    assert status["context_before_paragraphs"] == 1
    assert status["context_after_paragraphs"] == 2
    assert status["pixel_validation_available"] is True


def test_image_readiness_does_not_require_reserved_vl_chat(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MODEL_API_KEY", "")
    monkeypatch.setattr(settings, "IMAGE_VL_EMBED_API_KEY", "vl-key")
    monkeypatch.setattr(settings, "IMAGE_TEXT_EMBED_API_KEY", "text-key")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(settings, "IMAGE_VL_CHAT_URL", "")
    monkeypatch.setattr(settings, "IMAGE_VL_CHAT_MODEL", "")
    monkeypatch.setattr(settings, "IMAGE_VL_EMBED_URL", "http://models.test/vl-embed")
    monkeypatch.setattr(settings, "IMAGE_VL_EMBED_MODEL", "8b")
    monkeypatch.setattr(settings, "IMAGE_TEXT_EMBED_URL", "http://models.test/text-embed")
    monkeypatch.setattr(settings, "IMAGE_TEXT_EMBED_MODEL", "medium")

    status = build_image_readiness_status()

    assert status["feature_model_configured"] is True
    assert status["feature_auth_sources"]["vl_embed_api_key"] == "IMAGE_VL_EMBED_API_KEY"
    assert status["feature_auth_sources"]["text_embed_api_key"] == "IMAGE_TEXT_EMBED_API_KEY"


@pytest.mark.asyncio
async def test_download_bytes_success_returns_content():
    content = b"\x89PNG\r\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://files.test/images/fig1.png"
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "image/png"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ImageDownloader(
            base_url="http://files.test",
            client=client,
            sleep=_no_sleep,
        ).download_bytes("/images/fig1.png")

    assert result.url == "http://files.test/images/fig1.png"
    assert result.content == content
    assert result.content_type == "image/png"
    assert result.size_bytes == len(content)


@pytest.mark.asyncio
async def test_download_bytes_retries_network_failures_then_raises():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("network down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ImageDownloader(
            base_url="http://files.test",
            max_retries=2,
            client=client,
            sleep=_no_sleep,
        )
        with pytest.raises(ImageRetryableDownloadError):
            await downloader.download_bytes("/images/fig1.png")

    assert attempts == 3


@pytest.mark.asyncio
async def test_download_bytes_error_message_includes_empty_timeout_type():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ImageDownloader(
            base_url="http://files.test",
            max_retries=0,
            client=client,
            sleep=_no_sleep,
        )
        with pytest.raises(ImageRetryableDownloadError) as exc_info:
            await downloader.download_bytes("/images/fig1.png")

    message = str(exc_info.value)
    assert "ReadTimeout" in message
    assert "http://files.test/images/fig1.png" in message


@pytest.mark.asyncio
async def test_download_bytes_limits_concurrency():
    in_flight = 0
    max_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, content=b"ok", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ImageDownloader(
            base_url="http://files.test",
            concurrency=2,
            client=client,
            sleep=_no_sleep,
        )
        await asyncio.gather(
            *[
                downloader.download_bytes(f"/images/fig{index}.png")
                for index in range(6)
            ]
        )

    assert max_in_flight == 2


@pytest.mark.asyncio
async def test_download_bytes_4xx_is_non_retryable():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="missing", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ImageDownloader(
            base_url="http://files.test",
            max_retries=2,
            client=client,
            sleep=_no_sleep,
        )
        with pytest.raises(ImageNonRetryableDownloadError):
            await downloader.download_bytes("/images/missing.png")

    assert attempts == 1


@pytest.mark.asyncio
async def test_download_bytes_retries_5xx_then_succeeds():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="busy", request=request)
        return httpx.Response(200, content=b"ready", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ImageDownloader(
            base_url="http://files.test",
            max_retries=2,
            client=client,
            sleep=_no_sleep,
        ).download_bytes("/images/fig1.png")

    assert attempts == 2
    assert result.content == b"ready"


@pytest.mark.asyncio
async def test_download_bytes_uses_separate_fast_and_slow_timeouts(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_DOWNLOAD_LANE_ENABLED", True)
    monkeypatch.setattr(settings, "IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC", 3)
    monkeypatch.setattr(settings, "IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC", 9)
    seen_timeouts = []

    async with httpx.AsyncClient() as client:
        downloader = ImageDownloader(base_url="http://files.test", client=client)

        async def request_once(_url: str, *, timeout: int) -> httpx.Response:
            seen_timeouts.append(timeout)
            request = httpx.Request("GET", "http://files.test/images/fig.png")
            return httpx.Response(200, content=b"ok", request=request)

        monkeypatch.setattr(downloader, "_request_once", request_once)
        await downloader.download_bytes("/images/fig.png")
        await downloader.download_bytes("/images/fig.png", lane="slow")

    assert seen_timeouts == [3, 9]


async def _no_sleep(_: float) -> None:
    return None
