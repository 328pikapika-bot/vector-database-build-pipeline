from __future__ import annotations

import json
import asyncio

import httpx
import pytest

from app.services.embedder import EmbeddingClient


@pytest.mark.asyncio
async def test_embed_texts_detailed_sends_list_payloads_by_batch_size():
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_inputs.append(payload["input"])
        inputs = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
        return httpx.Response(
            200,
            json={"data": [{"embedding": [float(index), 1.0]} for index, _ in enumerate(inputs)]},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Content-Type": "application/json"},
    ) as http_client:
        client = EmbeddingClient(
            base_url="http://models.test/v1/embeddings",
            model="qwen3-embed",
            max_concurrency=4,
            max_retries=0,
            batch_size=2,
        )
        client._client = http_client

        results = await client.embed_texts_detailed(["first", "second", "third"])

    assert seen_inputs == [["first", "second"], "third"]
    assert [item.success for item in results] == [True, True, True]
    assert [item.vector for item in results] == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]


@pytest.mark.asyncio
async def test_embed_texts_detailed_preserves_empty_text_without_request():
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_inputs.append(payload["input"])
        return httpx.Response(
            200,
            json={"data": [{"embedding": [1.0, 2.0]}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(
            base_url="http://models.test/v1/embeddings",
            model="qwen3-embed",
            max_concurrency=4,
            max_retries=0,
            batch_size=8,
        )
        client._client = http_client

        results = await client.embed_texts_detailed(["  ", "valid"])

    assert seen_inputs == ["valid"]
    assert results[0].success is False
    assert results[0].skipped_empty is True
    assert results[0].vector is None
    assert results[0].error_code == "EMBEDDING_INPUT_EMPTY"
    assert results[1].vector == [1.0, 2.0]


@pytest.mark.asyncio
async def test_embed_texts_detailed_splits_batch_after_non_retryable_400():
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_inputs.append(payload["input"])
        if isinstance(payload["input"], list):
            return httpx.Response(400, json={"error": "batch too large"}, request=request)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [float(len(seen_inputs)), 2.0]}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(
            base_url="http://models.test/v1/embeddings",
            model="qwen3-embed",
            max_concurrency=4,
            max_retries=0,
            batch_size=2,
        )
        client._client = http_client

        results = await client.embed_texts_detailed(["first", "second"])

    assert seen_inputs == [["first", "second"], "first", "second"]
    assert [item.success for item in results] == [True, True]
    assert [item.vector for item in results] == [[2.0, 2.0], [3.0, 2.0]]


@pytest.mark.asyncio
async def test_embed_texts_detailed_coalesces_concurrent_short_requests(monkeypatch):
    monkeypatch.setattr("app.services.embedder.settings.EMBEDDING_CROSS_DOCUMENT_BATCH_ENABLED", True)
    monkeypatch.setattr("app.services.embedder.settings.EMBEDDING_CROSS_DOCUMENT_BATCH_WAIT_MS", 5)
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_inputs.append(payload["input"])
        inputs = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
        return httpx.Response(
            200,
            json={"data": [{"embedding": [float(index), 1.0]} for index, _ in enumerate(inputs)]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(
            base_url="http://models.test/v1/embeddings",
            model="qwen3-embed",
            max_concurrency=4,
            max_retries=0,
            batch_size=2,
        )
        client._client = http_client
        first, second = await asyncio.gather(
            client.embed_texts_detailed(["first"]),
            client.embed_texts_detailed(["second"]),
        )

    assert seen_inputs == [["first", "second"]]
    assert first[0].success is True
    assert second[0].success is True
    assert first[0].queue_wait_ms >= 0
    assert second[0].queue_wait_ms >= 0
