from __future__ import annotations

import asyncio
from io import BytesIO
import json
import threading

import httpx
import pytest

from app.models.image import ImageRecord
from app.services import image_offline_providers as providers_module
from app.services.image_offline_providers import (
    IMAGE_PROVIDER_CONFIG_MISSING,
    IMAGE_PROVIDER_RESPONSE_INVALID,
    ImageProviderConfigurationError,
    ImageProviderError,
    OfflineImageModelClient,
    IMAGE_VL_EMBED_INPUT_TOO_LARGE,
    build_offline_image_feature_providers,
)


@pytest.fixture(autouse=True)
def _enable_visual_embedding_for_visual_provider_tests(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_VISUAL_EMBED_ENABLED", True)


@pytest.mark.asyncio
async def test_visual_priority_gate_does_not_start_text_while_visual_is_waiting():
    gate = providers_module._VisualPriorityModelGate(1)
    events = []
    visual_started = asyncio.Event()
    release_visual = asyncio.Event()

    async def visual_call():
        events.append("visual-start")
        visual_started.set()
        await release_visual.wait()
        events.append("visual-end")

    async def text_call():
        events.append("text-start")

    visual_task = asyncio.create_task(gate.run(visual_call(), priority="visual"))
    await visual_started.wait()
    text_task = asyncio.create_task(gate.run(text_call(), priority="text"))
    await asyncio.sleep(0)

    assert events == ["visual-start"]
    release_visual.set()
    await asyncio.gather(visual_task, text_task)
    assert events == ["visual-start", "visual-end", "text-start"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("```json\n{\"a\": 2}\n```", {"a": 2}),
        ('Here is the result:\n{"a": {"nested": true}, "text": "brace } inside"}\nDone.', {"a": {"nested": True}, "text": "brace } inside"}),
        ([{"type": "text", "text": "```json\n{\"a\": 3}\n```"}], {"a": 3}),
    ],
)
def test_loads_json_object_accepts_common_vl_wrappers(raw, expected):
    assert providers_module._loads_json_object(raw, "stage") == expected


def test_loads_json_object_invalid_error_includes_stage_length_and_preview(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_VL_RESPONSE_PREVIEW_CHARS", 16)
    with pytest.raises(ImageProviderError) as exc_info:
        providers_module._loads_json_object("not json " + ("A" * 80), "draft analysis")

    message = str(exc_info.value)
    assert exc_info.value.error_code == IMAGE_PROVIDER_RESPONSE_INVALID
    assert "draft analysis returned invalid JSON" in message
    assert "raw_len=89" in message
    assert "preview=not json AAAAAAA..." in message
    assert "A" * 40 not in message


@pytest.mark.asyncio
async def test_embed_payloads_use_unified_api_input_shapes_and_normalize(monkeypatch):
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: "data:image/png;base64,abc",
    )
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        expected_key = "test-key" if str(request.url).endswith("/vl-embeddings") else "text-key"
        assert request.headers["authorization"] == f"Bearer {expected_key}"
        return httpx.Response(
            200,
            json={"data": [{"embedding": [3.0, 4.0]}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(
            http_client=http_client,
            vl_embed_api_key="test-key",
            text_embed_api_key="text-key",
        )
        image_vec = await client.embed_image(b"image")
        text_vec = await client.embed_text("caption")
        mixed_vec = await client.embed_image_text(b"image", "deep text")

    assert image_vec == pytest.approx([0.60009765625, 0.7998046875])
    assert text_vec == pytest.approx([0.60009765625, 0.7998046875])
    assert mixed_vec == pytest.approx([0.60009765625, 0.7998046875])
    assert seen_payloads[0] == {
        "model": "vl-embed",
        "input": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
            }
        ],
        "encoding_format": "float",
    }
    assert seen_payloads[1] == {
        "model": "text-model",
        "input": "caption",
        "encoding_format": "float",
    }
    assert seen_payloads[2]["input"][0]["content"][0] == {"type": "text", "text": "deep text"}
    assert seen_payloads[2]["input"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }


@pytest.mark.asyncio
async def test_embed_image_data_urls_sends_one_multimodal_input_per_image():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"data": [{"embedding": [3.0, 4.0]}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        vectors = await _client(http_client=http_client).embed_image_data_urls(
            ["data:image/png;base64,aaa", "data:image/png;base64,bbb"]
        )

    assert [payload["input"][0]["content"][0]["image_url"]["url"] for payload in seen_payloads] == [
        "data:image/png;base64,aaa",
        "data:image/png;base64,bbb",
    ]
    assert all("messages" not in payload for payload in seen_payloads)
    assert len(vectors) == 2
    for vector in vectors:
        assert vector == pytest.approx([0.60009765625, 0.7998046875])


@pytest.mark.asyncio
async def test_image_client_uses_dedicated_text_embedding_rate(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "EMBEDDING_RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(providers_module.settings, "EMBEDDING_TEXT_RATE", 48)
    monkeypatch.setattr(providers_module.settings, "IMAGE_TEXT_EMBED_RATE", 100)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: None)
    ) as http_client:
        client = _client(http_client=http_client)

        assert client._text_rate_limiter is not None
        assert client._text_rate_limiter.rate_per_second == 100


@pytest.mark.asyncio
async def test_interpret_image_cleans_context_and_uses_draft_only(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", True)
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: "data:image/png;base64,abc",
    )
    calls = []
    responses = [
        {"cleaned_context": "clean context", "removed_items": ["noise"], "removal_ratio": 0.5},
        {"deep_semantics": "draft", "search_keywords": ["draft"]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls.append(payload)
        body = {"choices": [{"message": {"content": json.dumps(responses[len(calls) - 1])}}]}
        return httpx.Response(200, json=body, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(
            http_client=http_client,
            max_retries=1,
        )
        result = await client.interpret_image(b"image", "raw context")

    assert result.deep_text == "draft"
    assert result.keyword_text == "draft"
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"] == providers_module.CONTEXT_CLEAN_PROMPT
    assert calls[1]["messages"][0]["content"] == providers_module.DRAFT_SYSTEM_PROMPT
    assert "clean context" in calls[1]["messages"][1]["content"][1]["text"]
    assert result.context_cleaning_fallback is False
    assert result.draft_chat_ms >= 0


@pytest.mark.asyncio
async def test_clean_context_falls_back_to_raw_context_on_invalid_json(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", True)
    monkeypatch.setattr(providers_module.settings, "IMAGE_VL_RESPONSE_PREVIEW_CHARS", 32)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "cleaned context: raw context"}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        result = await client._clean_context_with_diagnostics("raw context")
        public_result = await client.clean_context("raw context")

    assert result.cleaned_context == "raw context"
    assert result.fallback is True
    assert "context cleaning returned invalid JSON" in result.error_message
    assert result.raw_preview == "cleaned context: raw context"
    assert public_result == "raw context"


@pytest.mark.asyncio
async def test_clean_context_can_be_disabled(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("context clean request should not be sent")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        result = await client._clean_context_with_diagnostics("raw context")
        public_result = await client.clean_context("raw context")

    assert result.cleaned_context == "raw context"
    assert result.fallback is False
    assert public_result == "raw context"


@pytest.mark.asyncio
async def test_interpret_image_continues_when_context_cleaning_is_invalid(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", True)
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: "data:image/png;base64,abc",
    )
    calls = []
    responses = [
        "I cleaned it, but not as JSON.",
        {"deep_semantics": "draft", "search_keywords": ["draft"]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls.append(payload)
        content = responses[len(calls) - 1]
        if not isinstance(content, str):
            content = json.dumps(content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        result = await client.interpret_image(b"image", "raw context")

    assert result.deep_text == "draft"
    assert result.keyword_text == "draft"
    assert result.context_cleaning_fallback is True
    assert "context cleaning returned invalid JSON" in result.context_cleaning_error
    assert result.context_cleaning_preview == "I cleaned it, but not as JSON."
    assert "raw context" in calls[1]["messages"][1]["content"][1]["text"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_interpret_image_draft_invalid_json_still_fails(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", True)
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: "data:image/png;base64,abc",
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            content = {"cleaned_context": "clean", "removed_items": [], "removal_ratio": 0}
        else:
            content = "draft is not JSON"
        if not isinstance(content, str):
            content = json.dumps(content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        with pytest.raises(ImageProviderError) as exc_info:
            await client.interpret_image(b"image", "context")

    assert exc_info.value.error_code == IMAGE_PROVIDER_RESPONSE_INVALID
    assert "draft analysis returned invalid JSON" in str(exc_info.value)
    assert calls == 2


def test_build_offline_image_feature_providers_omits_reserved_text_providers():
    client = OfflineImageModelClient(
        api_key="test-key",
        vl_chat_url="",
        vl_chat_model="",
        vl_embed_url="embed",
        vl_embed_model="vl-embed",
        text_embed_url="text",
        text_embed_model="text-model",
    )

    providers = build_offline_image_feature_providers(client=client)

    assert providers.p_hash_provider is not None
    assert providers.visual_embed_provider is not None
    assert providers.caption_embed_provider is not None
    assert providers.context_embed_provider is not None
    assert providers.text_embed_provider is None
    assert providers.visual_text_embed_provider is None
    assert providers.text_sparse_vec_provider is None
    assert providers.bm25_text_provider is None


def test_disabled_visual_embedding_does_not_require_or_create_visual_provider(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_VISUAL_EMBED_ENABLED", False)
    client = OfflineImageModelClient(
        api_key="",
        text_embed_api_key="text-key",
        vl_embed_url="",
        vl_embed_model="",
        text_embed_url="text",
        text_embed_model="text-model",
    )

    providers = build_offline_image_feature_providers(client=client)

    assert providers.visual_embed_provider is None
    assert providers.caption_embed_provider is not None
    assert providers.context_embed_provider is not None


@pytest.mark.asyncio
async def test_interpret_image_uses_single_draft_chat(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", False)
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: "data:image/png;base64,abc",
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = {"deep_semantics": "draft only", "search_keywords": ["skip"]}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        result = await client.interpret_image(b"image", "context")

    assert result.deep_text == "draft only"
    assert result.keyword_text == "skip"
    assert calls == 1


@pytest.mark.asyncio
async def test_interpret_image_truncates_vl_context(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", False)
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: "data:image/png;base64,abc",
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls.append(payload)
        content = {"deep_semantics": "draft", "search_keywords": []}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(
            http_client=http_client,
            context_max_chars=5,
        )
        result = await client.interpret_image(b"image", "123456789")

    assert result.vl_context_chars == 5
    assert result.vl_context_truncated is True
    assert "Context:\n12345" in calls[0]["messages"][1]["content"][1]["text"]
    assert "123456" not in calls[0]["messages"][1]["content"][1]["text"]


@pytest.mark.asyncio
async def test_visual_provider_reuses_data_url_across_calls(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", False)
    conversion_calls = 0

    def fake_data_url(image_bytes, *, max_pixels):
        nonlocal conversion_calls
        conversion_calls += 1
        return "data:image/png;base64,abc"

    monkeypatch.setattr(providers_module, "image_bytes_to_data_url", fake_data_url)

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/vl-embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"embedding": [3.0, 4.0]}]},
                request=request,
            )
        content = {"deep_semantics": "draft", "search_keywords": ["alpha"]}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
            request=request,
        )

    draft = ImageRecord(img_id="DOC-1_img_0", sysId="DOC-1", img_meta={})
    context = providers_module.ImageFeaturePipelineContext(
        image_bytes=b"image",
        draft_record=draft,
        context_text="raw context",
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        providers = build_offline_image_feature_providers(client=client)
        await asyncio.gather(
            providers.visual_embed_provider(context),
            providers.visual_embed_provider(context),
        )

    assert conversion_calls == 1
    assert draft.img_meta["data_url_chars"] == len("data:image/png;base64,abc")
    assert draft.img_meta["feature_stage_timings_ms"]["data_url"] >= 0
    assert draft.img_meta["feature_stage_timings_ms"]["visual_embed"] >= 0


@pytest.mark.asyncio
async def test_visual_preprocessing_runs_in_parallel_threads(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_PREPROCESS_CONCURRENCY", 2)
    barrier = threading.Barrier(2, timeout=2)
    lock = threading.Lock()
    main_thread_id = threading.get_ident()
    worker_thread_ids = set()
    in_flight = 0
    max_in_flight = 0

    def fake_data_url(image_bytes, *, max_pixels):
        nonlocal in_flight, max_in_flight
        with lock:
            worker_thread_ids.add(threading.get_ident())
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            barrier.wait()
            return f"data:image/png;base64,{image_bytes.decode()}"
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(providers_module, "image_bytes_to_data_url", fake_data_url)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"embedding": [3.0, 4.0]}]},
            request=request,
        )

    contexts = [
        providers_module.ImageFeaturePipelineContext(
            image_bytes=f"image-{index}".encode(),
            draft_record=ImageRecord(
                img_id=f"DOC-1_img_{index}",
                sysId="DOC-1",
                sequence=index,
                img_meta={},
            ),
        )
        for index in range(2)
    ]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        feature_providers = build_offline_image_feature_providers(client=client)
        vectors = await asyncio.gather(
            *(feature_providers.visual_embed_provider(context) for context in contexts)
        )

    assert max_in_flight == 2
    assert main_thread_id not in worker_thread_ids
    assert len(worker_thread_ids) == 2
    assert len(vectors) == 2
    assert all(
        context.draft_record.img_meta["feature_stage_timings_ms"]["data_url_queue_wait"] >= 0
        for context in contexts
    )


@pytest.mark.asyncio
async def test_phash_provider_runs_outside_event_loop(monkeypatch):
    main_thread_id = threading.get_ident()
    monkeypatch.setattr(
        providers_module,
        "compute_phash",
        lambda image_bytes: threading.get_ident(),
    )
    context = providers_module.ImageFeaturePipelineContext(
        image_bytes=b"image",
        draft_record=ImageRecord(img_id="DOC-1_img_0", sysId="DOC-1", img_meta={}),
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as http_client:
        client = _client(http_client=http_client)
        feature_providers = build_offline_image_feature_providers(client=client)
        worker_thread_id = await feature_providers.p_hash_provider(context)

    assert worker_thread_id != main_thread_id
    assert context.draft_record.img_meta["feature_stage_timings_ms"]["p_hash"] >= 0


@pytest.mark.asyncio
async def test_text_embedding_batch_falls_back_to_single_requests(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_CONTEXT_CLEAN_ENABLED", False)
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_inputs.append(payload["input"])
        if isinstance(payload["input"], list):
            return httpx.Response(400, json={"error": "list unsupported"}, request=request)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [3.0, 4.0]}]},
            request=request,
        )

    draft = ImageRecord(
        img_id="DOC-1_img_0",
        sysId="DOC-1",
        img_meta={"caption": "caption"},
        context_before="before",
        context_after="after",
    )
    context = providers_module.ImageFeaturePipelineContext(
        image_bytes=b"image",
        draft_record=draft,
        context_text="before\nafter",
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        providers = build_offline_image_feature_providers(client=client)
        caption_vec, context_vec = await asyncio.gather(
            providers.caption_embed_provider(context),
            providers.context_embed_provider(context),
        )

    assert caption_vec == pytest.approx([0.60009765625, 0.7998046875])
    assert context_vec == pytest.approx([0.60009765625, 0.7998046875])
    assert seen_inputs[0] == ["caption", "before\n\nafter"]
    assert seen_inputs[1:] == ["caption", "before\n\nafter"]
    assert draft.img_meta["text_embedding_batch_fallback"] is True
    assert sorted(draft.img_meta["text_embedding_batch_fallback_features"]) == [
        "caption_embed",
        "context_embed",
    ]
    assert draft.img_meta["feature_stage_timings_ms"]["caption_embed"] >= 0
    assert draft.img_meta["feature_stage_timings_ms"]["context_embed"] >= 0


@pytest.mark.asyncio
async def test_text_embedding_batcher_can_dispatch_multiple_batches_in_parallel(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_TEXT_EMBED_BATCH_SIZE", 2)
    monkeypatch.setattr(providers_module.settings, "IMAGE_TEXT_EMBED_BATCH_DELAY_MS", 1000)
    monkeypatch.setattr(providers_module.settings, "IMAGE_TEXT_EMBED_CONCURRENCY", 4)

    in_flight = 0
    max_in_flight = 0
    seen_inputs = []
    release = asyncio.Event()

    async def fake_embed_texts(texts):
        nonlocal in_flight, max_in_flight
        seen_inputs.append(list(texts))
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await release.wait()
            return [[3.0, 4.0] for _ in texts]
        finally:
            in_flight -= 1

    client = _client(http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)))
    client.embed_texts = fake_embed_texts  # type: ignore[method-assign]
    semaphore = asyncio.Semaphore(4)
    batcher = providers_module._TextEmbeddingBatcher(client, semaphore)

    tasks = [
        asyncio.create_task(batcher.embed_text(f"text-{index}"))
        for index in range(4)
    ]
    await asyncio.sleep(0.05)
    release.set()
    vectors = await asyncio.gather(*tasks)

    assert max_in_flight >= 2
    assert seen_inputs == [["text-0", "text-1"], ["text-2", "text-3"]]
    for vector in vectors:
        assert vector == [3.0, 4.0]


@pytest.mark.asyncio
async def test_visual_embedding_uses_single_image_input_requests(monkeypatch):
    monkeypatch.setattr(
        providers_module,
        "image_bytes_to_data_url",
        lambda image_bytes, *, max_pixels: f"data:image/png;base64,{image_bytes.decode()}",
    )
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [3.0, 4.0]}]},
            request=request,
        )

    context_a = providers_module.ImageFeaturePipelineContext(
        image_bytes=b"aaa",
        draft_record=ImageRecord(img_id="DOC-1_img_0", sysId="DOC-1", img_meta={}),
    )
    context_b = providers_module.ImageFeaturePipelineContext(
        image_bytes=b"bbb",
        draft_record=ImageRecord(img_id="DOC-1_img_1", sysId="DOC-1", sequence=1, img_meta={}),
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client=http_client)
        providers = build_offline_image_feature_providers(client=client)
        vector_a, vector_b = await asyncio.gather(
            providers.visual_embed_provider(context_a),
            providers.visual_embed_provider(context_b),
        )

    assert len(seen_payloads) == 2
    assert [payload.get("messages") for payload in seen_payloads] == [None, None]
    assert [
        payload["input"][0]["content"][0]["image_url"]["url"]
        for payload in seen_payloads
    ] == ["data:image/png;base64,aaa", "data:image/png;base64,bbb"]
    assert vector_a == pytest.approx([0.60009765625, 0.7998046875])
    assert vector_b == pytest.approx([0.60009765625, 0.7998046875])


def test_image_bytes_to_data_url_uses_lightweight_jpeg(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_FORMAT", "jpeg")
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MAX_SIDE", 320)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MAX_PIXELS", 90_000)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_JPEG_QUALITY", 82)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MIN_JPEG_QUALITY", 70)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MAX_BYTES", 512_000)

    data_url = providers_module.image_bytes_to_data_url(
        _png_bytes(width=1600, height=900),
        max_pixels=2_000_000,
    )

    assert data_url.startswith("data:image/jpeg;base64,")
    image = _image_from_data_url(data_url)
    assert image.mode == "RGB"
    assert max(image.size) <= 320
    assert image.width * image.height <= 90_000


def test_image_bytes_to_data_url_reports_unshrinkable_model_input(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_FORMAT", "jpeg")
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MAX_SIDE", 512)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MAX_PIXELS", 262_144)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_JPEG_QUALITY", 82)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MIN_JPEG_QUALITY", 70)
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_INPUT_MAX_BYTES", 10)

    with pytest.raises(ImageProviderError) as exc_info:
        providers_module.image_bytes_to_data_url(_png_bytes(width=64, height=64), max_pixels=2_000_000)

    assert exc_info.value.error_code == IMAGE_VL_EMBED_INPUT_TOO_LARGE
    assert exc_info.value.retryable is False


def test_build_offline_image_feature_providers_requires_explicit_api_key(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_API_KEY", "")
    monkeypatch.setattr(providers_module.settings, "IMAGE_VL_EMBED_API_KEY", "")
    monkeypatch.setattr(providers_module.settings, "IMAGE_TEXT_EMBED_API_KEY", "")
    monkeypatch.setattr(providers_module.settings, "EMBEDDING_API_KEY", "")
    client = OfflineImageModelClient(
        api_key="",
        vl_chat_url="chat",
        vl_chat_model="chat-model",
        vl_embed_url="embed",
        vl_embed_model="vl-embed",
        text_embed_url="text",
        text_embed_model="text-model",
    )

    with pytest.raises(ImageProviderConfigurationError) as exc_info:
        build_offline_image_feature_providers(client=client)

    assert exc_info.value.error_code == IMAGE_PROVIDER_CONFIG_MISSING
    assert exc_info.value.retryable is False
    assert exc_info.value.missing_env_names == [
        "IMAGE_VL_EMBED_API_KEY|IMAGE_MODEL_API_KEY",
        "IMAGE_TEXT_EMBED_API_KEY|IMAGE_MODEL_API_KEY|EMBEDDING_API_KEY",
    ]


def test_build_offline_image_feature_providers_accepts_split_api_keys(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_API_KEY", "")
    monkeypatch.setattr(providers_module.settings, "EMBEDDING_API_KEY", "")
    client = OfflineImageModelClient(
        api_key="",
        vl_embed_api_key="vl-key",
        text_embed_api_key="text-key",
        vl_chat_url="",
        vl_chat_model="",
        vl_embed_url="embed",
        vl_embed_model="vl-embed",
        text_embed_url="text",
        text_embed_model="text-model",
    )

    providers = build_offline_image_feature_providers(client=client)

    assert providers.visual_embed_provider is not None
    assert providers.caption_embed_provider is not None
    assert providers.text_embed_provider is None


def test_build_offline_image_feature_providers_does_not_require_reserved_vl_chat(monkeypatch):
    monkeypatch.setattr(providers_module.settings, "IMAGE_MODEL_API_KEY", "")
    monkeypatch.setattr(providers_module.settings, "IMAGE_VL_CHAT_URL", "")
    monkeypatch.setattr(providers_module.settings, "IMAGE_VL_CHAT_MODEL", "")
    client = OfflineImageModelClient(
        api_key="test-key",
        vl_chat_url="",
        vl_chat_model="",
        vl_embed_url="embed",
        vl_embed_model="vl-embed",
        text_embed_url="text",
        text_embed_model="text-model",
    )

    providers = build_offline_image_feature_providers(client=client)

    assert providers.visual_embed_provider is not None
    assert providers.caption_embed_provider is not None
    assert providers.context_embed_provider is not None


def _client(
    *,
    http_client: httpx.AsyncClient,
    max_retries: int = 0,
    context_max_chars: int = 500,
    api_key: str = "test-key",
    vl_embed_api_key: str = "",
    text_embed_api_key: str = "text-key",
) -> OfflineImageModelClient:
    return OfflineImageModelClient(
        api_key=api_key,
        vl_embed_api_key=vl_embed_api_key,
        text_embed_api_key=text_embed_api_key,
        vl_chat_url="http://models.test/v1/chat/completions",
        vl_chat_model="chat-model",
        vl_embed_url="http://models.test/v1/vl-embeddings",
        vl_embed_model="vl-embed",
        text_embed_url="http://models.test/v1/text-embeddings",
        text_embed_model="text-model",
        timeout=5,
        max_retries=max_retries,
        context_max_chars=context_max_chars,
        http_client=http_client,
        sleep=_no_sleep,
    )


async def _no_sleep(_: float) -> None:
    return None


def _png_bytes(*, width: int, height: int) -> bytes:
    from PIL import Image

    image = Image.new("RGBA", (width, height), (20, 80, 180, 180))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_from_data_url(data_url: str):
    import base64

    from PIL import Image

    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded)))
