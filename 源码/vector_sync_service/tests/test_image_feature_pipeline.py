from hashlib import sha256
from pathlib import Path
import asyncio
import time

import pytest

from app.config import settings
from app.services.image_feature_pipeline import (
    IMAGE_FEATURE_PHASH_INVALID,
    IMAGE_FEATURE_PROVIDER_ERROR,
    IMAGE_FEATURE_PROVIDER_MISSING,
    IMAGE_FEATURE_SPARSE_INVALID,
    IMAGE_FEATURE_TEXT_MISSING,
    IMAGE_FEATURE_VECTOR_INVALID,
    IMAGE_MANIFEST_INVALID,
    ImageFeatureProviders,
    RESERVED_IMAGE_TEXT_PLACEHOLDER,
    finalize_image_feature_execution,
    prepare_image_feature_execution,
    process_image_features,
    run_image_feature_text_stage,
    run_image_feature_visual_stage,
)
from app.services.image_features import build_image_record_draft_from_manifest
from app.services.milvus_image_service import (
    IMAGE_SCHEMA_MODE_CAPTION_CONTEXT_2,
    IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    TEXT_VECTOR_DIM,
    VISUAL_TEXT_VECTOR_DIM,
    VISUAL_VECTOR_DIM,
    validate_writable_image_record,
)


@pytest.fixture(autouse=True)
def _enable_visual_embedding_for_legacy_contract_tests(monkeypatch):
    """Existing provider tests exercise the explicitly enabled visual path."""
    monkeypatch.setattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", True)


@pytest.mark.asyncio
async def test_process_image_features_success_from_manifest_builds_writable_record():
    seen_context = {}

    async def visual_provider(context):
        seen_context["image_bytes"] = context.image_bytes
        seen_context["draft_id"] = context.draft_record.img_id
        seen_context["context_text"] = context.context_text
        return [1.0] * VISUAL_VECTOR_DIM

    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(visual_embed_provider=visual_provider),
        context_text="nearby paragraph",
        schema_mode=IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    )

    assert result.success is True
    assert result.failure_items == []
    assert result.image_id == "DOC-20_img_1"
    assert result.doc_id == "DOC-20"
    assert result.feature_summary.p_hash_present is True
    assert result.feature_summary.visual_dim == 4096
    assert result.feature_summary.caption_dim == 2560
    assert result.feature_summary.context_dim == 2560
    assert result.feature_summary.text_dim == 2560
    assert result.feature_summary.visual_text_dim == 4096
    assert result.feature_summary.sparse_terms is None
    assert result.feature_summary.bm25_text_chars == len(RESERVED_IMAGE_TEXT_PLACEHOLDER)

    record = result.image_record
    validate_writable_image_record(record, expected_doc_id="DOC-20")
    assert record.img_id == "DOC-20_img_1"
    assert record.p_hash == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert record.caption_embed == [1.0] * TEXT_VECTOR_DIM
    assert record.context_embed == [1.0] * TEXT_VECTOR_DIM
    assert record.text == RESERVED_IMAGE_TEXT_PLACEHOLDER
    assert record.text_embed[0] > 0
    assert record.visual_text_embed[0] > 0
    assert record.text_sparse_vec is None
    assert record.img_meta["reserved_text_fields_placeholder"] is True
    assert seen_context == {
        "image_bytes": b"image-bytes",
        "draft_id": "DOC-20_img_1",
        "context_text": "nearby paragraph",
    }


@pytest.mark.asyncio
async def test_process_image_features_success_from_metadata_draft():
    draft = build_image_record_draft_from_manifest(_manifest())

    result = await process_image_features(
        b"image-bytes",
        draft,
        providers=_providers(),
        schema_mode=IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    )

    assert result.success is True
    assert result.image_record.img_id == draft.img_id
    assert result.image_record.caption_embed == [1.0] * TEXT_VECTOR_DIM
    assert result.image_record.context_embed == [1.0] * TEXT_VECTOR_DIM
    assert result.image_record.text_sparse_vec is None
    assert result.image_record.text is None
    assert result.image_record.text_embed is None
    assert result.image_record.visual_text_embed is None
    assert "reserved_text_fields_placeholder" not in result.image_record.img_meta
    assert result.feature_summary.caption_dim == TEXT_VECTOR_DIM
    assert result.feature_summary.context_dim == TEXT_VECTOR_DIM
    validate_writable_image_record(result.image_record, expected_doc_id=draft.docId)


@pytest.mark.asyncio
async def test_missing_caption_and_context_still_produce_visual_record():
    manifest = _manifest(
        title=None,
        raw_img_meta={"id": "DOC-20_img_1"},
        context_before=None,
        context_after=None,
    )

    result = await process_image_features(
        b"image-bytes",
        manifest,
        providers=_providers(
            caption_embed_provider=lambda _: None,
            context_embed_provider=lambda _: None,
        ),
        schema_mode=IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    )

    assert result.success is True
    assert result.image_record is not None
    assert result.image_record.visual_embed == [1.0] * VISUAL_VECTOR_DIM
    assert result.image_record.caption_embed is None
    assert result.image_record.context_embed is None
    assert result.image_record.img_meta["reserved_text_fields_placeholder"] is True


@pytest.mark.asyncio
async def test_legacy_caption_context_mode_resolves_to_history_v2_and_requires_visual_embedding():
    calls = []

    def forbidden_visual_provider(_context):
        calls.append("visual")
        raise RuntimeError("visual provider is required by the V2 contract")

    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(visual_embed_provider=forbidden_visual_provider),
        schema_mode=IMAGE_SCHEMA_MODE_CAPTION_CONTEXT_2,
    )

    assert result.success is False
    assert calls == ["visual"]
    assert result.image_record is None
    assert result.failure_items[0].feature_name == "visual_embed"


@pytest.mark.asyncio
async def test_process_image_features_collects_independent_providers_concurrently():
    async def slow_visual(_context):
        await asyncio.sleep(0.05)
        return [1.0] * VISUAL_VECTOR_DIM

    async def slow_caption(_context):
        await asyncio.sleep(0.05)
        return [1.0] * TEXT_VECTOR_DIM

    started = time.perf_counter()
    result = await process_image_features(
        b"image-bytes",
        build_image_record_draft_from_manifest(_manifest()),
        providers=_providers(
            visual_embed_provider=slow_visual,
            caption_embed_provider=slow_caption,
        ),
        schema_mode=IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    )
    elapsed = time.perf_counter() - started

    assert result.success is True
    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_staged_feature_execution_builds_record_in_two_steps():
    prepared = prepare_image_feature_execution(
        b"image-bytes",
        _manifest(),
        providers=_providers(),
        context_text="nearby paragraph",
        schema_mode=IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    )

    assert not isinstance(prepared, type(process_image_features))
    await run_image_feature_visual_stage(prepared, providers=_providers())
    assert prepared.summary.p_hash_present is True
    assert prepared.summary.visual_dim == VISUAL_VECTOR_DIM
    assert prepared.summary.caption_dim is None
    assert prepared.summary.context_dim is None

    await run_image_feature_text_stage(prepared, providers=_providers())
    result = finalize_image_feature_execution(prepared)

    assert result.success is True
    assert result.feature_summary.caption_dim == TEXT_VECTOR_DIM
    assert result.feature_summary.context_dim == TEXT_VECTOR_DIM
    assert result.image_record is not None
    validate_writable_image_record(result.image_record, expected_doc_id="DOC-20")


@pytest.mark.asyncio
async def test_reserved_text_fields_do_not_call_interpretation_providers():
    calls = []

    def reserved_provider(_context):
        calls.append("reserved")
        raise RuntimeError("reserved field should not be interpreted")

    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(
            text_embed_provider=reserved_provider,
            visual_text_embed_provider=reserved_provider,
            text_sparse_vec_provider=reserved_provider,
            bm25_text_provider=reserved_provider,
        ),
        schema_mode=IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    )

    assert result.success is True
    assert calls == []
    assert result.image_record.text == RESERVED_IMAGE_TEXT_PLACEHOLDER
    assert result.image_record.text_embed[0] > 0
    assert result.image_record.visual_text_embed[0] > 0


@pytest.mark.asyncio
async def test_disabled_visual_embedding_writes_sentinel_without_calling_provider(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", False)
    calls = []

    def forbidden_visual_provider(_context):
        calls.append("visual")
        raise RuntimeError("visual embedding must be disabled")

    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(visual_embed_provider=forbidden_visual_provider),
        schema_mode=IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    )

    assert result.success is True
    assert calls == []
    assert result.image_record.visual_embed[0] > 0
    assert result.image_record.img_meta["reserved_visual_embed_placeholder"] is True


@pytest.mark.asyncio
async def test_invalid_manifest_returns_failure_without_calling_providers():
    calls = []

    def called_provider(_):
        calls.append("called")
        return [1.0] * VISUAL_VECTOR_DIM

    result = await process_image_features(
        b"image-bytes",
        {"docId": "DOC-20"},
        providers=_providers(visual_embed_provider=called_provider),
    )

    assert result.success is False
    assert result.image_record is None
    assert [item.error_code for item in result.failure_items] == [IMAGE_MANIFEST_INVALID]
    assert calls == []


@pytest.mark.asyncio
async def test_missing_provider_returns_structured_failure_without_record():
    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(visual_embed_provider=None),
    )

    assert result.success is False
    assert result.image_record is None
    assert [item.feature_name for item in result.failure_items] == ["visual_embed"]
    assert [item.error_code for item in result.failure_items] == [IMAGE_FEATURE_PROVIDER_MISSING]


@pytest.mark.asyncio
async def test_provider_exception_returns_structured_failure_without_record():
    def broken_provider(_):
        raise RuntimeError("provider blew up")

    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(p_hash_provider=broken_provider),
    )

    assert result.success is False
    assert result.image_record is None
    assert result.failure_items[0].feature_name == "p_hash"
    assert result.failure_items[0].error_code == IMAGE_FEATURE_PROVIDER_ERROR
    assert "provider blew up" in result.failure_items[0].error_message


@pytest.mark.parametrize(
    "provider_name, provider_value, expected_feature, expected_code",
    [
        ("p_hash_provider", b"\x01", "p_hash", IMAGE_FEATURE_PHASH_INVALID),
        ("p_hash_provider", b"\x00" * 8, "p_hash", IMAGE_FEATURE_PHASH_INVALID),
        ("visual_embed_provider", [1.0] * 10, "visual_embed", IMAGE_FEATURE_VECTOR_INVALID),
        ("caption_embed_provider", [1.0] * 10, "caption_embed", IMAGE_FEATURE_VECTOR_INVALID),
        ("context_embed_provider", [1.0] * 10, "context_embed", IMAGE_FEATURE_VECTOR_INVALID),
    ],
)
@pytest.mark.asyncio
async def test_invalid_feature_outputs_keep_visual_record_for_optional_text(provider_name, provider_value, expected_feature, expected_code):
    result = await process_image_features(
        b"image-bytes",
        _manifest(),
        providers=_providers(**{provider_name: lambda _: provider_value}),
        schema_mode=(
            IMAGE_SCHEMA_MODE_SEVEN_VECTOR
            if provider_name in {"caption_embed_provider", "context_embed_provider"}
            else ""
        ),
    )

    if provider_name in {"caption_embed_provider", "context_embed_provider"}:
        assert result.success is True
        assert result.image_record is not None
        assert result.image_record.visual_embed == [1.0] * VISUAL_VECTOR_DIM
        assert result.image_record.img_meta["text_vectorization_failures"][0]["feature_name"] == expected_feature
        assert result.image_record.img_meta["text_vectorization_failures"][0]["error_code"] == expected_code
    else:
        assert result.success is False
        assert result.image_record is None
        assert any(
            item.feature_name == expected_feature and item.error_code == expected_code
            for item in result.failure_items
        )


def test_image_feature_pipeline_source_does_not_reference_live_clients_or_document_collection():
    source = Path(__file__).parents[1].joinpath("app/services/image_feature_pipeline.py").read_text(encoding="utf-8")

    forbidden = [
        "scientific_documents",
        "MILVUS_COLLECTION",
        "get_embedding_client",
        "EmbeddingClient",
        "MilvusClient",
        "httpx",
        "requests",
        "openai",
        "OpenAI",
    ]
    for text in forbidden:
        assert text not in source


def _manifest(image_bytes=b"image-bytes", **overrides):
    manifest = {
        "docId": "DOC-20",
        "imageId": "DOC-20_img_1",
        "relativeUrl": "/images/1.png",
        "contentType": "image/png",
        "sizeBytes": len(image_bytes),
        "bytesSha256": sha256(image_bytes).hexdigest(),
        "sequence": 1,
        "doc_title": ["Document 20"],
        "keyword": ["science"],
        "language": ["zh"],
        "sysDataStatus": "ACTIVE",
        "title": "Figure 1",
        "name": "1.png",
        "raw_img_meta": {"id": "DOC-20_img_1"},
        "context_before": "Previous paragraph",
        "context_after": "Figure 1\n\nNext paragraph one\n\nNext paragraph two",
    }
    manifest.update(overrides)
    return manifest


def _providers(**overrides):
    providers = {
        "p_hash_provider": lambda _: b"\x01\x02\x03\x04\x05\x06\x07\x08",
        "visual_embed_provider": lambda _: [1.0] * VISUAL_VECTOR_DIM,
        "caption_embed_provider": lambda _: [1.0] * TEXT_VECTOR_DIM,
        "context_embed_provider": lambda _: [1.0] * TEXT_VECTOR_DIM,
        "text_embed_provider": lambda _: [1.0] * TEXT_VECTOR_DIM,
        "visual_text_embed_provider": lambda _: [1.0] * VISUAL_TEXT_VECTOR_DIM,
        "text_sparse_vec_provider": lambda _: {1: 0.5},
        "bm25_text_provider": lambda _: "Figure caption and context",
    }
    providers.update(overrides)
    return ImageFeatureProviders(**providers)
