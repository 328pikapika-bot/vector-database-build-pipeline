"""Default-skipped live smoke tests for staging dependency checks.

These tests intentionally construct live clients only after both RUN_LIVE_SMOKE=1
and the component-specific RUN_LIVE_* flag are set.
"""

import pytest

from app.services.data_plane_smoke import build_staging_smoke_plan, live_smoke_skip_reason, sanitize_smoke_payload


def _skip_unless_live(component: str) -> None:
    reason = live_smoke_skip_reason(component)
    if reason:
        pytest.skip(reason)


def _skip_unless_live_consistency() -> dict:
    plan = build_staging_smoke_plan(
        ["mongo", "milvus"],
        expect_document=True,
        expect_chunks=None,
        expect_images=None,
    )
    if not plan["consistency"]["ready"]:
        pytest.skip(plan["consistency"]["skip_reason"])
    return plan


@pytest.mark.asyncio
async def test_live_mongo_ping_and_collections_are_reachable():
    _skip_unless_live("mongo")
    from app.services.mongo_state_manager import MongoStateManager

    manager = MongoStateManager()
    try:
        await manager.connect()
        assert await manager.ping() is True
        assert manager.status_collection.name
        assert manager.failures_collection.name
        assert manager.artifacts_collection.name
    finally:
        await manager.close()


def test_live_milvus_collection_presence_is_read_only():
    _skip_unless_live("milvus")
    from pymilvus import MilvusClient

    from app.config import settings
    from app.services.milvus_chunk_service import DEFAULT_CHUNK_COLLECTION
    from app.services.milvus_image_service import DEFAULT_IMAGE_COLLECTION

    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        user=settings.MILVUS_USER or "",
        password=settings.MILVUS_PASSWORD or "",
    )
    try:
        collections = {
            "document": settings.MILVUS_COLLECTION,
            "chunk": DEFAULT_CHUNK_COLLECTION,
            "image": DEFAULT_IMAGE_COLLECTION,
        }
        presence = {
            key: client.has_collection(collection_name)
            for key, collection_name in collections.items()
        }
        sanitized = sanitize_smoke_payload({"collections": collections, "presence": presence})
        assert sanitized["collections"]["chunk"] != sanitized["collections"]["document"]
        assert sanitized["collections"]["image"] != sanitized["collections"]["document"]
    finally:
        client.close()


def test_live_kafka_topic_metadata_is_reachable_without_commits():
    _skip_unless_live("kafka")
    from confluent_kafka import Consumer

    from app.config import settings
    from app.services.kafka_consumer import _build_consumer_config

    consumer = Consumer(_build_consumer_config())
    try:
        metadata = consumer.list_topics(topic=settings.KAFKA_TOPIC, timeout=10)
        assert settings.KAFKA_TOPIC in metadata.topics
    finally:
        consumer.close()


@pytest.mark.asyncio
async def test_live_embedding_returns_expected_dimension():
    _skip_unless_live("embedding")
    from app.config import settings
    from app.services.embedder import EmbeddingClient

    client = EmbeddingClient()
    try:
        result = await client.embed_single_detailed("NSTL live smoke ping")
        assert result.success is True
        assert result.vector is not None
        assert len(result.vector) == settings.VECTOR_DIM
    finally:
        await client.close()


def test_live_vl_provider_contract_is_not_configured_yet():
    _skip_unless_live("vl")
    pytest.fail("Live VL/image provider smoke is not implemented until real provider config is approved")


@pytest.mark.asyncio
async def test_live_consistency_verifier_for_safe_doc_is_read_only():
    plan = _skip_unless_live_consistency()
    from app.services.data_plane_consistency import verify_data_plane_consistency

    doc_id = plan["safe_ids"]["doc_id"]["value"]
    sys_id = plan["safe_ids"]["sys_id"]["value"] or doc_id

    result = await verify_data_plane_consistency(
        doc_id,
        sys_id=sys_id,
        allow_live_clients=True,
        expect_document=True,
        expect_chunks=None,
        expect_images=None,
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["doc_id"] == doc_id
    assert result["collections"]["chunk"] != result["collections"]["document"]
    assert result["collections"]["image"] != result["collections"]["document"]
