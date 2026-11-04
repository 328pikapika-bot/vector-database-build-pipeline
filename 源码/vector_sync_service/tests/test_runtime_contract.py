from __future__ import annotations

import pytest

from app.config import settings
from app.services.kafka_consumer import _build_consumer_config


def test_docx_runtime_performance_defaults_are_locked():
    assert settings.BATCH_MAX_SIZE == 50
    assert settings.BATCH_MAX_WAIT_MS == 5000
    assert settings.EMBEDDING_TIMEOUT == 30
    assert settings.EMBEDDING_MAX_CONCURRENCY == 10
    assert settings.EMBEDDING_MAX_RETRIES == 3
    assert settings.MILVUS_BATCH_SIZE == 100
    assert settings.DOCUMENT_INDEX_TYPE == "IVFPQ"
    assert settings.FULLTEXT_CHUNK_INDEX_TYPE == "IVFPQ"
    assert settings.IMAGE_DENSE_INDEX_TYPE == "IVFPQ"
    assert settings.DOCUMENT_IVFPQ_NBITS == 8
    assert settings.FULLTEXT_CHUNK_IVFPQ_NBITS == 8
    assert settings.IMAGE_DENSE_IVFPQ_NBITS == 8


def test_kafka_consumer_config_disables_auto_commit_without_creating_consumer():
    config = _build_consumer_config()

    assert config["enable.auto.commit"] is False
    assert config["auto.offset.reset"] == "earliest"
    assert config["group.id"] == settings.KAFKA_GROUP_ID
    assert config["bootstrap.servers"] == settings.KAFKA_BOOTSTRAP_SERVERS


@pytest.mark.asyncio
async def test_vector_collection_startup_init_is_default_off(monkeypatch):
    from app import main

    calls = []

    def forbidden_milvus_service():
        calls.append("milvus")
        raise AssertionError("startup must not initialize Milvus collections by default")

    monkeypatch.setattr(main.settings, "INIT_VECTOR_COLLECTIONS_ON_STARTUP", False, raising=False)
    monkeypatch.setattr(main, "get_milvus_service", forbidden_milvus_service)

    assert await main._initialize_vector_collections_on_startup() is False
    assert calls == []
