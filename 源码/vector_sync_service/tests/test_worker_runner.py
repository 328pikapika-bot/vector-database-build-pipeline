from __future__ import annotations

import pytest

from app.cli.worker_runner import initialize_independent_worker_target, run_independent_worker


def test_image_worker_initializes_collection_before_consumer_startup(monkeypatch):
    calls: list[object] = []

    class FakeImageService:
        def init_image_collection(self, collection_name: str):
            calls.append(("init", collection_name))
            return {
                "resolved_name": collection_name,
                "created": False,
                "reused": True,
                "target_signature": "image-target",
            }

        def close(self):
            calls.append("close")

    monkeypatch.setattr("app.cli.worker_runner.MilvusImageService", FakeImageService)

    initialize_independent_worker_target("image")

    assert calls[0][0] == "init"
    assert calls[-1] == "close"


def test_non_image_workers_do_not_initialize_collections():
    assert initialize_independent_worker_target("metadata") is None
    assert initialize_independent_worker_target("fulltext") is None


@pytest.mark.asyncio
async def test_image_collection_is_initialized_before_kafka_consumer(monkeypatch):
    order: list[str] = []

    class FakeManager:
        async def connect(self):
            order.append("mongo_connect")

    class FakeProcessor:
        def __init__(self, *_args, **_kwargs):
            order.append("processor")

        async def process_batch(self, _batch):
            return None

        async def close(self):
            order.append("processor_close")

    class FakeConsumer:
        def close(self):
            order.append("consumer_close")

    async def fake_run_consumer(_consumer, _processor):
        order.append("consume")

    async def async_marker(name: str):
        order.append(name)

    monkeypatch.setattr("app.cli.worker_runner.get_mongo_state_manager", lambda: FakeManager())
    monkeypatch.setattr(
        "app.cli.worker_runner.initialize_independent_worker_target",
        lambda _role: order.append("image_target_ready"),
    )
    monkeypatch.setattr("app.cli.worker_runner.IndependentKafkaWorker", FakeProcessor)
    monkeypatch.setattr(
        "app.cli.worker_runner.create_consumer",
        lambda **_kwargs: order.append("consumer") or FakeConsumer(),
    )
    monkeypatch.setattr("app.cli.worker_runner.kafka_group_id_for_role", lambda _role: "image-group")
    monkeypatch.setattr("app.cli.worker_runner.run_consumer", fake_run_consumer)
    monkeypatch.setattr(
        "app.cli.worker_runner.close_embedding_client",
        lambda: async_marker("embedding_close"),
    )
    monkeypatch.setattr("app.cli.worker_runner.close_milvus_service", lambda: order.append("milvus_close"))
    monkeypatch.setattr(
        "app.cli.worker_runner.close_mongo_state_manager",
        lambda: async_marker("mongo_close"),
    )

    await run_independent_worker("image")

    assert order.index("image_target_ready") < order.index("consumer") < order.index("consume")
