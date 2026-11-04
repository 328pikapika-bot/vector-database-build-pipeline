from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.models.sync_runtime import BatchProcessingResult, ConsumedRecord
from app.services.independent_kafka_worker import (
    IndependentKafkaWorker,
    kafka_group_id_for_role,
)


class _FailureManager:
    def __init__(self) -> None:
        self.persisted: list[dict] = []
        self.done: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    async def persist_failure(self, **kwargs) -> None:
        self.persisted.append(kwargs)

    async def mark_failure_done(self, failure_key: str, note: str) -> None:
        self.done.append((failure_key, note))

    async def mark_failure_skipped_stale(self, failure_key: str, reason: str) -> None:
        self.skipped.append((failure_key, reason))


class _VersionManager(_FailureManager):
    def __init__(self) -> None:
        super().__init__()
        self.versions: dict[tuple[str, str, str, str], dict] = {}

    async def get_pipeline_source_version(self, **kwargs):
        key = (
            kwargs["pipeline"],
            kwargs["logical_collection"],
            kwargs["target_signature"],
            kwargs["doc_id"],
        )
        return self.versions.get(key)

    async def upsert_pipeline_source_version(self, **kwargs) -> None:
        key = (
            kwargs["pipeline"],
            kwargs["logical_collection"],
            kwargs["target_signature"],
            kwargs["doc_id"],
        )
        self.versions[key] = {
            "source_version": kwargs["source_version"],
            "source_hash": kwargs["source_hash"],
        }


def _invalid_record() -> ConsumedRecord:
    return ConsumedRecord(
        topic="topic",
        partition=1,
        offset=2,
        raw_bytes=b"not-json",
        raw_text="not-json",
        event_fingerprint="topic|1|2|hash",
        decode_error="invalid json",
    )


def _delete_record() -> ConsumedRecord:
    return ConsumedRecord(
        topic="topic",
        partition=1,
        offset=3,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint="topic|1|3|delete-hash",
        parsed_payload={"operation": "DELETE", "data": [{"id": "doc-1"}]},
    )


@pytest.mark.asyncio
async def test_direct_fulltext_invalid_record_is_persisted_before_commit(monkeypatch):
    manager = _FailureManager()
    worker = IndependentKafkaWorker("fulltext", state_manager=manager)
    monkeypatch.setattr(
        worker,
        "_target_binding",
        lambda: {
            "pipeline": "fulltext",
            "source_version": "",
            "item_id": "",
            "logical_collection": "chunks_v2",
            "resolved_collection": "chunks_v2",
            "target_signature": "chunk-signature",
        },
    )

    result = await worker.process_batch([_invalid_record()])

    assert result.fail == 1
    assert result.commit_safe is True
    assert len(manager.persisted) == 1
    failure = manager.persisted[0]
    assert failure["pipeline"] == "fulltext"
    assert failure["resolved_collection"] == "chunks_v2"
    assert failure["target_signature"] == "chunk-signature"
    assert failure["replay_payload"] == {"raw_text": "not-json"}


@pytest.mark.asyncio
async def test_direct_replay_without_payload_is_held_as_history():
    manager = _FailureManager()
    worker = IndependentKafkaWorker("image", state_manager=manager)

    outcome = await worker.replay_failure({"failure_key": "failure-1", "doc_id": "doc-1"})

    assert outcome.commit_safe is True
    assert manager.skipped == [("failure-1", "direct worker failure has no replay payload")]


@pytest.mark.asyncio
async def test_fulltext_delete_event_is_acknowledged_without_calling_milvus(monkeypatch):
    manager = _FailureManager()
    worker = IndependentKafkaWorker("fulltext", state_manager=manager)
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.settings.LIVE_E2E_NO_DELETE",
        True,
    )
    worker._chunk_service = SimpleNamespace(
        delete_chunks_by_doc_id=lambda *_args, **_kwargs: pytest.fail("Milvus delete must not be called")
    )

    result = await worker.process_batch([_delete_record()])

    assert result.success == 0
    assert result.skip == 1
    assert result.commit_safe is True
    assert result.outcomes[0].message == "fulltext_delete_skipped_no_delete"


@pytest.mark.asyncio
async def test_fulltext_delete_replay_is_marked_done_without_calling_milvus(monkeypatch):
    manager = _FailureManager()
    worker = IndependentKafkaWorker("fulltext", state_manager=manager)
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.settings.LIVE_E2E_NO_DELETE",
        True,
    )
    worker._chunk_service = SimpleNamespace(
        delete_chunks_by_doc_id=lambda *_args, **_kwargs: pytest.fail("Milvus delete must not be called")
    )

    result = await worker.replay_failure(
        {
            "failure_key": "failure-delete-1",
            "doc_id": "doc-1",
            "replay_payload": {"kind": "delete_document", "doc_ids": ["doc-1"]},
        }
    )

    assert result.commit_safe is True
    assert result.final_status == "SKIPPED_STALE"
    assert result.message == "fulltext_delete_replay_skipped_no_delete"
    assert manager.skipped == [
        (
            "failure-delete-1",
            "fulltext delete replay skipped because LIVE_E2E_NO_DELETE is enabled",
        )
    ]


def test_each_role_resolves_to_a_different_kafka_group(monkeypatch):
    monkeypatch.setattr("app.services.independent_kafka_worker.settings.KAFKA_METADATA_GROUP_ID", "meta")
    monkeypatch.setattr("app.services.independent_kafka_worker.settings.KAFKA_FULLTEXT_GROUP_ID", "full")
    monkeypatch.setattr("app.services.independent_kafka_worker.settings.KAFKA_IMAGE_GROUP_ID", "image")

    groups = {
        kafka_group_id_for_role("metadata"),
        kafka_group_id_for_role("fulltext"),
        kafka_group_id_for_role("image"),
    }
    assert len(groups) == 3
    assert kafka_group_id_for_role("metadata").startswith("meta--target-")
    assert kafka_group_id_for_role("fulltext").startswith("full--target-")
    assert kafka_group_id_for_role("image").startswith("image--target-")


def test_target_group_changes_for_build_index_but_not_search_nprobe(monkeypatch):
    monkeypatch.setattr("app.services.independent_kafka_worker.settings.DOCUMENT_INDEX_TYPE", "IVFPQ")
    monkeypatch.setattr("app.services.independent_kafka_worker.settings.KAFKA_METADATA_GROUP_ID", "meta")
    initial = kafka_group_id_for_role("metadata")

    monkeypatch.setattr("app.services.independent_kafka_worker.settings.DOCUMENT_IVFPQ_NPROBE", 128)
    assert kafka_group_id_for_role("metadata") == initial

    monkeypatch.setattr("app.services.independent_kafka_worker.settings.DOCUMENT_IVFPQ_NLIST", 8192)
    assert kafka_group_id_for_role("metadata") != initial


class _DirectImageWorker:
    def __init__(self, *, fail_image_id: str = "", skip_image_id: str = "") -> None:
        self.fail_image_id = fail_image_id
        self.skip_image_id = skip_image_id

    async def process_task(self, task: dict):
        image_id = task["image_id"]
        if image_id == self.fail_image_id:
            return SimpleNamespace(
                success=False,
                image_record=None,
                error_message="image download failed",
                retryable=True,
            )
        if image_id == self.skip_image_id:
            return SimpleNamespace(
                success=True,
                image_record=None,
                error_message="",
                retryable=False,
            )
        return SimpleNamespace(
            success=True,
            image_record={"id": image_id, "doc_id": task["doc_id"]},
            error_message="",
            retryable=False,
        )


class _HangingDirectImageWorker:
    async def process_task(self, _task: dict):
        await asyncio.sleep(1)


class _DirectImageService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def replace_images_for_documents(self, collection_name: str, documents: dict, *, complete_snapshot: bool):
        self.calls.append(
            {
                "collection_name": collection_name,
                "documents": documents,
                "complete_snapshot": complete_snapshot,
            }
        )
        return {"image_count": sum(len(items) for items in documents.values())}

    def insert_or_skip_images_by_img_id_batch(self, collection_name: str, images: list[dict]):
        self.calls.append(
            {
                "collection_name": collection_name,
                "images": images,
                "mode": "insert_or_skip_if_exists",
            }
        )
        return {"new_count": len(images), "image_count": len(images)}


def _image_record() -> ConsumedRecord:
    return ConsumedRecord(
        topic="topic",
        partition=0,
        offset=1,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint="topic|0|1|hash",
        parsed_payload={
            "data": [
                {"sysId": "doc-1", "sysDataUpdateTime": "2026-08-05T00:00:00Z"},
                {"sysId": "doc-2", "sysDataUpdateTime": "2026-08-05T00:00:00Z"},
            ]
        },
    )


@pytest.mark.asyncio
async def test_direct_image_task_timeout_returns_replayable_failure(monkeypatch):
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.settings.IMAGE_WORKER_TASK_TIMEOUT_SEC",
        0.01,
    )
    worker = IndependentKafkaWorker("image", state_manager=_FailureManager())

    result = await worker._process_image_task_with_timeout(
        _HangingDirectImageWorker(),
        {"task_id": "task-timeout", "doc_id": "doc-timeout"},
    )

    assert result.success is False
    assert result.error_code == "IMAGE_WORKER_TASK_TIMEOUT"
    assert result.retryable is True


@pytest.mark.asyncio
async def test_direct_image_batch_writes_successes_once_without_mongo(monkeypatch):
    manager = _FailureManager()
    image_worker = _DirectImageWorker()
    worker = IndependentKafkaWorker("image", state_manager=manager, image_worker=image_worker)
    service = _DirectImageService()
    worker._image_service = service
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.build_image_task_documents",
        lambda _payload: [
            {"task_id": "task-1", "image_id": "image-1", "doc_id": "doc-1", "record_update_time": "v1"},
            {"task_id": "task-2", "image_id": "image-2", "doc_id": "doc-2", "record_update_time": "v1"},
        ],
    )

    result = await worker.process_batch([_image_record()])

    assert result.success == 1
    assert manager.persisted == []
    assert len(service.calls) == 1
    assert service.calls[0]["complete_snapshot"] is True
    assert set(service.calls[0]["documents"]) == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_direct_image_batch_uses_insert_or_skip_for_existing_collection(monkeypatch):
    manager = _FailureManager()
    image_worker = _DirectImageWorker()
    worker = IndependentKafkaWorker("image", state_manager=manager, image_worker=image_worker)
    service = _DirectImageService()
    worker._image_service = service
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.settings.IMAGE_MILVUS_WRITE_MODE",
        "insert_or_skip_if_exists",
    )
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.settings.IMAGE_DIRECT_STALE_CLEANUP_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.build_image_task_documents",
        lambda _payload: [
            {"task_id": "task-1", "image_id": "image-1", "doc_id": "doc-1", "record_update_time": "v1"},
        ],
    )

    result = await worker.process_batch([_image_record()])

    assert result.success == 1
    assert service.calls[0]["mode"] == "insert_or_skip_if_exists"


@pytest.mark.asyncio
async def test_direct_image_dedup_skips_only_matching_source_version_and_hash(monkeypatch):
    manager = _VersionManager()
    image_worker = _DirectImageWorker()
    worker = IndependentKafkaWorker("image", state_manager=manager, image_worker=image_worker)
    service = _DirectImageService()
    worker._image_service = service
    monkeypatch.setattr(
        worker,
        "_target_binding",
        lambda: {
            "pipeline": "image",
            "logical_collection": "images_v2",
            "resolved_collection": "images_v2",
            "target_signature": "image-signature",
        },
    )
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.build_image_task_documents",
        lambda _payload: [
            {"task_id": "task-1", "image_id": "image-1", "doc_id": "doc-1", "record_update_time": "v1"},
        ],
    )
    record = _image_record()
    record.parsed_payload = {"data": [record.parsed_payload["data"][0]]}

    first = await worker.process_batch([record])
    second = await worker.process_batch([record])

    assert first.success == 1
    assert second.skip == 1
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_direct_image_failure_is_persisted_as_replayable_document(monkeypatch):
    manager = _FailureManager()
    image_worker = _DirectImageWorker(fail_image_id="image-2")
    worker = IndependentKafkaWorker("image", state_manager=manager, image_worker=image_worker)
    monkeypatch.setattr(
        worker,
        "_target_binding",
        lambda: {
            "pipeline": "image",
            "source_version": "",
            "item_id": "",
            "logical_collection": "images_v2",
            "resolved_collection": "images_v2",
            "target_signature": "image-signature",
        },
    )
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.build_image_task_documents",
        lambda _payload: [
            {"task_id": "task-2", "image_id": "image-2", "doc_id": "doc-2", "record_update_time": "v1"},
        ],
    )

    result = await worker.process_batch([_image_record()])

    assert result.fail == 1
    assert result.commit_safe is True
    assert len(manager.persisted) == 1
    failure = manager.persisted[0]
    assert failure["doc_id"] == "doc-2"
    assert failure["replay_payload"]["kind"] == "image_snapshot"
    assert failure["replay_payload"]["tasks"][0]["image_id"] == "image-2"


@pytest.mark.asyncio
async def test_direct_image_quality_skip_is_not_treated_as_failure(monkeypatch):
    manager = _FailureManager()
    image_worker = _DirectImageWorker(skip_image_id="image-quarantined")
    worker = IndependentKafkaWorker("image", state_manager=manager, image_worker=image_worker)
    service = _DirectImageService()
    worker._image_service = service
    monkeypatch.setattr(
        "app.services.independent_kafka_worker.build_image_task_documents",
        lambda _payload: [
            {
                "task_id": "task-quarantined",
                "image_id": "image-quarantined",
                "doc_id": "doc-1",
                "record_update_time": "v1",
            }
        ],
    )

    result = await worker.process_batch([_image_record()])

    assert result.success == 1
    assert manager.persisted == []
    assert service.calls == []


@pytest.mark.asyncio
async def test_parallel_image_batches_serialize_same_document(monkeypatch):
    worker = IndependentKafkaWorker("image", state_manager=_FailureManager())
    active = 0
    max_active = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def process_image_batch(batch):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        await release.wait()
        active -= 1
        return BatchProcessingResult(outcomes=[])

    monkeypatch.setattr(worker, "_process_image_batch", process_image_batch)
    record = _image_record()
    first = asyncio.create_task(worker.process_batch([record]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(worker.process_batch([record]))
    await asyncio.sleep(0.02)
    assert max_active == 1
    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_parallel_image_batches_overlap_different_documents(monkeypatch):
    worker = IndependentKafkaWorker("image", state_manager=_FailureManager())
    active = 0
    max_active = 0
    release = asyncio.Event()

    async def process_image_batch(batch):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await release.wait()
        active -= 1
        return BatchProcessingResult(outcomes=[])

    monkeypatch.setattr(worker, "_process_image_batch", process_image_batch)
    first_record = _image_record()
    second_record = _image_record()
    second_record.parsed_payload = {"data": [{"sysId": "doc-3"}]}
    first = asyncio.create_task(worker.process_batch([first_record]))
    second = asyncio.create_task(worker.process_batch([second_record]))
    await asyncio.sleep(0.02)
    assert max_active == 2
    release.set()
    await asyncio.gather(first, second)
