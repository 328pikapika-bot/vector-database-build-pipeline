from __future__ import annotations

from typing import Any

import pytest

from app.models.sync_runtime import (
    ArtifactType,
    EmbeddingItemResult,
    FailureRef,
    ReplayMode,
    ScopeType,
    StageExecutionStatus,
    SyncOverallStatus,
    SyncStage,
)
from app.services import batch_processor_existing as pipeline_module
from app.services.milvus_chunk_service import DEFAULT_CHUNK_COLLECTION
from app.services.milvus_image_service import DEFAULT_IMAGE_COLLECTION
from tests.test_failure_persistence_pipeline import (
    FakeEmbedder,
    FakeManager,
    FakeMilvus,
    _make_record,
    _make_valid_payload,
)


class FakeMilvusChunkService:
    def __init__(self, *, delete_error: Exception | None = None, close_error: Exception | None = None):
        self.delete_error = delete_error
        self.close_error = close_error
        self.delete_calls: list[dict[str, Any]] = []
        self.closed = False

    def delete_chunks_by_doc_id(self, collection_name: str, doc_id: str):
        call = {"collection_name": collection_name, "doc_id": doc_id}
        self.delete_calls.append(call)
        if self.delete_error is not None:
            raise self.delete_error
        return {"delete_count": 7}

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeMilvusChunkServiceFactory:
    def __init__(self):
        self.instances: list[FakeMilvusChunkService] = []
        self.delete_error: Exception | None = None
        self.close_error: Exception | None = None

    def __call__(self):
        instance = FakeMilvusChunkService(
            delete_error=self.delete_error,
            close_error=self.close_error,
        )
        self.instances.append(instance)
        return instance


class FakeImageDeletePipeline:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None

    def __call__(self, doc_id: str):
        self.calls.append({"doc_id": doc_id})
        return self.result or {
            "success": True,
            "doc_id": doc_id,
            "collection_name": DEFAULT_IMAGE_COLLECTION,
            "delete_result": {"delete_count": 5},
            "error_code": None,
            "error_message": None,
            "retryable": False,
        }


@pytest.fixture()
def pipeline_dependencies(monkeypatch):
    manager = FakeManager()
    milvus = FakeMilvus()
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0, 0.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
        ]
    )
    chunk_factory = FakeMilvusChunkServiceFactory()
    monkeypatch.setattr(pipeline_module, "get_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(pipeline_module, "get_milvus_service", lambda: milvus)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(pipeline_module, "MilvusChunkService", chunk_factory)
    monkeypatch.setattr(pipeline_module.settings, "VECTOR_DIM", 3)
    monkeypatch.setattr(pipeline_module.settings, "MILVUS_COLLECTION", "unit_test_collection")
    monkeypatch.setattr(pipeline_module.settings, "MONGO_SUCCESS_AUDIT_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", False)
    return manager, milvus, chunk_factory


@pytest.fixture()
def image_delete_pipeline(monkeypatch):
    fake = FakeImageDeletePipeline()
    monkeypatch.setattr(pipeline_module, "delete_images_for_document", fake)
    return fake


@pytest.mark.asyncio
async def test_default_delete_chunks_disabled_preserves_document_delete_behavior(pipeline_dependencies):
    manager, milvus, chunk_factory = pipeline_dependencies
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=401)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert [call["op"] for call in milvus.call_order] == ["delete_by_sys_id"]
    assert chunk_factory.instances == []

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.DELETED.value
    assert status["current_stage"] == SyncStage.DELETE_DOCUMENT.value
    assert status["last_success_stage"] == SyncStage.DELETE_DOCUMENT.value
    assert SyncStage.DELETE_CHUNKS.value not in status.get("stage_statuses", {})
    assert SyncStage.DELETE_IMAGES.value not in status.get("stage_statuses", {})
    assert not [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.DELETE_CHUNKS.value
    ]
    assert not [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.DELETE_IMAGES.value
    ]


@pytest.mark.asyncio
async def test_enabled_delete_chunks_runs_after_document_delete_and_records_success(
    pipeline_dependencies,
    monkeypatch,
):
    manager, milvus, chunk_factory = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=402)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert [call["op"] for call in milvus.call_order] == ["delete_by_sys_id"]
    assert len(chunk_factory.instances) == 1
    chunk_service = chunk_factory.instances[0]
    assert chunk_service.delete_calls == [
        {"collection_name": DEFAULT_CHUNK_COLLECTION, "doc_id": "doc-1"}
    ]
    assert chunk_service.closed is True

    artifact_stages = [artifact["stage"] for artifact in manager.artifacts.values()]
    assert artifact_stages.index(SyncStage.DELETE_DOCUMENT.value) < artifact_stages.index(
        SyncStage.DELETE_CHUNKS.value
    )
    delete_chunks_artifacts = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.DELETE_CHUNKS.value
        and artifact["artifact_type"] == ArtifactType.MILVUS_WRITE_SUMMARY.value
    ]
    assert len(delete_chunks_artifacts) == 1
    assert delete_chunks_artifacts[0]["scope_type"] == ScopeType.CHUNK.value
    assert delete_chunks_artifacts[0]["scope_id"] == "doc-1"
    assert delete_chunks_artifacts[0]["payload"] == {
        "doc_id": "doc-1",
        "collection_name": DEFAULT_CHUNK_COLLECTION,
        "delete_result": {"delete_count": 7},
    }

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.DELETED.value
    assert status["current_stage"] == SyncStage.DELETE_CHUNKS.value
    assert status["last_success_stage"] == SyncStage.DELETE_CHUNKS.value
    assert status["stage_statuses"][SyncStage.DELETE_CHUNKS.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )


@pytest.mark.asyncio
async def test_enabled_delete_images_runs_after_document_delete_and_records_success(
    pipeline_dependencies,
    image_delete_pipeline,
    monkeypatch,
):
    manager, milvus, chunk_factory = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=406)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert [call["op"] for call in milvus.call_order] == ["delete_by_sys_id"]
    assert chunk_factory.instances == []
    assert image_delete_pipeline.calls == [{"doc_id": "doc-1"}]

    artifact_stages = [artifact["stage"] for artifact in manager.artifacts.values()]
    assert artifact_stages.index(SyncStage.DELETE_DOCUMENT.value) < artifact_stages.index(
        SyncStage.DELETE_IMAGES.value
    )
    delete_images_artifacts = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.DELETE_IMAGES.value
        and artifact["artifact_type"] == ArtifactType.MILVUS_WRITE_SUMMARY.value
    ]
    assert len(delete_images_artifacts) == 1
    assert delete_images_artifacts[0]["scope_type"] == ScopeType.IMAGE.value
    assert delete_images_artifacts[0]["scope_id"] == "doc-1"
    assert delete_images_artifacts[0]["payload"] == {
        "success": True,
        "doc_id": "doc-1",
        "collection_name": DEFAULT_IMAGE_COLLECTION,
        "delete_result": {"delete_count": 5},
        "error_code": None,
        "error_message": None,
        "retryable": False,
    }

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.DELETED.value
    assert status["current_stage"] == SyncStage.DELETE_IMAGES.value
    assert status["last_success_stage"] == SyncStage.DELETE_IMAGES.value
    assert status["stage_statuses"][SyncStage.DELETE_IMAGES.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )
    assert status["progress"]["image_vectors_deleted"] == 1


@pytest.mark.asyncio
async def test_delete_images_failure_is_persisted_and_not_clean_deleted(
    pipeline_dependencies,
    image_delete_pipeline,
    monkeypatch,
):
    manager, _, _ = pipeline_dependencies
    image_delete_pipeline.result = {
        "success": False,
        "doc_id": "doc-1",
        "collection_name": DEFAULT_IMAGE_COLLECTION,
        "delete_result": None,
        "error_code": "IMAGE_DELETE_WRITER_ERROR",
        "error_message": "image delete failed",
        "retryable": True,
    }
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=407)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.fail == 1
    outcome = result.outcomes[0]
    assert outcome.final_status == SyncOverallStatus.PARTIAL_FAILURE.value
    assert outcome.failure_persisted is True
    assert len(outcome.pending_failure_refs) == 1
    assert image_delete_pipeline.calls == [{"doc_id": "doc-1"}]

    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.DELETE_IMAGES.value
    assert failure["scope_type"] == ScopeType.IMAGE.value
    assert failure["scope_id"] == "doc-1"
    assert failure["error_type"] == "RETRYABLE"
    assert failure["error_code"] == "IMAGE_DELETE_WRITER_ERROR"
    assert failure["error_message"] == "image delete failed"
    assert failure["replay_mode"] == ReplayMode.FROM_DELETE_CONTEXT.value

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.PARTIAL_FAILURE.value
    assert status["current_stage"] == SyncStage.DELETE_IMAGES.value
    assert status["last_success_stage"] == SyncStage.DELETE_DOCUMENT.value
    assert status["stage_statuses"][SyncStage.DELETE_IMAGES.value]["status"] == (
        StageExecutionStatus.FAILURE.value
    )
    assert status["progress"]["image_vectors_deleted"] == 0


@pytest.mark.asyncio
async def test_delete_images_failure_not_commit_safe_when_failure_persist_fails(
    pipeline_dependencies,
    image_delete_pipeline,
    monkeypatch,
):
    manager, _, _ = pipeline_dependencies
    image_delete_pipeline.result = {
        "success": False,
        "doc_id": "doc-1",
        "collection_name": DEFAULT_IMAGE_COLLECTION,
        "delete_result": None,
        "error_code": "IMAGE_DELETE_VALIDATION_ERROR",
        "error_message": "validation failed",
        "retryable": False,
    }

    async def fail_persist_failure(**_: Any) -> FailureRef:
        raise RuntimeError("mongo unavailable")

    manager.persist_failure = fail_persist_failure
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=408)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is False
    assert result.fail == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.PARTIAL_FAILURE.value
    assert result.outcomes[0].failure_persisted is False
    assert result.outcomes[0].message == "IMAGE_DELETE_VALIDATION_ERROR"
    assert manager.failures == {}
    assert image_delete_pipeline.calls == [{"doc_id": "doc-1"}]


@pytest.mark.asyncio
async def test_delete_chunks_and_images_run_in_order_when_both_enabled(
    pipeline_dependencies,
    image_delete_pipeline,
    monkeypatch,
):
    manager, _, chunk_factory = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=409)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert len(chunk_factory.instances) == 1
    assert chunk_factory.instances[0].delete_calls == [
        {"collection_name": DEFAULT_CHUNK_COLLECTION, "doc_id": "doc-1"}
    ]
    assert image_delete_pipeline.calls == [{"doc_id": "doc-1"}]

    artifact_stages = [artifact["stage"] for artifact in manager.artifacts.values()]
    assert artifact_stages.index(SyncStage.DELETE_DOCUMENT.value) < artifact_stages.index(
        SyncStage.DELETE_CHUNKS.value
    )
    assert artifact_stages.index(SyncStage.DELETE_CHUNKS.value) < artifact_stages.index(
        SyncStage.DELETE_IMAGES.value
    )

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.DELETED.value
    assert status["current_stage"] == SyncStage.DELETE_IMAGES.value
    assert status["last_success_stage"] == SyncStage.DELETE_IMAGES.value
    assert status["stage_statuses"][SyncStage.DELETE_CHUNKS.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )
    assert status["stage_statuses"][SyncStage.DELETE_IMAGES.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )
    assert status["progress"]["chunk_vectors_deleted"] == 1
    assert status["progress"]["image_vectors_deleted"] == 1


@pytest.mark.asyncio
async def test_delete_images_failure_after_delete_chunks_success_preserves_chunk_stage(
    pipeline_dependencies,
    image_delete_pipeline,
    monkeypatch,
):
    manager, _, chunk_factory = pipeline_dependencies
    image_delete_pipeline.result = {
        "success": False,
        "doc_id": "doc-1",
        "collection_name": DEFAULT_IMAGE_COLLECTION,
        "delete_result": None,
        "error_code": "IMAGE_DELETE_WRITER_ERROR",
        "error_message": "image delete failed",
        "retryable": True,
    }
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=410)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.fail == 1
    assert len(chunk_factory.instances) == 1
    assert chunk_factory.instances[0].delete_calls == [
        {"collection_name": DEFAULT_CHUNK_COLLECTION, "doc_id": "doc-1"}
    ]
    assert image_delete_pipeline.calls == [{"doc_id": "doc-1"}]

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.PARTIAL_FAILURE.value
    assert status["current_stage"] == SyncStage.DELETE_IMAGES.value
    assert status["last_success_stage"] == SyncStage.DELETE_CHUNKS.value
    assert status["stage_statuses"][SyncStage.DELETE_CHUNKS.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )
    assert status["stage_statuses"][SyncStage.DELETE_IMAGES.value]["status"] == (
        StageExecutionStatus.FAILURE.value
    )
    assert status["progress"]["chunk_vectors_deleted"] == 1
    assert status["progress"]["image_vectors_deleted"] == 0


@pytest.mark.asyncio
async def test_delete_chunks_failure_is_persisted_and_commit_safe(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, chunk_factory = pipeline_dependencies
    chunk_factory.delete_error = RuntimeError("chunk delete failed")
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=403)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.fail == 1
    outcome = result.outcomes[0]
    assert outcome.final_status == SyncOverallStatus.PARTIAL_FAILURE.value
    assert outcome.failure_persisted is True
    assert len(outcome.pending_failure_refs) == 1
    assert chunk_factory.instances[0].closed is True

    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.DELETE_CHUNKS.value
    assert failure["scope_type"] == ScopeType.CHUNK.value
    assert failure["scope_id"] == "doc-1"
    assert failure["error_code"] == "FULLTEXT_DELETE_CHUNKS_ERROR"
    assert failure["replay_mode"] == ReplayMode.FROM_DELETE_CONTEXT.value

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.PARTIAL_FAILURE.value
    assert status["current_stage"] == SyncStage.DELETE_CHUNKS.value
    assert status["last_success_stage"] == SyncStage.DELETE_DOCUMENT.value
    assert status["stage_statuses"][SyncStage.DELETE_CHUNKS.value]["status"] == (
        StageExecutionStatus.FAILURE.value
    )


@pytest.mark.asyncio
async def test_delete_chunks_failure_not_commit_safe_when_failure_persist_fails(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, chunk_factory = pipeline_dependencies
    chunk_factory.delete_error = RuntimeError("chunk delete failed")

    async def fail_persist_failure(**_: Any) -> FailureRef:
        raise RuntimeError("mongo unavailable")

    manager.persist_failure = fail_persist_failure
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=404)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is False
    assert result.fail == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.PARTIAL_FAILURE.value
    assert result.outcomes[0].failure_persisted is False
    assert result.outcomes[0].message == "FULLTEXT_DELETE_CHUNKS_ERROR"
    assert manager.failures == {}
    assert chunk_factory.instances[0].closed is True


@pytest.mark.parametrize("event_type", ["ADD", "UPDATE"])
@pytest.mark.asyncio
async def test_add_and_update_do_not_trigger_delete_chunks_when_enabled(
    pipeline_dependencies,
    image_delete_pipeline,
    monkeypatch,
    event_type,
):
    _, milvus, chunk_factory = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type=event_type), offset=405)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert [call["op"] for call in milvus.call_order] == ["process_and_insert"]
    assert milvus.delete_calls == []
    assert chunk_factory.instances == []
    assert image_delete_pipeline.calls == []
