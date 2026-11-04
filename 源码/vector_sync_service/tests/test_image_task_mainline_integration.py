from __future__ import annotations

from typing import Any

import pytest

from app.models.sync_runtime import (
    ArtifactType,
    ErrorType,
    FailureRef,
    ScopeType,
    StageExecutionStatus,
    SyncOverallStatus,
    SyncStage,
)
from app.services import batch_processor_existing as pipeline_module
from tests.test_failure_persistence_pipeline import (
    FakeEmbedder,
    FakeManager,
    FakeMilvus,
    _make_docx_contract_payload,
    _make_record,
    _make_valid_payload,
)


@pytest.fixture()
def pipeline_dependencies(monkeypatch):
    manager = FakeManager()
    milvus = FakeMilvus()
    embedder = FakeEmbedder(
        [
            pipeline_module.EmbeddingItemResult(success=True, vector=[1.0, 0.0, 0.0]),
            pipeline_module.EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
            pipeline_module.EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
        ]
    )
    fulltext_repository = FakeTaskRepository()
    image_repository = FakeTaskRepository()
    monkeypatch.setattr(pipeline_module, "get_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(pipeline_module, "get_milvus_service", lambda: milvus)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(
        pipeline_module,
        "get_fulltext_task_repository",
        lambda: fulltext_repository,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_image_task_repository",
        lambda: image_repository,
    )
    monkeypatch.setattr(pipeline_module.settings, "VECTOR_DIM", 3)
    monkeypatch.setattr(pipeline_module.settings, "MILVUS_COLLECTION", "unit_test_collection")
    monkeypatch.setattr(pipeline_module.settings, "MONGO_SUCCESS_AUDIT_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_WORKER_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_FEATURE_PIPELINE_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_MILVUS_WRITE_ENABLED", False)
    return manager, milvus, fulltext_repository, image_repository


@pytest.mark.asyncio
async def test_default_image_task_upsert_disabled_preserves_mainline(
    pipeline_dependencies,
):
    manager, _, _, image_repository = pipeline_dependencies
    record = _make_record(_payload_with_image(), offset=201)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert image_repository.upsert_calls == []
    stage_statuses = manager.status_docs["doc-contract-1"].get("stage_statuses", {})
    assert SyncStage.IMAGE_DISCOVER.value not in stage_statuses
    assert SyncStage.MILVUS_WRITE_IMAGES.value not in stage_statuses
    assert SyncStage.DELETE_IMAGES.value not in stage_statuses


@pytest.mark.asyncio
async def test_worker_feature_and_write_flags_do_not_affect_mainline_when_task_upsert_disabled(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, _, image_repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_WORKER_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_FEATURE_PIPELINE_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_MILVUS_WRITE_ENABLED", True)
    record = _make_record(_payload_with_image(), offset=208)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert image_repository.upsert_calls == []
    stage_statuses = manager.status_docs["doc-contract-1"].get("stage_statuses", {})
    assert SyncStage.IMAGE_DISCOVER.value not in stage_statuses
    assert SyncStage.MILVUS_WRITE_IMAGES.value not in stage_statuses
    assert SyncStage.DELETE_IMAGES.value not in stage_statuses


@pytest.mark.asyncio
async def test_enabled_add_upserts_image_tasks_saves_manifest_and_marks_success(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, _, image_repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", True)
    record = _make_record(_payload_with_image(), offset=202)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert len(image_repository.upsert_calls) == 1
    tasks = image_repository.upsert_calls[0]
    assert len(tasks) == 1
    assert tasks[0].doc_id == "doc-contract-1"
    assert tasks[0].relative_url == "/images/fig1.png"
    assert tasks[0].mmd_relative_url == "/fulltext/paper.mmd"
    assert tasks[0].image_id == "image-1"

    image_artifacts = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.IMAGE_DISCOVER.value
        and artifact["artifact_type"] == ArtifactType.IMAGE_MANIFEST_JSON.value
    ]
    assert len(image_artifacts) == 1
    payload = image_artifacts[0]["payload"]
    assert payload["task_ids"] == [tasks[0].task_id]
    assert payload["image_ids"] == ["image-1"]
    assert payload["relative_urls"] == ["/images/fig1.png"]
    assert payload["mmd_relative_urls"] == ["/fulltext/paper.mmd"]
    assert payload["upsert_result"] == {"matched": 0, "upserted": 1}
    assert payload["tasks"][0]["title"] == "Figure 1"
    assert manager.status_docs["doc-contract-1"]["stage_statuses"][
        SyncStage.IMAGE_DISCOVER.value
    ]["status"] == StageExecutionStatus.SUCCESS.value
    assert manager.status_docs["doc-contract-1"]["core_status"] == "SUCCESS"
    assert manager.status_docs["doc-contract-1"]["image_status"] == "PENDING"
    assert manager.status_docs["doc-contract-1"]["image_required"] is True
    assert manager.status_docs["doc-contract-1"]["overall_status"] == (
        SyncOverallStatus.CORE_SUCCESS.value
    )


@pytest.mark.asyncio
async def test_enabled_update_without_image_marks_skipped_not_present(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, _, image_repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", True)
    payload = _make_valid_payload(doc_id="doc-no-image", event_type="UPDATE")
    record = _make_record(payload, offset=203)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert image_repository.upsert_calls == []
    stage_status = manager.status_docs["doc-no-image"]["stage_statuses"][
        SyncStage.IMAGE_DISCOVER.value
    ]
    assert stage_status["status"] == StageExecutionStatus.SKIPPED_NOT_PRESENT.value
    assert manager.status_docs["doc-no-image"]["image_status"] == "NOT_PRESENT"
    attempts = [
        attempt
        for attempt in manager.attempts
        if attempt["stage"] == SyncStage.IMAGE_DISCOVER
    ]
    assert attempts[-1]["stage_status"] == StageExecutionStatus.SKIPPED_NOT_PRESENT.value


@pytest.mark.asyncio
async def test_image_task_upsert_failure_is_persisted_and_commit_safe(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, _, image_repository = pipeline_dependencies
    image_repository.error = RuntimeError("image task queue unavailable")
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", True)
    record = _make_record(_payload_with_image(), offset=204)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    outcome = result.outcomes[0]
    assert outcome.failure_persisted is True
    assert len(outcome.pending_failure_refs) == 1
    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.IMAGE_DISCOVER.value
    assert failure["scope_type"] == ScopeType.DOCUMENT.value
    assert failure["error_type"] == ErrorType.RETRYABLE.value
    assert failure["error_code"] == "IMAGE_TASK_UPSERT_ERROR"
    assert manager.status_docs["doc-contract-1"]["stage_statuses"][
        SyncStage.IMAGE_DISCOVER.value
    ]["status"] == StageExecutionStatus.FAILURE.value


@pytest.mark.asyncio
async def test_image_task_upsert_failure_not_commit_safe_when_failure_persist_fails(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, _, image_repository = pipeline_dependencies
    image_repository.error = RuntimeError("image task queue unavailable")

    async def fail_persist_failure(**_: Any) -> FailureRef:
        raise RuntimeError("mongo unavailable")

    manager.persist_failure = fail_persist_failure
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", True)
    record = _make_record(_payload_with_image(), offset=205)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is False
    assert result.outcomes[0].final_status == SyncOverallStatus.SUCCESS.value
    assert result.outcomes[0].failure_persisted is False
    assert result.outcomes[0].message == "IMAGE_TASK_UPSERT_ERROR"


@pytest.mark.asyncio
async def test_delete_event_does_not_trigger_image_discover_or_delete_images(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, _, image_repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", True)
    record = _make_record(_payload_with_image(event_type="DELETE"), offset=206)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert image_repository.upsert_calls == []
    stage_statuses = manager.status_docs["doc-contract-1"].get("stage_statuses", {})
    assert SyncStage.IMAGE_DISCOVER.value not in stage_statuses
    assert SyncStage.DELETE_IMAGES.value not in stage_statuses


@pytest.mark.asyncio
async def test_fulltext_and_image_discover_both_run_when_both_enabled(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, fulltext_repository, image_repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", True)
    record = _make_record(_payload_with_image(), offset=207)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert len(fulltext_repository.upsert_calls) == 1
    assert len(image_repository.upsert_calls) == 1
    stage_statuses = manager.status_docs["doc-contract-1"]["stage_statuses"]
    assert stage_statuses[SyncStage.FULLTEXT_DISCOVER.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )
    assert stage_statuses[SyncStage.IMAGE_DISCOVER.value]["status"] == (
        StageExecutionStatus.SUCCESS.value
    )


def _payload_with_image(*, event_type: str = "ADD") -> dict[str, Any]:
    payload = _make_docx_contract_payload(event_type=event_type)
    payload["data"][0].setdefault("op", {})["hasPictureFile"] = {
        "field": "hasPictureFile",
        "value": [
            {
                "id": "image-1",
                "cls": "PictureFile",
                "dp": {
                    "name": {"field": "name", "value": ["fig1.png"]},
                    "title": {"field": "title", "value": ["Figure 1"]},
                    "url": {"field": "url", "value": ["/images/fig1.png"]},
                    "width": {"field": "width", "value": ["800"]},
                    "height": {"field": "height", "value": ["600"]},
                    "size": {"field": "size", "value": ["12345"]},
                    "sequence": {"field": "sequence", "value": ["1"]},
                    "fileType": {"field": "fileType", "value": ["png"]},
                },
            }
        ],
    }
    return payload


class FakeTaskRepository:
    def __init__(self) -> None:
        self.upsert_calls: list[list[Any]] = []
        self.error: Exception | None = None

    async def upsert_tasks(self, tasks):
        if self.error is not None:
            raise self.error
        self.upsert_calls.append(list(tasks))
        return {"matched": 0, "upserted": len(tasks)}
