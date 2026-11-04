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
    repository = FakeFullTextRepository()
    monkeypatch.setattr(pipeline_module, "get_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(pipeline_module, "get_milvus_service", lambda: milvus)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(
        pipeline_module,
        "get_fulltext_task_repository",
        lambda: repository,
    )
    monkeypatch.setattr(pipeline_module.settings, "VECTOR_DIM", 3)
    monkeypatch.setattr(pipeline_module.settings, "MILVUS_COLLECTION", "unit_test_collection")
    monkeypatch.setattr(pipeline_module.settings, "MONGO_SUCCESS_AUDIT_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", False)
    return manager, milvus, repository


@pytest.mark.asyncio
async def test_default_fulltext_task_upsert_disabled_preserves_mainline(
    pipeline_dependencies,
):
    manager, _, repository = pipeline_dependencies
    record = _make_record(_make_docx_contract_payload(), offset=101)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert repository.upsert_calls == []
    stage_statuses = manager.status_docs["doc-contract-1"].get("stage_statuses", {})
    assert SyncStage.FULLTEXT_DISCOVER.value not in stage_statuses


@pytest.mark.asyncio
async def test_enabled_add_upserts_tasks_saves_meta_and_marks_success(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", True)
    record = _make_record(_make_docx_contract_payload(), offset=102)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert len(repository.upsert_calls) == 1
    tasks = repository.upsert_calls[0]
    assert len(tasks) == 1
    assert tasks[0].doc_id == "doc-contract-1"
    assert tasks[0].relative_url == "/fulltext/paper.mmd"

    fulltext_artifacts = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.FULLTEXT_DISCOVER.value
        and artifact["artifact_type"] == ArtifactType.FULLTEXT_FILE_META.value
    ]
    assert len(fulltext_artifacts) == 1
    payload = fulltext_artifacts[0]["payload"]
    assert payload["task_ids"] == [tasks[0].task_id]
    assert payload["relative_urls"] == ["/fulltext/paper.mmd"]
    assert payload["upsert_result"] == {"matched": 0, "upserted": 1}
    assert manager.status_docs["doc-contract-1"]["stage_statuses"][
        SyncStage.FULLTEXT_DISCOVER.value
    ]["status"] == StageExecutionStatus.SUCCESS.value
    assert manager.status_docs["doc-contract-1"]["fulltext_status"] == "PENDING"
    assert manager.status_docs["doc-contract-1"]["core_status"] == "PROCESSING"
    assert manager.status_docs["doc-contract-1"]["fulltext_required"] is True


@pytest.mark.asyncio
async def test_enabled_update_without_fulltext_marks_skipped_not_present(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", True)
    payload = _make_valid_payload(doc_id="doc-no-fulltext", event_type="UPDATE")
    record = _make_record(payload, offset=103)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert repository.upsert_calls == []
    stage_status = manager.status_docs["doc-no-fulltext"]["stage_statuses"][
        SyncStage.FULLTEXT_DISCOVER.value
    ]
    assert stage_status["status"] == StageExecutionStatus.SKIPPED_NOT_PRESENT.value
    assert manager.status_docs["doc-no-fulltext"]["fulltext_status"] == "NOT_PRESENT"
    assert manager.status_docs["doc-no-fulltext"]["core_status"] == "SUCCESS"
    attempts = [
        attempt
        for attempt in manager.attempts
        if attempt["stage"] == SyncStage.FULLTEXT_DISCOVER
    ]
    assert attempts[-1]["stage_status"] == StageExecutionStatus.SKIPPED_NOT_PRESENT.value


@pytest.mark.asyncio
async def test_task_upsert_failure_is_persisted_and_commit_safe(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, repository = pipeline_dependencies
    repository.error = RuntimeError("task queue unavailable")
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", True)
    record = _make_record(_make_docx_contract_payload(), offset=104)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    outcome = result.outcomes[0]
    assert outcome.failure_persisted is True
    assert len(outcome.pending_failure_refs) == 1
    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.FULLTEXT_DISCOVER.value
    assert failure["scope_type"] == ScopeType.DOCUMENT.value
    assert failure["error_type"] == ErrorType.RETRYABLE.value
    assert failure["error_code"] == "FULLTEXT_TASK_UPSERT_ERROR"
    assert manager.status_docs["doc-contract-1"]["stage_statuses"][
        SyncStage.FULLTEXT_DISCOVER.value
    ]["status"] == StageExecutionStatus.FAILURE.value


@pytest.mark.asyncio
async def test_task_upsert_failure_not_commit_safe_when_failure_persist_fails(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, repository = pipeline_dependencies
    repository.error = RuntimeError("task queue unavailable")

    async def fail_persist_failure(**_: Any) -> FailureRef:
        raise RuntimeError("mongo unavailable")

    manager.persist_failure = fail_persist_failure
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", True)
    record = _make_record(_make_docx_contract_payload(), offset=105)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is False
    assert result.outcomes[0].final_status == SyncOverallStatus.SUCCESS.value
    assert result.outcomes[0].failure_persisted is False
    assert result.outcomes[0].message == "FULLTEXT_TASK_UPSERT_ERROR"


@pytest.mark.asyncio
async def test_delete_event_does_not_trigger_fulltext_or_delete_chunks(
    pipeline_dependencies,
    monkeypatch,
):
    manager, _, repository = pipeline_dependencies
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=106)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert repository.upsert_calls == []
    stage_statuses = manager.status_docs["doc-1"].get("stage_statuses", {})
    assert SyncStage.FULLTEXT_DISCOVER.value not in stage_statuses
    assert SyncStage.DELETE_CHUNKS.value not in stage_statuses


class FakeFullTextRepository:
    def __init__(self) -> None:
        self.upsert_calls: list[list[Any]] = []
        self.error: Exception | None = None

    async def upsert_tasks(self, tasks):
        if self.error is not None:
            raise self.error
        self.upsert_calls.append(list(tasks))
        return {"matched": 0, "upserted": len(tasks)}
