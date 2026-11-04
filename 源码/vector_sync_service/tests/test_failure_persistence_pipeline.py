from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from app.models.sync_runtime import (
    ArtifactType,
    ConsumedRecord,
    EmbeddingItemResult,
    ErrorType,
    FailureRef,
    ReplayMode,
    ScopeType,
    SyncOverallStatus,
    SyncStage,
    normalize_record_update_time,
)
from app.services import batch_processor_existing as pipeline_module


class FakeManager:
    def __init__(self) -> None:
        self.status_docs: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, dict[str, Any]] = {}
        self.attempts: list[dict[str, Any]] = []
        self.resolved: list[tuple[str, str]] = []
        self.connected = True
        self._artifact_counter = 0

    async def get_status(self, doc_id: str) -> Optional[dict[str, Any]]:
        return self.status_docs.get(doc_id)

    async def initialize_status(self, *, doc_id: str, record_update_time: str, sub_doc_type: str, event_type: str) -> None:
        self.status_docs[doc_id] = {
            "_id": doc_id,
            "latest_record_update_time": record_update_time,
            "sub_doc_type": sub_doc_type,
            "latest_event_type": event_type,
            "overall_status": "PROCESSING",
            "core_status": "PROCESSING",
            "document_status": "PROCESSING",
            "fulltext_status": "PENDING",
            "image_status": "PENDING",
            "fulltext_required": None,
            "image_required": None,
            "retry_summary": {
                "has_pending_failure": False,
                "pending_stage_count": 0,
                "next_retry_at": None,
                "gave_up_count": 0,
            },
        }

    async def save_artifact(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        stage,
        scope_type,
        scope_id: str,
        artifact_type,
        content: Any,
        success_path: bool,
        content_type: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        expire_days: Optional[int] = None,
    ) -> str:
        self._artifact_counter += 1
        artifact_id = f"artifact-{self._artifact_counter}"
        self.artifacts[artifact_id] = {
            "doc_id": doc_id,
            "record_update_time": record_update_time,
            "sub_doc_type": sub_doc_type,
            "stage": stage.value,
            "scope_type": scope_type.value,
            "scope_id": scope_id,
            "artifact_type": artifact_type.value,
            "payload": content,
            "success_path": success_path,
            "content_type": content_type,
            "meta": meta or {},
            "expire_days": expire_days,
        }
        return artifact_id

    async def mark_stage_started(self, *, doc_id: str, stage) -> None:
        stage_statuses = self.status_docs.setdefault(doc_id, {}).setdefault("stage_statuses", {})
        stage_statuses[stage.value] = {"status": "PROCESSING"}

    async def mark_stage_finished(
        self,
        *,
        doc_id: str,
        stage,
        status,
        artifact_ref: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        stage_statuses = self.status_docs.setdefault(doc_id, {}).setdefault("stage_statuses", {})
        stage_statuses[stage.value] = {
            "status": status.value,
            "artifact_ref": artifact_ref,
            "error_code": error_code,
            "error_message": error_message,
            "duration_ms": duration_ms,
        }
        self.status_docs[doc_id]["last_success_stage"] = stage.value if status.value != "FAILURE" else self.status_docs[doc_id].get("last_success_stage")

    async def record_attempt(self, **kwargs) -> None:
        self.attempts.append(kwargs)

    async def persist_failure(
        self,
        *,
        doc_id: Optional[str],
        record_update_time: Optional[str],
        sub_doc_type: Optional[str],
        event_type: Optional[str],
        failure_stage,
        scope_type,
        scope_id: str,
        error_type,
        error_code: str,
        error_message: str,
        exception_class: str,
        stacktrace: str,
        replay_mode,
        artifact_refs: list[str],
        source_event_meta,
        pipeline: str = "",
        source_version: str = "",
        item_id: str = "",
        logical_collection: str = "",
        resolved_collection: str = "",
        target_signature: str = "",
        retry_increment: int = 0,
        max_retries: Optional[int] = None,
    ) -> FailureRef:
        failure_key = "|".join(
            [
                doc_id or "",
                record_update_time or "",
                failure_stage.value,
                scope_type.value,
                scope_id,
            ]
        )
        self.failures[failure_key] = {
            "failure_key": failure_key,
            "doc_id": doc_id,
            "record_update_time": record_update_time,
            "sub_doc_type": sub_doc_type,
            "event_type": event_type,
            "failure_stage": failure_stage.value,
            "scope_type": scope_type.value,
            "scope_id": scope_id,
            "error_type": error_type.value,
            "error_code": error_code,
            "error_message": error_message,
            "exception_class": exception_class,
            "stacktrace": stacktrace,
            "replay_mode": replay_mode.value,
            "artifact_refs": artifact_refs,
            "retry_count": retry_increment,
            "pipeline": pipeline,
            "source_version": source_version,
            "item_id": item_id,
            "logical_collection": logical_collection,
            "resolved_collection": resolved_collection,
            "target_signature": target_signature,
        }
        if doc_id:
            self.status_docs.setdefault(doc_id, {})["retry_summary"] = {
                "has_pending_failure": True,
                "pending_stage_count": 1,
                "next_retry_at": None,
                "gave_up_count": 0,
            }
        return FailureRef(
            failure_key=failure_key,
            failure_stage=failure_stage,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    async def finalize_document_status(
        self,
        *,
        doc_id: str,
        overall_status,
        current_stage,
        last_success_stage,
        progress: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status_docs.setdefault(doc_id, {})
        self.status_docs[doc_id]["overall_status"] = overall_status.value
        self.status_docs[doc_id]["current_stage"] = current_stage.value if current_stage else None
        self.status_docs[doc_id]["last_success_stage"] = last_success_stage.value if last_success_stage else None
        self.status_docs[doc_id]["progress"] = progress or {}

    async def update_pipeline_status(
        self,
        *,
        doc_id: str,
        document_status: Optional[str] = None,
        fulltext_status: Optional[str] = None,
        image_status: Optional[str] = None,
        core_status: Optional[str] = None,
        overall_status: Optional[Any] = None,
        fulltext_required: Optional[bool] = None,
        image_required: Optional[bool] = None,
        progress: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status_docs.setdefault(doc_id, {})
        if document_status is not None:
            self.status_docs[doc_id]["document_status"] = document_status
        if fulltext_status is not None:
            self.status_docs[doc_id]["fulltext_status"] = fulltext_status
        if image_status is not None:
            self.status_docs[doc_id]["image_status"] = image_status
        if core_status is not None:
            self.status_docs[doc_id]["core_status"] = core_status
        if overall_status is not None:
            self.status_docs[doc_id]["overall_status"] = (
                overall_status.value if hasattr(overall_status, "value") else str(overall_status)
            )
        if fulltext_required is not None:
            self.status_docs[doc_id]["fulltext_required"] = bool(fulltext_required)
        if image_required is not None:
            self.status_docs[doc_id]["image_required"] = bool(image_required)
        if progress:
            current = self.status_docs[doc_id].setdefault("progress", {})
            current.update(progress)

    async def refresh_async_pipeline_status(
        self,
        *,
        doc_id: str,
        record_update_time: Optional[str] = None,
    ) -> None:
        self.status_docs.setdefault(doc_id, {})
        progress = self.status_docs[doc_id].setdefault("progress", {})
        fulltext_expected = int(progress.get("fulltext_task_count") or 0)
        image_expected = int(progress.get("image_count_expected") or 0)
        fulltext_status = (
            "PENDING" if fulltext_expected else "NOT_PRESENT"
        )
        image_status = "PENDING" if image_expected else "NOT_PRESENT"
        self.status_docs[doc_id]["fulltext_status"] = fulltext_status
        self.status_docs[doc_id]["image_status"] = image_status
        self.status_docs[doc_id]["core_status"] = (
            "PROCESSING" if fulltext_status == "PENDING" else "SUCCESS"
        )

    async def resolve_failures_for_document(self, doc_id: str, record_update_time: str) -> None:
        self.resolved.append((doc_id, record_update_time))
        for failure in self.failures.values():
            if failure["doc_id"] == doc_id and failure["record_update_time"] == record_update_time:
                failure["status"] = "DONE"
        self.status_docs.setdefault(doc_id, {}).setdefault("retry_summary", {})
        self.status_docs[doc_id]["retry_summary"]["has_pending_failure"] = False

    async def mark_failure_skipped_stale(self, failure_key: str, reason: str) -> None:
        self.failures[failure_key]["status"] = "SKIPPED_STALE"
        self.failures[failure_key]["error_message"] = reason

    async def mark_failure_done(self, failure_key: str, note: str = "") -> None:
        self.failures[failure_key]["status"] = "DONE"
        self.failures[failure_key]["resolution_note"] = note

    async def get_failure_artifacts(self, artifact_refs: list[str]) -> list[dict[str, Any]]:
        return [{**self.artifacts[ref], "_id": ref} for ref in artifact_refs if ref in self.artifacts]

    async def mark_mongo_unavailable(self, context: dict[str, Any]) -> None:
        self.last_mongo_error = context


class FakeMilvus:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        insert_should_fail: bool = False,
        delete_should_fail: bool = False,
    ) -> None:
        self.insert_should_fail = should_fail or insert_should_fail
        self.delete_should_fail = delete_should_fail
        self.insert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.call_order: list[dict[str, Any]] = []

    def process_and_insert(self, collection_name: str, articles: list[Any], vector_dim: int = 2560):
        call = {
            "op": "process_and_insert",
            "collection_name": collection_name,
            "articles": articles,
            "sys_ids": [getattr(article, "sysId", None) for article in articles],
            "vector_dim": vector_dim,
        }
        self.insert_calls.append(call)
        self.call_order.append(call)
        if self.insert_should_fail:
            raise RuntimeError("milvus temporary unavailable")
        return {"insert_count": len(articles)}

    def delete_by_sys_id(self, collection_name: str, sys_id: str):
        call = {
            "op": "delete_by_sys_id",
            "collection_name": collection_name,
            "sys_id": sys_id,
        }
        self.delete_calls.append(call)
        self.call_order.append(call)
        if self.delete_should_fail:
            raise RuntimeError("milvus delete temporary unavailable")
        return {"delete_count": 1}

    def delete_by_sys_ids(self, collection_name: str, sys_ids: list[str]):
        call = {
            "op": "delete_by_sys_ids",
            "collection_name": collection_name,
            "sys_ids": sys_ids,
        }
        self.delete_calls.append(call)
        self.call_order.append(call)
        if self.delete_should_fail:
            raise RuntimeError("milvus delete temporary unavailable")
        return {"delete_count": len(sys_ids)}


class FakeEmbedder:
    def __init__(self, results: list[EmbeddingItemResult]) -> None:
        self.results = results
        self.calls: list[list[str]] = []

    async def embed_texts_detailed(self, texts: list[str]) -> list[EmbeddingItemResult]:
        self.calls.append(texts)
        return self.results


def _make_valid_payload(
    *,
    doc_id: str = "doc-1",
    record_update_time: int = 1770502340000,
    event_type: str = "ADD",
) -> dict[str, Any]:
    return {
        "subDocType": "Article",
        "dataType": "Journal",
        "type": event_type,
        "data": [
            {
                "id": doc_id,
                "cls": "Article",
                "dp": {
                    "title": {"field": "title", "value": ["title text"]},
                    "abstract": {"field": "abstract", "value": ["abstract text"]},
                    "keyword": {
                        "field": "keyword",
                        "value": ["kw1", "kw2"],
                        "multiValue": True,
                    },
                    "sysDataUpdateTime": {
                        "field": "sysDataUpdateTime",
                        "value": [record_update_time],
                    },
                    "sysSubType": {
                        "field": "sysSubType",
                        "value": ["Article"],
                    },
                },
            }
        ],
    }


def _make_docx_contract_payload(
    *,
    doc_id: str = "doc-contract-1",
    record_update_time: int = 1770502340000,
    sub_doc_type: str = "Article",
    event_type: str = "ADD",
) -> dict[str, Any]:
    return {
        "subDocType": sub_doc_type,
        "dataType": "Journal",
        "type": event_type,
        "data": [
            {
                "sysId": doc_id,
                "cls": "Article",
                "dp": {
                    "title": {
                        "field": "title",
                        "value": ["Contract title"],
                        "multiValue": True,
                    },
                    "abstract": {
                        "field": "abstract",
                        "value": ["Contract abstract"],
                        "multiValue": True,
                    },
                    "keyword": {
                        "field": "keyword",
                        "value": ["contract-kw-1", "contract-kw-2"],
                        "multiValue": True,
                    },
                    "year": {"field": "year", "value": ["2026"]},
                    "sysDataUpdateTime": {
                        "field": "sysDataUpdateTime",
                        "value": [record_update_time],
                    },
                    "sysSubType": {
                        "field": "sysSubType",
                        "value": [sub_doc_type],
                    },
                    "contractExtraField": {
                        "field": "contractExtraField",
                        "value": ["extra-value"],
                    },
                },
                "op": {
                    "hasFullTextFile": {
                        "field": "hasFullTextFile",
                        "value": [
                            {
                                "id": "fulltext-1",
                                "cls": "FullTextFile",
                                "dp": {
                                    "fileName": {
                                        "field": "fileName",
                                        "value": ["paper.mmd"],
                                    },
                                    "fileUrl": {
                                        "field": "fileUrl",
                                        "value": ["/fulltext/paper.mmd"],
                                    },
                                },
                            }
                        ],
                    }
                },
            }
        ],
    }


def _success_embedder_with_dim(vector_dim: int) -> FakeEmbedder:
    return FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * vector_dim),
            EmbeddingItemResult(success=True, vector=[2.0] * vector_dim),
            EmbeddingItemResult(success=True, vector=[3.0] * vector_dim),
        ]
    )


def _make_record(payload: dict[str, Any], *, offset: int = 10) -> ConsumedRecord:
    return ConsumedRecord(
        topic="VECTOR_JOURNAL_ARTICLE",
        partition=0,
        offset=offset,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint=f"topic|0|{offset}|test",
        parsed_payload=payload,
    )


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    manager = FakeManager()
    milvus = FakeMilvus()
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0, 0.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
        ]
    )
    monkeypatch.setattr(pipeline_module, "get_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(pipeline_module, "get_milvus_service", lambda: milvus)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(pipeline_module.settings, "VECTOR_DIM", 3)
    monkeypatch.setattr(pipeline_module.settings, "MILVUS_COLLECTION", "unit_test_collection")
    # These legacy integration tests exercise the optional audit mode. The
    # production default remains failure-only and is covered explicitly below.
    monkeypatch.setattr(pipeline_module.settings, "MONGO_SUCCESS_AUDIT_ENABLED", True)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_TASK_UPSERT_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_TASK_UPSERT_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_DELETE_CHUNKS_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "IMAGE_DELETE_IMAGES_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "LIVE_E2E_NO_DELETE", False)
    return manager, milvus, embedder


@pytest.mark.parametrize("event_type", ["ADD", "UPDATE"])
@pytest.mark.asyncio
async def test_upsert_writes_without_predelete_and_records_summary(patch_dependencies, event_type):
    manager, milvus, _ = patch_dependencies
    record = _make_record(_make_valid_payload(event_type=event_type), offset=20)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert [call["op"] for call in milvus.call_order] == ["process_and_insert"]
    assert milvus.delete_calls == []
    assert milvus.insert_calls[0]["collection_name"] == "unit_test_collection"
    assert milvus.insert_calls[0]["sys_ids"] == ["doc-1"]

    summaries = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.MILVUS_WRITE_DOCUMENT.value
        and artifact["artifact_type"] == ArtifactType.MILVUS_WRITE_SUMMARY.value
    ]
    assert len(summaries) == 1
    summary = summaries[0]["payload"]
    assert summary["collection_name"] == "unit_test_collection"
    assert summary["sys_id"] == "doc-1"
    assert summary["write_mode"] == "primary_key_upsert"
    assert summary["insert_result"] == {"insert_count": 1}


@pytest.mark.asyncio
async def test_failure_only_mongo_mode_skips_success_audit_writes(
    patch_dependencies,
    monkeypatch,
):
    manager, milvus, _ = patch_dependencies
    monkeypatch.setattr(
        pipeline_module.settings,
        "MONGO_SUCCESS_AUDIT_ENABLED",
        False,
        raising=False,
    )

    result = await pipeline_module.process_batch_existing_arch(
        [_make_record(_make_valid_payload(), offset=21)]
    )

    assert result.commit_safe is True
    assert result.success == 1
    assert [call["op"] for call in milvus.call_order] == ["process_and_insert"]
    assert milvus.delete_calls == []
    assert manager.status_docs == {}
    assert manager.attempts == []
    assert manager.artifacts == {}


@pytest.mark.asyncio
async def test_metadata_bulk_batches_embeddings_and_milvus_writes(
    patch_dependencies,
    monkeypatch,
):
    manager, milvus, _ = patch_dependencies
    bulk_embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0, 0.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
            EmbeddingItemResult(success=True, vector=[2.0, 0.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 2.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 2.0]),
        ]
    )
    monkeypatch.setattr(
        pipeline_module.settings,
        "MONGO_SUCCESS_AUDIT_ENABLED",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module.settings,
        "DOCUMENT_BULK_MILVUS_BATCH_SIZE",
        512,
        raising=False,
    )
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: bulk_embedder)

    result = await pipeline_module.process_batch_existing_arch(
        [
            _make_record(_make_valid_payload(doc_id="doc-bulk-1"), offset=27),
            _make_record(_make_valid_payload(doc_id="doc-bulk-2"), offset=28),
        ],
        mode="metadata_bulk",
    )

    assert result.success == 2
    assert result.fail == 0
    assert len(bulk_embedder.calls) == 1
    assert len(bulk_embedder.calls[0]) == 6
    assert [call["op"] for call in milvus.call_order] == ["process_and_insert"]
    assert milvus.delete_calls == []
    assert milvus.insert_calls[0]["sys_ids"] == ["doc-bulk-1", "doc-bulk-2"]
    assert manager.status_docs == {}
    assert manager.attempts == []
    assert manager.artifacts == {}


@pytest.mark.asyncio
async def test_metadata_bulk_uses_placeholder_vector_for_empty_optional_field(
    patch_dependencies,
    monkeypatch,
):
    _, milvus, _ = patch_dependencies
    payload = _make_valid_payload(doc_id="doc-empty-abstract")
    payload["data"][0]["dp"].pop("abstract")
    empty_result = EmbeddingItemResult(
        success=False,
        vector=None,
        error_type=ErrorType.NON_RETRYABLE,
        error_code="EMBEDDING_INPUT_EMPTY",
        error_message="embedding input is empty",
        retryable=False,
        skipped_empty=True,
    )
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0, 0.0, 0.0]),
            empty_result,
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
        ]
    )
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(pipeline_module.settings, "MONGO_SUCCESS_AUDIT_ENABLED", False)

    result = await pipeline_module.process_batch_existing_arch(
        [_make_record(payload, offset=29)],
        mode="metadata_bulk",
    )

    assert result.success == 1
    article = milvus.insert_calls[0]["articles"][0]
    assert article.abstract_embed == [1e-6, 0.0, 0.0]


@pytest.mark.asyncio
async def test_metadata_bulk_bounds_overlong_embedding_field(
    patch_dependencies,
    monkeypatch,
):
    _, milvus, _ = patch_dependencies
    payload = _make_valid_payload(doc_id="doc-overlong-abstract")
    payload["data"][0]["dp"]["abstract"] = ["a" * 20]
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0, 0.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
        ]
    )
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(pipeline_module.settings, "MONGO_SUCCESS_AUDIT_ENABLED", False)
    monkeypatch.setattr(pipeline_module.settings, "METADATA_EMBEDDING_MAX_CHARS", 12)

    result = await pipeline_module.process_batch_existing_arch(
        [_make_record(payload, offset=30)],
        mode="metadata_bulk",
    )

    assert result.success == 1
    assert embedder.calls[0][1] == "a" * 12
    assert milvus.insert_calls[0]["sys_ids"] == ["doc-overlong-abstract"]


@pytest.mark.parametrize("event_type", ["ADD", "UPDATE"])
@pytest.mark.asyncio
async def test_docx_contract_upsert_preserves_article_fields_and_vectors(
    patch_dependencies,
    monkeypatch,
    event_type,
):
    manager, milvus, _ = patch_dependencies
    vector_dim = 2560
    doc_id = f"doc-contract-{event_type.lower()}"
    record_update_time = 1770502340000
    expected_update_time = normalize_record_update_time(record_update_time)
    monkeypatch.setattr(pipeline_module.settings, "VECTOR_DIM", vector_dim)
    monkeypatch.setattr(
        pipeline_module,
        "get_embedding_client",
        lambda: _success_embedder_with_dim(vector_dim),
    )
    record = _make_record(
        _make_docx_contract_payload(
            doc_id=doc_id,
            record_update_time=record_update_time,
            event_type=event_type,
        ),
        offset=30 if event_type == "ADD" else 31,
    )

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].doc_id == doc_id
    assert result.outcomes[0].record_update_time == expected_update_time
    assert [call["op"] for call in milvus.call_order] == ["process_and_insert"]
    assert milvus.delete_calls == []
    assert milvus.insert_calls[0]["vector_dim"] == vector_dim

    inserted_article = milvus.insert_calls[0]["articles"][0]
    assert inserted_article.ID == doc_id
    assert inserted_article.sysId == doc_id
    assert inserted_article.cls == "Article"
    assert inserted_article.title == ["Contract title"]
    assert inserted_article.abstract == ["Contract abstract"]
    assert inserted_article.keyword == ["contract-kw-1", "contract-kw-2"]
    assert inserted_article.year == "2026"
    assert inserted_article.sysSubType == "Article"
    assert inserted_article.dynamic_fields == {}
    assert inserted_article.hasFullTextFileid == ["fulltext-1"]
    assert inserted_article.hasFullTextFilename == ["paper.mmd"]

    for vector_field in ("title_vector", "abstract_vector", "keyword_vector"):
        vector = getattr(inserted_article, vector_field)
        assert isinstance(vector, list)
        assert len(vector) == vector_dim

    status = manager.status_docs[doc_id]
    assert status["_id"] == doc_id
    assert status["latest_record_update_time"] == expected_update_time
    assert status["sub_doc_type"] == "Article"
    assert status["latest_event_type"] == event_type


@pytest.mark.asyncio
async def test_live_e2e_no_delete_upsert_inserts_without_delete(patch_dependencies, monkeypatch):
    manager, milvus, _ = patch_dependencies
    monkeypatch.setattr(pipeline_module.settings, "LIVE_E2E_NO_DELETE", True)
    record = _make_record(_make_valid_payload(event_type="ADD"), offset=120)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert milvus.delete_calls == []
    assert [call["op"] for call in milvus.call_order] == ["process_and_insert"]
    milvus_artifacts = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.MILVUS_WRITE_DOCUMENT.value
    ]
    assert milvus_artifacts[-1]["payload"]["write_mode"] == "primary_key_upsert"


@pytest.mark.asyncio
async def test_document_delete_success_marks_deleted_and_deletes_milvus(patch_dependencies):
    manager, milvus, _ = patch_dependencies
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=21)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert [call["op"] for call in milvus.call_order] == ["delete_by_sys_id"]
    assert milvus.delete_calls[0]["sys_id"] == "doc-1"
    assert milvus.insert_calls == []

    status = manager.status_docs["doc-1"]
    assert status["overall_status"] == SyncOverallStatus.DELETED.value
    assert status["current_stage"] == SyncStage.DELETE_DOCUMENT.value
    assert status["last_success_stage"] == SyncStage.DELETE_DOCUMENT.value


@pytest.mark.asyncio
async def test_live_e2e_no_delete_delete_event_records_noop_without_delete(patch_dependencies, monkeypatch):
    manager, milvus, _ = patch_dependencies
    monkeypatch.setattr(pipeline_module.settings, "LIVE_E2E_NO_DELETE", True)
    record = _make_record(_make_valid_payload(event_type="DELETE"), offset=121)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.outcomes[0].final_status == SyncOverallStatus.DELETED.value
    assert milvus.delete_calls == []
    assert milvus.insert_calls == []
    delete_artifacts = [
        artifact
        for artifact in manager.artifacts.values()
        if artifact["stage"] == SyncStage.DELETE_DOCUMENT.value
        and artifact["artifact_type"] == ArtifactType.MILVUS_WRITE_SUMMARY.value
    ]
    assert delete_artifacts[-1]["payload"] == {
        "skipped": True,
        "status": "DELETE_NOOP_RECORDED",
        "reason": "LIVE_E2E_NO_DELETE",
        "operation": "delete_document",
        "collection_name": "unit_test_collection",
        "doc_id": "doc-1",
    }
    assert manager.status_docs["doc-1"]["progress"]["delete_noop_recorded"] is True


@pytest.mark.asyncio
async def test_update_does_not_call_delete_before_upsert(patch_dependencies, monkeypatch):
    manager, _, embedder = patch_dependencies
    failing_milvus = FakeMilvus(delete_should_fail=True)
    monkeypatch.setattr(pipeline_module, "get_milvus_service", lambda: failing_milvus)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    record = _make_record(_make_valid_payload(), offset=22)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.success == 1
    assert result.fail == 0
    assert failing_milvus.delete_calls == []
    assert [call["op"] for call in failing_milvus.call_order] == ["process_and_insert"]


@pytest.mark.asyncio
async def test_older_message_does_not_resurrect_newer_deleted_document(patch_dependencies):
    manager, milvus, _ = patch_dependencies
    older_time_raw = 1770502340000
    newer_time = normalize_record_update_time(1770502350000)
    manager.status_docs["doc-1"] = {
        "_id": "doc-1",
        "latest_record_update_time": newer_time,
        "overall_status": SyncOverallStatus.DELETED.value,
        "current_stage": SyncStage.DELETE_DOCUMENT.value,
        "last_success_stage": SyncStage.DELETE_DOCUMENT.value,
        "retry_summary": {"has_pending_failure": False},
    }
    record = _make_record(
        _make_docx_contract_payload(
            doc_id="doc-1",
            record_update_time=older_time_raw,
            event_type="ADD",
        ),
        offset=23,
    )

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.skip == 1
    assert result.outcomes[0].message == "older_than_latest_processed_version"
    assert manager.status_docs["doc-1"]["latest_record_update_time"] == newer_time
    assert manager.status_docs["doc-1"]["overall_status"] == SyncOverallStatus.DELETED.value
    assert milvus.call_order == []
    assert manager.artifacts == {}
    assert manager.failures == {}


@pytest.mark.asyncio
async def test_same_version_success_is_not_reprocessed(patch_dependencies):
    manager, milvus, _ = patch_dependencies
    same_time = normalize_record_update_time(1770502340000)
    manager.status_docs["doc-1"] = {
        "_id": "doc-1",
        "latest_record_update_time": same_time,
        "overall_status": SyncOverallStatus.SUCCESS.value,
        "current_stage": SyncStage.MILVUS_WRITE_DOCUMENT.value,
        "last_success_stage": SyncStage.MILVUS_WRITE_DOCUMENT.value,
        "retry_summary": {"has_pending_failure": False},
    }
    record = _make_record(_make_docx_contract_payload(doc_id="doc-1"), offset=24)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.skip == 1
    assert result.outcomes[0].message == "same_version_already_processed"
    assert milvus.call_order == []
    assert manager.artifacts == {}
    assert manager.failures == {}


@pytest.mark.asyncio
async def test_same_version_pending_failure_is_not_duplicated(patch_dependencies):
    manager, milvus, _ = patch_dependencies
    same_time = normalize_record_update_time(1770502340000)
    failure_key = "|".join(
        [
            "doc-1",
            same_time,
            SyncStage.MILVUS_WRITE_DOCUMENT.value,
            ScopeType.DOCUMENT.value,
            "doc-1",
        ]
    )
    manager.status_docs["doc-1"] = {
        "_id": "doc-1",
        "latest_record_update_time": same_time,
        "overall_status": SyncOverallStatus.FAILURE.value,
        "current_stage": SyncStage.MILVUS_WRITE_DOCUMENT.value,
        "last_success_stage": SyncStage.DOC_EMBED_KEYWORD.value,
        "retry_summary": {"has_pending_failure": True, "pending_stage_count": 1},
    }
    manager.failures[failure_key] = {
        "failure_key": failure_key,
        "doc_id": "doc-1",
        "record_update_time": same_time,
        "failure_stage": SyncStage.MILVUS_WRITE_DOCUMENT.value,
        "scope_type": ScopeType.DOCUMENT.value,
        "scope_id": "doc-1",
        "error_code": "MILVUS_WRITE_ERROR",
    }
    record = _make_record(_make_docx_contract_payload(doc_id="doc-1"), offset=25)

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.skip == 1
    assert result.outcomes[0].message == "same_version_pending_failure_already_recorded"
    assert milvus.call_order == []
    assert manager.artifacts == {}
    assert len(manager.failures) == 1
    assert manager.failures[failure_key]["error_code"] == "MILVUS_WRITE_ERROR"


@pytest.mark.asyncio
async def test_poison_message_persisted_commit_safe(patch_dependencies):
    manager, milvus, _ = patch_dependencies
    record = ConsumedRecord(
        topic="VECTOR_JOURNAL_ARTICLE",
        partition=0,
        offset=1,
        raw_bytes=b"{bad json",
        raw_text="{bad json",
        event_fingerprint="topic|0|1|abc",
        decode_error="Expecting property name enclosed in double quotes",
    )

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.fail == 1
    assert len(manager.failures) == 1
    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.KAFKA_DECODE.value
    assert failure["scope_type"] == ScopeType.EVENT.value
    assert failure["replay_mode"] == ReplayMode.FROM_RAW_EVENT.value
    assert milvus.delete_calls == []
    assert milvus.insert_calls == []


@pytest.mark.asyncio
async def test_embedding_failure_persisted_without_milvus_write(patch_dependencies, monkeypatch):
    manager, milvus, _ = patch_dependencies
    failing_embedder = FakeEmbedder(
        [
            EmbeddingItemResult(
                success=False,
                vector=None,
                error_type=ErrorType.RETRYABLE,
                error_code="EMBEDDING_RATE_LIMIT",
                error_message="429",
                retryable=True,
            ),
            EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
            EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
        ]
    )
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: failing_embedder)

    record = ConsumedRecord(
        topic="VECTOR_JOURNAL_ARTICLE",
        partition=0,
        offset=2,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint="topic|0|2|def",
        parsed_payload=_make_valid_payload(),
    )

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.fail == 1
    assert result.outcomes[0].final_status == "PARTIAL_FAILURE"
    assert len(result.outcomes[0].pending_failure_refs) == 1
    assert len(manager.failures) == 1
    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.DOC_EMBED_TITLE.value
    assert failure["scope_type"] == ScopeType.ARTICLE.value
    assert failure["replay_mode"] == ReplayMode.FROM_PARSED_DOCUMENT.value
    assert milvus.delete_calls == []
    assert milvus.insert_calls == []


@pytest.mark.asyncio
async def test_failure_only_mongo_mode_keeps_replay_payload_on_embedding_failure(
    patch_dependencies,
    monkeypatch,
):
    manager, milvus, _ = patch_dependencies
    monkeypatch.setattr(
        pipeline_module.settings,
        "MONGO_SUCCESS_AUDIT_ENABLED",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_embedding_client",
        lambda: FakeEmbedder(
            [
                EmbeddingItemResult(
                    success=False,
                    vector=None,
                    error_type=ErrorType.RETRYABLE,
                    error_code="EMBEDDING_RATE_LIMIT",
                    error_message="429",
                    retryable=True,
                ),
                EmbeddingItemResult(success=True, vector=[0.0, 1.0, 0.0]),
                EmbeddingItemResult(success=True, vector=[0.0, 0.0, 1.0]),
            ]
        ),
    )

    result = await pipeline_module.process_batch_existing_arch(
        [_make_record(_make_valid_payload(), offset=26)]
    )

    assert result.commit_safe is True
    assert result.fail == 1
    assert len(manager.failures) == 1
    failure = next(iter(manager.failures.values()))
    replay_payloads = [
        item for item in manager.artifacts.values()
        if item["artifact_type"] == ArtifactType.RAW_EVENT_JSON.value
    ]
    assert len(replay_payloads) == 1
    assert replay_payloads[0]["success_path"] is False
    assert replay_payloads[0]["payload"] == _make_valid_payload()
    assert replay_payloads[0]["doc_id"] == "doc-1"
    assert replay_payloads[0]["scope_id"] == "doc-1"
    assert replay_payloads[0]["stage"] == SyncStage.DOC_EMBED_TITLE.value
    assert replay_payloads[0] in manager.artifacts.values()
    assert failure["artifact_refs"]
    assert milvus.call_order == []


@pytest.mark.asyncio
async def test_milvus_failure_persisted_commit_safe(patch_dependencies, monkeypatch):
    manager, _, embedder = patch_dependencies
    failing_milvus = FakeMilvus(insert_should_fail=True)
    monkeypatch.setattr(pipeline_module, "get_milvus_service", lambda: failing_milvus)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)

    record = ConsumedRecord(
        topic="VECTOR_JOURNAL_ARTICLE",
        partition=0,
        offset=3,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint="topic|0|3|ghi",
        parsed_payload=_make_valid_payload(),
    )

    result = await pipeline_module.process_batch_existing_arch([record])

    assert result.commit_safe is True
    assert result.fail == 1
    assert [call["op"] for call in failing_milvus.call_order] == ["process_and_insert"]
    assert failing_milvus.delete_calls == []
    assert len(failing_milvus.delete_calls) == 0
    assert len(failing_milvus.insert_calls) == 1
    failure = next(iter(manager.failures.values()))
    assert failure["failure_stage"] == SyncStage.MILVUS_WRITE_DOCUMENT.value
    assert failure["error_code"] == "MILVUS_WRITE_ERROR"
