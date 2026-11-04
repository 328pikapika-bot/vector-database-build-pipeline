from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.sync_runtime import utcnow
from app.config import settings
from app.services.image_task_manager import ImageTask, ImageTaskStatus
from app.services.image_task_repository import ImageTaskRepository


@pytest.mark.asyncio
async def test_successful_task_is_deleted_and_republished_task_starts_pending():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.ensure_indexes()

    result = await repository.upsert_tasks([_task("task-1")])

    assert result["upserted"] == 1
    assert collection.docs["task-1"]["status"] == ImageTaskStatus.PENDING.value
    assert collection.indexes

    await repository.mark_task_success(
        "task-1",
        artifact_refs=["image-raw"],
        result_summary={"written": 1},
    )
    assert "task-1" not in collection.docs
    await repository.upsert_tasks([_task("task-1")])

    assert collection.docs["task-1"]["status"] == ImageTaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_pending_task_can_be_claimed():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    claimed = await repository.claim_next_task(
        worker_id="worker-1",
        lease_seconds=120,
    )

    assert claimed is not None
    assert claimed["task_id"] == "task-1"
    assert claimed["status"] == ImageTaskStatus.LEASED.value
    assert claimed["lease_owner"] == "worker-1"
    assert claimed["attempt_count"] == 1


@pytest.mark.asyncio
async def test_claim_next_tasks_respects_limit_and_claim_order():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1"), _task("task-2"), _task("task-3")])

    claimed = await repository.claim_next_tasks(
        worker_id="worker-1",
        lease_seconds=120,
        limit=2,
    )

    assert [task["task_id"] for task in claimed] == ["task-1", "task-2"]
    assert collection.docs["task-1"]["status"] == ImageTaskStatus.LEASED.value
    assert collection.docs["task-2"]["status"] == ImageTaskStatus.LEASED.value
    assert collection.docs["task-3"]["status"] == ImageTaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_claim_next_task_skips_excluded_active_task_ids():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1"), _task("task-2")])

    claimed = await repository.claim_next_task(
        worker_id="worker-1",
        lease_seconds=120,
        exclude_task_ids={"task-1"},
    )

    assert claimed is not None
    assert claimed["task_id"] == "task-2"
    assert collection.docs["task-1"]["status"] == ImageTaskStatus.PENDING.value
    assert collection.docs["task-2"]["status"] == ImageTaskStatus.LEASED.value


@pytest.mark.asyncio
async def test_unexpired_lease_is_not_claimed_but_expired_lease_is_reclaimed():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    collection.docs["task-1"].update(
        {
            "status": ImageTaskStatus.LEASED.value,
            "lease_owner": "worker-old",
            "lease_expires_at": utcnow() + timedelta(seconds=60),
        }
    )
    assert await repository.claim_next_task(worker_id="worker-2", lease_seconds=120) is None

    collection.docs["task-1"]["lease_expires_at"] = utcnow() - timedelta(seconds=1)
    claimed = await repository.claim_next_task(
        worker_id="worker-2",
        lease_seconds=120,
    )

    assert claimed is not None
    assert claimed["lease_owner"] == "worker-2"
    assert claimed["attempt_count"] == 1


@pytest.mark.asyncio
async def test_retryable_failure_can_be_claimed_only_after_next_retry_at():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "IMAGE_TASK_MAX_ATTEMPTS", 2)
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    try:
        await repository.upsert_tasks([_task("task-1")])
        await repository.mark_task_failure(
            "task-1",
            error_code="IMAGE_DOWNLOAD_RETRYABLE",
            error_message="network down",
            retryable=True,
        )

        assert collection.docs["task-1"]["retry_count"] == 1
        assert collection.docs["task-1"]["next_retry_at"] is not None
        assert await repository.claim_next_task(worker_id="worker-1", lease_seconds=120) is None

        collection.docs["task-1"]["next_retry_at"] = utcnow() - timedelta(seconds=1)
        claimed = await repository.claim_next_task(
            worker_id="worker-1",
            lease_seconds=120,
        )

        assert claimed is not None
        assert claimed["status"] == ImageTaskStatus.LEASED.value
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_retryable_image_task_is_not_retryable_when_only_one_attempt_allowed(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_TASK_MAX_ATTEMPTS", 1)
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    await repository.mark_task_failure(
        "task-1",
        error_code="IMAGE_PROVIDER_NETWORK_ERROR",
        error_message="network down",
        retryable=True,
    )

    failure = collection.docs["task-1"]
    assert failure["retry_count"] == 1
    assert failure["retryable"] is False
    assert failure["max_attempts"] == 1
    assert failure["next_retry_at"] is None


@pytest.mark.asyncio
async def test_retryable_image_task_stops_after_configured_attempts(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_TASK_MAX_ATTEMPTS", 2)
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    await repository.mark_task_failure(
        "task-1",
        error_code="IMAGE_PROVIDER_NETWORK_ERROR",
        error_message="network down",
        retryable=True,
    )
    first_failure = collection.docs["task-1"]
    assert first_failure["retry_count"] == 1
    assert first_failure["next_retry_at"] is not None

    await repository.mark_task_failure(
        "task-1",
        error_code="IMAGE_PROVIDER_NETWORK_ERROR",
        error_message="network down again",
        retryable=True,
    )
    second_failure = collection.docs["task-1"]
    assert second_failure["retry_count"] == 2
    assert second_failure["max_attempts"] == 2
    assert second_failure["next_retry_at"] is None


@pytest.mark.asyncio
async def test_held_history_is_counted_but_not_claimed():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])
    collection.docs["task-1"]["status"] = ImageTaskStatus.HELD_HISTORY.value

    summary = await repository.get_summary()
    claimed = await repository.claim_next_task(
        worker_id="worker-1",
        lease_seconds=120,
    )

    assert summary["counts"][ImageTaskStatus.HELD_HISTORY.value] == 1
    assert claimed is None


@pytest.mark.asyncio
async def test_success_task_is_deleted_while_skipped_task_remains_recorded():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1"), _task("task-2")])

    await repository.mark_task_success(
        "task-1",
        artifact_refs=["raw", "summary"],
        result_summary={"image_id": "doc-1_img_1"},
    )
    await repository.mark_task_skipped("task-2", reason="SKIPPED_NOT_PRESENT")

    task = await repository.get_task("task-1")
    summary = await repository.get_summary()

    assert task is None
    assert collection.docs["task-2"]["skip_reason"] == "SKIPPED_NOT_PRESENT"
    assert summary["collection"] == "doc_image_tasks"
    assert summary["total"] == 1
    assert summary["counts"][ImageTaskStatus.SUCCESS.value] == 0
    assert summary["counts"][ImageTaskStatus.SKIPPED.value] == 1


@pytest.mark.asyncio
async def test_success_summary_is_not_retained_in_task_queue():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    await repository.mark_task_success(
        "task-1",
        artifact_refs=[],
        result_summary={
            "task_id": "task-1",
            "image_id": "doc-1_img_1",
            "success_stage": "milvus_write",
            "milvus_write_mode": "upsert_if_exists",
        },
    )

    task = await repository.get_task("task-1")
    assert task is None


@pytest.mark.asyncio
async def test_status_only_success_summary_is_not_retained_in_task_queue():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    await repository.mark_task_success(
        "task-1",
        artifact_refs=[],
        result_summary={
            "task_id": "task-1",
            "image_id": "doc-1_img_1",
            "success_stage": "download",
            "milvus_write_mode": "replace",
        },
    )

    task = await repository.get_task("task-1")
    assert task is None


@pytest.mark.asyncio
async def test_mark_task_success_minimal_deletes_task():
    collection = FakeCollection()
    repository = ImageTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    await repository.mark_task_success_minimal(
        "task-1",
        image_id="doc-1_img_1",
        success_stage="milvus_write",
        milvus_write_mode="insert_only",
        artifact_refs=["manifest-1"],
    )

    task = await repository.get_task("task-1")
    assert task is None


def test_fake_collection_rejects_duplicate_update_fields_across_operators():
    collection = FakeCollection()

    with pytest.raises(ValueError, match="Cannot update 'task_id' and 'task_id' at the same time"):
        collection.update_one(
            {"_id": "task-1"},
            {
                "$setOnInsert": {"_id": "task-1", "task_id": "task-1"},
                "$set": {"task_id": "task-1"},
            },
            upsert=True,
        )


def _task(task_id: str) -> ImageTask:
    return ImageTask(
        task_id=task_id,
        doc_id="doc-1",
        record_update_time="2026-06-07T00:00:00+00:00",
        relative_url=f"/images/{task_id}.png",
        image_id="doc-1_img_1",
        sequence="1",
        title="Figure",
        name=f"{task_id}.png",
    )


class FakeCollection:
    name = "doc_image_tasks"

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.indexes: list[tuple] = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def update_one(self, filter_doc, update_doc, upsert=False):
        self._validate_update_doc(update_doc)
        task_id = filter_doc["_id"]
        exists = task_id in self.docs
        if not exists:
            if not upsert:
                return SimpleNamespace(matched_count=0, upserted_id=None)
            self.docs[task_id] = deepcopy(update_doc.get("$setOnInsert", {}))
            self.docs[task_id].setdefault("_id", task_id)
        self._apply_update(self.docs[task_id], update_doc)
        return SimpleNamespace(
            matched_count=1 if exists else 0,
            upserted_id=None if exists else task_id,
        )

    def find_one(self, filter_doc):
        for doc in self.docs.values():
            if self._matches(doc, filter_doc):
                return deepcopy(doc)
        return None

    def delete_one(self, filter_doc):
        self.docs.pop(filter_doc["_id"], None)

    def find_one_and_update(self, filter_doc, update_doc, sort=None, return_document=None):
        matches = [doc for doc in self.docs.values() if self._matches(doc, filter_doc)]
        if not matches:
            return None
        if sort:
            for field, direction in reversed(sort):
                matches.sort(
                    key=lambda item: item.get(field),
                    reverse=direction < 0,
                )
        selected = matches[0]
        self._apply_update(selected, update_doc)
        return deepcopy(selected)

    def count_documents(self, filter_doc):
        return sum(1 for doc in self.docs.values() if self._matches(doc, filter_doc))

    def _apply_update(self, doc, update_doc):
        for key, value in update_doc.get("$set", {}).items():
            doc[key] = value
        for key, value in update_doc.get("$inc", {}).items():
            doc[key] = int(doc.get(key, 0)) + value

    def _validate_update_doc(self, update_doc):
        overlap = set(update_doc.get("$setOnInsert", {})) & set(update_doc.get("$set", {}))
        if overlap:
            field = sorted(overlap)[0]
            raise ValueError(f"Cannot update '{field}' and '{field}' at the same time")

    def _matches(self, doc, filter_doc):
        if "$and" in filter_doc:
            return all(self._matches(doc, item) for item in filter_doc["$and"])
        if "$or" in filter_doc:
            return any(self._matches(doc, item) for item in filter_doc["$or"])
        for key, expected in filter_doc.items():
            actual = doc.get(key)
            if isinstance(expected, dict) and "$lte" in expected:
                limit = expected["$lte"]
                if actual is None or actual > limit:
                    return False
            elif isinstance(expected, dict) and "$lt" in expected:
                limit = expected["$lt"]
                if actual is None or actual >= limit:
                    return False
            elif isinstance(expected, dict) and "$nin" in expected:
                if actual in set(expected["$nin"]):
                    return False
            elif actual != expected:
                return False
        return True
