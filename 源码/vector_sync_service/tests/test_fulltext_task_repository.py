from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.sync_runtime import utcnow
from app.services.fulltext_task_manager import FullTextTask, FullTextTaskStatus
from app.services.fulltext_task_repository import FullTextTaskRepository


@pytest.mark.asyncio
async def test_successful_task_is_deleted_and_republished_task_starts_pending():
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.ensure_indexes()

    result = await repository.upsert_tasks([_task("task-1")])

    assert result["upserted"] == 1
    assert collection.docs["task-1"]["status"] == FullTextTaskStatus.PENDING.value
    assert collection.indexes

    await repository.mark_task_success(
        "task-1",
        artifact_refs=["artifact-1"],
        chunk_count=2,
    )
    assert "task-1" not in collection.docs
    await repository.upsert_tasks([_task("task-1")])

    assert collection.docs["task-1"]["status"] == FullTextTaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_pending_task_can_be_claimed():
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    claimed = await repository.claim_next_task(
        worker_id="worker-1",
        lease_seconds=120,
    )

    assert claimed is not None
    assert claimed["task_id"] == "task-1"
    assert claimed["status"] == FullTextTaskStatus.LEASED.value
    assert claimed["lease_owner"] == "worker-1"
    assert claimed["attempt_count"] == 1


@pytest.mark.asyncio
async def test_upsert_tasks_persists_doc_metadata_for_worker_claim():
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.upsert_tasks(
        [
            _task(
                "task-1",
                title=["Paper title"],
                keyword=["vector"],
                accessType="open",
                classificationCode="TP391",
                year="2026",
                hasAuthorname=["Alice"],
                hasCreateOrganizationname=["NSTL"],
                language=["en"],
            )
        ]
    )

    stored = collection.docs["task-1"]
    assert stored["title"] == ["Paper title"]
    assert stored["keyword"] == ["vector"]
    assert stored["accessType"] == "open"
    assert stored["classificationCode"] == "TP391"
    assert stored["year"] == "2026"
    assert stored["hasAuthorname"] == ["Alice"]
    assert stored["hasCreateOrganizationname"] == ["NSTL"]
    assert stored["language"] == ["en"]

    claimed = await repository.claim_next_task(worker_id="worker-1", lease_seconds=120)
    assert claimed["title"] == ["Paper title"]
    assert claimed["keyword"] == ["vector"]
    assert claimed["language"] == ["en"]


@pytest.mark.asyncio
async def test_unexpired_lease_is_not_claimed_but_expired_lease_is_reclaimed():
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])

    future = utcnow() + timedelta(seconds=60)
    collection.docs["task-1"].update(
        {
            "status": FullTextTaskStatus.LEASED.value,
            "lease_owner": "worker-old",
            "lease_expires_at": future,
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
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])
    await repository.mark_task_failure(
        "task-1",
        error_code="FULLTEXT_DOWNLOAD_RETRYABLE",
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
    assert claimed["status"] == FullTextTaskStatus.LEASED.value


@pytest.mark.asyncio
async def test_claim_next_task_can_skip_tasks_created_before_configured_cutoff(monkeypatch):
    cutoff = datetime(2026, 6, 14, 2, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.services.fulltext_task_repository.settings.FULLTEXT_WORKER_CLAIM_CREATED_AFTER",
        cutoff.isoformat(),
    )
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    old_task = _task("task-old", created_at=cutoff - timedelta(seconds=1))
    new_task = _task("task-new", created_at=cutoff + timedelta(seconds=1))
    await repository.upsert_tasks([old_task, new_task])

    claimed = await repository.claim_next_task(
        worker_id="worker-1",
        lease_seconds=120,
    )

    assert claimed is not None
    assert claimed["task_id"] == "task-new"
    assert collection.docs["task-old"]["status"] == FullTextTaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_held_history_is_counted_but_not_claimed():
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1")])
    collection.docs["task-1"]["status"] = FullTextTaskStatus.HELD_HISTORY.value
    collection.docs["task-1"]["claim_rank"] = 9

    summary = await repository.get_summary()
    claimed = await repository.claim_next_task(
        worker_id="worker-1",
        lease_seconds=120,
    )

    assert summary["counts"][FullTextTaskStatus.HELD_HISTORY.value] == 1
    assert claimed is None


@pytest.mark.asyncio
async def test_success_task_is_deleted_while_skipped_task_remains_recorded():
    collection = FakeCollection()
    repository = FullTextTaskRepository(collection=collection)
    await repository.upsert_tasks([_task("task-1"), _task("task-2")])

    await repository.mark_task_success(
        "task-1",
        artifact_refs=["raw", "chunks"],
        chunk_count=3,
    )
    await repository.mark_task_skipped("task-2", reason="SKIPPED_NOT_PRESENT")

    task = await repository.get_task("task-1")
    summary = await repository.get_summary()

    assert task is None
    assert collection.docs["task-2"]["skip_reason"] == "SKIPPED_NOT_PRESENT"
    assert summary["collection"] == "doc_fulltext_tasks"
    assert summary["total"] == 1
    assert summary["counts"][FullTextTaskStatus.SUCCESS.value] == 0
    assert summary["counts"][FullTextTaskStatus.SKIPPED.value] == 1


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


def _task(task_id: str, **overrides: Any) -> FullTextTask:
    defaults = {
        "task_id": task_id,
        "doc_id": "doc-1",
        "record_update_time": "2026-06-07T00:00:00+00:00",
        "relative_url": f"/fulltext/{task_id}.mmd",
    }
    defaults.update(overrides)
    return FullTextTask(
        **defaults,
    )


class FakeCollection:
    name = "doc_fulltext_tasks"

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
            if isinstance(expected, dict):
                if "$lte" in expected:
                    limit = expected["$lte"]
                    if actual is None or actual > limit:
                        return False
                elif "$gt" in expected:
                    limit = expected["$gt"]
                    if actual is None or actual <= limit:
                        return False
                elif "$nin" in expected:
                    if actual in expected["$nin"]:
                        return False
                else:
                    raise AssertionError(f"unsupported fake filter operator: {expected}")
            elif actual != expected:
                return False
        return True
