from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

from app.models.sync_runtime import FailureStatus, utcnow
from app.services.mongo_state_manager import MongoStateManager


def test_held_history_failure_is_not_claimed_for_retry(monkeypatch):
    collection = FakeFailureCollection()
    now = utcnow()
    collection.docs["held"] = {
        "_id": "held",
        "failure_key": "held",
        "status": FailureStatus.HELD_HISTORY.value,
        "next_retry_at": now - timedelta(seconds=10),
        "updated_at": now - timedelta(seconds=10),
    }
    collection.docs["pending"] = {
        "_id": "pending",
        "failure_key": "pending",
        "status": FailureStatus.PENDING.value,
        "next_retry_at": now - timedelta(seconds=10),
        "updated_at": now,
    }
    manager = MongoStateManager()
    monkeypatch.setattr(manager, "_require_sync", lambda: None)
    monkeypatch.setattr(
        MongoStateManager,
        "failures_collection",
        property(lambda _self: collection),
    )

    claimed = manager._claim_failure_for_retry_sync()

    assert claimed["failure_key"] == "pending"
    assert collection.docs["held"]["status"] == FailureStatus.HELD_HISTORY.value
    assert collection.docs["pending"]["status"] == FailureStatus.RETRYING.value


def test_targetless_legacy_failure_is_held_before_targeted_retry(monkeypatch):
    collection = FakeFailureCollection()
    now = utcnow()
    collection.docs["legacy"] = {
        "_id": "legacy",
        "failure_key": "legacy",
        "status": FailureStatus.PENDING.value,
        "next_retry_at": now - timedelta(seconds=10),
        "updated_at": now - timedelta(seconds=10),
    }
    collection.docs["bound"] = {
        "_id": "bound",
        "failure_key": "bound",
        "status": FailureStatus.PENDING.value,
        "next_retry_at": now - timedelta(seconds=10),
        "updated_at": now,
        "pipeline": "metadata",
        "resolved_collection": "documents_v2",
        "target_signature": "signature-v2",
    }
    manager = MongoStateManager()
    monkeypatch.setattr(manager, "_require_sync", lambda: None)
    monkeypatch.setattr(manager, "_refresh_pending_count_sync", lambda: None)
    monkeypatch.setattr(
        MongoStateManager,
        "failures_collection",
        property(lambda _self: collection),
    )

    claimed = manager._claim_failure_for_retry_filtered_sync(
        {"metadata"},
        {"metadata": ("documents_v2", "signature-v2")},
    )

    assert claimed["failure_key"] == "bound"
    assert collection.docs["legacy"]["status"] == FailureStatus.HELD_HISTORY.value
    assert collection.docs["legacy"]["resolution_note"] == (
        "skipped: legacy failure has no pipeline target binding"
    )


class FakeFailureCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def find_one_and_update(self, filter_doc, update_doc, sort=None, return_document=None):
        matches = [doc for doc in self.docs.values() if _matches(doc, filter_doc)]
        if not matches:
            return None
        if sort:
            for field, direction in reversed(sort):
                matches.sort(key=lambda item: item.get(field), reverse=direction < 0)
        selected = matches[0]
        for key, value in update_doc.get("$set", {}).items():
            selected[key] = value
        return deepcopy(selected)

    def update_many(self, filter_doc, update_doc):
        matched_count = 0
        for document in self.docs.values():
            if not _matches(document, filter_doc):
                continue
            matched_count += 1
            for key, value in update_doc.get("$set", {}).items():
                document[key] = value
        return SimpleNamespace(matched_count=matched_count)


def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    if "$and" in query:
        return all(_matches(doc, item) for item in query["$and"])
    if "$or" in query:
        return any(_matches(doc, item) for item in query["$or"])
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict) and "$lte" in expected:
            limit = expected["$lte"]
            if actual is None or actual > limit:
                return False
        elif isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif isinstance(expected, dict) and "$exists" in expected:
            if bool(key in doc) != bool(expected["$exists"]):
                return False
        elif actual != expected:
            return False
    return True
