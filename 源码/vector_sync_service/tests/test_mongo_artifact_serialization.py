from app.config import settings
import pytest
from datetime import timedelta
from types import SimpleNamespace

from app.services.mongo_state_manager import (
    MongoPersistenceError,
    MongoStateManager,
    _json_default,
)
from app.models.sync_runtime import ArtifactType, ScopeType, SyncStage, utcnow
from app.services.retention_cleanup import RetentionCleanupService


class FakeRepeatedContainer:
    def __iter__(self):
        return iter([101, 102, 103])


def test_json_default_converts_iterable_container_to_list():
    value = {"ids": FakeRepeatedContainer()}

    normalized = _json_default(value)

    assert normalized == {"ids": [101, 102, 103]}


def test_mongo_state_manager_indexes_use_short_explicit_names(monkeypatch):
    monkeypatch.setattr(settings, "MONGO_STATUS_COLLECTION", "doc_sync_status")
    monkeypatch.setattr(settings, "MONGO_FAILURE_COLLECTION", "doc_sync_failures")
    monkeypatch.setattr(settings, "MONGO_ATTEMPT_COLLECTION", "doc_sync_attempts")
    monkeypatch.setattr(settings, "MONGO_ARTIFACT_COLLECTION", "doc_sync_artifacts")
    monkeypatch.setattr(settings, "MONGO_FAILURE_RETENTION_ENABLED", True)

    fake_db = FakeDatabase()
    manager = MongoStateManager()
    manager._client = object()
    manager._db = fake_db
    manager._connected = True

    manager._ensure_indexes_sync()

    created_names = {
        collection_name: [kwargs.get("name") for _, kwargs in collection.indexes]
        for collection_name, collection in fake_db.collections.items()
    }
    assert created_names == {
        "doc_sync_status": [
            "idx_status_updated",
            "idx_next_retry",
            "idx_core_status_updated",
            "idx_image_status_updated",
        ],
        "doc_sync_failures": [
            "ux_failure_key",
            "idx_failure_retry",
            "idx_failure_target_retry",
            "idx_failure_doc_time",
            "ttl_failure_resolved",
        ],
        "doc_sync_attempts": ["idx_attempt_doc_time_start", "ttl_attempt_finished"],
        "doc_sync_artifacts": ["idx_artifact_doc_type_stage", "ttl_artifact_expire"],
        "doc_sync_pipeline_versions": ["ux_pipeline_target_doc", "idx_pipeline_version_updated"],
    }
    assert all(name and len(name.encode("utf-8")) < 40 for names in created_names.values() for name in names)


def test_mongo_state_manager_exposes_selected_database():
    fake_db = FakeDatabase()
    manager = MongoStateManager()
    manager._client = object()
    manager._db = fake_db
    manager._connected = True

    assert manager.db is fake_db


def test_mongo_state_manager_db_requires_connection():
    manager = MongoStateManager()

    with pytest.raises(MongoPersistenceError, match="mongo state manager is not connected"):
        _ = manager.db


def test_backfill_selects_all_pending_windows_in_order(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_PRIORITY_WINDOW_COLLECTION", "image_priority_windows")

    now = utcnow()
    fake_db = FakeDatabase()
    manager = MongoStateManager()
    manager._client = object()
    manager._db = fake_db
    manager._connected = True

    windows = fake_db["image_priority_windows"].docs
    windows["doc-ready|t1"] = {
        "_id": "doc-ready|t1",
        "doc_id": "doc-ready",
        "record_update_time": "t1",
        "window_id": "231-test",
        "metadata_backfill_status": "PENDING",
        "image_ready_at": now,
        "image_task_count": 2,
        "updated_at": now,
    }
    windows["doc-half|t1"] = {
        "_id": "doc-half|t1",
        "doc_id": "doc-half",
        "record_update_time": "t1",
        "window_id": "231-test",
        "metadata_backfill_status": "PENDING",
        "image_ready_at": now,
        "image_task_count": 2,
        "updated_at": now,
    }
    windows["doc-none|t1"] = {
        "_id": "doc-none|t1",
        "doc_id": "doc-none",
        "record_update_time": "t1",
        "window_id": "231-test",
        "metadata_backfill_status": "PENDING",
        "image_ready_at": now,
        "image_task_count": 0,
        "updated_at": now,
    }
    windows["doc-other-window|t1"] = {
        "_id": "doc-other-window|t1",
        "doc_id": "doc-other-window",
        "record_update_time": "t1",
        "window_id": "other",
        "metadata_backfill_status": "PENDING",
        "image_ready_at": now,
        "image_task_count": 1,
        "updated_at": now,
    }

    rows = manager._find_image_success_windows_for_metadata_backfill_sync(
        "231-test",
        ["PENDING", "FAILED"],
        10,
        "documents|IVFPQ|COSINE|NLIST=4096|M=64|NBITS=8",
    )

    assert [row["_id"] for row in rows] == ["doc-half|t1", "doc-none|t1", "doc-ready|t1"]


def test_success_and_failure_artifacts_use_distinct_retention(monkeypatch):
    monkeypatch.setattr(settings, "MONGO_ARTIFACT_SUCCESS_RETENTION_DAYS", 7)
    monkeypatch.setattr(settings, "MONGO_ARTIFACT_FAILURE_RETENTION_DAYS", 180)

    fake_db = FakeDatabase()
    manager = MongoStateManager()
    manager._client = object()
    manager._db = fake_db

    success_id = manager._save_artifact_sync(
        "doc-1",
        "2026-06-17T00:00:00+08:00",
        "A79",
        SyncStage.IMAGE_DISCOVER,
        ScopeType.DOCUMENT,
        "doc-1",
        ArtifactType.IMAGE_MANIFEST_JSON,
        {"ok": True},
        True,
        "application/json",
        {},
        None,
    )
    failure_id = manager._save_artifact_sync(
        "doc-1",
        "2026-06-17T00:00:00+08:00",
        "A79",
        SyncStage.IMAGE_DISCOVER,
        ScopeType.DOCUMENT,
        "doc-1",
        ArtifactType.IMAGE_MANIFEST_JSON,
        {"ok": False},
        False,
        "application/json",
        {},
        None,
    )

    now = utcnow()
    success = fake_db.collections[settings.MONGO_ARTIFACT_COLLECTION].docs[success_id]
    failure = fake_db.collections[settings.MONGO_ARTIFACT_COLLECTION].docs[failure_id]
    assert timedelta(days=6, hours=23) < success["expire_at"] - now < timedelta(days=7, minutes=1)
    assert timedelta(days=179, hours=23) < failure["expire_at"] - now < timedelta(days=180, minutes=1)


def test_retention_cleanup_deletes_only_old_success_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MONGO_SUCCESS_TASK_RETENTION_DAYS", 7)
    monkeypatch.setattr(settings, "MONGO_ARTIFACT_FAILURE_RETENTION_DAYS", 180)

    fake_db = FakeDatabase()
    manager = MongoStateManager()
    manager._client = object()
    manager._db = fake_db
    now = utcnow()
    fulltext = fake_db[settings.MONGO_FULLTEXT_TASK_COLLECTION]
    image = fake_db[settings.MONGO_IMAGE_TASK_COLLECTION]
    fulltext.docs["success-old"] = {
        "_id": "success-old",
        "status": "SUCCESS",
        "finished_at": now - timedelta(days=8),
    }
    fulltext.docs["failure-old"] = {
        "_id": "failure-old",
        "status": "FAILURE",
        "finished_at": now - timedelta(days=8),
    }
    image.docs["success-new"] = {
        "_id": "success-new",
        "status": "SUCCESS",
        "finished_at": now - timedelta(days=1),
    }
    image.docs["success-old"] = {
        "_id": "success-old",
        "status": "SUCCESS",
        "finished_at": now - timedelta(days=8),
    }

    service = RetentionCleanupService(manager=manager, upload_dir=tmp_path)
    result = service.run_sync(mode="weekly", dry_run=False)

    assert result["mongo"]["success_fulltext_tasks"] == 1
    assert result["mongo"]["success_image_tasks"] == 1
    assert "success-old" not in fulltext.docs
    assert "failure-old" in fulltext.docs
    assert "success-new" in image.docs


def test_retention_cleanup_deletes_only_confirmed_success_attachment_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "IMAGE_LOCAL_SUCCESS_RETENTION_HOURS", 24)

    fake_db = FakeDatabase()
    manager = MongoStateManager()
    manager._client = object()
    manager._db = fake_db
    manager._connected = True

    attachments_root = tmp_path / "attachments" / "doc-1"
    success_dir = attachments_root / "task-success"
    leased_dir = attachments_root / "task-leased"
    failure_dir = attachments_root / "task-failure"
    missing_task_dir = attachments_root / "task-missing"
    recent_success_dir = attachments_root / "task-recent"

    for task_dir in (success_dir, leased_dir, failure_dir, missing_task_dir, recent_success_dir):
        task_dir.mkdir(parents=True)
        (task_dir / "image.png").write_bytes(b"img")
        (task_dir / "content.mmd").write_text("mmd", encoding="utf-8")

    success_old = "2026-06-16T00:00:00+00:00"
    success_recent = "2099-06-18T00:00:00+00:00"
    _write_manifest(success_dir / "manifest.json", "task-success", success_old)
    _write_manifest(leased_dir / "manifest.json", "task-leased", success_old)
    _write_manifest(failure_dir / "manifest.json", "task-failure", success_old)
    _write_manifest(missing_task_dir / "manifest.json", "task-missing", success_old)
    _write_manifest(recent_success_dir / "manifest.json", "task-recent", success_recent)

    image_collection = fake_db[settings.MONGO_IMAGE_TASK_COLLECTION]
    image_collection.docs["task-success"] = {
        "_id": "task-success",
        "status": "SUCCESS",
        "finished_at": success_old,
    }
    image_collection.docs["task-leased"] = {
        "_id": "task-leased",
        "status": "LEASED",
        "lease_expires_at": "2099-06-18T00:00:00+00:00",
    }
    image_collection.docs["task-failure"] = {
        "_id": "task-failure",
        "status": "FAILURE",
        "finished_at": success_old,
    }
    image_collection.docs["task-recent"] = {
        "_id": "task-recent",
        "status": "SUCCESS",
        "finished_at": success_recent,
    }

    service = RetentionCleanupService(manager=manager, upload_dir=tmp_path)
    result = service.run_sync(mode="daily", dry_run=False)

    assert result["files"]["attachments_scanned"] == 5
    assert result["files"]["attachments_confirmed_success"] == 1
    assert result["files"]["attachments_deleted"] == 1
    assert result["files"]["attachments_deleted_images"] == 1
    assert result["files"]["attachments_deleted_mmd"] == 1
    assert result["files"]["attachments_skip_status_LEASED"] == 1
    assert result["files"]["attachments_skip_status_FAILURE"] == 1
    assert result["files"]["attachments_skip_task_not_found"] == 1
    assert result["files"]["attachments_skip_within_retention_window"] == 1
    assert not success_dir.exists()
    assert leased_dir.exists()
    assert failure_dir.exists()
    assert missing_task_dir.exists()
    assert recent_success_dir.exists()


class FakeCollection:
    def __init__(self) -> None:
        self.indexes = []
        self.docs = {}

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def insert_one(self, document):
        inserted_id = str(len(self.docs) + 1)
        self.docs[inserted_id] = {**document, "_id": inserted_id}
        return SimpleNamespace(inserted_id=inserted_id)

    def count_documents(self, filter_doc):
        return sum(1 for doc in self.docs.values() if _matches(doc, filter_doc or {}))

    def delete_many(self, filter_doc):
        keys = [key for key, doc in self.docs.items() if _matches(doc, filter_doc or {})]
        for key in keys:
            self.docs.pop(key, None)
        return SimpleNamespace(deleted_count=len(keys))

    def find_one(self, filter_doc):
        for doc in self.docs.values():
            if _matches(doc, filter_doc or {}):
                return dict(doc)
        return None

    def find(self, filter_doc, projection=None):
        rows = []
        for doc in self.docs.values():
            if not _matches(doc, filter_doc or {}):
                continue
            if projection:
                projected = {key: doc.get(key) for key, enabled in projection.items() if enabled}
                if "_id" not in projection:
                    projected["_id"] = doc.get("_id")
                rows.append(projected)
            else:
                rows.append(dict(doc))
        return FakeCursor(rows)


class FakeCursor(list):
    def sort(self, sort_spec):
        for key, direction in reversed(sort_spec):
            reverse = direction < 0
            super().sort(key=lambda row: row.get(key), reverse=reverse)
        return self

    def limit(self, count):
        return FakeCursor(self[:count])


class BackfillImageTaskCollection(FakeCollection):
    def __init__(self, database) -> None:
        super().__init__()
        self.database = database

    def aggregate(self, pipeline):
        match_success = pipeline[0]["$match"]
        success_before = (match_success.get("updated_at") or {}).get("$lte")
        window_match = pipeline[2]["$lookup"]["pipeline"][1]["$match"]
        allowed_statuses = window_match["backfill_status"]["$in"]
        window_id = window_match.get("window_id")
        limit = next((stage["$limit"] for stage in pipeline if "$limit" in stage), 0)

        grouped = {}
        for task in self.docs.values():
            if task.get("status") != "SUCCESS":
                continue
            if not task.get("doc_id") or not task.get("record_update_time"):
                continue
            if success_before is not None and task.get("updated_at") > success_before:
                continue
            key = (task["doc_id"], task["record_update_time"])
            item = grouped.setdefault(
                key,
                {
                    "success_image_count": 0,
                    "first_image_success_at": task["updated_at"],
                    "last_image_success_at": task["updated_at"],
                },
            )
            item["success_image_count"] += 1
            item["first_image_success_at"] = min(item["first_image_success_at"], task["updated_at"])
            item["last_image_success_at"] = max(item["last_image_success_at"], task["updated_at"])

        rows = []
        windows = self.database["image_priority_windows"].docs.values()
        for (doc_id, record_update_time), item in grouped.items():
            matches = [
                window
                for window in windows
                if window.get("doc_id") == doc_id
                and window.get("record_update_time") == record_update_time
                and window.get("backfill_status") in allowed_statuses
                and (not window_id or window.get("window_id") == window_id)
                and int(window.get("image_task_count") or 0) > 0
                and item["success_image_count"] >= int(window.get("image_task_count") or 0)
            ]
            for window in matches:
                rows.append(
                    {
                        **window,
                        "image_success_count": item["success_image_count"],
                        "first_image_success_at": item["first_image_success_at"],
                        "last_image_success_at": item["last_image_success_at"],
                    }
                )
        rows.sort(key=lambda row: (row["first_image_success_at"], row["_id"]))
        if limit:
            rows = rows[:limit]
        return rows


class FakeDatabase:
    def __init__(self) -> None:
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = FakeCollection()
        return self.collections[name]


def _matches(doc, filter_doc):
    for key, expected in filter_doc.items():
        if key == "$or":
            if not any(_matches(doc, branch) for branch in expected):
                return False
            continue
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$lte" in expected:
                if actual is None or actual > expected["$lte"]:
                    return False
            elif "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif "$gt" in expected:
                if actual is None or actual <= expected["$gt"]:
                    return False
            elif "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
            else:
                raise AssertionError(f"unsupported fake filter operator: {expected}")
        elif actual != expected:
            return False
    return True


def _write_manifest(path, task_id, created_at):
    path.write_text(
        (
            '{\n'
            f'  "task_id": "{task_id}",\n'
            f'  "created_at": "{created_at}"\n'
            '}\n'
        ),
        encoding="utf-8",
    )
