from __future__ import annotations

from datetime import datetime, timezone

from app.config import settings
from app.services.mongo_ops_backup import (
    BACKUP_COLLECTIONS,
    BACKUP_META_COLLECTION,
    build_backup_inspection,
    run_backup_sync,
)


def test_failure_ttl_index_is_dropped_when_retention_disabled(monkeypatch):
    monkeypatch.setattr(settings, "MONGO_FAILURE_RETENTION_ENABLED", False)

    fake_db = FakeDatabase()
    manager = _build_manager(fake_db)

    manager._ensure_indexes_sync()

    failure_collection = fake_db.collections[settings.MONGO_FAILURE_COLLECTION]
    ttl_indexes = [
        kwargs
        for _keys, kwargs in failure_collection.indexes
        if kwargs.get("name") == "ttl_failure_resolved"
    ]
    assert ttl_indexes == []
    assert failure_collection.dropped_indexes == ["ttl_failure_resolved"]


def test_run_backup_sync_copies_expected_collections_and_overwrites_targets(monkeypatch):
    monkeypatch.setattr(settings, "MONGO_BACKUP_DB", "backup_latest")
    monkeypatch.setattr(settings, "MONGO_DB", "source_db")

    source_db = FakeDatabase()
    backup_db = FakeDatabase()
    client = FakeMongoClient({"source_db": source_db, "backup_latest": backup_db})
    manager = _build_manager(source_db, client=client)

    for key, collection_name in BACKUP_COLLECTIONS.items():
        source_db[collection_name].docs = {
            f"{key}-1": {"_id": f"{key}-1", "kind": key},
            f"{key}-2": {"_id": f"{key}-2", "kind": key},
        }
        backup_db[collection_name].docs = {"stale": {"_id": "stale"}}

    result = run_backup_sync(manager)

    assert result["backup_db"] == "backup_latest"
    assert result["source_db"] == "source_db"
    assert all(count == 2 for count in result["copied_counts"].values())
    assert len(client.rename_calls) == len(BACKUP_COLLECTIONS)
    meta = backup_db[BACKUP_META_COLLECTION].find_one({"_id": "latest"})
    assert meta is not None
    assert meta["source_db"] == "source_db"
    assert meta["backup_db"] == "backup_latest"
    for key, collection_name in BACKUP_COLLECTIONS.items():
        assert set(backup_db[collection_name].docs) == {f"{key}-1", f"{key}-2"}


def test_build_backup_inspection_reports_source_and_backup_counts(monkeypatch):
    monkeypatch.setattr(settings, "MONGO_BACKUP_DB", "backup_latest")

    source_db = FakeDatabase()
    backup_db = FakeDatabase()
    client = FakeMongoClient({"source": source_db, "backup_latest": backup_db})
    manager = _build_manager(source_db, client=client)

    for key, collection_name in BACKUP_COLLECTIONS.items():
        source_db[collection_name].docs = {f"{key}-1": {"_id": f"{key}-1"}}
        backup_db[collection_name].docs = {
            f"{key}-1": {"_id": f"{key}-1"},
            f"{key}-2": {"_id": f"{key}-2"},
        }
    backup_db[BACKUP_META_COLLECTION].docs = {
        "latest": {
            "_id": "latest",
            "checked_at": datetime(2026, 6, 18, tzinfo=timezone.utc),
            "copied_counts": {key: 1 for key in BACKUP_COLLECTIONS},
        }
    }

    inspection = build_backup_inspection(manager)

    assert inspection["backup_db"] == "backup_latest"
    assert inspection["meta"]["_id"] == "latest"
    assert inspection["meta"]["checked_at"] == "2026-06-18T00:00:00+00:00"
    assert all(value == 1 for value in inspection["source_counts"].values())
    assert all(value == 2 for value in inspection["backup_counts"].values())


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.docs: dict[str, dict] = {}
        self.indexes: list[tuple[list[tuple[str, int]], dict]] = []
        self.dropped_indexes: list[str] = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((list(keys), dict(kwargs)))

    def drop_index(self, name: str):
        self.dropped_indexes.append(name)

    def drop(self):
        self.docs = {}

    def find(self, _filter):
        return [dict(value) for value in self.docs.values()]

    def find_one(self, filter_doc):
        target_id = filter_doc.get("_id")
        if target_id is None:
            return None
        value = self.docs.get(target_id)
        return None if value is None else dict(value)

    def insert_many(self, documents, ordered=False):
        for document in documents:
            self.docs[str(document["_id"])] = dict(document)

    def replace_one(self, filter_doc, document, upsert=False):
        target_id = filter_doc.get("_id") or document.get("_id")
        if target_id is None and not upsert:
            return
        self.docs[str(target_id)] = dict(document)

    def count_documents(self, _filter):
        return len(self.docs)


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]


class FakeAdmin:
    def __init__(self, client: "FakeMongoClient") -> None:
        self._client = client

    def command(self, _command_name: str, source_name: str, *, to: str, dropTarget: bool):
        assert dropTarget is True
        self._client.rename_calls.append({"source": source_name, "target": to})
        source_db_name, source_collection_name = source_name.split(".", 1)
        target_db_name, target_collection_name = to.split(".", 1)
        source_db = self._client[source_db_name]
        target_db = self._client[target_db_name]
        target_db[target_collection_name].docs = dict(source_db[source_collection_name].docs)
        source_db[source_collection_name].docs = {}


class FakeMongoClient:
    def __init__(self, databases: dict[str, FakeDatabase]) -> None:
        self.databases = dict(databases)
        self.rename_calls: list[dict[str, str]] = []
        self.admin = FakeAdmin(self)

    def __getitem__(self, name: str) -> FakeDatabase:
        if name not in self.databases:
            self.databases[name] = FakeDatabase()
        return self.databases[name]


def _build_manager(db: FakeDatabase, *, client: FakeMongoClient | None = None):
    from app.services.mongo_state_manager import MongoStateManager

    manager = MongoStateManager()
    manager._db = db
    manager._client = client or FakeMongoClient({settings.MONGO_DB: db})
    manager._connected = True
    return manager
