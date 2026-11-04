"""Mongo backup helpers for operational snapshots."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from app.config import settings


BACKUP_META_COLLECTION = "_ops_backup_meta"
BACKUP_COLLECTIONS: dict[str, str] = {
    "status": settings.MONGO_STATUS_COLLECTION,
    "failures": settings.MONGO_FAILURE_COLLECTION,
    "artifacts": settings.MONGO_ARTIFACT_COLLECTION,
    "fulltext_tasks": settings.MONGO_FULLTEXT_TASK_COLLECTION,
    "image_tasks": settings.MONGO_IMAGE_TASK_COLLECTION,
}


async def run_backup(manager) -> dict[str, Any]:
    return await asyncio.to_thread(run_backup_sync, manager)


def run_backup_sync(manager) -> dict[str, Any]:
    backup_db_name = str(settings.MONGO_BACKUP_DB or "").strip()
    if not backup_db_name:
        raise ValueError("MONGO_BACKUP_DB is required")
    manager._require_sync()
    backup_db = manager._client[backup_db_name]
    copied_counts: dict[str, int] = {}
    checked_at = _utcnow()

    for key, source_name in BACKUP_COLLECTIONS.items():
        target_name = source_name
        temp_name = f"{target_name}__tmp"
        target_collection = backup_db[temp_name]
        target_collection.drop()
        copied_counts[key] = _copy_collection(manager.db[source_name], target_collection)
        _rename_with_drop(manager._client, backup_db_name, temp_name, target_name)

    meta_collection = backup_db[BACKUP_META_COLLECTION]
    meta_collection.replace_one(
        {"_id": "latest"},
        {
            "_id": "latest",
            "checked_at": checked_at,
            "source_db": settings.MONGO_DB,
            "backup_db": backup_db_name,
            "copied_counts": copied_counts,
        },
        upsert=True,
    )
    return {
        "checked_at": checked_at.isoformat(),
        "source_db": settings.MONGO_DB,
        "backup_db": backup_db_name,
        "copied_counts": copied_counts,
    }


def build_backup_inspection(manager) -> dict[str, Any]:
    backup_db_name = str(settings.MONGO_BACKUP_DB or "").strip()
    manager._require_sync()
    backup_db = manager._client[backup_db_name] if backup_db_name else None
    meta = None
    if backup_db is not None:
        meta = backup_db[BACKUP_META_COLLECTION].find_one({"_id": "latest"})
    backup_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for key, collection_name in BACKUP_COLLECTIONS.items():
        source_counts[key] = int(manager.db[collection_name].count_documents({}))
        if backup_db is not None:
            backup_counts[key] = int(backup_db[collection_name].count_documents({}))
    return {
        "backup_db": backup_db_name,
        "meta": _serialize_meta(meta),
        "source_counts": source_counts,
        "backup_counts": backup_counts,
    }


def _copy_collection(source: Collection, target: Collection, *, batch_size: int = 1000) -> int:
    copied = 0
    batch: list[dict[str, Any]] = []
    for document in source.find({}):
        batch.append(document)
        if len(batch) >= batch_size:
            target.insert_many(batch, ordered=False)
            copied += len(batch)
            batch = []
    if batch:
        target.insert_many(batch, ordered=False)
        copied += len(batch)
    return copied


def _rename_with_drop(client, db_name: str, source_name: str, target_name: str) -> None:
    client.admin.command(
        "renameCollection",
        f"{db_name}.{source_name}",
        to=f"{db_name}.{target_name}",
        dropTarget=True,
    )


def _serialize_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    payload = dict(meta)
    payload["_id"] = str(payload.get("_id"))
    checked_at = payload.get("checked_at")
    if isinstance(checked_at, datetime):
        payload["checked_at"] = checked_at.astimezone(timezone.utc).isoformat()
    return payload


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
