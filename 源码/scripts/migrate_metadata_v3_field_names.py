"""Preserve the first v3 collection and resume it with corrected field names.

This migration never drops a collection or resets the SQLite checkpoint. It is
safe to rerun after an interruption: target upserts are keyed by the same IDs.
The v3 worker and Kafka snapshotter must be stopped while this runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from backfill_metadata_v3 import (
    DB_PATH,
    MERGED_VECTOR,
    NULL_FLAG,
    OLD_VECTORS,
    SOURCE,
    SOURCE_ID,
    STATE,
    TARGET,
    connect,
    ensure_target,
    validate_source,
)

LEGACY = "scientific_documents_v3_legacy_20261026"
OLD_VECTOR = "merged_embedding"
OLD_FLAG = "merged_embedding_is_null"
OLD_SCHEMA_VERSION = "metadata_v3_41"
NEW_SCHEMA_VERSION = "metadata_v3_41_text_embedding"
BATCH_SIZE = 128


def strong_count(client, collection: str) -> int:
    return int(client.query(
        collection, filter="", output_fields=["count(*)"],
        consistency_level="Strong", timeout=180,
    )[0]["count(*)"])


def field_names(client, collection: str) -> set[str]:
    return {field["name"] for field in client.describe_collection(collection)["fields"]}


def main() -> None:
    client = connect()
    validate_source(client)
    old_names = {OLD_VECTOR, OLD_FLAG}
    new_names = {MERGED_VECTOR, NULL_FLAG}
    if not client.has_collection(LEGACY):
        if not client.has_collection(TARGET):
            raise RuntimeError("neither original nor legacy v3 collection exists")
        names = field_names(client, TARGET)
        if not old_names <= names or names & new_names:
            raise RuntimeError("original v3 does not have the expected old fields")
        old_count = strong_count(client, TARGET)
        print(json.dumps({"event": "legacy_rename_start", "count": old_count}), flush=True)
        client.rename_collection(TARGET, LEGACY)
    else:
        if client.has_collection(TARGET) and old_names & field_names(client, TARGET):
            raise RuntimeError("old-named target and legacy collection both exist")

    legacy = client.describe_collection(LEGACY)
    if len(legacy["fields"]) != 41 or not old_names <= field_names(client, LEGACY):
        raise RuntimeError("legacy v3 schema changed")
    legacy_count = strong_count(client, LEGACY)
    print(json.dumps({"event": "legacy_preserved", "collection": LEGACY,
                      "collection_id": str(legacy["collection_id"]),
                      "strong_count": legacy_count}), flush=True)

    ensure_target(client)
    expected = set(validate_source(client)) | new_names | {"zijian_version"}
    if field_names(client, TARGET) != expected:
        raise RuntimeError("new v3 schema is not the corrected 41 fields")
    if any(name in expected for name in OLD_VECTORS | old_names):
        raise RuntimeError("new v3 unexpectedly includes an old vector or flag")

    if client.get_load_state(LEGACY).get("state").name != "Loaded":
        client.load_collection(LEGACY, timeout=600)
    fields = [field["name"] for field in legacy["fields"]]
    iterator = client.query_iterator(
        collection_name=LEGACY, filter="", output_fields=fields,
        batch_size=BATCH_SIZE, consistency_level="Strong", timeout=180,
    )
    copied = 0
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            for row in rows:
                row[MERGED_VECTOR] = row.pop(OLD_VECTOR)
                row[NULL_FLAG] = row.pop(OLD_FLAG)
            client.upsert(TARGET, data=rows, timeout=180)
            copied += len(rows)
            if copied % 4096 < len(rows) or copied == legacy_count:
                print(json.dumps({"event": "copy_progress", "copied": copied,
                                  "expected": legacy_count}), flush=True)
    finally:
        iterator.close()
    if copied != legacy_count:
        raise RuntimeError(f"legacy iterator copied {copied}, expected {legacy_count}")

    client.flush(TARGET)
    target_count = strong_count(client, TARGET)
    if target_count != legacy_count:
        raise RuntimeError(f"new v3 count {target_count}, legacy count {legacy_count}")

    with sqlite3.connect(DB_PATH, timeout=60) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        if meta.get("source_id") != SOURCE_ID or meta.get("target") != TARGET:
            raise RuntimeError("checkpoint collection identity mismatch")
        if meta.get("schema_version") not in {OLD_SCHEMA_VERSION, NEW_SCHEMA_VERSION}:
            raise RuntimeError("checkpoint schema version mismatch")
        done = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
        if target_count < done:
            raise RuntimeError(f"new v3 count {target_count} is below checkpoint done {done}")
        db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (NEW_SCHEMA_VERSION,))

    try:
        client.release_collection(LEGACY)
    except Exception as exc:
        print(json.dumps({"event": "legacy_release_skipped",
                          "error_type": type(exc).__name__}), flush=True)
    report = {
        "event": "migration_complete",
        "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "legacy_collection": LEGACY,
        "target_collection": TARGET,
        "copied": copied,
        "strong_target_count": target_count,
        "checkpoint_done": done,
        "vector_field": MERGED_VECTOR,
        "null_flag": NULL_FLAG,
        "field_count": len(expected),
        "index_count": len(client.list_indexes(TARGET)),
    }
    (STATE / "field_name_migration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
