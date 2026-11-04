"""Compare exact sequential writes with grouped batch writes in local Milvus only.

The script creates an isolated database/collection and never deletes or drops data.
Both measured passes update the same 1,000 seeded records; reset is outside timings.
"""
from __future__ import annotations

import collections
import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from incremental_sync.adapters import MilvusTarget, _equal  # noqa: E402
from incremental_sync.contract import (  # noqa: E402
    FIELDS, VECTOR_FIELDS, new_record, update_payload, validate_value,
)

OUT = ROOT / "out/metadata_incremental_perf_20261020"
DATASET = OUT / "dataset.jsonl"
URI = "http://127.0.0.1:19539"
DATABASE = "metadata_incremental_perf_20261020"
COLLECTION_PREFIX = "scientific_documents_perf_"
ADDED_FIELDS = tuple(list(FIELDS)[24:])
BATCH_SIZE = 50
assert len(FIELDS) == 41 and len(ADDED_FIELDS) == 17


def dataset() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1000 or len({item["id"] for item in rows}) != 1000:
        raise RuntimeError("fixed dataset must contain exactly 1,000 distinct documents")
    for item in rows:
        for field, vector in item["vectors"].items():
            validate_value(field, vector)
    return rows


def groups(rows: list[Any], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def create_local_collection() -> tuple[MilvusClient, str]:
    if URI != "http://127.0.0.1:19539":
        raise RuntimeError("benchmark must use the isolated loopback Milvus")
    admin = MilvusClient(uri=URI, timeout=90)
    try:
        if DATABASE not in admin.list_databases():
            admin.create_database(db_name=DATABASE)
    finally:
        admin.close()
    client = MilvusClient(uri=URI, db_name=DATABASE, timeout=90)
    collection = COLLECTION_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)
    if client.has_collection(collection):
        raise RuntimeError("new benchmark collection already exists")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, spec in FIELDS.items():
        if name == "id":
            schema.add_field(name, DataType.VARCHAR, max_length=spec.max_length, is_primary=True)
        elif spec.kind == "varchar":
            schema.add_field(name, DataType.VARCHAR, max_length=spec.max_length, nullable=True)
        elif spec.kind == "array":
            schema.add_field(name, DataType.ARRAY, element_type=DataType.VARCHAR,
                             max_capacity=spec.max_capacity, max_length=spec.max_length, nullable=True)
        elif spec.kind == "vector":
            schema.add_field(name, DataType.FLOAT_VECTOR, dim=spec.dim)
    indexes = client.prepare_index_params()
    for name in VECTOR_FIELDS:
        indexes.add_index(field_name=name, index_type="FLAT", metric_type="COSINE")
    for name in ("id", "year", "sysSubType", "sysDataStatus", "sysDataUpdateTime",
                 "sysSourceId", "sync_status", "accessType", "classificationCode",
                 "doi", "hasAuthorname", "hasCreateOrganizationname",
                 "hasSourcesysSourceId", "hasSourceidSet", "language"):
        indexes.add_index(field_name=name, index_type="AUTOINDEX")
    client.create_collection(collection_name=collection, schema=schema, index_params=indexes, timeout=180)
    client.load_collection(collection, timeout=180)
    return client, collection


def seed_record(item: dict[str, Any]) -> dict[str, Any]:
    row = new_record(item["id"], item["values"], item["vectors"])
    for name in ADDED_FIELDS:
        row[name] = None
    current = row["title_embed"]
    row["title_embed"] = [current[0] + 0.01, *current[1:]]
    return row


def batch_get(client: MilvusClient, collection: str, ids: list[str]) -> dict[str, dict[str, Any]]:
    result = client.get(collection_name=collection, ids=ids, output_fields=list(FIELDS),
                        consistency_level="Strong", timeout=90)
    rows = {}
    for raw in result:
        row = dict(raw)
        for name, spec in FIELDS.items():
            if spec.kind == "array" and row.get(name) is not None:
                row[name] = list(row[name])
        rows[row["id"]] = row
    if set(rows) != set(ids):
        raise RuntimeError("Strong batch read did not return the exact requested IDs")
    return rows


def verify_pass(client: MilvusClient, collection: str, rows: list[dict[str, Any]]) -> None:
    for group in groups(rows, BATCH_SIZE):
        current = batch_get(client, collection, [item["id"] for item in group])
        for item in group:
            # A missing Solr text field must preserve its seeded historical vector.
            expected = seed_record(item)
            expected.update(item["values"])
            expected.update(item["vectors"])
            actual = current[item["id"]]
            mismatch = [name for name in FIELDS if not _equal(actual.get(name), expected.get(name))]
            if mismatch:
                raise RuntimeError("final audit found mismatched fields: " + ",".join(mismatch))


def sequential_pass(rows: list[dict[str, Any]], collection: str) -> dict[str, Any]:
    target = MilvusTarget(URI, DATABASE, collection, "", 90)
    started = time.monotonic()
    outcomes = collections.Counter()
    try:
        for position, item in enumerate(rows, 1):
            before = target.get(item["id"])
            if before is None:
                raise RuntimeError("seeded record is missing before sequential update")
            payload = update_payload(item["id"], item["values"], item["vectors"], before)
            if len(payload) == 1:
                outcomes["unchanged"] += 1
            else:
                target.write(item["id"], before, payload, new=False)
                outcomes["updated"] += 1
            if position % 100 == 0:
                print(json.dumps({"phase": "sequential", "completed": position}), flush=True)
    finally:
        target.close()
    elapsed = time.monotonic() - started
    return {"seconds": round(elapsed, 3), "rows_per_second": round(len(rows) / elapsed, 3),
            "outcomes": dict(outcomes), "read_requests": len(rows) + outcomes["updated"],
            "upsert_requests": outcomes["updated"]}


def reset_seed(client: MilvusClient, collection: str, rows: list[dict[str, Any]]) -> None:
    for group in groups(rows, BATCH_SIZE):
        reset_rows = []
        for item in group:
            seed = seed_record(item)
            reset_rows.append({"id": item["id"], "title_embed": seed["title_embed"],
                               **{name: None for name in ADDED_FIELDS}})
        client.upsert(collection_name=collection, data=reset_rows, partial_update=True, timeout=90)
    client.flush(collection, timeout=180)
    for group in groups(rows, BATCH_SIZE):
        current = batch_get(client, collection, [item["id"] for item in group])
        for item in group:
            seed = seed_record(item)
            if any(not _equal(current[item["id"]].get(name), seed.get(name)) for name in FIELDS):
                raise RuntimeError("reset did not restore the fixed 1,000-row seed")


def grouped_batch_pass(client: MilvusClient, collection: str,
                       rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    timings = collections.Counter()
    masks = collections.Counter()
    read_requests = upsert_requests = verified = unchanged = 0
    for position, group in enumerate(groups(rows, BATCH_SIZE), 1):
        ids = [item["id"] for item in group]
        tick = time.monotonic()
        before = batch_get(client, collection, ids)
        timings["before_read"] += time.monotonic() - tick
        read_requests += 1
        payloads: dict[str, dict[str, Any]] = {}
        by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
        tick = time.monotonic()
        for item in group:
            payload = update_payload(item["id"], item["values"], item["vectors"], before[item["id"]])
            if len(payload) == 1:
                unchanged += 1
                payloads[item["id"]] = payload
                continue
            payloads[item["id"]] = payload
            mask = tuple(sorted(payload))
            by_mask[mask].append(payload)
            masks[mask] += 1
        timings["convert"] += time.monotonic() - tick
        tick = time.monotonic()
        for same_mask in by_mask.values():
            client.upsert(collection_name=collection, data=same_mask,
                          partial_update=True, timeout=90)
            upsert_requests += 1
        timings["upsert"] += time.monotonic() - tick
        tick = time.monotonic()
        after = batch_get(client, collection, ids)
        read_requests += 1
        for doc_id in ids:
            payload = payloads[doc_id]
            for name in FIELDS:
                expected = payload[name] if name in payload else before[doc_id].get(name)
                if not _equal(after[doc_id].get(name), expected):
                    raise RuntimeError("batch readback differs for field " + name)
            verified += 1
        timings["readback"] += time.monotonic() - tick
        if position % 4 == 0:
            print(json.dumps({"phase": "batch", "completed": verified,
                              "upsert_requests": upsert_requests}), flush=True)
    elapsed = time.monotonic() - started
    return {"seconds": round(elapsed, 3), "rows_per_second": round(len(rows) / elapsed, 3),
            "outcomes": {"updated": verified - unchanged, "unchanged": unchanged},
            "read_requests": read_requests,
            "upsert_requests": upsert_requests, "distinct_payload_masks": len(masks),
            "largest_mask_group": max(masks.values()),
            "stage_seconds": {name: round(value, 3) for name, value in timings.items()}}


def grouped_batch_global_pass(client: MilvusClient, collection: str,
                              rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group exact sparse masks across the full sample to minimize RPC count."""
    started = time.monotonic()
    timings = collections.Counter()
    masks = collections.Counter()
    before_all: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    read_requests = upsert_requests = 0

    tick = time.monotonic()
    for group in groups(rows, BATCH_SIZE):
        result = batch_get(client, collection, [item["id"] for item in group])
        before_all.update(result)
        read_requests += 1
    timings["before_read"] = time.monotonic() - tick

    tick = time.monotonic()
    for item in rows:
        payload = update_payload(item["id"], item["values"], item["vectors"], before_all[item["id"]])
        payloads[item["id"]] = payload
        if len(payload) == 1:
            continue
        mask = tuple(sorted(payload))
        by_mask[mask].append(payload)
        masks[mask] += 1
    timings["convert"] = time.monotonic() - tick

    tick = time.monotonic()
    for same_mask in by_mask.values():
        client.upsert(collection_name=collection, data=same_mask,
                      partial_update=True, timeout=120)
        upsert_requests += 1
    timings["upsert"] = time.monotonic() - tick

    tick = time.monotonic()
    verified = unchanged = 0
    for group in groups(rows, BATCH_SIZE):
        ids = [item["id"] for item in group]
        after = batch_get(client, collection, ids)
        read_requests += 1
        for doc_id in ids:
            payload = payloads[doc_id]
            for name in FIELDS:
                expected = payload[name] if name in payload else before_all[doc_id].get(name)
                if not _equal(after[doc_id].get(name), expected):
                    raise RuntimeError("global batch readback differs for field " + name)
            verified += 1
            unchanged += int(len(payload) == 1)
    timings["readback"] = time.monotonic() - tick
    elapsed = time.monotonic() - started
    return {"seconds": round(elapsed, 3), "rows_per_second": round(len(rows) / elapsed, 3),
            "outcomes": {"updated": verified - unchanged, "unchanged": unchanged},
            "read_requests": read_requests, "upsert_requests": upsert_requests,
            "distinct_payload_masks": len(masks),
            "largest_mask_group": max(masks.values(), default=0),
            "stage_seconds": {name: round(value, 3) for name, value in timings.items()}}


def main() -> None:
    rows = dataset()
    client, collection = create_local_collection()
    report = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "uri": URI, "database": DATABASE, "collection": collection,
              "records": len(rows), "schema_fields": len(FIELDS), "index_count": 18,
              "vector_index_type": "FLAT", "production_milvus_writes": 0,
              "kafka_commits": 0}
    report_path = OUT / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        for group in groups(rows, 25):
            client.insert(collection_name=collection, data=[seed_record(item) for item in group], timeout=90)
        client.flush(collection, timeout=180)
        count = client.query(collection, filter="", output_fields=["count(*)"],
                             consistency_level="Strong", timeout=90)[0]["count(*)"]
        if count != 1000:
            raise RuntimeError("seed logical count is not 1,000")
        report["seeded_count"] = count
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        report["sequential"] = sequential_pass(rows, collection)
        client.flush(collection, timeout=180)
        verify_pass(client, collection, rows)
        report["sequential_verified"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        reset_seed(client, collection, rows)
        report["reset_verified"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        report["batched"] = grouped_batch_pass(client, collection, rows)
        client.flush(collection, timeout=180)
        verify_pass(client, collection, rows)
        after_count = client.query(collection, filter="", output_fields=["count(*)"],
                                   consistency_level="Strong", timeout=90)[0]["count(*)"]
        if after_count != 1000:
            raise RuntimeError("final logical count changed")
        report["batched_verified"] = True
        report["final_count"] = after_count
        report["speedup"] = round(report["batched"]["rows_per_second"] /
                                  report["sequential"]["rows_per_second"], 3)
        report["passed"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
