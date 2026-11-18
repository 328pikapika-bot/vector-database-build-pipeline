"""Add one nullable field and patch exactly 100 existing chunks on three hosts.

Fixed scope: only scientific_fulltext_chunks_v2.hasSourcesysSourceId is written.
There are no collection/row deletion, full-row replacement or restart operations.
The explicit --apply argument is required to perform the authorized small test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from pymilvus import DataType, MilvusClient


HOSTS = ("192.0.2.15", "192.0.2.14", "192.0.2.16")
COLLECTION = "scientific_fulltext_chunks_v2"
FIELD = "hasSourcesysSourceId"
SOLR_FIELD = "stringIMS_hasSourcesysSourceId"
SOLR = "http://service.example.invalid:8984/solr/paper6.6/select"
SIZE = 100
TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value, Message):
        return MessageToDict(value, preserving_proto_field_name=True)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return [plain(v) for v in value]


def digest(value):
    raw = json.dumps(plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def connect(host):
    return MilvusClient(uri=f"http://{host}:19533", db_name="default", timeout=30)


def count(client, collection):
    return int(client.query(collection, filter="", output_fields=["count(*)"],
                            consistency_level="Strong", timeout=30)[0]["count(*)"])


def snapshot(client):
    result = {}
    for name in sorted(client.list_collections()):
        schema = client.describe_collection(name)
        result[name] = {
            "collection_id": str(schema["collection_id"]),
            "fields": plain(schema["fields"]),
            "enable_dynamic_field": schema.get("enable_dynamic_field"),
            "count": count(client, name),
            "load_state": str(client.get_load_state(name)["state"]),
            "indexes": sorted(client.list_indexes(name)),
        }
    return result


def read_rows(client, ids, fields):
    rows = client.get(COLLECTION, ids=ids, output_fields=fields,
                      consistency_level="Strong", timeout=30)
    mapped = {r["id"]: plain(dict(r)) for r in rows}
    if set(mapped) != set(ids):
        raise RuntimeError("Selected primary keys do not all exist")
    return mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Use --apply only for the explicitly authorized 100-row test")

    folder = ROOT / "outputs" / ("fulltext_source_id_100_" + datetime.now(TZ).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now(TZ).isoformat(), "collection": COLLECTION,
              "field": FIELD, "records_per_host": SIZE, "status": "preflight",
              "hosts": {}, "timing": {}}
    def save():
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    clients = {host: connect(host) for host in HOSTS}
    try:
        for host, client in clients.items():
            baseline = snapshot(client)
            if COLLECTION not in baseline:
                raise RuntimeError(f"Missing collection on {host}")
            if FIELD in {f["name"] for f in baseline[COLLECTION]["fields"]}:
                raise RuntimeError(f"Field already exists on {host}; refusing a new benchmark")
            report["hosts"][host] = {"before": baseline}
            print("BASELINE_OK", host, baseline[COLLECTION]["count"], flush=True)
        fields = [f["name"] for f in report["hosts"][HOSTS[0]]["before"][COLLECTION]["fields"]]
        selected = clients[HOSTS[0]].query(COLLECTION, filter="chunkIndex == 0",
            output_fields=["id", "doc_id"], limit=SIZE, consistency_level="Strong", timeout=30)
        selected = sorted([plain(dict(r)) for r in selected], key=lambda r: r["id"])
        ids = [r["id"] for r in selected]
        doc_ids = sorted({r["doc_id"] for r in selected})
        if len(ids) != SIZE or len(set(ids)) != SIZE or len(doc_ids) != SIZE:
            raise RuntimeError("Selection must contain 100 distinct existing chunks/documents")
        if any(not isinstance(i, str) or len(i) != 32 or any(c not in "0123456789abcdef" for c in i) for i in doc_ids):
            raise RuntimeError("Unexpected document ID format")

        started = time.perf_counter()
        session = requests.Session()
        session.trust_env = False
        response = session.get(SOLR, params={"q": "{!terms f=id}" + ",".join(doc_ids),
            "fl": "id," + SOLR_FIELD, "rows": SIZE, "wt": "json", "timeAllowed": 10000}, timeout=(6, 20))
        response.raise_for_status()
        body = response.json()
        if body.get("responseHeader", {}).get("partialResults"):
            raise RuntimeError("Partial Solr results")
        docs = {row["id"]: row for row in body["response"]["docs"]}
        if set(docs) != set(doc_ids):
            raise RuntimeError("Not all selected documents matched in Solr")
        values = {}
        for doc_id, row in docs.items():
            value = row.get(SOLR_FIELD)
            if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
                raise RuntimeError(f"Missing/invalid journal-source IDs for {doc_id}")
            values[doc_id] = value
        report["timing"]["solr_100_documents_seconds"] = time.perf_counter() - started
        patches = [{"id": r["id"], FIELD: values[r["doc_id"]]} for r in selected]
        report["selected"] = selected
        report["patches_sha256"] = digest(patches)
        (folder / "patches.json").write_text(json.dumps(patches, ensure_ascii=False, indent=2), encoding="utf-8")

        before_rows = {}
        for host, client in clients.items():
            before_rows[host] = read_rows(client, ids, fields)
            report["hosts"][host]["sample_before_sha256"] = digest(before_rows[host])
            (folder / (host.rsplit(".", 1)[-1] + "_original_100.json")).write_text(
                json.dumps(before_rows[host], ensure_ascii=False), encoding="utf-8")
        if len({digest(rows) for rows in before_rows.values()}) != 1:
            raise RuntimeError("The original 100 records/vectors differ across hosts")
        report["status"] = "preflight_passed"
        save()
        print("PREFLIGHT_PASSED", SIZE, "matching chunks; Solr source values and all original vectors saved locally", flush=True)

        barrier = threading.Barrier(3)
        def add_field(host):
            barrier.wait(timeout=15)
            started = time.perf_counter()
            clients[host].add_collection_field(COLLECTION, FIELD, DataType.ARRAY,
                element_type=DataType.VARCHAR, max_capacity=4096, max_length=65535,
                nullable=True, desc="Journal/source sysSourceId from Solr stringIMS_hasSourcesysSourceId", timeout=60)
            return host, time.perf_counter() - started
        report["status"] = "adding_field"
        save()
        add_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(add_field, host) for host in HOSTS]):
                host, elapsed = future.result()
                report["hosts"][host]["add_field_seconds"] = elapsed
                save()
                print("FIELD_ADDED", host, round(elapsed, 4), "seconds", flush=True)
        report["timing"]["add_field_three_hosts_wall_seconds"] = time.perf_counter() - add_started
        for host, client in clients.items():
            desc = client.describe_collection(COLLECTION)
            field = next(f for f in desc["fields"] if f["name"] == FIELD)
            if int(field["type"]) != int(DataType.ARRAY) or not field.get("nullable") or int(field.get("element_type", 0)) != int(DataType.VARCHAR):
                raise RuntimeError(f"New field schema mismatch on {host}")
            if len(desc["fields"]) != len(fields) + 1:
                raise RuntimeError(f"Unexpected schema change on {host}")

        barrier = threading.Barrier(3)
        def patch_host(host):
            barrier.wait(timeout=15)
            started = time.perf_counter()
            result = clients[host].upsert(COLLECTION, data=patches, partial_update=True, timeout=90)
            return host, time.perf_counter() - started, result
        report["status"] = "patching_exact_100"
        save()
        patch_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(patch_host, host) for host in HOSTS]):
                host, elapsed, result = future.result()
                report["hosts"][host]["partial_update_seconds"] = elapsed
                report["hosts"][host]["upsert_count"] = int(result.get("upsert_count", -1))
                report["hosts"][host]["update_timestamp"] = str(result.get("timestamp", ""))
                save()
                print("PATCHED", host, "count", result.get("upsert_count"), "seconds", round(elapsed, 4), flush=True)
                if int(result.get("upsert_count", -1)) != SIZE:
                    raise RuntimeError(f"Unexpected upsert count on {host}")
        report["timing"]["patch_100_three_hosts_wall_seconds"] = time.perf_counter() - patch_started

        report["status"] = "verifying"
        save()
        patched_rows = {}
        for host, client in clients.items():
            started = time.perf_counter()
            after = read_rows(client, ids, fields + [FIELD])
            expected = {r["id"]: r[FIELD] for r in patches}
            if any(after[i].get(FIELD) != expected[i] for i in ids):
                raise RuntimeError(f"New source IDs differ from Solr on {host}")
            original = {i: {k: v for k, v in row.items() if k != FIELD} for i, row in after.items()}
            if digest(original) != digest(before_rows[host]):
                raise RuntimeError(f"Original fields or vectors changed on {host}")
            nonnull = int(client.query(COLLECTION, filter=f"{FIELD} is not null",
                output_fields=["count(*)"], consistency_level="Strong", timeout=30)[0]["count(*)"])
            if nonnull != SIZE:
                raise RuntimeError(f"Unexpected total filled field count on {host}: {nonnull}")
            patched_rows[host] = after
            state = snapshot(client)
            baseline = report["hosts"][host]["before"]
            if set(state) != set(baseline):
                raise RuntimeError(f"Collection inventory changed on {host}")
            for name in baseline:
                normalized = dict(state[name])
                if name == COLLECTION:
                    normalized["fields"] = [f for f in normalized["fields"] if f["name"] != FIELD]
                if normalized != baseline[name]:
                    raise RuntimeError(f"Counts/schema/indexes/load-state changed for {host}/{name}")
            report["hosts"][host].update({"verification_seconds": time.perf_counter()-started,
                "after": state, "original_100_unchanged": True, "source_values_match": True,
                "nonnull_source_id_count": nonnull,
                "sample_after_sha256": digest(after)})
            save()
            print("VERIFIED", host, "original fields/vectors unchanged; source IDs match; collection counts unchanged", flush=True)
        if len({digest(rows) for rows in patched_rows.values()}) != 1:
            raise RuntimeError("The three hosts differ after the patch")
        report["three_hosts_100_identical"] = True
        report["status"] = "completed"
        report["finished_at"] = datetime.now(TZ).isoformat()
        save()
        lines = ["# 全文期刊来源 ID 三节点 100 条补值测试", "", f"集合：`{COLLECTION}`；字段：`{FIELD}`。", "",
                 "| 节点 | 新增字段（秒） | 100 条局部更新（秒） | 原字段与向量 |", "| --- | ---: | ---: | --- |"]
        for host in HOSTS:
            result = report["hosts"][host]
            lines.append(f"| {host} | {result['add_field_seconds']:.3f} | {result['partial_update_seconds']:.3f} | 核对一致 |")
        lines += ["", f"三台并行更新 100 条的总等待时间：{report['timing']['patch_100_three_hosts_wall_seconds']:.3f} 秒。",
                  "", "100 条采用同一主键清单和 Solr 字段值，三台查询结果完全一致。未新增第二个来源字段。",
                  "所有集合条数、其他集合 Schema/索引/加载状态与执行前一致。样本原文、其他字段和向量逐条核对一致。",
                  "", "该结果只代表 100 条的小批量更新，不直接代表近 2 亿切片的长期处理速度。"]
        (folder / "测试结果.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print("REPORT", folder / "report.json", flush=True)
        print("RESULT", folder / "测试结果.md", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        save()
        print("STOPPED", repr(exc), "REPORT", folder / "report.json", flush=True)
        raise
    finally:
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    main()
