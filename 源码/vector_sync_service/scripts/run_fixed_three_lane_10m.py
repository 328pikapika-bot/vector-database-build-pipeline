#!/usr/bin/env python3
"""Run the fixed, isolated three-lane 10-minute local benchmark.

The script is intentionally test-only: it creates only the three ``*_10M_*``
Kafka topics and V2 collections, uses the existing test Mongo database, and
never drops a database or an existing collection.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "vector_sync_service"
DRIVER = SERVICE / "scripts" / "run_independent_kafka_collection_benchmark.py"
SOURCE = ROOT / "bakkafkadata" / "31" / "kafka_messages.jsonl"
MONGO_FAILURES = "doc_sync_failures"
TOPICS = {
    "metadata": "VECTOR_JOURNAL_ARTICLE_V2_10M_METADATA_TEST",
    "fulltext": "VECTOR_JOURNAL_ARTICLE_V2_10M_FULLTEXT_TEST",
    "image": "VECTOR_JOURNAL_ARTICLE_V2_10M_IMAGE_TEST",
}
GROUPS = {
    "metadata": "vector_sync_v2_10m_metadata_test",
    "fulltext": "vector_sync_v2_10m_fulltext_test",
    "image": "vector_sync_v2_10m_image_test",
}
COLLECTIONS = {
    "metadata": "scientific_documents_v2_10m",
    "fulltext": "scientific_fulltext_chunks_v2_10m",
    "image": "scientific_images_v2_10m",
}
CORE_COLLECTIONS = {
    "metadata": "scientific_documents",
    "fulltext": "scientific_fulltext_chunks",
    "image": "scientific_images",
}
COUNTS = {"metadata": 15_000, "fulltext": 3_000, "image": 6_000}


def _kafka_config() -> dict[str, str]:
    return {
        "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        "security.protocol": os.environ.get("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
        "sasl.mechanism": os.environ.get("KAFKA_SASL_MECHANISM", "PLAIN"),
        "sasl.username": os.environ.get("KAFKA_SASL_USERNAME", ""),
        "sasl.password": os.environ.get("KAFKA_SASL_PASSWORD", ""),
    }


def _milvus() -> MilvusClient:
    return MilvusClient(uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ.get('MILVUS_PORT', '19530')}", timeout=60)


def _topic_probe() -> dict[str, Any]:
    admin = AdminClient(_kafka_config())
    metadata = admin.list_topics(timeout=20)
    result: dict[str, Any] = {}
    for lane, topic in TOPICS.items():
        item = metadata.topics.get(topic)
        result[lane] = {
            "topic": topic,
            "exists": item is not None and not item.error,
            "partitions": sorted(item.partitions) if item is not None and not item.error else [],
        }
    return result


def _ensure_topics() -> dict[str, Any]:
    probe = _topic_probe()
    missing = [item["topic"] for item in probe.values() if not item["exists"]]
    if missing:
        admin = AdminClient(_kafka_config())
        futures = admin.create_topics([NewTopic(t, num_partitions=1, replication_factor=1) for t in missing], request_timeout=30)
        for topic, future in futures.items():
            try:
                future.result()
            except Exception as exc:
                if "TOPIC_ALREADY_EXISTS" not in str(exc).upper():
                    raise
    return _topic_probe()


def _rebuild_empty_targets() -> None:
    """Clean only this benchmark's empty targets after an interrupted run."""
    client = _milvus()
    try:
        for name in COLLECTIONS.values():
            if client.has_collection(collection_name=name):
                stats = client.get_collection_stats(collection_name=name)
                if int(stats.get("row_count", 0)) != 0:
                    raise RuntimeError(f"refusing to drop non-empty test collection {name}: {stats}")
                client.drop_collection(collection_name=name)
    finally:
        client.close()
    admin = AdminClient(_kafka_config())
    futures = admin.delete_topics(list(TOPICS.values()), request_timeout=30)
    for topic, future in futures.items():
        try:
            future.result()
        except Exception as exc:
            if "UNKNOWN_TOPIC" not in str(exc).upper():
                raise


def _collection_probe(client: MilvusClient) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lane, name in {**COLLECTIONS, **{f"core_{k}": v for k, v in CORE_COLLECTIONS.items()}}.items():
        exists = client.has_collection(collection_name=name)
        item: dict[str, Any] = {"name": name, "exists": bool(exists)}
        if exists:
            try:
                client.flush(collection_name=name)
                item["row_count"] = int(client.get_collection_stats(collection_name=name).get("row_count", 0))
            except Exception as exc:
                item["stats_error"] = f"{type(exc).__name__}: {exc}"
            try:
                item["indexes"] = [client.describe_index(collection_name=name, index_name=i) for i in client.list_indexes(collection_name=name)]
            except Exception as exc:
                item["index_error"] = f"{type(exc).__name__}: {exc}"
        result[lane] = item
    return result


def _initialize_fixed_collections() -> dict[str, Any]:
    """Create the three V2 schemas through the production initializers."""
    os.environ.update(
        {
            "MILVUS_COLLECTION": COLLECTIONS["metadata"],
            "DOCUMENT_COLLECTION_NAME": COLLECTIONS["metadata"],
            "FULLTEXT_CHUNK_COLLECTION": COLLECTIONS["fulltext"],
            "FULLTEXT_CHUNK_COLLECTION_NAME": COLLECTIONS["fulltext"],
            "IMAGE_COLLECTION_NAME": COLLECTIONS["image"],
            "IMAGE_VECTOR_SCHEMA_MODE": "history_compatible_v2",
            "IMAGE_VISUAL_EMBED_ENABLED": "true",
            "INIT_VECTOR_COLLECTIONS_ON_STARTUP": "false",
            "DOCUMENT_INDEX_TYPE": "IVFPQ",
            "DOCUMENT_INDEX_METRIC_TYPE": "COSINE",
            "DOCUMENT_IVFPQ_NLIST": "4096",
            "DOCUMENT_IVFPQ_M": "64",
            "DOCUMENT_IVFPQ_NBITS": "8",
            "FULLTEXT_CHUNK_INDEX_TYPE": "IVFPQ",
            "FULLTEXT_CHUNK_INDEX_METRIC_TYPE": "COSINE",
            "FULLTEXT_CHUNK_IVFPQ_NLIST": "4096",
            "FULLTEXT_CHUNK_IVFPQ_M": "64",
            "FULLTEXT_CHUNK_IVFPQ_NBITS": "8",
            "IMAGE_DENSE_INDEX_TYPE": "IVFPQ",
            "IMAGE_DENSE_INDEX_METRIC_TYPE": "IP",
            "IMAGE_DENSE_IVFPQ_NLIST": "4096",
            "IMAGE_DENSE_IVFPQ_M": "64",
            "IMAGE_DENSE_IVFPQ_NBITS": "8",
        }
    )
    if str(SERVICE) not in sys.path:
        sys.path.insert(0, str(SERVICE))
    from app.config import settings
    from app.services.milvus_chunk_service import MilvusChunkService
    from app.services.milvus_image_service import MilvusImageService
    from app.services.milvus_service import get_milvus_service

    created: dict[str, Any] = {}
    created["metadata"] = get_milvus_service().init_collection(COLLECTIONS["metadata"], settings.VECTOR_DIM)
    created["fulltext"] = MilvusChunkService().init_chunk_collection(COLLECTIONS["fulltext"], settings.VECTOR_DIM)
    created["image"] = MilvusImageService(schema_mode="history_compatible_v2").init_image_collection(COLLECTIONS["image"])
    return created


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")[:42]


def _child_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if len(normalized) > 30:
        normalized = f"{normalized[:20]}_{normalized[-9:]}"
    return normalized[:30] or "benchmark"


def _child_cmd(lane: str, run_id: str, duration: int, mongo_db: str) -> list[str]:
    common = [
        sys.executable, str(DRIVER), "--target", lane,
        "--duration-seconds", str(duration), "--messages", str(COUNTS[lane]),
        "--run-id", run_id, "--topic", TOPICS[lane], "--source-jsonl", str(SOURCE),
        "--collection-name", COLLECTIONS[lane], "--mongo-db", mongo_db,
        "--consumer-group", GROUPS[lane], "--skip-mongo-reset", "--skip-collection-reset",
        "--embedding-text-rate", str({"metadata": 72, "fulltext": 168, "image": 20}[lane]),
    ]
    if lane == "metadata":
        common += ["--metadata-mode", "metadata_bulk", "--metadata-batch-size", "32",
                   "--embedding-batch-size", "32", "--document-concurrency", "2",
                   "--metadata-pipeline-inflight-batches", "16", "--embedding-concurrency", "2"]
    elif lane == "fulltext":
        common += ["--fulltext-direct", "--fulltext-direct-batch-size", "32",
                   "--fulltext-direct-pipeline-inflight-batches", "8", "--fulltext-concurrency", "32",
                   "--embedding-batch-size", "32", "--image-attachment-prefix", os.environ.get("ATTACHMENT_PREFIX", "")]
    else:
        common += ["--image-direct", "--image-direct-batch-size", "32",
                   "--image-direct-pipeline-inflight-batches", "6", "--image-concurrency", "32",
                   "--image-visual-concurrency", "32", "--image-text-concurrency", "12",
                   "--image-download-concurrency", "24", "--image-text-batch-size", "32",
                   "--image-text-max-in-flight-batches", "12", "--image-mmd-hot-path-mode", "required",
                   "--image-attachment-prefix", os.environ.get("ATTACHMENT_PREFIX", "")]
    if lane == "image":
        common += ["--image-direct"]
    return common


def _failure_summary(mongo_uri: str, db: str, run_id: str) -> dict[str, Any]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    try:
        # Payload IDs are prefixed with lt_<run-id>_<ordinal> by the child.
        regex = {"$regex": f"^lt_{re.escape(run_id)}_"}
        coll = client[db][MONGO_FAILURES]
        count = int(coll.count_documents({"doc_id": regex}))
        samples = list(coll.find({"doc_id": regex}, {"_id": 0, "doc_id": 1, "failure_stage": 1, "error_code": 1, "status": 1}).limit(20))
        return {"count": count, "samples": samples}
    finally:
        client.close()


def _read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"report_missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"report_missing": True, "path": str(path), "read_error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--max-wait-seconds", type=int, default=900)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--rebuild-empty-test-targets", action="store_true",
                        help="After an interrupted test, delete only empty fixed test topics/collections and recreate them.")
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.max_wait_seconds <= 0:
        raise ValueError("duration and max-wait must be positive")
    load_dotenv(SERVICE / ".env", override=False)
    mongo_uri = os.environ["MONGO_URI"]
    mongo_db = "nstl_vector_sync_scientific_20260617"
    run_id = _safe_slug(args.run_id or datetime.now(timezone.utc).strftime("three_lane_10m_%Y%m%dT%H%M%SZ"))
    out = SERVICE / "out" / f"codex_fixed_three_lane_10m_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    before_topics = _topic_probe()
    before_collections = _collection_probe(_milvus())
    # Existing fixed targets are never deleted or silently reused.
    existing_targets = [v for k, v in before_collections.items() if k in COLLECTIONS and v["exists"]]
    if existing_targets:
        if not args.rebuild_empty_test_targets:
            raise RuntimeError("固定测试集合已存在，未获清理确认，停止运行: " + json.dumps(existing_targets, ensure_ascii=False))
        _rebuild_empty_targets()
        before_topics = _topic_probe()
        before_collections = _collection_probe(_milvus())
    topic_state = _ensure_topics()
    initialized = _initialize_fixed_collections()
    after_create_collections = _collection_probe(_milvus())
    core_before = {k: before_collections[f"core_{k}"].get("row_count", 0) for k in CORE_COLLECTIONS}

    processes: dict[str, subprocess.Popen[Any]] = {}
    handles: list[Any] = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        for lane in ("metadata", "fulltext", "image"):
            stdout = (out / f"{lane}.stdout.log").open("w", encoding="utf-8")
            stderr = (out / f"{lane}.stderr.log").open("w", encoding="utf-8")
            handles.extend([stdout, stderr])
            processes[lane] = subprocess.Popen(_child_cmd(lane, run_id, args.duration_seconds, mongo_db), cwd=SERVICE, stdout=stdout, stderr=stderr)
        deadline = time.monotonic() + args.max_wait_seconds
        exit_codes: dict[str, int | None] = {}
        for lane, process in processes.items():
            try:
                exit_codes[lane] = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_codes[lane] = None
        reports: dict[str, Any] = {}
        # Child reports live in their standard per-lane directories.
        for lane in processes:
            child_slug = _child_slug(f"{run_id}_{lane}")
            reports[lane] = _read_report(SERVICE / "out" / f"codex_independent_kafka_{child_slug}" / "report.json")
        final_collections = _collection_probe(_milvus())
        core_after = {k: final_collections[f"core_{k}"].get("row_count", 0) for k in CORE_COLLECTIONS}
        report = {
            "run_id": run_id, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_requested_seconds": args.duration_seconds, "mongo_db": mongo_db, "topics": topic_state,
            "consumer_groups": GROUPS, "collections": COLLECTIONS, "message_counts": COUNTS,
            "lanes": reports, "exit_codes": exit_codes,
            "mongo_failures": _failure_summary(mongo_uri, mongo_db, run_id),
            "collection_initialization": initialized, "milvus_before": before_collections, "milvus_after": final_collections,
            "historical_core_row_counts": {"before": core_before, "after": core_after, "unchanged": core_before == core_after},
            "safety": {"topics_before": before_topics, "fixed_targets_preexisting": False, "mongo_database_reset": False},
        }
        path = out / "report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / "report.md").write_text("# 本地三路全链路 10 分钟测试\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n```\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if all(v == 0 for v in exit_codes.values()) else 1
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
