#!/usr/bin/env python3
"""Safely rebuild the three V2 collections and benchmark all Kafka lanes.

This command is intentionally limited to the V2 targets used by the three
independent workers.  It never drops or writes the historical core targets.
Every lane uses a new Kafka group, reads the real topic from its beginning,
resolves the document from the test Solr service, and uses the production
direct worker implementation to write the V2 collection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, TopicPartition
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymongo import MongoClient


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

# Kafka and embedding diagnostics may contain arbitrary Unicode.  Benchmark
# failures must reflect the data path rather than the Windows console codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

DEFAULT_V2_COLLECTIONS = {
    "metadata": "scientific_documents_v2",
    "fulltext": "scientific_fulltext_chunks_v2",
    "image": "scientific_images_v2",
}
V2_COLLECTIONS = dict(DEFAULT_V2_COLLECTIONS)
HISTORICAL_CORE_COLLECTIONS = {
    "scientific_documents",
    "scientific_fulltext_chunks",
    "scientific_images",
}
TOPICS = {
    "metadata": "VECTOR_JOURNAL_ARTICLE_METADATA_BASE",
    "fulltext": "VECTOR_JOURNAL_ARTICLE_FULLTEXT",
    "image": "VECTOR_JOURNAL_ARTICLE_IMAGE",
}
TEST_SOLR_PAPER_URL = "http://solr-bak1.example.invalid:8984/solr/paper6.6"
TEST_SOLR_IMAGE_URL = "http://solr-bak1.example.invalid:8984/solr/standard6.6"
IMAGE_ATTACHMENT_PREFIX = "http://127.0.0.1:12001/"

# The prior run used the same V2 targets and real topics.  Keep it in the
# report so the result is comparable without mixing in historical collections.
BASELINE = {
    "metadata": {"rows": 743, "throughput": 2.48, "unit": "documents/s"},
    "fulltext": {"rows": 12_474, "throughput": 41.58, "unit": "chunks/s"},
    "image": {"rows": 296, "throughput": 0.99, "unit": "images/s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "reset", "lane"), default="run")
    parser.add_argument("--lane", choices=tuple(V2_COLLECTIONS), default="")
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--drain-seconds", type=int, default=180)
    parser.add_argument(
        "--post-run-settle-seconds",
        type=int,
        default=30,
        help="Wait after all lanes exit before recording stable V2 row counts.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--collection-suffix",
        default="",
        help=(
            "Use isolated scientific_*_<suffix> targets. Custom values must start "
            "with test_; omit the option to preserve the existing V2 targets."
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse existing isolated test collections without dropping or clearing them.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required for the reset that drops and recreates the three V2 collections.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_id(value: str) -> str:
    return value.strip() or datetime.now(timezone.utc).strftime("v2_optimized_5m_%Y%m%dT%H%M%SZ")


def _configure_collection_targets(suffix: str) -> None:
    """Select the fixed V2 targets or a strictly isolated three-collection set."""
    global V2_COLLECTIONS
    normalized = suffix.strip().lower()
    if not normalized:
        V2_COLLECTIONS = dict(DEFAULT_V2_COLLECTIONS)
        return
    if not re.fullmatch(r"test_[a-z0-9_]{1,48}", normalized):
        raise ValueError("--collection-suffix must match test_[a-z0-9_]{1,48}")
    V2_COLLECTIONS = {
        "metadata": f"scientific_documents_{normalized}",
        "fulltext": f"scientific_fulltext_chunks_{normalized}",
        "image": f"scientific_images_{normalized}",
    }


def _benchmark_image_setting(name: str, default: int) -> str:
    """Read a positive image benchmark override without weakening production defaults."""
    try:
        value = int(str(os.environ.get(f"BENCH_IMAGE_{name}") or default).strip())
    except (TypeError, ValueError):
        value = default
    return str(max(value, 1))


def _benchmark_setting(name: str, default: int) -> str:
    """Read a positive benchmark-only override without changing service defaults."""
    try:
        value = int(str(os.environ.get(f"BENCH_{name}") or default).strip())
    except (TypeError, ValueError):
        value = default
    return str(max(value, 1))


def _benchmark_enabled(name: str, default: bool) -> bool:
    raw = str(os.environ.get(f"BENCH_{name}") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _load_environment() -> None:
    load_dotenv(SERVICE_ROOT / ".env", override=False)


def _milvus_uri() -> str:
    host = str(os.environ.get("MILVUS_HOST") or "127.0.0.1").strip()
    port = str(os.environ.get("MILVUS_PORT") or "19533").strip()
    return f"http://{host}:{port}"


def _milvus_client() -> MilvusClient:
    return MilvusClient(
        uri=_milvus_uri(),
        user=str(os.environ.get("MILVUS_USER") or ""),
        password=str(os.environ.get("MILVUS_PASSWORD") or ""),
        timeout=180,
    )


def _consumer_config(group_id: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": str(os.environ["KAFKA_BOOTSTRAP_SERVERS"]),
        "group.id": group_id,
        "security.protocol": str(os.environ.get("KAFKA_SECURITY_PROTOCOL") or "SASL_PLAINTEXT"),
        "sasl.mechanism": str(os.environ.get("KAFKA_SASL_MECHANISM") or "PLAIN"),
        "sasl.username": str(os.environ.get("KAFKA_SASL_USERNAME") or ""),
        "sasl.password": str(os.environ.get("KAFKA_SASL_PASSWORD") or ""),
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 1_800_000,
        "session.timeout.ms": 30_000,
        "heartbeat.interval.ms": 10_000,
    }


def _topic_availability() -> dict[str, Any]:
    client = Consumer(_consumer_config(f"vector_sync_v2_probe_{int(time.time())}"))
    try:
        report: dict[str, Any] = {}
        metadata = client.list_topics(timeout=15)
        for lane, topic in TOPICS.items():
            partitions = metadata.topics.get(topic)
            if partitions is None or partitions.error is not None:
                raise RuntimeError(f"Kafka topic unavailable: {topic}")
            rows: list[dict[str, int]] = []
            total = 0
            for partition_id in sorted(partitions.partitions):
                low, high = client.get_watermark_offsets(
                    TopicPartition(topic, partition_id), timeout=15
                )
                available = max(int(high) - int(low), 0)
                total += available
                rows.append({"partition": partition_id, "low": low, "high": high, "available": available})
            if total <= 0:
                raise RuntimeError(f"Kafka topic has no available messages: {topic}")
            report[lane] = {"topic": topic, "available_messages": total, "partitions": rows}
        return report
    finally:
        client.close()


def _snapshot_core_collections() -> dict[str, int]:
    client = _milvus_client()
    try:
        rows: dict[str, int] = {}
        for collection in sorted(HISTORICAL_CORE_COLLECTIONS):
            if not client.has_collection(collection):
                raise RuntimeError(f"Protected historical collection is missing: {collection}")
            rows[collection] = int(client.get_collection_stats(collection).get("row_count", 0))
        return rows
    finally:
        client.close()


def _apply_runtime_environment(*, lane: str, run_id: str, out_dir: Path) -> None:
    """Set one process' production-equivalent worker configuration."""
    collection = V2_COLLECTIONS[lane]
    group_id = f"vector_sync_{lane}_v2_load_{run_id}"[:220]
    runtime_dir = out_dir / "runtime" / lane
    runtime_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "SYNC_WORKER_ROLE": lane,
        "EMBEDDING_RATE_LIMIT_ENABLED": "true",
        "EMBEDDING_LANE": lane,
        "EMBEDDING_TEXT_BURST": "32",
        "KAFKA_TOPIC": TOPICS[lane],
        "KAFKA_GROUP_ID": group_id,
        f"KAFKA_{lane.upper()}_GROUP_ID": group_id,
        "KAFKA_TARGET_VERSIONED_GROUP_ENABLED": "true",
        "KAFKA_CONSUME_LIMIT": "0",
        "KAFKA_BACKPRESSURE_ENABLED": "false",
        "KAFKA_DUMP_FILE": str(runtime_dir / "kafka_messages.jsonl"),
        "KAFKA_ERROR_DUMP_FILE": str(runtime_dir / "kafka_messages_error.jsonl"),
        "KAFKA_PROCESSED_DUMP_FILE": str(runtime_dir / "kafka_messages_processed.jsonl"),
        "MONGO_SUCCESS_AUDIT_ENABLED": "false",
        "MILVUS_COLLECTION": V2_COLLECTIONS["metadata"],
        "DOCUMENT_COLLECTION_NAME": V2_COLLECTIONS["metadata"],
        "FULLTEXT_CHUNK_COLLECTION": V2_COLLECTIONS["fulltext"],
        "FULLTEXT_CHUNK_COLLECTION_NAME": V2_COLLECTIONS["fulltext"],
        "IMAGE_COLLECTION_NAME": V2_COLLECTIONS["image"],
        "DOCUMENT_INDEX_TYPE": "IVFPQ",
        "FULLTEXT_CHUNK_INDEX_TYPE": "IVFPQ",
        "IMAGE_DENSE_INDEX_TYPE": "IVFPQ",
        "IMAGE_VECTOR_SCHEMA_MODE": "history_compatible_v2",
        "IMAGE_VISUAL_EMBED_ENABLED": "false",
        # Use the current internal high-throughput origin for both images and MMD.
        "ATTACHMENT_PREFIX": IMAGE_ATTACHMENT_PREFIX,
        "SOLR_PAPER_URL": TEST_SOLR_PAPER_URL,
        "SOLR_IMAGE_URL": TEST_SOLR_IMAGE_URL,
        "PIPELINE_VERSIONED_DEDUP_ENABLED": "true",
        "EMBEDDING_CROSS_DOCUMENT_BATCH_ENABLED": "true",
        "EMBEDDING_CROSS_DOCUMENT_BATCH_WAIT_MS": "25",
        "FULLTEXT_CHUNK_WRITE_MODE": "safe_replace",
        "FULLTEXT_CHUNK_VISIBILITY_STRICT": "true",
        "FULLTEXT_TASK_UPSERT_ENABLED": "false",
        "IMAGE_TASK_UPSERT_ENABLED": "false",
        "IMAGE_MILVUS_WRITE_ENABLED": "false",
        "IMAGE_DIRECT_STALE_CLEANUP_ENABLED": "true",
        "IMAGE_VISIBILITY_CHECK_ENABLED": "false",
        "IMAGE_RUNTIME_ATTACHMENT_MODE": "failure_only",
        "IMAGE_MMD_HOT_PATH_MODE": "required",
        "IMAGE_DOWNLOAD_LANE_ENABLED": "true",
        "IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC": "15",
        "IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC": "120",
        "IMAGE_SLOW_DOWNLOAD_CONCURRENCY": "8",
        "METADATA_BULK_TIMING_ENABLED": "true",
        "SYNC_PIPELINE_MODE": "metadata_bulk",
        "DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES": "16",
        "DOCUMENT_BULK_PIPELINE_ENABLED": "true",
    }
    # The image lane uses the deployment-sized execution window.  This is a
    # real end-to-end throughput test, so limiting it to the old eight-document
    # serial microbatch would hide available worker/model parallelism.
    if lane == "metadata":
        metadata_batch_size = _benchmark_setting("METADATA_BATCH_SIZE", 128)
        common.update(
            {
                "BATCH_MAX_SIZE": metadata_batch_size,
                "BATCH_MAX_WAIT_MS": _benchmark_setting("METADATA_BATCH_WAIT_MS", 500),
                "KAFKA_CONSUMER_MICRO_BATCH_SIZE": metadata_batch_size,
                "EMBEDDING_BATCH_SIZE": _benchmark_setting("METADATA_EMBEDDING_BATCH_SIZE", 32),
                "EMBEDDING_MAX_CONCURRENCY": _benchmark_setting(
                    "METADATA_EMBEDDING_MAX_CONCURRENCY", 2
                ),
                "EMBEDDING_TEXT_RATE": _benchmark_setting(
                    "METADATA_EMBEDDING_TEXT_RATE", 80
                ),
                "DOCUMENT_BULK_MILVUS_BATCH_SIZE": _benchmark_setting(
                    "METADATA_MILVUS_BATCH_SIZE", 512
                ),
                "DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES": _benchmark_setting(
                    "METADATA_PIPELINE_INFLIGHT_BATCHES", 16
                ),
            }
        )
        if not _benchmark_enabled("METADATA_SUCCESS_DUMP_ENABLED", True):
            common["KAFKA_DUMP_FILE"] = ""
            common["KAFKA_PROCESSED_DUMP_FILE"] = ""
    elif lane == "image":
        common.update(
            {
                "BATCH_MAX_SIZE": _benchmark_image_setting("BATCH_SIZE", 8),
                "BATCH_MAX_WAIT_MS": "1000",
                "KAFKA_CONSUMER_MICRO_BATCH_SIZE": _benchmark_image_setting("BATCH_SIZE", 8),
                "IMAGE_WORKER_CONCURRENCY": _benchmark_image_setting("WORKER_CONCURRENCY", 24),
                # The static-resource origin plateaus around 24-48 parallel
                # requests; keep the benchmark default aligned with the
                # deployment setting and avoid inflating raw in-flight work.
                "IMAGE_DOWNLOAD_CONCURRENCY": _benchmark_image_setting("DOWNLOAD_CONCURRENCY", 24),
                "IMAGE_VL_EMBED_CONCURRENCY": _benchmark_image_setting("VL_EMBED_CONCURRENCY", 16),
                "EMBEDDING_TEXT_RATE": _benchmark_image_setting("TEXT_EMBED_RATE", 24),
                "IMAGE_TEXT_EMBED_MAX_IN_FLIGHT_BATCHES": _benchmark_image_setting(
                    "TEXT_EMBED_MAX_IN_FLIGHT_BATCHES", 12
                ),
                "IMAGE_TEXT_EMBED_CONCURRENCY": _benchmark_image_setting(
                    "TEXT_EMBED_CONCURRENCY", 12
                ),
                "IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES": _benchmark_image_setting("PIPELINE_INFLIGHT_BATCHES", 6),
            }
        )
    else:
        fulltext_batch_size = _benchmark_setting("FULLTEXT_BATCH_SIZE", 32)
        common.update(
            {
                "BATCH_MAX_SIZE": fulltext_batch_size,
                "BATCH_MAX_WAIT_MS": "500",
                "KAFKA_CONSUMER_MICRO_BATCH_SIZE": fulltext_batch_size,
                "EMBEDDING_BATCH_SIZE": _benchmark_setting("FULLTEXT_EMBEDDING_BATCH_SIZE", 32),
                "EMBEDDING_MAX_CONCURRENCY": _benchmark_setting(
                    "EMBEDDING_MAX_CONCURRENCY", 4
                ),
                "EMBEDDING_TEXT_RATE": _benchmark_setting(
                    "FULLTEXT_EMBEDDING_TEXT_RATE", 150
                ),
                "FULLTEXT_WORKER_CONCURRENCY": _benchmark_setting(
                    "FULLTEXT_WORKER_CONCURRENCY", 32
                ),
                "FULLTEXT_PIPELINE_INFLIGHT_BATCHES": _benchmark_setting(
                    "FULLTEXT_PIPELINE_INFLIGHT_BATCHES", 8
                ),
            }
        )
    os.environ.update(common)


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json(item) for item in value]
    if hasattr(value, "to_dict"):
        return _to_json(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collection_snapshot(collection_name: str) -> dict[str, Any]:
    client = _milvus_client()
    try:
        if not client.has_collection(collection_name):
            return {"exists": False, "row_count": 0, "indexes": [], "schema": {}}
        client.flush(collection_name=collection_name)
        indexes = []
        for index_name in client.list_indexes(collection_name=collection_name):
            indexes.append(client.describe_index(collection_name=collection_name, index_name=index_name))
        return {
            "exists": True,
            "row_count": int(client.get_collection_stats(collection_name).get("row_count", 0)),
            "indexes": _to_json(indexes),
            "schema": _to_json(client.describe_collection(collection_name)),
        }
    finally:
        client.close()


def _reset_v2_collections(*, execute: bool, lane: str = "") -> dict[str, Any]:
    if not execute:
        raise ValueError("Refusing to drop collections without --execute")
    if set(V2_COLLECTIONS.values()) & HISTORICAL_CORE_COLLECTIONS:
        raise RuntimeError("V2 collection allowlist overlaps protected historical collections")
    selected_lanes = (lane,) if lane else tuple(V2_COLLECTIONS)
    if any(item not in V2_COLLECTIONS for item in selected_lanes):
        raise ValueError(f"Unsupported V2 reset lane: {lane}")

    before_core = _snapshot_core_collections()
    client = _milvus_client()
    dropped: dict[str, bool] = {}
    try:
        for selected_lane in selected_lanes:
            collection_name = V2_COLLECTIONS[selected_lane]
            if collection_name in HISTORICAL_CORE_COLLECTIONS:
                raise RuntimeError(f"Refusing to drop protected collection: {collection_name}")
            exists = bool(client.has_collection(collection_name))
            if exists:
                client.drop_collection(collection_name)
            dropped[selected_lane] = exists
    finally:
        client.close()

    mongo_uri = str(os.environ["MONGO_URI"])
    mongo_db = str(os.environ["MONGO_DB"])
    mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    try:
        version_collection = mongo[mongo_db]["doc_sync_pipeline_versions"]
        cleared = version_collection.delete_many(
            {"logical_collection": {"$in": [V2_COLLECTIONS[item] for item in selected_lanes]}}
        ).deleted_count
    finally:
        mongo.close()

    # Import application services only after the V2-specific env is active.
    from app.config import settings
    from app.services.milvus_chunk_service import MilvusChunkService
    from app.services.milvus_image_service import MilvusImageService
    from app.services.milvus_service import get_milvus_service

    document_service = get_milvus_service()
    chunk_service = MilvusChunkService()
    image_service = MilvusImageService()
    try:
        created: dict[str, Any] = {}
        if "metadata" in selected_lanes:
            created["metadata"] = document_service.init_collection(
                V2_COLLECTIONS["metadata"], settings.VECTOR_DIM
            )
        if "fulltext" in selected_lanes:
            created["fulltext"] = chunk_service.init_chunk_collection(
                V2_COLLECTIONS["fulltext"], settings.VECTOR_DIM
            )
        if "image" in selected_lanes:
            created["image"] = image_service.init_image_collection(V2_COLLECTIONS["image"])
    finally:
        document_service.close()
        chunk_service.close()
        image_service.close()

    after_core = _snapshot_core_collections()
    if before_core != after_core:
        raise RuntimeError(
            "Protected historical collection row counts changed during V2 reset: "
            f"before={before_core}, after={after_core}"
        )
    return {
        "dropped_v2": dropped,
        "cleared_pipeline_version_watermarks": int(cleared),
        "created_targets": _to_json(created),
        "protected_core_rows_before": before_core,
        "protected_core_rows_after": after_core,
    }


def _verify_incremental_targets() -> dict[str, Any]:
    """Verify isolated targets exist without changing or clearing any data."""
    before_core = _snapshot_core_collections()
    targets: dict[str, Any] = {}
    for lane, collection_name in V2_COLLECTIONS.items():
        snapshot = _collection_snapshot(collection_name)
        if not snapshot.get("exists"):
            raise RuntimeError(f"Incremental target collection is missing: {collection_name}")
        targets[lane] = {
            "collection_name": collection_name,
            "row_count_before": int(snapshot.get("row_count") or 0),
        }
    return {
        "mode": "incremental",
        "dropped_v2": {lane: False for lane in V2_COLLECTIONS},
        "cleared_pipeline_version_watermarks": 0,
        "created_targets": targets,
        "protected_core_rows_before": before_core,
        "protected_core_rows_after": before_core,
    }


class TimingRecorder:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._items: Counter[str] = Counter()

    def observe(self, stage: str, value_ms: Any, *, items: int = 0) -> None:
        try:
            value = float(value_ms)
        except (TypeError, ValueError):
            return
        if value < 0:
            return
        self._values[str(stage)].append(value)
        self._items[str(stage)] += max(int(items), 0)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for stage, values in sorted(self._values.items()):
            ordered = sorted(values)
            result[stage] = {
                "count": len(ordered),
                "items": int(self._items[stage]),
                "total_ms": round(sum(ordered), 3),
                "avg_ms": round(statistics.fmean(ordered), 3),
                "p50_ms": round(statistics.median(ordered), 3),
                "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
            }
        return result


def _install_task_timing_hooks(*, lane: str, processor: Any, recorder: TimingRecorder, counters: Counter[str]) -> None:
    if lane == "fulltext":
        worker = processor._get_fulltext_worker()  # Production worker; timing wrapper only.
        original = worker.process_task

        async def timed(task: dict[str, Any]) -> Any:
            result = await original(task)
            for stage, elapsed in (getattr(result, "timings_ms", {}) or {}).items():
                recorder.observe(stage, elapsed, items=1)
            if result.success:
                counters["fulltext_task_success"] += 1
                counters["written_chunks"] += int(getattr(result, "chunk_count", 0) or 0)
            else:
                counters["fulltext_task_failed"] += 1
                counters[f"fulltext_task_error:{str(getattr(result, 'error_code', '') or 'UNKNOWN')}"] += 1
            return result

        worker.process_task = timed
        return

    if lane == "image":
        # These wrappers instrument the real direct worker. They deliberately
        # do not alter scheduling, source data, or write behavior.
        original_hydrate = processor._hydrate_record_from_solr

        async def timed_hydrate(record: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original_hydrate(record, **kwargs)
            finally:
                recorder.observe("solr_resolve", (time.perf_counter() - started) * 1000, items=1)

        processor._hydrate_record_from_solr = timed_hydrate
        original_version_check = processor._is_unchanged_source

        async def timed_version_check(snapshot: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original_version_check(snapshot)
            finally:
                recorder.observe("mongo_source_version", (time.perf_counter() - started) * 1000, items=1)

        processor._is_unchanged_source = timed_version_check
        original_mark_processed = processor._mark_source_processed

        async def timed_mark_processed(snapshot: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original_mark_processed(snapshot)
            finally:
                recorder.observe("mongo_source_mark", (time.perf_counter() - started) * 1000, items=1)

        processor._mark_source_processed = timed_mark_processed

        worker = processor._get_image_worker()  # Production worker; timing wrapper only.
        original_task = worker.process_task

        async def timed(task: dict[str, Any]) -> Any:
            result = await original_task(task)
            for stage, elapsed in (getattr(result, "timings_ms", {}) or {}).items():
                recorder.observe(stage, elapsed, items=1)
            if result.success:
                counters["image_task_success"] += 1
            else:
                counters["image_task_failed"] += 1
                counters[f"image_task_error:{str(getattr(result, 'error_code', '') or 'UNKNOWN')}"] += 1
            return result

        worker.process_task = timed
        image_service = processor._get_image_service()
        original_write = image_service.replace_images_for_documents

        def timed_write(collection_name: str, documents: dict[str, Any], **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                result = original_write(collection_name, documents, **kwargs)
                counters["written_images"] += sum(len(rows) for rows in documents.values())
                return result
            finally:
                recorder.observe(
                    "milvus_document_batch",
                    (time.perf_counter() - started) * 1000,
                    items=sum(len(rows) for rows in documents.values()),
                )

        image_service.replace_images_for_documents = timed_write


async def _run_lane(*, lane: str, duration_seconds: int, drain_seconds: int) -> dict[str, Any]:
    from app.services.batch_processor_existing import get_metadata_bulk_timing_snapshot, reset_metadata_bulk_timing
    from app.services.independent_kafka_worker import IndependentKafkaWorker
    from app.services.kafka_consumer import _build_commit_offsets, create_consumer, parse_message
    from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager

    if lane == "metadata":
        reset_metadata_bulk_timing()
    manager = get_mongo_state_manager()
    await manager.connect()
    processor = IndependentKafkaWorker(lane, state_manager=manager)
    group_id = str(os.environ[f"KAFKA_{lane.upper()}_GROUP_ID"])
    consumer = create_consumer(group_id=group_id, topic=TOPICS[lane])
    counters: Counter[str] = Counter()
    counters["compensator_replay_attempts"] = 0
    counters["slow_download_replays"] = 0
    timings = TimingRecorder()
    _install_task_timing_hooks(lane=lane, processor=processor, recorder=timings, counters=counters)
    batch: list[Any] = []
    batches: list[dict[str, Any]] = []
    pipeline_inflight_limit = max(
        int(
            os.environ.get(
                {
                    "metadata": "DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES",
                    "fulltext": "BENCH_FULLTEXT_PIPELINE_INFLIGHT_BATCHES",
                    "image": "IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES",
                }[lane]
            )
            or 1
        ),
        1,
    )
    inflight: list[tuple[list[Any], asyncio.Task[tuple[Any, float]]]] = []
    stop_receiving_at = time.monotonic() + duration_seconds
    started = time.monotonic()
    accepting_elapsed = 0.0

    async def process_active(active: list[Any]) -> tuple[Any, float]:
        batch_started = time.perf_counter()
        result = await processor.process_batch(active)
        elapsed_ms = (time.perf_counter() - batch_started) * 1000
        return result, elapsed_ms

    async def complete_oldest() -> None:
        active, task = inflight.pop(0)
        result, elapsed_ms = await task
        timings.observe("kafka_micro_batch", elapsed_ms, items=len(active))
        counters["outcome_success"] += int(result.success)
        counters["outcome_skipped"] += int(result.skip)
        counters["outcome_failed"] += int(result.fail)
        counters["processed_records"] += int(result.total)
        counters["batches"] += 1
        for outcome in result.outcomes:
            if str(getattr(outcome, "message", "")) == "source_version_unchanged":
                counters["version_dedup_skipped"] += 1
        committed = False
        if result.commit_safe:
            commit_started = time.perf_counter()
            consumer.commit(offsets=_build_commit_offsets(active), asynchronous=False)
            timings.observe("kafka_commit", (time.perf_counter() - commit_started) * 1000, items=len(active))
            counters["committed_records"] += len(active)
            counters["committed_batches"] += 1
            committed = True
        else:
            counters["uncommitted_batches"] += 1
        batches.append(
            {
                "records": len(active),
                "elapsed_ms": round(elapsed_ms, 3),
                "success": int(result.success),
                "skipped": int(result.skip),
                "failed": int(result.fail),
                "commit_safe": bool(result.commit_safe),
                "committed": committed,
            }
        )

    async def dispatch_active(active: list[Any]) -> None:
        if not active:
            return
        inflight.append((active, asyncio.create_task(process_active(active))))
        if len(inflight) >= pipeline_inflight_limit:
            await complete_oldest()

    async def drain_inflight(*, timeout_seconds: float) -> None:
        deadline = time.monotonic() + max(timeout_seconds, 1.0)
        while inflight:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                counters["drain_timeout_batches"] += len(inflight)
                for _, task in inflight:
                    task.cancel()
                await asyncio.gather(*(task for _, task in inflight), return_exceptions=True)
                inflight.clear()
                return
            try:
                await asyncio.wait_for(complete_oldest(), timeout=remaining)
            except asyncio.TimeoutError:
                counters["drain_timeout_batches"] += len(inflight) + 1
                for _, task in inflight:
                    task.cancel()
                await asyncio.gather(*(task for _, task in inflight), return_exceptions=True)
                inflight.clear()
                return

    try:
        while time.monotonic() < stop_receiving_at:
            poll_started = time.perf_counter()
            message = await asyncio.to_thread(consumer.poll, 0.2)
            timings.observe("kafka_poll", (time.perf_counter() - poll_started) * 1000, items=1)
            if message is None:
                if batch:
                    active, batch = batch, []
                    await dispatch_active(active)
                await drain_inflight(timeout_seconds=drain_seconds)
                continue
            record = parse_message(message)
            if record is None:
                continue
            counters["kafka_received"] += 1
            batch.append(record)
            micro_batch_size = int(os.environ.get("KAFKA_CONSUMER_MICRO_BATCH_SIZE") or 8)
            if len(batch) >= max(micro_batch_size, 1):
                active, batch = batch, []
                await dispatch_active(active)
        accepting_elapsed = time.monotonic() - started
        # At the boundary we only finish already-polled micro-batches. The
        # bounded drain keeps the benchmark's ordered commit contract aligned
        # with the image Stack's direct Kafka pipeline.
        if batch:
            active, batch = batch, []
            await dispatch_active(active)
        await drain_inflight(timeout_seconds=drain_seconds)
        total_elapsed = time.monotonic() - started
        batch_elapsed = [float(item["elapsed_ms"]) for item in batches]
        return {
            "lane": lane,
            "topic": TOPICS[lane],
            "collection": V2_COLLECTIONS[lane],
            "consumer_group": group_id,
            "pipeline_inflight_batches": pipeline_inflight_limit,
            "runtime_tuning": {
                "kafka_micro_batch_size": int(
                    os.environ.get("KAFKA_CONSUMER_MICRO_BATCH_SIZE") or 1
                ),
                "embedding_max_concurrency": int(
                    os.environ.get("EMBEDDING_MAX_CONCURRENCY") or 0
                ),
                "embedding_rate_limit_enabled": str(
                    os.environ.get("EMBEDDING_RATE_LIMIT_ENABLED") or ""
                ).lower()
                in {"1", "true", "yes", "on"},
                "embedding_lane": os.environ.get("EMBEDDING_LANE") or "",
                "embedding_text_rate": float(
                    os.environ.get("EMBEDDING_TEXT_RATE") or 0
                ),
                "embedding_text_burst": int(
                    os.environ.get("EMBEDDING_TEXT_BURST") or 0
                ),
                "embedding_batch_size": int(os.environ.get("EMBEDDING_BATCH_SIZE") or 0),
                "metadata_milvus_batch_size": int(
                    os.environ.get("DOCUMENT_BULK_MILVUS_BATCH_SIZE") or 0
                ),
                "metadata_pipeline_inflight_batches": int(
                    os.environ.get("DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES") or 1
                ),
                "fulltext_worker_concurrency": int(
                    os.environ.get("FULLTEXT_WORKER_CONCURRENCY") or 0
                ),
                "fulltext_pipeline_inflight_batches": int(
                    os.environ.get("FULLTEXT_PIPELINE_INFLIGHT_BATCHES") or 1
                ),
                "image_worker_concurrency": int(
                    os.environ.get("IMAGE_WORKER_CONCURRENCY") or 0
                ),
                "image_download_concurrency": int(
                    os.environ.get("IMAGE_DOWNLOAD_CONCURRENCY") or 0
                ),
                "image_vl_embed_concurrency": int(
                    os.environ.get("IMAGE_VL_EMBED_CONCURRENCY") or 0
                ),
                "image_text_embed_concurrency": int(
                    os.environ.get("IMAGE_TEXT_EMBED_CONCURRENCY") or 0
                ),
                "image_pipeline_inflight_batches": int(
                    os.environ.get("IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES") or 1
                ),
            },
            "accepting_elapsed_seconds": round(accepting_elapsed, 3),
            "total_elapsed_seconds": round(total_elapsed, 3),
            "counters": dict(counters),
            "batch_latency_ms": _latency_summary(batch_elapsed),
            "stage_timing_ms": timings.summary(),
            "metadata_bulk_timing": get_metadata_bulk_timing_snapshot() if lane == "metadata" else {},
        }
    finally:
        await asyncio.to_thread(consumer.close)
        await processor.close()
        await close_mongo_state_manager()


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "avg_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
    }


def _lane_main(args: argparse.Namespace) -> int:
    if not args.lane:
        raise ValueError("--lane is required for --mode lane")
    if args.out_dir is None:
        raise ValueError("--out-dir is required for --mode lane")
    _load_environment()
    _apply_runtime_environment(lane=args.lane, run_id=_run_id(args.run_id), out_dir=args.out_dir)
    result = asyncio.run(
        _run_lane(
            lane=args.lane,
            duration_seconds=args.duration_seconds,
            drain_seconds=args.drain_seconds,
        )
    )
    result["milvus_after_flush"] = _collection_snapshot(V2_COLLECTIONS[args.lane])
    result["finished_at"] = _utc_now()
    path = args.out_dir / f"{args.lane}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _written_rows(lane: str, counters: dict[str, Any]) -> int:
    if lane == "metadata":
        return int(counters.get("outcome_success") or 0)
    return int(counters.get("written_chunks" if lane == "fulltext" else "written_images") or 0)


def _build_comparison(
    lanes: dict[str, dict[str, Any]],
    settled_v2: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lane, lane_result in lanes.items():
        counters = lane_result.get("counters") or {}
        written_rows = _written_rows(lane, counters)
        rows_at_lane_finish = int(((lane_result.get("milvus_after_flush") or {}).get("row_count") or 0))
        settled_rows = int((settled_v2.get(lane) or {}).get("row_count") or rows_at_lane_finish)
        elapsed = max(float(lane_result.get("accepting_elapsed_seconds") or 0), 0.001)
        throughput = written_rows / elapsed
        baseline = BASELINE[lane]
        result[lane] = {
            "baseline_rows": baseline["rows"],
            "baseline_throughput": baseline["throughput"],
            "unit": baseline["unit"],
            "optimized_rows": written_rows,
            "window_successful_write_rows": written_rows,
            "milvus_rows_at_lane_finish": rows_at_lane_finish,
            "post_settlement_rows": settled_rows,
            "optimized_throughput": round(throughput, 3),
            "throughput_change_percent": round((throughput / float(baseline["throughput"]) - 1) * 100, 2),
            "failure_rate_percent": round(
                100 * int(counters.get("outcome_failed") or 0) / max(int(counters.get("processed_records") or 0), 1),
                3,
            ),
        }
    return result


def _stage_time_shares(lane: str, lane_result: dict[str, Any]) -> dict[str, Any]:
    """Return major-stage work shares; concurrent task time may overlap."""
    if lane == "metadata":
        stages = (lane_result.get("metadata_bulk_timing") or {}).get("stages") or {}
        totals = {
            stage: float(values.get("total_seconds") or 0.0) * 1000.0
            for stage, values in stages.items()
            if float(values.get("total_seconds") or 0.0) > 0
        }
    else:
        stage_timing = lane_result.get("stage_timing_ms") or {}
        candidates = (
            (
                "solr",
                "mmd_download",
                "parse",
                "chunk_build",
                "pipeline_embedding",
                "pipeline_milvus_write",
                "pipeline_visibility_check",
            )
            if lane == "fulltext"
            else (
                "solr_resolve",
                "download_fast",
                "download_slow",
                "feature_total",
                "milvus_document_batch",
            )
        )
        totals = {
            stage: float((stage_timing.get(stage) or {}).get("total_ms") or 0.0)
            for stage in candidates
            if float((stage_timing.get(stage) or {}).get("total_ms") or 0.0) > 0
        }
    overall = sum(totals.values())
    return {
        "basis": "major_stage_work_totals; concurrent stages may overlap",
        "total_ms": round(overall, 3),
        "stages": {
            stage: {"total_ms": round(value, 3), "share_percent": round(100 * value / overall, 3)}
            for stage, value in totals.items()
        }
        if overall
        else {},
    }


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    duration_minutes = float(report.get("duration_requested_seconds") or 0) / 60.0
    duration_label = (
        f"{duration_minutes:.0f} 分钟"
        if duration_minutes.is_integer()
        else f"{duration_minutes:.1f} 分钟"
    )
    lines = [
        f"# 三路 V2 真实链路 {duration_label} 压测报告",
        "",
        f"- 运行编号：`{report['run_id']}`",
        f"- 时间：{report['started_at']} 至 {report['finished_at']}",
        "- 链路：Kafka -> 测试 Solr -> 静态资源 -> Embedding -> Milvus V2",
        f"- 安全边界：{'复用已有隔离测试集合并增量写入，不删除既有测试数据' if report.get('incremental') else '仅删除并重建三个 V2 集合'}；历史三个核心集合未删除且行数复核一致。",
        "",
        "## 吞吐对比",
        "",
        "| 链路 | 上轮吞吐 | 本轮吞吐 | 变化 | 窗口成功写入 | 稳定行数 | 失败率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"metadata": "元数据", "fulltext": "全文切片", "image": "图片"}
    for lane in ("metadata", "fulltext", "image"):
        item = report["comparison"][lane]
        lines.append(
            f"| {labels[lane]} | {item['baseline_throughput']:.2f} {item['unit']} | "
            f"{item['optimized_throughput']:.3f} {item['unit']} | {item['throughput_change_percent']:+.2f}% | "
            f"{item['window_successful_write_rows']} | {item['post_settlement_rows']} | "
            f"{item['failure_rate_percent']:.3f}% |"
        )
    lines.extend(["", "## 分阶段耗时", ""])
    for lane in ("metadata", "fulltext", "image"):
        lane_result = report["lanes"][lane]
        lines.extend([f"### {labels[lane]}", "", "| 阶段 | 样本数 | 累积耗时(ms) | 平均(ms) | P95(ms) |", "|---|---:|---:|---:|---:|"])
        timing = lane_result.get("stage_timing_ms") or {}
        if not timing:
            lines.append("| 无可用单项阶段样本 | 0 | 0 | 0 | 0 |")
        else:
            for stage, values in timing.items():
                lines.append(
                    f"| {stage} | {values['count']} | {values['total_ms']:.3f} | "
                    f"{values['avg_ms']:.3f} | {values['p95_ms']:.3f} |"
                )
        batch = lane_result.get("batch_latency_ms") or {}
        lines.extend([
            "",
            f"Kafka 微批端到端：平均 {batch.get('avg_ms', 0):.3f} ms，"
            f"P50 {batch.get('p50_ms', 0):.3f} ms，P95 {batch.get('p95_ms', 0):.3f} ms。",
            "",
        ])
        shares = report["stage_time_shares"].get(lane) or {}
        if shares.get("stages"):
            lines.extend([
                "主要阶段累计耗时占比：",
                "",
                "| 阶段 | 累计耗时(ms) | 占比 |",
                "|---|---:|---:|",
            ])
            for stage, values in shares["stages"].items():
                lines.append(f"| {stage} | {values['total_ms']:.3f} | {values['share_percent']:.3f}% |")
            lines.append("")
    lines.extend([
        "## 口径",
        "",
        "吞吐只统计 300 秒接收窗口内成功写入的行数；窗口结束后只收尾已接收微批，不再读取新的 Kafka 消息。",
        "稳定行数在三个 worker 退出后的稳定期再次读取。图片使用同文献完整快照写入，后续快照会清理该文献已失效的旧图片，因此它可能小于窗口内成功写入次数。",
        "阶段累计耗时是任务级 CPU/IO 等待时间的累加。阶段占比用于定位资源消耗；存在并发时，不能把它当作 300 秒墙钟时间的直接拆分。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_main(args: argparse.Namespace) -> int:
    if args.duration_seconds <= 0 or args.drain_seconds <= 0 or args.post_run_settle_seconds < 0:
        raise ValueError("duration and drain seconds must be positive; post-run settle seconds cannot be negative")
    if not args.execute:
        raise ValueError("--execute is required for the benchmark")
    if args.incremental and not args.collection_suffix:
        raise ValueError("--incremental requires --collection-suffix for an isolated target set")
    _load_environment()
    run_id = _run_id(args.run_id)
    # Child lane processes run with SERVICE_ROOT as cwd.  Resolve the output
    # directory once so their JSON artifacts land beside the parent logs even
    # when the caller supplies a relative path.
    out_dir = (args.out_dir or SERVICE_ROOT / "out" / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    for lane in V2_COLLECTIONS:
        _apply_runtime_environment(lane=lane, run_id=run_id, out_dir=out_dir)
    topic_availability = _topic_availability()
    reset = _verify_incremental_targets() if args.incremental else _reset_v2_collections(execute=True)
    started_at = _utc_now()
    processes: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    try:
        for lane in V2_COLLECTIONS:
            stdout = (out_dir / f"{lane}.stdout.log").open("w", encoding="utf-8")
            stderr = (out_dir / f"{lane}.stderr.log").open("w", encoding="utf-8")
            logs[lane] = (stdout, stderr)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode", "lane",
                "--lane", lane,
                "--duration-seconds", str(args.duration_seconds),
                "--drain-seconds", str(args.drain_seconds),
                "--run-id", run_id,
                "--collection-suffix", args.collection_suffix,
                "--out-dir", str(out_dir),
            ]
            processes[lane] = subprocess.Popen(command, cwd=SERVICE_ROOT, stdout=stdout, stderr=stderr)
        max_wait = args.duration_seconds + args.drain_seconds + 120
        exit_codes: dict[str, int | None] = {}
        for lane, process in processes.items():
            try:
                exit_codes[lane] = process.wait(timeout=max_wait)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_codes[lane] = None
        lane_results = {lane: _read_json(out_dir / f"{lane}.json") for lane in V2_COLLECTIONS}
        final_v2 = {lane: _collection_snapshot(collection) for lane, collection in V2_COLLECTIONS.items()}
        if args.post_run_settle_seconds:
            time.sleep(args.post_run_settle_seconds)
        settled_v2 = {lane: _collection_snapshot(collection) for lane, collection in V2_COLLECTIONS.items()}
        final_core = _snapshot_core_collections()
        report = {
            "run_id": run_id,
            "incremental": bool(args.incremental),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_requested_seconds": args.duration_seconds,
            "drain_seconds": args.drain_seconds,
            "post_run_settle_seconds": args.post_run_settle_seconds,
            "target_collections": dict(V2_COLLECTIONS),
            "test_solr": {"paper": TEST_SOLR_PAPER_URL, "image": TEST_SOLR_IMAGE_URL},
            "topic_availability_before_run": topic_availability,
            "reset": reset,
            "exit_codes": exit_codes,
            "lanes": lane_results,
            "final_v2_collections": final_v2,
            "settled_v2_collections": settled_v2,
            "protected_historical_core_rows_after": final_core,
            "comparison": _build_comparison(lane_results, settled_v2),
            "stage_time_shares": {
                lane: _stage_time_shares(lane, result) for lane, result in lane_results.items()
            },
        }
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_markdown_report(out_dir / "report.md", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(code == 0 for code in exit_codes.values()) else 1
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for stdout, stderr in logs.values():
            stdout.close()
            stderr.close()


def main() -> int:
    args = parse_args()
    _configure_collection_targets(args.collection_suffix)
    if args.mode == "lane":
        return _lane_main(args)
    _load_environment()
    if args.mode == "reset":
        print(
            json.dumps(
                _reset_v2_collections(execute=args.execute, lane=args.lane),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return _run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
