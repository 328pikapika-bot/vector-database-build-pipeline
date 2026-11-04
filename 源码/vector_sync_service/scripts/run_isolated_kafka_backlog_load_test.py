#!/usr/bin/env python3
"""Run a 10-minute isolated Kafka backlog -> Mongo -> Milvus load test."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from confluent_kafka import Consumer, TopicPartition
from pymilvus import MilvusClient
from pymongo import MongoClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "vector_sync_service"
VENV_PYTHON = SERVICE_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_RUN_PREFIX = "codex_kafka_backlog_loadtest"
DEFAULT_TARGET_HOST = "127.0.0.1"
KAFKA_SASL_USERNAME = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.environ.get("KAFKA_SASL_PASSWORD", "")
DEFAULT_TARGET_PORT = 19533
DEFAULT_TOPIC = "VECTOR_JOURNAL_ARTICLE"
DEFAULT_BOOTSTRAP = "127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092"
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "")
DEFAULT_EMBEDDING_URL = "http://127.0.0.1:16703/v1/embeddings"
DEFAULT_IMAGE_CHAT_URL = "http://127.0.0.1:50019/v1/chat/completions"
DEFAULT_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated Kafka backlog load test against fresh scientific Milvus collections.")
    parser.add_argument("--observe-seconds", type=int, default=600, help="How long to observe ingestion.")
    parser.add_argument("--monitor-interval", type=int, default=30, help="Seconds between status snapshots.")
    parser.add_argument(
        "--consume-limit",
        type=int,
        default=100,
        help="Maximum number of real historical Kafka messages to process; 0 means unlimited.",
    )
    parser.add_argument("--port", type=int, default=8012, help="Local uvicorn port.")
    parser.add_argument("--run-id", default="", help="Optional run id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_prereqs()

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{DEFAULT_RUN_PREFIX}_{run_id}"
    run_dir = SERVICE_ROOT / "out" / run_name
    runtime_dir = SERVICE_ROOT / ".runtime" / run_name
    upload_dir = SERVICE_ROOT / "upload" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    report_path = run_dir / "report.json"
    service_stdout = run_dir / "service.stdout.log"
    service_stderr = run_dir / "service.stderr.log"

    run_slug = _safe_run_slug(run_id)
    kafka_group = f"codex_backlog_loadtest_{run_slug}"
    mongo_db = f"nstl_vs_bl_{run_slug}"
    collection_names = {
        "documents": f"scientific_documents_{run_slug}",
        "chunks": f"scientific_fulltext_chunks_{run_slug}",
        "images": f"scientific_images_{run_slug}",
    }

    status: dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "phase": "initializing",
        "started_at": iso_now(),
        "observe_seconds": args.observe_seconds,
        "monitor_interval": args.monitor_interval,
        "consume_limit": args.consume_limit,
        "local_service_port": args.port,
        "milvus": {
            "host": DEFAULT_TARGET_HOST,
            "port": DEFAULT_TARGET_PORT,
            "collections": collection_names,
        },
        "mongo_db": mongo_db,
        "kafka_group_id": kafka_group,
        "timeline": [],
        "artifacts": {
            "status_path": str(status_path),
            "report_path": str(report_path),
            "service_stdout": str(service_stdout),
            "service_stderr": str(service_stderr),
        },
    }
    write_json(status_path, status)

    service_process: subprocess.Popen[str] | None = None
    try:
        reset_mongo_database(mongo_db)
        reset_milvus_collections(collection_names)
        topic_offsets = get_topic_offsets(DEFAULT_TOPIC)
        status["topic_offsets_before"] = topic_offsets
        status["phase"] = "starting_service"
        write_json(status_path, status)

        service_process = start_service(
            port=args.port,
            kafka_group=kafka_group,
            mongo_db=mongo_db,
            runtime_dir=runtime_dir,
            upload_dir=upload_dir,
            stdout_path=service_stdout,
            stderr_path=service_stderr,
            collection_names=collection_names,
            consume_limit=args.consume_limit,
        )

        health = wait_for_service(args.port, timeout_seconds=180)
        status["service"] = {
            "pid": service_process.pid,
            "health": health,
        }
        status["phase"] = "collecting_initial_counts"
        write_json(status_path, status)

        initial_rows = get_milvus_row_counts(collection_names)
        status["initial_rows"] = initial_rows
        write_json(status_path, status)

        status["phase"] = "observing"
        write_json(status_path, status)

        end_time = time.time() + args.observe_seconds
        while time.time() < end_time:
            snapshot = collect_snapshot(
                args.port,
                mongo_db,
                collection_names,
            )
            snapshot["elapsed_seconds"] = int(args.observe_seconds - max(end_time - time.time(), 0))
            snapshot["topic_offsets"] = get_topic_offsets(DEFAULT_TOPIC)
            snapshot["consumer_group_offsets"] = get_group_offsets(DEFAULT_TOPIC, kafka_group)
            status["timeline"].append(snapshot)
            write_json(status_path, status)
            sleep_seconds = min(args.monitor_interval, max(int(end_time - time.time()), 0))
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)

        final_snapshot = collect_snapshot(args.port, mongo_db, collection_names)
        final_snapshot["topic_offsets"] = get_topic_offsets(DEFAULT_TOPIC)
        final_snapshot["consumer_group_offsets"] = get_group_offsets(DEFAULT_TOPIC, kafka_group)
        status["timeline"].append(final_snapshot)
        status["phase"] = "finalizing"
        write_json(status_path, status)

        final_rows = final_snapshot["milvus"]["row_counts"]
        row_delta = {
            name: int(final_rows.get(name, 0)) - int(initial_rows.get(name, 0))
            for name in collection_names
        }

        consumed_delta = 0
        group_offsets = final_snapshot.get("consumer_group_offsets") or {}
        for partition_info in group_offsets.get("partitions", []):
            consumed_delta += max(int(partition_info.get("current_offset", 0)) - int(partition_info.get("low", 0)), 0)

        report = {
            "run_name": run_name,
            "run_id": run_id,
            "started_at": status["started_at"],
            "finished_at": iso_now(),
            "observe_seconds": args.observe_seconds,
            "target": {
                "attu_url": "http://127.0.0.1:18002/#/connect",
                "milvus_host": DEFAULT_TARGET_HOST,
                "milvus_port": DEFAULT_TARGET_PORT,
                "collections": collection_names,
            },
            "isolation": {
                "mongo_db": mongo_db,
                "kafka_group_id": kafka_group,
            },
            "service": {
                "local_port": args.port,
                "pid": service_process.pid if service_process is not None else None,
            },
            "kafka": {
                "topic_offsets_before": topic_offsets,
                "topic_offsets_after": final_snapshot.get("topic_offsets"),
                "consumer_group_offsets": final_snapshot.get("consumer_group_offsets"),
                "estimated_messages_consumed_from_backlog": consumed_delta,
            },
            "counts": {
                "initial_rows": initial_rows,
                "final_rows": final_rows,
                "row_delta": row_delta,
                "total_inserted_rows": sum(row_delta.values()),
            },
            "mongo": final_snapshot["mongo"],
            "health": final_snapshot["health"],
            "timeline": status["timeline"],
            "artifacts": status["artifacts"],
        }
        write_json(report_path, report)
        status["phase"] = "completed"
        status["report_path"] = str(report_path)
        write_json(status_path, status)
        return 0
    except Exception as error:
        status["phase"] = "failed"
        status["failed_at"] = iso_now()
        status["error"] = f"{type(error).__name__}: {error}"
        write_json(status_path, status)
        raise
    finally:
        stop_service(service_process)


def ensure_prereqs() -> None:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"venv python not found: {VENV_PYTHON}")


def _safe_run_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not normalized:
        normalized = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
    return normalized[:32]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def kafka_config(group_id: str, *, auto_offset_reset: str = "earliest") -> dict[str, Any]:
    return {
        "bootstrap.servers": DEFAULT_BOOTSTRAP,
        "group.id": group_id,
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "PLAIN",
        "sasl.username": KAFKA_SASL_USERNAME,
        "sasl.password": KAFKA_SASL_PASSWORD,
        "enable.auto.commit": False,
        "auto.offset.reset": auto_offset_reset,
    }


def get_topic_offsets(topic: str) -> dict[str, Any]:
    consumer = Consumer(kafka_config("codex_backlog_topic_probe", auto_offset_reset="earliest"))
    try:
        md = consumer.list_topics(topic, timeout=15)
        topic_meta = md.topics.get(topic)
        if topic_meta is None:
            raise ValueError(f"Topic {topic} metadata not found")
        partitions = []
        for partition_id in sorted(topic_meta.partitions.keys()):
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, partition_id),
                timeout=10,
                cached=False,
            )
            partitions.append(
                {
                    "partition": partition_id,
                    "low": low,
                    "high": high,
                    "available": high - low,
                }
            )
        return {"topic": topic, "partitions": partitions}
    finally:
        consumer.close()


def get_group_offsets(topic: str, group_id: str) -> dict[str, Any]:
    consumer = Consumer(kafka_config(group_id, auto_offset_reset="earliest"))
    try:
        md = consumer.list_topics(topic, timeout=15)
        topic_meta = md.topics.get(topic)
        if topic_meta is None:
            raise ValueError(f"Topic {topic} metadata not found")

        partitions = [TopicPartition(topic, partition_id) for partition_id in sorted(topic_meta.partitions.keys())]
        committed = consumer.committed(partitions, timeout=10)
        results = []
        for item in committed:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, item.partition),
                timeout=10,
                cached=False,
            )
            current_offset = int(item.offset if item.offset is not None and item.offset >= 0 else low)
            lag = max(high - current_offset, 0)
            results.append(
                {
                    "partition": item.partition,
                    "low": low,
                    "high": high,
                    "current_offset": current_offset,
                    "lag": lag,
                }
            )
        return {"topic": topic, "group_id": group_id, "partitions": results}
    finally:
        consumer.close()


def start_service(
    *,
    port: int,
    kafka_group: str,
    mongo_db: str,
    runtime_dir: Path,
    upload_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    collection_names: dict[str, str],
    consume_limit: int,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "KAFKA_BOOTSTRAP_SERVERS": DEFAULT_BOOTSTRAP,
            "KAFKA_TOPIC": DEFAULT_TOPIC,
            "KAFKA_GROUP_ID": kafka_group,
            "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
            "KAFKA_SASL_MECHANISM": "PLAIN",
            "KAFKA_SASL_USERNAME": KAFKA_SASL_USERNAME,
            "KAFKA_SASL_PASSWORD": KAFKA_SASL_PASSWORD,
            "KAFKA_DUMP_FILE": str(runtime_dir / "kafka_messages.jsonl"),
            "KAFKA_ERROR_DUMP_FILE": str(runtime_dir / "kafka_messages_error.jsonl"),
            "KAFKA_PROCESSED_DUMP_FILE": str(runtime_dir / "kafka_messages_processed.jsonl"),
            "MONGO_URI": DEFAULT_MONGO_URI,
            "MONGO_DB": mongo_db,
            "MILVUS_HOST": DEFAULT_TARGET_HOST,
            "MILVUS_PORT": str(DEFAULT_TARGET_PORT),
            "MILVUS_COLLECTION": collection_names["documents"],
            "DOCUMENT_COLLECTION_NAME": collection_names["documents"],
            "FULLTEXT_CHUNK_COLLECTION": collection_names["chunks"],
            "FULLTEXT_CHUNK_COLLECTION_NAME": collection_names["chunks"],
            "IMAGE_COLLECTION_NAME": collection_names["images"],
            "INIT_VECTOR_COLLECTIONS_ON_STARTUP": "true",
            "DOCUMENT_INDEX_TYPE": "IVFPQ",
            "DOCUMENT_INDEX_METRIC_TYPE": "COSINE",
            "DOCUMENT_IVFPQ_NLIST": "64",
            "DOCUMENT_IVFPQ_M": "64",
            "DOCUMENT_IVFPQ_NBITS": "8",
            "DOCUMENT_IVFPQ_NPROBE": "16",
            "FULLTEXT_CHUNK_INDEX_TYPE": "IVFPQ",
            "FULLTEXT_CHUNK_INDEX_METRIC_TYPE": "COSINE",
            "FULLTEXT_CHUNK_IVFPQ_NLIST": "64",
            "FULLTEXT_CHUNK_IVFPQ_M": "64",
            "FULLTEXT_CHUNK_IVFPQ_NBITS": "8",
            "FULLTEXT_CHUNK_IVFPQ_NPROBE": "16",
            "IMAGE_DENSE_INDEX_TYPE": "IVFPQ",
            "IMAGE_DENSE_INDEX_METRIC_TYPE": "IP",
            "IMAGE_DENSE_IVFPQ_NLIST": "64",
            "IMAGE_DENSE_IVFPQ_M": "64",
            "IMAGE_DENSE_IVFPQ_NBITS": "8",
            "IMAGE_DENSE_IVFPQ_NPROBE": "16",
            "EMBEDDING_API_URL": DEFAULT_EMBEDDING_URL,
            "EMBEDDING_API_KEY": DEFAULT_API_KEY,
            "EMBEDDING_MODEL": "qwen3-embed",
            "AUTO_PUBLISH_ENABLED": "false",
            "KAFKA_CONSUMER_ENABLED": "true",
            "KAFKA_CONSUME_LIMIT": str(max(int(consume_limit), 0)),
            "KAFKA_BACKPRESSURE_ENABLED": "false",
            "COMPENSATOR_ENABLED": "false",
            "FULLTEXT_WORKER_ENABLED": "true",
            "FULLTEXT_TASK_UPSERT_ENABLED": "true",
            "IMAGE_WORKER_ENABLED": "true",
            "IMAGE_TASK_UPSERT_ENABLED": "true",
            "IMAGE_FEATURE_PIPELINE_ENABLED": "true",
            "IMAGE_MILVUS_WRITE_ENABLED": "true",
            "IMAGE_VL_CHAT_URL": DEFAULT_IMAGE_CHAT_URL,
            "IMAGE_VL_CHAT_MODEL": "qwen3.5-plus",
            "IMAGE_VL_EMBED_URL": DEFAULT_EMBEDDING_URL,
            "IMAGE_VL_EMBED_MODEL": "qwen3-vl-embed",
            "IMAGE_TEXT_EMBED_URL": DEFAULT_EMBEDDING_URL,
            "IMAGE_TEXT_EMBED_MODEL": "qwen3-embed",
            "IMAGE_MODEL_API_KEY": DEFAULT_API_KEY,
            "ADMIN_AUTH_ENABLED": "false",
            "UPLOAD_DIR": str(upload_dir),
        }
    )
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(SERVICE_ROOT),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )


def stop_service(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_for_service(port: int, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    last_error = "unknown"
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}"
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(2)
    raise TimeoutError(f"service did not become healthy on {url}: {last_error}")


def collect_snapshot(port: int, mongo_db: str, collections: dict[str, str]) -> dict[str, Any]:
    return {
        "ts": iso_now(),
        "health": safe_health(port),
        "milvus": {
            "row_counts": get_milvus_row_counts(collections),
        },
        "mongo": get_mongo_summary(mongo_db),
    }


def safe_health(port: int) -> dict[str, Any]:
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5)
        return {
            "status_code": response.status_code,
            "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else {},
        }
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def get_milvus_row_counts(collections: dict[str, str]) -> dict[str, int]:
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=20)
    try:
        counts: dict[str, int] = {}
        for label, name in collections.items():
            if not client.has_collection(name):
                counts[label] = 0
                continue
            client.flush(collection_name=name)
            stats = client.get_collection_stats(collection_name=name)
            counts[label] = int(stats.get("row_count", 0))
        return counts
    finally:
        client.close()


def get_mongo_summary(mongo_db: str) -> dict[str, Any]:
    client = MongoClient(DEFAULT_MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000)
    try:
        db = client[mongo_db]
        return {
            "status_counts": group_counts(db["doc_sync_status"], "overall_status"),
            "fulltext_task_counts": group_counts(db["doc_fulltext_tasks"], "status"),
            "image_task_counts": group_counts(db["doc_image_tasks"], "status"),
            "failure_counts": group_counts(db["doc_sync_failures"], "status"),
        }
    finally:
        client.close()


def group_counts(collection: Any, field_name: str) -> dict[str, int]:
    rows = collection.aggregate(
        [
            {"$group": {"_id": f"${field_name}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
    )
    return {str(row.get("_id") or "UNKNOWN"): int(row.get("count") or 0) for row in rows}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_mongo_database(mongo_db: str) -> None:
    client = MongoClient(DEFAULT_MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000)
    try:
        client.drop_database(mongo_db)
    finally:
        client.close()


def reset_milvus_collections(collections: dict[str, str]) -> None:
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=20)
    try:
        for name in collections.values():
            if client.has_collection(name):
                client.drop_collection(collection_name=name)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
