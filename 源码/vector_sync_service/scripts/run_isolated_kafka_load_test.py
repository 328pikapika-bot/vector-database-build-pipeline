#!/usr/bin/env python3
"""Run an isolated 10-minute Kafka -> Mongo -> Milvus load test."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from confluent_kafka import Consumer, ConsumerGroupTopicPartitions, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymongo import MongoClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "vector_sync_service"
VENV_PYTHON = SERVICE_ROOT / ".venv" / "Scripts" / "python.exe"
load_dotenv(SERVICE_ROOT / ".env", override=False)
DEFAULT_REAL_PAYLOAD_SOURCE = REPO_ROOT / "bakkafkadata" / "31" / "kafka_messages.jsonl"
DEFAULT_RUN_PREFIX = "codex_kafka_loadtest"
DEFAULT_TARGET_HOST = "127.0.0.1"
KAFKA_SASL_USERNAME = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.environ.get("KAFKA_SASL_PASSWORD", "")
DEFAULT_TARGET_PORT = 19533
PRODUCTION_TOPIC = "VECTOR_JOURNAL_ARTICLE"
INCREMENTAL_TOPIC = "VECTOR_JOURNAL_ARTICLE_INCREMENTAL"
READ_ONLY_TOPICS = {PRODUCTION_TOPIC, INCREMENTAL_TOPIC}
DEFAULT_TOPIC = "VECTOR_JOURNAL_ARTICLE_IVFPQ_E2E_TEST"
DEFAULT_BOOTSTRAP = "127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092"
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "")
DEFAULT_EMBEDDING_URL = "http://127.0.0.1:16703/v1/embeddings"
DEFAULT_IMAGE_CHAT_URL = "http://127.0.0.1:50019/v1/chat/completions"
DEFAULT_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
TEST_IMAGE_SCHEMA_MODE = "history_compatible_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated real-payload Kafka -> Mongo -> Milvus end-to-end test.")
    parser.add_argument("--messages", type=int, default=5, help="Number of real historical Kafka messages to replay.")
    parser.add_argument("--observe-seconds", type=int, default=600, help="How long to observe ingestion after publish.")
    parser.add_argument("--monitor-interval", type=int, default=30, help="Seconds between status snapshots.")
    parser.add_argument("--port", type=int, default=8011, help="Local uvicorn port.")
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_REAL_PAYLOAD_SOURCE, help="Historical real Kafka payload JSONL source.")
    parser.add_argument(
        "--history-root",
        type=Path,
        default=SERVICE_ROOT / ".runtime",
        help="Additional runtime directories containing historical Kafka payload dumps.",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Dedicated Kafka topic for this test. Production topic is rejected.")
    parser.add_argument(
        "--consume-existing-topic",
        action="store_true",
        help="Consume an existing topic with a new isolated group; never create or publish to the topic.",
    )
    parser.add_argument("--run-id", default="", help="Optional run id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_prereqs(None if args.consume_existing_topic else args.source_jsonl)
    topic = args.topic.strip()
    if not topic:
        raise ValueError("--topic must not be empty")
    if topic in READ_ONLY_TOPICS and not args.consume_existing_topic:
        raise ValueError("refusing to publish test payloads to a production Kafka topic")
    if not args.consume_existing_topic and args.messages <= 0:
        raise ValueError("--messages must be greater than zero for a real-payload test")

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
    kafka_group = f"codex_e2e_test_{run_slug}"
    mongo_db = f"nstl_vs_e2e_{run_slug}_test"
    collection_names = {
        "documents": f"scientific_documents_{run_slug}_test",
        "chunks": f"scientific_fulltext_chunks_{run_slug}_test",
        "images": f"scientific_images_{run_slug}_test",
    }

    status: dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "phase": "initializing",
        "started_at": iso_now(),
        "messages": args.messages,
        "consume_existing_topic": args.consume_existing_topic,
        "source_jsonl": str(args.source_jsonl),
        "kafka_topic": topic,
        "image_schema_mode": TEST_IMAGE_SCHEMA_MODE,
        "observe_seconds": args.observe_seconds,
        "monitor_interval": args.monitor_interval,
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
        assert_isolated_targets_absent(mongo_db, collection_names)
        topic_tail = get_topic_watermarks(topic)
        status["topic_baseline"] = topic_tail
        if not args.consume_existing_topic:
            ensure_test_topic(topic)
            topic_tail = get_topic_watermarks(topic)
            status["topic_baseline"] = topic_tail
            status["phase"] = "committing_group_to_tail"
            write_json(status_path, status)
            commit_group_to_offset(topic, kafka_group, topic_tail["partitions"][0]["high"])

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
            topic=topic,
            consume_limit=0 if args.consume_existing_topic else args.messages,
            existing_topic=args.consume_existing_topic,
        )

        health = (
            wait_for_image_worker(service_process, collection_names["images"], timeout_seconds=180)
            if args.consume_existing_topic
            else wait_for_service(args.port, timeout_seconds=180)
        )
        status["service"] = {
            "pid": service_process.pid,
            "health": health,
        }
        status["phase"] = "waiting_for_empty_collections"
        write_json(status_path, status)

        initial_rows = get_milvus_row_counts(collection_names)
        status["initial_rows"] = initial_rows
        write_json(status_path, status)

        if args.consume_existing_topic:
            source_summary = {
                "selection": "existing_topic_from_new_group_earliest",
                "topic": topic,
            }
            publish_meta = {
                "messages_published": 0,
                "publish_duration_seconds": 0,
                "original_document_ids": [],
            }
            status["payload_source"] = source_summary
        else:
            source_payloads, source_summary = load_real_payloads(
                args.source_jsonl,
                history_root=args.history_root,
                limit=args.messages,
            )
            status["payload_source"] = source_summary
            status["phase"] = "publishing"
            write_json(status_path, status)
            publish_meta = produce_messages(
                source_payloads=source_payloads,
                topic=topic,
                run_id=run_id,
            )
            status["publish"] = publish_meta
        status["phase"] = "observing"
        status["published_at"] = iso_now()
        write_json(status_path, status)

        end_time = time.time() + args.observe_seconds
        while time.time() < end_time:
            snapshot = collect_snapshot(
                args.port,
                mongo_db,
                collection_names,
                worker_process=service_process if args.consume_existing_topic else None,
            )
            snapshot["elapsed_seconds"] = int(args.observe_seconds - max(end_time - time.time(), 0))
            status["timeline"].append(snapshot)
            write_json(status_path, status)
            sleep_seconds = min(args.monitor_interval, max(int(end_time - time.time()), 0))
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)

        final_snapshot = collect_snapshot(
            args.port,
            mongo_db,
            collection_names,
            worker_process=service_process if args.consume_existing_topic else None,
        )
        status["timeline"].append(final_snapshot)
        status["phase"] = "finalizing"
        write_json(status_path, status)

        final_rows = final_snapshot["milvus"]["row_counts"]
        initial_rows = status.get("initial_rows", {})
        row_delta = {
            name: int(final_rows.get(name, 0)) - int(initial_rows.get(name, 0))
            for name in collection_names
        }
        report = {
            "run_name": run_name,
            "run_id": run_id,
            "started_at": status["started_at"],
            "finished_at": iso_now(),
            "observe_seconds": args.observe_seconds,
            "payload_source": {
                "source_jsonl": None if args.consume_existing_topic else str(args.source_jsonl),
                "selection": source_summary.get("selection", "messages containing both hasFullTextFile and hasPictureFile"),
                "original_document_ids": publish_meta["original_document_ids"],
                **source_summary,
            },
            "image_schema_mode": TEST_IMAGE_SCHEMA_MODE,
            "messages_published": publish_meta["messages_published"],
            "publish_duration_seconds": publish_meta["publish_duration_seconds"],
            "target": {
                "attu_url": "http://127.0.0.1:18002/#/connect",
                "milvus_host": DEFAULT_TARGET_HOST,
                "milvus_port": DEFAULT_TARGET_PORT,
                "collections": collection_names,
            },
            "isolation": {
                "mongo_db": mongo_db,
                "kafka_group_id": kafka_group,
                "topic": topic,
                "topic_baseline_watermarks": topic_tail["partitions"],
            },
            "service": {
                "local_port": args.port,
                "pid": service_process.pid if service_process is not None else None,
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


def ensure_prereqs(source_path: Path | None) -> None:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"venv python not found: {VENV_PYTHON}")
    if source_path is not None and not source_path.exists():
        raise FileNotFoundError(f"historical Kafka payload source not found: {source_path}")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_run_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not normalized:
        normalized = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
    return normalized[:32]


def kafka_config(group_id: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": DEFAULT_BOOTSTRAP,
        "group.id": group_id,
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "PLAIN",
        "sasl.username": KAFKA_SASL_USERNAME,
        "sasl.password": KAFKA_SASL_PASSWORD,
        "enable.auto.commit": False,
        "auto.offset.reset": "latest",
    }


def kafka_admin_config() -> dict[str, str]:
    return {
        "bootstrap.servers": DEFAULT_BOOTSTRAP,
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "PLAIN",
        "sasl.username": KAFKA_SASL_USERNAME,
        "sasl.password": KAFKA_SASL_PASSWORD,
    }


def ensure_test_topic(topic: str) -> None:
    """Create the isolated topic once, without altering an existing topic."""
    consumer = Consumer(kafka_config("codex_e2e_topic_probe"))
    try:
        metadata = consumer.list_topics(topic=topic, timeout=15)
        if topic in metadata.topics and not metadata.topics[topic].error:
            return
    finally:
        consumer.close()

    admin = AdminClient(kafka_admin_config())
    futures = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)], request_timeout=20)
    try:
        futures[topic].result()
    except Exception as error:  # Kafka may report an expected TopicAlreadyExists race.
        if "TOPIC_ALREADY_EXISTS" not in str(error).upper():
            raise


def get_topic_watermarks(topic: str) -> dict[str, Any]:
    consumer = Consumer(kafka_config("codex_tail_probe"))
    try:
        metadata = consumer.list_topics(topic=topic, timeout=15)
        topic_metadata = metadata.topics.get(topic)
        if topic_metadata is None or topic_metadata.error:
            raise RuntimeError(f"Kafka topic is unavailable: {topic}")
        partitions = []
        for partition_id in sorted(topic_metadata.partitions):
            partition = TopicPartition(topic, partition_id)
            low, high = consumer.get_watermark_offsets(partition, timeout=10, cached=False)
            partitions.append({"partition": partition_id, "low": low, "high": high})
        return {"topic": topic, "partitions": partitions}
    finally:
        consumer.close()


def commit_group_to_offset(topic: str, group_id: str, offset: int) -> None:
    admin = AdminClient(kafka_admin_config())
    request = [ConsumerGroupTopicPartitions(group_id, [TopicPartition(topic, 0, offset)])]
    futures = admin.alter_consumer_group_offsets(request, request_timeout=15)
    for future in futures.values():
        future.result()


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
    topic: str,
    consume_limit: int,
    existing_topic: bool = False,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "KAFKA_BOOTSTRAP_SERVERS": DEFAULT_BOOTSTRAP,
            "KAFKA_TOPIC": topic,
            "KAFKA_GROUP_ID": kafka_group,
            "KAFKA_IMAGE_GROUP_ID": kafka_group,
            "KAFKA_TARGET_VERSIONED_GROUP_ENABLED": "false",
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
            "COLLECTION_AUTO_CREATE_ON_EMPTY": "true" if existing_topic else "false",
            "COLLECTION_TARGET_STRICT": "true",
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
            "KAFKA_CONSUME_LIMIT": str(consume_limit),
            "KAFKA_BACKPRESSURE_ENABLED": "true" if existing_topic else "false",
            "KAFKA_MAX_POLL_INTERVAL_MS": "1800000",
            "BATCH_MAX_SIZE": "8",
            "KAFKA_CONSUMER_MICRO_BATCH_SIZE": "8",
            "KAFKA_CONSUMER_TIME_BUDGET_MS": "60000",
            "COMPENSATOR_ENABLED": "false",
            # This run measures the image path; the isolated topic still
            # carries real fulltext metadata, but no fulltext worker is needed
            # for the image throughput/field-integrity result.
            "FULLTEXT_WORKER_ENABLED": "false",
            "SYNC_WORKER_ROLE": "image" if existing_topic else "",
            "FULLTEXT_TASK_UPSERT_ENABLED": "true",
            "IMAGE_WORKER_ENABLED": "true",
            "IMAGE_TASK_UPSERT_ENABLED": "true",
            "IMAGE_FEATURE_PIPELINE_ENABLED": "true",
            "IMAGE_MILVUS_WRITE_ENABLED": "true",
            "IMAGE_VECTOR_SCHEMA_MODE": "history_compatible_v2",
            "IMAGE_VL_CHAT_URL": DEFAULT_IMAGE_CHAT_URL,
            "IMAGE_VL_CHAT_MODEL": "qwen3.5-plus",
            "IMAGE_VL_EMBED_URL": DEFAULT_EMBEDDING_URL,
            "IMAGE_VL_EMBED_MODEL": "qwen3-vl-embed",
            "IMAGE_TEXT_EMBED_URL": DEFAULT_EMBEDDING_URL,
            "IMAGE_TEXT_EMBED_MODEL": "qwen3-embed",
            "IMAGE_MODEL_API_KEY": DEFAULT_API_KEY,
            "IMAGE_MMD_HOT_PATH_MODE": "required",
            "IMAGE_TEXT_INDEX_POLICY": "verified_only",
            "IMAGE_MMD_CONTEXT_REQUIRED_FOR_TEXT_INDEX": "true",
            "IMAGE_ALLOW_DOCUMENT_TITLE_CONTEXT_FALLBACK": "false",
            "IMAGE_TASK_MAX_ATTEMPTS": "3",
            "IMAGE_MMD_DOWNLOAD_MAX_RETRIES": "0",
            "IMAGE_WORKER_BATCH_SIZE": "256",
            "IMAGE_WORKER_CONCURRENCY": "48",
            "IMAGE_DOWNLOAD_CONCURRENCY": "48",
            "IMAGE_VL_EMBED_CONCURRENCY": "32",
            "IMAGE_MODEL_TOTAL_CONCURRENCY": "32",
            "IMAGE_TEXT_EMBED_CONCURRENCY": "20",
            "IMAGE_TEXT_EMBED_BATCH_SIZE": "32",
            "IMAGE_TEXT_EMBED_MAX_IN_FLIGHT_BATCHES": "8",
            "IMAGE_MILVUS_WRITE_CONCURRENCY": "8",
            "IMAGE_MILVUS_WRITE_BATCH_SIZE": "64",
            "IMAGE_MILVUS_WRITE_MODE": "upsert",
            "IMAGE_DELETE_IMAGES_ENABLED": "false",
            "IMAGE_VISIBILITY_CHECK_ENABLED": "false",
            "IMAGE_DIRECT_STALE_CLEANUP_ENABLED": "false",
            "ATTACHMENT_PREFIX": "http://127.0.0.1:12001/",
            "SOLR_PAPER_URL": "http://solr-bak1.example.invalid:8984/solr/paper6.6",
            "SOLR_IMAGE_URL": "http://solr-bak1.example.invalid:8984/solr/standard6.6",
            "SOLR_REQUEST_TIMEOUT_SEC": "20",
            "SOLR_IMAGE_ROWS": "1000",
            "SOLR_RESOLVE_CONCURRENCY": "24",
            "ADMIN_AUTH_ENABLED": "false",
            "UPLOAD_DIR": str(upload_dir),
        }
    )
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = (
        [str(VENV_PYTHON), "-m", "app.cli.image_worker"]
        if existing_topic
        else [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    )
    return subprocess.Popen(
        command,
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


def wait_for_image_worker(
    process: subprocess.Popen[str],
    image_collection: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=20)
    try:
        while time.time() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(f"image worker exited during startup with code {exit_code}")
            if client.has_collection(image_collection):
                return {"mode": "image_worker_process", "running": True, "collection_ready": True}
            time.sleep(2)
    finally:
        client.close()
    raise TimeoutError(f"image worker did not create collection within {timeout_seconds}s: {image_collection}")


def load_real_payloads(
    source_path: Path,
    *,
    history_root: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load unique real article payloads, then cycle them for a larger controlled run."""
    source_paths = [source_path]
    if history_root.exists():
        source_paths.extend(history_root.rglob("kafka_messages.jsonl"))

    unique_payloads: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()
    scanned_files: set[Path] = set()
    for candidate_path in source_paths:
        resolved_path = candidate_path.resolve()
        if resolved_path in scanned_files or not resolved_path.is_file():
            continue
        scanned_files.add(resolved_path)
        with resolved_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not payload_has_fulltext_and_images(payload):
                    continue
                payload_key = payload_attachment_key(payload)
                if not payload_key or payload_key in seen_payloads:
                    continue
                seen_payloads.add(payload_key)
                payload["_e2e_source_line"] = line_number
                payload["_e2e_source_path"] = str(resolved_path)
                unique_payloads.append(payload)

    if not unique_payloads:
        raise RuntimeError("no historical messages with both fulltext and images were found")
    selected = [unique_payloads[index % len(unique_payloads)] for index in range(limit)]
    return selected, {
        "history_files_scanned": len(scanned_files),
        "unique_real_payloads": len(unique_payloads),
        "replayed_payloads": limit,
        "repeated_payloads": max(limit - len(unique_payloads), 0),
    }


def payload_has_fulltext_and_images(payload: dict[str, Any]) -> bool:
    for item in payload.get("data") or []:
        op = item.get("op") if isinstance(item, dict) else None
        if isinstance(op, dict) and op.get("hasFullTextFile") and op.get("hasPictureFile"):
            return True
    return False


def payload_attachment_key(payload: dict[str, Any]) -> str:
    """Use attachment relations, not a mutable document ID, to deduplicate real input."""
    relation_values: list[Any] = []
    for item in payload.get("data") or []:
        op = item.get("op") if isinstance(item, dict) else None
        if not isinstance(op, dict):
            continue
        relation_values.extend([op.get("hasFullTextFile"), op.get("hasPictureFile")])
    if not relation_values:
        return ""
    return json.dumps(relation_values, ensure_ascii=False, sort_keys=True, default=str)


def produce_messages(
    *,
    source_payloads: list[dict[str, Any]],
    topic: str,
    run_id: str,
    rewrite_image_ids: bool = False,
) -> dict[str, Any]:
    started = time.time()
    producer = Producer(
        {
            "bootstrap.servers": DEFAULT_BOOTSTRAP,
            "security.protocol": "SASL_PLAINTEXT",
            "sasl.mechanism": "PLAIN",
            "sasl.username": KAFKA_SASL_USERNAME,
            "sasl.password": KAFKA_SASL_PASSWORD,
        }
    )
    produced_count = 0
    first_doc_id: str | None = None
    last_doc_id: str | None = None
    original_document_ids: list[str] = []
    fixed_update_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    queue_backpressure_events = 0
    for index, base_payload in enumerate(source_payloads):
        doc_id = f"lt_{run_id}_{index:08d}"
        original_document_ids.append(extract_original_document_id(base_payload))
        payload = mutate_payload(
            base_payload=base_payload,
            doc_id=doc_id,
            update_ms=fixed_update_ms,
            publish_ms=fixed_update_ms + index,
            rewrite_image_ids=rewrite_image_ids,
        )
        encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # Large real payloads can exceed librdkafka's local queue before the
        # broker acknowledges them.  Wait for delivery instead of aborting the
        # whole run with BufferError and leaving a partial Topic behind.
        while True:
            try:
                producer.produce(topic, encoded_payload)
                break
            except BufferError:
                queue_backpressure_events += 1
                producer.poll(0.5)
        if index % 20 == 0:
            producer.poll(0)
        if index and index % 1000 == 0:
            # Bound the client-side queue for multi-gigabyte real-payload
            # replays.  A periodic flush also makes publish progress visible
            # to the broker before the next chunk is serialized.
            producer.flush(60)
        if first_doc_id is None:
            first_doc_id = doc_id
        last_doc_id = doc_id
        produced_count += 1
    producer.flush(60)
    return {
        "topic": topic,
        "messages_published": produced_count,
        "publish_duration_seconds": round(time.time() - started, 3),
        "publish_mode": "fixed_real_payloads",
        "queue_backpressure_events": queue_backpressure_events,
        "fixed_update_ms": fixed_update_ms,
        "doc_id_first": first_doc_id,
        "doc_id_last": last_doc_id,
        "original_document_ids": original_document_ids,
        "image_ids_rewritten_for_test": rewrite_image_ids,
    }


def extract_original_document_id(payload: dict[str, Any]) -> str:
    data = payload.get("data") or []
    item = data[0] if data and isinstance(data[0], dict) else {}
    return str(item.get("id") or payload.get("id") or "")


def mutate_payload(
    *,
    base_payload: dict[str, Any],
    doc_id: str,
    update_ms: int,
    publish_ms: int,
    rewrite_image_ids: bool = False,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["id"] = doc_id
    payload["publishTime"] = publish_ms
    data_items = payload.get("data") or []
    if not data_items:
        raise ValueError("sample payload has no data items")

    item = data_items[0]
    item["id"] = doc_id
    if "sysId" in item:
        item["sysId"] = doc_id

    dp = item.get("dp") or {}
    if "sysId" in dp and isinstance(dp["sysId"], dict):
        dp["sysId"]["value"] = [doc_id]
    if "sysDataUpdateTime" in dp and isinstance(dp["sysDataUpdateTime"], dict):
        dp["sysDataUpdateTime"]["value"] = [update_ms]
    if "sysDataStorageTime" in dp and isinstance(dp["sysDataStorageTime"], dict):
        dp["sysDataStorageTime"]["value"] = [update_ms]
    if "updateTime" in dp and isinstance(dp["updateTime"], dict):
        dp["updateTime"]["value"] = [update_ms]
    if rewrite_image_ids:
        _rewrite_image_relation_ids(payload, doc_id=doc_id)
    return payload


def _rewrite_image_relation_ids(payload: dict[str, Any], *, doc_id: str) -> None:
    """Give repeated real images isolated primary keys without changing their sources.

    The benchmark replays a finite historical payload set.  Reusing an original
    picture id would make Milvus upsert/insert semantics overwrite an earlier
    test row, even though the downloader, MMD parser, embedder and Milvus write
    path all executed.  Only this test-only primary key is changed; the source
    picture URL, MMD URL and all content fields stay untouched.
    """
    image_index = 0
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        if not isinstance(op, dict):
            continue
        relation = op.get("hasPictureFile")
        values = relation.get("value") if isinstance(relation, dict) else relation
        if not isinstance(values, list):
            values = [values]
        for picture in values:
            if not isinstance(picture, dict):
                continue
            picture["id"] = f"{doc_id}__image_{image_index:04d}"
            image_index += 1


def collect_snapshot(
    port: int,
    mongo_db: str,
    collections: dict[str, str],
    *,
    worker_process: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
    return {
        "ts": iso_now(),
        "health": (
            {
                "mode": "image_worker_process",
                "running": worker_process.poll() is None,
                "exit_code": worker_process.poll(),
            }
            if worker_process is not None
            else safe_health(port)
        ),
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


def assert_isolated_targets_absent(mongo_db: str, collections: dict[str, str]) -> None:
    """Fail closed instead of deleting any pre-existing Mongo or Milvus data."""
    client = MongoClient(DEFAULT_MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000)
    try:
        if mongo_db in client.list_database_names():
            raise RuntimeError(f"refusing to reuse existing Mongo database: {mongo_db}")
    finally:
        client.close()
    milvus = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=20)
    try:
        for name in collections.values():
            if milvus.has_collection(name):
                raise RuntimeError(f"refusing to reuse existing Milvus collection: {name}")
    finally:
        milvus.close()


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
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: OSError | None = None
    for _ in range(3):
        temporary_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            temporary_path.write_text(content, encoding="utf-8")
            os.replace(temporary_path, path)
            return
        except OSError as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            time.sleep(0.2)
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    sys.exit(main())
