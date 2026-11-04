#!/usr/bin/env python3
"""Run a clean full-chain Kafka -> Mongo -> Milvus test on formal collections.

This script is intentionally destructive for the configured test targets:
it drops the configured Mongo database and the three formal Milvus collections,
then recreates them through the local service startup path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import httpx
from confluent_kafka import Consumer, ConsumerGroupTopicPartitions, KafkaError, Producer, TopicPartition
from confluent_kafka.admin import AdminClient
from pymilvus import MilvusClient
from pymongo import MongoClient


REPO_ROOT = Path(__file__).resolve().parents[2]
KAFKA_SASL_USERNAME = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.environ.get("KAFKA_SASL_PASSWORD", "")
SERVICE_ROOT = REPO_ROOT / "vector_sync_service"
VENV_PYTHON = SERVICE_ROOT / ".venv" / "Scripts" / "python.exe"

DEFAULT_RUN_PREFIX = "formal_fullchain_10m"
DEFAULT_TARGET_HOST = "127.0.0.1"
DEFAULT_TARGET_PORT = 19533
DEFAULT_COLLECTIONS = {
    "documents": "scientific_documents",
    "chunks": "scientific_fulltext_chunks",
    "images": "scientific_images",
}
DEFAULT_TOPIC = "VECTOR_JOURNAL_ARTICLE"
DEFAULT_BOOTSTRAP = "127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092"
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "")
DEFAULT_MONGO_DB = "nstl_vector_sync_scientific_20260617"
DEFAULT_EMBEDDING_URL = "http://127.0.0.1:16703/v1/embeddings"
DEFAULT_IMAGE_CHAT_URL = "http://127.0.0.1:50019/v1/chat/completions"
DEFAULT_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
DEFAULT_PUBLISH_URL = "http://127.0.0.1:17123/api/service/interfaces/vector/job/publishBatch"


@dataclass
class PublishPageResult:
    page: int
    ok: bool
    status_code: int | None
    elapsed_seconds: float
    body_excerpt: str
    error: str = ""


class DurationPublisher(threading.Thread):
    def __init__(
        self,
        *,
        url: str,
        token: str,
        sub_doc_type: str,
        start_page: int,
        duration_seconds: int,
        timeout_seconds: int,
        interval_seconds: float,
        output_path: Path,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.token = token
        self.sub_doc_type = sub_doc_type
        self.start_page = start_page
        self.duration_seconds = duration_seconds
        self.timeout_seconds = timeout_seconds
        self.interval_seconds = interval_seconds
        self.output_path = output_path
        self.stop_event = stop_event
        self.results: list[PublishPageResult] = []
        self.current_page = start_page - 1

    def run(self) -> None:
        deadline = time.monotonic() + max(int(self.duration_seconds), 0)
        page = self.start_page
        while time.monotonic() < deadline and not self.stop_event.is_set():
            result = self._publish_page(page)
            self.results.append(result)
            self.current_page = page
            self._write_results()
            page += 1
            remaining = deadline - time.monotonic()
            if self.interval_seconds > 0 and remaining > 0:
                time.sleep(min(self.interval_seconds, remaining))

    def summary(self) -> dict[str, Any]:
        ok = [item for item in self.results if item.ok]
        failed = [item for item in self.results if not item.ok]
        elapsed = [item.elapsed_seconds for item in ok]
        return {
            "mode": "publish-batch",
            "requested_duration_seconds": self.duration_seconds,
            "start_page": self.start_page,
            "last_attempted_page": self.current_page,
            "inflight_page": self.current_page + 1 if self.is_alive() else None,
            "attempted_pages": len(self.results),
            "successful_pages": len(ok),
            "failed_pages": len(failed),
            "failed_examples": [asdict(item) for item in failed[:10]],
            "page_latency_seconds": latency_snapshot(elapsed),
        }

    def _publish_page(self, page: int) -> PublishPageResult:
        form = {"subDocType": self.sub_doc_type, "pageNumer": str(page)}
        payload = urlencode(form).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "vector-sync-formal-fullchain-test/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.url, data=payload, method="POST", headers=headers)
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                status_code = int(response.status)
            elapsed = time.monotonic() - started
            excerpt = decode_body_excerpt(body)
            ensure_successful_body(body, excerpt)
            return PublishPageResult(page, True, status_code, elapsed, excerpt)
        except HTTPError as error:
            body = error.read()
            return PublishPageResult(
                page,
                False,
                int(error.code),
                time.monotonic() - started,
                decode_body_excerpt(body),
                f"HTTPError: {error}",
            )
        except (TimeoutError, URLError, RuntimeError, OSError) as error:
            return PublishPageResult(
                page,
                False,
                None,
                time.monotonic() - started,
                "",
                f"{type(error).__name__}: {error}",
            )

    def _write_results(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": iso_now(),
            "summary": self.summary(),
            "results": [asdict(item) for item in self.results],
        }
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            )


class TopicReplayPublisher(threading.Thread):
    """Replay real historical Kafka payloads to the topic tail for a duration."""

    def __init__(
        self,
        *,
        topic: str,
        source_offsets: dict[str, Any],
        duration_seconds: int,
        output_path: Path,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.topic = topic
        self.source_offsets = source_offsets
        self.duration_seconds = duration_seconds
        self.output_path = output_path
        self.stop_event = stop_event
        self.started_at = 0.0
        self.finished_at = 0.0
        self.messages_published = 0
        self.bytes_published = 0
        self.delivery_failures = 0
        self.skipped_messages = 0
        self.last_source_offset: dict[int, int] = {}
        self.last_error = ""
        self._lock = threading.Lock()

    def run(self) -> None:
        self.started_at = time.monotonic()
        deadline = self.started_at + max(int(self.duration_seconds), 0)
        consumer = Consumer(
            {
                **kafka_config(
                    f"formal_fullchain_replay_source_{int(time.time())}",
                    auto_offset_reset="earliest",
                ),
                "enable.partition.eof": True,
            }
        )
        producer = Producer(kafka_admin_config())
        end_offsets = {
            int(item["partition"]): int(item["high"])
            for item in self.source_offsets.get("partitions", [])
        }
        assignments = [
            TopicPartition(self.topic, int(item["partition"]), int(item["low"]))
            for item in self.source_offsets.get("partitions", [])
            if int(item.get("low", 0)) < int(item.get("high", 0))
        ]
        completed: set[int] = set()
        try:
            if not assignments:
                self.last_error = "no source messages available"
                return
            consumer.assign(assignments)
            while time.monotonic() < deadline and not self.stop_event.is_set():
                if len(completed) >= len(assignments):
                    completed.clear()
                    consumer.assign(assignments)
                    consumer.seek(assignments[0])
                msg = consumer.poll(timeout=1.0)
                producer.poll(0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        completed.add(int(msg.partition()))
                        continue
                    self.last_error = str(msg.error())
                    continue
                partition = int(msg.partition())
                end_offset = end_offsets.get(partition)
                if end_offset is not None and int(msg.offset()) >= end_offset:
                    completed.add(partition)
                    continue
                value = msg.value()
                if value is None:
                    continue
                produced = False
                while not self.stop_event.is_set():
                    try:
                        producer.produce(
                            self.topic,
                            value=value,
                            key=msg.key(),
                            callback=self._delivery_callback,
                        )
                        produced = True
                        break
                    except BufferError:
                        producer.poll(0.5)
                    except Exception as error:  # noqa: BLE001
                        with self._lock:
                            self.skipped_messages += 1
                            self.last_error = f"{type(error).__name__}: {error}"
                        break
                if not produced:
                    if end_offset is not None and int(msg.offset()) + 1 >= end_offset:
                        completed.add(partition)
                    continue
                with self._lock:
                    self.messages_published += 1
                    self.bytes_published += len(value)
                    self.last_source_offset[partition] = int(msg.offset())
                if end_offset is not None and int(msg.offset()) + 1 >= end_offset:
                    completed.add(partition)
                if self.messages_published % 500 == 0:
                    self._write_results()
            producer.flush(60)
        except Exception as error:  # noqa: BLE001
            self.last_error = f"{type(error).__name__}: {error}"
        finally:
            self.finished_at = time.monotonic()
            consumer.close()
            self._write_results()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            elapsed = (
                (self.finished_at or time.monotonic()) - self.started_at
                if self.started_at
                else 0.0
            )
            return {
                "mode": "topic-replay",
                "requested_duration_seconds": self.duration_seconds,
                "elapsed_seconds": round(max(elapsed, 0.0), 3),
                "messages_published": self.messages_published,
                "bytes_published": self.bytes_published,
                "mb_published": round(self.bytes_published / 1024 / 1024, 3),
                "messages_per_second": round(
                    self.messages_published / elapsed,
                    3,
                )
                if elapsed > 0
                else 0.0,
                "delivery_failures": self.delivery_failures,
                "skipped_messages": self.skipped_messages,
                "last_source_offset": dict(self.last_source_offset),
                "source_offsets": self.source_offsets,
                "last_error": self.last_error,
                "running": self.is_alive(),
            }

    def _delivery_callback(self, error, _msg) -> None:
        if error is not None:
            with self._lock:
                self.delivery_failures += 1
                self.last_error = str(error)

    def _write_results(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "updated_at": iso_now(),
                    "summary": self.summary(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and run a formal collection full-chain load test.",
    )
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--kafka-group", default="")
    parser.add_argument("--mongo-db", default=DEFAULT_MONGO_DB)
    parser.add_argument("--publish-url", default=DEFAULT_PUBLISH_URL)
    parser.add_argument("--publish-token", default=os.getenv("PUBLISH_BATCH_TOKEN", ""))
    parser.add_argument(
        "--publish-mode",
        choices=["topic-replay", "publish-batch"],
        default="topic-replay",
        help="topic-replay replays real Kafka payloads to the topic tail; publish-batch calls the upstream HTTP endpoint.",
    )
    parser.add_argument("--sub-doc-type", default="A79")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--publish-timeout-seconds", type=int, default=900)
    parser.add_argument("--publish-interval-seconds", type=float, default=0.0)
    parser.add_argument("--settle-seconds", type=int, default=0)
    parser.add_argument(
        "--pipeline-mode",
        choices=["standard", "image_priority"],
        default="standard",
        help="standard runs all three paths; image_priority only discovers image tasks and saves docs for later backfill.",
    )
    parser.add_argument("--window-id", default="")
    parser.add_argument("--image-backlog-high-watermark", type=int, default=20000)
    parser.add_argument("--image-backlog-resume-threshold", type=int, default=2000)
    parser.add_argument("--image-worker-concurrency", type=int, default=0)
    parser.add_argument("--image-collection", default="")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument(
        "--drop-extra-milvus-collections",
        action="store_true",
        help="Drop non-formal Milvus collections before the run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_prereqs()

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{DEFAULT_RUN_PREFIX}_{safe_slug(run_id)}"
    run_dir = SERVICE_ROOT / "out" / run_name
    runtime_dir = SERVICE_ROOT / ".runtime" / run_name
    upload_dir = SERVICE_ROOT / "upload" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    report_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    publisher_path = run_dir / "publisher.json"
    service_stdout = run_dir / "service.stdout.log"
    service_stderr = run_dir / "service.stderr.log"

    kafka_group = args.kafka_group.strip() or f"python_consumer_group_scientific_clean_{safe_slug(run_id)}"
    collections = dict(DEFAULT_COLLECTIONS)
    if args.image_collection.strip():
        collections["images"] = args.image_collection.strip()
    window_id = args.window_id.strip() or run_name
    stop_event = threading.Event()
    publisher: DurationPublisher | TopicReplayPublisher | None = None
    service_process: subprocess.Popen[str] | None = None

    status: dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "phase": "initializing",
        "started_at": iso_now(),
        "duration_seconds": args.duration_seconds,
        "settle_seconds": args.settle_seconds,
        "pipeline_mode": args.pipeline_mode,
        "window_id": window_id,
        "monitor_interval": args.monitor_interval,
        "local_service_port": args.port,
        "mongo_db": args.mongo_db,
        "kafka_group_id": kafka_group,
        "milvus": {
            "host": DEFAULT_TARGET_HOST,
            "port": DEFAULT_TARGET_PORT,
            "collections": collections,
        },
        "timeline": [],
        "artifacts": {
            "status_path": str(status_path),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "publisher_path": str(publisher_path),
            "service_stdout": str(service_stdout),
            "service_stderr": str(service_stderr),
        },
    }
    write_json(status_path, status)

    try:
        if not args.skip_clean:
            status["phase"] = "cleaning"
            write_json(status_path, status)
            reset_mongo_database(args.mongo_db)
            reset_milvus_collections(
                collections,
                drop_extra=bool(args.drop_extra_milvus_collections),
            )

        topic_offsets_before = get_topic_offsets(DEFAULT_TOPIC)
        status["topic_offsets_before"] = topic_offsets_before
        status["phase"] = "resetting_kafka_group_to_tail"
        write_json(status_path, status)
        reset_group_to_tail(DEFAULT_TOPIC, kafka_group, topic_offsets_before)

        status["phase"] = "starting_service"
        write_json(status_path, status)
        service_process = start_service(
            port=args.port,
            kafka_group=kafka_group,
            mongo_db=args.mongo_db,
            runtime_dir=runtime_dir,
            upload_dir=upload_dir,
            stdout_path=service_stdout,
            stderr_path=service_stderr,
            collections=collections,
            pipeline_mode=args.pipeline_mode,
            window_id=window_id,
            image_backlog_high_watermark=args.image_backlog_high_watermark,
            image_backlog_resume_threshold=args.image_backlog_resume_threshold,
            image_worker_concurrency=args.image_worker_concurrency,
        )
        health = wait_for_service(args.port, timeout_seconds=240)
        status["service"] = {"pid": service_process.pid, "health": health}

        initial_snapshot = collect_snapshot(args.port, args.mongo_db, collections, kafka_group)
        status["initial_snapshot"] = initial_snapshot
        status["phase"] = "publishing_and_observing"
        write_json(status_path, status)

        if args.publish_mode == "publish-batch":
            publisher = DurationPublisher(
                url=args.publish_url,
                token=args.publish_token,
                sub_doc_type=args.sub_doc_type,
                start_page=args.start_page,
                duration_seconds=args.duration_seconds,
                timeout_seconds=args.publish_timeout_seconds,
                interval_seconds=args.publish_interval_seconds,
                output_path=publisher_path,
                stop_event=stop_event,
            )
        else:
            publisher = TopicReplayPublisher(
                topic=DEFAULT_TOPIC,
                source_offsets=topic_offsets_before,
                duration_seconds=args.duration_seconds,
                output_path=publisher_path,
                stop_event=stop_event,
            )
        publisher.start()

        observe_deadline = time.monotonic() + args.duration_seconds
        while time.monotonic() < observe_deadline:
            snapshot = collect_snapshot(args.port, args.mongo_db, collections, kafka_group)
            snapshot["elapsed_seconds"] = round(
                args.duration_seconds - max(observe_deadline - time.monotonic(), 0),
                1,
            )
            snapshot["publisher"] = publisher.summary()
            status["timeline"].append(snapshot)
            write_json(status_path, status)
            sleep_seconds = min(args.monitor_interval, max(observe_deadline - time.monotonic(), 0))
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)

        if publisher.is_alive():
            if isinstance(publisher, TopicReplayPublisher):
                stop_event.set()
                publisher.join(timeout=90)
            else:
                publisher.join(timeout=5)
        if args.settle_seconds > 0:
            status["phase"] = "settling"
            write_json(status_path, status)
            settle_deadline = time.monotonic() + args.settle_seconds
            while time.monotonic() < settle_deadline:
                snapshot = collect_snapshot(args.port, args.mongo_db, collections, kafka_group)
                snapshot["elapsed_seconds"] = round(
                    args.duration_seconds + args.settle_seconds - max(settle_deadline - time.monotonic(), 0),
                    1,
                )
                snapshot["publisher"] = publisher.summary()
                status["timeline"].append(snapshot)
                write_json(status_path, status)
                sleep_seconds = min(args.monitor_interval, max(settle_deadline - time.monotonic(), 0))
                if sleep_seconds <= 0:
                    break
                time.sleep(sleep_seconds)

        final_snapshot = collect_snapshot(args.port, args.mongo_db, collections, kafka_group)
        final_snapshot["publisher"] = publisher.summary() if publisher is not None else {}
        status["timeline"].append(final_snapshot)
        status["phase"] = "finalizing"
        write_json(status_path, status)

        report = build_report(
            run_name=run_name,
            run_id=run_id,
            started_at=status["started_at"],
            duration_seconds=args.duration_seconds,
            settle_seconds=args.settle_seconds,
            pipeline_mode=args.pipeline_mode,
            window_id=window_id,
            kafka_group=kafka_group,
            mongo_db=args.mongo_db,
            collections=collections,
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            publisher_summary=publisher.summary() if publisher is not None else {},
            timeline=status["timeline"],
            artifacts=status["artifacts"],
            service_pid=service_process.pid if service_process is not None else None,
        )
        write_json(report_path, report)
        markdown_path.write_text(render_markdown_report_clean(report), encoding="utf-8")
        status["phase"] = "completed"
        status["finished_at"] = report["finished_at"]
        status["report_path"] = str(report_path)
        status["markdown_path"] = str(markdown_path)
        write_json(status_path, status)
        return 0
    except Exception as error:
        stop_event.set()
        status["phase"] = "failed"
        status["failed_at"] = iso_now()
        status["error"] = f"{type(error).__name__}: {error}"
        write_json(status_path, status)
        raise
    finally:
        stop_event.set()
        stop_service(service_process)


def ensure_prereqs() -> None:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"venv python not found: {VENV_PYTHON}")


def start_service(
    *,
    port: int,
    kafka_group: str,
    mongo_db: str,
    runtime_dir: Path,
    upload_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    collections: dict[str, str],
    pipeline_mode: str,
    window_id: str,
    image_backlog_high_watermark: int,
    image_backlog_resume_threshold: int,
    image_worker_concurrency: int,
) -> subprocess.Popen[str]:
    image_priority = str(pipeline_mode or "standard").strip().lower() == "image_priority"
    fulltext_enabled = "false" if image_priority else "true"
    fulltext_task_upsert_enabled = "false" if image_priority else "true"
    image_concurrency = (
        str(max(int(image_worker_concurrency), 1))
        if int(image_worker_concurrency or 0) > 0
        else "24"
    )
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
            "KAFKA_CONSUMER_ENABLED": "true",
            "KAFKA_CONSUME_LIMIT": "0",
            "KAFKA_MAX_POLL_INTERVAL_MS": "900000",
            "KAFKA_CONSUMER_TIME_BUDGET_MS": "8000",
            "KAFKA_CONSUMER_MICRO_BATCH_SIZE": "8",
            "KAFKA_BACKPRESSURE_ENABLED": "true",
            "KAFKA_BACKPRESSURE_MODE": "image_priority",
            "KAFKA_BACKPRESSURE_HIGH_WATERMARK": str(max(int(image_backlog_high_watermark), 0)),
            "KAFKA_BACKPRESSURE_LOW_WATERMARK": str(max(int(image_backlog_resume_threshold), 0)),
            "KAFKA_BACKPRESSURE_CHECK_INTERVAL_SEC": "2",
            "KAFKA_BACKPRESSURE_IMAGE_PENDING_SOFT_LIMIT": str(max(int(image_backlog_high_watermark), 0)),
            "KAFKA_BACKPRESSURE_IMAGE_PENDING_RESUME_THRESHOLD": str(max(int(image_backlog_resume_threshold), 0)),
            "KAFKA_BACKPRESSURE_SOFT_WATERMARK_FACTOR": "0.55",
            "KAFKA_BACKPRESSURE_SOFT_THROTTLE_SLEEP_MS": "200",
            "KAFKA_BACKPRESSURE_SOFT_MICRO_BATCH_MIN": "1",
            "MONGO_URI": DEFAULT_MONGO_URI,
            "MONGO_DB": mongo_db,
            "MILVUS_HOST": DEFAULT_TARGET_HOST,
            "MILVUS_PORT": str(DEFAULT_TARGET_PORT),
            "MILVUS_COLLECTION": collections["documents"],
            "FULLTEXT_CHUNK_COLLECTION": collections["chunks"],
            "IMAGE_COLLECTION_NAME": collections["images"],
            "INIT_VECTOR_COLLECTIONS_ON_STARTUP": "true",
            "SYNC_PIPELINE_MODE": pipeline_mode,
            "IMAGE_PRIORITY_WINDOW_ID": window_id,
            "EMBEDDING_API_URL": DEFAULT_EMBEDDING_URL,
            "EMBEDDING_API_KEY": DEFAULT_API_KEY,
            "EMBEDDING_MODEL": "qwen3-embed",
            "AUTO_PUBLISH_ENABLED": "false",
            "COMPENSATOR_ENABLED": "false",
            "FULLTEXT_WORKER_ENABLED": fulltext_enabled,
            "FULLTEXT_TASK_UPSERT_ENABLED": fulltext_task_upsert_enabled,
            "FULLTEXT_WORKER_CONCURRENCY": "16",
            "FULLTEXT_WORKER_BATCH_SIZE": "50",
            "IMAGE_WORKER_ENABLED": "true",
            "IMAGE_TASK_UPSERT_ENABLED": "true",
            "IMAGE_FEATURE_PIPELINE_ENABLED": "true",
            "IMAGE_MILVUS_WRITE_ENABLED": "true",
            "IMAGE_WORKER_BATCH_SIZE": "64",
            "IMAGE_WORKER_CONCURRENCY": image_concurrency,
            "IMAGE_DOWNLOAD_CONCURRENCY": "24",
            "IMAGE_VL_CHAT_CONCURRENCY": "8",
            "IMAGE_VL_EMBED_CONCURRENCY": "24",
            "IMAGE_TEXT_EMBED_CONCURRENCY": "32",
            "IMAGE_MILVUS_WRITE_CONCURRENCY": "8",
            "IMAGE_MILVUS_WRITE_BATCH_SIZE": "48",
            "IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS": "1000",
            "IMAGE_MILVUS_WRITE_MODE": "insert_only",
            "IMAGE_MMD_HOT_PATH_MODE": "background",
            "IMAGE_VL_CHAT_URL": DEFAULT_IMAGE_CHAT_URL,
            "IMAGE_VL_CHAT_MODEL": "qwen3.5-plus",
            "IMAGE_VL_EMBED_URL": DEFAULT_EMBEDDING_URL,
            "IMAGE_VL_EMBED_MODEL": "qwen3-vl-embed",
            "IMAGE_TEXT_EMBED_URL": DEFAULT_EMBEDDING_URL,
            "IMAGE_TEXT_EMBED_MODEL": "qwen3-embed",
            "IMAGE_MODEL_API_KEY": DEFAULT_API_KEY,
            "IMAGE_VL_EMBED_API_KEY": DEFAULT_API_KEY,
            "IMAGE_TEXT_EMBED_API_KEY": DEFAULT_API_KEY,
            "IMAGE_SUCCESS_RESULT_MODE": "status_only",
            "IMAGE_SUCCESS_ARTIFACT_MODE": "summary",
            "IMAGE_RUNTIME_ATTACHMENT_MODE": "failure_only",
            "ADMIN_AUTH_ENABLED": "false",
            "UPLOAD_DIR": str(upload_dir),
            "RETENTION_CLEANUP_ENABLED": "false",
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


def wait_for_service(port: int, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    last_error = "unknown"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(2)
    raise TimeoutError(f"service did not become healthy on {url}: {last_error}")


def collect_snapshot(
    port: int,
    mongo_db: str,
    collections: dict[str, str],
    kafka_group: str,
) -> dict[str, Any]:
    health = safe_health(port)
    runtime = ((health.get("body") or {}).get("runtime") or {}) if health else {}
    return {
        "ts": iso_now(),
        "health": health,
        "milvus": {
            "row_counts": get_milvus_row_counts(collections),
        },
        "mongo": get_mongo_summary(mongo_db),
        "kafka": {
            "topic_offsets": get_topic_offsets(DEFAULT_TOPIC),
            "consumer_group_offsets": get_group_offsets(DEFAULT_TOPIC, kafka_group),
        },
        "backpressure": ((runtime.get("consumer") or {}).get("backpressure") or {}),
        "image_stage_executors": (runtime.get("image") or {}).get("stage_executors") or {},
        "image_runtime": summarize_image_runtime(runtime.get("image") or {}),
        "fulltext_runtime": summarize_worker_runtime(runtime.get("fulltext") or {}),
    }


def safe_health(port: int) -> dict[str, Any]:
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=10)
        return {
            "status_code": response.status_code,
            "body": response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {},
        }
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def summarize_worker_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "worker_running",
        "batch_size",
        "concurrency",
        "last_processed_count",
        "last_error",
        "task_summary",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def summarize_image_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "worker_running",
        "batch_size",
        "concurrency",
        "download_concurrency",
        "model_concurrency",
        "milvus_write",
        "last_processed_count",
        "last_error",
        "task_summary",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def build_report(
    *,
    run_name: str,
    run_id: str,
    started_at: str,
    duration_seconds: int,
    settle_seconds: int,
    pipeline_mode: str,
    window_id: str,
    kafka_group: str,
    mongo_db: str,
    collections: dict[str, str],
    initial_snapshot: dict[str, Any],
    final_snapshot: dict[str, Any],
    publisher_summary: dict[str, Any],
    timeline: list[dict[str, Any]],
    artifacts: dict[str, str],
    service_pid: int | None,
) -> dict[str, Any]:
    initial_rows = (initial_snapshot.get("milvus") or {}).get("row_counts") or {}
    final_rows = (final_snapshot.get("milvus") or {}).get("row_counts") or {}
    row_delta = {
        key: int(final_rows.get(key, 0)) - int(initial_rows.get(key, 0))
        for key in collections
    }
    image_stage_executors = final_snapshot.get("image_stage_executors") or {}
    return {
        "run_name": run_name,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": iso_now(),
        "duration_seconds": duration_seconds,
        "settle_seconds": settle_seconds,
        "pipeline_mode": pipeline_mode,
        "window_id": window_id,
        "target": {
            "attu_url": "http://127.0.0.1:18002/#/connect",
            "milvus_host": DEFAULT_TARGET_HOST,
            "milvus_port": DEFAULT_TARGET_PORT,
            "collections": collections,
        },
        "clean_environment": {
            "mongo_db": mongo_db,
            "kafka_group_id": kafka_group,
            "milvus_collections_dropped_before_run": list(collections.values()),
        },
        "service": {
            "local_port": (final_snapshot.get("health") or {}).get("port"),
            "pid": service_pid,
        },
        "publisher": publisher_summary,
        "counts": {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "row_delta": row_delta,
            "total_inserted_rows": sum(row_delta.values()),
            "throughput_rows_per_second": {
                key: round(value / max(duration_seconds + settle_seconds, 1), 3)
                for key, value in row_delta.items()
            },
        },
        "mongo": final_snapshot.get("mongo") or {},
        "kafka": (final_snapshot.get("kafka") or {}),
        "backpressure": final_snapshot.get("backpressure") or {},
        "image_stage_executors": image_stage_executors,
        "image_stage_summary": summarize_image_stages(image_stage_executors),
        "image_runtime": final_snapshot.get("image_runtime") or {},
        "fulltext_runtime": final_snapshot.get("fulltext_runtime") or {},
        "timeline_summary": summarize_timeline(timeline),
        "timeline": timeline,
        "artifacts": artifacts,
    }


def summarize_image_stages(stages: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, payload in stages.items():
        if not isinstance(payload, dict):
            continue
        queue_wait = payload.get("queue_wait_ms") or {}
        run_latency = payload.get("run_latency_ms") or {}
        summary[name] = {
            "concurrency": payload.get("concurrency"),
            "backlog": payload.get("backlog"),
            "inflight": payload.get("inflight"),
            "success_count": payload.get("success_count"),
            "failure_count": payload.get("failure_count"),
            "queue_wait_avg_ms": queue_wait.get("avg"),
            "queue_wait_p95_ms": queue_wait.get("p95"),
            "run_avg_ms": run_latency.get("avg"),
            "run_p95_ms": run_latency.get("p95"),
            "run_max_ms": run_latency.get("max"),
        }
    return summary


def summarize_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    if not timeline:
        return {}
    max_rows = {"documents": 0, "chunks": 0, "images": 0}
    max_image_pending = 0
    max_backpressure_backlog = 0
    max_image_priority_windows = 0
    for item in timeline:
        rows = ((item.get("milvus") or {}).get("row_counts") or {})
        for key in max_rows:
            max_rows[key] = max(max_rows[key], int(rows.get(key, 0) or 0))
        image_counts = (((item.get("mongo") or {}).get("image_task_counts") or {}))
        max_image_pending = max(max_image_pending, int(image_counts.get("PENDING", 0) or 0))
        max_image_priority_windows = max(
            max_image_priority_windows,
            int((item.get("mongo") or {}).get("image_priority_windows", 0) or 0),
        )
        backpressure = item.get("backpressure") or {}
        max_backpressure_backlog = max(
            max_backpressure_backlog,
            int(backpressure.get("last_backlog", 0) or 0),
        )
    return {
        "sample_count": len(timeline),
        "max_row_counts": max_rows,
        "max_image_pending": max_image_pending,
        "max_image_priority_windows": max_image_priority_windows,
        "max_backpressure_backlog": max_backpressure_backlog,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    delta = counts.get("row_delta") or {}
    throughput = counts.get("throughput_rows_per_second") or {}
    mongo = report.get("mongo") or {}
    publisher = report.get("publisher") or {}
    stages = report.get("image_stage_summary") or {}
    lines = [
        "# 正式三集合全链路压测报告",
        "",
        f"- 运行时间: {report.get('started_at')} 至 {report.get('finished_at')}",
        f"- 压测窗口: {report.get('duration_seconds')} 秒，额外沉淀: {report.get('settle_seconds')} 秒",
        f"- MongoDB: `{(report.get('clean_environment') or {}).get('mongo_db')}`",
        f"- Kafka group: `{(report.get('clean_environment') or {}).get('kafka_group_id')}`",
        f"- Milvus: `{(report.get('target') or {}).get('milvus_host')}:{(report.get('target') or {}).get('milvus_port')}`",
        "",
        "## Kafka 投递",
        "",
        f"- 成功投递页数: {publisher.get('successful_pages', 0)} / {publisher.get('attempted_pages', 0)}",
        f"- Kafka replay 消息数: {publisher.get('messages_published', 0)}",
        f"- Kafka replay 速率: {publisher.get('messages_per_second', 0)} msg/s",
        f"- 最后尝试页: {publisher.get('last_attempted_page')}",
        f"- 单页投递 p95: {(publisher.get('page_latency_seconds') or {}).get('p95', 0)} 秒",
        "",
        "## Milvus 入库增量",
        "",
        "| 集合 | 增量 | 平均 rows/s |",
        "| --- | ---: | ---: |",
        f"| scientific_documents | {delta.get('documents', 0)} | {throughput.get('documents', 0)} |",
        f"| scientific_fulltext_chunks | {delta.get('chunks', 0)} | {throughput.get('chunks', 0)} |",
        f"| scientific_images | {delta.get('images', 0)} | {throughput.get('images', 0)} |",
        "",
        "## Mongo 任务状态",
        "",
        f"- 文档状态: `{json.dumps(mongo.get('status_counts', {}), ensure_ascii=False)}`",
        f"- 全文任务: `{json.dumps(mongo.get('fulltext_task_counts', {}), ensure_ascii=False)}`",
        f"- 图片任务: `{json.dumps(mongo.get('image_task_counts', {}), ensure_ascii=False)}`",
        f"- 失败记录: `{json.dumps(mongo.get('failure_counts', {}), ensure_ascii=False)}`",
        "",
        "## 图片阶段耗时",
        "",
        "| 阶段 | 成功 | 失败 | 并发 | 队列等待 p95(ms) | 执行 avg(ms) | 执行 p95(ms) | 执行 max(ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("download", "feature_visual", "feature_text", "feature_pipeline", "milvus_write", "mongo_success"):
        item = stages.get(name) or {}
        lines.append(
            f"| {name} | {item.get('success_count', 0)} | {item.get('failure_count', 0)} | "
            f"{item.get('concurrency', 0)} | {item.get('queue_wait_p95_ms', 0)} | "
            f"{item.get('run_avg_ms', 0)} | {item.get('run_p95_ms', 0)} | {item.get('run_max_ms', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Backpressure",
            "",
            f"- 状态: `{json.dumps(report.get('backpressure') or {}, ensure_ascii=False)}`",
            "",
            "## 产物",
            "",
            f"- JSON 报告: `{(report.get('artifacts') or {}).get('report_path')}`",
            f"- 采样状态: `{(report.get('artifacts') or {}).get('status_path')}`",
            f"- 服务 stdout: `{(report.get('artifacts') or {}).get('service_stdout')}`",
            f"- 服务 stderr: `{(report.get('artifacts') or {}).get('service_stderr')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_markdown_report_clean(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    delta = counts.get("row_delta") or {}
    throughput = counts.get("throughput_rows_per_second") or {}
    mongo = report.get("mongo") or {}
    publisher = report.get("publisher") or {}
    stages = report.get("image_stage_summary") or {}
    clean_env = report.get("clean_environment") or {}
    lines = [
        "# 正式三集合全链路压测报告",
        "",
        f"- 运行时间: {report.get('started_at')} 到 {report.get('finished_at')}",
        f"- 模式: `{report.get('pipeline_mode') or 'standard'}`",
        f"- 窗口 ID: `{report.get('window_id') or ''}`",
        f"- MongoDB: `{clean_env.get('mongo_db')}`",
        f"- Kafka group: `{clean_env.get('kafka_group_id')}`",
        f"- Milvus: `{(report.get('target') or {}).get('milvus_host')}:{(report.get('target') or {}).get('milvus_port')}`",
        "",
        "## Kafka 投递",
        "",
        f"- 成功投递页数: {publisher.get('successful_pages', 0)} / {publisher.get('attempted_pages', 0)}",
        f"- Kafka replay 消息数: {publisher.get('messages_published', 0)}",
        f"- Kafka replay 速率: {publisher.get('messages_per_second', 0)} msg/s",
        f"- 最后尝试页: {publisher.get('last_attempted_page')}",
        f"- 单页投递 p95: {(publisher.get('page_latency_seconds') or {}).get('p95', 0)} 秒",
        "",
        "## Milvus 入库增量",
        "",
        "| 集合 | 增量 | 平均 rows/s |",
        "| --- | ---: | ---: |",
        f"| scientific_documents | {delta.get('documents', 0)} | {throughput.get('documents', 0)} |",
        f"| scientific_fulltext_chunks | {delta.get('chunks', 0)} | {throughput.get('chunks', 0)} |",
        f"| scientific_images | {delta.get('images', 0)} | {throughput.get('images', 0)} |",
        "",
        "## Mongo 任务状态",
        "",
        f"- 文献状态: `{json.dumps(mongo.get('status_counts', {}), ensure_ascii=False)}`",
        f"- 全文任务: `{json.dumps(mongo.get('fulltext_task_counts', {}), ensure_ascii=False)}`",
        f"- 图片任务: `{json.dumps(mongo.get('image_task_counts', {}), ensure_ascii=False)}`",
        f"- 图片优先窗口: `{json.dumps(mongo.get('image_priority_window_counts', {}), ensure_ascii=False)}`",
        f"- 失败记录: `{json.dumps(mongo.get('failure_counts', {}), ensure_ascii=False)}`",
        "",
        "## 图片阶段耗时",
        "",
        "| 阶段 | 成功 | 失败 | 并发 | 队列等待 p95(ms) | 执行 avg(ms) | 执行 p95(ms) | 执行 max(ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("download", "feature_visual", "feature_text", "feature_pipeline", "milvus_write", "mongo_success"):
        item = stages.get(name) or {}
        lines.append(
            f"| {name} | {item.get('success_count', 0)} | {item.get('failure_count', 0)} | "
            f"{item.get('concurrency', 0)} | {item.get('queue_wait_p95_ms', 0)} | "
            f"{item.get('run_avg_ms', 0)} | {item.get('run_p95_ms', 0)} | {item.get('run_max_ms', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Backpressure",
            "",
            f"- 状态: `{json.dumps(report.get('backpressure') or {}, ensure_ascii=False)}`",
            "",
            "## 产物",
            "",
            f"- JSON 报告: `{(report.get('artifacts') or {}).get('report_path')}`",
            f"- 采样状态: `{(report.get('artifacts') or {}).get('status_path')}`",
            f"- 服务 stdout: `{(report.get('artifacts') or {}).get('service_stdout')}`",
            f"- 服务 stderr: `{(report.get('artifacts') or {}).get('service_stderr')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def get_mongo_summary(mongo_db: str) -> dict[str, Any]:
    client = MongoClient(
        DEFAULT_MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )
    try:
        db = client[mongo_db]
        return {
            "doc_sync_status": int(db["doc_sync_status"].count_documents({})),
            "doc_fulltext_tasks": int(db["doc_fulltext_tasks"].count_documents({})),
            "doc_image_tasks": int(db["doc_image_tasks"].count_documents({})),
            "image_priority_windows": int(
                db["image_priority_windows"].count_documents({})
            ),
            "doc_sync_failures": int(db["doc_sync_failures"].count_documents({})),
            "status_counts": group_counts(db["doc_sync_status"], "overall_status"),
            "document_pipeline_counts": group_counts(db["doc_sync_status"], "document_status"),
            "fulltext_pipeline_counts": group_counts(db["doc_sync_status"], "fulltext_status"),
            "image_pipeline_counts": group_counts(db["doc_sync_status"], "image_status"),
            "fulltext_task_counts": group_counts(db["doc_fulltext_tasks"], "status"),
            "image_task_counts": group_counts(db["doc_image_tasks"], "status"),
            "image_priority_window_counts": group_counts(
                db["image_priority_windows"], "backfill_status"
            ),
            "image_priority_window_stage_counts": group_counts(
                db["image_priority_windows"], "status"
            ),
            "image_error_counts": group_counts(db["doc_image_tasks"], "error_code"),
            "fulltext_error_counts": group_counts(db["doc_fulltext_tasks"], "error_code"),
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
    return {str(row.get("_id") or "NONE"): int(row.get("count") or 0) for row in rows}


def reset_mongo_database(mongo_db: str) -> None:
    client = MongoClient(
        DEFAULT_MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )
    try:
        client.drop_database(mongo_db)
    finally:
        client.close()


def reset_milvus_collections(collections: dict[str, str], *, drop_extra: bool = False) -> None:
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=60)
    try:
        formal = set(collections.values())
        names = set(formal)
        if drop_extra:
            names.update(str(name) for name in (client.list_collections() or []))
        for name in sorted(names):
            if client.has_collection(name):
                client.drop_collection(collection_name=name)
    finally:
        client.close()


def get_milvus_row_counts(collections: dict[str, str]) -> dict[str, int]:
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=30)
    try:
        counts: dict[str, int] = {}
        for label, name in collections.items():
            if not client.has_collection(name):
                counts[label] = 0
                continue
            try:
                client.flush(collection_name=name)
            except Exception:
                pass
            stats = client.get_collection_stats(collection_name=name)
            counts[label] = int(stats.get("row_count", 0))
        return counts
    finally:
        client.close()


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


def kafka_admin_config() -> dict[str, Any]:
    return {
        "bootstrap.servers": DEFAULT_BOOTSTRAP,
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "PLAIN",
        "sasl.username": KAFKA_SASL_USERNAME,
        "sasl.password": KAFKA_SASL_PASSWORD,
    }


def get_topic_offsets(topic: str) -> dict[str, Any]:
    consumer = Consumer(kafka_config("formal_fullchain_topic_probe", auto_offset_reset="earliest"))
    try:
        md = consumer.list_topics(topic, timeout=20)
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
                    "low": int(low),
                    "high": int(high),
                    "available": int(high - low),
                }
            )
        return {"topic": topic, "partitions": partitions}
    finally:
        consumer.close()


def reset_group_to_tail(topic: str, group_id: str, topic_offsets: dict[str, Any]) -> None:
    partitions = [
        TopicPartition(topic, int(item["partition"]), int(item["high"]))
        for item in topic_offsets.get("partitions", [])
    ]
    if not partitions:
        return
    admin = AdminClient(kafka_admin_config())
    request = [ConsumerGroupTopicPartitions(group_id, partitions)]
    futures = admin.alter_consumer_group_offsets(request, request_timeout=30)
    for future in futures.values():
        future.result()


def get_group_offsets(topic: str, group_id: str) -> dict[str, Any]:
    consumer = Consumer(kafka_config(group_id, auto_offset_reset="earliest"))
    try:
        md = consumer.list_topics(topic, timeout=20)
        topic_meta = md.topics.get(topic)
        if topic_meta is None:
            raise ValueError(f"Topic {topic} metadata not found")
        partitions = [
            TopicPartition(topic, partition_id)
            for partition_id in sorted(topic_meta.partitions.keys())
        ]
        committed = consumer.committed(partitions, timeout=10)
        results = []
        total_lag = 0
        for item in committed:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, item.partition),
                timeout=10,
                cached=False,
            )
            current_offset = int(item.offset if item.offset is not None and item.offset >= 0 else low)
            lag = max(int(high) - current_offset, 0)
            total_lag += lag
            results.append(
                {
                    "partition": int(item.partition),
                    "low": int(low),
                    "high": int(high),
                    "current_offset": current_offset,
                    "lag": lag,
                }
            )
        return {
            "topic": topic,
            "group_id": group_id,
            "total_lag": total_lag,
            "partitions": results,
        }
    finally:
        consumer.close()


def stop_service(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def latency_snapshot(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(item) for item in values)
    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 3),
        "p50": round(percentile(ordered, 50), 3),
        "p95": round(percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * max(min(pct, 100.0), 0.0) / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def decode_body_excerpt(body: bytes, limit: int = 500) -> str:
    if not body:
        return ""
    return body.decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")[:limit]


def ensure_successful_body(body: bytes, excerpt: str) -> None:
    if not body:
        return
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if isinstance(parsed, dict) and parsed.get("success") is False:
        raise RuntimeError(f"Response success=false: {excerpt}")
    if isinstance(parsed, dict) and str(parsed.get("code", "")).lower() in {"500", "-1", "error"}:
        raise RuntimeError(f"Response code indicates failure: {excerpt}")


def safe_slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(value or ""))
    text = "_".join(part for part in text.split("_") if part)
    return (text or datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S"))[:48]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
