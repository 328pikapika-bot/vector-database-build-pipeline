#!/usr/bin/env python3
"""Measure one real Kafka-to-Milvus collection pipeline without the other two.

This is deliberately a test-only driver.  Production keeps its normal fan-out
architecture (Kafka -> document -> Mongo tasks).  Here fulltext and image use
their own Kafka group only to create their normal Mongo tasks, so each worker
can be measured without competing with document or the other worker.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AbstractContextManager
from io import BytesIO
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from confluent_kafka import Consumer, TopicPartition
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pymilvus import MilvusClient
from pymongo import MongoClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "vector_sync_service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

# Load the same local connection settings used by the service before the
# isolated-test helpers capture their module-level defaults.
load_dotenv(SERVICE_ROOT / ".env", override=False)
KAFKA_SASL_USERNAME = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.environ.get("KAFKA_SASL_PASSWORD", "")

from run_isolated_kafka_load_test import (  # noqa: E402
    DEFAULT_BOOTSTRAP,
    DEFAULT_MONGO_URI,
    DEFAULT_TARGET_HOST,
    DEFAULT_TARGET_PORT,
    ensure_test_topic,
    load_real_payloads,
    produce_messages,
)
from run_isolated_kafka_backlog_load_test import (  # noqa: E402
    reset_milvus_collections,
    reset_mongo_database,
)


# Use the production-compatible V2 image schema for real image-chain tests.
# The older caption_context_2_vector name is not accepted by current target
# resolution and causes every direct write to be rejected before Milvus.
TEST_IMAGE_SCHEMA_MODE = "history_compatible_v2"
DEFAULT_MESSAGES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("metadata", "fulltext", "image"), required=True)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES,
                        help="Preloaded real messages. Consumption itself has no configured limit.")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--collection-name",
        default="",
        help="Explicit test collection name (used by the fixed three-lane orchestrator).",
    )
    parser.add_argument(
        "--mongo-db",
        default="",
        help="Explicit Mongo database name. The database is never reset when --skip-mongo-reset is set.",
    )
    parser.add_argument(
        "--consumer-group",
        default="",
        help="Explicit Kafka consumer group for this lane.",
    )
    parser.add_argument(
        "--skip-mongo-reset",
        action="store_true",
        help="Do not drop/reset the selected Mongo database before running.",
    )
    parser.add_argument(
        "--skip-collection-reset",
        action="store_true",
        help="Do not drop the selected Milvus collection before running.",
    )
    parser.add_argument(
        "--reuse-topic",
        default="",
        help="Reuse an existing isolated Kafka topic without publishing another fixture set.",
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Explicit test topic name. The topic is created and populated unless --reuse-topic is used.",
    )
    parser.add_argument("--source-jsonl", type=Path,
                        default=REPO_ROOT / "bakkafkadata" / "31" / "kafka_messages.jsonl")
    parser.add_argument(
        "--include-runtime-history",
        action="store_true",
        help="Also use historical runtime dumps in addition to --source-jsonl.",
    )
    parser.add_argument("--document-concurrency", type=int, default=48)
    parser.add_argument("--metadata-mode", choices=("standard", "metadata_bulk"), default="standard")
    parser.add_argument("--metadata-batch-size", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--metadata-pipeline-inflight-batches", type=int, default=1)
    parser.add_argument("--fulltext-concurrency", type=int, default=64)
    parser.add_argument(
        "--fulltext-direct",
        action="store_true",
        help="Process fulltext directly from Kafka without retaining successful Mongo tasks.",
    )
    parser.add_argument("--fulltext-direct-batch-size", type=int, default=32)
    parser.add_argument("--fulltext-direct-pipeline-inflight-batches", type=int, default=1)
    parser.add_argument("--image-concurrency", type=int, default=64)
    parser.add_argument("--image-visual-concurrency", type=int, default=0,
                        help="Override qwen3-vl-embed concurrency (defaults to image concurrency).")
    parser.add_argument("--image-text-concurrency", type=int, default=0,
                        help="Override qwen3-embed text concurrency for image lane.")
    parser.add_argument("--image-download-concurrency", type=int, default=0,
                        help="Override image download concurrency.")
    parser.add_argument(
        "--image-direct",
        action="store_true",
        help="Process images directly from the test Kafka topic without Mongo task queuing.",
    )
    parser.add_argument("--image-direct-batch-size", type=int, default=16)
    parser.add_argument("--image-direct-pipeline-inflight-batches", type=int, default=1)
    parser.add_argument(
        "--image-mmd-hot-path-mode",
        choices=("disabled", "background", "required"),
        default="background",
    )
    parser.add_argument("--image-text-batch-size", type=int, default=32)
    parser.add_argument(
        "--image-text-batch-delay-ms",
        type=int,
        default=50,
        help="Short coalescing window so caption/context calls form real batches.",
    )
    parser.add_argument("--image-text-max-in-flight-batches", type=int, default=4)
    parser.add_argument(
        "--image-model-total-concurrency",
        type=int,
        default=32,
        help="Shared visual/text gateway concurrency; higher values can reduce throughput via queueing.",
    )
    parser.add_argument("--image-preprocess-concurrency", type=int, default=32)
    parser.add_argument(
        "--image-attachment-prefix",
        default="http://127.0.0.1:12001/",
        help="Image download origin used by the isolated image benchmark.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Publish project-shaped synthetic image messages and serve fixture images locally.",
    )
    parser.add_argument("--synthetic-images-per-document", type=int, default=5)
    parser.add_argument("--embedding-concurrency", type=int, default=256)
    parser.add_argument(
        "--embedding-text-rate",
        type=float,
        default=0.0,
        help="Optional per-process qwen3-embed budget in input texts per second.",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    # Mongo database names are limited to 63 bytes; the fixed prefix/suffix
    # below needs room as well. Preserve the suffix because the orchestrator
    # appends the lane name there; truncating only the prefix would make all
    # three lanes share one database and report directory.
    if len(value) > 30:
        value = f"{value[:20]}_{value[-9:]}"
    return value[:30] or datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _synthetic_image_jpeg() -> bytes:
    """Create one realistic-size JPEG used through the normal HTTP downloader."""
    image = Image.new("RGB", (1024, 768), color=(237, 241, 244))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(((25, 92, 145), (47, 140, 114), (202, 122, 55))):
        left = 80 + index * 270
        draw.rectangle((left, 100, left + 190, 620), fill=color)
        draw.line((left, 660, left + 190, 660), fill=(30, 30, 30), width=4)
    output = BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


class _SyntheticAttachmentServer(AbstractContextManager["_SyntheticAttachmentServer"]):
    """In-process HTTP origin so download behavior stays inside the benchmark."""

    def __init__(self) -> None:
        self._image_bytes = _synthetic_image_jpeg()
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def prefix(self) -> str:
        if self._server is None:
            raise RuntimeError("synthetic attachment server is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def __enter__(self) -> "_SyntheticAttachmentServer":
        image_bytes = self._image_bytes

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.endswith(".mmd"):
                    body = b"# Synthetic full text\n\nSynthetic image context.\n"
                    content_type = "text/plain; charset=utf-8"
                else:
                    body = image_bytes
                    content_type = "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None


def _build_synthetic_image_payloads(
    *,
    messages: int,
    images_per_document: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_count = max(int(images_per_document), 1)
    payloads: list[dict[str, Any]] = []
    for document_index in range(max(int(messages), 1)):
        document_id = f"synthetic-image-document-{document_index:08d}"
        relations: list[dict[str, Any]] = []
        mmd_parts = [
            "本文讨论科技文献中的图像语义检索、实验结果对比和数据质量控制。",
            "实验采用统一采样条件，并在多个指标上对模型输出进行复核。",
        ]
        for image_index in range(image_count):
            file_name = f"figure_{image_index + 1:02d}.jpg"
            caption = (
                f"图 {image_index + 1}：第 {document_index + 1} 篇模拟科技文献的实验结果，"
                "展示指标变化、样本分布与对比分析。"
            )
            relations.append(
                {
                    "id": f"synthetic-image-{document_index:08d}-{image_index:02d}",
                    "fileUrl": f"/synthetic/{document_id}/figures/{file_name}",
                    "sequence": image_index + 1,
                    "title": caption,
                    "fileName": file_name,
                    "width": 1024,
                    "height": 768,
                    "fileType": "jpeg",
                    "size": 96_000,
                    "context_before": (
                        "本段说明实验数据来源、变量设置和统计方法，"
                        "用于模拟真实图片前的论文上下文。"
                    ),
                    "context_after": (
                        "图中结果用于比较不同参数下的召回表现，"
                        "并给出误差来源和后续分析结论。"
                    ),
                }
            )
            mmd_parts.extend(
                [
                    f"图 {image_index + 1} 前的实验说明，包含方法、数据集和评估指标。",
                    f"![{caption}](figures/{file_name})",
                    "图后给出结果解释、参数敏感性分析和可复现实验结论。",
                ]
            )
        payloads.append(
            {
                "id": document_id,
                "data": [
                    {
                        "id": document_id,
                        "sysId": document_id,
                        "sysDataUpdateTime": "2026-08-05T00:00:00+00:00",
                        "dp": {
                            "sysId": {"field": "sysId", "value": [document_id]},
                            "sysDataUpdateTime": {
                                "field": "sysDataUpdateTime",
                                "value": ["2026-08-05T00:00:00+00:00"],
                            },
                            "title": {
                                "field": "title",
                                "value": ["面向科技文献图像检索的向量化入库性能模拟研究"],
                            },
                            "keyword": {"field": "keyword", "value": ["向量检索", "全文图片", "性能评估"]},
                            "year": {"field": "year", "value": ["2026"]},
                            "language": {"field": "language", "value": ["zh"]},
                            "accessType": {"field": "accessType", "value": ["OA"]},
                            "classificationCode": {"field": "classificationCode", "value": ["TP391"]},
                            "sysDataStatus": {"field": "sysDataStatus", "value": ["ACTIVE"]},
                            "hasAuthorname": {"field": "hasAuthorname", "value": ["模拟作者"]},
                            "hasCreateOrganizationname": {"field": "hasCreateOrganizationname", "value": ["NSTL"]},
                            "hasSourcesysSourceId": {
                                "field": "hasSourcesysSourceId",
                                "value": ["SYNTHETIC-SOURCE"],
                            },
                            "hasSourceidSet": {
                                "field": "hasSourceidSet",
                                "value": ["SYNTHETIC-JOURNAL"],
                            },
                        },
                        "title": ["面向科技文献图像检索的向量化入库性能模拟研究"],
                        "keyword": ["向量检索", "全文图片", "性能评估"],
                        "year": "2026",
                        "language": ["zh"],
                        "accessType": ["OA"],
                        "classificationCode": ["TP391"],
                        "sysDataStatus": "ACTIVE",
                        "hasAuthorname": ["模拟作者"],
                        "hasCreateOrganizationname": ["NSTL"],
                        "hasSourcesysSourceId": ["SYNTHETIC-SOURCE"],
                        "hasSourceidSet": ["SYNTHETIC-JOURNAL"],
                        "op": {
                            "hasPictureFile": {"field": "hasPictureFile", "value": relations},
                            "hasFullTextFile": {
                                "field": "hasFullTextFile",
                                "value": [{"fileUrl": f"/synthetic/{document_id}/{document_id}.mmd"}],
                            },
                        },
                        "mmd_text": "\n\n".join(mmd_parts),
                    }
                ],
            }
        )
    return payloads, {
        "synthetic": True,
        "synthetic_documents": len(payloads),
        "synthetic_images_per_document": image_count,
        "synthetic_total_images": len(payloads) * image_count,
        "synthetic_text_profile": "Chinese title/caption/context/MMD paragraphs",
    }


def _configure_environment(
    args: argparse.Namespace,
    *,
    run_slug: str,
    collection: str,
    mongo_db: str,
    topic: str,
    attachment_prefix: str,
) -> None:
    """Must run before importing app modules because settings is a singleton."""
    runtime = SERVICE_ROOT / ".runtime" / f"independent_{run_slug}"
    runtime.mkdir(parents=True, exist_ok=True)
    common = {
        "KAFKA_BOOTSTRAP_SERVERS": DEFAULT_BOOTSTRAP,
        "KAFKA_TOPIC": topic,
        "KAFKA_GROUP_ID": str(args.consumer_group or f"codex_independent_{args.target}_{run_slug}"),
        "SYNC_WORKER_ROLE": args.target,
        "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
        "KAFKA_SASL_MECHANISM": "PLAIN",
        "KAFKA_SASL_USERNAME": KAFKA_SASL_USERNAME,
        "KAFKA_SASL_PASSWORD": KAFKA_SASL_PASSWORD,
        "KAFKA_CONSUME_LIMIT": "0",
        "KAFKA_BACKPRESSURE_ENABLED": "false",
        "KAFKA_DUMP_FILE": str(runtime / "kafka_messages.jsonl"),
        "KAFKA_ERROR_DUMP_FILE": str(runtime / "kafka_messages_error.jsonl"),
        "KAFKA_PROCESSED_DUMP_FILE": str(runtime / "kafka_messages_processed.jsonl"),
        "MONGO_URI": DEFAULT_MONGO_URI,
        "MONGO_DB": mongo_db,
        # Benchmark the intended high-throughput metadata path: successful
        # documents go directly to Milvus, while failures remain in Mongo.
        "MONGO_SUCCESS_AUDIT_ENABLED": "false",
        "MILVUS_HOST": DEFAULT_TARGET_HOST,
        "MILVUS_PORT": str(DEFAULT_TARGET_PORT),
        "MILVUS_COLLECTION": collection,
        "DOCUMENT_COLLECTION_NAME": collection if args.target == "metadata" else "",
        "FULLTEXT_CHUNK_COLLECTION": collection,
        "FULLTEXT_CHUNK_COLLECTION_NAME": collection if args.target == "fulltext" else "",
        "IMAGE_COLLECTION_NAME": collection if args.target == "image" else "scientific_images_unused_test",
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
        "EMBEDDING_MAX_CONCURRENCY": str(args.embedding_concurrency),
        "EMBEDDING_BATCH_SIZE": str(args.embedding_batch_size),
        # Each lane is an independent process.  Set the rate limiter here so
        # a parent .env cannot accidentally make all lanes inherit its image
        # budget and over- or under-drive the shared embedding endpoint.
        "EMBEDDING_RATE_LIMIT_ENABLED": "true" if float(args.embedding_text_rate) > 0 else "false",
        "EMBEDDING_LANE": args.target,
        "EMBEDDING_TEXT_RATE": str(max(float(args.embedding_text_rate), 0.0)),
        "EMBEDDING_TEXT_BURST": "32",
        "DOCUMENT_PIPELINE_CONCURRENCY": str(args.document_concurrency),
        "KAFKA_CONSUMER_MICRO_BATCH_SIZE": str(args.metadata_batch_size),
        "BATCH_MAX_SIZE": str(args.metadata_batch_size),
        "DOCUMENT_BULK_PIPELINE_ENABLED": "true" if args.metadata_mode == "metadata_bulk" else "false",
        "DOCUMENT_BULK_MILVUS_BATCH_SIZE": str(args.metadata_batch_size),
        "METADATA_BULK_TIMING_ENABLED": "true" if args.target == "metadata" else "false",
        "BENCHMARK_METADATA_MODE": args.metadata_mode,
        "BENCHMARK_METADATA_PIPELINE_INFLIGHT_BATCHES": str(args.metadata_pipeline_inflight_batches),
        "FULLTEXT_CHUNK_WRITE_MODE": "insert",
        "FULLTEXT_WORKER_BATCH_SIZE": "256",
        "FULLTEXT_WORKER_CONCURRENCY": str(args.fulltext_concurrency),
        "FULLTEXT_WORKER_MIN_TASK_TIMEOUT_SEC": "300",
        "IMAGE_WORKER_BATCH_SIZE": "256",
        "IMAGE_WORKER_CONCURRENCY": str(args.image_concurrency),
        "IMAGE_DOWNLOAD_CONCURRENCY": str(args.image_download_concurrency or args.image_concurrency),
        "ATTACHMENT_PREFIX": str(attachment_prefix or "").strip(),
        "IMAGE_MMD_HOT_PATH_MODE": args.image_mmd_hot_path_mode,
        "IMAGE_DOWNLOAD_LANE_ENABLED": "true",
        "IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC": "15",
        "IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC": "120",
        "IMAGE_SLOW_DOWNLOAD_CONCURRENCY": "8",
        "IMAGE_SLOW_RETRY_DELAY_SEC": "30",
        "IMAGE_TASK_MAX_ATTEMPTS": "3",
        "IMAGE_FEATURE_PIPELINE_ENABLED": "true",
        # The image primary vector is a required real visual embedding. The
        # three history-compatible text fields remain placeholders.
        "IMAGE_VISUAL_EMBED_ENABLED": "true",
        "IMAGE_MILVUS_WRITE_ENABLED": "false" if args.image_direct else "true",
        "IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES": str(
            args.image_direct_pipeline_inflight_batches
        ),
        # Every benchmark writes to a newly-created, isolated collection. Old
        # image cleanup has no value there and would measure a different path
        # from the intended first full replay.
        "IMAGE_DIRECT_STALE_CLEANUP_ENABLED": "false" if args.image_direct else "true",
        "IMAGE_MILVUS_WRITE_MODE": "insert_only",
        "IMAGE_VECTOR_SCHEMA_MODE": TEST_IMAGE_SCHEMA_MODE,
        "IMAGE_VL_CHAT_CONCURRENCY": "0",
        "IMAGE_VL_EMBED_CONCURRENCY": str(min(max(int(args.image_visual_concurrency or args.image_concurrency), 1), 32)),
        "IMAGE_TEXT_EMBED_CONCURRENCY": str(args.image_text_concurrency or args.image_concurrency),
        "IMAGE_MODEL_TOTAL_CONCURRENCY": str(max(int(args.image_model_total_concurrency), 1)),
        "IMAGE_PREPROCESS_CONCURRENCY": str(max(int(args.image_preprocess_concurrency), 1)),
        "IMAGE_TEXT_EMBED_BATCH_SIZE": str(args.image_text_batch_size),
        "IMAGE_TEXT_EMBED_BATCH_DELAY_MS": str(args.image_text_batch_delay_ms),
        "IMAGE_TEXT_EMBED_MAX_IN_FLIGHT_BATCHES": str(
            args.image_text_max_in_flight_batches
        ),
        "IMAGE_MILVUS_WRITE_CONCURRENCY": "16",
        "IMAGE_MILVUS_WRITE_BATCH_SIZE": "128",
        "IMAGE_VISIBILITY_CHECK_ENABLED": "false" if args.image_direct else "true",
        "IMAGE_SUCCESS_ARTIFACT_MODE": "summary",
        "IMAGE_SUCCESS_RESULT_MODE": "full" if args.target == "image" else "status_only",
        "LIVE_E2E_NO_DELETE": "true",
        "FULLTEXT_TASK_UPSERT_ENABLED": "false",
        "IMAGE_TASK_UPSERT_ENABLED": "false",
        "FULLTEXT_WORKER_ENABLED": "false",
        "IMAGE_WORKER_ENABLED": "false",
        "COMPENSATOR_ENABLED": "false",
    }
    os.environ.update(common)


def _milvus_count(collection: str, *, flush: bool = True) -> int:
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=60)
    try:
        if flush:
            client.flush(collection_name=collection)
        return int(client.get_collection_stats(collection_name=collection).get("row_count", 0))
    finally:
        client.close()


def _index_description(collection: str) -> list[dict[str, Any]]:
    client = MilvusClient(uri=f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}", timeout=60)
    try:
        rows: list[dict[str, Any]] = []
        for index_name in client.list_indexes(collection_name=collection):
            rows.append(client.describe_index(collection_name=collection, index_name=index_name))
        return rows
    finally:
        client.close()


def _task_counts(db_name: str, target: str) -> dict[str, int]:
    collection_name = "doc_fulltext_tasks" if target == "fulltext" else "doc_image_tasks"
    client = MongoClient(DEFAULT_MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        return {
            str(row.get("_id") or "UNKNOWN"): int(row.get("count") or 0)
            for row in client[db_name][collection_name].aggregate(
                [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
            )
        }
    finally:
        client.close()


def _consumer_config(group_id: str) -> dict[str, Any]:
    """Every fresh benchmark group must replay the preloaded isolated topic."""
    return {
        "bootstrap.servers": DEFAULT_BOOTSTRAP,
        "group.id": group_id,
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "PLAIN",
        "sasl.username": KAFKA_SASL_USERNAME,
        "sasl.password": KAFKA_SASL_PASSWORD,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 900_000,
    }


async def _consume_metadata(deadline: float, topic: str) -> Counter[str]:
    from app.services.batch_processor_existing import process_batch_existing_arch
    from app.services.kafka_consumer import parse_message

    consumer = Consumer(_consumer_config(os.environ["KAFKA_GROUP_ID"]))
    counters: Counter[str] = Counter()
    batch: list[Any] = []
    inflight: list[tuple[asyncio.Task[Any], Any]] = []
    inflight_limit = max(
        int(os.environ.get("BENCHMARK_METADATA_PIPELINE_INFLIGHT_BATCHES", "1")),
        1,
    )

    async def complete_oldest() -> None:
        task, last_record = inflight.pop(0)
        counters.update(_metadata_result(await task))
        consumer.commit(
            offsets=[
                TopicPartition(
                    last_record.topic,
                    last_record.partition,
                    last_record.offset + 1,
                )
            ],
            asynchronous=False,
        )

    async def dispatch(batch_to_process: list[Any]) -> None:
        if not batch_to_process:
            return
        task = asyncio.create_task(
            process_batch_existing_arch(
                batch_to_process,
                mode=os.environ.get("BENCHMARK_METADATA_MODE", "standard"),
            )
        )
        inflight.append((task, batch_to_process[-1]))
        if len(inflight) >= inflight_limit:
            await complete_oldest()

    try:
        consumer.subscribe([topic])
        while time.monotonic() < deadline:
            message = await asyncio.to_thread(consumer.poll, 0.2)
            if message is None:
                if batch:
                    await dispatch(batch)
                    batch = []
                if inflight:
                    await complete_oldest()
                continue
            record = parse_message(message)
            if record is None:
                continue
            counters["kafka_received"] += 1
            batch.append(record)
            if len(batch) >= int(os.environ.get("BATCH_MAX_SIZE", "512")):
                await dispatch(batch)
                batch = []
    finally:
        # A five-minute benchmark measures only work accepted inside its window.
        # Do not start a new trailing batch after the deadline merely to drain Kafka.
        if batch and time.monotonic() < deadline:
            await dispatch(batch)
        while inflight:
            await complete_oldest()
        consumer.close()
    return counters


def _metadata_result(result: Any) -> Counter[str]:
    return Counter(metadata_success=int(result.success), metadata_failed=int(result.fail), metadata_skipped=int(result.skip))


async def _consume_tasks(deadline: float, topic: str, target: str) -> Counter[str]:
    from app.services.kafka_consumer import parse_message
    if target == "fulltext":
        from app.services.fulltext_task_manager import extract_fulltext_tasks as extract
        from app.services.fulltext_task_repository import get_fulltext_task_repository as get_repo
    else:
        from app.services.image_task_manager import extract_image_tasks as extract
        from app.services.image_task_repository import get_image_task_repository as get_repo

    repository = get_repo()
    await repository.ensure_indexes()
    consumer = Consumer(_consumer_config(os.environ["KAFKA_GROUP_ID"]))
    counters: Counter[str] = Counter()
    try:
        consumer.subscribe([topic])
        while time.monotonic() < deadline:
            message = await asyncio.to_thread(consumer.poll, 0.2)
            if message is None:
                continue
            record = parse_message(message)
            if record is None:
                continue
            counters["kafka_received"] += 1
            tasks = extract(record.parsed_payload or {}) if record.parsed_payload else []
            result = await repository.upsert_tasks(tasks)
            counters["tasks_extracted"] += len(tasks)
            counters["tasks_upserted"] += int(result.get("upserted") or 0)
            consumer.commit(asynchronous=False)
    finally:
        consumer.close()
    return counters


async def _consume_direct_images(
    deadline: float,
    topic: str,
    *,
    batch_size: int,
    pipeline_inflight_batches: int,
) -> tuple[Counter[str], dict[str, Any]]:
    """Run the production direct Kafka image pipeline against an isolated topic."""
    from app.services.independent_kafka_worker import IndependentKafkaWorker
    from app.services.kafka_consumer import parse_message
    from app.services.mongo_state_manager import get_mongo_state_manager

    consumer = Consumer(_consumer_config(os.environ["KAFKA_GROUP_ID"]))
    processor = IndependentKafkaWorker("image", state_manager=get_mongo_state_manager())
    counters: Counter[str] = Counter()
    records: list[Any] = []
    timing_values: dict[str, list[float]] = defaultdict(list)
    timing_items: Counter[str] = Counter()
    metrics_lock = Lock()

    def observe(name: str, started: float, *, items: int = 0) -> None:
        with metrics_lock:
            timing_values[name].append((time.perf_counter() - started) * 1000)
            timing_items[name] += max(int(items), 0)

    # Profile boundary calls only in the benchmark process.  Production code
    # keeps no per-item timing allocations on the direct path.
    image_worker = processor._get_image_worker()
    original_process_task = image_worker.process_task

    async def timed_process_task(task: dict[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            return await original_process_task(task)
        finally:
            observe("image_task", started)

    image_worker.process_task = timed_process_task
    original_download_bytes = image_worker.downloader.download_bytes

    async def timed_download_bytes(relative_url: str, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original_download_bytes(relative_url, *args, **kwargs)
        finally:
            observe("image_download", started)

    image_worker.downloader.download_bytes = timed_download_bytes
    original_feature_pipeline = image_worker._run_feature_pipeline_with_timeout

    async def timed_feature_pipeline(**kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original_feature_pipeline(**kwargs)
        finally:
            observe("image_feature_pipeline", started)

    image_worker._run_feature_pipeline_with_timeout = timed_feature_pipeline
    model_client = getattr(getattr(image_worker, "_resource_pool", None), "model_client", None)
    if model_client is not None:
        original_embed_texts = model_client.embed_texts

        async def timed_embed_texts(texts: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original_embed_texts(texts)
            finally:
                observe("embedding_api_batch", started, items=len(texts))

        model_client.embed_texts = timed_embed_texts
        original_embed_image = model_client.embed_image

        async def timed_embed_image(image_bytes: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original_embed_image(image_bytes, **kwargs)
            finally:
                observe("visual_embedding_api", started, items=1)

        model_client.embed_image = timed_embed_image

    image_service = processor._get_image_service()
    original_replace_documents = image_service.replace_images_for_documents

    def timed_replace_documents(collection_name: str, documents: dict[str, Any], **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_replace_documents(collection_name, documents, **kwargs)
        finally:
            observe(
                "milvus_document_batch",
                started,
                items=sum(len(images) for images in documents.values()),
            )

    image_service.replace_images_for_documents = timed_replace_documents
    original_upsert_records = image_service._upsert_records

    def timed_upsert_records(collection_name: str, image_records: list[dict[str, Any]]) -> Any:
        started = time.perf_counter()
        try:
            result = original_upsert_records(collection_name, image_records)
            with metrics_lock:
                counters["direct_milvus_rows"] += len(image_records)
            return result
        finally:
            observe("milvus_vector_upsert", started, items=len(image_records))

    image_service._upsert_records = timed_upsert_records
    original_delete = image_service.client.delete

    def timed_delete(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_delete(*args, **kwargs)
        finally:
            observe("milvus_stale_cleanup", started)

    image_service.client.delete = timed_delete

    inflight: list[tuple[asyncio.Task[Any], list[Any], float]] = []
    started_at = time.monotonic()
    window_snapshot: dict[str, Any] = {}

    async def complete_oldest() -> None:
        if not inflight:
            return
        task, active, started = inflight.pop(0)
        result = await task
        observe("kafka_micro_batch", started, items=len(active))
        counters["direct_success"] += int(result.success)
        counters["direct_failed"] += int(result.fail)
        counters["direct_skipped"] += int(result.skip)
        if result.commit_safe:
            from app.services.kafka_consumer import _build_commit_offsets

            consumer.commit(offsets=_build_commit_offsets(active), asynchronous=False)
            counters["direct_commit_safe_batches"] += 1
        else:
            counters["direct_uncommitted_batches"] += 1

    async def dispatch() -> None:
        if not records:
            return
        active = list(records)
        records.clear()
        started = time.perf_counter()
        inflight.append((asyncio.create_task(processor.process_batch(active)), active, started))
        if len(inflight) >= max(int(pipeline_inflight_batches), 1):
            if not window_snapshot and time.monotonic() >= deadline:
                window_snapshot.update(
                    {
                        "elapsed_seconds": round(max(deadline - started_at, 0.0), 3),
                        "counters": dict(counters),
                    }
                )
            await complete_oldest()

    try:
        consumer.subscribe([topic])
        while time.monotonic() < deadline:
            message = await asyncio.to_thread(consumer.poll, 0.2)
            if message is None:
                if records:
                    await dispatch()
                continue
            record = parse_message(message)
            if record is None:
                continue
            counters["kafka_received"] += 1
            records.append(record)
            if len(records) >= max(int(batch_size), 1):
                await dispatch()
        window_snapshot = {
            **window_snapshot,
            "elapsed_seconds": round(
                float(window_snapshot.get("elapsed_seconds") or max(deadline - started_at, 0.0)),
                3,
            ),
            "counters": window_snapshot.get("counters") or dict(counters),
        }
    finally:
        if records and time.monotonic() < deadline:
            await dispatch()
        while inflight:
            await complete_oldest()
        consumer.close()
        await processor.close()
    timing = {}
    for name, values in sorted(timing_values.items()):
        ordered = sorted(values)
        timing[name] = {
            "count": len(ordered),
            "items": int(timing_items[name]),
            "avg_ms": round(statistics.fmean(ordered), 3),
            "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
            "total_ms": round(sum(ordered), 3),
        }
    timing["_window"] = window_snapshot
    return counters, timing


async def _consume_direct_fulltext(
    deadline: float,
    topic: str,
    *,
    batch_size: int,
    pipeline_inflight_batches: int,
) -> tuple[Counter[str], dict[str, Any]]:
    """Run the production direct Kafka fulltext worker without Mongo task leasing."""
    from app.services.independent_kafka_worker import IndependentKafkaWorker
    from app.services.kafka_consumer import _build_commit_offsets, parse_message
    from app.services.mongo_state_manager import get_mongo_state_manager

    consumer = Consumer(_consumer_config(os.environ["KAFKA_GROUP_ID"]))
    processor = IndependentKafkaWorker("fulltext", state_manager=get_mongo_state_manager())
    counters: Counter[str] = Counter()
    records: list[Any] = []
    timing_values: dict[str, list[float]] = defaultdict(list)
    timing_items: Counter[str] = Counter()
    metrics_lock = Lock()

    def observe_duration(name: str, duration_ms: Any, *, items: int = 0) -> None:
        try:
            value = float(duration_ms)
        except (TypeError, ValueError):
            return
        if value < 0:
            return
        with metrics_lock:
            timing_values[name].append(value)
            timing_items[name] += max(int(items), 0)

    fulltext_worker = processor._get_fulltext_worker()
    original_process_task = fulltext_worker.process_task

    async def timed_process_task(task: dict[str, Any]) -> Any:
        result = await original_process_task(task)
        for stage, duration_ms in (getattr(result, "timings_ms", {}) or {}).items():
            observe_duration(f"fulltext_{stage}", duration_ms)
        if getattr(result, "success", False):
            counters["direct_chunks_written"] += int(getattr(result, "chunk_count", 0) or 0)
        return result

    fulltext_worker.process_task = timed_process_task
    inflight: list[tuple[asyncio.Task[Any], list[Any], float]] = []

    async def complete_oldest() -> None:
        if not inflight:
            return
        task, active, started = inflight.pop(0)
        result = await task
        observe_duration("kafka_micro_batch", (time.perf_counter() - started) * 1000, items=len(active))
        counters["direct_success"] += int(result.success)
        counters["direct_failed"] += int(result.fail)
        counters["direct_skipped"] += int(result.skip)
        if result.commit_safe:
            consumer.commit(offsets=_build_commit_offsets(active), asynchronous=False)
            counters["direct_commit_safe_batches"] += 1
        else:
            counters["direct_uncommitted_batches"] += 1

    async def dispatch() -> None:
        if not records:
            return
        active = list(records)
        records.clear()
        started = time.perf_counter()
        inflight.append((asyncio.create_task(processor.process_batch(active)), active, started))
        if len(inflight) >= max(int(pipeline_inflight_batches), 1):
            await complete_oldest()

    try:
        consumer.subscribe([topic])
        while time.monotonic() < deadline:
            message = await asyncio.to_thread(consumer.poll, 0.2)
            if message is None:
                if records:
                    await dispatch()
                continue
            record = parse_message(message)
            if record is None:
                continue
            counters["kafka_received"] += 1
            records.append(record)
            if len(records) >= max(int(batch_size), 1):
                await dispatch()
    finally:
        if records and time.monotonic() < deadline:
            await dispatch()
        while inflight:
            await complete_oldest()
        consumer.close()
        await processor.close()

    timing: dict[str, Any] = {}
    for name, values in sorted(timing_values.items()):
        ordered = sorted(values)
        timing[name] = {
            "count": len(ordered),
            "items": int(timing_items[name]),
            "avg_ms": round(statistics.fmean(ordered), 3),
            "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
            "total_ms": round(sum(ordered), 3),
        }
    return counters, timing


async def _run_worker(deadline: float, target: str) -> tuple[Counter[str], dict[str, Any]]:
    if target == "fulltext":
        from app.services.fulltext_worker_runner import FullTextWorkerRunner
        runner = FullTextWorkerRunner(batch_size=256, concurrency=int(os.environ["FULLTEXT_WORKER_CONCURRENCY"]))
    else:
        from app.services.image_worker_runner import ImageWorkerRunner
        runner = ImageWorkerRunner(batch_size=256, concurrency=int(os.environ["IMAGE_WORKER_CONCURRENCY"]))
    counters: Counter[str] = Counter()
    timing_samples: list[dict[str, Any]] = []
    try:
        while time.monotonic() < deadline:
            result = await runner.run_once(worker_id=f"benchmark-{target}")
            if not result.get("claimed"):
                await asyncio.sleep(0.05)
                continue
            for item in result.get("results") or []:
                counters["worker_success" if item.get("processed") else "worker_failed"] += 1
                worker_result = item.get("result") or {}
                timings = worker_result.get("timings_ms") if isinstance(worker_result, dict) else None
                if isinstance(timings, dict):
                    timing_samples.append(timings)
    finally:
        worker = getattr(runner, "worker", None)
        close = getattr(worker, "close", None)
        if close is not None:
            await close()
    return counters, _summarize_timing_samples(timing_samples)


def _summarize_timing_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[float]] = {}
    for sample in samples:
        for stage, value in sample.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                by_stage.setdefault(str(stage), []).append(number)

    summary: dict[str, Any] = {"sample_count": len(samples), "stages": {}}
    for stage, values in sorted(by_stage.items()):
        ordered = sorted(values)
        summary["stages"][stage] = {
            "count": len(ordered),
            "total_ms": round(sum(ordered), 3),
            "avg_ms": round(statistics.fmean(ordered), 3),
            "p50_ms": round(statistics.median(ordered), 3),
            "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
        }
    return summary


def _image_timing_summary(db_name: str) -> dict[str, Any]:
    client = MongoClient(DEFAULT_MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        samples: list[dict[str, Any]] = []
        for row in client[db_name]["doc_image_tasks"].find(
            {"status": "SUCCESS"},
            {"result_summary.image_diagnostics": 1},
        ):
            diagnostics = ((row.get("result_summary") or {}).get("image_diagnostics") or {})
            sample: dict[str, Any] = {}
            stage_timings = diagnostics.get("feature_stage_timings_ms")
            if isinstance(stage_timings, dict):
                sample.update(stage_timings)
            for key, target in (("milvus_write_ms", "milvus_write"), ("embed_text_ms", "embed_text")):
                if key in diagnostics:
                    sample[target] = diagnostics[key]
            if sample:
                samples.append(sample)
        return _summarize_timing_samples(samples)
    finally:
        client.close()


async def run_benchmark(args: argparse.Namespace, *, topic: str, mongo_db: str, collection: str) -> dict[str, Any]:
    from app.config import settings
    from app.services.milvus_chunk_service import MilvusChunkService
    from app.services.milvus_image_service import MilvusImageService
    from app.services.milvus_service import get_milvus_service
    from app.services.mongo_state_manager import get_mongo_state_manager
    from app.services.batch_processor_existing import (
        get_metadata_bulk_timing_snapshot,
        reset_metadata_bulk_timing,
    )

    mongo = get_mongo_state_manager()
    await mongo.connect()
    if args.target == "metadata":
        reset_metadata_bulk_timing()
    if args.target == "metadata":
        get_milvus_service().init_collection(collection, settings.VECTOR_DIM)
    elif args.target == "fulltext":
        MilvusChunkService().init_chunk_collection(collection, settings.VECTOR_DIM)
    else:
        MilvusImageService().init_image_collection(collection)

    deadline = time.monotonic() + args.duration_seconds
    started = time.monotonic()
    try:
        if args.target == "metadata":
            counters = await _consume_metadata(deadline, topic)
        elif args.target == "image" and args.image_direct:
            counters, direct_pipeline_timing = await _consume_direct_images(
                deadline,
                topic,
                batch_size=args.image_direct_batch_size,
                pipeline_inflight_batches=args.image_direct_pipeline_inflight_batches,
            )
            worker_pipeline_timing = {}
        elif args.target == "fulltext" and args.fulltext_direct:
            counters, direct_pipeline_timing = await _consume_direct_fulltext(
                deadline,
                topic,
                batch_size=args.fulltext_direct_batch_size,
                pipeline_inflight_batches=args.fulltext_direct_pipeline_inflight_batches,
            )
            worker_pipeline_timing = {}
        else:
            consumer_task = asyncio.create_task(_consume_tasks(deadline, topic, args.target))
            worker_task = asyncio.create_task(_run_worker(deadline, args.target))
            consumer_counts, worker_result = await asyncio.gather(consumer_task, worker_task)
            worker_counts, worker_pipeline_timing = worker_result
            counters = consumer_counts + worker_counts
        elapsed = time.monotonic() - started
        # IVFPQ index building can keep a flush call blocked well after the
        # write request has returned. Direct-image throughput is therefore
        # measured from successful upsert rows, with a non-blocking collection
        # count kept only as an observation of current visibility.
        observed_rows = _milvus_count(
            collection,
            flush=not (args.target == "image" and args.image_direct),
        )
        direct_timing = (
            direct_pipeline_timing
            if (args.target == "image" and args.image_direct)
            or (args.target == "fulltext" and args.fulltext_direct)
            else {}
        )
        window_timing = direct_timing.pop("_window", {}) if isinstance(direct_timing, dict) else {}
        rows = (
            int(counters.get("direct_milvus_rows") or 0)
            if args.target == "image" and args.image_direct
            else observed_rows
        )
        return {
            "elapsed_seconds": round(elapsed, 3),
            "counters": dict(counters),
            "milvus_rows": rows,
            "milvus_observed_rows": observed_rows,
            "milvus_rows_per_hour": round(rows * 3600 / elapsed, 2) if elapsed else 0,
            "task_statuses": _task_counts(mongo_db, args.target) if args.target != "metadata" else {},
            "index_descriptions": _index_description(collection),
            "metadata_pipeline_timing": (
                get_metadata_bulk_timing_snapshot() if args.target == "metadata" else {}
            ),
            "worker_pipeline_timing": (
                _select_image_timing_summary(mongo_db, worker_pipeline_timing)
                if args.target == "image"
                else worker_pipeline_timing if args.target == "fulltext" else {}
            ),
            "direct_pipeline_timing": (
                direct_timing
            ),
            "window": window_timing,
        }
    finally:
        await mongo.close()


def _select_image_timing_summary(
    mongo_db: str,
    worker_pipeline_timing: dict[str, Any],
) -> dict[str, Any]:
    persisted = _image_timing_summary(mongo_db)
    if int(persisted.get("sample_count") or 0) > 0:
        return persisted
    return worker_pipeline_timing


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0 or args.messages <= 0:
        raise ValueError("duration and messages must be positive")
    if args.synthetic and args.reuse_topic:
        raise ValueError("--synthetic cannot be combined with --reuse-topic")
    if args.synthetic and (args.target != "image" or not args.image_direct):
        raise ValueError("--synthetic requires --target image --image-direct")
    if args.fulltext_direct and args.target != "fulltext":
        raise ValueError("--fulltext-direct requires --target fulltext")
    if args.synthetic_images_per_document <= 0:
        raise ValueError("--synthetic-images-per-document must be positive")
    if args.topic and args.reuse_topic:
        raise ValueError("--topic cannot be combined with --reuse-topic")
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_slug = _slug(f"{run_id}_{args.target}")
    topic = str(args.topic or args.reuse_topic or "").strip() or f"VECTOR_JOURNAL_ARTICLE_IVFPQ_{args.target.upper()}_{run_slug}_TEST"[:249]
    collection = str(args.collection_name or f"scientific_{args.target}_{run_slug}_test")
    mongo_db = str(args.mongo_db or f"nstl_vs_independent_{run_slug}_test")
    out_dir = SERVICE_ROOT / "out" / f"codex_independent_kafka_{run_slug}"
    report_path = out_dir / "report.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    attachment_server: _SyntheticAttachmentServer | None = None
    try:
        attachment_prefix = str(args.image_attachment_prefix or "").strip()
        if args.synthetic:
            attachment_server = _SyntheticAttachmentServer().__enter__()
            attachment_prefix = attachment_server.prefix

        started_at = _now()
        if not args.skip_mongo_reset:
            reset_mongo_database(mongo_db)
        if not args.skip_collection_reset:
            reset_milvus_collections({args.target: collection})
        if args.reuse_topic:
            source_summary = {
                "history_files_scanned": 0,
                "unique_real_payloads": 0,
                "replayed_payloads": 0,
                "repeated_payloads": 0,
                "topic_reused": topic,
            }
            publish = {
                "topic": topic,
                "messages_published": 0,
                "publish_duration_seconds": 0,
                "publish_mode": "reused_isolated_topic",
            }
        else:
            ensure_test_topic(topic)
            if args.synthetic:
                payloads, source_summary = _build_synthetic_image_payloads(
                    messages=args.messages,
                    images_per_document=args.synthetic_images_per_document,
                )
            else:
                history_root = (
                    SERVICE_ROOT / ".runtime"
                    if args.include_runtime_history
                    else SERVICE_ROOT / ".runtime-history-disabled"
                )
                payloads, source_summary = load_real_payloads(
                    args.source_jsonl,
                    history_root=history_root,
                    limit=args.messages,
                )
            publish = produce_messages(
                source_payloads=payloads,
                topic=topic,
                run_id=run_slug,
                # Historical sources have fewer unique image documents than a
                # five-minute run needs. Keep real URLs/content but isolate
                # image primary keys so repeated source messages create rows.
                rewrite_image_ids=args.target == "image" and args.image_direct,
            )
        _configure_environment(
            args,
            run_slug=run_slug,
            collection=collection,
            mongo_db=mongo_db,
            topic=topic,
            attachment_prefix=attachment_prefix,
        )
        result = asyncio.run(run_benchmark(args, topic=topic, mongo_db=mongo_db, collection=collection))
        report = {
            "started_at": started_at, "finished_at": _now(), "run_id": run_id, "target": args.target,
            "duration_requested_seconds": args.duration_seconds,
            "test_only_isolation": {"topic": topic, "consumer_group": os.environ["KAFKA_GROUP_ID"], "mongo_db": mongo_db, "collection": collection},
            "index": {"type": "IVF_PQ", "nlist": 64, "m": 64, "nbits": 8},
            "image_schema_mode": TEST_IMAGE_SCHEMA_MODE if args.target == "image" else None,
            "tuning": {"metadata_mode": args.metadata_mode, "metadata_batch_size": args.metadata_batch_size, "embedding_batch_size": args.embedding_batch_size, "metadata_pipeline_inflight_batches": args.metadata_pipeline_inflight_batches, "document_concurrency": args.document_concurrency, "fulltext_concurrency": args.fulltext_concurrency, "fulltext_direct": args.fulltext_direct, "fulltext_direct_batch_size": args.fulltext_direct_batch_size, "fulltext_direct_pipeline_inflight_batches": args.fulltext_direct_pipeline_inflight_batches, "image_concurrency": args.image_concurrency, "image_direct": args.image_direct, "image_direct_batch_size": args.image_direct_batch_size, "image_direct_pipeline_inflight_batches": args.image_direct_pipeline_inflight_batches, "image_attachment_prefix": attachment_prefix, "image_mmd_hot_path_mode": args.image_mmd_hot_path_mode, "image_text_batch_size": args.image_text_batch_size, "image_text_batch_delay_ms": args.image_text_batch_delay_ms, "image_text_max_in_flight_batches": args.image_text_max_in_flight_batches, "image_model_total_concurrency": args.image_model_total_concurrency, "image_preprocess_concurrency": args.image_preprocess_concurrency, "embedding_concurrency": args.embedding_concurrency, "stale_cleanup_disabled_for_new_test_collection": bool(args.image_direct), "predelete_disabled_for_unique_test_ids": True},
            "input": {**source_summary, **publish, "consume_limit": 0},
            "result": result,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        if attachment_server is not None:
            attachment_server.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
