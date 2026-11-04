#!/usr/bin/env python3
"""Run the three direct Kafka lanes concurrently against isolated test targets."""

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

from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymongo import MongoClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "vector_sync_service"
DRIVER = SERVICE_ROOT / "scripts" / "run_independent_kafka_collection_benchmark.py"

# Topic names are generated per run unless --reuse-topic is supplied.  This
# keeps the test isolated even when an earlier run left a topic behind.
TOPIC_PREFIX = "VECTOR_JOURNAL_ARTICLE_CODEX_40M"
MESSAGE_COUNTS = {"metadata": 40_000, "fulltext": 5_000, "image": 4_000}
TEXT_RATES = {"metadata": 80, "fulltext": 150, "image": 24}
MONGO_COLLECTIONS = (
    "doc_sync_status",
    "doc_sync_failures",
    "doc_sync_attempts",
    "doc_fulltext_tasks",
    "doc_image_tasks",
    "doc_sync_artifacts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-wait-seconds", type=int, default=410)
    parser.add_argument("--source-jsonl", type=Path, default=REPO_ROOT / "bakkafkadata" / "31" / "kafka_messages.jsonl")
    parser.add_argument("--messages-metadata", type=int, default=MESSAGE_COUNTS["metadata"])
    parser.add_argument("--messages-fulltext", type=int, default=MESSAGE_COUNTS["fulltext"])
    parser.add_argument("--messages-image", type=int, default=MESSAGE_COUNTS["image"])
    parser.add_argument(
        "--reuse-topic",
        action="store_true",
        help="Use the legacy preloaded topic names instead of publishing a new isolated payload set.",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if len(normalized) > 30:
        normalized = f"{normalized[:20]}_{normalized[-9:]}"
    return normalized[:30] or "three_lane"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _lane_command(
    lane: str,
    *,
    duration_seconds: int,
    run_id: str,
    topic: str,
    messages: int,
    source_jsonl: Path,
    reuse_topic: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(DRIVER),
        "--target",
        lane,
        "--duration-seconds",
        str(duration_seconds),
        "--messages",
        str(messages),
        "--run-id",
        run_id,
        "--embedding-text-rate",
        str(TEXT_RATES[lane]),
    ]
    if reuse_topic:
        command.extend(["--reuse-topic", topic])
    else:
        command.extend(["--topic", topic, "--source-jsonl", str(source_jsonl)])
    if lane == "metadata":
        command.extend(
            [
                "--metadata-mode",
                "metadata_bulk",
                "--metadata-batch-size",
                "1024",
                "--embedding-batch-size",
                "64",
                "--document-concurrency",
                "48",
                "--metadata-pipeline-inflight-batches",
                "2",
                "--embedding-concurrency",
                "48",
            ]
        )
    elif lane == "fulltext":
        command.extend(
            [
                "--fulltext-direct",
                "--fulltext-direct-batch-size",
                "32",
                "--fulltext-direct-pipeline-inflight-batches",
                "1",
                "--fulltext-concurrency",
                "64",
                "--embedding-batch-size",
                "64",
                "--image-attachment-prefix",
                "http://127.0.0.1:12001/",
            ]
        )
    else:
        command.extend(
            [
                "--image-direct",
                "--image-direct-batch-size",
                "18",
                "--image-direct-pipeline-inflight-batches",
                "1",
                "--image-concurrency",
                "96",
                "--image-text-batch-size",
                "32",
                "--image-text-batch-delay-ms",
                "250",
                "--image-text-max-in-flight-batches",
                "16",
                "--image-mmd-hot-path-mode",
                "required",
                "--image-attachment-prefix",
                "http://127.0.0.1:12001/",
            ]
        )
    return command


def _report_path(run_id: str, lane: str) -> Path:
    return SERVICE_ROOT / "out" / f"codex_independent_kafka_{_slug(f'{run_id}_{lane}')}" / "report.json"


def _read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"report_missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _mongo_counts(mongo_uri: str, database: str) -> dict[str, int]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    try:
        db = client[database]
        return {name: int(db[name].count_documents({})) for name in MONGO_COLLECTIONS}
    finally:
        client.close()


def _verified_milvus_rows(milvus_uri: str, collection: str) -> int:
    client = MilvusClient(uri=milvus_uri, timeout=180)
    try:
        client.flush(collection_name=collection)
        stats = client.get_collection_stats(collection_name=collection)
        return int(stats.get("row_count", 0))
    finally:
        client.close()


def _summary(report: dict[str, Any], *, mongo_uri: str, milvus_uri: str) -> dict[str, Any]:
    if report.get("report_missing"):
        return report
    isolation = report["test_only_isolation"]
    result = report["result"]
    collection = str(isolation["collection"])
    return {
        "collection": collection,
        "mongo_db": isolation["mongo_db"],
        "elapsed_seconds": result["elapsed_seconds"],
        "counters": result["counters"],
        "reported_milvus_rows": result["milvus_rows"],
        "verified_milvus_rows_after_flush": _verified_milvus_rows(milvus_uri, collection),
        "milvus_rows_per_hour": result["milvus_rows_per_hour"],
        "mongo_counts": _mongo_counts(mongo_uri, str(isolation["mongo_db"])),
        "report_path": str(_report_path(str(report["run_id"]), str(report["target"]))),
    }


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0 or args.max_wait_seconds <= 0:
        raise ValueError("duration and max wait must be positive")

    load_dotenv(SERVICE_ROOT / ".env", override=False)
    mongo_uri = str(os.environ["MONGO_URI"])
    milvus_host = str(os.environ["MILVUS_HOST"])
    milvus_port = str(os.environ.get("MILVUS_PORT", "19530"))
    milvus_uri = f"http://{milvus_host}:{milvus_port}"

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("three_lane_%Y%m%dT%H%M%SZ")
    if args.reuse_topic:
        topics = {
            "metadata": "VECTOR_JOURNAL_ARTICLE_IVFPQ_METADATA_VALID_REAL5M_TEST",
            "fulltext": "VECTOR_JOURNAL_ARTICLE_IVFPQ_FULLTEXT_real_5m_fulltext_direct_b32_fu_TEST",
            "image": "VECTOR_JOURNAL_ARTICLE_IVFPQ_IMAGE_real_5m_full_b18_unique_image_TEST",
        }
    else:
        topic_slug = _slug(run_id).upper()[:80]
        topics = {
            lane: f"{TOPIC_PREFIX}_{topic_slug}_{lane.upper()}"[:249]
            for lane in ("metadata", "fulltext", "image")
        }
    message_counts = {
        "metadata": max(int(args.messages_metadata), 1),
        "fulltext": max(int(args.messages_fulltext), 1),
        "image": max(int(args.messages_image), 1),
    }
    out_dir = SERVICE_ROOT / "out" / f"codex_three_lane_concurrent_{_slug(run_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    handles: dict[str, tuple[Any, Any]] = {}
    processes: dict[str, subprocess.Popen[Any]] = {}
    try:
        for lane in ("metadata", "fulltext", "image"):
            stdout = (out_dir / f"{lane}.stdout.log").open("w", encoding="utf-8")
            stderr = (out_dir / f"{lane}.stderr.log").open("w", encoding="utf-8")
            handles[lane] = (stdout, stderr)
            processes[lane] = subprocess.Popen(
                _lane_command(
                    lane,
                    duration_seconds=args.duration_seconds,
                    # The child appends its target to this base id when it
                    # builds its isolated Mongo/collection/report slug.
                    run_id=run_id,
                    topic=topics[lane],
                    messages=message_counts[lane],
                    source_jsonl=args.source_jsonl,
                    reuse_topic=args.reuse_topic,
                ),
                cwd=SERVICE_ROOT,
                stdout=stdout,
                stderr=stderr,
            )

        deadline = time.monotonic() + args.max_wait_seconds
        exit_codes: dict[str, int | None] = {}
        for lane, process in processes.items():
            remaining = max(deadline - time.monotonic(), 1)
            try:
                exit_codes[lane] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_codes[lane] = None
        reports = {
            lane: _read_report(_report_path(run_id, lane))
            for lane in processes
        }
        report = {
            "started_at": started_at,
            "finished_at": _now(),
            "run_id": run_id,
            "duration_requested_seconds": args.duration_seconds,
            "test_only_topics": topics,
            "message_counts": message_counts,
            "publish_mode": "reused_isolated_topic" if args.reuse_topic else "new_real_payload_topics",
            "exit_codes": exit_codes,
            "lanes": {
                lane: _summary(item, mongo_uri=mongo_uri, milvus_uri=milvus_uri)
                for lane, item in reports.items()
            },
        }
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(code == 0 for code in exit_codes.values()) else 1
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for stdout, stderr in handles.values():
            stdout.close()
            stderr.close()


if __name__ == "__main__":
    raise SystemExit(main())
