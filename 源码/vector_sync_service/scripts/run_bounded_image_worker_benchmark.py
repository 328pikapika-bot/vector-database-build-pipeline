#!/usr/bin/env python3
"""Run a bounded real image-worker benchmark over preloaded test tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--mongo-db", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=96)
    return parser.parse_args()


def _milvus_client() -> MilvusClient:
    return MilvusClient(
        uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ['MILVUS_PORT']}",
        timeout=60,
    )


def _milvus_rows(collection: str) -> int:
    client = _milvus_client()
    try:
        client.flush(collection_name=collection)
        return int(client.get_collection_stats(collection_name=collection).get("row_count", 0))
    finally:
        client.close()


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_stage: dict[str, list[float]] = {}
    for sample in samples:
        for stage, value in sample.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                values_by_stage.setdefault(str(stage), []).append(number)
    stages: dict[str, Any] = {}
    for stage, values in sorted(values_by_stage.items()):
        ordered = sorted(values)
        stages[stage] = {
            "count": len(ordered),
            "avg_ms": round(statistics.fmean(ordered), 3),
            "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
        }
    return {"sample_count": len(samples), "stages": stages}


class _Timer:
    def __init__(self) -> None:
        self.values_ms: list[float] = []
        self.items = 0

    def add(self, elapsed_seconds: float, *, items: int = 0) -> None:
        self.values_ms.append(elapsed_seconds * 1000)
        self.items += max(int(items), 0)

    def summary(self) -> dict[str, Any]:
        if not self.values_ms:
            return {"count": 0, "items": self.items}
        ordered = sorted(self.values_ms)
        return {
            "count": len(ordered),
            "items": self.items,
            "total_ms": round(sum(ordered), 3),
            "avg_ms": round(statistics.fmean(ordered), 3),
            "p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 3),
        }


def _instrument_worker(runner: Any) -> tuple[dict[str, _Timer], list[dict[str, Any]]]:
    """Time boundary calls without changing the worker implementation."""
    timers = {
        name: _Timer()
        for name in (
            "mongo_claim_one",
            "mongo_success_delete",
            "mongo_failure_update",
            "embedding_api_batch",
            "milvus_batch_flush",
            "milvus_insert",
            "milvus_visibility_check",
            "worker_batch",
        )
    }
    batch_samples: list[dict[str, Any]] = []
    repository = runner.repository

    def wrap_async(instance: Any, method_name: str, timer_name: str) -> None:
        original = getattr(instance, method_name)

        async def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original(*args, **kwargs)
            finally:
                timers[timer_name].add(time.perf_counter() - started)

        setattr(instance, method_name, measured)

    wrap_async(repository, "claim_next_task", "mongo_claim_one")
    wrap_async(repository, "mark_task_success", "mongo_success_delete")
    wrap_async(repository, "mark_task_success_minimal", "mongo_success_delete")
    wrap_async(repository, "mark_task_failure", "mongo_failure_update")

    worker = runner.worker
    resource_pool = getattr(worker, "_resource_pool", None)
    model_client = getattr(resource_pool, "model_client", None)
    if model_client is not None:
        original_embed_texts = model_client.embed_texts

        async def measured_embed_texts(texts: list[str]) -> list[list[float]]:
            started = time.perf_counter()
            try:
                return await original_embed_texts(texts)
            finally:
                timers["embedding_api_batch"].add(
                    time.perf_counter() - started,
                    items=len(texts),
                )

        model_client.embed_texts = measured_embed_texts

    writer = getattr(worker, "image_writer", None)
    if writer is not None:
        original_flush_batch = writer._flush_batch

        async def measured_flush_batch(batch: list[Any]) -> None:
            started = time.perf_counter()
            try:
                return await original_flush_batch(batch)
            finally:
                timers["milvus_batch_flush"].add(
                    time.perf_counter() - started,
                    items=len(batch),
                )

        writer._flush_batch = measured_flush_batch
        original_insert_images = writer.service.insert_images

        def measured_insert_images(collection_name: str, images: list[Any]) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                return original_insert_images(collection_name, images)
            finally:
                timers["milvus_insert"].add(
                    time.perf_counter() - started,
                    items=len(images),
                )

        writer.service.insert_images = measured_insert_images
        original_verify_visibility = writer._verify_visibility

        def measured_verify_visibility(collection_name: str, images: list[Any]) -> None:
            started = time.perf_counter()
            try:
                return original_verify_visibility(collection_name, images)
            finally:
                timers["milvus_visibility_check"].add(
                    time.perf_counter() - started,
                    items=len(images),
                )

        writer._verify_visibility = measured_verify_visibility

    return timers, batch_samples


def _task_statuses(db_name: str) -> dict[str, int]:
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=5000)
    try:
        rows = client[db_name]["doc_image_tasks"].aggregate(
            [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        )
        return {str(row.get("_id") or "UNKNOWN"): int(row.get("count") or 0) for row in rows}
    finally:
        client.close()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from app.services.image_task_repository import get_image_task_repository
    from app.services.image_worker_runner import ImageWorkerRunner
    from app.services.milvus_image_service import MilvusImageService
    from app.services.mongo_state_manager import get_mongo_state_manager

    mongo = get_mongo_state_manager()
    await mongo.connect()
    service = MilvusImageService()
    try:
        service.init_image_collection(args.collection)
    finally:
        service.close()

    repository = get_image_task_repository()
    await repository.ensure_indexes()
    before = _milvus_rows(args.collection)
    runner = ImageWorkerRunner(batch_size=args.batch_size, concurrency=args.concurrency)
    timers, batch_samples = _instrument_worker(runner)
    counters: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + args.duration_seconds
    try:
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            batch_started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    runner.run_once(worker_id="bounded-image-benchmark"),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                timers["worker_batch"].add(time.perf_counter() - batch_started)
                break
            batch_elapsed = time.perf_counter() - batch_started
            result_items = list(result.get("results") or [])
            timers["worker_batch"].add(batch_elapsed, items=len(result_items))
            batch_samples.append(
                {
                    "elapsed_ms": round(batch_elapsed * 1000, 3),
                    "claimed": bool(result.get("claimed")),
                    "result_count": len(result_items),
                }
            )
            if not result.get("claimed"):
                await asyncio.sleep(0.05)
                continue
            for item in result_items:
                counters["worker_success" if item.get("processed") else "worker_failed"] += 1
                worker_result = item.get("result") or {}
                timings = worker_result.get("timings_ms") if isinstance(worker_result, dict) else None
                if isinstance(timings, dict):
                    samples.append(timings)
    finally:
        worker = getattr(runner, "worker", None)
        background_tasks = list(getattr(worker, "_background_tasks", set()))
        for task in background_tasks:
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        close = getattr(worker, "close", None)
        if close is not None:
            try:
                await asyncio.wait_for(close(), timeout=10)
            except (asyncio.TimeoutError, Exception):
                pass
        try:
            await asyncio.wait_for(mongo.close(), timeout=10)
        except (asyncio.TimeoutError, Exception):
            pass

    elapsed = time.monotonic() - started
    after = _milvus_rows(args.collection)
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_requested_seconds": args.duration_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "download_base_url": os.environ.get("ATTACHMENT_PREFIX", ""),
        "mongo_db": args.mongo_db,
        "collection": args.collection,
        "index": {"type": "IVF_PQ", "nlist": 64, "m": 64, "nbits": 8},
        "tuning": {"batch_size": args.batch_size, "concurrency": args.concurrency},
        "milvus_rows_before": before,
        "milvus_rows_after": after,
        "milvus_rows_inserted": after - before,
        "milvus_rows_per_hour": round((after - before) * 3600 / elapsed, 2) if elapsed else 0,
        "counters": dict(counters),
        "task_statuses_after": _task_statuses(args.mongo_db),
        "worker_pipeline_timing": _summarize(samples),
        "boundary_timing": {name: timer.summary() for name, timer in timers.items()},
        "worker_batches": batch_samples,
    }


def main() -> int:
    args = parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
