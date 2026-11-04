#!/usr/bin/env python3
"""Backfill metadata/fulltext independently from image-priority intake windows."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill metadata/fulltext saved by SYNC_PIPELINE_MODE=image_priority.",
    )
    parser.add_argument("--window-id", default="", help="Only backfill one image-priority window.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum window records to process; 0 means no limit.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mongo-uri", default="")
    parser.add_argument("--mongo-db", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-fulltext-drain", action="store_true")
    parser.add_argument("--fulltext-drain-timeout-seconds", type=int, default=3600)
    parser.add_argument("--report-path", default="")
    parser.add_argument(
        "--lane",
        default="combined",
        choices=("combined", "metadata", "fulltext"),
        help="Backfill lane to execute.",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    _prepare_env(args)

    from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager

    manager = get_mongo_state_manager()
    started = time.perf_counter()
    report = _base_report(args)
    try:
        await manager.connect()
        if args.lane in {"combined", "metadata"}:
            report["metadata"] = await run_metadata_backfill(
                manager=manager,
                window_id=args.window_id,
                limit=args.limit,
                batch_size=args.batch_size,
                dry_run=bool(args.dry_run),
            )
        if args.lane in {"combined", "fulltext"}:
            report["fulltext"] = await run_fulltext_backfill(
                manager=manager,
                window_id=args.window_id,
                limit=args.limit,
                batch_size=args.batch_size,
                dry_run=bool(args.dry_run),
                skip_drain=bool(args.skip_fulltext_drain),
                drain_timeout_seconds=int(args.fulltext_drain_timeout_seconds or 0),
            )
    finally:
        report["finished_at"] = iso_now()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if args.report_path:
            path = Path(args.report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        await close_mongo_state_manager()
    return _exit_code(report)


async def run_metadata_backfill(
    *,
    manager: Any,
    window_id: str,
    limit: int,
    batch_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    from app.models.sync_runtime import ConsumedRecord, sha256_hex
    from app.services.batch_processor_existing import process_batch_existing_arch
    from app.services.collection_targets import (
        collection_target_key,
        resolve_document_collection_target,
    )
    from app.services.milvus_service import get_milvus_service

    milvus = get_milvus_service()
    target = resolve_document_collection_target(
        milvus.client,
        logical_name=None,
        ensure_collection=lambda name: milvus._ensure_document_collection_created(  # noqa: SLF001
            name,
            vector_dim=milvus_vector_dim(),
        ),
    )
    target_key = collection_target_key(target)
    report = _lane_report("metadata", target=target.to_dict(), dry_run=dry_run)
    remaining = int(limit or 0)
    while True:
        pull_limit = max(int(batch_size or 1), 1)
        if remaining > 0:
            pull_limit = min(pull_limit, remaining)
        rows = await manager.find_image_success_windows_for_metadata_backfill(
            window_id=window_id or None,
            statuses=["PENDING", "FAILED"],
            limit=pull_limit,
            target=target_key,
        )
        if not rows:
            break
        records = [_record_from_window(row, ConsumedRecord, sha256_hex) for row in rows]
        row_ids = [str(row.get("_id")) for row in rows]
        if dry_run:
            report["processed"] += len(rows)
            report["success"] += len(rows)
        else:
            try:
                result = await process_batch_existing_arch(
                    records,
                    mode="metadata_bulk",
                    document_collection_name=target.logical_name,
                )
                report["processed"] += len(rows)
                report["success"] += result.success
                report["skip"] += result.skip
                report["fail"] += result.fail
                if result.commit_safe:
                    await manager.mark_metadata_backfill_result(
                        ids=row_ids,
                        target=target_key,
                        status="DONE",
                    )
                    report["marked_done"] += len(row_ids)
                else:
                    await manager.mark_metadata_backfill_result(
                        ids=row_ids,
                        target=target_key,
                        status="FAILED",
                        error="metadata_pipeline_returned_commit_unsafe",
                    )
                    report["marked_failed"] += len(row_ids)
            except Exception as error:  # noqa: BLE001
                message = f"{type(error).__name__}: {error}"
                await manager.mark_metadata_backfill_result(
                    ids=row_ids,
                    target=target_key,
                    status="FAILED",
                    error=message,
                )
                report["processed"] += len(rows)
                report["fail"] += len(rows)
                report["marked_failed"] += len(row_ids)
                report["errors"].append({"ids": row_ids[:20], "error": message})
        if remaining > 0:
            remaining -= len(rows)
            if remaining <= 0:
                break
    return report


async def run_fulltext_backfill(
    *,
    manager: Any,
    window_id: str,
    limit: int,
    batch_size: int,
    dry_run: bool,
    skip_drain: bool,
    drain_timeout_seconds: int,
) -> dict[str, Any]:
    from app.services.collection_targets import (
        collection_target_key,
        resolve_fulltext_collection_target,
    )
    from app.services.fulltext_task_manager import extract_fulltext_tasks
    from app.services.fulltext_task_repository import get_fulltext_task_repository
    from app.services.milvus_chunk_service import MilvusChunkService

    chunk_writer = MilvusChunkService()
    target = resolve_fulltext_collection_target(
        chunk_writer.client,
        logical_name=None,
        ensure_collection=lambda name: chunk_writer._ensure_chunk_collection_created(  # noqa: SLF001
            name,
            vector_dim=milvus_vector_dim(),
        ),
    )
    target_key = collection_target_key(target)
    repository = get_fulltext_task_repository()
    report = _lane_report("fulltext", target=target.to_dict(), dry_run=dry_run)
    remaining = int(limit or 0)
    while True:
        pull_limit = max(int(batch_size or 1), 1)
        if remaining > 0:
            pull_limit = min(pull_limit, remaining)
        rows = await manager.find_image_success_windows_for_fulltext_backfill(
            window_id=window_id or None,
            statuses=["PENDING", "FAILED"],
            limit=pull_limit,
            target=target_key,
        )
        if not rows:
            break
        row_ids = [str(row.get("_id")) for row in rows]
        tasks = []
        for row in rows:
            extracted = extract_fulltext_tasks(row.get("payload") or {})
            tasks.extend(extracted)
        if dry_run:
            report["processed"] += len(rows)
            report["success"] += len(rows)
            report["task_count"] += len(tasks)
        else:
            try:
                if tasks:
                    upsert_result = await repository.upsert_tasks(tasks)
                    report["task_count"] += len(tasks)
                    report["task_upsert_result"] = upsert_result
                await manager.mark_fulltext_backfill_result(
                    ids=row_ids,
                    target=target_key,
                    status="DONE",
                )
                report["processed"] += len(rows)
                report["success"] += len(rows)
                report["marked_done"] += len(row_ids)
            except Exception as error:  # noqa: BLE001
                message = f"{type(error).__name__}: {error}"
                await manager.mark_fulltext_backfill_result(
                    ids=row_ids,
                    target=target_key,
                    status="FAILED",
                    error=message,
                )
                report["processed"] += len(rows)
                report["fail"] += len(rows)
                report["marked_failed"] += len(row_ids)
                report["errors"].append({"ids": row_ids[:20], "error": message})
        if remaining > 0:
            remaining -= len(rows)
            if remaining <= 0:
                break
    if not dry_run and not skip_drain:
        report["fulltext_drain"] = await drain_fulltext_tasks(
            timeout_seconds=drain_timeout_seconds,
            collection_name=target.logical_name,
        )
    return report


async def drain_fulltext_tasks(*, timeout_seconds: int, collection_name: str) -> dict[str, Any]:
    from app.services.fulltext_worker_runner import FullTextWorkerRunner
    from app.services.fulltext_worker import FullTextWorker

    worker = FullTextWorker(chunk_collection_name=collection_name)
    runner = FullTextWorkerRunner(worker=worker)
    worker_id = f"image_priority_backfill:{int(time.time())}"
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    summary: dict[str, Any] = {
        "enabled": True,
        "timeout_seconds": timeout_seconds,
        "iterations": 0,
        "processed": 0,
        "last_error": None,
        "timed_out": False,
        "collection_name": collection_name,
    }
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            summary["timed_out"] = True
            break
        result = await runner.run_once(worker_id=worker_id)
        summary["iterations"] += 1
        if not result.get("claimed"):
            break
        summary["processed"] += int(runner.status().get("last_processed_count") or 0)
        summary["last_error"] = runner.status().get("last_error")
    return summary


def _prepare_env(args: argparse.Namespace) -> None:
    os.environ["SYNC_PIPELINE_MODE"] = "standard"
    os.environ["IMAGE_TASK_UPSERT_ENABLED"] = "false"
    os.environ["FULLTEXT_TASK_UPSERT_ENABLED"] = "true"
    os.environ.setdefault("FULLTEXT_WORKER_ENABLED", "true")
    if args.mongo_uri:
        os.environ["MONGO_URI"] = args.mongo_uri
    if args.mongo_db:
        os.environ["MONGO_DB"] = args.mongo_db


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "started_at": iso_now(),
        "lane": args.lane,
        "window_id": args.window_id,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "dry_run": bool(args.dry_run),
        "metadata": None,
        "fulltext": None,
    }


def _lane_report(lane: str, *, target: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    return {
        "lane": lane,
        "dry_run": dry_run,
        "target": target,
        "processed": 0,
        "success": 0,
        "skip": 0,
        "fail": 0,
        "marked_done": 0,
        "marked_failed": 0,
        "task_count": 0,
        "errors": [],
    }


def _record_from_window(row: dict[str, Any], record_cls: Callable[..., Any], sha256_hex_func: Callable[[bytes], str]) -> Any:
    payload = row.get("payload") or {}
    raw_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw_bytes = raw_text.encode("utf-8")
    topic = str(row.get("topic") or "image_priority_backfill")
    partition = int(row.get("partition") or 0)
    offset = int(row.get("offset") or 0)
    fingerprint = str(row.get("event_fingerprint") or "")
    if not fingerprint:
        fingerprint = "|".join([topic, str(partition), str(offset), sha256_hex_func(raw_bytes)])
    return record_cls(
        topic=topic,
        partition=partition,
        offset=offset,
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        event_fingerprint=fingerprint,
        parsed_payload=payload,
    )


def _exit_code(report: dict[str, Any]) -> int:
    failures = 0
    for lane in ("metadata", "fulltext"):
        lane_report = report.get(lane) or {}
        failures += int(lane_report.get("marked_failed") or 0)
    return 0 if failures == 0 else 2


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def milvus_vector_dim() -> int:
    from app.config import settings

    return int(settings.VECTOR_DIM or 2560)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
