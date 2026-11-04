#!/usr/bin/env python3
"""Benchmark the concurrent full image ingestion path into Milvus."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from pymilvus import MilvusClient

from benchmark_shared import (
    BenchmarkCaseResult,
    Deadline,
    DEFAULT_BENCHMARK_PREFIX,
    DEFAULT_UPLOAD_DIR,
    allocate_case_durations,
    build_pipeline_input,
    cycle_samples,
    default_run_dir,
    force_utf8_stdout,
    scan_upload_samples,
    should_stop_escalation,
    summarize_stage,
    write_json,
)


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.services.image_milvus_batch_writer import MilvusImageBatchWriter
from app.services.image_feature_pipeline import process_image_features
from app.services.image_offline_providers import (
    OfflineImageModelClient,
    build_offline_image_feature_providers,
)
from app.services.milvus_image_service import MilvusImageService


DEFAULT_CONCURRENCY_LADDER = (4, 8, 12, 16)
FEATURE_STAGE_KEYS = (
    "p_hash",
    "data_url",
    "visual_embed",
    "caption_embed",
    "context_embed",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a concurrent full image-write benchmark.")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--warmup-seconds", type=int, default=0)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(item) for item in DEFAULT_CONCURRENCY_LADDER),
        help="Comma-separated concurrency ladder.",
    )
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage-name", default="image_full_write")
    parser.add_argument("--collection-name", default="")
    parser.add_argument(
        "--mode",
        choices=["replace", "insert-only", "clean-collection"],
        default="clean-collection",
    )
    parser.add_argument("--reuse-existing-collection", action="store_true")
    parser.add_argument(
        "--prepare-collection-only",
        action="store_true",
        help="Prepare the target collection and exit without running the timed benchmark.",
    )
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args(argv)


def _milvus_client() -> MilvusClient:
    return MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        user=settings.MILVUS_USER or "",
        password=settings.MILVUS_PASSWORD or "",
        timeout=20,
    )


def _ensure_collection(collection_name: str) -> None:
    service = MilvusImageService()
    try:
        if service.client.has_collection(collection_name):
            service._validate_existing_image_schema(collection_name)
            return
        service.init_image_collection(collection_name)
    finally:
        service.close()


def _require_existing_collection(collection_name: str) -> None:
    service = MilvusImageService()
    try:
        if not service.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")
        service._validate_existing_image_schema(collection_name)
    finally:
        service.close()


def _ensure_clean_collection(collection_name: str) -> None:
    service = MilvusImageService()
    try:
        if service.client.has_collection(collection_name):
            service.client.drop_collection(collection_name)
        service.init_image_collection(collection_name)
    finally:
        service.close()


def _prepare_collection(
    *,
    collection_name: str,
    mode: str,
    reuse_existing_collection: bool,
) -> str:
    if mode == "clean-collection":
        if reuse_existing_collection:
            _require_existing_collection(collection_name)
            return "reused_existing_collection"
        _ensure_clean_collection(collection_name)
        return "recreated_clean_collection"
    if reuse_existing_collection:
        _require_existing_collection(collection_name)
        return "reused_existing_collection"
    _ensure_collection(collection_name)
    return "ensured_collection"


def _escape_milvus_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _visibility_probe(
    milvus: MilvusClient,
    *,
    collection_name: str,
    image_ids: Sequence[str],
) -> dict[str, object]:
    probe_ids = [str(item).strip() for item in image_ids if str(item or "").strip()]
    if not probe_ids:
        return {"probed": 0, "visible": 0, "sample_ids": []}
    visible_ids: list[str] = []
    batch_size = 64
    for index in range(0, len(probe_ids), batch_size):
        batch = probe_ids[index : index + batch_size]
        escaped = ",".join(f'"{_escape_milvus_string(item)}"' for item in batch)
        rows = milvus.query(
            collection_name=collection_name,
            filter=f"id in [{escaped}]",
            output_fields=["id"],
        )
        visible_ids.extend(str(row.get("id") or "") for row in rows if row.get("id"))
    visible_ids = sorted({item for item in visible_ids if item})
    return {
        "probed": len(probe_ids),
        "visible": len(visible_ids),
        "sample_ids": probe_ids[:8],
        "visible_ids": visible_ids[:8],
    }


async def _run_case(
    *,
    samples,
    duration_seconds: int,
    concurrency: int,
    collection_name: str,
    sequence_offset: int,
    mode: str,
) -> BenchmarkCaseResult:
    selector = cycle_samples(samples)
    writer = MilvusImageBatchWriter(
        batch_size=settings.IMAGE_MILVUS_WRITE_BATCH_SIZE,
        flush_interval_ms=settings.IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS,
        concurrency=settings.IMAGE_MILVUS_WRITE_CONCURRENCY,
    )
    model_client = OfflineImageModelClient.from_settings()
    providers = build_offline_image_feature_providers(client=model_client)
    success_ids: list[str] = []
    result = BenchmarkCaseResult(
        case_name=f"full_c{concurrency}",
        stage="image_full_write",
        concurrency=concurrency,
        batch_size=settings.IMAGE_MILVUS_WRITE_BATCH_SIZE,
    )
    errors = Counter()
    deadline = Deadline(duration_seconds)
    counter = sequence_offset
    result.extras["sample_count"] = len(samples)
    result.extras["collection_name"] = collection_name
    visual_latencies_ms: list[float] = []
    write_latencies_ms: list[float] = []
    feature_stage_latencies_ms: dict[str, list[float]] = {
        stage_name: [] for stage_name in FEATURE_STAGE_KEYS
    }

    async def worker() -> None:
        nonlocal counter
        while not deadline.expired:
            sample = selector(counter)
            counter += 1
            pipeline_input = build_pipeline_input(sample, sequence=counter)
            started = time.perf_counter()
            result.attempts += 1
            try:
                feature_started = time.perf_counter()
                feature_result = await process_image_features(
                    pipeline_input.image_bytes,
                    pipeline_input.draft_record,
                    providers=providers,
                    context_text=pipeline_input.context_text,
                    schema_mode=settings.IMAGE_VECTOR_SCHEMA_MODE,
                )
                visual_latencies_ms.append((time.perf_counter() - feature_started) * 1000.0)
                if not feature_result.success or feature_result.image_record is None:
                    raise RuntimeError(
                        "; ".join(
                            str(item.error_message or item.error_code)
                            for item in (feature_result.failure_items or [])
                        )
                        or "image feature pipeline failed"
                    )
                image_record = feature_result.image_record
                timings = image_record.img_meta.get("feature_stage_timings_ms") or {}
                if isinstance(timings, dict):
                    for stage_name in FEATURE_STAGE_KEYS:
                        value = timings.get(stage_name)
                        if isinstance(value, (int, float)):
                            feature_stage_latencies_ms[stage_name].append(float(value))
                write_started = time.perf_counter()
                if mode == "replace":
                    await writer.replace_image_by_img_id(collection_name, image_record)
                else:
                    await writer.insert_images(collection_name, [image_record])
                write_latencies_ms.append((time.perf_counter() - write_started) * 1000.0)
            except Exception as error:  # noqa: BLE001
                result.failures += 1
                errors[type(error).__name__] += 1
                continue
            result.successes += 1
            success_ids.append(str(image_record.img_id))
            result.bytes_processed += len(pipeline_input.image_bytes)
            result.latencies_ms.append((time.perf_counter() - started) * 1000.0)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    started = time.perf_counter()
    try:
        await asyncio.gather(*tasks)
    finally:
        result.duration_seconds = time.perf_counter() - started
        result.error_counts = dict(sorted(errors.items()))
        await writer.close()
        http_client = getattr(model_client, "_http_client", None)
        if http_client is not None:
            await http_client.aclose()
    result.extras["stage_latency_ms"] = {
        "feature_pipeline_total": BenchmarkCaseResult(
            case_name="feature_pipeline_total",
            stage="feature_pipeline_total",
            concurrency=concurrency,
            latencies_ms=visual_latencies_ms,
        ).to_summary()["latency_ms"],
        "milvus_write": BenchmarkCaseResult(
            case_name="milvus_write",
            stage="milvus_write",
            concurrency=concurrency,
            latencies_ms=write_latencies_ms,
        ).to_summary()["latency_ms"],
    }
    for stage_name, latencies in feature_stage_latencies_ms.items():
        result.extras["stage_latency_ms"][stage_name] = BenchmarkCaseResult(
            case_name=stage_name,
            stage=stage_name,
            concurrency=concurrency,
            latencies_ms=latencies,
        ).to_summary()["latency_ms"]
    result.extras["successful_image_ids"] = list(success_ids)
    return result


async def run_image_full_write_benchmark(args: argparse.Namespace) -> dict[str, object]:
    samples = scan_upload_samples(args.upload_dir)
    samples = [sample for sample in samples if sample.image_path.exists() and sample.image_path.is_file()]
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise ValueError(f"no valid upload samples found under {args.upload_dir}")

    run_id = args.run_id.strip() or None
    run_dir = default_run_dir(f"{DEFAULT_BENCHMARK_PREFIX}_{args.stage_name}", run_id=run_id)
    status_path = run_dir / "status.json"
    report_path = run_dir / "report.json"
    sequence_seed = int(args.sequence_offset or 0)
    if sequence_seed <= 0:
        sequence_seed = int(time.time() * 1000)
    collection_name = (
        args.collection_name.strip()
        or f"{settings.IMAGE_COLLECTION_NAME}_{(run_id or 'bench').lower()}"
    )
    collection_prepare_mode = _prepare_collection(
        collection_name=collection_name,
        mode=args.mode,
        reuse_existing_collection=args.reuse_existing_collection,
    )
    if args.prepare_collection_only:
        report = summarize_stage(
            stage=args.stage_name,
            cases=[],
            extra={
                "upload_dir": str(args.upload_dir),
                "sample_count": len(samples),
                "collection_name": collection_name,
                "mode": args.mode,
                "collection_prepare_mode": collection_prepare_mode,
                "prepared_collection_only": True,
                "artifact_paths": {
                    "status_path": str(status_path),
                    "report_path": str(report_path),
                },
            },
        )
        write_json(
            status_path,
            {
                "prepared_collection_only": True,
                "collection_name": collection_name,
                "mode": args.mode,
                "collection_prepare_mode": collection_prepare_mode,
            },
        )
        write_json(report_path, report)
        return {"run_dir": str(run_dir), "report_path": str(report_path), "report": report}

    milvus = _milvus_client()
    before_count = int(milvus.get_collection_stats(collection_name=collection_name).get("row_count") or 0)

    concurrency_ladder = [int(item) for item in str(args.concurrency).split(",") if str(item).strip()]
    status = {
        "started_at": time.time(),
        "stage": args.stage_name,
        "upload_dir": str(args.upload_dir),
        "sample_count": len(samples),
        "collection_name": collection_name,
        "mode": args.mode,
        "collection_prepare_mode": collection_prepare_mode,
        "sequence_seed": sequence_seed,
        "concurrency_ladder": concurrency_ladder,
        "cases": [],
    }
    write_json(status_path, status)

    cases: list[BenchmarkCaseResult] = []
    previous: BenchmarkCaseResult | None = None
    if args.warmup_seconds > 0:
        warmup = await _run_case(
            samples=samples,
            duration_seconds=args.warmup_seconds,
            concurrency=min(concurrency_ladder[0], len(samples)),
            collection_name=collection_name,
            sequence_offset=sequence_seed,
            mode=args.mode,
        )
        status["warmup"] = warmup.to_summary()
        write_json(status_path, status)

    case_durations = allocate_case_durations(args.duration_seconds, len(concurrency_ladder))
    for index, (concurrency, case_duration) in enumerate(zip(concurrency_ladder, case_durations), start=1):
        case = await _run_case(
            samples=samples,
            duration_seconds=case_duration,
            concurrency=concurrency,
            collection_name=collection_name,
            sequence_offset=sequence_seed + (index * 100_000),
            mode=args.mode,
        )
        cases.append(case)
        status["cases"] = [item.to_summary() for item in cases]
        write_json(status_path, status)
        if should_stop_escalation(previous, case):
            case.notes.append("escalation_stop_boundary")
            break
        previous = case

    await asyncio.sleep(2)
    after_count = int(milvus.get_collection_stats(collection_name=collection_name).get("row_count") or 0)
    visible_probe_ids: list[str] = []
    for case in cases:
        if isinstance(case.extras, dict):
            visible_probe_ids.extend(case.extras.get("successful_image_ids") or [])
    visibility = _visibility_probe(
        milvus,
        collection_name=collection_name,
        image_ids=visible_probe_ids,
    )
    milvus.close()

    report = summarize_stage(
        stage=args.stage_name,
        cases=cases,
        extra={
            "upload_dir": str(args.upload_dir),
            "sample_count": len(samples),
            "collection_name": collection_name,
            "mode": args.mode,
            "collection_prepare_mode": collection_prepare_mode,
            "before_count": before_count,
            "after_count": after_count,
            "delta_count": after_count - before_count,
            "visibility_probe": visibility,
            "artifact_paths": {
                "status_path": str(status_path),
                "report_path": str(report_path),
            },
        },
    )
    write_json(report_path, report)
    return {"run_dir": str(run_dir), "report_path": str(report_path), "report": report}


def main(argv: Sequence[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    result = asyncio.run(run_image_full_write_benchmark(args))
    if args.print_report:
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    else:
        print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
