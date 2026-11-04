#!/usr/bin/env python3
"""Benchmark the isolated image model/vectorization path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import httpx

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
from app.services.image_offline_providers import OfflineImageModelClient


DEFAULT_CONCURRENCY_LADDER = (1, 2, 4, 8, 16, 24, 32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated image model benchmark.")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--downloaded-dir", type=Path, default=Path())
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--warmup-seconds", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(item) for item in DEFAULT_CONCURRENCY_LADDER),
        help="Comma-separated concurrency ladder.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage-name", default="image_model")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--model-url", default=settings.IMAGE_VL_EMBED_URL)
    parser.add_argument("--model-name", default=settings.IMAGE_VL_EMBED_MODEL)
    parser.add_argument("--api-key", default=settings.IMAGE_MODEL_API_KEY)
    parser.add_argument("--timeout", type=int, default=settings.IMAGE_FEATURE_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=settings.IMAGE_FEATURE_MAX_RETRIES)
    parser.add_argument("--max-pixels", type=int, default=settings.IMAGE_MAX_PIXELS)
    parser.add_argument("--batch-mode", action="store_true")
    parser.add_argument("--batch-size", type=int, default=settings.IMAGE_VL_EMBED_BATCH_SIZE)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args(argv)


def _client(args: argparse.Namespace, *, concurrency: int) -> OfflineImageModelClient:
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 32),
        max_keepalive_connections=max(concurrency, 16),
    )
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout),
        limits=limits,
        headers={"Content-Type": "application/json"},
    )
    return OfflineImageModelClient(
        api_key=args.api_key,
        vl_embed_url=args.model_url,
        vl_embed_model=args.model_name,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_pixels=args.max_pixels,
        http_client=http_client,
    )


async def _close_model_client(client: OfflineImageModelClient) -> None:
    http_client = getattr(client, "_http_client", None)
    if http_client is not None:
        await http_client.aclose()


async def _run_case(
    *,
    samples,
    duration_seconds: int,
    concurrency: int,
    args: argparse.Namespace,
    batch_mode: bool,
) -> BenchmarkCaseResult:
    selector = cycle_samples(samples)
    client = _client(args, concurrency=concurrency)
    result = BenchmarkCaseResult(
        case_name=f"model_c{concurrency}{'_batch' if batch_mode else ''}",
        stage="image_model",
        concurrency=concurrency,
        batch_size=max(int(args.batch_size or 1), 1) if batch_mode else 1,
    )
    errors = Counter()
    deadline = Deadline(duration_seconds)
    counter = 0
    result.extras["sample_count"] = len(samples)
    result.extras["batch_mode"] = batch_mode
    result.extras["model_url"] = args.model_url

    async def worker() -> None:
        nonlocal counter
        while not deadline.expired:
            sample = selector(counter)
            counter += 1
            pipeline_input = build_pipeline_input(sample, sequence=counter)
            started = time.perf_counter()
            result.attempts += 1
            try:
                if batch_mode:
                    await asyncio.wait_for(
                        client.embed_image_data_urls([_data_url(pipeline_input)]),
                        timeout=max(deadline.remaining, 0.5),
                    )
                else:
                    await asyncio.wait_for(
                        client.embed_image(
                            pipeline_input.image_bytes,
                        ),
                        timeout=max(deadline.remaining, 0.5),
                    )
            except Exception as error:  # noqa: BLE001 - benchmark needs error census
                result.failures += 1
                errors[type(error).__name__] += 1
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result.successes += 1
            result.bytes_processed += len(pipeline_input.image_bytes)
            result.latencies_ms.append(elapsed_ms)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    started = time.perf_counter()
    try:
        await asyncio.gather(*tasks)
    finally:
        result.duration_seconds = time.perf_counter() - started
        result.error_counts = dict(sorted(errors.items()))
        await _close_model_client(client)
    return result


def _data_url(pipeline_input) -> str:
    from app.services.image_offline_providers import image_bytes_to_data_url

    return image_bytes_to_data_url(
        pipeline_input.image_bytes,
        max_pixels=settings.IMAGE_MAX_PIXELS,
    )


async def run_image_model_benchmark(args: argparse.Namespace) -> dict[str, object]:
    samples = scan_upload_samples(args.upload_dir)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise ValueError(f"no upload samples found under {args.upload_dir}")

    run_dir = default_run_dir(
        f"{DEFAULT_BENCHMARK_PREFIX}_{args.stage_name}",
        run_id=args.run_id.strip() or None,
    )
    concurrency_ladder = [int(item) for item in str(args.concurrency).split(",") if str(item).strip()]
    status_path = run_dir / "status.json"
    report_path = run_dir / "report.json"
    cases: list[BenchmarkCaseResult] = []
    previous: BenchmarkCaseResult | None = None
    status = {
        "started_at": time.time(),
        "stage": args.stage_name,
        "upload_dir": str(args.upload_dir),
        "sample_count": len(samples),
        "concurrency_ladder": concurrency_ladder,
        "cases": [],
        "batch_mode": bool(args.batch_mode),
    }
    write_json(status_path, status)

    if args.warmup_seconds > 0:
        warmup = await _run_case(
            samples=samples,
            duration_seconds=args.warmup_seconds,
            concurrency=min(concurrency_ladder[0], len(samples)),
            args=args,
            batch_mode=args.batch_mode,
        )
        status["warmup"] = warmup.to_summary()
        write_json(status_path, status)

    case_durations = allocate_case_durations(args.duration_seconds, len(concurrency_ladder))
    for concurrency, case_duration in zip(concurrency_ladder, case_durations):
        case = await _run_case(
            samples=samples,
            duration_seconds=case_duration,
            concurrency=concurrency,
            args=args,
            batch_mode=args.batch_mode,
        )
        cases.append(case)
        status["cases"] = [item.to_summary() for item in cases]
        write_json(status_path, status)
        if should_stop_escalation(previous, case):
            case.notes.append("escalation_stop_boundary")
            break
        previous = case

    report = summarize_stage(
        stage=args.stage_name,
        cases=cases,
        extra={
            "upload_dir": str(args.upload_dir),
            "sample_count": len(samples),
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "batch_mode": bool(args.batch_mode),
            "batch_size": args.batch_size,
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
    result = asyncio.run(run_image_model_benchmark(args))
    if args.print_report:
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    else:
        print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
