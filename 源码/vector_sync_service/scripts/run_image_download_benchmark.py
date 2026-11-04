#!/usr/bin/env python3
"""Benchmark the isolated image download path."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import httpx
import json

from benchmark_shared import (
    BenchmarkCaseResult,
    Deadline,
    DEFAULT_BENCHMARK_PREFIX,
    DEFAULT_UPLOAD_DIR,
    allocate_case_durations,
    cycle_samples,
    default_run_dir,
    force_utf8_stdout,
    json_safe,
    scan_upload_samples,
    should_stop_escalation,
    summarize_stage,
    write_json,
)


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.services.image_downloader import ImageDownloader


DEFAULT_CONCURRENCY_LADDER = (8, 16, 32, 48, 64, 96, 128)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated image download benchmark.")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--warmup-seconds", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(item) for item in DEFAULT_CONCURRENCY_LADDER),
        help="Comma-separated concurrency ladder.",
    )
    parser.add_argument("--timeout", type=int, default=settings.IMAGE_DOWNLOAD_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=settings.IMAGE_DOWNLOAD_MAX_RETRIES)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage-name", default="image_download")
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional cap on scanned samples.")
    parser.add_argument("--shared-client", action="store_true", default=True)
    parser.add_argument("--no-shared-client", dest="shared_client", action="store_false")
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args(argv)


async def _run_case(
    *,
    samples,
    duration_seconds: int,
    concurrency: int,
    timeout: int,
    max_retries: int,
    shared_client: bool,
) -> BenchmarkCaseResult:
    selector = cycle_samples(samples)
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 32),
        max_keepalive_connections=max(concurrency, 16),
    )
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), limits=limits) if shared_client else None
    downloader = ImageDownloader(
        timeout=timeout,
        max_retries=max_retries,
        concurrency=concurrency,
        client=client,
    )
    result = BenchmarkCaseResult(
        case_name=f"download_c{concurrency}",
        stage="image_download",
        concurrency=concurrency,
        batch_size=1,
    )
    errors = Counter()
    deadline = Deadline(duration_seconds)
    counter = 0
    result.extras["sample_count"] = len(samples)
    result.extras["shared_client"] = shared_client

    async def worker() -> None:
        nonlocal counter
        while not deadline.expired:
            sample = selector(counter)
            counter += 1
            started = time.perf_counter()
            result.attempts += 1
            try:
                download = await asyncio.wait_for(
                    downloader.download_bytes(sample.image_relative_url),
                    timeout=max(deadline.remaining, 0.5),
                )
            except Exception as error:  # noqa: BLE001 - benchmark needs error census
                result.failures += 1
                errors[type(error).__name__] += 1
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result.successes += 1
            result.bytes_processed += int(download.size_bytes or 0)
            result.latencies_ms.append(elapsed_ms)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    started = time.perf_counter()
    try:
        await asyncio.gather(*tasks)
    finally:
        result.duration_seconds = time.perf_counter() - started
        result.error_counts = dict(sorted(errors.items()))
        if client is not None:
            await client.aclose()
    return result


async def run_download_benchmark(args: argparse.Namespace) -> dict[str, object]:
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
    }
    write_json(status_path, status)

    if args.warmup_seconds > 0:
        warmup = await _run_case(
            samples=samples,
            duration_seconds=args.warmup_seconds,
            concurrency=min(concurrency_ladder[0], len(samples)),
            timeout=args.timeout,
            max_retries=args.max_retries,
            shared_client=args.shared_client,
        )
        status["warmup"] = warmup.to_summary()
        write_json(status_path, status)

    case_durations = allocate_case_durations(args.duration_seconds, len(concurrency_ladder))
    for concurrency, case_duration in zip(concurrency_ladder, case_durations):
        case = await _run_case(
            samples=samples,
            duration_seconds=case_duration,
            concurrency=concurrency,
            timeout=args.timeout,
            max_retries=args.max_retries,
            shared_client=args.shared_client,
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
            "shared_client": args.shared_client,
            "artifact_paths": {
                "status_path": str(status_path),
                "report_path": str(report_path),
            },
        },
    )
    write_json(report_path, report)
    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "report": report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    result = asyncio.run(run_download_benchmark(args))
    if args.print_report:
        print(json.dumps(json_safe(result["report"]), ensure_ascii=False, indent=2))
    else:
        print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
