#!/usr/bin/env python3
"""Benchmark the isolated text embedding path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from benchmark_shared import (
    BenchmarkCaseResult,
    DEFAULT_BENCHMARK_PREFIX,
    DEFAULT_UPLOAD_DIR,
    allocate_case_durations,
    Deadline,
    build_pipeline_input,
    cycle_samples,
    default_run_dir,
    force_utf8_stdout,
    scan_upload_samples,
    summarize_stage,
    write_json,
)


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.services.embedder import EmbeddingClient


DEFAULT_CONCURRENCY_LADDER = (4, 8, 16, 32)
DEFAULT_BATCH_LADDER = (1, 8, 16, 32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated text embedding benchmark.")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--duration-seconds", type=int, default=180)
    parser.add_argument("--warmup-seconds", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(item) for item in DEFAULT_CONCURRENCY_LADDER),
        help="Comma-separated concurrency ladder.",
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(item) for item in DEFAULT_BATCH_LADDER),
        help="Comma-separated batch-size ladder.",
    )
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage-name", default="text_embedding")
    parser.add_argument("--api-url", default=settings.EMBEDDING_API_URL)
    parser.add_argument("--model-name", default=settings.EMBEDDING_MODEL)
    parser.add_argument("--api-key", default=settings.EMBEDDING_API_KEY)
    parser.add_argument("--timeout", type=int, default=settings.EMBEDDING_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=settings.EMBEDDING_MAX_RETRIES)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args(argv)


def _text_from_sample(sample, *, sequence: int) -> str:
    pipeline_input = build_pipeline_input(sample, sequence=sequence)
    context = pipeline_input.context_text.strip()
    caption = str(pipeline_input.draft_record.img_meta.get("caption") or "").strip()
    return "\n".join(part for part in (caption, context) if part).strip() or sample.image_relative_url


async def _run_case(
    *,
    samples,
    duration_seconds: int,
    concurrency: int,
    batch_size: int,
    args: argparse.Namespace,
) -> BenchmarkCaseResult:
    selector = cycle_samples(samples)
    client = EmbeddingClient(
        base_url=args.api_url,
        model=args.model_name,
        api_key=args.api_key,
        max_concurrency=concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        batch_size=batch_size,
    )
    result = BenchmarkCaseResult(
        case_name=f"text_c{concurrency}_b{batch_size}",
        stage="text_embedding",
        concurrency=concurrency,
        batch_size=batch_size,
    )
    errors = Counter()
    deadline = Deadline(duration_seconds)
    counter = 0

    async def worker() -> None:
        nonlocal counter
        while not deadline.expired:
            texts: list[str] = []
            for _ in range(batch_size):
                sample = selector(counter)
                counter += 1
                texts.append(_text_from_sample(sample, sequence=counter))
            started = time.perf_counter()
            result.attempts += len(texts)
            try:
                responses = await asyncio.wait_for(
                    client.embed_texts_detailed(texts),
                    timeout=max(deadline.remaining, 0.5),
                )
            except Exception as error:  # noqa: BLE001 - benchmark needs error census
                result.failures += len(texts)
                errors[type(error).__name__] += len(texts)
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            success_count = sum(1 for item in responses if getattr(item, "success", False))
            failure_count = len(responses) - success_count
            result.successes += success_count
            result.failures += failure_count
            if success_count:
                result.latencies_ms.extend([elapsed_ms / max(success_count, 1)] * success_count)
            for item in responses:
                if not getattr(item, "success", False):
                    errors[str(getattr(item, "error_code", "UNKNOWN"))] += 1
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    started = time.perf_counter()
    try:
        await asyncio.gather(*tasks)
    finally:
        result.duration_seconds = time.perf_counter() - started
        result.error_counts = dict(sorted(errors.items()))
        result.extras["sample_count"] = len(samples)
        result.extras["api_url"] = args.api_url
        await client.close()
    return result


async def run_text_embedding_benchmark(args: argparse.Namespace) -> dict[str, object]:
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
    batch_ladder = [int(item) for item in str(args.batch_sizes).split(",") if str(item).strip()]
    status_path = run_dir / "status.json"
    report_path = run_dir / "report.json"
    cases: list[BenchmarkCaseResult] = []

    status = {
        "started_at": time.time(),
        "stage": args.stage_name,
        "upload_dir": str(args.upload_dir),
        "sample_count": len(samples),
        "concurrency_ladder": concurrency_ladder,
        "batch_ladder": batch_ladder,
        "cases": [],
    }
    write_json(status_path, status)

    if args.warmup_seconds > 0:
        warmup = await _run_case(
            samples=samples,
            duration_seconds=args.warmup_seconds,
            concurrency=concurrency_ladder[0],
            batch_size=batch_ladder[0],
            args=args,
        )
        status["warmup"] = warmup.to_summary()
        write_json(status_path, status)

    case_durations = allocate_case_durations(
        args.duration_seconds,
        len(concurrency_ladder) * len(batch_ladder),
    )
    duration_index = 0
    for concurrency in concurrency_ladder:
        for batch_size in batch_ladder:
            case = await _run_case(
                samples=samples,
                duration_seconds=case_durations[duration_index],
                concurrency=concurrency,
                batch_size=batch_size,
                args=args,
            )
            duration_index += 1
            cases.append(case)
            status["cases"] = [item.to_summary() for item in cases]
            write_json(status_path, status)

    report = summarize_stage(
        stage=args.stage_name,
        cases=cases,
        extra={
            "upload_dir": str(args.upload_dir),
            "sample_count": len(samples),
            "timeout": args.timeout,
            "max_retries": args.max_retries,
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
    result = asyncio.run(run_text_embedding_benchmark(args))
    if args.print_report:
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    else:
        print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
