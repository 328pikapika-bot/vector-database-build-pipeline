#!/usr/bin/env python3
"""Sustained batch-load ladder for the shared qwen3-embed endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx


DEFAULT_URL = "http://127.0.0.1:16703/v1/embeddings"
DEFAULT_MODEL = "qwen3-embed"
DEFAULT_KEY = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="4,8,16,32,64")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("EMBEDDING_API_KEY", DEFAULT_KEY))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_texts() -> list[str]:
    path = Path(__file__).with_name("test_text_samples.txt")
    if path.exists():
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if values:
            return values
    return [
        "Scientific document embedding throughput sample for sustained load testing.",
        "This request represents a title, abstract, keyword, or full-text chunk.",
    ]


async def run_case(
    client: httpx.AsyncClient,
    texts: list[str],
    *,
    url: str,
    concurrency: int,
    batch_size: int,
    seconds: int,
    model: str,
    api_key: str,
) -> dict[str, object]:
    deadline = time.monotonic() + seconds
    cursor = 0
    requests = 0
    successes = 0
    failures = 0
    errors: dict[str, int] = {}
    latencies: list[float] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal cursor, requests, successes, failures
        while time.monotonic() < deadline:
            async with lock:
                start_index = cursor
                cursor += batch_size
            payload_texts = [texts[(start_index + index) % len(texts)] for index in range(batch_size)]
            started = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    json={"model": model, "input": payload_texts, "encoding_format": "float"},
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                async with lock:
                    requests += 1
                    if response.status_code == 200:
                        body = response.json()
                        data = body.get("data") or []
                        if len(data) == batch_size and all(len(item.get("embedding") or []) == 2560 for item in data):
                            successes += 1
                            latencies.append(elapsed)
                        else:
                            failures += 1
                            errors["count_or_dimension_mismatch"] = errors.get("count_or_dimension_mismatch", 0) + 1
                    else:
                        failures += 1
                        key = f"HTTP_{response.status_code}"
                        errors[key] = errors.get(key, 0) + 1
            except Exception as error:  # noqa: BLE001 - load test must census transport failures
                async with lock:
                    requests += 1
                    failures += 1
                    key = type(error).__name__
                    errors[key] = errors.get(key, 0) + 1

    started = time.perf_counter()
    await asyncio.gather(*(asyncio.create_task(worker()) for _ in range(concurrency)))
    elapsed = max(time.perf_counter() - started, 0.001)
    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        if not ordered:
            return 0.0
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    return {
        "concurrency": concurrency,
        "batch_size": batch_size,
        "duration_seconds": round(elapsed, 3),
        "requests": requests,
        "successful_requests": successes,
        "failed_requests": failures,
        "success_rate": round(successes / max(requests, 1), 4),
        "request_throughput_per_second": round(successes / elapsed, 3),
        "text_throughput_per_second": round(successes * batch_size / elapsed, 3),
        "latency_ms": {
            "avg": round(statistics.fmean(ordered), 3) if ordered else 0.0,
            "p50": round(percentile(0.50), 3),
            "p95": round(percentile(0.95), 3),
            "p99": round(percentile(0.99), 3),
        },
        "errors": dict(sorted(errors.items())),
    }


async def main() -> None:
    args = parse_args()
    levels = [max(int(value), 1) for value in args.concurrency.split(",") if value.strip()]
    texts = load_texts()
    limits = httpx.Limits(max_connections=max(levels) * 2, max_keepalive_connections=max(levels))
    timeout = httpx.Timeout(60.0)
    headers = {"Authorization": f"Bearer {args.api_key}"}
    results = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers) as client:
        warmup = await client.post(
            args.url,
            json={"model": args.model, "input": [texts[0]], "encoding_format": "float"},
        )
        if warmup.status_code != 200:
            raise RuntimeError(f"warmup failed: HTTP_{warmup.status_code} {warmup.text[:300]}")
        for level in levels:
            result = await run_case(
                client,
                texts,
                url=args.url,
                concurrency=level,
                batch_size=args.batch_size,
                seconds=args.seconds,
                model=args.model,
                api_key=args.api_key,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {
        "endpoint": args.url,
        "model": args.model,
        "concurrency_levels": levels,
        "batch_size": args.batch_size,
        "seconds_per_level": args.seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
