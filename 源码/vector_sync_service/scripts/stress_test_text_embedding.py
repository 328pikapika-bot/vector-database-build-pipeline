"""
Text embedding (qwen3-embed) concurrency stress test.
Tests the embedding API at increasing concurrency levels with real text data.
Supports batch embedding testing too.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

# --- Config ---
EMBED_URL = "http://127.0.0.1:16703/v1/embeddings"
EMBED_MODEL = "qwen3-embed"
API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

# Test concurrency: single-text and batch-text
SINGLE_CONCURRENCY = [1, 4, 8, 16, 24, 32, 48, 64, 96, 128]
BATCH_CONCURRENCY = [1, 2, 4, 8, 16, 24, 32]
REQUESTS_PER_LEVEL = 80
BATCH_SIZE_FOR_TEST = 32  # batch size for batch embedding test
TIMEOUT_SEC = 60

TEXT_SAMPLES_FILE = Path(__file__).resolve().parent / "test_text_samples.txt"


@dataclass
class RequestResult:
    success: bool
    latency_sec: float
    vector_dim: int = 0
    error: str = ""
    status_code: int = 0


@dataclass
class LevelReport:
    concurrency: int
    mode: str  # "single" or "batch"
    total: int
    success: int
    failed: int
    total_duration_sec: float
    latencies: list[float]
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    avg_vector_dim: int
    error_counts: dict[str, int]


def load_texts(limit: int) -> list[str]:
    if TEXT_SAMPLES_FILE.exists():
        with open(TEXT_SAMPLES_FILE, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts[:limit]
    # Fallback scientific texts
    return [
        "Astaxanthin (ASTA) is a ketocarotenoid found in many marine organisms and that affords many benefits to human health. ASTA is particularly effective against radical-mediated lipid peroxidation.",
        "The adoption of mobile commerce and service in developing countries has been studied extensively. Factors influencing small enterprises include perceived usefulness and ease of use.",
        "Introduction: mutations in mtDNA cause mitochondrial cytopathies, the most common being MELAS syndrome. The A3243G transition in tRNA leucine is present in 80% of patients.",
        "This paper presents a systematic review and meta-analysis of preventive interventions for eating disorders. The findings suggest early intervention programs are effective.",
        "Quantum computing offers significant advantages for solving certain classes of optimization problems. The development of error-corrected qubits remains a key challenge.",
        "The CRISPR-Cas9 system has revolutionized gene editing technology. Its applications span from basic research to therapeutic development for genetic diseases.",
        "Climate change impacts on biodiversity have been documented across multiple ecosystems. Rising temperatures correlate with species migration patterns and extinction rates.",
        "Deep learning models have achieved state-of-the-art performance in natural language processing tasks. Transformer architectures particularly excel at sequence modeling.",
        "The pharmacokinetics of novel drug delivery systems depend on particle size, surface charge, and biocompatibility. Nanoparticle carriers show promise for targeted therapy.",
        "Synthetic biology combines principles from engineering and biology to design novel biological systems. Applications include biosensors, biofuels, and therapeutic production.",
    ] * (limit // 10 + 1)


def build_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }


async def embed_single(client: httpx.AsyncClient, text: str, sem: asyncio.Semaphore) -> RequestResult:
    result = RequestResult(success=False, latency_sec=0)
    payload = {"model": EMBED_MODEL, "input": text, "encoding_format": "float"}
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(EMBED_URL, json=payload, headers=build_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            emb_data = data.get("data", [])
            if emb_data and isinstance(emb_data[0], dict):
                vec = emb_data[0].get("embedding", [])
                result.vector_dim = len(vec)
                result.success = bool(vec)
            else:
                result.error = "empty_embedding_data"
        else:
            result.error = f"HTTP_{resp.status_code}"
    except httpx.TimeoutException:
        result.latency_sec = time.perf_counter() - start
        result.error = "timeout"
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


async def embed_batch(
    client: httpx.AsyncClient, texts: list[str], sem: asyncio.Semaphore
) -> RequestResult:
    """Embed a batch of texts in one API call."""
    result = RequestResult(success=False, latency_sec=0)
    payload = {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"}
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(EMBED_URL, json=payload, headers=build_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            emb_data = data.get("data", [])
            if emb_data and len(emb_data) == len(texts):
                result.vector_dim = len(emb_data[0].get("embedding", []))
                result.success = True
            else:
                result.error = f"count_mismatch: expected={len(texts)} got={len(emb_data)}"
        else:
            result.error = f"HTTP_{resp.status_code}"
    except httpx.TimeoutException:
        result.latency_sec = time.perf_counter() - start
        result.error = "timeout"
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


def compute_report(
    results: list[RequestResult], concurrency: int, mode: str, duration: float
) -> LevelReport:
    success = sum(1 for r in results if r.success)
    failed = len(results) - success
    latencies = [r.latency_sec for r in results if r.success]
    dims = [r.vector_dim for r in results if r.success]
    error_counts: dict[str, int] = {}
    for r in results:
        if not r.success:
            key = r.error.split(":")[0] if r.error else f"HTTP_{r.status_code}"
            error_counts[key] = error_counts.get(key, 0) + 1

    sorted_lat = sorted(latencies)
    return LevelReport(
        concurrency=concurrency,
        mode=mode,
        total=len(results),
        success=success,
        failed=failed,
        total_duration_sec=duration,
        latencies=sorted_lat,
        throughput_rps=len(results) / duration if duration > 0 else 0,
        min_ms=sorted_lat[0] * 1000 if sorted_lat else 0,
        max_ms=sorted_lat[-1] * 1000 if sorted_lat else 0,
        avg_ms=statistics.mean(latencies) * 1000 if latencies else 0,
        p50_ms=_percentile(sorted_lat, 0.50) * 1000,
        p95_ms=_percentile(sorted_lat, 0.95) * 1000,
        p99_ms=_percentile(sorted_lat, 0.99) * 1000,
        avg_vector_dim=int(statistics.mean(dims)) if dims else 0,
        error_counts=error_counts,
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(p * len(sorted_values))))
    return sorted_values[idx]


def print_level(report: LevelReport) -> None:
    success_rate = (report.success / report.total * 100) if report.total > 0 else 0
    print(
        f"  [{report.mode}] conc={report.concurrency:>4} | "
        f"succ={report.success:>3}/{report.total:>3} ({success_rate:>5.1f}%) | "
        f"avg={report.avg_ms:>7.0f}ms p50={report.p50_ms:>7.0f}ms p95={report.p95_ms:>7.0f}ms "
        f"p99={report.p99_ms:>7.0f}ms | rps={report.throughput_rps:>7.1f} | dim={report.avg_vector_dim}"
    )
    if report.error_counts:
        top_errors = sorted(report.error_counts.items(), key=lambda x: -x[1])[:3]
        print(f"         errors: {' | '.join(f'{k}={v}' for k, v in top_errors)}")


def print_summary(reports: dict[str, list[LevelReport]]) -> None:
    print(f"\n{'='*110}")
    header = (
        f"{'Mode':<10} {'Conc':>5} {'Succ':>5} {'Rate':>7} {'RPS':>8} "
        f"{'avg_ms':>7} {'p50_ms':>7} {'p95_ms':>8} {'p99_ms':>8} {'max_ms':>7} {'dim':>5}"
    )
    print(header)
    print("-" * 110)
    for mode in ["single", "batch"]:
        if mode in reports:
            for r in reports[mode]:
                rate = (r.success / r.total * 100) if r.total > 0 else 0
                print(
                    f"{r.mode:<10} {r.concurrency:>5} {r.success:>5} {rate:>6.1f}% "
                    f"{r.throughput_rps:>7.1f} {r.avg_ms:>7.0f} {r.p50_ms:>7.0f} "
                    f"{r.p95_ms:>8.0f} {r.p99_ms:>8.0f} {r.max_ms:>7.0f} {r.avg_vector_dim:>5}"
                )
    print(f"{'='*110}")


async def run_single_text_tests(
    client: httpx.AsyncClient, texts: list[str]
) -> list[LevelReport]:
    print("\n=== SINGLE TEXT EMBEDDING STRESS TEST ===")
    print(f"  Model: {EMBED_MODEL}")
    print(f"  Concurrency levels: {SINGLE_CONCURRENCY}")
    print(f"  Requests per level: {REQUESTS_PER_LEVEL}")

    reports: list[LevelReport] = []
    for concurrency in SINGLE_CONCURRENCY:
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            embed_single(client, random.choice(texts), sem)
            for _ in range(REQUESTS_PER_LEVEL)
        ]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        report = compute_report(list(results), concurrency, "single", duration)
        print_level(report)
        reports.append(report)
        if concurrency != SINGLE_CONCURRENCY[-1]:
            await asyncio.sleep(1.0)
    return reports


async def run_batch_text_tests(
    client: httpx.AsyncClient, texts: list[str]
) -> list[LevelReport]:
    print(f"\n\n=== BATCH TEXT EMBEDDING STRESS TEST (batch_size={BATCH_SIZE_FOR_TEST}) ===")
    print(f"  Model: {EMBED_MODEL}")
    print(f"  Batch concurrency levels: {BATCH_CONCURRENCY}")
    print(f"  Requests per level: {REQUESTS_PER_LEVEL}")

    # Prepare batches
    batches = []
    for i in range(REQUESTS_PER_LEVEL):
        batch_texts = [random.choice(texts) for _ in range(BATCH_SIZE_FOR_TEST)]
        batches.append(batch_texts)

    reports: list[LevelReport] = []
    for concurrency in BATCH_CONCURRENCY:
        sem = asyncio.Semaphore(concurrency)
        tasks = [embed_batch(client, batch, sem) for batch in batches]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        report = compute_report(list(results), concurrency, "batch", duration)
        print_level(report)
        reports.append(report)
        if concurrency != BATCH_CONCURRENCY[-1]:
            await asyncio.sleep(1.5)
    return reports


async def main() -> None:
    print("=" * 70)
    print("  TEXT EMBEDDING STRESS TEST")
    print(f"  Endpoint: {EMBED_URL}")
    print(f"  Model:    {EMBED_MODEL}")
    print(f"  Timeout:  {TIMEOUT_SEC}s")
    print("=" * 70)

    # Load texts
    all_texts = load_texts(100)
    print(f"\nLoaded {len(all_texts)} unique text samples")

    limits = httpx.Limits(
        max_connections=max(SINGLE_CONCURRENCY) * 2,
        max_keepalive_connections=max(SINGLE_CONCURRENCY),
    )
    timeout = httpx.Timeout(TIMEOUT_SEC)
    all_reports: dict[str, list[LevelReport]] = {}

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Warm-up
        print("\n[WARMUP] Sanity check...")
        warm_sem = asyncio.Semaphore(1)
        w = await embed_single(client, all_texts[0], warm_sem)
        print(f"  Single embed: {'OK' if w.success else 'FAIL'} ({w.latency_sec*1000:.0f}ms, dim={w.vector_dim}) error={w.error[:80] if w.error else ''}")

        if not w.success:
            print(f"\n*** ERROR: Warm-up failed. Check endpoint and API key. ***")
            print(f"    URL: {EMBED_URL}")
            print(f"    Error: {w.error}")
            return

        # Run tests
        all_reports["single"] = await run_single_text_tests(client, all_texts)
        all_reports["batch"] = await run_batch_text_tests(client, all_texts)

    # Summary
    print_summary(all_reports)

    # Recommendations
    print("\n--- Tuning Recommendations ---")
    for mode, reports in all_reports.items():
        print(f"\n  {mode.upper()} TEXT EMBEDDING:")
        for r in reports:
            success_rate = (r.success / r.total * 100) if r.total > 0 else 0
            if success_rate >= 99 and r.p95_ms < 5000:
                status = "SAFE"
            elif success_rate >= 95 and r.p95_ms < 10000:
                status = "CAUTIOUS"
            elif success_rate >= 90:
                status = "RISKY"
            else:
                status = "UNSAFE"
            extra = ""
            if mode == "batch":
                texts_per_sec = r.throughput_rps * BATCH_SIZE_FOR_TEST
                extra = f"  texts/s={texts_per_sec:.0f}"
            print(
                f"    concurrency={r.concurrency:>4}: {status:>9}  "
                f"success={success_rate:.1f}%  p95={r.p95_ms:.0f}ms  rps={r.throughput_rps:.1f}{extra}"
            )

    # Best concurrency suggestion
    print("\n--- Suggested Settings ---")
    for mode, reports in all_reports.items():
        best = None
        for r in reports:
            success_rate = (r.success / r.total * 100) if r.total > 0 else 0
            if success_rate >= 99.5 and r.p95_ms < 3000:
                best = r
        if best:
            print(f"  {mode}: IMAGE_TEXT_EMBED_CONCURRENCY = {best.concurrency}  (safe, p95={best.p95_ms:.0f}ms)")
        else:
            # Pick the highest safe one
            for r in reports:
                success_rate = (r.success / r.total * 100) if r.total > 0 else 0
                if success_rate >= 99:
                    best = r
            if best:
                print(f"  {mode}: IMAGE_TEXT_EMBED_CONCURRENCY = {best.concurrency}  (borderline, p95={best.p95_ms:.0f}ms)")


if __name__ == "__main__":
    asyncio.run(main())
