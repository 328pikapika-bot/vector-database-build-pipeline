"""
Image download concurrency stress test (local mode).
Since static.nstl.gov.cn is not accessible from this machine,
this test creates a local HTTP server serving cached images and
tests the httpx downloader's concurrency handling capability.
"""
from __future__ import annotations

import asyncio
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

# --- Config ---
LOCAL_PORT = 18790
LOCAL_BASE_URL = f"http://127.0.0.1:{LOCAL_PORT}"
CONCURRENCY_LEVELS = [1, 4, 8, 16, 24, 32, 48, 64, 96, 128]
REQUESTS_PER_LEVEL = 80
TIMEOUT_SEC = 60


@dataclass
class RequestResult:
    url: str = ""
    status_code: int = 0
    success: bool = False
    latency_sec: float = 0.0
    size_bytes: int = 0
    error: str = ""


@dataclass
class LevelReport:
    concurrency: int
    total: int = 0
    success: int = 0
    failed: int = 0
    total_duration_sec: float = 0.0
    latencies: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    throughput_rps: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    avg_size_kb: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)


def find_local_images(limit: int) -> list[str]:
    """Find local image files for testing."""
    images = []
    upload_dir = Path(__file__).resolve().parent.parent / "upload"
    if upload_dir.exists():
        for root, dirs, files in os.walk(upload_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and len(images) < limit:
                    fpath = os.path.join(root, f)
                    if os.path.getsize(fpath) > 100:
                        images.append(fpath)
                if len(images) >= limit:
                    break
            if len(images) >= limit:
                break
    return images


def start_local_server(serve_dir: str) -> HTTPServer:
    """Start a simple HTTP file server in a thread."""

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress logs

    server = HTTPServer(("127.0.0.1", LOCAL_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def download_one(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    result = RequestResult(url=url)
    start = time.perf_counter()
    try:
        async with semaphore:
            response = await client.get(url)
        result.latency_sec = time.perf_counter() - start
        result.status_code = response.status_code
        result.size_bytes = len(response.content)
        result.success = response.status_code == 200 and len(response.content) > 100
    except httpx.TimeoutException:
        result.latency_sec = time.perf_counter() - start
        result.error = "timeout"
    except httpx.ConnectError as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"connect_error: {e}"
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


def compute_report(results: list[RequestResult], concurrency: int, duration: float) -> LevelReport:
    report = LevelReport(concurrency=concurrency)
    report.total = len(results)
    report.total_duration_sec = duration

    for r in results:
        if r.success:
            report.success += 1
            report.latencies.append(r.latency_sec)
            report.sizes.append(r.size_bytes)
        else:
            report.failed += 1
            error_type = r.error.split(":")[0] if r.error else f"HTTP_{r.status_code}"
            report.errors[error_type] = report.errors.get(error_type, 0) + 1

    report.throughput_rps = report.total / duration if duration > 0 else 0

    if report.latencies:
        sorted_lat = sorted(report.latencies)
        report.min_ms = sorted_lat[0] * 1000
        report.max_ms = sorted_lat[-1] * 1000
        report.avg_ms = statistics.mean(report.latencies) * 1000
        report.p50_ms = _percentile(sorted_lat, 0.50) * 1000
        report.p95_ms = _percentile(sorted_lat, 0.95) * 1000
        report.p99_ms = _percentile(sorted_lat, 0.99) * 1000

    if report.sizes:
        report.avg_size_kb = statistics.mean(report.sizes) / 1024

    return report


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(p * len(sorted_values))))
    return sorted_values[idx]


def print_report(report: LevelReport) -> None:
    success_rate = (report.success / report.total * 100) if report.total > 0 else 0
    print(
        f"  conc={report.concurrency:>4} | "
        f"succ={report.success:>3}/{report.total:>3} ({success_rate:>5.1f}%) | "
        f"avg={report.avg_ms:>7.1f}ms p50={report.p50_ms:>7.1f}ms p95={report.p95_ms:>7.1f}ms "
        f"p99={report.p99_ms:>7.1f}ms | rps={report.throughput_rps:>8.1f} | "
        f"avg_size={report.avg_size_kb:>6.1f}KB"
    )
    if report.errors:
        top_errors = sorted(report.errors.items(), key=lambda x: -x[1])[:3]
        print(f"         errors: {' | '.join(f'{k}={v}' for k, v in top_errors)}")


def print_summary_table(reports: list[LevelReport]) -> None:
    print(f"\n{'='*100}")
    header = (
        f"{'Conc':>5} {'Succ':>5} {'Fail':>5} {'Rate':>7} {'RPS':>8} "
        f"{'avg_ms':>7} {'p50_ms':>7} {'p95_ms':>8} {'p99_ms':>8} {'max_ms':>7} {'KB':>7}"
    )
    print(header)
    print("-" * 100)
    for r in reports:
        rate = (r.success / r.total * 100) if r.total > 0 else 0
        print(
            f"{r.concurrency:>5} {r.success:>5} {r.failed:>5} {rate:>6.1f}% "
            f"{r.throughput_rps:>7.1f} {r.avg_ms:>7.1f} {r.p50_ms:>7.1f} "
            f"{r.p95_ms:>8.1f} {r.p99_ms:>8.1f} {r.max_ms:>7.1f} {r.avg_size_kb:>6.1f}"
        )
    print(f"{'='*100}")


async def run_level(
    client: httpx.AsyncClient,
    server_dir: str,
    image_files: list[str],
    concurrency: int,
    total_requests: int,
) -> LevelReport:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for i in range(total_requests):
        fname = os.path.basename(image_files[i % len(image_files)])
        url = f"{LOCAL_BASE_URL}/{fname}"
        tasks.append(download_one(client, url, semaphore))

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start

    return compute_report(list(results), concurrency, duration)


async def main() -> None:
    print("=" * 70)
    print("  IMAGE DOWNLOAD CONCURRENCY STRESS TEST (LOCAL MODE)")
    print(f"  Note: static.nstl.gov.cn returns 404, using local server")
    print(f"  Local server: {LOCAL_BASE_URL}")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"  Requests per level: {REQUESTS_PER_LEVEL}")
    print("=" * 70)

    # Find local images
    images = find_local_images(30)
    if not images:
        print("[ERROR] No local images found in upload/ directory")
        return
    print(f"\nFound {len(images)} local images for testing")

    # Determine the directory containing the images to serve
    serve_dir = str(Path(images[0]).parent)
    # Copy images to a single temp dir for simpler serving
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="stress_test_images_")
    for img in images:
        shutil.copy2(img, os.path.join(tmpdir, os.path.basename(img)))
    print(f"Copied images to temp serving dir: {tmpdir}")

    # Start local server
    server = start_local_server(tmpdir)
    print(f"Local HTTP server started on port {LOCAL_PORT}")

    try:
        limits = httpx.Limits(
            max_connections=max(CONCURRENCY_LEVELS) * 2,
            max_keepalive_connections=max(CONCURRENCY_LEVELS),
        )
        timeout = httpx.Timeout(TIMEOUT_SEC)
        image_basenames = [os.path.basename(img) for img in images]

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            # Warm-up
            print("\n[WARMUP] Sanity check...")
            warm = await download_one(client, f"{LOCAL_BASE_URL}/{image_basenames[0]}", asyncio.Semaphore(1))
            if warm.success:
                print(f"  OK - {warm.size_bytes} bytes, {warm.latency_sec*1000:.0f}ms")
            else:
                print(f"  FAIL - {warm.error}")
                return

            reports: list[LevelReport] = []
            for concurrency in CONCURRENCY_LEVELS:
                print(f"\n[TEST] Concurrency = {concurrency}...")
                report = await run_level(client, tmpdir, images, concurrency, REQUESTS_PER_LEVEL)
                print_report(report)
                reports.append(report)
                if concurrency != CONCURRENCY_LEVELS[-1]:
                    await asyncio.sleep(1.0)

        # Summary
        print_summary_table(reports)

        # Recommendations
        print("\n--- Tuning Recommendations ---")
        for r in reports:
            success_rate = (r.success / r.total * 100) if r.total > 0 else 0
            if success_rate >= 99 and r.p95_ms < 100:
                status = "SAFE"
            elif success_rate >= 95 and r.p95_ms < 500:
                status = "CAUTIOUS"
            elif success_rate >= 90:
                status = "RISKY"
            else:
                status = "UNSAFE"
            print(
                f"  concurrency={r.concurrency:>4}: {status:>9}  "
                f"success={success_rate:.1f}%  p95={r.p95_ms:.1f}ms  rps={r.throughput_rps:.1f}"
            )

    finally:
        server.shutdown()
        # Clean up temp dir
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\nCleaned up temp directory")


if __name__ == "__main__":
    asyncio.run(main())
