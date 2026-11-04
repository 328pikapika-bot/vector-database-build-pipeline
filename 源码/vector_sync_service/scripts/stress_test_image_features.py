"""
Image feature pipeline stress test.
Tests VL embed, VL chat, and text embed model endpoints at increasing concurrency.
Reports throughput, latency percentiles, and error rates for each model endpoint.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image as PILImage

# --- Config ---
VL_EMBED_URL = "http://127.0.0.1:16703/v1/embeddings"
VL_EMBED_MODEL = "qwen3-vl-embed"
VL_CHAT_URL = "http://127.0.0.1:50019/v1/chat/completions"
VL_CHAT_MODEL = "qwen3.5-122b"
TEXT_EMBED_URL = "http://127.0.0.1:16703/v1/embeddings"
TEXT_EMBED_MODEL = "qwen3-embed"
API_KEY = os.environ.get("IMAGE_MODEL_API_KEY") or os.environ.get(
    "EMBEDDING_API_KEY", ""
)

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 24, 32, 48]
REQUESTS_PER_LEVEL = 40
FEATURE_TIMEOUT_SEC = 60
DOWNLOAD_URLS_FILE = Path(__file__).resolve().parent / "test_image_urls.txt"
TEXT_SAMPLES_FILE = Path(__file__).resolve().parent / "test_text_samples.txt"
STATIC_BASE_URL = "http://127.0.0.1:12001"

# For VL embed, use a small test image encoded as data URL
# We'll download real images first for proper testing
SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent.parent / "upload" / "stress_test_samples"


@dataclass
class RequestResult:
    endpoint: str = ""
    success: bool = False
    latency_sec: float = 0.0
    size_bytes: int = 0
    error: str = ""
    status_code: int = 0


@dataclass
class LevelReport:
    concurrency: int
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
    error_counts: dict[str, int]


async def download_sample_images(limit: int = 5) -> list[bytes]:
    """Download a few real images to use as test payloads, with local fallback."""
    images = []

    # Try downloading from static server first
    urls = load_lines(DOWNLOAD_URLS_FILE, limit)
    full_urls = [f"{STATIC_BASE_URL.rstrip('/')}/{u.lstrip('/')}" for u in urls]
    async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
        for url in full_urls[:limit]:
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and len(resp.content) > 100:
                    images.append(resp.content)
                    print(f"  Downloaded: {len(resp.content)} bytes from {url[:80]}...")
            except Exception:
                pass

    # Fallback: load from local upload/ directory
    if not images:
        print("  Static server unavailable, loading from local cache...")
        upload_dir = Path(__file__).resolve().parent.parent / "upload"
        for root, dirs, files in os.walk(upload_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and len(images) < limit:
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, 'rb') as fh:
                            data = fh.read()
                            if len(data) > 100:
                                images.append(data)
                                print(f"  Loaded local: {f} ({len(data)} bytes)")
                    except Exception:
                        pass
                if len(images) >= limit:
                    break
            if len(images) >= limit:
                break

    return images


def image_bytes_to_data_url(image_bytes: bytes, max_pixels: int = 1_200_000) -> str:
    """Convert image bytes to a data URL suitable for VL model input."""
    with PILImage.open(BytesIO(image_bytes)) as img:
        from PIL import ImageOps
        converted = ImageOps.exif_transpose(img)
        if converted.mode in {"RGBA", "LA"}:
            bg = PILImage.new("RGB", converted.size, "white")
            if converted.mode == "RGBA":
                bg.paste(converted, mask=converted.split()[-1])
            else:
                bg.paste(converted)
            converted = bg
        else:
            converted = converted.convert("RGB")

        # Resize if too large
        pixels = converted.width * converted.height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            new_size = (max(int(converted.width * scale), 1), max(int(converted.height * scale), 1))
            converted = converted.resize(new_size, PILImage.LANCZOS)

        buf = BytesIO()
        converted.save(buf, format="JPEG", quality=82, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_lines(filepath: Path, limit: int) -> list[str]:
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()][:limit]
    return []


def build_auth_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }


# ===== VL Embed =====
async def vl_embed_one(client: httpx.AsyncClient, data_url: str, sem: asyncio.Semaphore) -> RequestResult:
    result = RequestResult(endpoint="vl_embed")
    payload = {
        "model": VL_EMBED_MODEL,
        "input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}],
        "encoding_format": "float",
    }
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(VL_EMBED_URL, json=payload, headers=build_auth_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        result.success = resp.status_code == 200
        if result.success:
            result.size_bytes = len(resp.content)
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


# ===== VL Chat =====
async def vl_chat_one(client: httpx.AsyncClient, data_url: str, sem: asyncio.Semaphore) -> RequestResult:
    result = RequestResult(endpoint="vl_chat")
    payload = {
        "model": VL_CHAT_MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Describe this scientific image briefly in 1-2 sentences."},
            ]},
        ],
        "max_tokens": 500,
        "temperature": 0.2,
    }
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(VL_CHAT_URL, json=payload, headers=build_auth_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        result.success = resp.status_code == 200
        if result.success:
            result.size_bytes = len(resp.content)
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


# ===== Text Embed =====
async def text_embed_one(client: httpx.AsyncClient, text: str, sem: asyncio.Semaphore) -> RequestResult:
    result = RequestResult(endpoint="text_embed")
    payload = {
        "model": TEXT_EMBED_MODEL,
        "input": text,
        "encoding_format": "float",
    }
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(TEXT_EMBED_URL, json=payload, headers=build_auth_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        result.success = resp.status_code == 200
        if result.success:
            result.size_bytes = len(resp.content)
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


def compute_report(results: list[RequestResult], concurrency: int, duration: float) -> LevelReport:
    success = sum(1 for r in results if r.success)
    failed = len(results) - success
    latencies = [r.latency_sec for r in results if r.success]
    error_counts: dict[str, int] = {}
    for r in results:
        if not r.success:
            key = r.error.split(":")[0] if r.error else f"HTTP_{r.status_code}"
            error_counts[key] = error_counts.get(key, 0) + 1

    sorted_lat = sorted(latencies)
    return LevelReport(
        concurrency=concurrency,
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
        error_counts=error_counts,
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(p * len(sorted_values))))
    return sorted_values[idx]


def print_report(report: LevelReport, label: str) -> None:
    success_rate = (report.success / report.total * 100) if report.total > 0 else 0
    print(f"  [{label}] concurrency={report.concurrency} | "
          f"success={report.success}/{report.total} ({success_rate:.1f}%) | "
          f"avg={report.avg_ms:.0f}ms p50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms p99={report.p99_ms:.0f}ms | "
          f"rps={report.throughput_rps:.1f}")
    if report.error_counts:
        for err, cnt in sorted(report.error_counts.items(), key=lambda x: -x[1])[:3]:
            print(f"         errors: {err}={cnt}")


def print_summary(reports: dict[str, list[LevelReport]]) -> None:
    endpoints = list(reports.keys())
    print(f"\n{'='*100}")
    header = f"{'Endpoint':<15} {'Conc':>5} {'Succ':>5} {'Rate':>7} {'RPS':>7} {'avg_ms':>7} {'p50_ms':>7} {'p95_ms':>8} {'p99_ms':>8} {'max_ms':>7}"
    print(header)
    print("-" * 100)
    for ep in endpoints:
        for r in reports[ep]:
            rate = (r.success / r.total * 100) if r.total > 0 else 0
            print(f"{ep:<15} {r.concurrency:>5} {r.success:>5} {rate:>6.1f}% {r.throughput_rps:>6.1f} "
                  f"{r.avg_ms:>7.0f} {r.p50_ms:>7.0f} {r.p95_ms:>8.0f} {r.p99_ms:>8.0f} {r.max_ms:>7.0f}")
    print(f"{'='*100}")


async def run_stress_test(
    test_fn,
    client: httpx.AsyncClient,
    payloads: list[str],
    concurrency_levels: list[int],
    requests_per_level: int,
    label: str,
) -> list[LevelReport]:
    reports: list[LevelReport] = []
    for concurrency in concurrency_levels:
        sem = asyncio.Semaphore(concurrency)
        tasks = [test_fn(client, payloads[i % len(payloads)], sem) for i in range(requests_per_level)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        report = compute_report(list(results), concurrency, duration)
        print_report(report, label)
        reports.append(report)
        if concurrency != concurrency_levels[-1]:
            await asyncio.sleep(1.5)
    return reports


async def main() -> None:
    print("=" * 70)
    print("  IMAGE FEATURE PIPELINE STRESS TEST")
    print(f"  VL Embed URL:  {VL_EMBED_URL} ({VL_EMBED_MODEL})")
    print(f"  VL Chat URL:   {VL_CHAT_URL} ({VL_CHAT_MODEL})")
    print(f"  Text Embed URL: {TEXT_EMBED_URL} ({TEXT_EMBED_MODEL})")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"  Requests per level: {REQUESTS_PER_LEVEL}")
    print("=" * 70)

    # Step 1: Download sample images
    print("\n[SETUP] Downloading sample images for VL model testing...")
    image_bytes_list = await download_sample_images(5)
    if not image_bytes_list:
        print("[ERROR] Could not download any sample images. Check network.")
        # Create a tiny synthetic image as fallback
        print("[FALLBACK] Creating synthetic 1x1 JPEG image...")
        buf = BytesIO()
        img = PILImage.new("RGB", (100, 100), color="blue")
        img.save(buf, format="JPEG")
        image_bytes_list = [buf.getvalue()]
        print(f"  Created synthetic image: {len(image_bytes_list[0])} bytes")

    # Step 2: Convert to data URLs
    print(f"\n[SETUP] Converting {len(image_bytes_list)} images to data URLs...")
    data_urls = []
    for i, img_bytes in enumerate(image_bytes_list):
        try:
            data_url = image_bytes_to_data_url(img_bytes)
            data_urls.append(data_url)
            print(f"  Image {i+1}: {len(img_bytes)} bytes -> data_url {len(data_url)} chars")
        except Exception as e:
            print(f"  Image {i+1}: conversion failed: {e}")

    if not data_urls:
        print("[ERROR] No data URLs generated. Aborting.")
        return

    # Step 3: Load text samples
    text_samples = load_lines(TEXT_SAMPLES_FILE, 20)
    if not text_samples:
        print("[WARN] No text samples found, using defaults")
        text_samples = [
            "Astaxanthin is a ketocarotenoid found in many marine organisms.",
            "The study examines factors influencing mobile commerce adoption in Ghana.",
            "Mitochondrial DNA mutations cause various cytopathies including MELAS syndrome.",
        ] * 7

    print(f"\n[SETUP] Testing with {len(data_urls)} image data URLs and {len(text_samples)} text samples")

    limits = httpx.Limits(
        max_connections=max(CONCURRENCY_LEVELS) * 2,
        max_keepalive_connections=max(CONCURRENCY_LEVELS),
    )
    timeout = httpx.Timeout(FEATURE_TIMEOUT_SEC)
    all_reports: dict[str, list[LevelReport]] = {}

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Warm-up / sanity check
        print("\n[WARMUP] Sanity checks...")
        warm_sem = asyncio.Semaphore(1)
        w1 = await vl_embed_one(client, data_urls[0], warm_sem)
        print(f"  VL Embed: {'OK' if w1.success else 'FAIL'} ({w1.latency_sec*1000:.0f}ms) error={w1.error[:60] if w1.error else ''}")
        w2 = await text_embed_one(client, text_samples[0], warm_sem)
        print(f"  Text Embed: {'OK' if w2.success else 'FAIL'} ({w2.latency_sec*1000:.0f}ms) error={w2.error[:60] if w2.error else ''}")
        w3 = await vl_chat_one(client, data_urls[0], warm_sem)
        print(f"  VL Chat: {'OK' if w3.success else 'FAIL'} ({w3.latency_sec*1000:.0f}ms) error={w3.error[:60] if w3.error else ''}")

        # VL Embed stress test
        print("\n\n=== VL EMBED STRESS TEST (qwen3-vl-embed) ===")
        all_reports["vl_embed"] = await run_stress_test(
            vl_embed_one, client, data_urls, CONCURRENCY_LEVELS, REQUESTS_PER_LEVEL, "vl_embed"
        )

        # Text Embed stress test
        print("\n\n=== TEXT EMBED STRESS TEST (qwen3-embed) ===")
        all_reports["text_embed"] = await run_stress_test(
            text_embed_one, client, text_samples, CONCURRENCY_LEVELS, REQUESTS_PER_LEVEL, "text_embed"
        )

        # VL Chat stress test (shorter concurrency levels since this is heavier)
        chat_levels = [1, 2, 4, 8, 16]
        print("\n\n=== VL CHAT STRESS TEST (qwen3.5-plus) ===")
        all_reports["vl_chat"] = await run_stress_test(
            vl_chat_one, client, data_urls, chat_levels, max(20, REQUESTS_PER_LEVEL // 2), "vl_chat"
        )

    # Summary
    print_summary(all_reports)

    # Recommendations
    print("\n--- Tuning Recommendations ---")
    for ep_name, reports in all_reports.items():
        print(f"\n  {ep_name.upper()}:")
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
            print(f"    concurrency={r.concurrency:>3}: {status:>9}  success={success_rate:.1f}%  p95={r.p95_ms:.0f}ms  rps={r.throughput_rps:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
