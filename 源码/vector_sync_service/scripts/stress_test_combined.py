"""
Combined VL Embed + Text Embed mixed stress test.
Simulates real-world load where both pipelines run simultaneously.
Finds optimal concurrency pair for highest total throughput with p95 < 2000ms.
"""
from __future__ import annotations

import asyncio
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image as PILImage
from io import BytesIO
import base64

# --- Config ---
VL_EMBED_URL = "http://127.0.0.1:16703/v1/embeddings"
VL_EMBED_MODEL = "qwen3-vl-embed"
TEXT_EMBED_URL = "http://127.0.0.1:16703/v1/embeddings"
TEXT_EMBED_MODEL = "qwen3-embed"
API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

TOTAL_REQUESTS = 60
TIMEOUT_SEC = 120

# Test matrix: (vl_concurrency, text_concurrency) pairs to test
# Total concurrency covers range from 4 to 64
TEST_MATRIX = [
    # (vl, text, total)
    (2, 2, 4),
    (2, 8, 10),
    (4, 8, 12),
    (4, 16, 20),
    (8, 16, 24),
    (6, 24, 30),
    (8, 24, 32),
    (8, 32, 40),
    (12, 32, 44),
    (12, 36, 48),
    (16, 32, 48),
    (16, 40, 56),
    (20, 44, 64),
]


@dataclass
class RequestResult:
    kind: str  # "vl" or "text"
    success: bool = False
    latency_sec: float = 0.0
    error: str = ""
    status_code: int = 0


@dataclass
class CombinedReport:
    vl_concurrency: int
    text_concurrency: int
    total_concurrency: int
    duration_sec: float
    vl_success: int
    vl_failed: int
    vl_avg_ms: float
    vl_p50_ms: float
    vl_p95_ms: float
    vl_p99_ms: float
    vl_rps: float
    text_success: int
    text_failed: int
    text_avg_ms: float
    text_p50_ms: float
    text_p95_ms: float
    text_p99_ms: float
    text_rps: float
    total_rps: float
    vl_errors: dict[str, int]
    text_errors: dict[str, int]
    score: float = 0.0  # composite score (higher = better)


def find_local_images(limit: int) -> list[bytes]:
    """Load local images for VL embed testing."""
    images = []
    upload_dir = Path(__file__).resolve().parent.parent / "upload"
    if upload_dir.exists():
        for root, dirs, files in os.walk(upload_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and len(images) < limit:
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, 'rb') as fh:
                            data = fh.read()
                            if len(data) > 1000:
                                images.append(data)
                    except Exception:
                        pass
                if len(images) >= limit:
                    break
            if len(images) >= limit:
                break
    return images


def image_bytes_to_data_url(image_bytes: bytes, max_pixels: int = 600_000) -> str:
    """Convert to small data URL for faster VL embed testing."""
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
        pixels = converted.width * converted.height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            new_size = (max(int(converted.width * scale), 1), max(int(converted.height * scale), 1))
            converted = converted.resize(new_size, PILImage.LANCZOS)
        buf = BytesIO()
        converted.save(buf, format="JPEG", quality=78, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}


async def vl_embed_one(client: httpx.AsyncClient, data_url: str, sem: asyncio.Semaphore) -> RequestResult:
    result = RequestResult(kind="vl")
    payload = {
        "model": VL_EMBED_MODEL,
        "input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}],
        "encoding_format": "float",
    }
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(VL_EMBED_URL, json=payload, headers=build_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        result.success = resp.status_code == 200
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


async def text_embed_one(client: httpx.AsyncClient, text: str, sem: asyncio.Semaphore) -> RequestResult:
    result = RequestResult(kind="text")
    payload = {"model": TEXT_EMBED_MODEL, "input": text, "encoding_format": "float"}
    start = time.perf_counter()
    try:
        async with sem:
            resp = await client.post(TEXT_EMBED_URL, json=payload, headers=build_headers())
        result.latency_sec = time.perf_counter() - start
        result.status_code = resp.status_code
        result.success = resp.status_code == 200
    except Exception as e:
        result.latency_sec = time.perf_counter() - start
        result.error = f"{type(e).__name__}: {e}"
    return result


def _p(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(pct * len(sorted_vals))))
    return sorted_vals[idx] * 1000


def compute_combined(
    vl_results: list[RequestResult],
    text_results: list[RequestResult],
    vl_conc: int,
    text_conc: int,
    duration: float,
) -> CombinedReport:
    r = CombinedReport(
        vl_concurrency=vl_conc,
        text_concurrency=text_conc,
        total_concurrency=vl_conc + text_conc,
        duration_sec=duration,
        vl_success=0, vl_failed=0, vl_avg_ms=0, vl_p50_ms=0, vl_p95_ms=0, vl_p99_ms=0, vl_rps=0,
        text_success=0, text_failed=0, text_avg_ms=0, text_p50_ms=0, text_p95_ms=0, text_p99_ms=0, text_rps=0,
        total_rps=0,
        vl_errors={}, text_errors={},
    )

    for results, kind in [(vl_results, "vl"), (text_results, "text")]:
        success = sum(1 for res in results if res.success)
        failed = len(results) - success
        lats = sorted([res.latency_sec for res in results if res.success])
        errors: dict[str, int] = {}
        for res in results:
            if not res.success:
                k = res.error.split(":")[0] if res.error else f"HTTP_{res.status_code}"
                errors[k] = errors.get(k, 0) + 1

        avg = statistics.mean(lats) * 1000 if lats else 0
        p50 = _p(lats, 0.50)
        p95 = _p(lats, 0.95)
        p99 = _p(lats, 0.99)
        rps = len(results) / duration if duration > 0 else 0

        if kind == "vl":
            r.vl_success = success; r.vl_failed = failed
            r.vl_avg_ms = avg; r.vl_p50_ms = p50; r.vl_p95_ms = p95; r.vl_p99_ms = p99
            r.vl_rps = rps; r.vl_errors = errors
        else:
            r.text_success = success; r.text_failed = failed
            r.text_avg_ms = avg; r.text_p50_ms = p50; r.text_p95_ms = p95; r.text_p99_ms = p99
            r.text_rps = rps; r.text_errors = errors

    r.total_rps = r.vl_rps + r.text_rps

    # Composite score: reward throughput, penalize high p95 and errors
    vl_rate = r.vl_success / max(len(vl_results), 1)
    text_rate = r.text_success / max(len(text_results), 1)
    success_penalty = vl_rate * text_rate  # 0-1

    # Penalty for p95 > 2000ms
    vl_p95_penalty = max(0, (r.vl_p95_ms - 2000) / 2000)
    text_p95_penalty = max(0, (r.text_p95_ms - 2000) / 2000)
    p95_penalty = 1.0 / (1.0 + vl_p95_penalty + text_p95_penalty)

    r.score = r.total_rps * success_penalty * p95_penalty
    return r


def print_row(r: CombinedReport) -> None:
    vl_rate = (r.vl_success / max(r.vl_success + r.vl_failed, 1)) * 100
    text_rate = (r.text_success / max(r.text_success + r.text_failed, 1)) * 100
    print(
        f"  VL={r.vl_concurrency:>3} T={r.text_concurrency:>3} | "
        f"总={r.total_concurrency:>3} | "
        f"VL: {r.vl_success:>3}/{r.vl_success+r.vl_failed:>3} {vl_rate:>5.1f}% "
        f"p50={r.vl_p50_ms:>6.0f}ms p95={r.vl_p95_ms:>6.0f}ms | "
        f"Text: {r.text_success:>3}/{r.text_success+r.text_failed:>3} {text_rate:>5.1f}% "
        f"p50={r.text_p50_ms:>5.0f}ms p95={r.text_p95_ms:>5.0f}ms | "
        f"总RPS={r.total_rps:>5.1f} 评分={r.score:>5.1f}"
    )
    if r.vl_errors or r.text_errors:
        errs = {}
        errs.update({f"VL:{k}": v for k, v in r.vl_errors.items()})
        errs.update({f"T:{k}": v for k, v in r.text_errors.items()})
        print(f"         ERR {errs}")


async def main() -> None:
    print("=" * 75)
    print("  混合压测: VL Embed + Text Embed 同时并发")
    print(f"  服务器: {VL_EMBED_URL}")
    print(f"  总请求数/轮: {TOTAL_REQUESTS}")
    print("=" * 75)

    # Load test data
    images = find_local_images(8)
    if not images:
        print("[ERROR] No local images found")
        return
    data_urls = [image_bytes_to_data_url(img) for img in images]
    print(f"\n加载了 {len(data_urls)} 张本地图片用于 VL Embed 测试")

    texts = [
        "Astaxanthin is a ketocarotenoid found in many marine organisms with significant antioxidant properties and health benefits.",
        "The study examines factors influencing mobile commerce adoption among small enterprises in developing economies.",
        "Mutations in mitochondrial DNA cause various cytopathies including MELAS syndrome. The A3243G transition is most common.",
        "Quantum computing offers advantages for optimization problems, with error-corrected qubits remaining a key research challenge.",
        "Deep learning transformer architectures have achieved state-of-the-art performance in natural language processing.",
        "Climate change impacts on biodiversity have been documented across multiple ecosystems worldwide.",
        "The CRISPR-Cas9 system has revolutionized gene editing with applications from basic research to therapeutic development.",
        "Synthetic biology combines engineering principles with biological systems for novel biosensor and therapeutic applications.",
    ] * 10
    print(f"加载了 {len(texts)} 条文本样本\n")

    limits = httpx.Limits(max_connections=256, max_keepalive_connections=128)
    timeout = httpx.Timeout(TIMEOUT_SEC)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Warm-up
        print("[预热] 连通性检查...")
        w1 = await vl_embed_one(client, data_urls[0], asyncio.Semaphore(1))
        w2 = await text_embed_one(client, texts[0], asyncio.Semaphore(1))
        print(f"  VL Embed: {'OK' if w1.success else 'FAIL'} ({w1.latency_sec*1000:.0f}ms)")
        print(f"  Text Embed: {'OK' if w2.success else 'FAIL'} ({w2.latency_sec*1000:.0f}ms)")
        if not w1.success or not w2.success:
            print("[ERROR] 连通性检查失败")
            return

        # Run test matrix
        print(f"\n{'='*75}")
        print(f"  VL并发 Text并发 | 总计 | VL结果                     | Text结果                   | 总吞吐  评分")
        print(f"{'-'*75}")

        reports: list[CombinedReport] = []
        for vl_conc, text_conc, total in TEST_MATRIX:
            vl_sem = asyncio.Semaphore(vl_conc)
            text_sem = asyncio.Semaphore(text_conc)

            vl_count = int(TOTAL_REQUESTS * 0.3)  # 30% VL, 70% text (realistic ratio)
            text_count = TOTAL_REQUESTS - vl_count

            vl_tasks = [vl_embed_one(client, random.choice(data_urls), vl_sem) for _ in range(vl_count)]
            text_tasks = [text_embed_one(client, random.choice(texts), text_sem) for _ in range(text_count)]

            start = time.perf_counter()
            all_results = await asyncio.gather(*(vl_tasks + text_tasks))
            duration = time.perf_counter() - start

            vl_results = [r for r in all_results if r.kind == "vl"]
            text_results = [r for r in all_results if r.kind == "text"]

            report = compute_combined(vl_results, text_results, vl_conc, text_conc, duration)
            print_row(report)
            reports.append(report)

            await asyncio.sleep(1.5)

    # Find best configs
    print(f"\n{'='*75}")
    print("  TOP 5 最佳配置 (按综合评分排序)")
    print(f"{'='*75}")
    sorted_reports = sorted(reports, key=lambda r: r.score, reverse=True)
    for i, r in enumerate(sorted_reports[:5]):
        print(f"  #{i+1} VL={r.vl_concurrency} Text={r.text_concurrency} "
              f"总并发={r.total_concurrency} 总RPS={r.total_rps:.1f} 评分={r.score:.1f} "
              f"VL_p95={r.vl_p95_ms:.0f}ms Text_p95={r.text_p95_ms:.0f}ms")

    # Recommendation
    best = sorted_reports[0]
    print(f"\n{'='*75}")
    print(f"  [推荐配置] (本地32G内存 / 最高吞吐)")
    print(f"{'='*75}")
    print(f"  IMAGE_VL_EMBED_CONCURRENCY  = {best.vl_concurrency}")
    print(f"  IMAGE_TEXT_EMBED_CONCURRENCY = {best.text_concurrency}")
    print(f"  EMBEDDING_MAX_CONCURRENCY    = {best.total_concurrency}")
    print(f"")
    print(f"  预期性能:")
    print(f"    VL Embed:  p50={best.vl_p50_ms:.0f}ms  p95={best.vl_p95_ms:.0f}ms  RPS={best.vl_rps:.1f}")
    print(f"    Text Embed: p50={best.text_p50_ms:.0f}ms  p95={best.text_p95_ms:.0f}ms  RPS={best.text_rps:.1f}")
    print(f"    总吞吐: {best.total_rps:.1f} req/s")

    # Also show safe upper bound
    safe_reports = [r for r in reports if r.vl_p95_ms < 1500 and r.text_p95_ms < 1500
                    and (r.vl_success+r.vl_failed > 0 and r.vl_success/(r.vl_success+r.vl_failed) > 0.98)
                    and (r.text_success+r.text_failed > 0 and r.text_success/(r.text_success+r.text_failed) > 0.98)]
    if safe_reports:
        safe_best = max(safe_reports, key=lambda r: r.total_rps)
        print(f"\n  Safe upper bound 安全上限配置 (p95<1500ms, 成功率>98%):")
        print(f"    IMAGE_VL_EMBED_CONCURRENCY  = {safe_best.vl_concurrency}")
        print(f"    IMAGE_TEXT_EMBED_CONCURRENCY = {safe_best.text_concurrency}")
        print(f"    总RPS={safe_best.total_rps:.1f} VL_p95={safe_best.vl_p95_ms:.0f}ms Text_p95={safe_best.text_p95_ms:.0f}ms")
    print(f"{'='*75}")


if __name__ == "__main__":
    asyncio.run(main())
