#!/usr/bin/env python3
"""Run the staged sync-path benchmark and output a unified report."""

from __future__ import annotations

import argparse
import asyncio
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
    build_pipeline_input,
    compare_stage_bottlenecks,
    cycle_samples,
    default_run_dir,
    force_utf8_stdout,
    scan_upload_samples,
    write_json,
)
from run_image_download_benchmark import run_download_benchmark
from run_image_model_benchmark import run_image_model_benchmark
from run_text_embedding_benchmark import run_text_embedding_benchmark


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.contracts.docx_contract import TEXT_VECTOR_DIM, VISUAL_TEXT_VECTOR_DIM
from app.services.image_milvus_batch_writer import MilvusImageBatchWriter
from app.services.image_offline_providers import OfflineImageModelClient
from app.services.image_features import compute_phash


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged sync-path benchmark.")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--download-seconds", type=int, default=600)
    parser.add_argument("--image-model-seconds", type=int, default=600)
    parser.add_argument("--text-seconds", type=int, default=180)
    parser.add_argument("--write-seconds", type=int, default=120)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--image-collection", default=settings.IMAGE_COLLECTION_NAME)
    parser.add_argument("--print-report", action="store_true")
    return parser.parse_args(argv)


async def _run_milvus_write_probe(
    *,
    samples,
    duration_seconds: int,
    collection_name: str,
) -> dict[str, object]:
    selector = cycle_samples(samples)
    writer = MilvusImageBatchWriter(
        batch_size=settings.IMAGE_MILVUS_WRITE_BATCH_SIZE,
        flush_interval_ms=settings.IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS,
        concurrency=settings.IMAGE_MILVUS_WRITE_CONCURRENCY,
    )
    model_client = OfflineImageModelClient.from_settings()
    result = BenchmarkCaseResult(
        case_name="milvus_write_probe",
        stage="milvus_write_probe",
        concurrency=settings.IMAGE_MILVUS_WRITE_CONCURRENCY,
        batch_size=settings.IMAGE_MILVUS_WRITE_BATCH_SIZE,
    )
    errors = Counter()
    end_time = time.perf_counter() + max(duration_seconds, 1)
    sequence = 0
    try:
        while time.perf_counter() < end_time:
            sample = selector(sequence)
            sequence += 1
            pipeline_input = build_pipeline_input(sample, sequence=sequence)
            started = time.perf_counter()
            result.attempts += 1
            try:
                visual_embed = await model_client.embed_image(pipeline_input.image_bytes)
                text = pipeline_input.context_text.strip() or str(
                    pipeline_input.draft_record.img_meta.get("caption") or "No caption"
                )
                text_embed = await model_client.embed_text(text)
                image_record = pipeline_input.draft_record.model_copy(deep=True)
                image_record.p_hash = pipeline_input.draft_record.p_hash or _compute_phash_bytes(pipeline_input.image_bytes)
                image_record.visual_embed = visual_embed
                image_record.caption_embed = text_embed
                image_record.context_embed = text_embed
                image_record.text = text
                image_record.text_embed = text_embed
                image_record.visual_text_embed = visual_embed[:VISUAL_TEXT_VECTOR_DIM]
                await writer.replace_image_by_img_id(collection_name, image_record)
            except Exception as error:  # noqa: BLE001 - benchmark needs error census
                result.failures += 1
                errors[type(error).__name__] += 1
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result.successes += 1
            result.latencies_ms.append(elapsed_ms)
            result.bytes_processed += len(pipeline_input.image_bytes)
    finally:
        result.duration_seconds = max(duration_seconds, 1)
        result.error_counts = dict(sorted(errors.items()))
        await writer.close()
        http_client = getattr(model_client, "_http_client", None)
        if http_client is not None:
            await http_client.aclose()
    return summarize_stage(
        stage="milvus_write_probe",
        cases=[result],
        extra={
            "collection_name": collection_name,
            "duration_seconds": duration_seconds,
            "vector_dims": {
                "text": TEXT_VECTOR_DIM,
                "visual_text": VISUAL_TEXT_VECTOR_DIM,
            },
        },
    )


def _compute_phash_bytes(image_bytes: bytes) -> bytes:
    result = compute_phash(image_bytes)
    if result.success and result.p_hash is not None:
        return result.p_hash
    return bytes.fromhex("0123456789abcdef")


async def run_sync_path_benchmark(args: argparse.Namespace) -> dict[str, object]:
    samples = scan_upload_samples(args.upload_dir)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise ValueError(f"no upload samples found under {args.upload_dir}")

    run_dir = default_run_dir(
        f"{DEFAULT_BENCHMARK_PREFIX}_full",
        run_id=args.run_id.strip() or None,
    )
    status_path = run_dir / "status.json"
    report_path = run_dir / "report.json"

    status = {
        "started_at": time.time(),
        "upload_dir": str(args.upload_dir),
        "sample_count": len(samples),
        "phases": [],
    }
    write_json(status_path, status)

    download_args = argparse.Namespace(
        upload_dir=args.upload_dir,
        duration_seconds=args.download_seconds,
        warmup_seconds=min(10, args.download_seconds),
        concurrency="8,16,32,48,64,96,128",
        timeout=settings.IMAGE_DOWNLOAD_TIMEOUT,
        max_retries=settings.IMAGE_DOWNLOAD_MAX_RETRIES,
        run_id=args.run_id,
        stage_name="image_download",
        sample_limit=args.sample_limit,
        shared_client=True,
        print_report=False,
    )
    download_result = await run_download_benchmark(download_args)
    status["phases"].append({"stage": "image_download", "report_path": download_result["report_path"]})
    write_json(status_path, status)

    image_model_args = argparse.Namespace(
        upload_dir=args.upload_dir,
        downloaded_dir=Path(),
        duration_seconds=args.image_model_seconds,
        warmup_seconds=min(10, args.image_model_seconds),
        concurrency="1,2,4,8,16,24,32",
        run_id=args.run_id,
        stage_name="image_model",
        sample_limit=args.sample_limit,
        model_url=settings.IMAGE_VL_EMBED_URL,
        model_name=settings.IMAGE_VL_EMBED_MODEL,
        api_key=settings.IMAGE_MODEL_API_KEY,
        timeout=settings.IMAGE_FEATURE_TIMEOUT,
        max_retries=settings.IMAGE_FEATURE_MAX_RETRIES,
        max_pixels=settings.IMAGE_MAX_PIXELS,
        batch_mode=False,
        batch_size=settings.IMAGE_VL_EMBED_BATCH_SIZE,
        print_report=False,
    )
    image_model_result = await run_image_model_benchmark(image_model_args)
    status["phases"].append({"stage": "image_model", "report_path": image_model_result["report_path"]})
    write_json(status_path, status)

    text_args = argparse.Namespace(
        upload_dir=args.upload_dir,
        duration_seconds=args.text_seconds,
        warmup_seconds=min(10, args.text_seconds),
        concurrency="4,8,16,32",
        batch_sizes="1,8,16,32",
        sample_limit=args.sample_limit,
        run_id=args.run_id,
        stage_name="text_embedding",
        api_url=settings.EMBEDDING_API_URL,
        model_name=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY,
        timeout=settings.EMBEDDING_TIMEOUT,
        max_retries=settings.EMBEDDING_MAX_RETRIES,
        print_report=False,
    )
    text_result = await run_text_embedding_benchmark(text_args)
    status["phases"].append({"stage": "text_embedding", "report_path": text_result["report_path"]})
    write_json(status_path, status)

    milvus_probe = await _run_milvus_write_probe(
        samples=samples,
        duration_seconds=args.write_seconds,
        collection_name=args.image_collection,
    )
    write_json(run_dir / "milvus_write_probe.json", milvus_probe)
    status["phases"].append({"stage": "milvus_write_probe", "report_path": str(run_dir / "milvus_write_probe.json")})
    write_json(status_path, status)

    stage_reports = [
        download_result["report"],
        image_model_result["report"],
        text_result["report"],
        milvus_probe,
    ]
    comparison = compare_stage_bottlenecks(stage_reports)
    report = {
        "started_at": status["started_at"],
        "finished_at": time.time(),
        "upload_dir": str(args.upload_dir),
        "sample_count": len(samples),
        "stages": stage_reports,
        "comparison": comparison,
        "recommendations": _recommendations(comparison),
        "artifact_paths": {
            "status_path": str(status_path),
            "report_path": str(report_path),
        },
    }
    write_json(report_path, report)
    return {"run_dir": str(run_dir), "report_path": str(report_path), "report": report}


def _recommendations(comparison: dict[str, object]) -> list[str]:
    stage = str(comparison.get("bottleneck_stage") or "")
    recommendations = []
    if stage == "image_download":
        recommendations.append("优先增加下载链路连接池、DNS 复用和本地缓存盘写入能力。")
        recommendations.append("如果静态资源站存在限流，优先和服务端确认 429/超时阈值。")
    elif stage == "image_model":
        recommendations.append("图片向量化是当前最低点，优先扩图片 embedding 并发、模型实例数，或降低输入分辨率上限。")
        recommendations.append("优先复用预处理后的 JPEG data URL，减少重复转码。")
    elif stage == "text_embedding":
        recommendations.append("文本 embedding 已成为最低点，优先调大 batch size 并检查 embedding 服务限流。")
    elif stage == "milvus_write_probe":
        recommendations.append("Milvus 写入已成为最低点，优先调大批量写入 batch 和 flush 间隔，并检查集合索引负载。")
    else:
        recommendations.append("未识别出明确瓶颈阶段，需要复核各阶段报告中的错误率和 p95 延迟。")
    return recommendations


def main(argv: Sequence[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    result = asyncio.run(run_sync_path_benchmark(args))
    if args.print_report:
        print(result["report"])
    else:
        print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
