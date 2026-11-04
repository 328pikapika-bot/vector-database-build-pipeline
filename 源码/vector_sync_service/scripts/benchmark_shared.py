"""Shared helpers for sync-path benchmark scripts."""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.models.image import ImageRecord
from app.services.image_context_extractor import extract_mmd_image_contexts, find_mmd_context_for_image
from app.services.image_features import (
    IMAGE_PIXEL_LIMIT_EXCEEDED,
    build_image_record_draft_from_manifest,
    compute_phash,
    validate_image_bytes_dimensions,
    validate_image_manifest,
)


DEFAULT_UPLOAD_DIR = SERVICE_ROOT / "upload"
DEFAULT_OUT_DIR = SERVICE_ROOT / "out"
DEFAULT_BENCHMARK_PREFIX = "codex_sync_path_benchmark"


@dataclass(slots=True)
class UploadImageSample:
    doc_id: str
    task_id: str
    image_relative_url: str
    image_url: str
    image_path: Path
    manifest_path: Path
    mmd_relative_url: str = ""
    mmd_path: Optional[Path] = None
    image_name: str = ""
    content_type: str = ""
    size_bytes: int = 0
    upload_state: str = ""
    bytes_sha256: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineInput:
    sample: UploadImageSample
    sequence: int
    image_bytes: bytes
    draft_record: ImageRecord
    context_text: str
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class BenchmarkCaseResult:
    case_name: str
    stage: str
    concurrency: int
    batch_size: int = 1
    duration_seconds: float = 0.0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    bytes_processed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    error_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        duration = max(float(self.duration_seconds), 1e-9)
        throughput = self.successes / duration
        mbps = (self.bytes_processed / duration) / (1024 * 1024)
        return {
            "case_name": self.case_name,
            "stage": self.stage,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "duration_seconds": round(self.duration_seconds, 3),
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.successes / max(self.attempts, 1), 4),
            "throughput_per_second": round(throughput, 4),
            "throughput_per_minute": round(throughput * 60.0, 2),
            "bytes_processed": self.bytes_processed,
            "mb_per_second": round(mbps, 4),
            "latency_ms": latency_summary(self.latencies_ms),
            "error_counts": dict(sorted(self.error_counts.items())),
            "notes": list(self.notes),
            "extras": json_safe(self.extras),
        }


def force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in latencies_ms if value is not None]
    if not values:
        return {
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "avg": round(statistics.fmean(sorted_values), 3),
        "min": round(sorted_values[0], 3),
        "max": round(sorted_values[-1], 3),
        "p50": round(percentile(sorted_values, 50), 3),
        "p95": round(percentile(sorted_values, 95), 3),
        "p99": round(percentile(sorted_values, 99), 3),
    }


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * max(min(pct, 100.0), 0.0) / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def best_case(results: Sequence[BenchmarkCaseResult]) -> Optional[BenchmarkCaseResult]:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            item.successes / max(item.duration_seconds, 1e-9),
            -item.failures,
            -latency_summary(item.latencies_ms)["p95"],
        ),
    )


def should_stop_escalation(
    previous: Optional[BenchmarkCaseResult],
    current: BenchmarkCaseResult,
    *,
    min_gain_ratio: float = 0.05,
    failure_rate_threshold: float = 0.05,
) -> bool:
    if previous is None:
        return False
    prev_tps = previous.successes / max(previous.duration_seconds, 1e-9)
    curr_tps = current.successes / max(current.duration_seconds, 1e-9)
    if prev_tps <= 0:
        return False
    gain_ratio = (curr_tps - prev_tps) / prev_tps
    failure_rate = current.failures / max(current.attempts, 1)
    return gain_ratio < min_gain_ratio or failure_rate > failure_rate_threshold


def scan_upload_samples(upload_dir: Path = DEFAULT_UPLOAD_DIR) -> list[UploadImageSample]:
    manifest_paths = sorted(upload_dir.rglob("manifest.json"))
    samples: dict[tuple[str, str], UploadImageSample] = {}
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample = _sample_from_manifest(manifest_path, manifest)
        if sample is None:
            continue
        key = (sample.doc_id, sample.image_relative_url)
        existing = samples.get(key)
        if existing is None or _prefer_sample(sample, existing):
            samples[key] = sample
    return list(samples.values())


def _sample_from_manifest(manifest_path: Path, manifest: dict[str, Any]) -> Optional[UploadImageSample]:
    if not isinstance(manifest, dict):
        return None
    image_relative_url = str(manifest.get("image_relative_url") or "").strip()
    downloads = manifest.get("downloads") or {}
    image_download = downloads.get("image") or {}
    if not image_relative_url:
        image_relative_url = str(image_download.get("relative_url") or "").strip()
    if not image_relative_url:
        return None
    image_local_path = Path(str(image_download.get("local_path") or "")).expanduser()
    if not image_local_path.exists():
        image_local_path = manifest_path.parent / Path(image_relative_url).name
    if not image_local_path.exists():
        candidates = sorted(manifest_path.parent.glob("*.jpg")) + sorted(manifest_path.parent.glob("*.png"))
        if not candidates:
            return None
        image_local_path = candidates[0]
    mmd_relative_url = str(manifest.get("mmd_relative_url") or "").strip()
    mmd_download = downloads.get("mmd") or {}
    mmd_local_raw = str(mmd_download.get("local_path") or "").strip()
    mmd_path = Path(mmd_local_raw).expanduser() if mmd_local_raw else None
    if mmd_path is not None and not mmd_path.exists():
        fallback = manifest_path.parent / Path(mmd_relative_url).name
        mmd_path = fallback if fallback.exists() else None
    content_type = str(image_download.get("content_type") or "").strip()
    return UploadImageSample(
        doc_id=str(manifest.get("doc_id") or "").strip(),
        task_id=str(manifest.get("task_id") or manifest.get("batch_id") or "").strip(),
        image_relative_url=image_relative_url,
        image_url=str(image_download.get("url") or "").strip(),
        image_path=image_local_path,
        manifest_path=manifest_path,
        mmd_relative_url=mmd_relative_url,
        mmd_path=mmd_path if mmd_path is not None and mmd_path.exists() else None,
        image_name=image_local_path.name,
        content_type=content_type,
        size_bytes=int(image_download.get("size_bytes") or image_local_path.stat().st_size),
        upload_state=_upload_state_from_path(manifest_path),
        bytes_sha256=str(image_download.get("sha256") or "").strip(),
        manifest=manifest,
    )


def _upload_state_from_path(manifest_path: Path) -> str:
    for name in manifest_path.parts:
        if name in {"attachments", "failed", "_cleanup_backups"} or name.startswith("codex_"):
            return name
    return manifest_path.parent.name


def _prefer_sample(candidate: UploadImageSample, current: UploadImageSample) -> bool:
    ranking = {
        "attachments": 3,
        "codex_kafka_loadtest_20260616T1700Z": 2,
        "codex_kafka_backlog_loadtest_20260616T100613Z": 2,
        "failed": 1,
        "_cleanup_backups": 0,
    }
    return ranking.get(candidate.upload_state, 0) > ranking.get(current.upload_state, 0)


def cycle_samples(samples: Sequence[UploadImageSample]) -> Callable[[int], UploadImageSample]:
    if not samples:
        raise ValueError("samples are required")

    def selector(index: int) -> UploadImageSample:
        return samples[index % len(samples)]

    return selector


def build_pipeline_input(sample: UploadImageSample, *, sequence: int) -> PipelineInput:
    image_bytes = sample.image_path.read_bytes()
    dimensions = validate_image_bytes_dimensions(image_bytes, max_pixels=settings.IMAGE_MAX_PIXELS)
    oversize_for_model = (
        not dimensions.success and dimensions.error_code == IMAGE_PIXEL_LIMIT_EXCEEDED
    )
    if not dimensions.success and not oversize_for_model:
        raise ValueError(f"invalid image bytes: {dimensions.error_code}: {dimensions.error_message}")

    raw_img_meta = _dict_or_empty(sample.manifest.get("raw_img_meta"))
    original_image_id = str(
        sample.manifest.get("image_id")
        or sample.manifest.get("imageId")
        or raw_img_meta.get("id")
        or ""
    ).strip()
    if not original_image_id:
        raise ValueError(f"sample has no original image id: {sample.manifest_path}")
    raw_img_meta["id"] = original_image_id
    manifest = {
        "doc_id": sample.doc_id,
        "image_id": original_image_id,
        "raw_img_meta": raw_img_meta,
        "relative_url": sample.image_relative_url,
        "content_type": sample.content_type or _content_type_from_name(sample.image_name),
        "size_bytes": sample.size_bytes or len(image_bytes),
        "bytes_sha256": sample.bytes_sha256 or sha256(image_bytes).hexdigest(),
        "sequence": sequence,
        "name": sample.image_name,
        "file_type": _file_type_from_name(sample.image_name),
        "title": _manifest_title(sample.manifest),
    }
    manifest_result = validate_image_manifest(manifest, image_bytes=image_bytes)
    if not manifest_result.success or manifest_result.manifest is None:
        raise ValueError(f"invalid manifest for sample {sample.manifest_path}")

    context_before, context_after, context_text, caption = _extract_context(sample)
    draft_record = build_image_record_draft_from_manifest(
        manifest_result,
        dimension_result=dimensions,
        feature_metadata={
            "name": sample.image_name,
            "image_title": caption,
            "doc_meta": _doc_meta_from_manifest(sample),
            "raw_img_meta": raw_img_meta,
            "context_before": context_before,
            "context_after": context_after,
            "img_meta": {
                "source_task_id": sample.task_id,
                "mmd_context_located": bool(context_text),
                "oversize_for_model": oversize_for_model,
                "original_pixel_count": dimensions.pixel_count,
            },
        },
    )
    diagnostics = {
        "oversize_for_model": oversize_for_model,
        "original_pixel_count": dimensions.pixel_count,
        "width": dimensions.width,
        "height": dimensions.height,
        "image_format": dimensions.image_format,
        "caption_chars": len(caption),
        "context_chars": len(context_text),
    }
    return PipelineInput(
        sample=sample,
        sequence=sequence,
        image_bytes=image_bytes,
        draft_record=draft_record,
        context_text=context_text,
        diagnostics=diagnostics,
    )


def _extract_context(sample: UploadImageSample) -> tuple[str, str, str, str]:
    context = sample.manifest.get("context") or {}
    caption = str(context.get("caption") or context.get("caption_text") or "").strip()
    if sample.mmd_path is None or not sample.mmd_path.exists():
        return "", caption or _default_caption(), caption, caption or _default_caption()
    try:
        mmd_text = sample.mmd_path.read_text(encoding="utf-8")
        refs = extract_mmd_image_contexts(mmd_text)
        located = find_mmd_context_for_image(
            refs,
            image_name=sample.image_name,
            relative_url=sample.image_relative_url,
        )
    except Exception:
        return "", caption or _default_caption(), caption, caption or _default_caption()
    if located is None:
        return "", caption or _default_caption(), caption, caption or _default_caption()
    resolved_caption = caption or str(located.context_after or "").split("\n\n", 1)[0].strip() or _default_caption()
    before = str(located.context_before or "").strip()
    context_after = str(located.context_after or resolved_caption or _default_caption()).strip()
    joined = str(located.context_for_model or "").strip()
    if not joined:
        joined = "\n\n".join(part for part in (before, context_after) if part)
    return before, context_after, joined, resolved_caption


def _manifest_title(manifest: dict[str, Any]) -> str:
    for key in ("title", "image_title", "caption"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _doc_meta_from_manifest(sample: UploadImageSample) -> dict[str, Any]:
    manifest = sample.manifest
    doc_meta = _dict_or_empty(manifest.get("doc_meta"))
    doc_meta.setdefault("docId", sample.doc_id)
    doc_meta.setdefault("language", ["unknown"])
    doc_meta.setdefault("sysDataStatus", "ACTIVE")
    title = manifest.get("title")
    if title:
        doc_meta.setdefault("title", title)
    keyword = manifest.get("keyword")
    if keyword:
        doc_meta.setdefault("keyword", keyword)
    return doc_meta


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _file_type_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    return suffix or "jpg"


def _content_type_from_name(name: str) -> str:
    suffix = _file_type_from_name(name)
    if suffix in {"jpg", "jpeg"}:
        return "image/jpeg"
    if suffix == "png":
        return "image/png"
    return "application/octet-stream"


def _default_caption() -> str:
    return str(settings.IMAGE_DEFAULT_CAPTION or "No caption").strip() or "No caption"


def default_run_dir(prefix: str, *, run_id: Optional[str] = None) -> Path:
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(DEFAULT_OUT_DIR / f"{prefix}_{resolved_run_id}")


def summarize_stage(
    *,
    stage: str,
    cases: Sequence[BenchmarkCaseResult],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    best = best_case(cases)
    return {
        "stage": stage,
        "cases": [case.to_summary() for case in cases],
        "best_case": None if best is None else best.to_summary(),
        "extra": json_safe(extra or {}),
    }


def compare_stage_bottlenecks(stage_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[tuple[str, float, dict[str, Any]]] = []
    for report in stage_reports:
        best = report.get("best_case") or {}
        throughput = float(best.get("throughput_per_second") or 0.0)
        candidates.append((str(report.get("stage") or ""), throughput, best))
    if not candidates:
        return {"bottleneck_stage": "", "ordered": []}
    ordered = sorted(candidates, key=lambda item: item[1])
    bottleneck_stage, bottleneck_tps, bottleneck_case = ordered[0]
    return {
        "bottleneck_stage": bottleneck_stage,
        "bottleneck_throughput_per_second": round(bottleneck_tps, 4),
        "bottleneck_case": bottleneck_case,
        "ordered": [
            {
                "stage": stage,
                "throughput_per_second": round(throughput, 4),
                "best_case": case,
            }
            for stage, throughput, case in ordered
        ],
    }


def format_error_counts(errors: Iterable[str]) -> dict[str, int]:
    counter = Counter(str(item or "UNKNOWN") for item in errors)
    return dict(sorted(counter.items()))


def allocate_case_durations(total_seconds: int, case_count: int) -> list[int]:
    total = max(int(total_seconds or 0), 1)
    count = max(int(case_count or 0), 1)
    base = max(total // count, 1)
    durations = [base] * count
    remainder = max(total - base * count, 0)
    for index in range(remainder):
        durations[index % count] += 1
    return durations


class Deadline:
    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = max(float(duration_seconds), 0.0)
        self.started = time.perf_counter()

    @property
    def remaining(self) -> float:
        return max(self.duration_seconds - (time.perf_counter() - self.started), 0.0)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0
