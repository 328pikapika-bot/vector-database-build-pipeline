#!/usr/bin/env python3
"""Publish paged batches to the upstream Kafka publishing endpoint.

The endpoint is side-effecting: one successful page publishes a batch into
Kafka. To avoid duplicate pages, this script records successful pages in a
checkpoint file and does not retry ambiguous timeouts by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.services.auto_publisher import _decode_body_excerpt, _ensure_successful_body


DEFAULT_URL = "http://127.0.0.1:17123/api/service/interfaces/vector/job/publishBatch"
DEFAULT_SUB_DOC_TYPE = "A79"
DEFAULT_PAGE_PARAM = "pageNumer"
DEFAULT_START_PAGE = 10
DEFAULT_PAGE_COUNT = 50
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_SLOW_THRESHOLD_SECONDS = 30.0
RUNTIME_DIR = Path(__file__).resolve().parents[1] / ".runtime"
DEFAULT_CHECKPOINT = RUNTIME_DIR / "publish_batch_pages_checkpoint.json"
DEFAULT_LOG = RUNTIME_DIR / "publish_batch_pages.log"
DEFAULT_AUDIT = RUNTIME_DIR / "publish_batch_pages_audit.jsonl"
DEFAULT_SUMMARY = RUNTIME_DIR / "publish_batch_pages_summary.json"


@dataclass(frozen=True)
class PublishResult:
    page: int
    status_code: int
    body_excerpt: str
    elapsed_seconds: float
    requested_topic: str | None
    response_page: int | None
    current_page_size: int | None
    success_count: int | None
    fail_count: int | None
    total_count: int | None
    response_code: str | None
    response_success: bool | None
    response_message: str | None
    slow: bool


class AmbiguousPublishError(RuntimeError):
    """Raised when the client cannot know whether the server published a page."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish consecutive pages to the Kafka batch publishing endpoint. "
            "Defaults publish pages 10-59, 50 pages total."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="publishBatch endpoint URL.")
    parser.add_argument("--sub-doc-type", default=DEFAULT_SUB_DOC_TYPE, help="subDocType value.")
    parser.add_argument("--page-param", default=DEFAULT_PAGE_PARAM, help="Page parameter name.")
    parser.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE, help="First page number.")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGE_COUNT, help="How many pages to publish.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-page request timeout. Default is 900 seconds.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Delay after each successful page.",
    )
    parser.add_argument(
        "--slow-threshold-seconds",
        type=float,
        default=DEFAULT_SLOW_THRESHOLD_SECONDS,
        help="Warn and mark a page slow when elapsed time reaches this threshold.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint JSON path for successful pages.",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG, help="Log file path.")
    parser.add_argument("--audit-file", type=Path, default=DEFAULT_AUDIT, help="Append-only NDJSON audit log path.")
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY, help="Per-run summary JSON path.")
    parser.add_argument("--topic", default=None, help="Optional topic form field passed upstream.")
    parser.add_argument("--token", default=None, help="Bearer token. Prefer env PUBLISH_BATCH_TOKEN.")
    parser.add_argument(
        "--authorization",
        default=None,
        help="Full Authorization header. Prefer env PUBLISH_BATCH_AUTHORIZATION.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore successful pages in the checkpoint and attempt all requested pages.",
    )
    parser.add_argument(
        "--retry-unknown",
        action="store_true",
        help="Retry pages that previously timed out or had an unknown result.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pages and form parameters without calling the endpoint.",
    )
    parser.add_argument(
        "--continue-after-timeout",
        action="store_true",
        help=(
            "Record timeout pages as unknown and continue with later pages. "
            "Use only when skipping an uncertain page is acceptable."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pages < 1:
        parser.error("--pages must be at least 1")
    if args.start_page < 1:
        parser.error("--start-page must be at least 1")

    setup_logging(args.log_file)
    auth_header = build_authorization_header(args)
    checkpoint = load_checkpoint(args.checkpoint)
    requested_pages = list(range(args.start_page, args.start_page + args.pages))
    run_started_monotonic = time.monotonic()
    summary = build_run_summary(args, requested_pages)
    completed_pages = {
        int(page)
        for page, record in checkpoint.get("completed_pages", {}).items()
        if isinstance(record, dict)
    }
    unknown_pages = {
        int(page)
        for page, record in checkpoint.get("unknown_pages", {}).items()
        if isinstance(record, dict)
    }

    logging.info(
        "Starting publish run: url=%s subDocType=%s pageParam=%s pages=%s-%s count=%s timeout=%ss dryRun=%s topic=%s",
        args.url,
        args.sub_doc_type,
        args.page_param,
        requested_pages[0],
        requested_pages[-1],
        len(requested_pages),
        args.timeout_seconds,
        args.dry_run,
        args.topic or "-",
    )

    try:
        if args.dry_run:
            for page in requested_pages:
                form = build_form(args.sub_doc_type, args.page_param, page, args.topic)
                logging.info("DRY RUN page=%s form=%s", page, form)
                record = {
                    "timestamp": utc_now(),
                    "event": "dry_run",
                    "page": page,
                    "sub_doc_type": args.sub_doc_type,
                    "requested_topic": args.topic,
                    "form": form,
                }
                summary["pages"].append(record)
                write_audit_record(args.audit_file, record)
            return 0

        for page in requested_pages:
            if not args.no_resume and page in completed_pages:
                logging.info("Skipping page=%s because checkpoint already marks it successful.", page)
                summary["counts"]["skipped_completed_pages"] += 1
                record = {
                    "timestamp": utc_now(),
                    "event": "skip_completed",
                    "page": page,
                    "sub_doc_type": args.sub_doc_type,
                    "requested_topic": args.topic,
                }
                summary["pages"].append(record)
                write_audit_record(args.audit_file, record)
                continue

            if not args.no_resume and not args.retry_unknown and page in unknown_pages:
                logging.info("Skipping page=%s because checkpoint marks its result unknown.", page)
                summary["counts"]["skipped_unknown_pages"] += 1
                record = {
                    "timestamp": utc_now(),
                    "event": "skip_unknown",
                    "page": page,
                    "sub_doc_type": args.sub_doc_type,
                    "requested_topic": args.topic,
                }
                summary["pages"].append(record)
                write_audit_record(args.audit_file, record)
                continue

            form = build_form(args.sub_doc_type, args.page_param, page, args.topic)
            logging.info("Publishing page=%s form=%s", page, form)
            summary["counts"]["attempted_pages"] += 1
            try:
                result = publish_page(
                    url=args.url,
                    authorization=auth_header,
                    form=form,
                    page=page,
                    timeout_seconds=args.timeout_seconds,
                    slow_threshold_seconds=args.slow_threshold_seconds,
                    requested_topic=args.topic,
                )
            except AmbiguousPublishError as exc:
                logging.error("Ambiguous timeout for page=%s: %s", page, exc)
                record_unknown_page(args.checkpoint, checkpoint, page, args.sub_doc_type, str(exc))
                summary["counts"]["unknown_pages"] += 1
                record = {
                    "timestamp": utc_now(),
                    "event": "page_unknown",
                    "page": page,
                    "sub_doc_type": args.sub_doc_type,
                    "requested_topic": args.topic,
                    "reason": str(exc),
                }
                summary["pages"].append(record)
                write_audit_record(args.audit_file, record)
                if args.continue_after_timeout:
                    continue
                logging.error(
                    "Stopped to avoid duplicate publishing. Increase --timeout-seconds or inspect Kafka/checkpoint."
                )
                return 2
            except Exception as exc:  # noqa: BLE001 - top-level CLI reporting
                logging.exception("Failed publishing page=%s: %s", page, exc)
                summary["counts"]["failed_pages"] += 1
                record = {
                    "timestamp": utc_now(),
                    "event": "page_failed",
                    "page": page,
                    "sub_doc_type": args.sub_doc_type,
                    "requested_topic": args.topic,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                summary["pages"].append(record)
                write_audit_record(args.audit_file, record)
                return 1

            record_success(args.checkpoint, checkpoint, result, args.sub_doc_type)
            update_summary_counts(summary, result)
            event_name = "page_slow" if result.slow else "page_ok"
            page_record = build_page_record(event_name, result, args.sub_doc_type)
            summary["pages"].append(page_record)
            write_audit_record(args.audit_file, page_record)

            log_fn = logging.warning if result.slow else logging.info
            log_fn(
                "Published page=%s status=%s elapsed=%.1fs topic=%s pageSize=%s success=%s fail=%s total=%s responsePage=%s body=%s",
                result.page,
                result.status_code,
                result.elapsed_seconds,
                result.requested_topic or "-",
                result.current_page_size,
                result.success_count,
                result.fail_count,
                result.total_count,
                result.response_page,
                result.body_excerpt,
            )

            if args.interval_seconds > 0 and page != requested_pages[-1]:
                time.sleep(args.interval_seconds)

        logging.info("Publish run finished.")
        return 0
    finally:
        finalize_summary(args.summary_file, summary, run_started_monotonic)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def build_authorization_header(args: argparse.Namespace) -> str:
    full_header = args.authorization or os.getenv("PUBLISH_BATCH_AUTHORIZATION")
    if full_header:
        return full_header
    token = args.token or os.getenv("PUBLISH_BATCH_TOKEN")
    if not token:
        raise SystemExit(
            "Missing bearer token. Set env PUBLISH_BATCH_TOKEN or pass --token/--authorization."
        )
    return f"Bearer {token}"


def build_form(sub_doc_type: str, page_param: str, page: int, topic: str | None) -> dict[str, str]:
    form = {"subDocType": sub_doc_type, page_param: str(page)}
    if topic:
        form["topic"] = topic
    return form


def publish_page(
    *,
    url: str,
    authorization: str,
    form: dict[str, str],
    page: int,
    timeout_seconds: int,
    slow_threshold_seconds: float,
    requested_topic: str | None,
) -> PublishResult:
    payload = urlencode(form).encode("utf-8")
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "vector-sync-publish-batch/1.0",
        },
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            status_code = int(response.status)
    except (TimeoutError, socket.timeout) as exc:
        raise AmbiguousPublishError(f"request timed out after {timeout_seconds}s") from exc
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise AmbiguousPublishError(f"request timed out after {timeout_seconds}s") from exc
        raise
    except HTTPError as exc:
        body = exc.read()
        excerpt = _decode_body_excerpt(body)
        raise RuntimeError(f"HTTP {exc.code}: {excerpt}") from exc

    elapsed = time.monotonic() - started
    excerpt = _decode_body_excerpt(body)
    _ensure_successful_body(body, excerpt)
    metrics = parse_response_metrics(body)
    return PublishResult(
        page=page,
        status_code=status_code,
        body_excerpt=excerpt,
        elapsed_seconds=elapsed,
        requested_topic=requested_topic,
        response_page=metrics.get("response_page"),
        current_page_size=metrics.get("current_page_size"),
        success_count=metrics.get("success_count"),
        fail_count=metrics.get("fail_count"),
        total_count=metrics.get("total_count"),
        response_code=metrics.get("response_code"),
        response_success=metrics.get("response_success"),
        response_message=metrics.get("response_message"),
        slow=elapsed >= slow_threshold_seconds,
    )


def parse_response_metrics(body: bytes) -> dict[str, Any]:
    parsed = load_response_json(body)
    if not isinstance(parsed, dict):
        return {}

    payload = parsed
    if isinstance(parsed.get("data"), dict):
        payload = parsed["data"]

    return {
        "response_page": coerce_int(first_present(payload, "currentPage", "pageNumer", "pageNumber", "page")),
        "current_page_size": coerce_int(first_present(payload, "currentPageSize", "pageSize", "size")),
        "success_count": coerce_int(first_present(payload, "successCount")),
        "fail_count": coerce_int(first_present(payload, "failCount", "failureCount", "errorCount")),
        "total_count": coerce_int(first_present(payload, "totalCount", "count")),
        "response_code": coerce_str(first_present(parsed, "code", "status")),
        "response_success": coerce_bool(first_present(parsed, "success")),
        "response_message": coerce_str(first_present(parsed, "message", "msg")),
    }


def load_response_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def first_present(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def build_run_summary(args: argparse.Namespace, requested_pages: list[int]) -> dict[str, Any]:
    return {
        "run_started_at": utc_now(),
        "run_finished_at": None,
        "run_elapsed_seconds": None,
        "hostname": socket.gethostname(),
        "url": args.url,
        "sub_doc_type": args.sub_doc_type,
        "page_param": args.page_param,
        "start_page": args.start_page,
        "pages_requested": args.pages,
        "requested_page_range": [requested_pages[0], requested_pages[-1]],
        "timeout_seconds": args.timeout_seconds,
        "interval_seconds": args.interval_seconds,
        "slow_threshold_seconds": args.slow_threshold_seconds,
        "requested_topic": args.topic,
        "dry_run": args.dry_run,
        "checkpoint": str(args.checkpoint),
        "log_file": str(args.log_file),
        "audit_file": str(args.audit_file),
        "summary_file": str(args.summary_file),
        "counts": {
            "requested_pages": len(requested_pages),
            "attempted_pages": 0,
            "successful_pages": 0,
            "unknown_pages": 0,
            "failed_pages": 0,
            "skipped_completed_pages": 0,
            "skipped_unknown_pages": 0,
            "slow_pages": 0,
            "reported_success_records": 0,
            "reported_fail_records": 0,
            "reported_total_records": 0,
            "reported_page_sizes": 0,
        },
        "pages": [],
    }


def update_summary_counts(summary: dict[str, Any], result: PublishResult) -> None:
    counts = summary["counts"]
    counts["successful_pages"] += 1
    if result.slow:
        counts["slow_pages"] += 1
    if result.success_count is not None:
        counts["reported_success_records"] += result.success_count
    if result.fail_count is not None:
        counts["reported_fail_records"] += result.fail_count
    if result.total_count is not None:
        counts["reported_total_records"] += result.total_count
    if result.current_page_size is not None:
        counts["reported_page_sizes"] += result.current_page_size


def build_page_record(event: str, result: PublishResult, sub_doc_type: str) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "event": event,
        "page": result.page,
        "sub_doc_type": sub_doc_type,
        "requested_topic": result.requested_topic,
        "status_code": result.status_code,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "slow": result.slow,
        "response_page": result.response_page,
        "current_page_size": result.current_page_size,
        "success_count": result.success_count,
        "fail_count": result.fail_count,
        "total_count": result.total_count,
        "response_code": result.response_code,
        "response_success": result.response_success,
        "response_message": result.response_message,
        "body_excerpt": result.body_excerpt,
    }


def write_audit_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, sort_keys=True)
        fh.write("\n")


def finalize_summary(path: Path, summary: dict[str, Any], run_started_monotonic: float) -> None:
    summary["run_finished_at"] = utc_now()
    summary["run_elapsed_seconds"] = round(time.monotonic() - run_started_monotonic, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_pages": {}, "unknown_pages": {}}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid checkpoint JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid checkpoint shape: {path}")
    data.setdefault("completed_pages", {})
    data.setdefault("unknown_pages", {})
    return data


def record_success(path: Path, checkpoint: dict[str, Any], result: PublishResult, sub_doc_type: str) -> None:
    checkpoint.setdefault("completed_pages", {})[str(result.page)] = {
        "subDocType": sub_doc_type,
        "requested_topic": result.requested_topic,
        "status_code": result.status_code,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "response_page": result.response_page,
        "current_page_size": result.current_page_size,
        "success_count": result.success_count,
        "fail_count": result.fail_count,
        "total_count": result.total_count,
        "response_code": result.response_code,
        "response_success": result.response_success,
        "response_message": result.response_message,
        "slow": result.slow,
        "body_excerpt": result.body_excerpt,
        "finished_at": utc_now(),
    }
    checkpoint.setdefault("unknown_pages", {}).pop(str(result.page), None)
    save_checkpoint(path, checkpoint)


def record_unknown_page(
    path: Path,
    checkpoint: dict[str, Any],
    page: int,
    sub_doc_type: str,
    reason: str,
) -> None:
    checkpoint.setdefault("unknown_pages", {})[str(page)] = {
        "subDocType": sub_doc_type,
        "reason": reason,
        "recorded_at": utc_now(),
    }
    save_checkpoint(path, checkpoint)


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = utc_now()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
