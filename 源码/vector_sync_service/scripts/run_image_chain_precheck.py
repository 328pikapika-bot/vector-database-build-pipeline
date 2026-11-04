#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import httpx
import sys
from pymongo import MongoClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "vector_sync_service"
VENV_PYTHON = SERVICE_ROOT / ".venv" / "Scripts" / "python.exe"
OUT_DIR = SERVICE_ROOT / "out"

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings

READ_TOKEN = os.environ.get("ADMIN_READ_TOKEN") or secrets.token_urlsafe(24)
WRITE_TOKEN = os.environ.get("ADMIN_WRITE_TOKEN") or secrets.token_urlsafe(24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart local image worker with load-test settings and run a 5-minute precheck."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--sample-interval-seconds", type=int, default=30)
    parser.add_argument("--settle-timeout-seconds", type=int, default=240)
    parser.add_argument("--seed-count", type=int, default=3000)
    parser.add_argument("--seed-limit", type=int, default=200)
    parser.add_argument("--doc-prefix", default="loadtest_img")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--vl-embed-concurrency", type=int, default=4)
    parser.add_argument("--text-embed-concurrency", type=int, default=16)
    parser.add_argument("--embedding-max-concurrency", type=int, default=20)
    parser.add_argument("--image-worker-concurrency", type=int, default=24)
    parser.add_argument("--image-download-concurrency", type=int, default=24)
    parser.add_argument("--image-worker-batch-size", type=int, default=64)
    parser.add_argument("--image-write-concurrency", type=int, default=8)
    parser.add_argument("--image-write-batch-size", type=int, default=48)
    parser.add_argument("--image-write-flush-interval-ms", type=int, default=1000)
    parser.add_argument(
        "--image-write-mode",
        default="insert_or_skip_if_exists",
        choices=[
            "replace",
            "insert_only",
            "insert_or_skip_if_exists",
            "upsert_if_exists",
        ],
    )
    parser.add_argument("--health-timeout-seconds", type=int, default=180)
    parser.add_argument("--startup-timeout-seconds", type=int, default=180)
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--keep-running", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_prereqs()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id.strip() or f"precheck_vl{args.vl_embed_concurrency}_{timestamp}"
    base_name = f"image_chain_precheck_vl{args.vl_embed_concurrency}_{timestamp}"

    jsonl_path = OUT_DIR / f"{base_name}.jsonl"
    stdout_path = OUT_DIR / f"{base_name}.stdout.log"
    stderr_path = OUT_DIR / f"{base_name}.stderr.log"
    summary_path = OUT_DIR / f"{base_name}_summary.json"
    seed_report_path = OUT_DIR / f"{base_name}_seed.json"
    startup_health_path = OUT_DIR / f"{base_name}_startup_health.json"
    startup_status_path = OUT_DIR / f"{base_name}_startup_image_status.json"
    final_health_path = OUT_DIR / f"{base_name}_final_health.json"
    final_status_path = OUT_DIR / f"{base_name}_final_image_status.json"
    last10_path = OUT_DIR / f"{base_name}_last10.json"

    service_process: subprocess.Popen[str] | None = None
    started_at = iso_now()
    baseline_health = try_fetch_json(f"http://127.0.0.1:{args.port}/health")
    baseline_status = try_fetch_json(
        f"http://127.0.0.1:{args.port}/admin/workers/image/status",
        headers=auth_headers(READ_TOKEN),
    )
    baseline_counts = image_counts_from_status(baseline_status)
    if sum(baseline_counts.values()) == 0:
        baseline_counts = query_image_task_counts()

    try:
        if not args.skip_seed:
            seed_tasks(
                count=args.seed_count,
                seed_limit=args.seed_limit,
                run_id=run_id,
                doc_prefix=args.doc_prefix,
                report_path=seed_report_path,
            )

        stop_listener_on_port(args.port)
        env = build_env(args)
        service_process = start_service(
            port=args.port,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        startup_health = wait_for_service(args.port, args.startup_timeout_seconds)
        startup_status = fetch_json(
            f"http://127.0.0.1:{args.port}/admin/workers/image/status",
            headers=auth_headers(READ_TOKEN),
        )
        write_json(startup_health_path, startup_health)
        write_json(startup_status_path, startup_status)
        ensure_startup_ready(startup_health, startup_status)

        resume_status = post_json(
            f"http://127.0.0.1:{args.port}/admin/workers/image/resume",
            headers=auth_headers(WRITE_TOKEN),
        )
        if bool(((resume_status.get("control") or {}).get("paused"))):
            raise RuntimeError("image worker resume did not clear paused state")

        samples: list[dict[str, Any]] = []
        deadline = time.monotonic() + max(int(args.duration_seconds or 0), 0)
        sample_index = 0
        while True:
            now = time.monotonic()
            sample = collect_sample(args.port, sample_index)
            samples.append(sample)
            append_jsonl(jsonl_path, sample)
            sample_index += 1
            if now >= deadline:
                break
            sleep_seconds = min(max(int(args.sample_interval_seconds or 1), 1), max(deadline - now, 0.0))
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)

        drained_status = post_json(
            f"http://127.0.0.1:{args.port}/admin/workers/image/drain",
            headers=auth_headers(WRITE_TOKEN),
            timeout_seconds=max(int(args.settle_timeout_seconds or 0) + 30, 60),
        )
        wait_for_drain(
            port=args.port,
            timeout_seconds=args.settle_timeout_seconds,
        )

        final_health = fetch_json(f"http://127.0.0.1:{args.port}/health")
        final_status = fetch_json(
            f"http://127.0.0.1:{args.port}/admin/workers/image/status",
            headers=auth_headers(READ_TOKEN),
        )
        write_json(final_health_path, final_health)
        write_json(final_status_path, final_status)
        write_json(last10_path, {"samples": samples[-10:]})

        summary = build_summary(
            args=args,
            run_id=run_id,
            started_at=started_at,
            finished_at=iso_now(),
            baseline_counts=baseline_counts,
            startup_health=startup_health,
            startup_status=startup_status,
            final_health=final_health,
            final_status=final_status,
            drained_status=drained_status,
            samples=samples,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            jsonl_path=jsonl_path,
            seed_report_path=seed_report_path if not args.skip_seed else None,
        )
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary.get("passed") else 2
    finally:
        if service_process is not None and not args.keep_running:
            stop_process(service_process)


def ensure_prereqs() -> None:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"python not found: {VENV_PYTHON}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "KAFKA_CONSUMER_ENABLED": "false",
            "FULLTEXT_WORKER_ENABLED": "false",
            "IMAGE_WORKER_ENABLED": "true",
            "IMAGE_FEATURE_PIPELINE_ENABLED": "true",
            "IMAGE_TASK_UPSERT_ENABLED": "true",
            "EMBEDDING_MAX_CONCURRENCY": str(max(int(args.embedding_max_concurrency or 1), 1)),
            "IMAGE_VL_EMBED_CONCURRENCY": str(max(int(args.vl_embed_concurrency or 1), 1)),
            "IMAGE_TEXT_EMBED_CONCURRENCY": str(max(int(args.text_embed_concurrency or 1), 1)),
            "IMAGE_WORKER_CONCURRENCY": str(max(int(args.image_worker_concurrency or 1), 1)),
            "IMAGE_DOWNLOAD_CONCURRENCY": str(max(int(args.image_download_concurrency or 1), 1)),
            "IMAGE_WORKER_BATCH_SIZE": str(max(int(args.image_worker_batch_size or 1), 1)),
            "IMAGE_MILVUS_WRITE_CONCURRENCY": str(max(int(args.image_write_concurrency or 1), 1)),
            "IMAGE_MILVUS_WRITE_BATCH_SIZE": str(max(int(args.image_write_batch_size or 1), 1)),
            "IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS": str(
                max(int(args.image_write_flush_interval_ms or 1), 1)
            ),
            "IMAGE_MILVUS_WRITE_MODE": str(args.image_write_mode or "insert_or_skip_if_exists"),
            "ADMIN_AUTH_ENABLED": "true",
            "ADMIN_READ_TOKENS": READ_TOKEN,
            "ADMIN_WRITE_TOKENS": WRITE_TOKEN,
        }
    )
    return env


def seed_tasks(
    *,
    count: int,
    seed_limit: int,
    run_id: str,
    doc_prefix: str,
    report_path: Path,
) -> None:
    command = [
        str(VENV_PYTHON),
        str(SERVICE_ROOT / "scripts" / "seed_repeatable_image_tasks.py"),
        "--count",
        str(max(int(count or 0), 0)),
        "--seed-limit",
        str(max(int(seed_limit or 1), 1)),
        "--run-id",
        run_id,
        "--doc-prefix",
        doc_prefix,
        "--report-path",
        str(report_path),
    ]
    subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def start_service(
    *,
    port: int,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen[str]:
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(SERVICE_ROOT),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )


def wait_for_service(port: int, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(int(timeout_seconds or 1), 1)
    last_error = "unknown"
    while time.monotonic() < deadline:
        try:
            payload = fetch_json(f"http://127.0.0.1:{port}/health")
            if payload.get("status") == "ok":
                return payload
            last_error = json.dumps(payload, ensure_ascii=False)[:300]
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(2)
    raise TimeoutError(f"service did not become healthy within {timeout_seconds}s: {last_error}")


def ensure_startup_ready(health: dict[str, Any], image_status: dict[str, Any]) -> None:
    runtime = (health.get("runtime") or {})
    image_runtime = (runtime.get("image") or {})
    components = health.get("components") or {}
    worker_status = image_status.get("status") or {}
    if health.get("status") != "ok":
        raise RuntimeError("startup health status is not ok")
    if components.get("milvus") != "connected":
        raise RuntimeError("milvus not connected after restart")
    if components.get("mongodb") != "connected":
        raise RuntimeError("mongodb not connected after restart")
    if components.get("embedding_api") != "available":
        raise RuntimeError("embedding_api not available after restart")
    if not bool(image_runtime.get("worker_enabled")):
        raise RuntimeError("image worker not enabled after restart")
    if not bool(image_runtime.get("worker_running")):
        raise RuntimeError("image worker not running after restart")
    if not bool(worker_status.get("worker_enabled")):
        raise RuntimeError("image worker status endpoint reports worker_enabled=false")
    if not bool(worker_status.get("worker_running")):
        raise RuntimeError("image worker status endpoint reports worker_running=false")


def collect_sample(port: int, sample_index: int) -> dict[str, Any]:
    health = fetch_json(f"http://127.0.0.1:{port}/health")
    image_status = fetch_json(
        f"http://127.0.0.1:{port}/admin/workers/image/status",
        headers=auth_headers(READ_TOKEN),
    )
    counts = image_counts_from_status(image_status)
    image_runtime = ((health.get("runtime") or {}).get("image") or {})
    control = image_status.get("control") or {}
    return {
        "ts": iso_now(),
        "sample_index": sample_index,
        "health": summarize_health(health),
        "image": {
            "counts": counts,
            "control": {
                "paused": control.get("paused"),
                "draining": control.get("draining"),
                "pause_reason": control.get("pause_reason"),
                "inflight_runs": control.get("inflight_runs"),
            },
            "runner": {
                "worker_enabled": image_runtime.get("worker_enabled"),
                "worker_running": image_runtime.get("worker_running"),
                "last_processed_count": image_runtime.get("last_processed_count"),
                "last_error": image_runtime.get("last_error"),
                "model_concurrency": image_runtime.get("model_concurrency"),
                "milvus_write": image_runtime.get("milvus_write"),
            },
            "stage_executors": image_runtime.get("stage_executors") or {},
        },
    }


def summarize_health(health: dict[str, Any]) -> dict[str, Any]:
    components = health.get("components") or {}
    runtime = health.get("runtime") or {}
    return {
        "status": health.get("status"),
        "components": {
            "milvus": components.get("milvus"),
            "mongodb": components.get("mongodb"),
            "embedding_api": components.get("embedding_api"),
            "kafka": components.get("kafka"),
            "consumer": components.get("consumer"),
            "compensator": components.get("compensator"),
        },
        "runtime_control": runtime.get("control") or {},
    }


def build_summary(
    *,
    args: argparse.Namespace,
    run_id: str,
    started_at: str,
    finished_at: str,
    baseline_counts: dict[str, int],
    startup_health: dict[str, Any],
    startup_status: dict[str, Any],
    final_health: dict[str, Any],
    final_status: dict[str, Any],
    drained_status: dict[str, Any],
    samples: list[dict[str, Any]],
    stdout_path: Path,
    stderr_path: Path,
    jsonl_path: Path,
    seed_report_path: Path | None,
) -> dict[str, Any]:
    first_counts = image_counts_from_sample(samples[0]) if samples else baseline_counts
    last_counts = image_counts_from_sample(samples[-1]) if samples else baseline_counts
    delta = {
        "baseline_pending_seeded": first_counts.get("PENDING", 0) - baseline_counts.get("PENDING", 0),
        "success": last_counts.get("SUCCESS", 0) - first_counts.get("SUCCESS", 0),
        "pending": last_counts.get("PENDING", 0) - first_counts.get("PENDING", 0),
        "failure": last_counts.get("FAILURE", 0) - first_counts.get("FAILURE", 0),
        "leased": last_counts.get("LEASED", 0) - first_counts.get("LEASED", 0),
    }
    completed_count = max(delta["success"], 0)
    failure_delta = max(delta["failure"], 0)
    failure_ratio = round((failure_delta / completed_count), 4) if completed_count > 0 else None
    dependency_issues = dependency_issue_samples(samples)
    error_series = [str((((item.get("image") or {}).get("runner") or {}).get("last_error")) or "") for item in samples]
    repeated_error = most_common_non_empty(error_series)
    leased_values = [image_counts_from_sample(item).get("LEASED", 0) for item in samples]
    stage_summary = summarize_stage_executors(samples)

    reasons: list[str] = []
    if completed_count <= 0:
        reasons.append("success_not_growing")
    if delta["pending"] >= 0:
        reasons.append("pending_not_decreasing")
    if dependency_issues:
        reasons.append("dependency_unhealthy")
    if failure_ratio is None and failure_delta > 0:
        reasons.append("failure_without_success")
    elif failure_ratio is not None and failure_ratio > 0.05:
        reasons.append("failure_ratio_exceeds_5pct")
    if repeated_error["count"] >= 3:
        reasons.append("repeated_worker_error")
    final_control = final_status.get("control") or {}
    if not bool(final_control.get("draining")):
        reasons.append("worker_not_drained_at_end")

    return {
        "phase": "precheck",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": int(args.duration_seconds or 0),
        "sample_interval_seconds": int(args.sample_interval_seconds or 0),
        "seed_count_requested": int(args.seed_count or 0),
        "config": {
            "embedding_max_concurrency": int(args.embedding_max_concurrency or 0),
            "vl_embed_concurrency": int(args.vl_embed_concurrency or 0),
            "text_embed_concurrency": int(args.text_embed_concurrency or 0),
            "image_write_mode": args.image_write_mode,
            "image_write_batch_size": int(args.image_write_batch_size or 0),
            "image_write_flush_interval_ms": int(args.image_write_flush_interval_ms or 0),
        },
        "baseline_counts": baseline_counts,
        "start_counts": first_counts,
        "end_counts": last_counts,
        "delta": delta,
        "completed_count": completed_count,
        "failure_ratio": failure_ratio,
        "dependency_unhealthy_timestamps": dependency_issues,
        "leased": {
            "start": first_counts.get("LEASED", 0),
            "end": last_counts.get("LEASED", 0),
            "max": max(leased_values) if leased_values else 0,
        },
        "most_common_worker_error": repeated_error,
        "startup": {
            "health": summarize_health(startup_health),
            "image_status": summarize_worker_status(startup_status),
        },
        "final": {
            "health": summarize_health(final_health),
            "image_status": summarize_worker_status(final_status),
            "drained_status": summarize_worker_status(drained_status),
        },
        "stage_executor_summary": stage_summary,
        "artifacts": {
            "samples_jsonl": str(jsonl_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "seed_report": str(seed_report_path) if seed_report_path else None,
        },
        "sample_count": len(samples),
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }


def summarize_worker_status(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    if "status" in payload:
        control = payload.get("control") or {}
        status = payload.get("status") or {}
    else:
        control = payload.get("control") or {}
        status = payload
    return {
        "control": {
            "paused": control.get("paused"),
            "draining": control.get("draining"),
            "pause_reason": control.get("pause_reason"),
            "inflight_runs": control.get("inflight_runs"),
        },
        "status": {
            "worker_enabled": status.get("worker_enabled"),
            "worker_running": status.get("worker_running"),
            "last_processed_count": status.get("last_processed_count"),
            "last_error": status.get("last_error"),
            "task_summary": status.get("task_summary"),
            "stage_executors": status.get("stage_executors"),
        },
    }


def summarize_stage_executors(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, float | int]] = {}
    if not samples:
        return summary
    latest = ((samples[-1].get("image") or {}).get("stage_executors") or {})
    for name, payload in latest.items():
        queue_wait = payload.get("queue_wait_ms") or {}
        run_latency = payload.get("run_latency_ms") or {}
        summary[name] = {
            "success_count": int(payload.get("success_count") or 0),
            "failure_count": int(payload.get("failure_count") or 0),
            "backlog": int(payload.get("backlog") or 0),
            "inflight": int(payload.get("inflight") or 0),
            "queue_wait_avg_ms": float(queue_wait.get("avg") or 0.0),
            "queue_wait_p95_ms": float(queue_wait.get("p95") or 0.0),
            "run_avg_ms": float(run_latency.get("avg") or 0.0),
            "run_p95_ms": float(run_latency.get("p95") or 0.0),
            "run_max_ms": float(run_latency.get("max") or 0.0),
        }
    return summary


def dependency_issue_samples(samples: list[dict[str, Any]]) -> list[str]:
    bad: list[str] = []
    for item in samples:
        components = ((item.get("health") or {}).get("components") or {})
        if (
            components.get("milvus") != "connected"
            or components.get("mongodb") != "connected"
            or components.get("embedding_api") != "available"
        ):
            bad.append(str(item.get("ts") or ""))
    return bad


def most_common_non_empty(values: list[str]) -> dict[str, Any]:
    filtered = [value for value in values if value]
    if not filtered:
        return {"message": "", "count": 0}
    counter = Counter(filtered)
    message, count = counter.most_common(1)[0]
    return {"message": message, "count": count}


def image_counts_from_status(payload: dict[str, Any]) -> dict[str, int]:
    if not payload:
        return normalize_counts({})
    status = payload.get("status") or payload
    task_summary = (status.get("task_summary") or {})
    counts = task_summary.get("counts") or {}
    return normalize_counts(counts)


def image_counts_from_sample(sample: dict[str, Any]) -> dict[str, int]:
    return normalize_counts((((sample.get("image") or {}).get("counts")) or {}))


def normalize_counts(counts: dict[str, Any]) -> dict[str, int]:
    return {
        "PENDING": int(counts.get("PENDING", 0) or 0),
        "LEASED": int(counts.get("LEASED", 0) or 0),
        "SUCCESS": int(counts.get("SUCCESS", 0) or 0),
        "FAILURE": int(counts.get("FAILURE", 0) or 0),
        "SKIPPED": int(counts.get("SKIPPED", 0) or 0),
        "HELD_HISTORY": int(counts.get("HELD_HISTORY", 0) or 0),
    }


def wait_for_drain(*, port: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + max(int(timeout_seconds or 1), 1)
    while time.monotonic() < deadline:
        payload = fetch_json(
            f"http://127.0.0.1:{port}/admin/workers/image/status",
            headers=auth_headers(READ_TOKEN),
        )
        control = payload.get("control") or {}
        if bool(control.get("draining")) and int(control.get("inflight_runs") or 0) == 0:
            return
        time.sleep(2)
    raise TimeoutError("image worker did not drain within timeout")


def stop_listener_on_port(port: int) -> None:
    pid = listener_pid(port)
    if pid is None:
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if listener_pid(port) is None:
            return
        time.sleep(1)
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if listener_pid(port) is None:
            return
        time.sleep(1)
    raise TimeoutError(f"port {port} still has listener pid {pid}")


def listener_pid(port: int) -> int | None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) != 0:
            return None
    if os.name != "nt":
        return None
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"$conn=Get-NetTCPConnection -LocalPort {port} -State Listen "
            "-ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess; "
            "if ($conn) { Write-Output $conn }"
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    text = (result.stdout or "").strip()
    return int(text) if text.isdigit() else None


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def fetch_json(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    response = httpx.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


def try_fetch_json(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        return fetch_json(url, headers=headers)
    except Exception:
        return {}


def query_image_task_counts() -> dict[str, int]:
    client = MongoClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )
    try:
        collection = client[settings.MONGO_DB][settings.MONGO_IMAGE_TASK_COLLECTION]
        return normalize_counts(
            {
                "PENDING": collection.count_documents({"status": "PENDING"}),
                "LEASED": collection.count_documents({"status": "LEASED"}),
                "SUCCESS": collection.count_documents({"status": "SUCCESS"}),
                "FAILURE": collection.count_documents({"status": "FAILURE"}),
                "SKIPPED": collection.count_documents({"status": "SKIPPED"}),
                "HELD_HISTORY": collection.count_documents({"status": "HELD_HISTORY"}),
            }
        )
    finally:
        client.close()


def post_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    request = Request(url, method="POST", headers=headers or {})
    try:
        with urlopen(request, timeout=max(int(timeout_seconds or 1), 1)) as response:
            body = response.read()
            return json.loads(body.decode("utf-8")) if body else {}
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {error.code} {body[:300]}") from error
    except URLError as error:
        raise RuntimeError(f"POST {url} failed: {error}") from error


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
