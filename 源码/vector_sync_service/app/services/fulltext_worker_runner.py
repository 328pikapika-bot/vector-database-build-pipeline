"""Lifecycle wrapper for the full-text worker skeleton."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.config import settings
from app.models.sync_runtime import utcnow
from app.observability.logger import get_logger
from app.observability.metrics import (
    worker_loop_running,
    worker_task_claims_total,
    worker_task_processed_total,
)
from app.services.fulltext_task_repository import (
    FullTextTaskRepository,
    get_fulltext_task_repository,
)
from app.services.fulltext_worker import FullTextWorker, FullTextWorkerResult
from app.services.runtime_control import (
    BackgroundLoopControl,
    get_background_loop_control,
)

logger = get_logger(__name__)


class FullTextWorkerRunner:
    def __init__(
        self,
        *,
        repository: Optional[FullTextTaskRepository] = None,
        worker: Optional[FullTextWorker] = None,
        lease_seconds: Optional[int] = None,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        poll_interval_seconds: Optional[int] = None,
        task_timeout_seconds: Optional[float] = None,
        control: Optional[BackgroundLoopControl] = None,
    ) -> None:
        self.repository = repository or get_fulltext_task_repository()
        self.worker = worker or FullTextWorker(repository=self.repository)
        self.lease_seconds = lease_seconds or settings.FULLTEXT_WORKER_LEASE_SEC
        self.batch_size = max(int(batch_size or settings.FULLTEXT_WORKER_BATCH_SIZE), 1)
        self.concurrency = max(
            int(concurrency or settings.FULLTEXT_WORKER_CONCURRENCY or 1),
            1,
        )
        self.poll_interval_seconds = (
            poll_interval_seconds or settings.FULLTEXT_WORKER_POLL_INTERVAL_SEC
        )
        self.task_timeout_seconds = (
            float(settings.FULLTEXT_WORKER_TASK_TIMEOUT_SEC)
            if task_timeout_seconds is None
            else float(task_timeout_seconds)
        )
        min_timeout_seconds = float(
            getattr(settings, "FULLTEXT_WORKER_MIN_TASK_TIMEOUT_SEC", 0) or 0
        )
        if task_timeout_seconds is None and min_timeout_seconds > 0:
            self.task_timeout_seconds = max(
                self.task_timeout_seconds,
                min_timeout_seconds,
            )
        self.control = control or get_background_loop_control("fulltext")
        self._running = False
        self._last_run_at: Optional[str] = None
        self._last_task_id: Optional[str] = None
        self._last_processed_count = 0
        self._last_error: Optional[str] = None

    def status(self) -> dict[str, Any]:
        control_status = self.control.snapshot()
        return {
            "enabled": settings.FULLTEXT_WORKER_ENABLED,
            "running": self._running,
            "last_run_at": self._last_run_at,
            "last_task_id": self._last_task_id,
            "last_processed_count": self._last_processed_count,
            "last_error": self._last_error,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "claim_created_after": getattr(settings, "FULLTEXT_WORKER_CLAIM_CREATED_AFTER", ""),
            "poll_interval_seconds": self.poll_interval_seconds,
            "task_timeout_seconds": self.task_timeout_seconds,
            "paused": control_status["paused"],
            "draining": control_status["draining"],
            "pause_reason": control_status["pause_reason"],
            "inflight_runs": control_status["inflight_runs"],
        }

    async def run_once(self, worker_id: str) -> dict[str, Any]:
        self._last_run_at = utcnow().isoformat()
        self._last_task_id = None
        self._last_processed_count = 0

        if await self.control.is_paused_or_draining():
            worker_task_claims_total.labels(
                worker_type="fulltext",
                result="paused",
            ).inc()
            self._last_error = None
            return {
                "claimed": False,
                "processed": False,
                "task_id": None,
                "result": None,
                "paused": True,
            }

        async with self.control.track_inflight("fulltext_run_once"):
            if await self.control.is_paused_or_draining():
                worker_task_claims_total.labels(
                    worker_type="fulltext",
                    result="paused",
                ).inc()
                self._last_error = None
                return {
                    "claimed": False,
                    "processed": False,
                    "task_id": None,
                    "result": None,
                    "paused": True,
                }

            results = await self._process_tasks_with_refill(worker_id=worker_id)
            if not results:
                worker_task_claims_total.labels(
                    worker_type="fulltext",
                    result="empty",
                ).inc()
                self._last_error = None
                return {
                    "claimed": False,
                    "processed": False,
                    "task_id": None,
                    "result": None,
                }

            worker_task_claims_total.labels(
                worker_type="fulltext",
                result="claimed",
            ).inc(len(results))
            errors = _result_errors(results)
            self._last_processed_count = sum(
                1 for item in results if bool(item.get("processed"))
            )
            self._last_error = "; ".join(errors) if errors else None
            first_result = results[0] if results else None
            return {
                "claimed": True,
                "processed": any(bool(item.get("processed")) for item in results),
                "task_id": first_result.get("task_id") if first_result else None,
                "task_ids": [item.get("task_id") for item in results],
                "result": first_result.get("result") if first_result else None,
                "results": results,
                "error": first_result.get("error") if first_result else None,
            }

    async def _process_tasks_with_refill(self, *, worker_id: str) -> list[dict[str, Any]]:
        total_limit = max(self.batch_size, 1)
        concurrency = min(max(self.concurrency, 1), total_limit)
        next_index = 0
        running: dict[asyncio.Task[tuple[int, dict[str, Any]]], int] = {}
        active_task_ids: set[str] = set()
        results_by_index: dict[int, dict[str, Any]] = {}

        async def run_one(index: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            task_id = str(task.get("task_id") or task.get("_id") or "")
            result = await self._process_claimed_task(
                task,
                worker_id=worker_id,
                task_id=task_id,
            )
            return index, result

        async def claim_and_start() -> bool:
            nonlocal next_index
            if next_index >= total_limit:
                return False
            if await self.control.is_paused_or_draining():
                return False
            task = await self.repository.claim_next_task(
                worker_id=worker_id,
                lease_seconds=self.lease_seconds,
                exclude_task_ids=active_task_ids,
            )
            if task is None:
                return False
            task_id = str(task.get("task_id") or task.get("_id") or "")
            index = next_index
            next_index += 1
            active_task_ids.add(task_id)
            future = asyncio.create_task(run_one(index, task))
            running[future] = index
            return True

        while len(running) < concurrency and await claim_and_start():
            pass

        while running:
            done, _ = await asyncio.wait(
                running.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                running.pop(future, None)
                index, result = await future
                task_id = str(result.get("task_id") or "")
                if task_id:
                    active_task_ids.discard(task_id)
                results_by_index[index] = result
            while len(running) < concurrency and await claim_and_start():
                pass

        results = [results_by_index[index] for index in sorted(results_by_index)]
        last_task_id = None
        for item in results:
            if item.get("processed") or item.get("error"):
                last_task_id = item.get("task_id") or last_task_id
        self._last_task_id = last_task_id
        return results

    async def _process_claimed_task(
        self,
        task: dict[str, Any],
        *,
        worker_id: str,
        task_id: str,
    ) -> dict[str, Any]:
            try:
                if self.task_timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        self.worker.process_task(task),
                        timeout=self.task_timeout_seconds,
                    )
                else:
                    result = await self.worker.process_task(task)
            except asyncio.TimeoutError:
                error_code = "FULLTEXT_WORKER_TASK_TIMEOUT"
                error_message = (
                    f"fulltext worker task exceeded {self.task_timeout_seconds:g}s timeout"
                )
                await self.repository.mark_task_failure(
                    task_id,
                    error_code=error_code,
                    error_message=error_message,
                    retryable=True,
                )
                self._last_processed_count = 1
                self._last_error = error_message
                worker_task_processed_total.labels(
                    worker_type="fulltext",
                    status="timeout",
                    error_code=error_code,
                ).inc()
                logger.warning(
                    "fulltext_worker_task_timeout",
                    worker_id=worker_id,
                    task_id=task_id,
                    timeout_seconds=self.task_timeout_seconds,
                )
                result = FullTextWorkerResult(
                    task_id=task_id,
                    success=False,
                    artifact_refs=[],
                    chunk_count=0,
                    error_code=error_code,
                    error_message=error_message,
                    retryable=True,
                )
                return {
                    "claimed": True,
                    "processed": True,
                    "task_id": task_id,
                    "result": _worker_result_to_dict(result),
                }
            except Exception as error:
                worker_task_processed_total.labels(
                    worker_type="fulltext",
                    status="exception",
                    error_code="worker_exception",
                ).inc()
                logger.error(
                    "fulltext_worker_run_once_failed",
                    worker_id=worker_id,
                    task_id=task_id,
                    error=str(error),
                )
                return {
                    "claimed": True,
                    "processed": False,
                    "task_id": task_id,
                    "error": str(error),
                }

            worker_task_processed_total.labels(
                worker_type="fulltext",
                status="success" if result.success else "failure",
                error_code=result.error_code or "none",
            ).inc()
            return {
                "claimed": True,
                "processed": True,
                "task_id": task_id,
                "result": _worker_result_to_dict(result),
            }

    async def run_loop(self, worker_id: str) -> None:
        self._running = True
        worker_loop_running.labels(worker_type="fulltext").set(1.0)
        logger.info("fulltext_worker_loop_started", worker_id=worker_id)
        try:
            while True:
                await self.control.wait_if_paused_or_draining()
                await self.run_once(worker_id=worker_id)
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("fulltext_worker_loop_cancelled", worker_id=worker_id)
            raise
        finally:
            self._running = False
            worker_loop_running.labels(worker_type="fulltext").set(0.0)
            logger.info("fulltext_worker_loop_stopped", worker_id=worker_id)


def _worker_result_to_dict(result: FullTextWorkerResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "success": result.success,
        "artifact_refs": result.artifact_refs,
        "chunk_count": result.chunk_count,
        "timings_ms": result.timings_ms,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "retryable": result.retryable,
    }


def _result_errors(results: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for item in results:
        result = item.get("result")
        if isinstance(result, dict) and result.get("error_message"):
            errors.append(str(result["error_message"]))
        elif item.get("error"):
            errors.append(str(item["error"]))
    return errors[:5]


_fulltext_worker_runner: Optional[FullTextWorkerRunner] = None


def get_fulltext_worker_runner() -> FullTextWorkerRunner:
    global _fulltext_worker_runner
    if _fulltext_worker_runner is None:
        _fulltext_worker_runner = FullTextWorkerRunner()
    return _fulltext_worker_runner
