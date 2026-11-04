"""Lifecycle wrapper for the image worker skeleton."""
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
from app.services.image_task_repository import (
    ImageTaskRepository,
    get_image_task_repository,
)
from app.services.image_worker import ImageWorker, ImageWorkerResult
from app.services.runtime_control import (
    BackgroundLoopControl,
    get_background_loop_control,
)

logger = get_logger(__name__)


class ImageWorkerRunner:
    def __init__(
        self,
        *,
        repository: Optional[ImageTaskRepository] = None,
        worker: Optional[ImageWorker] = None,
        lease_seconds: Optional[int] = None,
        batch_size: Optional[int] = None,
        poll_interval_seconds: Optional[int] = None,
        task_timeout_seconds: Optional[float] = None,
        concurrency: Optional[int] = None,
        download_lane: str = "fast",
        control: Optional[BackgroundLoopControl] = None,
    ) -> None:
        self.repository = repository or get_image_task_repository()
        self.download_lane = str(download_lane or "fast").strip().lower()
        if self.download_lane not in {"fast", "slow"}:
            raise ValueError("download_lane must be fast or slow")
        self.worker = worker or ImageWorker(
            repository=self.repository,
            download_timeout=(
                settings.IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC
                if self.download_lane == "slow"
                else settings.IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC
            ),
        )
        self.lease_seconds = lease_seconds or settings.IMAGE_WORKER_LEASE_SEC
        self.batch_size = max(int(batch_size or settings.IMAGE_WORKER_BATCH_SIZE), 1)
        configured_concurrency = (
            settings.IMAGE_SLOW_DOWNLOAD_CONCURRENCY
            if self.download_lane == "slow" and concurrency is None
            else concurrency or settings.IMAGE_WORKER_CONCURRENCY
        )
        self.concurrency = max(int(configured_concurrency or 1), 1)
        self.poll_interval_seconds = (
            poll_interval_seconds or settings.IMAGE_WORKER_POLL_INTERVAL_SEC
        )
        self.task_timeout_seconds = (
            float(settings.IMAGE_WORKER_TASK_TIMEOUT_SEC)
            if task_timeout_seconds is None
            else float(task_timeout_seconds)
        )
        self.control = control or get_background_loop_control("image")
        self._running = False
        self._last_run_at: Optional[str] = None
        self._last_task_id: Optional[str] = None
        self._last_processed_count = 0
        self._last_error: Optional[str] = None

    def status(self) -> dict[str, Any]:
        control_status = self.control.snapshot()
        return {
            "enabled": settings.IMAGE_WORKER_ENABLED,
            "running": self._running,
            "last_run_at": self._last_run_at,
            "last_task_id": self._last_task_id,
            "last_processed_count": self._last_processed_count,
            "last_error": self._last_error,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "poll_interval_seconds": self.poll_interval_seconds,
            "task_timeout_seconds": self.task_timeout_seconds,
            "download_lane": self.download_lane,
            "model_concurrency": {
                "vl_chat": settings.IMAGE_VL_CHAT_CONCURRENCY,
                "vl_embed": settings.IMAGE_VL_EMBED_CONCURRENCY,
                "text_embed": settings.IMAGE_TEXT_EMBED_CONCURRENCY,
                "total": settings.IMAGE_MODEL_TOTAL_CONCURRENCY,
                "visual_priority": True,
            },
            "milvus_write": {
                "concurrency": settings.IMAGE_MILVUS_WRITE_CONCURRENCY,
                "batch_size": settings.IMAGE_MILVUS_WRITE_BATCH_SIZE,
                "flush_interval_ms": settings.IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS,
            },
            "model_input": {
                "format": settings.IMAGE_MODEL_INPUT_FORMAT,
                "max_pixels": settings.IMAGE_MODEL_INPUT_MAX_PIXELS,
                "max_side": settings.IMAGE_MODEL_INPUT_MAX_SIDE,
                "jpeg_quality": settings.IMAGE_MODEL_INPUT_JPEG_QUALITY,
                "min_jpeg_quality": settings.IMAGE_MODEL_INPUT_MIN_JPEG_QUALITY,
                "max_bytes": settings.IMAGE_MODEL_INPUT_MAX_BYTES,
            },
            "runtime_attachment_mode": settings.IMAGE_RUNTIME_ATTACHMENT_MODE,
            "success_artifact_mode": settings.IMAGE_SUCCESS_ARTIFACT_MODE,
            "stage_executors": getattr(
                self.worker,
                "stage_executor_status",
                lambda: {},
            )(),
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
                worker_type="image",
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

        async with self.control.track_inflight("image_run_once"):
            if await self.control.is_paused_or_draining():
                worker_task_claims_total.labels(
                    worker_type="image",
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
                    worker_type="image",
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
                worker_type="image",
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

    async def _process_tasks_with_refill(
        self,
        *,
        worker_id: str,
    ) -> list[dict[str, Any]]:
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
            task = await self._claim_one_task(
                worker_id=worker_id,
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

        try:
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
        except asyncio.CancelledError:
            for future in running:
                future.cancel()
            if running:
                await asyncio.gather(*list(running), return_exceptions=True)
            raise

        results = [results_by_index[index] for index in sorted(results_by_index)]
        last_task_id = None
        for item in results:
            if item.get("processed") or item.get("error"):
                last_task_id = item.get("task_id") or last_task_id
        self._last_task_id = last_task_id
        return results

    async def _claim_one_task(
        self,
        *,
        worker_id: str,
        exclude_task_ids: set[str],
    ) -> Optional[dict[str, Any]]:
        kwargs = {
            "worker_id": worker_id,
            "lease_seconds": self.lease_seconds,
            "exclude_task_ids": exclude_task_ids,
        }
        if self.download_lane == "slow":
            kwargs["download_lane"] = "slow"
        return await self.repository.claim_next_task(**kwargs)

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
            error_code = "IMAGE_WORKER_TASK_TIMEOUT"
            error_message = (
                f"image worker task exceeded {self.task_timeout_seconds:g}s timeout"
            )
            retryable = _task_retryable_after_failure(task)
            await self.repository.mark_task_failure(
                task_id,
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
            )
            worker_task_processed_total.labels(
                worker_type="image",
                status="timeout",
                error_code=error_code,
            ).inc()
            logger.warning(
                "image_worker_task_timeout",
                worker_id=worker_id,
                task_id=task_id,
                timeout_seconds=self.task_timeout_seconds,
            )
            result = ImageWorkerResult(
                task_id=task_id,
                success=False,
                artifact_refs=[],
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
            )
            return {
                "claimed": True,
                "processed": True,
                "task_id": task_id,
                "result": _worker_result_to_dict(result),
            }
        except Exception as error:
            worker_task_processed_total.labels(
                worker_type="image",
                status="exception",
                error_code="worker_exception",
            ).inc()
            logger.error(
                "image_worker_run_once_failed",
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
            worker_type="image",
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
        worker_loop_running.labels(worker_type="image").set(1.0)
        logger.info("image_worker_loop_started", worker_id=worker_id)
        try:
            while True:
                await self.control.wait_if_paused_or_draining()
                await self.run_once(worker_id=worker_id)
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("image_worker_loop_cancelled", worker_id=worker_id)
            raise
        finally:
            close = getattr(self.worker, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            self._running = False
            worker_loop_running.labels(worker_type="image").set(0.0)
            logger.info("image_worker_loop_stopped", worker_id=worker_id)


def _worker_result_to_dict(result: ImageWorkerResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "success": result.success,
        "artifact_refs": result.artifact_refs,
        "timings_ms": result.timings_ms,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "retryable": result.retryable,
    }


def _result_errors(results: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for item in results:
        if item.get("error"):
            errors.append(str(item["error"]))
        worker_result = item.get("result") or {}
        if worker_result.get("error_message"):
            errors.append(str(worker_result["error_message"]))
    return errors


def _task_retryable_after_failure(task: dict[str, Any]) -> bool:
    max_attempts = max(int(settings.IMAGE_TASK_MAX_ATTEMPTS or 1), 1)
    attempt_count = int(task.get("attempt_count") or 1)
    return attempt_count < max_attempts


_image_worker_runner: Optional[ImageWorkerRunner] = None


def get_image_worker_runner() -> ImageWorkerRunner:
    global _image_worker_runner
    if _image_worker_runner is None:
        _image_worker_runner = ImageWorkerRunner()
    return _image_worker_runner
