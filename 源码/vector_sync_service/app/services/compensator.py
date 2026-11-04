"""Mongo failure compensator with strict target binding and time windows."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from app.config import settings
from app.observability.logger import get_logger
from app.observability.metrics import compensator_rounds_total, worker_loop_running
from app.services.batch_processor_existing import retry_failure_task
from app.services.collection_targets import (
    resolve_document_collection_target,
    resolve_fulltext_collection_target,
)
from app.services.document_status_sync import get_document_status_sync
from app.services.independent_kafka_worker import IndependentKafkaWorker
from app.services.milvus_chunk_service import MilvusChunkService
from app.services.milvus_image_service import MilvusImageService
from app.services.mongo_state_manager import get_mongo_state_manager
from app.services.runtime_control import get_background_loop_control

logger = get_logger(__name__)
_REPLAYABLE_PIPELINES = {"metadata", "fulltext", "image"}
_SUPPORTED_PIPELINES = {"metadata", "fulltext", "image"}


class CompensatorRuntimeStatus:
    """In-memory runtime view for the compensator loop."""

    def __init__(self) -> None:
        self.running = False
        self.last_round_at: Optional[str] = None
        self.last_processed_count = 0
        self.last_error: Optional[str] = None
        self.mongo_connected = False
        self.updated_at: Optional[str] = None
        self.in_window: Optional[bool] = None
        self.selected_pipelines: list[str] = []
        self.replayable_pipelines: list[str] = []

    def mark_started(self, *, selected_pipelines: set[str], replayable_pipelines: set[str]) -> None:
        self.running = True
        self.selected_pipelines = sorted(selected_pipelines)
        self.replayable_pipelines = sorted(replayable_pipelines)
        self.updated_at = _utc_iso()

    def mark_round(self, *, processed: int, mongo_connected: bool, in_window: bool) -> None:
        now = _utc_iso()
        self.running = True
        self.last_round_at = now
        self.last_processed_count = processed
        self.mongo_connected = mongo_connected
        self.in_window = in_window
        self.updated_at = now

    def mark_error(self, error: str) -> None:
        self.last_error = error
        self.updated_at = _utc_iso()

    def mark_stopped(self) -> None:
        self.running = False
        self.updated_at = _utc_iso()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "last_round_at": self.last_round_at,
            "last_processed_count": self.last_processed_count,
            "last_error": self.last_error,
            "mongo_connected": self.mongo_connected,
            "in_window": self.in_window,
            "run_mode": _run_mode(),
            "start_time": settings.COMPENSATOR_START_TIME,
            "run_hours": settings.COMPENSATOR_RUN_HOURS,
            "timezone": settings.COMPENSATOR_TIMEZONE,
            "selected_pipelines": self.selected_pipelines,
            "replayable_pipelines": self.replayable_pipelines,
            "updated_at": self.updated_at,
        }


_compensator_status = CompensatorRuntimeStatus()


def get_compensator_status() -> dict[str, Any]:
    control_status = get_background_loop_control("compensator").snapshot()
    return {
        **_compensator_status.status(),
        "paused": control_status["paused"],
        "draining": control_status["draining"],
        "pause_reason": control_status["pause_reason"],
        "inflight_runs": control_status["inflight_runs"],
    }


async def run_compensator() -> None:
    """Retry only target-bound failures during the configured operating window."""
    manager = get_mongo_state_manager()
    control = get_background_loop_control("compensator")
    selected_pipelines = _selected_pipelines()
    replayable_pipelines = selected_pipelines & _REPLAYABLE_PIPELINES
    _compensator_status.mark_started(
        selected_pipelines=selected_pipelines,
        replayable_pipelines=replayable_pipelines,
    )
    worker_loop_running.labels(worker_type="compensator").set(1.0)
    logger.info(
        "compensator_loop_started",
        run_mode=_run_mode(),
        interval_sec=settings.COMPENSATOR_INTERVAL_SEC,
        batch_size=settings.COMPENSATOR_BATCH_SIZE,
        concurrency=settings.COMPENSATOR_CONCURRENCY,
        pipelines=sorted(replayable_pipelines),
    )

    try:
        if _run_mode() == "once":
            await _run_one_controlled_round(manager, control, replayable_pipelines, in_window=True)
            return

        while True:
            try:
                in_window, sleep_seconds = _window_state()
                if not in_window:
                    _compensator_status.mark_round(
                        processed=0,
                        mongo_connected=bool(manager.connected),
                        in_window=False,
                    )
                    await asyncio.sleep(sleep_seconds)
                    continue

                processed = await _run_one_controlled_round(
                    manager,
                    control,
                    replayable_pipelines,
                    in_window=True,
                )
                if processed == 0:
                    await asyncio.sleep(max(int(settings.COMPENSATOR_INTERVAL_SEC), 1))
            except asyncio.CancelledError:
                logger.info("compensator_cancelled")
                break
            except Exception as error:
                _compensator_status.mark_error(str(error))
                logger.error("compensator_loop_error", error=str(error))
                await asyncio.sleep(max(int(settings.COMPENSATOR_INTERVAL_SEC), 1))
    finally:
        _compensator_status.mark_stopped()
        worker_loop_running.labels(worker_type="compensator").set(0.0)
        logger.info("compensator_loop_stopped")


async def _run_one_controlled_round(
    manager: Any,
    control: Any,
    pipelines: set[str],
    *,
    in_window: bool,
) -> int:
    await control.wait_if_paused_or_draining()
    async with control.track_inflight("compensator_round"):
        if await control.is_paused_or_draining():
            return 0
        processed = await _run_compensator_round(
            manager,
            pipelines,
            stop_claiming_when_window_closes=_run_mode() == "scheduled",
        )
    _compensator_status.mark_round(
        processed=processed,
        mongo_connected=bool(manager.connected),
        in_window=in_window,
    )
    return processed


async def _run_compensator_round(
    manager: Any,
    pipelines: set[str],
    *,
    stop_claiming_when_window_closes: bool = False,
) -> int:
    compensator_rounds_total.inc()
    if not manager.connected or not pipelines:
        return 0

    targets = _current_retry_targets(pipelines)
    if not targets:
        return 0

    failures: list[dict[str, Any]] = []
    for _ in range(max(int(settings.COMPENSATOR_BATCH_SIZE), 1)):
        if stop_claiming_when_window_closes:
            in_window, _ = _window_state()
            if not in_window:
                break
        failure_task = await manager.claim_failure_for_retry_filtered(
            pipelines=pipelines,
            targets=targets,
        )
        if failure_task is None:
            break
        failures.append(failure_task)

    if not failures:
        return 0

    semaphore = asyncio.Semaphore(max(int(settings.COMPENSATOR_CONCURRENCY), 1))
    replayers = {
        pipeline: IndependentKafkaWorker(pipeline, state_manager=manager)
        for pipeline in pipelines
        if pipeline != "metadata"
    }

    async def _retry(failure_task: dict[str, Any]) -> None:
        async with semaphore:
            failure_key = failure_task.get("failure_key")
            try:
                pipeline = str(failure_task.get("pipeline") or "").strip().lower()
                if pipeline == "metadata":
                    # Preserve the established metadata retry semantics while
                    # fulltext and image move to their direct Kafka replayers.
                    outcome = await retry_failure_task(failure_task)
                else:
                    replayer = replayers.get(pipeline)
                    if replayer is None:
                        raise ValueError(f"no compensator replayer for pipeline={pipeline!r}")
                    outcome = await replayer.replay_failure(failure_task)
                logger.info(
                    "compensator_retry_finished",
                    failure_key=failure_key,
                    pipeline=failure_task.get("pipeline"),
                    doc_id=failure_task.get("doc_id"),
                    record_update_time=failure_task.get("record_update_time"),
                    final_status=outcome.final_status,
                    commit_safe=outcome.commit_safe,
                    pending_failures=outcome.pending_failure_refs,
                )
            except Exception as error:
                _compensator_status.mark_error(str(error))
                logger.error(
                    "compensator_retry_failed",
                    failure_key=failure_key,
                    pipeline=failure_task.get("pipeline"),
                    doc_id=failure_task.get("doc_id"),
                    error=str(error),
                )

    try:
        await asyncio.gather(*(_retry(failure_task) for failure_task in failures))
        reconciled = await _reconcile_milvus_sync_status(manager)
        return len(failures) + reconciled
    finally:
        await asyncio.gather(
            *(replayer.close() for replayer in replayers.values()),
            return_exceptions=True,
        )


def _current_retry_targets(pipelines: set[str]) -> dict[str, tuple[str, str]]:
    targets: dict[str, tuple[str, str]] = {}
    if "metadata" in pipelines:
        from app.services.milvus_service import get_milvus_service

        service = get_milvus_service()
        target = resolve_document_collection_target(
            service.client,
            logical_name=settings.resolved_document_collection_name,
            vector_dim=settings.VECTOR_DIM,
            ensure_collection=lambda _name: None,
        )
        targets["metadata"] = (target.resolved_name, target.target_signature)

    if "fulltext" in pipelines:
        service = MilvusChunkService()
        try:
            target = resolve_fulltext_collection_target(
                service.client,
                logical_name=settings.resolved_fulltext_chunk_collection_name,
                vector_dim=settings.VECTOR_DIM,
                ensure_collection=lambda _name: None,
            )
            targets["fulltext"] = (target.resolved_name, target.target_signature)
        finally:
            service.close()

    if "image" in pipelines:
        service = MilvusImageService()
        try:
            target = service.resolve_collection_target(
                collection_name=settings.IMAGE_COLLECTION_NAME,
            )
            targets["image"] = (target.resolved_name, target.target_signature)
        finally:
            service.close()
    return targets


async def _reconcile_milvus_sync_status(manager: Any) -> int:
    """Mirror recently-updated terminal MongoDB statuses to Milvus sync_status."""
    try:
        docs = await manager.find_terminal_status_docs(
            since_minutes=int(settings.COMPENSATOR_INTERVAL_SEC / 60) + 5,
            limit=settings.COMPENSATOR_BATCH_SIZE,
        )
    except Exception:
        return 0

    if not docs:
        return 0

    doc_ids = [str(d.get("_id", "")) for d in docs if d.get("_id")]
    if not doc_ids:
        return 0

    try:
        syncer = get_document_status_sync()
        result = await syncer.sync_batch(doc_ids)
        synced = result.get("synced", 0)
        if synced:
            logger.info("compensator_status_synced", synced=synced, total=len(doc_ids))
        return synced
    except Exception:
        return 0


def _selected_pipelines() -> set[str]:
    values = {
        item.strip().lower()
        for item in str(settings.COMPENSATOR_PIPELINES or "").split(",")
        if item.strip()
    }
    unknown = values - _SUPPORTED_PIPELINES
    if unknown:
        raise ValueError(f"unsupported COMPENSATOR_PIPELINES values: {sorted(unknown)}")
    return values


def _run_mode() -> str:
    mode = str(settings.COMPENSATOR_RUN_MODE or "scheduled").strip().lower()
    if mode not in {"scheduled", "continuous", "once"}:
        raise ValueError("COMPENSATOR_RUN_MODE must be scheduled, continuous, or once")
    return mode


def _window_state(now: Optional[datetime] = None) -> tuple[bool, float]:
    mode = _run_mode()
    if mode == "continuous":
        return True, 0.0
    if mode == "once":
        return True, 0.0

    tz = ZoneInfo(str(settings.COMPENSATOR_TIMEZONE or "Asia/Shanghai"))
    local_now = (now or datetime.now(tz)).astimezone(tz)
    start_hour, start_minute = _parse_start_time(settings.COMPENSATOR_START_TIME)
    run_hours = float(settings.COMPENSATOR_RUN_HOURS)
    if not 0 < run_hours <= 24:
        raise ValueError("COMPENSATOR_RUN_HOURS must be greater than 0 and no more than 24")

    start_today = local_now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    previous_start = start_today - timedelta(days=1)
    current_window_end = start_today + timedelta(hours=run_hours)
    previous_window_end = previous_start + timedelta(hours=run_hours)
    if start_today <= local_now < current_window_end:
        return True, 0.0
    if previous_start <= local_now < previous_window_end:
        return True, 0.0

    next_start = start_today if local_now < start_today else start_today + timedelta(days=1)
    return False, max((next_start - local_now).total_seconds(), 1.0)


def _parse_start_time(value: str) -> tuple[int, int]:
    parts = str(value or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError("COMPENSATOR_START_TIME must use HH:mm")
    try:
        hour, minute = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("COMPENSATOR_START_TIME must use HH:mm") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("COMPENSATOR_START_TIME must use HH:mm")
    return hour, minute


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
