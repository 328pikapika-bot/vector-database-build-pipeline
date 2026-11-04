"""Independent ops service for publisher, cleanup, inspection, and split backfill lanes."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
import json
from typing import Any, Awaitable, Callable, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.config import settings
from app.observability.logger import get_logger, setup_logging
from app.services.admin_auth import require_admin_read, require_admin_write
from app.services.auto_publisher import AutoPublishController
from app.services.collection_targets import (
    build_document_index_spec,
    build_fulltext_chunk_index_spec,
    resolve_document_collection_target,
    resolve_fulltext_collection_target,
)
from app.services.milvus_chunk_service import MilvusChunkService
from app.services.milvus_service import get_milvus_service
from app.services.mongo_ops_backup import build_backup_inspection
from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager
from app.services.ops_scheduler import seconds_until_next_run
from app.services.retention_cleanup import RetentionCleanupService


setup_logging()
logger = get_logger(__name__)


class OpsControlRequest(BaseModel):
    reason: Optional[str] = None


class OpsCleanupRequest(BaseModel):
    mode: str = "daily"
    dry_run: bool = False


class OpsPublishRequest(BaseModel):
    record_limit: Optional[int] = None


class OpsBackfillRequest(BaseModel):
    window_id: str = ""
    limit: Optional[int] = None
    batch_size: Optional[int] = None
    dry_run: bool = False
    skip_fulltext_drain: bool = False
    fulltext_drain_timeout_seconds: Optional[int] = None


class OpsLaneState:
    def __init__(self, lane: str) -> None:
        self.lane = lane
        self.running = False
        self.startup_completed = False
        self.skip_while_running = 0
        self.last_result: Optional[dict[str, Any]] = None


class OpsServiceState:
    def __init__(self) -> None:
        self.mongo_connected = False
        self.publisher = AutoPublishController(sync_health_url=settings.OPS_SYNC_HEALTH_URL)
        self.scheduler_tasks: dict[str, asyncio.Task] = {}
        self.last_publish: Optional[dict[str, Any]] = None
        self.last_cleanup: Optional[dict[str, Any]] = None
        self.last_backfill: Optional[dict[str, Any]] = None
        self.last_inspection: Optional[dict[str, Any]] = None
        self.started_at: Optional[str] = None
        self.metadata_backfill = OpsLaneState("metadata")
        self.fulltext_backfill = OpsLaneState("fulltext")


state = OpsServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.started_at = _utc_iso()
    try:
        mongo = get_mongo_state_manager()
        await mongo.connect()
        state.mongo_connected = True
    except Exception as error:
        state.mongo_connected = False
        logger.error("ops_service_mongo_init_failed", error=str(error))

    try:
        if settings.OPS_PUBLISH_ENABLED:
            state.scheduler_tasks["publish"] = asyncio.create_task(
                _run_named_schedule("publish", settings.OPS_PUBLISH_SCHEDULE, _run_publish_once)
            )
        if settings.OPS_STARTUP_PUBLISH_ENABLED:
            state.scheduler_tasks["startup_publish"] = asyncio.create_task(_run_startup_publish_once())
        if settings.OPS_CLEANUP_ENABLED:
            state.scheduler_tasks["cleanup"] = asyncio.create_task(
                _run_named_schedule("cleanup", settings.OPS_CLEANUP_SCHEDULE, _run_cleanup_once)
            )
        if settings.OPS_METADATA_BACKFILL_ENABLED:
            state.scheduler_tasks["metadata_backfill"] = asyncio.create_task(
                _run_interval_lane_loop(
                    state.metadata_backfill,
                    interval_hours=float(settings.OPS_METADATA_BACKFILL_INTERVAL_HOURS or 24.0),
                    startup_enabled=bool(settings.OPS_METADATA_BACKFILL_STARTUP_RUN_ENABLED),
                    runner=_run_metadata_backfill_once,
                )
            )
        if settings.OPS_FULLTEXT_BACKFILL_ENABLED:
            state.scheduler_tasks["fulltext_backfill"] = asyncio.create_task(
                _run_interval_lane_loop(
                    state.fulltext_backfill,
                    interval_hours=float(settings.OPS_FULLTEXT_BACKFILL_INTERVAL_HOURS or 1.0),
                    startup_enabled=bool(settings.OPS_FULLTEXT_BACKFILL_STARTUP_RUN_ENABLED),
                    runner=_run_fulltext_backfill_once,
                )
            )
        if settings.OPS_BACKFILL_ENABLED:
            state.scheduler_tasks["legacy_backfill"] = asyncio.create_task(
                _run_named_schedule("backfill", settings.OPS_BACKFILL_SCHEDULE, _run_backfill_once)
            )
        if settings.OPS_INSPECTION_ENABLED:
            state.scheduler_tasks["inspection"] = asyncio.create_task(
                _run_named_schedule("inspection", settings.OPS_INSPECTION_SCHEDULE, _run_inspection_once)
            )
        yield
    finally:
        for task in state.scheduler_tasks.values():
            task.cancel()
        for task in state.scheduler_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        state.scheduler_tasks.clear()
        await close_mongo_state_manager()
        state.mongo_connected = False


app = FastAPI(
    title="NSTL Vector Ops Service",
    description="Publisher / cleanup / inspection / split backfill control plane",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ops", "runtime": await _build_runtime()}


@app.get("/ops/runtime")
async def runtime_status(_auth: dict = Depends(require_admin_read)) -> dict[str, Any]:
    return await _build_runtime()


@app.post("/ops/publisher/pause")
async def pause_publisher(
    payload: Optional[OpsControlRequest] = None,
    _auth: dict = Depends(require_admin_write),
) -> dict[str, Any]:
    return state.publisher.pause(payload.reason if payload else None)


@app.post("/ops/publisher/resume")
async def resume_publisher(_auth: dict = Depends(require_admin_write)) -> dict[str, Any]:
    return state.publisher.resume()


@app.post("/ops/cleanup/run")
async def run_cleanup(
    payload: OpsCleanupRequest,
    _auth: dict = Depends(require_admin_write),
) -> dict[str, Any]:
    result = await _run_cleanup_once(mode=payload.mode, dry_run=payload.dry_run)
    state.last_cleanup = result
    return result


@app.post("/ops/publisher/run")
async def run_publisher(
    payload: Optional[OpsPublishRequest] = None,
    _auth: dict = Depends(require_admin_write),
) -> dict[str, Any]:
    result = await _run_publish_once(
        record_limit=payload.record_limit if payload else None,
        run_reason="manual",
    )
    state.last_publish = result
    return result


@app.post("/ops/metadata-backfill/run")
async def run_metadata_backfill(
    payload: OpsBackfillRequest,
    _auth: dict = Depends(require_admin_write),
) -> dict[str, Any]:
    result = await _run_metadata_backfill_once(
        window_id=payload.window_id,
        limit=payload.limit,
        batch_size=payload.batch_size,
        dry_run=payload.dry_run,
    )
    state.metadata_backfill.last_result = result
    return result


@app.post("/ops/fulltext-backfill/run")
async def run_fulltext_backfill(
    payload: OpsBackfillRequest,
    _auth: dict = Depends(require_admin_write),
) -> dict[str, Any]:
    result = await _run_fulltext_backfill_once(
        window_id=payload.window_id,
        limit=payload.limit,
        batch_size=payload.batch_size,
        dry_run=payload.dry_run,
        skip_fulltext_drain=payload.skip_fulltext_drain,
        fulltext_drain_timeout_seconds=payload.fulltext_drain_timeout_seconds,
    )
    state.fulltext_backfill.last_result = result
    return result


@app.post("/ops/backfill/run")
async def run_backfill(
    payload: OpsBackfillRequest,
    _auth: dict = Depends(require_admin_write),
) -> dict[str, Any]:
    result = await _run_backfill_once(
        window_id=payload.window_id,
        limit=payload.limit,
        batch_size=payload.batch_size,
        dry_run=payload.dry_run,
        skip_fulltext_drain=payload.skip_fulltext_drain,
        fulltext_drain_timeout_seconds=payload.fulltext_drain_timeout_seconds,
    )
    state.last_backfill = result
    return result


@app.post("/ops/inspection/run")
async def run_inspection(_auth: dict = Depends(require_admin_write)) -> dict[str, Any]:
    result = await _run_inspection_once()
    state.last_inspection = result
    return result


async def _build_runtime() -> dict[str, Any]:
    document_target = _resolve_document_target_runtime()
    fulltext_target = _resolve_fulltext_target_runtime()
    return {
        "started_at": state.started_at,
        "mongo_connected": state.mongo_connected,
        "publisher": state.publisher.status(),
        "cleanup_enabled": settings.OPS_CLEANUP_ENABLED,
        "inspection_enabled": settings.OPS_INSPECTION_ENABLED,
        "publish_enabled": settings.OPS_PUBLISH_ENABLED,
        "backfill_enabled": settings.OPS_BACKFILL_ENABLED,
        "metadata_backfill_enabled": settings.OPS_METADATA_BACKFILL_ENABLED,
        "fulltext_backfill_enabled": settings.OPS_FULLTEXT_BACKFILL_ENABLED,
        "startup_publish_enabled": settings.OPS_STARTUP_PUBLISH_ENABLED,
        "startup_publish_record_limit": settings.OPS_STARTUP_PUBLISH_RECORD_LIMIT,
        "schedules": {
            "publish": settings.OPS_PUBLISH_SCHEDULE,
            "cleanup": settings.OPS_CLEANUP_SCHEDULE,
            "backfill": settings.OPS_BACKFILL_SCHEDULE,
            "inspection": settings.OPS_INSPECTION_SCHEDULE,
        },
        "scheduler_running": {
            name: bool(task and not task.done())
            for name, task in state.scheduler_tasks.items()
        },
        "targets": {
            "document": document_target,
            "fulltext": fulltext_target,
        },
        "metadata_backfill": _lane_runtime_dict(
            state.metadata_backfill,
            interval_hours=float(settings.OPS_METADATA_BACKFILL_INTERVAL_HOURS or 24.0),
            startup_enabled=bool(settings.OPS_METADATA_BACKFILL_STARTUP_RUN_ENABLED),
            window_id=settings.OPS_METADATA_BACKFILL_WINDOW_ID or settings.OPS_BACKFILL_WINDOW_ID,
            batch_size=int(settings.OPS_METADATA_BACKFILL_BATCH_SIZE or settings.OPS_BACKFILL_BATCH_SIZE),
            limit=int(settings.OPS_METADATA_BACKFILL_LIMIT or settings.OPS_BACKFILL_LIMIT),
        ),
        "fulltext_backfill": _lane_runtime_dict(
            state.fulltext_backfill,
            interval_hours=float(settings.OPS_FULLTEXT_BACKFILL_INTERVAL_HOURS or 1.0),
            startup_enabled=bool(settings.OPS_FULLTEXT_BACKFILL_STARTUP_RUN_ENABLED),
            window_id=settings.OPS_FULLTEXT_BACKFILL_WINDOW_ID or settings.OPS_BACKFILL_WINDOW_ID,
            batch_size=int(settings.OPS_FULLTEXT_BACKFILL_BATCH_SIZE or settings.OPS_BACKFILL_BATCH_SIZE),
            limit=int(settings.OPS_FULLTEXT_BACKFILL_LIMIT or settings.OPS_BACKFILL_LIMIT),
        ),
        "last_publish": state.last_publish,
        "last_cleanup": state.last_cleanup,
        "last_backfill": state.last_backfill,
        "last_inspection": state.last_inspection,
    }


async def _run_named_schedule(name: str, expression: str, runner) -> None:
    while True:
        try:
            await asyncio.sleep(seconds_until_next_run(expression))
            result = await runner()
            if name == "publish":
                state.last_publish = result
            if name == "cleanup":
                state.last_cleanup = result
            elif name == "backfill":
                state.last_backfill = result
            elif name == "inspection":
                state.last_inspection = result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("ops_service_schedule_failed", name=name, error=str(error))
            await asyncio.sleep(30)


async def _run_interval_lane_loop(
    lane_state: OpsLaneState,
    *,
    interval_hours: float,
    startup_enabled: bool,
    runner: Callable[[], Awaitable[dict[str, Any]]],
) -> None:
    try:
        if startup_enabled:
            await _run_lane_once(lane_state, runner)
            lane_state.startup_completed = True
        while True:
            await asyncio.sleep(max(float(interval_hours or 0) * 3600.0, 1.0))
            await _run_lane_once(lane_state, runner)
    except asyncio.CancelledError:
        raise


async def _run_lane_once(
    lane_state: OpsLaneState,
    runner: Callable[[], Awaitable[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    if lane_state.running:
        lane_state.skip_while_running += 1
        return None
    lane_state.running = True
    try:
        result = await runner()
        lane_state.last_result = result
        return result
    finally:
        lane_state.running = False


async def _run_cleanup_once(*, mode: str = "daily", dry_run: bool = False) -> dict[str, Any]:
    return await RetentionCleanupService().run(mode=mode, dry_run=dry_run)


async def _run_startup_publish_once() -> None:
    try:
        record_limit = int(settings.OPS_STARTUP_PUBLISH_RECORD_LIMIT or 0) or None
        result = await _run_publish_once(record_limit=record_limit, run_reason="startup")
        state.last_publish = result
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001
        state.last_publish = {
            "run_reason": "startup",
            "started_at": _utc_iso(),
            "finished_at": _utc_iso(),
            "stopped_reason": "startup_publish_failed",
            "errors": [f"{type(error).__name__}: {error}"],
        }
        logger.warning("ops_service_startup_publish_failed", error=str(error))


async def _run_publish_once(
    *,
    record_limit: Optional[int] = None,
    run_reason: str = "scheduled",
) -> dict[str, Any]:
    result = await state.publisher.run_daily_once(record_limit=record_limit, run_reason=run_reason)
    report_path = _runtime_report_path("publish")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["report_path"] = str(report_path)
    return result


async def _run_metadata_backfill_once(
    *,
    window_id: str = "",
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return await _run_backfill_script(
        lane="metadata",
        report_prefix="metadata_backfill",
        window_id=window_id or settings.OPS_METADATA_BACKFILL_WINDOW_ID or settings.OPS_BACKFILL_WINDOW_ID,
        limit=settings.OPS_METADATA_BACKFILL_LIMIT if limit is None else limit,
        batch_size=settings.OPS_METADATA_BACKFILL_BATCH_SIZE if batch_size is None else batch_size,
        dry_run=dry_run,
        skip_fulltext_drain=True,
        fulltext_drain_timeout_seconds=0,
    )


async def _run_fulltext_backfill_once(
    *,
    window_id: str = "",
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    dry_run: bool = False,
    skip_fulltext_drain: bool = False,
    fulltext_drain_timeout_seconds: Optional[int] = None,
) -> dict[str, Any]:
    return await _run_backfill_script(
        lane="fulltext",
        report_prefix="fulltext_backfill",
        window_id=window_id or settings.OPS_FULLTEXT_BACKFILL_WINDOW_ID or settings.OPS_BACKFILL_WINDOW_ID,
        limit=settings.OPS_FULLTEXT_BACKFILL_LIMIT if limit is None else limit,
        batch_size=settings.OPS_FULLTEXT_BACKFILL_BATCH_SIZE if batch_size is None else batch_size,
        dry_run=dry_run,
        skip_fulltext_drain=skip_fulltext_drain,
        fulltext_drain_timeout_seconds=(
            settings.OPS_FULLTEXT_BACKFILL_DRAIN_TIMEOUT_SECONDS
            if fulltext_drain_timeout_seconds is None
            else fulltext_drain_timeout_seconds
        ),
    )


async def _run_backfill_once(
    *,
    window_id: str = "",
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    dry_run: bool = False,
    skip_fulltext_drain: bool = False,
    fulltext_drain_timeout_seconds: Optional[int] = None,
) -> dict[str, Any]:
    metadata = await _run_metadata_backfill_once(
        window_id=window_id,
        limit=limit,
        batch_size=batch_size,
        dry_run=dry_run,
    )
    fulltext = await _run_fulltext_backfill_once(
        window_id=window_id,
        limit=limit,
        batch_size=batch_size,
        dry_run=dry_run,
        skip_fulltext_drain=skip_fulltext_drain,
        fulltext_drain_timeout_seconds=fulltext_drain_timeout_seconds,
    )
    return {
        "started_at": metadata.get("started_at"),
        "finished_at": fulltext.get("finished_at"),
        "metadata": metadata,
        "fulltext": fulltext,
        "processed": int((metadata.get("metadata") or {}).get("processed") or 0)
        + int((fulltext.get("fulltext") or {}).get("processed") or 0),
        "marked_done": int((metadata.get("metadata") or {}).get("marked_done") or 0)
        + int((fulltext.get("fulltext") or {}).get("marked_done") or 0),
        "marked_failed": int((metadata.get("metadata") or {}).get("marked_failed") or 0)
        + int((fulltext.get("fulltext") or {}).get("marked_failed") or 0),
    }


async def _run_backfill_script(
    *,
    lane: str,
    report_prefix: str,
    window_id: str,
    limit: int,
    batch_size: int,
    dry_run: bool,
    skip_fulltext_drain: bool,
    fulltext_drain_timeout_seconds: int,
) -> dict[str, Any]:
    from scripts.run_image_priority_backfill import main_async, parse_args

    report_path = _runtime_report_path(report_prefix)
    args = parse_args(
        [
            "--lane",
            lane,
            "--window-id",
            window_id,
            "--limit",
            str(limit),
            "--batch-size",
            str(batch_size),
            "--fulltext-drain-timeout-seconds",
            str(fulltext_drain_timeout_seconds),
            "--report-path",
            str(report_path),
        ]
    )
    args.dry_run = bool(dry_run)
    args.skip_fulltext_drain = bool(skip_fulltext_drain)
    exit_code = await main_async(args)
    result = json.loads(report_path.read_text(encoding="utf-8"))
    result["exit_code"] = exit_code
    result["report_path"] = str(report_path)
    return result


async def _run_inspection_once() -> dict[str, Any]:
    mongo = get_mongo_state_manager()
    if not mongo.connected:
        await mongo.connect()
    inspection = await asyncio.to_thread(_build_inspection_report_sync, mongo)
    return inspection


def _build_inspection_report_sync(mongo) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    backup = build_backup_inspection(mongo)
    expired_artifacts_visible = int(
        mongo.artifacts_collection.count_documents({"expire_at": {"$lte": now}})
    )
    return {
        "checked_at": now.isoformat(),
        "backup": backup,
        "expired_artifacts_visible": expired_artifacts_visible,
        "notes": [
            "Mongo success artifact TTL is validated by checking expired artifacts visibility.",
            "Backup freshness must be validated against the most recent backup database refresh.",
        ],
    }


def _resolve_document_target_runtime() -> dict[str, Any]:
    service = get_milvus_service()
    target = resolve_document_collection_target(
        service.client,
        logical_name=settings.resolved_document_collection_name,
        ensure_collection=lambda _name: None,
    )
    spec = build_document_index_spec()
    return {
        **target.to_dict(),
        "configured_logical_name": settings.resolved_document_collection_name,
        "current_index_signature": spec.signature,
    }


def _resolve_fulltext_target_runtime() -> dict[str, Any]:
    service = MilvusChunkService()
    try:
        target = resolve_fulltext_collection_target(
            service.client,
            logical_name=settings.resolved_fulltext_chunk_collection_name,
            ensure_collection=lambda _name: None,
        )
        spec = build_fulltext_chunk_index_spec()
        return {
            **target.to_dict(),
            "configured_logical_name": settings.resolved_fulltext_chunk_collection_name,
            "current_index_signature": spec.signature,
        }
    finally:
        service.close()


def _lane_runtime_dict(
    lane_state: OpsLaneState,
    *,
    interval_hours: float,
    startup_enabled: bool,
    window_id: str,
    batch_size: int,
    limit: int,
) -> dict[str, Any]:
    return {
        "lane": lane_state.lane,
        "running": lane_state.running,
        "startup_run_enabled": startup_enabled,
        "startup_completed": lane_state.startup_completed,
        "interval_hours": interval_hours,
        "window_id": window_id,
        "batch_size": batch_size,
        "limit": limit,
        "skip_while_running": lane_state.skip_while_running,
        "last_result": lane_state.last_result,
    }


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_report_path(prefix: str):
    from pathlib import Path

    root = Path("/app/out")
    if not root.exists():
        root = Path(__file__).resolve().parents[1] / "out"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{prefix}_{stamp}.json"
