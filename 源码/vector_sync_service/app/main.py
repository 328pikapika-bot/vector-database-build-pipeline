"""NSTL 向量库同步服务 - FastAPI 应用入口。"""
from contextlib import asynccontextmanager
import asyncio
from functools import partial
from typing import Optional

from confluent_kafka import Consumer
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import settings
from app.observability.logger import setup_logging, get_logger
from app.observability.metrics import (
    CONTENT_TYPE_LATEST,
    dependency_readiness,
    diagnostics_generation_total,
    get_metrics_response,
    worker_loop_running,
    worker_task_backlog,
)
from app.services.admin_auth import require_admin_read, require_admin_write
from app.services.admin_diagnostics import (
    build_admin_diagnostics,
    build_document_diagnostics,
    build_document_recovery_plan,
)
from app.services.auto_publisher import AutoPublishController, get_auto_publisher
from app.services.kafka_consumer import create_consumer, run_consumer
from app.services.batch_processor_existing import process_batch_existing_arch
from app.services.image_priority_pipeline import process_batch_image_priority
from app.services.compensator import get_compensator_status, run_compensator
from app.services.dashboard import build_stats_summary, render_dashboard_html
from app.services.fulltext_task_repository import get_fulltext_task_repository
from app.services.fulltext_worker import build_fulltext_runtime_status
from app.services.fulltext_worker_runner import (
    FullTextWorkerRunner,
    get_fulltext_worker_runner,
)
from app.services.image_task_repository import get_image_task_repository
from app.services.image_worker import build_image_runtime_status
from app.services.image_worker_runner import (
    ImageWorkerRunner,
    get_image_worker_runner,
)
from app.services.kafka_backpressure import get_kafka_backpressure_controller
from app.services.collection_targets import (
    resolve_document_collection_target,
    resolve_fulltext_collection_target,
)
from app.services.milvus_service import get_milvus_service, close_milvus_service
from app.services.mongo_state_manager import (
    close_mongo_state_manager,
    get_existing_mongo_state_manager,
    get_mongo_state_manager,
)
from app.services.retention_cleanup import (
    RetentionCleanupService,
    get_retention_cleanup_scheduler,
)
from app.services.embedder import get_embedding_client, close_embedding_client
from app.services.runtime_control import (
    BACKGROUND_WORKER_TYPES,
    get_background_loop_control,
    get_runtime_control,
)


setup_logging()
logger = get_logger(__name__)


class AppState:
    """应用全局状态，存放各种客户端实例。"""

    def __init__(self) -> None:
        self.milvus_connected: bool = False
        self.mongo_connected: bool = False
        self.kafka_connected: bool = False
        self.embedding_available: bool = False
        self.consumer_running: bool = False
        self.consumer: Optional[Consumer] = None
        self.consumer_task: Optional[asyncio.Task] = None
        self.compensator_running: bool = False
        self.compensator_configured: bool = settings.COMPENSATOR_ENABLED
        self.compensator_task: Optional[asyncio.Task] = None
        self.fulltext_worker_runner: Optional[FullTextWorkerRunner] = None
        self.fulltext_worker_task: Optional[asyncio.Task] = None
        self.image_worker_runner: Optional[ImageWorkerRunner] = None
        self.image_worker_task: Optional[asyncio.Task] = None
        self.image_slow_worker_runner: Optional[ImageWorkerRunner] = None
        self.image_slow_worker_task: Optional[asyncio.Task] = None
        self.auto_publisher: Optional[AutoPublishController] = None
        self.auto_publisher_task: Optional[asyncio.Task] = None
        self.retention_cleanup_task: Optional[asyncio.Task] = None


state = AppState()


class ControlPauseRequest(BaseModel):
    reason: Optional[str] = None


class KafkaBackpressureUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    mode: Optional[str] = None
    high_watermark: Optional[int] = None
    low_watermark: Optional[int] = None
    check_interval_seconds: Optional[float] = None
    consumer_time_budget_ms: Optional[int] = None
    consumer_micro_batch_size: Optional[int] = None
    image_pending_soft_limit: Optional[int] = None
    image_pending_resume_threshold: Optional[int] = None
    soft_watermark_factor: Optional[float] = None
    soft_throttle_sleep_ms: Optional[int] = None
    soft_micro_batch_min: Optional[int] = None


class RetentionCleanupRequest(BaseModel):
    mode: str = "daily"
    dry_run: bool = False


async def _initialize_vector_collections_on_startup() -> bool:
    if not settings.INIT_VECTOR_COLLECTIONS_ON_STARTUP:
        logger.info("vector_collection_init_skipped", reason="INIT_VECTOR_COLLECTIONS_ON_STARTUP=false")
        return False

    milvus = get_milvus_service()
    document_target = milvus.init_collection(settings.resolved_document_collection_name, settings.VECTOR_DIM)

    from app.services.milvus_chunk_service import MilvusChunkService
    from app.services.milvus_image_service import MilvusImageService

    chunk_service = MilvusChunkService()
    image_service = MilvusImageService()
    try:
        fulltext_target = chunk_service.init_chunk_collection(
            settings.resolved_fulltext_chunk_collection_name,
            settings.VECTOR_DIM,
        )
        image_service.init_image_collection()
    finally:
        chunk_service.close()
        image_service.close()

    logger.info(
        "vector_collections_initialized",
        document_collection=document_target["resolved_name"],
        fulltext_collection=fulltext_target["resolved_name"],
        image_collection=settings.IMAGE_COLLECTION_NAME,
    )
    return True


def _resolve_kafka_batch_processor():
    mode = str(settings.SYNC_PIPELINE_MODE or "standard").strip().lower()
    if mode == "image_priority":
        return process_batch_image_priority
    if mode == "metadata_bulk":
        return partial(process_batch_existing_arch, mode="metadata_bulk")
    return process_batch_existing_arch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("app_startup", message="Starting NSTL Vector Sync Service")

    try:
        mongo = get_mongo_state_manager()
        await mongo.connect()
        await get_fulltext_task_repository().ensure_indexes()
        await get_image_task_repository().ensure_indexes()
        await mongo.ensure_image_priority_window_indexes()
        state.mongo_connected = True
        logger.info("mongo_initialized", database=settings.MONGO_DB)
    except Exception as error:
        logger.error("mongo_init_failed", error=str(error))
        state.mongo_connected = False

    try:
        state.milvus_connected = await _initialize_vector_collections_on_startup()
    except Exception as error:
        logger.error("milvus_init_failed", error=str(error))
        state.milvus_connected = False

    try:
        embedder = get_embedding_client()
        state.embedding_available = await embedder.health_check()
        if state.embedding_available:
            logger.info("embedding_api_available", url=settings.EMBEDDING_API_URL)
        else:
            logger.warning("embedding_api_unavailable", url=settings.EMBEDDING_API_URL)
    except Exception as error:
        logger.error("embedding_init_failed", error=str(error))
        state.embedding_available = False

    try:
        compensator_enabled = bool(settings.COMPENSATOR_ENABLED)
        state.compensator_configured = compensator_enabled
        if state.mongo_connected and settings.KAFKA_CONSUMER_ENABLED:
            runtime_control = get_runtime_control()
            if settings.KAFKA_START_PAUSED:
                await runtime_control.pause("startup_paused")
            state.consumer = create_consumer()
            state.kafka_connected = True
            state.consumer_task = asyncio.create_task(
                run_consumer(
                    state.consumer,
                    _resolve_kafka_batch_processor(),
                    runtime_control=runtime_control,
                )
            )
            state.consumer_running = True
            if compensator_enabled:
                state.compensator_task = asyncio.create_task(run_compensator())
                state.compensator_running = True
            else:
                state.compensator_running = False
            logger.info("consumer_task_started", topic=settings.KAFKA_TOPIC)
            if compensator_enabled:
                logger.info("compensator_task_started")
            else:
                logger.info("compensator_task_disabled", reason="COMPENSATOR_ENABLED=false")
            if settings.FULLTEXT_WORKER_ENABLED:
                state.fulltext_worker_runner = get_fulltext_worker_runner()
                state.fulltext_worker_task = asyncio.create_task(
                    state.fulltext_worker_runner.run_loop(
                        worker_id=f"{mongo.worker_id}:fulltext",
                    )
                )
                logger.info("fulltext_worker_task_started")
            if settings.IMAGE_WORKER_ENABLED:
                state.image_worker_runner = get_image_worker_runner()
                state.image_worker_task = asyncio.create_task(
                    state.image_worker_runner.run_loop(
                        worker_id=f"{mongo.worker_id}:image",
                    )
                )
                logger.info("image_worker_task_started")
                if settings.IMAGE_DOWNLOAD_LANE_ENABLED:
                    state.image_slow_worker_runner = ImageWorkerRunner(
                        repository=get_image_task_repository(),
                        download_lane="slow",
                    )
                    state.image_slow_worker_task = asyncio.create_task(
                        state.image_slow_worker_runner.run_loop(
                            worker_id=f"{mongo.worker_id}:image-slow",
                        )
                    )
                    logger.info("image_slow_worker_task_started")
            state.auto_publisher = None
            state.auto_publisher_task = None
        elif state.mongo_connected:
            state.kafka_connected = False
            state.consumer_running = False
            if compensator_enabled:
                state.compensator_task = asyncio.create_task(run_compensator())
                state.compensator_running = True
            else:
                state.compensator_running = False
            logger.info("consumer_not_started", reason="KAFKA_CONSUMER_ENABLED=false")
            if compensator_enabled:
                logger.info("compensator_task_started")
            else:
                logger.info("compensator_task_disabled", reason="COMPENSATOR_ENABLED=false")
            if settings.FULLTEXT_WORKER_ENABLED:
                state.fulltext_worker_runner = get_fulltext_worker_runner()
                state.fulltext_worker_task = asyncio.create_task(
                    state.fulltext_worker_runner.run_loop(
                        worker_id=f"{mongo.worker_id}:fulltext",
                    )
                )
                logger.info("fulltext_worker_task_started")
            if settings.IMAGE_WORKER_ENABLED:
                state.image_worker_runner = get_image_worker_runner()
                state.image_worker_task = asyncio.create_task(
                    state.image_worker_runner.run_loop(
                        worker_id=f"{mongo.worker_id}:image",
                    )
                )
                logger.info("image_worker_task_started")
                if settings.IMAGE_DOWNLOAD_LANE_ENABLED:
                    state.image_slow_worker_runner = ImageWorkerRunner(
                        repository=get_image_task_repository(),
                        download_lane="slow",
                    )
                    state.image_slow_worker_task = asyncio.create_task(
                        state.image_slow_worker_runner.run_loop(
                            worker_id=f"{mongo.worker_id}:image-slow",
                        )
                    )
                    logger.info("image_slow_worker_task_started")
            state.auto_publisher = None
            state.auto_publisher_task = None
        else:
            state.kafka_connected = False
            state.consumer_running = False
            state.compensator_running = False
            logger.warning("consumer_not_started_without_mongo")
    except Exception as error:
        logger.error("consumer_start_failed", error=str(error))
        state.kafka_connected = False
        state.consumer_running = False
        state.compensator_running = False

    try:
        # 启动完成，打印服务地址
        _log_service_urls()
        yield
    finally:
        logger.info("app_shutdown", message="Shutting down NSTL Vector Sync Service")

        if state.fulltext_worker_task is not None:
            state.fulltext_worker_task.cancel()
            try:
                await state.fulltext_worker_task
            except asyncio.CancelledError:
                pass

        if state.image_worker_task is not None:
            state.image_worker_task.cancel()
            try:
                await state.image_worker_task
            except asyncio.CancelledError:
                pass

        if state.image_slow_worker_task is not None:
            state.image_slow_worker_task.cancel()
            try:
                await state.image_slow_worker_task
            except asyncio.CancelledError:
                pass

        if state.auto_publisher_task is not None:
            state.auto_publisher_task.cancel()
            try:
                await state.auto_publisher_task
            except asyncio.CancelledError:
                pass

        if state.compensator_task is not None:
            state.compensator_task.cancel()
            try:
                await state.compensator_task
            except asyncio.CancelledError:
                pass

        if state.consumer_task is not None:
            state.consumer_task.cancel()
            try:
                await state.consumer_task
            except asyncio.CancelledError:
                pass

        if state.retention_cleanup_task is not None:
            await get_retention_cleanup_scheduler().stop()

        if state.consumer is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(state.consumer.close),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning("consumer_close_timeout")
            except Exception as error:
                logger.warning("consumer_close_failed", error=str(error))

        state.consumer_running = False
        state.compensator_running = False
        state.fulltext_worker_task = None
        state.image_worker_task = None
        state.image_slow_worker_task = None
        state.retention_cleanup_task = None

        await close_embedding_client()
        await close_mongo_state_manager()
        close_milvus_service()
        state.milvus_connected = False
        state.mongo_connected = False
        state.embedding_available = False
        logger.info("milvus_client_closed")
        logger.info("mongo_client_closed")
        logger.info("embedding_client_closed")


app = FastAPI(
    title="NSTL Vector Sync Service",
    description="NSTL 文献数据 -> Kafka -> NLP向量化入库 -> Milvus 同步服务",
    version="0.1.0",
    lifespan=lifespan,
)

from app.services.fulltext_search import router as fulltext_search_router
app.include_router(fulltext_search_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查端点：返回各组件连通状态。"""
    runtime = await _build_runtime_status()
    mongo_connected = runtime["mongodb"]["connected"]
    milvus_connected = runtime["milvus"]["connected"]

    return {
        "status": "ok",
        "version": "0.1.0",
        "components": {
            "milvus": "connected" if milvus_connected else "disconnected",
            "mongodb": "connected" if mongo_connected else "disconnected",
            "kafka": "connected" if state.kafka_connected else "disconnected",
            "embedding_api": "available" if state.embedding_available else "unavailable",
            "consumer": "running" if runtime["consumer"]["running"] else "stopped",
            "compensator": (
                "running" if runtime["compensator"]["running"] else "stopped"
            ),
        },
        "runtime": runtime,
        "mode": "phase1-scalar-embedding-with-mongo-failover",
    }


@app.post("/admin/control/pause")
async def admin_control_pause(
    payload: Optional[ControlPauseRequest] = None,
    _auth: dict = Depends(require_admin_write),
):
    """暂停源 Kafka 消费，不再主动 poll 新消息。"""
    reason = payload.reason if payload is not None else None
    control_status = await get_runtime_control().pause(reason)
    return await _build_runtime_status(control_status=control_status)


@app.post("/admin/control/resume")
async def admin_control_resume(_auth: dict = Depends(require_admin_write)):
    """恢复源 Kafka 消费。"""
    control_status = await get_runtime_control().resume()
    return await _build_runtime_status(control_status=control_status)


@app.post("/admin/control/drain")
async def admin_control_drain(_auth: dict = Depends(require_admin_write)):
    """停止拉新并等待已进入 consumer 的批次到安全点。"""
    control_status = await get_runtime_control().drain()
    return await _build_runtime_status(control_status=control_status)


@app.get("/admin/control/status")
async def admin_control_status(_auth: dict = Depends(require_admin_read)):
    """查看运行控制和后台任务状态。"""
    return await _build_runtime_status()


@app.get("/admin/control/backpressure")
async def admin_control_backpressure_status(_auth: dict = Depends(require_admin_read)):
    """查看 Kafka 自动背压状态。"""
    return await get_kafka_backpressure_controller().status()


@app.post("/admin/control/backpressure")
async def admin_control_backpressure_update(
    payload: KafkaBackpressureUpdateRequest,
    _auth: dict = Depends(require_admin_write),
):
    """热更新 Kafka 自动背压配置。"""
    controller = get_kafka_backpressure_controller()
    return await controller.configure(
        enabled=payload.enabled,
        mode=payload.mode,
        high_watermark=payload.high_watermark,
        low_watermark=payload.low_watermark,
        check_interval_sec=payload.check_interval_seconds,
        consumer_time_budget_ms=payload.consumer_time_budget_ms,
        consumer_micro_batch_size=payload.consumer_micro_batch_size,
        image_pending_soft_limit=payload.image_pending_soft_limit,
        image_pending_resume_threshold=payload.image_pending_resume_threshold,
        soft_watermark_factor=payload.soft_watermark_factor,
        soft_throttle_sleep_ms=payload.soft_throttle_sleep_ms,
        soft_micro_batch_min=payload.soft_micro_batch_min,
    )


@app.post("/admin/control/backpressure/reset")
async def admin_control_backpressure_reset(_auth: dict = Depends(require_admin_write)):
    """清除 Kafka 自动背压的运行时覆盖配置，回到环境变量默认值。"""
    controller = get_kafka_backpressure_controller()
    await controller.reset()
    return await controller.status()


@app.get("/admin/retention/cleanup/status")
async def admin_retention_cleanup_status(_auth: dict = Depends(require_admin_read)):
    """查看保留清理后台任务状态。"""
    return get_retention_cleanup_scheduler().status()


@app.post("/admin/retention/cleanup/run")
async def admin_retention_cleanup_run(
    payload: RetentionCleanupRequest,
    _auth: dict = Depends(require_admin_write),
):
    """手动触发一次 Mongo/本地文件保留清理。"""
    return await RetentionCleanupService().run(
        mode=payload.mode,
        dry_run=payload.dry_run,
    )


@app.get("/admin/workers/{worker_type}/status")
async def admin_worker_status(
    worker_type: str,
    _auth: dict = Depends(require_admin_read),
):
    """查看后台 worker/compensator 控制状态。"""
    return await _build_worker_control_status(worker_type)


@app.post("/admin/workers/{worker_type}/pause")
async def admin_worker_pause(
    worker_type: str,
    payload: Optional[ControlPauseRequest] = None,
    _auth: dict = Depends(require_admin_write),
):
    """暂停指定后台 worker/compensator 的新 claim/scan。"""
    control = _get_worker_control_or_404(worker_type)
    await control.pause(payload.reason if payload else None)
    return await _build_worker_control_status(worker_type)


@app.post("/admin/workers/{worker_type}/resume")
async def admin_worker_resume(
    worker_type: str,
    _auth: dict = Depends(require_admin_write),
):
    """恢复指定后台 worker/compensator。"""
    control = _get_worker_control_or_404(worker_type)
    await control.resume()
    return await _build_worker_control_status(worker_type)


@app.post("/admin/workers/{worker_type}/drain")
async def admin_worker_drain(
    worker_type: str,
    _auth: dict = Depends(require_admin_write),
):
    """停止指定后台 worker/compensator 拉新并等待当前 run 边界。"""
    control = _get_worker_control_or_404(worker_type)
    await control.drain()
    return await _build_worker_control_status(worker_type)


@app.get("/admin/diagnostics")
async def admin_diagnostics(_auth: dict = Depends(require_admin_read)):
    """返回脱敏 JSON 诊断信息。"""
    try:
        diagnostics = build_admin_diagnostics(await _build_runtime_status())
    except Exception:
        diagnostics_generation_total.labels(status="failure").inc()
        raise
    diagnostics_generation_total.labels(status="success").inc()
    return diagnostics


@app.get("/admin/diagnostics/document/{doc_id}")
async def admin_document_diagnostics(
    doc_id: str,
    _auth: dict = Depends(require_admin_read),
):
    """返回单篇文献的 Mongo metadata-only 诊断摘要。"""
    try:
        diagnostics = await build_document_diagnostics(
            doc_id,
            get_existing_mongo_state_manager(),
        )
    except Exception:
        diagnostics_generation_total.labels(status="document_error").inc()
        raise
    diagnostics_generation_total.labels(
        status=_document_diagnostics_metric_status(
            diagnostics.get("diagnostic_status")
        )
    ).inc()
    return diagnostics


@app.get("/admin/diagnostics/document/{doc_id}/recovery-plan")
async def admin_document_recovery_plan(
    doc_id: str,
    _auth: dict = Depends(require_admin_read),
):
    """返回单篇文献的只读恢复/行动建议。"""
    try:
        diagnostics = await build_document_diagnostics(
            doc_id,
            get_existing_mongo_state_manager(),
        )
        recovery_plan = build_document_recovery_plan(diagnostics)
    except Exception:
        diagnostics_generation_total.labels(status="recovery_plan_error").inc()
        raise
    diagnostics_generation_total.labels(
        status=_document_recovery_plan_metric_status(
            recovery_plan.get("diagnostic_status"),
            recovery_plan.get("plan_status"),
        )
    ).inc()
    return recovery_plan


def _document_diagnostics_metric_status(status: Optional[str]) -> str:
    mapping = {
        "success": "document_success",
        "not_found": "document_not_found",
        "mongo_unavailable": "document_mongo_unavailable",
        "error": "document_error",
    }
    return mapping.get(str(status or ""), "document_error")


def _document_recovery_plan_metric_status(
    diagnostic_status: Optional[str],
    plan_status: Optional[str],
) -> str:
    if diagnostic_status == "mongo_unavailable" or plan_status == "mongo_unavailable":
        return "recovery_plan_mongo_unavailable"
    if diagnostic_status == "not_found" or plan_status == "no_status":
        return "recovery_plan_not_found"
    if plan_status in {"no_action_required", "action_recommended"}:
        return "recovery_plan_success"
    return "recovery_plan_error"


def _get_worker_control_or_404(worker_type: str):
    if worker_type not in BACKGROUND_WORKER_TYPES:
        raise HTTPException(status_code=404, detail="unsupported worker_type")
    return get_background_loop_control(worker_type)


async def _build_worker_control_status(worker_type: str) -> dict:
    control = _get_worker_control_or_404(worker_type)
    control_status = await control.status()
    mongo_connected = state.mongo_connected
    if worker_type == "fulltext":
        worker_status = await _build_fulltext_status(mongo_connected)
    elif worker_type == "image":
        worker_status = await _build_image_status(mongo_connected)
    else:
        compensator_status = get_compensator_status()
        worker_status = {
            **compensator_status,
            "running": compensator_status["running"]
            or _task_running(state.compensator_task),
            "task_running": _task_running(state.compensator_task),
            "configured_running": state.compensator_running,
            "configured_enabled": state.compensator_configured,
        }
    return {
        "worker_type": worker_type,
        "control": control_status,
        "status": worker_status,
    }


async def _build_runtime_status(control_status: Optional[dict] = None) -> dict:
    if control_status is None:
        control_status = await get_runtime_control().status()
    backpressure_status = await get_kafka_backpressure_controller().status()

    mongo_connected = state.mongo_connected
    mongo_worker_id = None
    try:
        mongo = get_mongo_state_manager()
        mongo_connected = mongo.connected
        mongo_worker_id = mongo.worker_id
        state.mongo_connected = mongo_connected
    except Exception:
        mongo_connected = False
        state.mongo_connected = False

    compensator_status = get_compensator_status()
    compensator_task_running = _task_running(state.compensator_task)
    milvus_connected = await _probe_milvus_readiness()
    fulltext_status = await _build_fulltext_status(mongo_connected)
    image_status = await _build_image_status(mongo_connected)

    runtime = {
        "control": control_status,
        "consumer": {
            "running": _task_running(state.consumer_task),
            "configured_running": state.consumer_running,
            "kafka_connected": state.kafka_connected,
            "task_done": state.consumer_task.done() if state.consumer_task else None,
            "topic": settings.KAFKA_TOPIC,
            "group_id": settings.KAFKA_GROUP_ID,
            "enabled": settings.KAFKA_CONSUMER_ENABLED,
            "consume_limit": settings.KAFKA_CONSUME_LIMIT,
            "backpressure": backpressure_status,
        },
        "compensator": {
            **compensator_status,
            "running": compensator_status["running"] or compensator_task_running,
            "task_running": compensator_task_running,
            "configured_running": state.compensator_running,
            "configured_enabled": state.compensator_configured,
        },
        "mongodb": {
            "connected": mongo_connected,
            "database": settings.MONGO_DB,
            "worker_id": mongo_worker_id,
        },
        "milvus": {
            "connected": milvus_connected,
            "collection": settings.resolved_document_collection_name,
        },
        "embedding_api": {
            "available": state.embedding_available,
            "url": settings.EMBEDDING_API_URL,
        },
        "retention_cleanup": get_retention_cleanup_scheduler().status(),
        "fulltext": fulltext_status,
        "image": image_status,
    }
    _update_runtime_metrics(runtime)
    return runtime


async def _probe_milvus_readiness() -> bool:
    try:
        return await asyncio.to_thread(_probe_milvus_readiness_sync)
    except Exception as error:
        logger.warning("milvus_readiness_probe_failed", error=str(error))
        state.milvus_connected = False
        return False


def _probe_milvus_readiness_sync() -> bool:
    service = get_milvus_service()
    target = resolve_document_collection_target(
        service.client,
        logical_name=settings.resolved_document_collection_name,
        ensure_collection=lambda _name: None,
    )
    connected = bool(service.client.has_collection(target.resolved_name))
    state.milvus_connected = connected
    return connected


def _task_running(task: Optional[asyncio.Task]) -> bool:
    return task is not None and not task.done()


def _start_retention_cleanup_scheduler() -> None:
    if not settings.RETENTION_CLEANUP_ENABLED:
        return
    scheduler = get_retention_cleanup_scheduler()
    scheduler.start()
    state.retention_cleanup_task = scheduler._task
    logger.info("retention_cleanup_scheduler_started")


def _update_runtime_metrics(runtime: dict) -> None:
    dependency_readiness.labels(dependency="mongodb").set(
        1.0 if runtime["mongodb"]["connected"] else 0.0
    )
    dependency_readiness.labels(dependency="milvus").set(
        1.0 if runtime["milvus"]["connected"] else 0.0
    )
    dependency_readiness.labels(dependency="embedding_api").set(
        1.0 if runtime["embedding_api"]["available"] else 0.0
    )
    dependency_readiness.labels(dependency="fulltext_download_base").set(
        1.0 if runtime["fulltext"].get("base_url_configured") else 0.0
    )
    dependency_readiness.labels(dependency="image_download_base").set(
        1.0 if runtime["image"].get("base_url_configured") else 0.0
    )

    _update_worker_metrics("fulltext", runtime["fulltext"])
    _update_worker_metrics("image", runtime["image"])
    compensator = runtime["compensator"]
    worker_loop_running.labels(worker_type="compensator").set(
        1.0 if compensator.get("running") else 0.0
    )


def _update_worker_metrics(worker_type: str, status: dict) -> None:
    worker_loop_running.labels(worker_type=worker_type).set(
        1.0 if status.get("worker_running") else 0.0
    )
    task_summary = status.get("task_summary") or {}
    counts = task_summary.get("counts") or {}
    for task_status in (
        "PENDING",
        "LEASED",
        "SUCCESS",
        "FAILURE",
        "SKIPPED",
        "HELD_HISTORY",
    ):
        worker_task_backlog.labels(
            worker_type=worker_type,
            status=task_status.lower(),
        ).set(float(counts.get(task_status, 0)))


async def _build_fulltext_status(mongo_connected: bool) -> dict:
    task_summary = None
    if mongo_connected:
        try:
            task_summary = await get_fulltext_task_repository().get_summary()
        except Exception as error:
            task_summary = {"error": str(error)}

    runner_status = None
    if state.fulltext_worker_runner is not None:
        runner_status = state.fulltext_worker_runner.status()
        runner_status["running"] = runner_status["running"] or _task_running(
            state.fulltext_worker_task
        )
    else:
        control_status = get_background_loop_control("fulltext").snapshot()
        runner_status = {
            "enabled": settings.FULLTEXT_WORKER_ENABLED,
            "running": _task_running(state.fulltext_worker_task),
            "last_run_at": None,
            "last_task_id": None,
            "last_processed_count": 0,
            "last_error": None,
            "poll_interval_seconds": settings.FULLTEXT_WORKER_POLL_INTERVAL_SEC,
            "paused": control_status["paused"],
            "draining": control_status["draining"],
            "pause_reason": control_status["pause_reason"],
            "inflight_runs": control_status["inflight_runs"],
        }

    return build_fulltext_runtime_status(
        worker_running=_task_running(state.fulltext_worker_task),
        worker_status=runner_status,
        task_summary=task_summary,
    )


async def _build_image_status(mongo_connected: bool) -> dict:
    task_summary = None
    if mongo_connected:
        try:
            task_summary = await get_image_task_repository().get_summary()
        except Exception as error:
            task_summary = {"error": str(error)}

    runner_status = None
    if state.image_worker_runner is not None:
        runner_status = state.image_worker_runner.status()
        runner_status["running"] = runner_status["running"] or _task_running(
            state.image_worker_task
        )
    else:
        control_status = get_background_loop_control("image").snapshot()
        runner_status = {
            "enabled": settings.IMAGE_WORKER_ENABLED,
            "running": _task_running(state.image_worker_task),
            "last_run_at": None,
            "last_task_id": None,
            "last_processed_count": 0,
            "last_error": None,
            "poll_interval_seconds": settings.IMAGE_WORKER_POLL_INTERVAL_SEC,
            "paused": control_status["paused"],
            "draining": control_status["draining"],
            "pause_reason": control_status["pause_reason"],
            "inflight_runs": control_status["inflight_runs"],
        }

    return build_image_runtime_status(
        worker_running=_task_running(state.image_worker_task),
        worker_status={
            **runner_status,
            "slow_lane": (
                state.image_slow_worker_runner.status()
                if state.image_slow_worker_runner is not None
                else None
            ),
        },
        task_summary=task_summary,
    )


def _log_service_urls() -> None:
    """启动完成时打印各访问地址，方便在 Portainer 日志中直接看到。"""
    port = 8700
    lines = [
        "=" * 60,
        "  NSTL Vector Sync Service — 启动完成",
        "=" * 60,
        f"  Dashboard  : http://<host>:{port}/",
        f"  Health     : http://<host>:{port}/health",
        f"  Metrics    : http://<host>:{port}/metrics",
        f"  API Docs   : http://<host>:{port}/docs",
        f"  Stats JSON : http://<host>:{port}/api/stats/summary",
        "=" * 60,
        f"  Mongo      : {state.mongo_connected}",
        f"  Milvus     : {state.milvus_connected}",
        f"  Kafka      : {state.kafka_connected}",
        f"  Embedding  : {state.embedding_available}",
        f"  Consumer   : {state.consumer_running}",
        f"  Compensator: {state.compensator_running}",
        f"  Auto Pub   : {state.auto_publisher is not None and state.auto_publisher_task is not None}",
        "=" * 60,
    ]
    for line in lines:
        logger.info("service_url", line=line)


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点。"""
    return Response(content=get_metrics_response(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", response_class=HTMLResponse)
async def root():
    """监控仪表盘。"""
    return render_dashboard_html()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """监控仪表盘。"""
    return render_dashboard_html()


@app.get("/api/stats/summary")
async def stats_summary():
    """仪表盘数据接口 —— 查 MongoDB 状态统计。"""
    mongo = get_existing_mongo_state_manager()
    publisher = state.auto_publisher
    return await build_stats_summary(
        app_state=state,
        mongo_manager=mongo,
        auto_publisher=publisher,
    )
