"""自适应背压自动投递引擎。

持续投递 Kafka 数据 → 等待消费端处理 → 继续投递下一批，直到全部跑完。
支持断点续传、运行时暂停/恢复控制。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from confluent_kafka import Consumer, KafkaException, TopicPartition

from app.config import settings
from app.observability.logger import get_logger
from app.services.runtime_control import (
    RuntimeControl,
    get_runtime_control,
)
from app.services.task_backpressure import count_actionable_backlog

logger = get_logger(__name__)

RUNTIME_DIR = Path(__file__).resolve().parents[2] / ".runtime"
DEFAULT_CHECKPOINT = RUNTIME_DIR / "auto_publish_checkpoint.json"


@dataclass(frozen=True)
class PublishPageResult:
    page: int
    status_code: int
    body_excerpt: str
    elapsed_seconds: float
    success_count: Optional[int] = None
    current_page_size: Optional[int] = None
    fail_count: Optional[int] = None
    total_count: Optional[int] = None
    kafka_records: Optional[int] = None


class AutoPublishController:
    """自适应背压自动投递控制器。

    启动后循环执行：投递一批页面 → 等 MongoDB pending 降到阈值 → 投递下一批。
    通过 RuntimeControl 支持暂停/恢复/排空。
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        token: str | None = None,
        sub_doc_type: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        pages_per_round: int | None = None,
        pending_threshold: int | None = None,
        check_interval_sec: int | None = None,
        timeout_sec: int | None = None,
        interval_sec: float | None = None,
        checkpoint_path: Path | None = None,
        runtime_control: RuntimeControl | None = None,
        sync_health_url: str | None = None,
    ) -> None:
        self._url = url or settings.PUBLISH_BATCH_URL
        self._token = token or settings.PUBLISH_BATCH_TOKEN
        self._sub_doc_type = sub_doc_type or settings.PUBLISH_BATCH_SUB_DOC_TYPE
        self._topic = settings.PUBLISH_BATCH_TOPIC
        self._start_page = start_page if start_page is not None else settings.PUBLISH_BATCH_START_PAGE
        self._end_page = end_page if end_page is not None else settings.PUBLISH_BATCH_END_PAGE
        self._pages_per_round = pages_per_round or settings.PUBLISH_BATCH_PAGES_PER_ROUND
        self._pending_threshold = pending_threshold or settings.PUBLISH_BATCH_PENDING_THRESHOLD
        self._check_interval_sec = check_interval_sec or settings.PUBLISH_BATCH_CHECK_INTERVAL_SEC
        self._timeout_sec = timeout_sec or settings.PUBLISH_BATCH_TIMEOUT_SEC
        self._interval_sec = interval_sec if interval_sec is not None else settings.PUBLISH_BATCH_INTERVAL_SEC
        self._checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self._runtime_control = runtime_control or get_runtime_control()
        self._sync_health_url = sync_health_url or settings.OPS_SYNC_HEALTH_URL

        self._current_page: int = 0
        self._total_published: int = 0
        self._total_failed: int = 0
        self._running: bool = False
        self._finished: bool = False
        self._last_error: Optional[str] = None
        self._last_daily_run: Optional[dict[str, Any]] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._paused: bool = False
        self._pause_reason: Optional[str] = None

    # ── public status ──────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running": self._running,
            "finished": self._finished,
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "current_page": self._current_page,
            "start_page": self._start_page,
            "end_page": self._end_page,
            "total_published": self._total_published,
            "total_failed": self._total_failed,
            "pending_threshold": self._pending_threshold,
            "daily_record_limit": settings.PUBLISH_BATCH_DAILY_RECORD_LIMIT,
            "page_record_count": settings.PUBLISH_BATCH_PAGE_RECORD_COUNT,
            "count_mode": settings.PUBLISH_BATCH_COUNT_MODE,
            "last_daily_run": self._last_daily_run,
            "last_error": self._last_error,
            "sync_health_url": self._sync_health_url,
            "publish_topic": self._topic,
        }

    def pause(self, reason: str | None = None) -> dict:
        self._paused = True
        self._pause_reason = reason or "manual_pause"
        return self.status()

    def resume(self) -> dict:
        self._paused = False
        self._pause_reason = None
        return self.status()

    # ── main loop ──────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """主投递循环，作为 asyncio task 在 lifespan 中启动。"""
        logger.info(
            "auto_publisher_starting",
            url=self._url,
            sub_doc_type=self._sub_doc_type,
            start_page=self._start_page,
            end_page=self._end_page,
            pages_per_round=self._pages_per_round,
            pending_threshold=self._pending_threshold,
        )

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_sec),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._running = True

        try:
            await self._run()
        except asyncio.CancelledError:
            logger.info("auto_publisher_cancelled", page=self._current_page)
        except Exception:
            logger.exception("auto_publisher_fatal_error")
        finally:
            self._running = False
            if self._http is not None:
                await self._http.aclose()
                self._http = None

    async def run_daily_once(
        self,
        *,
        record_limit: int | None = None,
        run_reason: str = "scheduled",
    ) -> dict[str, Any]:
        """Publish one quota and then stop.

        `record_limit` lets ops seed Kafka before the scheduled midnight run
        without changing the daily Stack default.
        """
        started = datetime.now(timezone.utc)
        resolved_record_limit = (
            int(record_limit)
            if record_limit is not None and int(record_limit) > 0
            else int(settings.PUBLISH_BATCH_DAILY_RECORD_LIMIT or 0)
        )
        resolved_record_limit = max(resolved_record_limit, 0)
        page_record_count = max(int(settings.PUBLISH_BATCH_PAGE_RECORD_COUNT or 1), 1)
        count_mode = _normalize_count_mode(settings.PUBLISH_BATCH_COUNT_MODE)
        summary: dict[str, Any] = {
            "run_reason": run_reason,
            "started_at": started.isoformat(),
            "finished_at": None,
            "record_limit": resolved_record_limit,
            "page_record_count": page_record_count,
            "count_mode": count_mode,
            "start_page": None,
            "last_success_page": None,
            "attempted_pages": 0,
            "successful_pages": 0,
            "failed_pages": 0,
            "published_records": 0,
            "reported_success_records": 0,
            "reported_page_records": 0,
            "fallback_record_pages": 0,
            "kafka_record_pages": 0,
            "kafka_records": 0,
            "max_pages_per_run": max(int(settings.PUBLISH_BATCH_MAX_PAGES_PER_RUN or 0), 0),
            "stopped_reason": None,
            "errors": [],
        }
        if resolved_record_limit <= 0:
            summary["stopped_reason"] = "disabled_record_limit"
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._last_daily_run = summary
            return summary

        previous_http = self._http
        created_http = previous_http is None
        if created_http:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_sec),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
        self._running = True
        try:
            self._current_page = self._load_checkpoint()
            summary["start_page"] = self._current_page + 1
            while int(summary["published_records"]) < resolved_record_limit:
                max_pages = int(summary["max_pages_per_run"])
                if max_pages > 0 and int(summary["attempted_pages"]) >= max_pages:
                    summary["stopped_reason"] = "max_pages_reached"
                    break
                control = await self._runtime_control.status()
                if control.get("draining") or control.get("paused") or self._paused:
                    summary["stopped_reason"] = "paused_or_draining"
                    break
                drained = await self._wait_for_drain()
                if not drained:
                    summary["stopped_reason"] = "drain_stopped"
                    break

                page = self._current_page + 1
                if self._end_page > 0 and page > self._end_page:
                    summary["stopped_reason"] = "end_page_reached"
                    break
                try:
                    before_offsets = self._read_topic_end_offsets() if count_mode == "kafka_offset" else None
                    result = await self._publish_page(page)
                    if before_offsets is not None:
                        after_offsets = self._read_topic_end_offsets()
                        result = _with_kafka_records(
                            result,
                            _offset_delta(before_offsets, after_offsets),
                        )
                except Exception as error:  # noqa: BLE001 - report and stop this daily run
                    self._last_error = f"page {page} publish failed"
                    self._total_failed += 1
                    summary["attempted_pages"] += 1
                    summary["failed_pages"] += 1
                    summary["errors"].append(f"{type(error).__name__}: {error}")
                    summary["stopped_reason"] = "page_failed"
                    logger.exception("auto_publisher_daily_page_failed", page=page)
                    break

                if result is None:
                    summary["stopped_reason"] = "publish_returned_none"
                    break

                self._current_page = page
                self._total_published += 1
                self._save_checkpoint(page)
                records = _published_record_count(result, page_record_count, mode=count_mode)
                summary["attempted_pages"] += 1
                summary["successful_pages"] += 1
                summary["published_records"] += records
                summary["last_success_page"] = page
                success_count = getattr(result, "success_count", None)
                current_page_size = getattr(result, "current_page_size", None)
                kafka_records = getattr(result, "kafka_records", None)
                if success_count is not None:
                    summary["reported_success_records"] += max(int(success_count), 0)
                if current_page_size is not None:
                    summary["reported_page_records"] += max(int(current_page_size), 0)
                if kafka_records is not None:
                    summary["kafka_record_pages"] += 1
                    summary["kafka_records"] += max(int(kafka_records), 0)
                if success_count is None and current_page_size is None:
                    summary["fallback_record_pages"] += 1
                logger.info(
                    "auto_publisher_daily_page_ok",
                    page=page,
                    records=records,
                    kafka_records=kafka_records,
                    published_records=summary["published_records"],
                    record_limit=resolved_record_limit,
                )
                if self._interval_sec > 0:
                    await asyncio.sleep(self._interval_sec)

            if summary["stopped_reason"] is None:
                summary["stopped_reason"] = "record_limit_reached"
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._last_daily_run = summary
            return summary
        finally:
            self._running = False
            if created_http and self._http is not None:
                await self._http.aclose()
                self._http = previous_http

    async def _run(self) -> None:
        self._current_page = self._load_checkpoint()

        while True:
            control = await self._runtime_control.status()
            if control.get("draining"):
                logger.info("auto_publisher_drain_complete", page=self._current_page)
                self._finished = True
                return
            if control.get("paused") or self._paused:
                await asyncio.sleep(5)
                continue

            if self._end_page > 0 and self._current_page >= self._end_page:
                logger.info("auto_publisher_all_pages_done", last_page=self._current_page)
                self._finished = True
                return

            # 等待消费端追上
            drained = await self._wait_for_drain()
            if not drained:
                continue

            # 投递一轮
            start = self._current_page + 1
            count = self._pages_per_round
            if self._end_page > 0:
                remaining = self._end_page - self._current_page
                count = min(count, remaining)

            logger.info("auto_publisher_round_start", start_page=start, page_count=count)
            last_success = await self._publish_round(start, count)

            if last_success is not None:
                self._current_page = last_success
                self._save_checkpoint(last_success)

            if self._end_page > 0 and self._current_page >= self._end_page:
                logger.info("auto_publisher_all_pages_done", last_page=self._current_page)
                self._finished = True
                return

    # ── publish round ──────────────────────────────────────────────

    async def _publish_round(self, start_page: int, count: int) -> Optional[int]:
        """投递一轮页面，返回最后一页的页码（成功）或 None（全部失败）。"""
        last_success: Optional[int] = None
        end = start_page + count

        for page in range(start_page, end):
            control = await self._runtime_control.status()
            if control.get("draining") or control.get("paused") or self._paused:
                break

            try:
                result = await self._publish_page(page)
            except Exception:
                logger.exception("auto_publisher_page_failed", page=page)
                self._last_error = f"page {page} publish failed"
                self._total_failed += 1
                continue

            if result is not None:
                last_success = page
                self._current_page = page
                self._total_published += 1
                self._save_checkpoint(last_success)
                logger.info(
                    "auto_publisher_page_ok",
                    page=page,
                    status=result.status_code,
                    elapsed=round(result.elapsed_seconds, 1),
                )

            if self._interval_sec > 0 and page < end - 1:
                await asyncio.sleep(self._interval_sec)

        return last_success

    async def _publish_page(self, page: int) -> Optional[PublishPageResult]:
        """投递单个页面。"""
        if self._http is None:
            return None

        form_data = {
            "subDocType": self._sub_doc_type,
            "pageNumer": str(page),
        }
        if self._topic:
            form_data["topic"] = self._topic
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "vector-sync-auto-publish/1.0",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        started = time.monotonic()
        try:
            response = await self._http.post(
                self._url,
                content=_encode_form(form_data),
                headers=headers,
            )
        except httpx.ReadTimeout:
            elapsed = time.monotonic() - started
            if not settings.PUBLISH_BATCH_ASSUME_TIMEOUT_SUCCESS:
                raise
            logger.warning(
                "auto_publisher_page_timeout_assumed_accepted",
                page=page,
                elapsed=round(elapsed, 1),
                timeout_seconds=self._timeout_sec,
            )
            return PublishPageResult(
                page=page,
                status_code=202,
                body_excerpt="read_timeout_assumed_accepted",
                elapsed_seconds=elapsed,
            )
        elapsed = time.monotonic() - started

        body = response.read()
        excerpt = _decode_body_excerpt(body)

        if not response.is_success:
            raise RuntimeError(f"HTTP {response.status_code}: {excerpt}")

        _ensure_successful_body(body, excerpt)
        metrics = parse_response_metrics(body)

        return PublishPageResult(
            page=page,
            status_code=response.status_code,
            body_excerpt=excerpt,
            elapsed_seconds=elapsed,
            success_count=_coerce_int(metrics.get("success_count")),
            current_page_size=_coerce_int(metrics.get("current_page_size")),
            fail_count=_coerce_int(metrics.get("fail_count")),
            total_count=_coerce_int(metrics.get("total_count")),
        )

    def _read_topic_end_offsets(self) -> dict[int, int]:
        topic = self._topic or settings.KAFKA_TOPIC
        return read_kafka_topic_end_offsets(topic)

    # ── drain wait ─────────────────────────────────────────────────

    async def _wait_for_drain(self) -> bool:
        """轮询 MongoDB，等 pending 数量降到阈值以下。返回 True 表示可以继续投递。"""
        from app.services.mongo_state_manager import (
            get_existing_mongo_state_manager,
            get_mongo_state_manager,
        )

        while True:
            control = await self._runtime_control.status()
            if control.get("draining"):
                return False
            if control.get("paused") or self._paused:
                await asyncio.sleep(5)
                continue

            if not await self._sync_service_healthy():
                self._last_error = "sync service unhealthy"
                await asyncio.sleep(self._check_interval_sec)
                continue

            try:
                mongo = get_existing_mongo_state_manager()
                if mongo is None or not mongo.connected:
                    mongo = get_mongo_state_manager()
                    if not mongo.connected:
                        await mongo.connect()

                pending_count, backlog_detail = _count_actionable_backlog(mongo)

                logger.info(
                    "auto_publisher_drain_check",
                    pending_count=pending_count,
                    backlog_detail=backlog_detail,
                    threshold=self._pending_threshold,
                    current_page=self._current_page,
                )

                if pending_count <= self._pending_threshold:
                    return True

            except Exception:
                logger.exception("auto_publisher_drain_check_error")

            await asyncio.sleep(self._check_interval_sec)

    async def _sync_service_healthy(self) -> bool:
        if not self._sync_health_url:
            return True
        if self._http is None:
            return False
        try:
            response = await self._http.get(self._sync_health_url)
        except Exception:
            logger.warning("auto_publisher_sync_health_unreachable", url=self._sync_health_url)
            return False
        if not response.is_success:
            logger.warning(
                "auto_publisher_sync_health_bad_status",
                url=self._sync_health_url,
                status_code=response.status_code,
            )
            return False
        return True

    # ── checkpoint ─────────────────────────────────────────────────

    def _load_checkpoint(self) -> int:
        if not self._checkpoint_path.exists():
            return self._start_page - 1

        try:
            data = json.loads(self._checkpoint_path.read_text("utf-8"))
            page = int(data.get("last_success_page", self._start_page - 1))
            logger.info("auto_publisher_checkpoint_loaded", last_page=page)
            return page
        except Exception:
            logger.warning("auto_publisher_checkpoint_load_failed")
            return self._start_page - 1

    def _save_checkpoint(self, page: int) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._checkpoint_path.with_suffix(".tmp")
        data = {
            "last_success_page": page,
            "sub_doc_type": self._sub_doc_type,
            "url": self._url,
        }
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self._checkpoint_path)


# ── helpers (shared with publish_batch_pages.py) ──────────────────

def _encode_form(data: dict[str, str]) -> str:
    from urllib.parse import urlencode
    return urlencode(data)


def _decode_body_excerpt(body: bytes, limit: int = 500) -> str:
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")
    return text[:limit]


def _ensure_successful_body(body: bytes, excerpt: str) -> None:
    if not body:
        return
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if isinstance(parsed, dict) and parsed.get("success") is False:
        raise RuntimeError(f"Response success=false: {excerpt}")
    if isinstance(parsed, dict) and str(parsed.get("code", "")).lower() in {"500", "-1", "error"}:
        raise RuntimeError(f"Response code indicates failure: {excerpt}")


def parse_response_metrics(body: bytes) -> dict[str, Any]:
    parsed = _load_response_json(body)
    if not isinstance(parsed, dict):
        return {}
    payload = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
    return {
        "current_page_size": _coerce_int(_first_present(payload, "currentPageSize", "pageSize", "size")),
        "success_count": _coerce_int(_first_present(payload, "successCount")),
        "fail_count": _coerce_int(_first_present(payload, "failCount", "failureCount", "errorCount")),
        "total_count": _coerce_int(_first_present(payload, "totalCount", "count")),
    }


def _published_record_count(
    result: PublishPageResult,
    fallback_page_count: int,
    *,
    mode: str = "response_or_fallback",
) -> int:
    kafka_records = getattr(result, "kafka_records", None)
    if mode == "kafka_offset" and kafka_records is not None:
        return max(int(kafka_records), 0)
    success_count = getattr(result, "success_count", None)
    current_page_size = getattr(result, "current_page_size", None)
    if success_count is not None:
        return max(int(success_count), 0)
    if current_page_size is not None:
        return max(int(current_page_size), 0)
    return max(int(fallback_page_count), 1)


def _normalize_count_mode(value: str | None) -> str:
    mode = str(value or "response_or_fallback").strip().lower()
    if mode in {"kafka", "kafka_offsets", "kafka_offset", "topic_offset"}:
        return "kafka_offset"
    return "response_or_fallback"


def _offset_delta(before: dict[int, int], after: dict[int, int]) -> int:
    total = 0
    for partition, after_offset in after.items():
        before_offset = before.get(partition, after_offset)
        total += max(int(after_offset) - int(before_offset), 0)
    return total


def _with_kafka_records(
    result: Optional[PublishPageResult],
    kafka_records: int,
) -> Optional[PublishPageResult]:
    if result is None:
        return None
    return PublishPageResult(
        page=result.page,
        status_code=result.status_code,
        body_excerpt=getattr(result, "body_excerpt", ""),
        elapsed_seconds=getattr(result, "elapsed_seconds", 0.0),
        success_count=getattr(result, "success_count", None),
        current_page_size=getattr(result, "current_page_size", None),
        fail_count=getattr(result, "fail_count", None),
        total_count=getattr(result, "total_count", None),
        kafka_records=max(int(kafka_records), 0),
    )


def read_kafka_topic_end_offsets(topic: str) -> dict[int, int]:
    """Read Kafka end offsets without joining the sync consumer group."""
    if not topic:
        return {}

    config: dict[str, object] = {
        "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": f"vector-sync-publisher-offset-probe-{int(time.time() * 1000)}",
        "enable.auto.commit": False,
        "enable.partition.eof": False,
        "session.timeout.ms": int(settings.KAFKA_SESSION_TIMEOUT_MS),
    }
    if settings.KAFKA_SECURITY_PROTOCOL in ("SASL_SSL", "SASL_PLAINTEXT"):
        config["security.protocol"] = settings.KAFKA_SECURITY_PROTOCOL
        config["sasl.mechanism"] = settings.KAFKA_SASL_MECHANISM or "PLAIN"
        if settings.KAFKA_SASL_USERNAME:
            config["sasl.username"] = settings.KAFKA_SASL_USERNAME
        if settings.KAFKA_SASL_PASSWORD:
            config["sasl.password"] = settings.KAFKA_SASL_PASSWORD
    else:
        config["security.protocol"] = "PLAINTEXT"

    consumer = Consumer(config)
    try:
        metadata = consumer.list_topics(
            topic=topic,
            timeout=float(settings.PUBLISH_BATCH_KAFKA_OFFSET_TIMEOUT_SEC),
        )
        topic_meta = metadata.topics.get(topic)
        if topic_meta is None:
            raise RuntimeError(f"Kafka topic not found: {topic}")
        if topic_meta.error is not None:
            raise KafkaException(topic_meta.error)

        offsets: dict[int, int] = {}
        timeout = float(settings.PUBLISH_BATCH_KAFKA_OFFSET_TIMEOUT_SEC)
        for partition in sorted(topic_meta.partitions):
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, partition),
                timeout=timeout,
                cached=False,
            )
            offsets[int(partition)] = int(high)
        return offsets
    finally:
        consumer.close()


def _load_response_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _first_present(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _coerce_int(value: Any) -> int | None:
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


def _count_actionable_backlog(mongo) -> tuple[int, dict[str, int]]:
    return count_actionable_backlog(
        mongo,
        mode=getattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "all_tasks"),
    )


# ── singleton ─────────────────────────────────────────────────────

_auto_publisher: Optional[AutoPublishController] = None


def get_auto_publisher() -> AutoPublishController:
    global _auto_publisher
    if _auto_publisher is None:
        _auto_publisher = AutoPublishController()
    return _auto_publisher
