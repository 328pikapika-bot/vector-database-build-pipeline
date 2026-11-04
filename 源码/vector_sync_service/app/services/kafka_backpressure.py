"""Automatic Kafka intake backpressure based on downstream task backlog."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import settings
from app.observability.logger import get_logger
from app.services.mongo_state_manager import get_existing_mongo_state_manager
from app.services.task_backpressure import count_actionable_backlog

logger = get_logger(__name__)


class KafkaBackpressureController:
    def __init__(
        self,
        *,
        manager_getter: Optional[Callable[[], Any]] = None,
        enabled: Optional[bool] = None,
        high_watermark: Optional[int] = None,
        low_watermark: Optional[int] = None,
        check_interval_sec: Optional[float] = None,
        mode: Optional[str] = None,
        consumer_time_budget_ms: Optional[int] = None,
        consumer_micro_batch_size: Optional[int] = None,
        image_pending_soft_limit: Optional[int] = None,
        image_pending_resume_threshold: Optional[int] = None,
        soft_watermark_factor: Optional[float] = None,
        soft_throttle_sleep_ms: Optional[int] = None,
        soft_micro_batch_min: Optional[int] = None,
    ) -> None:
        self._manager_getter = manager_getter or get_existing_mongo_state_manager
        self._enabled_override = enabled
        self._high_override = high_watermark
        self._low_override = low_watermark
        self._check_interval_override = check_interval_sec
        self._mode_override = mode
        self._consumer_time_budget_override = consumer_time_budget_ms
        self._consumer_micro_batch_size_override = consumer_micro_batch_size
        self._image_pending_soft_limit_override = image_pending_soft_limit
        self._image_pending_resume_threshold_override = image_pending_resume_threshold
        self._soft_watermark_factor_override = soft_watermark_factor
        self._soft_throttle_sleep_ms_override = soft_throttle_sleep_ms
        self._soft_micro_batch_min_override = soft_micro_batch_min
        self._lock = asyncio.Lock()
        self._active = False
        self._soft_active = False
        self._lead_limit_active = False
        self._last_backlog = 0
        self._last_detail: dict[str, Any] = {}
        self._last_checked_at: Optional[str] = None
        self._last_transition_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_check_monotonic = 0.0

    async def should_throttle(self) -> bool:
        if not self._enabled():
            async with self._lock:
                if self._active:
                    self._set_active_unlocked(False, reason="disabled")
                self._soft_active = False
                self._lead_limit_active = False
                return False

        async with self._lock:
            return await self._evaluate_unlocked(force=False)

    async def wait_for_capacity(self) -> None:
        if not self._enabled():
            return

        sleep_seconds = max(float(self._check_interval_sec()), 0.2)
        while True:
            throttled = await self.should_throttle()
            if not throttled:
                return
            await asyncio.sleep(sleep_seconds)

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            return self._status_unlocked()

    async def configure(
        self,
        *,
        enabled: Optional[bool] = None,
        high_watermark: Optional[int] = None,
        low_watermark: Optional[int] = None,
        check_interval_sec: Optional[float] = None,
        mode: Optional[str] = None,
        consumer_time_budget_ms: Optional[int] = None,
        consumer_micro_batch_size: Optional[int] = None,
        image_pending_soft_limit: Optional[int] = None,
        image_pending_resume_threshold: Optional[int] = None,
        soft_watermark_factor: Optional[float] = None,
        soft_throttle_sleep_ms: Optional[int] = None,
        soft_micro_batch_min: Optional[int] = None,
        force_recheck: bool = True,
    ) -> dict[str, Any]:
        async with self._lock:
            if enabled is not None:
                self._enabled_override = bool(enabled)
            if high_watermark is not None:
                self._high_override = max(int(high_watermark), 0)
            if low_watermark is not None:
                self._low_override = max(int(low_watermark), 0)
            if check_interval_sec is not None:
                self._check_interval_override = max(float(check_interval_sec), 0.2)
            if mode is not None:
                self._mode_override = str(mode).strip().lower()
            if consumer_time_budget_ms is not None:
                self._consumer_time_budget_override = max(int(consumer_time_budget_ms), 0)
            if consumer_micro_batch_size is not None:
                self._consumer_micro_batch_size_override = max(int(consumer_micro_batch_size), 1)
            if image_pending_soft_limit is not None:
                self._image_pending_soft_limit_override = max(int(image_pending_soft_limit), 0)
            if image_pending_resume_threshold is not None:
                self._image_pending_resume_threshold_override = max(
                    int(image_pending_resume_threshold),
                    0,
                )
            if soft_watermark_factor is not None:
                self._soft_watermark_factor_override = float(soft_watermark_factor)
            if soft_throttle_sleep_ms is not None:
                self._soft_throttle_sleep_ms_override = max(int(soft_throttle_sleep_ms), 0)
            if soft_micro_batch_min is not None:
                self._soft_micro_batch_min_override = max(int(soft_micro_batch_min), 1)

            if self._low_override is not None and self._high_override is not None:
                self._low_override = min(self._low_override, self._high_override)
            if (
                self._image_pending_soft_limit_override is not None
                and self._image_pending_resume_threshold_override is not None
            ):
                self._image_pending_resume_threshold_override = min(
                    self._image_pending_resume_threshold_override,
                    self._image_pending_soft_limit_override,
                )

            self._last_check_monotonic = 0.0
            if force_recheck:
                await self._evaluate_unlocked(force=True)
            elif not self._enabled():
                self._set_active_unlocked(False, reason="disabled")
                self._soft_active = False
                self._lead_limit_active = False
            return self._status_unlocked()

    async def reset(self) -> None:
        async with self._lock:
            self._active = False
            self._last_backlog = 0
            self._last_detail = {}
            self._last_checked_at = None
            self._last_transition_at = None
            self._last_error = None
            self._last_check_monotonic = 0.0
            self._enabled_override = None
            self._high_override = None
            self._low_override = None
            self._check_interval_override = None
            self._mode_override = None
            self._consumer_time_budget_override = None
            self._consumer_micro_batch_size_override = None
            self._image_pending_soft_limit_override = None
            self._image_pending_resume_threshold_override = None
            self._soft_watermark_factor_override = None
            self._soft_throttle_sleep_ms_override = None
            self._soft_micro_batch_min_override = None
            self._soft_active = False
            self._lead_limit_active = False

    def snapshot(self) -> dict[str, Any]:
        return self._status_unlocked()

    async def _evaluate_unlocked(self, *, force: bool) -> bool:
        now_monotonic = time.monotonic()
        interval = max(float(self._check_interval_sec()), 0.0)
        if (
            not force
            and self._last_check_monotonic > 0
            and interval > 0
            and now_monotonic - self._last_check_monotonic < interval
        ):
            return self._active

        manager = self._manager_getter()
        self._last_check_monotonic = now_monotonic
        self._last_checked_at = _utc_iso()

        if manager is None or not getattr(manager, "connected", False):
            self._last_backlog = 0
            self._last_detail = {
                "reason": "mongo_unavailable",
                "backpressure_mode": self._mode(),
            }
            self._last_error = None
            if self._active:
                self._set_active_unlocked(False, reason="mongo_unavailable")
            return False

        try:
            backlog_count, detail = await asyncio.to_thread(
                count_actionable_backlog,
                manager,
                mode=self._mode(),
            )
        except Exception as error:
            self._last_error = str(error)
            logger.warning("kafka_backpressure_check_failed", error=self._last_error)
            if self._active:
                self._set_active_unlocked(False, reason="check_failed")
            return False

        self._last_backlog = int(backlog_count)
        self._last_detail = detail
        self._last_error = None

        effective_high = self._high_watermark()
        effective_low = self._low_watermark()
        self._soft_active = False
        self._lead_limit_active = False
        if self._mode() in {"image", "image_only", "image_priority", "image_core"}:
            soft_limit = self._image_pending_soft_limit()
            resume_threshold = self._image_pending_resume_threshold()
            if soft_limit > 0:
                effective_high = min(effective_high, soft_limit) if effective_high > 0 else soft_limit
                effective_low = (
                    min(effective_low, resume_threshold)
                    if effective_low > 0 and resume_threshold > 0
                    else min(resume_threshold, effective_high)
                )

        soft_high = self._soft_high_watermark(effective_high)
        if soft_high > 0 and backlog_count >= soft_high and backlog_count < effective_high:
            self._soft_active = True

        image_doc_lead = int(detail.get("image_doc_lead") or 0)
        fulltext_doc_lead = int(detail.get("fulltext_doc_lead") or 0)
        image_doc_lead_limit = self._image_doc_lead_limit()
        fulltext_doc_lead_limit = self._fulltext_doc_lead_limit()
        if (
            (image_doc_lead_limit > 0 and image_doc_lead >= image_doc_lead_limit)
            or (fulltext_doc_lead_limit > 0 and fulltext_doc_lead >= fulltext_doc_lead_limit)
        ):
            self._soft_active = True
            self._lead_limit_active = True

        if self._active:
            image_doc_lead_resume = self._image_doc_lead_resume_limit()
            fulltext_doc_lead_resume = self._fulltext_doc_lead_resume_limit()
            lead_released = (
                (image_doc_lead_limit <= 0 or image_doc_lead <= image_doc_lead_resume)
                and (
                    fulltext_doc_lead_limit <= 0
                    or fulltext_doc_lead <= fulltext_doc_lead_resume
                )
            )
            if backlog_count <= effective_low and lead_released:
                self._set_active_unlocked(False, reason="low_watermark_reached")
        elif backlog_count >= effective_high:
            self._set_active_unlocked(True, reason="high_watermark_reached")

        return self._active

    def _set_active_unlocked(self, active: bool, *, reason: str) -> None:
        if self._active == active:
            return
        self._active = active
        self._last_transition_at = _utc_iso()
        log_payload = {
            "active": active,
            "reason": reason,
            "backlog": self._last_backlog,
            "detail": self._last_detail,
            "high_watermark": self._high_watermark(),
            "low_watermark": self._low_watermark(),
            "mode": self._mode(),
        }
        if active:
            logger.warning("kafka_backpressure_engaged", **log_payload)
        else:
            logger.info("kafka_backpressure_released", **log_payload)

    def _enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return bool(getattr(settings, "KAFKA_BACKPRESSURE_ENABLED", False))

    def _high_watermark(self) -> int:
        value = (
            self._high_override
            if self._high_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_HIGH_WATERMARK", 0)
        )
        return max(int(value or 0), 0)

    def _low_watermark(self) -> int:
        value = (
            self._low_override
            if self._low_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_LOW_WATERMARK", 0)
        )
        resolved = max(int(value or 0), 0)
        return min(resolved, self._high_watermark())

    def _check_interval_sec(self) -> float:
        value = (
            self._check_interval_override
            if self._check_interval_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_CHECK_INTERVAL_SEC", 5)
        )
        return max(float(value or 0), 0.2)

    def _mode(self) -> str:
        value = (
            self._mode_override
            if self._mode_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_MODE", "image_priority")
        )
        return str(value or "image_priority").strip().lower()

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled(),
            "active": self._active,
            "soft_active": self._soft_active,
            "lead_limit_active": self._lead_limit_active,
            "mode": self._mode(),
            "high_watermark": self._high_watermark(),
            "low_watermark": self._low_watermark(),
            "soft_high_watermark": self._soft_high_watermark(self._high_watermark()),
            "check_interval_seconds": self._check_interval_sec(),
            "consumer_time_budget_ms": self._consumer_time_budget_ms(),
            "consumer_micro_batch_size": self._consumer_micro_batch_size(),
            "soft_watermark_factor": self._soft_watermark_factor(),
            "soft_throttle_sleep_ms": self.soft_throttle_sleep_ms(),
            "soft_micro_batch_min": self.soft_micro_batch_min(),
            "image_pending_soft_limit": self._image_pending_soft_limit(),
            "image_pending_resume_threshold": self._image_pending_resume_threshold(),
            "image_doc_lead_limit": self._image_doc_lead_limit(),
            "image_doc_lead_resume_limit": self._image_doc_lead_resume_limit(),
            "fulltext_doc_lead_limit": self._fulltext_doc_lead_limit(),
            "fulltext_doc_lead_resume_limit": self._fulltext_doc_lead_resume_limit(),
            "last_backlog": self._last_backlog,
            "last_detail": self._last_detail,
            "last_checked_at": self._last_checked_at,
            "last_transition_at": self._last_transition_at,
            "last_error": self._last_error,
        }

    def _consumer_time_budget_ms(self) -> int:
        value = (
            self._consumer_time_budget_override
            if self._consumer_time_budget_override is not None
            else getattr(settings, "KAFKA_CONSUMER_TIME_BUDGET_MS", 0)
        )
        return max(int(value or 0), 0)

    def _consumer_micro_batch_size(self) -> int:
        value = (
            self._consumer_micro_batch_size_override
            if self._consumer_micro_batch_size_override is not None
            else getattr(settings, "KAFKA_CONSUMER_MICRO_BATCH_SIZE", 1)
        )
        return max(int(value or 1), 1)

    def _image_pending_soft_limit(self) -> int:
        value = (
            self._image_pending_soft_limit_override
            if self._image_pending_soft_limit_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_IMAGE_PENDING_SOFT_LIMIT", 0)
        )
        return max(int(value or 0), 0)

    def _image_pending_resume_threshold(self) -> int:
        value = (
            self._image_pending_resume_threshold_override
            if self._image_pending_resume_threshold_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_IMAGE_PENDING_RESUME_THRESHOLD", 0)
        )
        resolved = max(int(value or 0), 0)
        return min(resolved, self._image_pending_soft_limit())

    def soft_active(self) -> bool:
        return self._soft_active

    def soft_throttle_sleep_ms(self) -> int:
        value = (
            self._soft_throttle_sleep_ms_override
            if self._soft_throttle_sleep_ms_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_SOFT_THROTTLE_SLEEP_MS", 250)
        )
        return max(int(value or 0), 0)

    def soft_micro_batch_min(self) -> int:
        value = (
            self._soft_micro_batch_min_override
            if self._soft_micro_batch_min_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_SOFT_MICRO_BATCH_MIN", 1)
        )
        return max(int(value or 1), 1)

    def _soft_watermark_factor(self) -> float:
        value = (
            self._soft_watermark_factor_override
            if self._soft_watermark_factor_override is not None
            else getattr(settings, "KAFKA_BACKPRESSURE_SOFT_WATERMARK_FACTOR", 0.75)
        )
        try:
            resolved = float(value)
        except Exception:
            resolved = 0.75
        return min(max(resolved, 0.1), 0.95)

    def _soft_high_watermark(self, hard_high: int) -> int:
        if hard_high <= 0:
            return 0
        return max(int(hard_high * self._soft_watermark_factor()), 1)

    def _doc_lead_resume_factor(self) -> float:
        value = getattr(settings, "KAFKA_BACKPRESSURE_DOC_LEAD_RESUME_FACTOR", 0.5)
        try:
            resolved = float(value)
        except Exception:
            resolved = 0.5
        return min(max(resolved, 0.1), 0.95)

    def _image_doc_lead_limit(self) -> int:
        value = getattr(settings, "KAFKA_BACKPRESSURE_IMAGE_DOC_LEAD_LIMIT", 0)
        return max(int(value or 0), 0)

    def _fulltext_doc_lead_limit(self) -> int:
        value = getattr(settings, "KAFKA_BACKPRESSURE_FULLTEXT_DOC_LEAD_LIMIT", 0)
        return max(int(value or 0), 0)

    def _image_doc_lead_resume_limit(self) -> int:
        limit = self._image_doc_lead_limit()
        if limit <= 0:
            return 0
        return max(int(limit * self._doc_lead_resume_factor()), 0)

    def _fulltext_doc_lead_resume_limit(self) -> int:
        limit = self._fulltext_doc_lead_limit()
        if limit <= 0:
            return 0
        return max(int(limit * self._doc_lead_resume_factor()), 0)


_kafka_backpressure_controller: Optional[KafkaBackpressureController] = None


def get_kafka_backpressure_controller() -> KafkaBackpressureController:
    global _kafka_backpressure_controller
    if _kafka_backpressure_controller is None:
        _kafka_backpressure_controller = KafkaBackpressureController()
    return _kafka_backpressure_controller


def reset_kafka_backpressure_controller() -> None:
    global _kafka_backpressure_controller
    _kafka_backpressure_controller = None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
