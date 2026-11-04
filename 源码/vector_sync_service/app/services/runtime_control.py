"""Runtime pause/resume/drain coordination."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from app.observability.metrics import (
    runtime_draining,
    runtime_inflight_batches,
    runtime_paused,
    worker_control_draining,
    worker_control_inflight_runs,
    worker_control_paused,
)


BACKGROUND_WORKER_TYPES = ("fulltext", "image", "compensator")


class RuntimeControl:
    """Coordinates manual runtime control for the source consumer."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._paused = False
        self._draining = False
        self._pause_reason: Optional[str] = None
        self._inflight_batches = 0
        self._active_polls = 0
        self._buffered_records = 0
        self._updated_at = _utc_iso()
        self._update_metrics_unlocked()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def pause_reason(self) -> Optional[str]:
        return self._pause_reason

    @property
    def inflight_batches(self) -> int:
        return self._inflight_batches

    async def pause(self, reason: Optional[str] = None) -> dict[str, Any]:
        async with self._condition:
            self._paused = True
            self._pause_reason = reason or "manual_pause"
            self._touch_unlocked()
            self._condition.notify_all()
            return self._status_unlocked()

    async def resume(self) -> dict[str, Any]:
        async with self._condition:
            self._paused = False
            self._draining = False
            self._pause_reason = None
            self._touch_unlocked()
            self._condition.notify_all()
            return self._status_unlocked()

    async def drain(self) -> dict[str, Any]:
        async with self._condition:
            self._paused = True
            self._draining = True
            if not self._pause_reason:
                self._pause_reason = "drain_requested"
            self._touch_unlocked()
            self._condition.notify_all()
            while (
                self._inflight_batches > 0
                or self._active_polls > 0
                or self._buffered_records > 0
            ):
                await self._condition.wait()
            return self._status_unlocked()

    async def status(self) -> dict[str, Any]:
        async with self._condition:
            return self._status_unlocked()

    async def is_paused_or_draining(self) -> bool:
        async with self._condition:
            return self._paused or self._draining

    async def wait_if_paused_or_draining(self) -> None:
        async with self._condition:
            while self._paused or self._draining:
                await self._condition.wait()

    @asynccontextmanager
    async def track_inflight(self, batch_id: Optional[str] = None) -> AsyncIterator[None]:
        await self._enter_inflight(batch_id)
        try:
            yield
        finally:
            await self._exit_inflight(batch_id)

    async def enter_poll(self) -> None:
        async with self._condition:
            self._active_polls += 1
            self._touch_unlocked()
            self._condition.notify_all()

    async def exit_poll(self) -> None:
        async with self._condition:
            self._active_polls = max(0, self._active_polls - 1)
            self._touch_unlocked()
            self._condition.notify_all()

    async def set_buffered_records(self, count: int) -> None:
        async with self._condition:
            self._buffered_records = max(0, count)
            self._touch_unlocked()
            self._condition.notify_all()

    async def _enter_inflight(self, batch_id: Optional[str]) -> None:
        async with self._condition:
            self._inflight_batches += 1
            self._touch_unlocked()
            self._condition.notify_all()

    async def _exit_inflight(self, batch_id: Optional[str]) -> None:
        async with self._condition:
            self._inflight_batches = max(0, self._inflight_batches - 1)
            self._touch_unlocked()
            self._condition.notify_all()

    def _touch_unlocked(self) -> None:
        self._updated_at = _utc_iso()
        self._update_metrics_unlocked()

    def _update_metrics_unlocked(self) -> None:
        runtime_paused.set(1.0 if self._paused else 0.0)
        runtime_draining.set(1.0 if self._draining else 0.0)
        runtime_inflight_batches.set(float(self._inflight_batches))

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "draining": self._draining,
            "pause_reason": self._pause_reason,
            "inflight_batches": self._inflight_batches,
            "active_polls": self._active_polls,
            "buffered_records": self._buffered_records,
            "updated_at": self._updated_at,
        }


class BackgroundLoopControl:
    """Pause/resume/drain coordination for a background loop."""

    def __init__(self, worker_type: str) -> None:
        if worker_type not in BACKGROUND_WORKER_TYPES:
            raise ValueError(f"unsupported worker_type: {worker_type}")
        self.worker_type = worker_type
        self._condition = asyncio.Condition()
        self._paused = False
        self._draining = False
        self._pause_reason: Optional[str] = None
        self._inflight_runs = 0
        self._updated_at = _utc_iso()
        self._update_metrics_unlocked()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def inflight_runs(self) -> int:
        return self._inflight_runs

    async def pause(self, reason: Optional[str] = None) -> dict[str, Any]:
        async with self._condition:
            self._paused = True
            self._pause_reason = reason or "manual_pause"
            self._touch_unlocked()
            self._condition.notify_all()
            return self._status_unlocked()

    async def resume(self) -> dict[str, Any]:
        async with self._condition:
            self._paused = False
            self._draining = False
            self._pause_reason = None
            self._touch_unlocked()
            self._condition.notify_all()
            return self._status_unlocked()

    async def drain(self) -> dict[str, Any]:
        async with self._condition:
            self._paused = True
            self._draining = True
            if not self._pause_reason:
                self._pause_reason = "drain_requested"
            self._touch_unlocked()
            self._condition.notify_all()
            while self._inflight_runs > 0:
                await self._condition.wait()
            return self._status_unlocked()

    async def status(self) -> dict[str, Any]:
        async with self._condition:
            return self._status_unlocked()

    def snapshot(self) -> dict[str, Any]:
        return self._status_unlocked()

    async def is_paused_or_draining(self) -> bool:
        async with self._condition:
            return self._paused or self._draining

    async def wait_if_paused_or_draining(self) -> None:
        async with self._condition:
            while self._paused or self._draining:
                await self._condition.wait()

    @asynccontextmanager
    async def track_inflight(self, run_id: Optional[str] = None) -> AsyncIterator[None]:
        await self._enter_inflight(run_id)
        try:
            yield
        finally:
            await self._exit_inflight(run_id)

    async def _enter_inflight(self, run_id: Optional[str]) -> None:
        async with self._condition:
            self._inflight_runs += 1
            self._touch_unlocked()
            self._condition.notify_all()

    async def _exit_inflight(self, run_id: Optional[str]) -> None:
        async with self._condition:
            self._inflight_runs = max(0, self._inflight_runs - 1)
            self._touch_unlocked()
            self._condition.notify_all()

    def _touch_unlocked(self) -> None:
        self._updated_at = _utc_iso()
        self._update_metrics_unlocked()

    def _update_metrics_unlocked(self) -> None:
        worker_control_paused.labels(worker_type=self.worker_type).set(
            1.0 if self._paused else 0.0
        )
        worker_control_draining.labels(worker_type=self.worker_type).set(
            1.0 if self._draining else 0.0
        )
        worker_control_inflight_runs.labels(worker_type=self.worker_type).set(
            float(self._inflight_runs)
        )

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "worker_type": self.worker_type,
            "paused": self._paused,
            "draining": self._draining,
            "pause_reason": self._pause_reason,
            "inflight_runs": self._inflight_runs,
            "updated_at": self._updated_at,
        }


_runtime_control: Optional[RuntimeControl] = None
_background_loop_controls: dict[str, BackgroundLoopControl] = {}


def get_runtime_control() -> RuntimeControl:
    global _runtime_control
    if _runtime_control is None:
        _runtime_control = RuntimeControl()
    return _runtime_control


def get_background_loop_control(worker_type: str) -> BackgroundLoopControl:
    if worker_type not in BACKGROUND_WORKER_TYPES:
        raise ValueError(f"unsupported worker_type: {worker_type}")
    control = _background_loop_controls.get(worker_type)
    if control is None:
        control = BackgroundLoopControl(worker_type)
        _background_loop_controls[worker_type] = control
    return control


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
