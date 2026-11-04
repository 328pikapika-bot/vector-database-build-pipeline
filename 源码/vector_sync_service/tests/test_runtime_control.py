from __future__ import annotations

import asyncio

import pytest

from app.services.runtime_control import BackgroundLoopControl, RuntimeControl


@pytest.mark.asyncio
async def test_runtime_control_initial_pause_resume_status():
    control = RuntimeControl()

    initial = await control.status()
    assert initial["paused"] is False
    assert initial["draining"] is False
    assert initial["pause_reason"] is None
    assert initial["inflight_batches"] == 0

    paused = await control.pause("maintenance")
    assert paused["paused"] is True
    assert paused["pause_reason"] == "maintenance"

    resumed = await control.resume()
    assert resumed["paused"] is False
    assert resumed["draining"] is False
    assert resumed["pause_reason"] is None


@pytest.mark.asyncio
async def test_wait_if_paused_or_draining_unblocks_on_resume():
    control = RuntimeControl()
    await control.pause("operator_check")

    waiter = asyncio.create_task(control.wait_if_paused_or_draining())
    await asyncio.sleep(0)
    assert waiter.done() is False

    await control.resume()
    await asyncio.wait_for(waiter, timeout=1)


@pytest.mark.asyncio
async def test_drain_waits_for_inflight_batch_to_exit():
    control = RuntimeControl()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run_batch() -> None:
        async with control.track_inflight("batch-1"):
            entered.set()
            await release.wait()

    batch_task = asyncio.create_task(run_batch())
    await asyncio.wait_for(entered.wait(), timeout=1)

    drain_task = asyncio.create_task(control.drain())
    await asyncio.sleep(0)
    assert drain_task.done() is False

    status = await control.status()
    assert status["paused"] is True
    assert status["draining"] is True
    assert status["inflight_batches"] == 1

    release.set()
    drained = await asyncio.wait_for(drain_task, timeout=1)
    await batch_task

    assert drained["paused"] is True
    assert drained["draining"] is True
    assert drained["inflight_batches"] == 0

    resumed = await control.resume()
    assert resumed["draining"] is False


@pytest.mark.asyncio
async def test_background_loop_control_pause_resume_status():
    control = BackgroundLoopControl("fulltext")

    initial = await control.status()
    assert initial["paused"] is False
    assert initial["draining"] is False
    assert initial["inflight_runs"] == 0

    paused = await control.pause("maintenance")
    assert paused["paused"] is True
    assert paused["pause_reason"] == "maintenance"

    resumed = await control.resume()
    assert resumed["paused"] is False
    assert resumed["draining"] is False
    assert resumed["pause_reason"] is None


@pytest.mark.asyncio
async def test_background_loop_control_wait_unblocks_on_resume():
    control = BackgroundLoopControl("image")
    await control.pause("operator_check")

    waiter = asyncio.create_task(control.wait_if_paused_or_draining())
    await asyncio.sleep(0)
    assert waiter.done() is False

    await control.resume()
    await asyncio.wait_for(waiter, timeout=1)


@pytest.mark.asyncio
async def test_background_loop_control_drain_waits_for_inflight_run():
    control = BackgroundLoopControl("compensator")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run_once_boundary() -> None:
        async with control.track_inflight("round-1"):
            entered.set()
            await release.wait()

    run_task = asyncio.create_task(run_once_boundary())
    await asyncio.wait_for(entered.wait(), timeout=1)

    drain_task = asyncio.create_task(control.drain())
    await asyncio.sleep(0)
    assert drain_task.done() is False

    status = await control.status()
    assert status["paused"] is True
    assert status["draining"] is True
    assert status["inflight_runs"] == 1

    release.set()
    drained = await asyncio.wait_for(drain_task, timeout=1)
    await run_task

    assert drained["paused"] is True
    assert drained["draining"] is True
    assert drained["inflight_runs"] == 0
