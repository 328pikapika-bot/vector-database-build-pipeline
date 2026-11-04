from __future__ import annotations

import asyncio

import pytest

from app.services.fulltext_worker import FullTextWorkerResult
from app.services.fulltext_worker_runner import FullTextWorkerRunner
from app.services.runtime_control import BackgroundLoopControl


@pytest.mark.asyncio
async def test_run_once_claims_one_task_and_processes_once():
    repository = FakeRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-2", "doc_id": "doc-2"},
        ]
    )
    worker = FakeWorker()
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        lease_seconds=30,
        batch_size=1,
        poll_interval_seconds=1,
        control=BackgroundLoopControl("fulltext"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is True
    assert result["task_id"] == "task-1"
    assert result["result"]["success"] is True
    assert repository.claim_calls == [{"worker_id": "worker-1", "lease_seconds": 30}]
    assert worker.calls == [{"task_id": "task-1", "doc_id": "doc-1"}]
    assert len(repository.tasks) == 1
    status = runner.status()
    assert status["last_task_id"] == "task-1"
    assert status["last_processed_count"] == 1
    assert status["last_error"] is None


@pytest.mark.asyncio
async def test_run_once_processes_batch_with_concurrency():
    repository = FakeRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-2", "doc_id": "doc-2"},
            {"task_id": "task-3", "doc_id": "doc-3"},
        ]
    )
    worker = FakeWorker()
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        batch_size=3,
        concurrency=2,
        control=BackgroundLoopControl("fulltext"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is True
    assert result["task_ids"] == ["task-1", "task-2", "task-3"]
    assert [call["task_id"] for call in worker.calls] == [
        "task-1",
        "task-2",
        "task-3",
    ]
    assert len(repository.claim_calls) == 3
    status = runner.status()
    assert status["last_task_id"] == "task-3"
    assert status["last_processed_count"] == 3


@pytest.mark.asyncio
async def test_run_once_without_task_does_not_call_worker():
    repository = FakeRepository([])
    worker = FakeWorker()
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        control=BackgroundLoopControl("fulltext"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is False
    assert result["processed"] is False
    assert worker.calls == []
    assert runner.status()["last_processed_count"] == 0


@pytest.mark.asyncio
async def test_run_once_records_process_exception_in_status():
    repository = FakeRepository([{"task_id": "task-1"}])
    worker = FakeWorker(error=RuntimeError("boom"))
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        control=BackgroundLoopControl("fulltext"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is False
    assert result["task_id"] == "task-1"
    assert result["error"] == "boom"
    status = runner.status()
    assert status["last_task_id"] == "task-1"
    assert status["last_processed_count"] == 0
    assert status["last_error"] == "boom"


@pytest.mark.asyncio
async def test_run_once_times_out_and_marks_task_failure():
    repository = FakeRepository([{"task_id": "task-1"}])
    worker = FakeWorker(release=asyncio.Event())
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        task_timeout_seconds=0.01,
        control=BackgroundLoopControl("fulltext"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is True
    assert result["task_id"] == "task-1"
    assert result["result"]["success"] is False
    assert result["result"]["error_code"] == "FULLTEXT_WORKER_TASK_TIMEOUT"
    assert result["result"]["retryable"] is True
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": "FULLTEXT_WORKER_TASK_TIMEOUT",
            "retryable": True,
        }
    ]
    status = runner.status()
    assert status["last_processed_count"] == 1
    assert status["last_error"] == result["result"]["error_message"]


@pytest.mark.asyncio
async def test_run_once_paused_does_not_claim_task():
    control = BackgroundLoopControl("fulltext")
    await control.pause("maintenance")
    repository = FakeRepository([{"task_id": "task-1"}])
    worker = FakeWorker()
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        control=control,
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is False
    assert result["processed"] is False
    assert result["paused"] is True
    assert repository.claim_calls == []
    assert worker.calls == []


@pytest.mark.asyncio
async def test_drain_waits_for_inflight_run_once_boundary():
    control = BackgroundLoopControl("fulltext")
    entered = asyncio.Event()
    release = asyncio.Event()
    repository = FakeRepository([{"task_id": "task-1", "doc_id": "doc-1"}])
    worker = FakeWorker(entered=entered, release=release)
    runner = FullTextWorkerRunner(
        repository=repository,
        worker=worker,
        control=control,
    )

    run_task = asyncio.create_task(runner.run_once(worker_id="worker-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    drain_task = asyncio.create_task(control.drain())
    await asyncio.sleep(0)
    assert drain_task.done() is False

    release.set()
    run_result = await asyncio.wait_for(run_task, timeout=1)
    drain_result = await asyncio.wait_for(drain_task, timeout=1)

    assert run_result["claimed"] is True
    assert run_result["processed"] is True
    assert drain_result["draining"] is True
    assert drain_result["inflight_runs"] == 0


class FakeRepository:
    def __init__(self, tasks: list[dict]) -> None:
        self.tasks = list(tasks)
        self.claim_calls: list[dict] = []
        self.failure_calls: list[dict] = []

    async def claim_next_task(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        exclude_task_ids: set[str] | None = None,
    ):
        self.claim_calls.append(
            {"worker_id": worker_id, "lease_seconds": lease_seconds}
        )
        excluded = set(exclude_task_ids or set())
        for index, task in enumerate(self.tasks):
            task_id = str(task.get("_id") or task.get("task_id") or "")
            if task_id not in excluded:
                return self.tasks.pop(index)
        return None

    async def mark_task_failure(
        self,
        task_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> None:
        self.failure_calls.append(
            {
                "task_id": task_id,
                "error_code": error_code,
                "retryable": retryable,
            }
        )


class FakeWorker:
    def __init__(
        self,
        error: Exception | None = None,
        entered=None,
        release=None,
    ) -> None:
        self.error = error
        self.entered = entered
        self.release = release
        self.calls: list[dict] = []

    async def process_task(self, task: dict) -> FullTextWorkerResult:
        self.calls.append(task)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return FullTextWorkerResult(
            task_id=str(task["task_id"]),
            success=True,
            artifact_refs=["raw", "chunk-plan"],
            chunk_count=2,
        )
