from __future__ import annotations

import asyncio

import pytest

from app.services.image_worker import ImageWorkerResult
from app.services.image_worker_runner import ImageWorkerRunner
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
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        lease_seconds=30,
        batch_size=1,
        poll_interval_seconds=1,
        control=BackgroundLoopControl("image"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is True
    assert result["task_id"] == "task-1"
    assert result["result"]["success"] is True
    assert repository.claim_calls == [
        {"worker_id": "worker-1", "lease_seconds": 30, "exclude_task_ids": []}
    ]
    assert worker.calls == [{"task_id": "task-1", "doc_id": "doc-1"}]
    assert len(repository.tasks) == 1
    status = runner.status()
    assert status["last_task_id"] == "task-1"
    assert status["last_processed_count"] == 1
    assert status["last_error"] is None


@pytest.mark.asyncio
async def test_run_once_without_task_does_not_call_worker():
    repository = FakeRepository([])
    worker = FakeWorker()
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        control=BackgroundLoopControl("image"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is False
    assert result["processed"] is False
    assert worker.calls == []
    assert runner.status()["last_processed_count"] == 0


@pytest.mark.asyncio
async def test_run_once_processes_claimed_batch():
    repository = FakeBatchRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-2", "doc_id": "doc-2"},
            {"task_id": "task-3", "doc_id": "doc-3"},
        ]
    )
    worker = FakeWorker()
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        batch_size=2,
        control=BackgroundLoopControl("image"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is True
    assert result["task_ids"] == ["task-1", "task-2"]
    assert len(result["results"]) == 2
    assert [call["task_id"] for call in worker.calls] == ["task-1", "task-2"]
    assert repository.claim_many_calls == []
    assert repository.claim_calls == [
        {"worker_id": "worker-1", "lease_seconds": 180, "exclude_task_ids": []},
        {"worker_id": "worker-1", "lease_seconds": 180, "exclude_task_ids": ["task-1"]},
    ]
    assert runner.status()["last_task_id"] == "task-2"
    assert runner.status()["last_processed_count"] == 2


@pytest.mark.asyncio
async def test_run_once_processes_batch_concurrently():
    repository = FakeBatchRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-2", "doc_id": "doc-2"},
        ]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    worker = FakeWorker(entered=entered, release=release, entered_after=2)
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        batch_size=2,
        concurrency=2,
        control=BackgroundLoopControl("image"),
    )

    run_task = asyncio.create_task(runner.run_once(worker_id="worker-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    assert len(worker.calls) == 2
    assert run_task.done() is False

    release.set()
    result = await asyncio.wait_for(run_task, timeout=1)

    assert result["task_ids"] == ["task-1", "task-2"]
    assert runner.status()["last_processed_count"] == 2
    assert runner.status()["concurrency"] == 2


@pytest.mark.asyncio
async def test_run_once_refills_finished_concurrency_slot():
    repository = FakeRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-2", "doc_id": "doc-2"},
            {"task_id": "task-3", "doc_id": "doc-3"},
        ]
    )
    task2_entered = asyncio.Event()
    task2_release = asyncio.Event()
    task3_entered = asyncio.Event()
    worker = RefillWorker(
        task2_entered=task2_entered,
        task2_release=task2_release,
        task3_entered=task3_entered,
    )
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        batch_size=3,
        concurrency=2,
        control=BackgroundLoopControl("image"),
    )

    run_task = asyncio.create_task(runner.run_once(worker_id="worker-1"))
    await asyncio.wait_for(task2_entered.wait(), timeout=1)
    await asyncio.wait_for(task3_entered.wait(), timeout=1)

    assert run_task.done() is False
    assert [call["task_id"] for call in worker.calls] == ["task-1", "task-2", "task-3"]

    task2_release.set()
    result = await asyncio.wait_for(run_task, timeout=1)

    assert result["task_ids"] == ["task-1", "task-2", "task-3"]
    assert runner.status()["last_processed_count"] == 3


@pytest.mark.asyncio
async def test_run_once_does_not_reclaim_active_task_id():
    repository = FakeRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-1", "doc_id": "doc-1-duplicate"},
            {"task_id": "task-2", "doc_id": "doc-2"},
        ]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    worker = FakeWorker(entered=entered, release=release)
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        batch_size=2,
        concurrency=2,
        control=BackgroundLoopControl("image"),
    )

    run_task = asyncio.create_task(runner.run_once(worker_id="worker-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.sleep(0)

    assert [call["task_id"] for call in worker.calls] == ["task-1", "task-2"]
    assert repository.claim_calls[1]["exclude_task_ids"] == ["task-1"]

    release.set()
    result = await asyncio.wait_for(run_task, timeout=1)

    assert result["task_ids"] == ["task-1", "task-2"]


@pytest.mark.asyncio
async def test_run_once_records_process_exception_in_status():
    repository = FakeRepository([{"task_id": "task-1"}])
    worker = FakeWorker(error=RuntimeError("boom"))
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        control=BackgroundLoopControl("image"),
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
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        task_timeout_seconds=0.01,
        control=BackgroundLoopControl("image"),
    )

    result = await runner.run_once(worker_id="worker-1")

    assert result["claimed"] is True
    assert result["processed"] is True
    assert result["task_id"] == "task-1"
    assert result["result"]["success"] is False
    assert result["result"]["error_code"] == "IMAGE_WORKER_TASK_TIMEOUT"
    assert result["result"]["retryable"] is False
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": "IMAGE_WORKER_TASK_TIMEOUT",
            "retryable": False,
        }
    ]
    status = runner.status()
    assert status["last_processed_count"] == 1
    assert status["last_error"] == result["result"]["error_message"]


@pytest.mark.asyncio
async def test_run_once_paused_does_not_claim_task():
    control = BackgroundLoopControl("image")
    await control.pause("maintenance")
    repository = FakeRepository([{"task_id": "task-1"}])
    worker = FakeWorker()
    runner = ImageWorkerRunner(
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
    control = BackgroundLoopControl("image")
    entered = asyncio.Event()
    release = asyncio.Event()
    repository = FakeRepository([{"task_id": "task-1", "doc_id": "doc-1"}])
    worker = FakeWorker(entered=entered, release=release)
    runner = ImageWorkerRunner(
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


@pytest.mark.asyncio
async def test_run_once_cancels_inflight_child_tasks_on_shutdown():
    repository = FakeRepository(
        [
            {"task_id": "task-1", "doc_id": "doc-1"},
            {"task_id": "task-2", "doc_id": "doc-2"},
        ]
    )
    entered = asyncio.Event()
    worker = FakeWorker(entered=entered, release=asyncio.Event(), entered_after=2)
    runner = ImageWorkerRunner(
        repository=repository,
        worker=worker,
        batch_size=2,
        concurrency=2,
        control=BackgroundLoopControl("image"),
    )

    run_task = asyncio.create_task(runner.run_once(worker_id="worker-1"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=1)

    assert worker.cancelled_count == 2


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
        excluded = set(exclude_task_ids or set())
        self.claim_calls.append(
            {
                "worker_id": worker_id,
                "lease_seconds": lease_seconds,
                "exclude_task_ids": sorted(excluded),
            }
        )
        if not self.tasks:
            return None
        for index, task in enumerate(self.tasks):
            task_id = str(task.get("task_id") or task.get("_id") or "")
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


class FakeBatchRepository(FakeRepository):
    def __init__(self, tasks: list[dict]) -> None:
        super().__init__(tasks)
        self.claim_many_calls: list[dict] = []

    async def claim_next_tasks(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        limit: int,
        exclude_task_ids: set[str] | None = None,
    ):
        self.claim_many_calls.append(
            {"worker_id": worker_id, "lease_seconds": lease_seconds, "limit": limit}
        )
        excluded = set(exclude_task_ids or set())
        claimed = []
        for _ in range(limit):
            task = None
            for index, item in enumerate(self.tasks):
                task_id = str(item.get("task_id") or item.get("_id") or "")
                if task_id not in excluded:
                    task = self.tasks.pop(index)
                    excluded.add(task_id)
                    break
            if task is None:
                break
            claimed.append(task)
        return claimed


class FakeWorker:
    def __init__(
        self,
        error: Exception | None = None,
        entered=None,
        release=None,
        entered_after: int = 1,
    ) -> None:
        self.error = error
        self.entered = entered
        self.release = release
        self.entered_after = entered_after
        self.calls: list[dict] = []
        self.cancelled_count = 0

    async def process_task(self, task: dict) -> ImageWorkerResult:
        self.calls.append(task)
        if self.entered is not None and len(self.calls) >= self.entered_after:
            self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled_count += 1
            raise
        if self.error is not None:
            raise self.error
        return ImageWorkerResult(
            task_id=str(task["task_id"]),
            success=True,
            artifact_refs=["manifest"],
        )


class RefillWorker:
    def __init__(
        self,
        *,
        task2_entered: asyncio.Event,
        task2_release: asyncio.Event,
        task3_entered: asyncio.Event,
    ) -> None:
        self.task2_entered = task2_entered
        self.task2_release = task2_release
        self.task3_entered = task3_entered
        self.calls: list[dict] = []

    async def process_task(self, task: dict) -> ImageWorkerResult:
        self.calls.append(task)
        if task["task_id"] == "task-2":
            self.task2_entered.set()
            await self.task2_release.wait()
        if task["task_id"] == "task-3":
            self.task3_entered.set()
        return ImageWorkerResult(
            task_id=str(task["task_id"]),
            success=True,
            artifact_refs=["manifest"],
        )
