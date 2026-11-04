from __future__ import annotations

import asyncio
import threading
from typing import Callable

import pytest

from app.models.sync_runtime import (
    BatchProcessingResult,
    ConsumedRecord,
    ProcessOutcome,
    SyncOverallStatus,
)
from app.services.kafka_backpressure import KafkaBackpressureController
from app.services import kafka_consumer as kafka_module
from app.services.runtime_control import RuntimeControl


class FakeConsumer:
    def __init__(self) -> None:
        self.poll_calls = 0
        self.commits = []

    def poll(self, timeout: float = 1.0):
        self.poll_calls += 1
        return None

    def metrics(self):
        return {}

    def commit(self, offsets, asynchronous: bool = False):
        self.commits.append(
            {
                "offsets": offsets,
                "asynchronous": asynchronous,
            }
        )


class FixedMessagesConsumer(FakeConsumer):
    def __init__(self, messages: list[object]) -> None:
        super().__init__()
        self.messages = list(messages)

    def poll(self, timeout: float = 1.0):
        self.poll_calls += 1
        if self.messages:
            return self.messages.pop(0)
        return None


class BlockingSecondPollConsumer(FakeConsumer):
    def __init__(self) -> None:
        super().__init__()
        self.second_poll_started = threading.Event()
        self.release_second_poll = threading.Event()

    def poll(self, timeout: float = 1.0):
        self.poll_calls += 1
        if self.poll_calls == 1:
            return object()
        self.second_poll_started.set()
        self.release_second_poll.wait(timeout=2.0)
        return None


class _FakeCollection:
    def __init__(self, docs_or_count) -> None:
        if isinstance(docs_or_count, int):
            self.docs = [{"status": "PENDING"} for _ in range(docs_or_count)]
        else:
            self.docs = [dict(item) for item in docs_or_count]

    def count_documents(self, query: dict) -> int:
        return sum(1 for doc in self.docs if _matches(doc, query or {}))


class _FakeDb:
    def __init__(self, collections: dict[str, int | list[dict]]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> _FakeCollection:
        return _FakeCollection(self.collections.get(name, 0))


class _FakeMongo:
    def __init__(
        self,
        collections: dict[str, int | list[dict]],
        *,
        status_count: int | list[dict] = 0,
        connected: bool = True,
    ) -> None:
        self.connected = connected
        self.db = _FakeDb(collections)
        self.status_collection = _FakeCollection(status_count)


def _record(offset: int = 1) -> ConsumedRecord:
    return ConsumedRecord(
        topic="VECTOR_JOURNAL_ARTICLE",
        partition=0,
        offset=offset,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint=f"topic|0|{offset}|abc",
    )


def _success_result() -> BatchProcessingResult:
    return BatchProcessingResult(
        outcomes=[
            ProcessOutcome(
                doc_id="doc-1",
                record_update_time="2026-06-07T00:00:00+00:00",
                final_status=SyncOverallStatus.SUCCESS.value,
                commit_safe=True,
            )
        ]
    )


def _unsafe_result() -> BatchProcessingResult:
    return BatchProcessingResult(
        outcomes=[
            ProcessOutcome(
                doc_id="doc-1",
                record_update_time="2026-06-07T00:00:00+00:00",
                final_status=SyncOverallStatus.FAILURE.value,
                commit_safe=False,
            )
        ]
    )


async def _empty_process(batch):
    return BatchProcessingResult()


async def _stop_consumer_task(task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.wait_for(task, timeout=1)


async def _wait_for_status(
    control: RuntimeControl,
    predicate: Callable[[dict], bool],
    timeout: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        status = await control.status()
        if predicate(status):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("runtime status condition was not met before timeout")


@pytest.mark.asyncio
async def test_consumer_does_not_poll_when_paused():
    control = RuntimeControl()
    await control.pause("unit_test")
    consumer = FakeConsumer()

    task = asyncio.create_task(
        kafka_module.run_consumer(
            consumer,
            _empty_process,
            batch_max_size=10,
            batch_max_wait_ms=1000,
            runtime_control=control,
        )
    )
    await asyncio.sleep(0.05)

    assert consumer.poll_calls == 0
    await _stop_consumer_task(task)


@pytest.mark.asyncio
async def test_consumer_does_not_poll_when_draining():
    control = RuntimeControl()
    await control.drain()
    consumer = FakeConsumer()

    task = asyncio.create_task(
        kafka_module.run_consumer(
            consumer,
            _empty_process,
            batch_max_size=10,
            batch_max_wait_ms=1000,
            runtime_control=control,
        )
    )
    await asyncio.sleep(0.05)

    assert consumer.poll_calls == 0
    await _stop_consumer_task(task)


@pytest.mark.asyncio
async def test_drain_flushes_buffered_record_before_returning(monkeypatch):
    control = RuntimeControl()
    consumer = BlockingSecondPollConsumer()
    processed_batches = []
    process_entered = asyncio.Event()
    release_process = asyncio.Event()

    monkeypatch.setattr(kafka_module, "parse_message", lambda msg: _record(5))

    async def process_batch(batch):
        processed_batches.append(batch)
        process_entered.set()
        await release_process.wait()
        return _success_result()

    task = asyncio.create_task(
        kafka_module.run_consumer(
            consumer,
            process_batch,
            batch_max_size=10,
            batch_max_wait_ms=60_000,
            runtime_control=control,
        )
    )

    await _wait_for_status(control, lambda status: status["buffered_records"] == 1)

    drain_task = asyncio.create_task(control.drain())
    consumer.release_second_poll.set()

    await asyncio.wait_for(process_entered.wait(), timeout=1)
    assert drain_task.done() is False

    release_process.set()
    drained = await asyncio.wait_for(drain_task, timeout=1)

    assert drained["paused"] is True
    assert drained["draining"] is True
    assert drained["buffered_records"] == 0
    assert drained["inflight_batches"] == 0
    assert len(processed_batches) == 1
    assert consumer.commits

    await _stop_consumer_task(task)


@pytest.mark.asyncio
async def test_handle_batch_tracks_inflight_and_commits_offsets():
    control = RuntimeControl()
    consumer = FakeConsumer()
    seen_inflight = asyncio.Event()

    async def process_batch(batch):
        status = await control.status()
        assert status["inflight_batches"] == 1
        seen_inflight.set()
        return _success_result()

    await kafka_module._handle_batch(
        [_record(8)],
        consumer,
        process_batch,
        runtime_control=control,
    )

    await asyncio.wait_for(seen_inflight.wait(), timeout=1)
    status = await control.status()
    assert status["inflight_batches"] == 0
    assert len(consumer.commits) == 1
    committed = consumer.commits[0]["offsets"][0]
    assert (committed.topic, committed.partition, committed.offset) == (
        "VECTOR_JOURNAL_ARTICLE",
        0,
        9,
    )


@pytest.mark.asyncio
async def test_ordered_pipeline_keeps_kafka_heartbeat_while_head_batch_waits():
    consumer = FakeConsumer()
    partition = kafka_module.TopicPartition("VECTOR_JOURNAL_ARTICLE", 0)
    consumer.assignment = lambda: [partition]
    consumer.pause_calls = 0
    consumer.resume_calls = 0
    consumer.pause = lambda _partitions: setattr(
        consumer, "pause_calls", consumer.pause_calls + 1
    )
    consumer.resume = lambda _partitions: setattr(
        consumer, "resume_calls", consumer.resume_calls + 1
    )

    release = asyncio.Event()

    async def process_batch(_batch):
        await release.wait()
        return _success_result()

    pipeline = kafka_module._OrderedBatchPipeline(
        consumer=consumer,
        process_batch_fn=process_batch,
        runtime_control=RuntimeControl(),
        inflight_limit=1,
    )
    dispatch_task = asyncio.create_task(
        pipeline.dispatch([_record(21)], force_commit=True)
    )

    await asyncio.sleep(0.1)
    assert consumer.poll_calls > 0
    assert consumer.pause_calls == 1
    assert not dispatch_task.done()

    release.set()
    await asyncio.wait_for(dispatch_task, timeout=1)
    assert consumer.resume_calls == 1
    assert consumer.commits


@pytest.mark.asyncio
async def test_handle_batch_commits_in_micro_batches(monkeypatch):
    control = RuntimeControl()
    consumer = FakeConsumer()
    seen_sizes = []

    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUMER_MICRO_BATCH_SIZE", 2)
    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUMER_TIME_BUDGET_MS", 30_000)

    async def process_batch(batch):
        seen_sizes.append(len(batch))
        return _success_result()

    await kafka_module._handle_batch(
        [_record(1), _record(2), _record(3), _record(4), _record(5)],
        consumer,
        process_batch,
        runtime_control=control,
    )

    assert seen_sizes == [2, 2, 1]
    assert len(consumer.commits) == 3


@pytest.mark.asyncio
async def test_handle_batch_does_not_commit_when_result_is_not_commit_safe_during_drain():
    control = RuntimeControl()
    await control.drain()
    consumer = FakeConsumer()
    seen_draining = asyncio.Event()

    async def process_batch(batch):
        status = await control.status()
        assert status["draining"] is True
        assert status["inflight_batches"] == 1
        seen_draining.set()
        return _unsafe_result()

    await kafka_module._handle_batch(
        [_record(12)],
        consumer,
        process_batch,
        runtime_control=control,
    )

    await asyncio.wait_for(seen_draining.wait(), timeout=1)
    status = await control.status()
    assert status["inflight_batches"] == 0
    assert consumer.commits == []


@pytest.mark.asyncio
async def test_handle_batch_does_not_commit_after_partition_revoke():
    control = RuntimeControl()
    consumer = FakeConsumer()
    state = kafka_module._AssignmentState()
    assigned = [kafka_module.TopicPartition("VECTOR_JOURNAL_ARTICLE", 0)]
    state.assigned(assigned)
    kafka_module._assignment_states[id(consumer)] = state

    async def process_batch(batch):
        state.revoked(assigned)
        return _success_result()

    try:
        result = await kafka_module._handle_batch(
            [_record(12)],
            consumer,
            process_batch,
            runtime_control=control,
        )
    finally:
        kafka_module._assignment_states.pop(id(consumer), None)

    assert result.commit_safe is False
    assert consumer.commits == []


@pytest.mark.asyncio
async def test_handle_batch_commits_unsafe_result_only_after_dlq_publish(monkeypatch):
    control = RuntimeControl()
    consumer = FakeConsumer()
    published = []

    monkeypatch.setattr(kafka_module.settings, "KAFKA_DLQ_ENABLED", True)
    monkeypatch.setattr(kafka_module.settings, "KAFKA_DLQ_TOPIC", "unit-test-dlq")

    def publish(_self, records, *, reason):
        published.append({"records": records, "reason": reason})
        return True

    monkeypatch.setattr(kafka_module._KafkaDlqPublisher, "publish", publish)

    result = await kafka_module._handle_batch(
        [_record(15)],
        consumer,
        lambda batch: asyncio.sleep(0, result=_unsafe_result()),
        runtime_control=control,
    )

    assert result.commit_safe is True
    assert len(consumer.commits) == 1
    assert published[0]["reason"] == "batch_outcome_not_commit_safe"


@pytest.mark.asyncio
async def test_consumer_stops_after_configured_consume_limit(monkeypatch):
    control = RuntimeControl()
    consumer = FixedMessagesConsumer([object(), object(), object()])
    processed_batches = []

    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUME_LIMIT", 2)
    monkeypatch.setattr(kafka_module, "parse_message", lambda msg: _record(consumer.poll_calls))

    async def process_batch(batch):
        processed_batches.append(batch)
        return _success_result()

    await asyncio.wait_for(
        kafka_module.run_consumer(
            consumer,
            process_batch,
            batch_max_size=10,
            batch_max_wait_ms=60_000,
            runtime_control=control,
        ),
        timeout=1,
    )

    assert [len(batch) for batch in processed_batches] == [2]
    assert len(consumer.commits) == 1
    assert consumer.poll_calls == 2


@pytest.mark.asyncio
async def test_metadata_bulk_ordered_pipeline_overlaps_batches_and_commits_in_order(monkeypatch):
    control = RuntimeControl()
    consumer = FixedMessagesConsumer([object(), object(), object(), object()])
    active = 0
    max_active = 0

    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUME_LIMIT", 4)
    monkeypatch.setattr(kafka_module.settings, "SYNC_PIPELINE_MODE", "metadata_bulk")
    monkeypatch.setattr(kafka_module.settings, "DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES", 2)
    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUMER_MICRO_BATCH_SIZE", 2)
    monkeypatch.setattr(kafka_module, "parse_message", lambda msg: _record(consumer.poll_calls))

    async def process_batch(batch):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _success_result()

    await asyncio.wait_for(
        kafka_module.run_consumer(
            consumer,
            process_batch,
            batch_max_size=2,
            batch_max_wait_ms=60_000,
            runtime_control=control,
        ),
        timeout=2,
    )

    assert max_active == 2
    assert [commit["offsets"][0].offset for commit in consumer.commits] == [3, 5]


@pytest.mark.asyncio
async def test_fulltext_ordered_pipeline_overlaps_batches_and_commits_in_order(monkeypatch):
    control = RuntimeControl()
    consumer = FixedMessagesConsumer([object(), object(), object(), object()])
    active = 0
    max_active = 0

    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUME_LIMIT", 4)
    monkeypatch.setattr(kafka_module.settings, "SYNC_WORKER_ROLE", "fulltext")
    monkeypatch.setattr(kafka_module.settings, "FULLTEXT_PIPELINE_INFLIGHT_BATCHES", 2)
    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUMER_MICRO_BATCH_SIZE", 2)
    monkeypatch.setattr(kafka_module, "parse_message", lambda msg: _record(consumer.poll_calls))

    async def process_batch(batch):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _success_result()

    await asyncio.wait_for(
        kafka_module.run_consumer(
            consumer,
            process_batch,
            batch_max_size=2,
            batch_max_wait_ms=60_000,
            runtime_control=control,
        ),
        timeout=2,
    )

    assert max_active == 2
    assert [commit["offsets"][0].offset for commit in consumer.commits] == [3, 5]


@pytest.mark.asyncio
async def test_image_ordered_pipeline_overlaps_batches_with_milvus_writes_enabled(monkeypatch):
    control = RuntimeControl()
    consumer = FixedMessagesConsumer([object(), object(), object(), object()])
    active = 0
    max_active = 0

    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUME_LIMIT", 4)
    monkeypatch.setattr(kafka_module.settings, "SYNC_WORKER_ROLE", "image")
    monkeypatch.setattr(kafka_module.settings, "IMAGE_MILVUS_WRITE_ENABLED", True)
    monkeypatch.setattr(kafka_module.settings, "IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES", 2)
    monkeypatch.setattr(kafka_module.settings, "KAFKA_CONSUMER_MICRO_BATCH_SIZE", 2)
    monkeypatch.setattr(
        kafka_module,
        "parse_message",
        lambda msg: ConsumedRecord(
            topic="VECTOR_JOURNAL_ARTICLE_IMAGE",
            partition=0,
            offset=consumer.poll_calls,
            raw_bytes=b"{}",
            raw_text="{}",
            event_fingerprint=f"image|0|{consumer.poll_calls}",
            parsed_payload={"data": [{"sysId": f"doc-{consumer.poll_calls}"}]},
        ),
    )

    async def process_batch(batch):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _success_result()

    await asyncio.wait_for(
        kafka_module.run_consumer(
            consumer,
            process_batch,
            batch_max_size=2,
            batch_max_wait_ms=60_000,
            runtime_control=control,
        ),
        timeout=2,
    )

    assert max_active == 2
    assert [commit["offsets"][0].offset for commit in consumer.commits] == [3, 5]


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_uses_hysteresis(monkeypatch):
    collections = {
        kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
        kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 6,
    }
    mongo = _FakeMongo(collections)
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=5,
        low_watermark=2,
        check_interval_sec=0,
        mode="image_priority",
    )

    assert await controller.should_throttle() is True
    status = await controller.status()
    assert status["active"] is True
    assert status["last_backlog"] == 6

    collections[kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION] = 4
    await asyncio.sleep(0.25)
    assert await controller.should_throttle() is True

    collections[kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION] = 2
    await asyncio.sleep(0.25)
    assert await controller.should_throttle() is False
    status = await controller.status()
    assert status["active"] is False
    assert status["last_backlog"] == 2


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_exposes_runtime_budget(monkeypatch):
    mongo = _FakeMongo(
        {
            kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
            kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 7,
        }
    )
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=5,
        low_watermark=2,
        check_interval_sec=0,
        mode="image_priority",
        consumer_time_budget_ms=9000,
        consumer_micro_batch_size=6,
        image_pending_soft_limit=4,
        image_pending_resume_threshold=1,
    )

    status = await controller.status()
    assert status["consumer_time_budget_ms"] == 9000
    assert status["consumer_micro_batch_size"] == 6
    assert status["image_pending_soft_limit"] == 4
    assert status["image_pending_resume_threshold"] == 1


@pytest.mark.asyncio
async def test_consumer_waits_when_backpressure_is_active(monkeypatch):
    control = RuntimeControl()
    consumer = FakeConsumer()
    wait_started = asyncio.Event()

    class _FakeBackpressure:
        async def should_throttle(self):
            return True

        async def wait_for_capacity(self):
            wait_started.set()
            await asyncio.sleep(10)

    monkeypatch.setattr(
        kafka_module,
        "get_kafka_backpressure_controller",
        lambda: _FakeBackpressure(),
    )

    task = asyncio.create_task(
        kafka_module.run_consumer(
            consumer,
            _empty_process,
            batch_max_size=10,
            batch_max_wait_ms=1000,
            runtime_control=control,
        )
    )

    await asyncio.wait_for(wait_started.wait(), timeout=1)
    assert consumer.poll_calls == 0
    await _stop_consumer_task(task)


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_exposes_soft_throttle_state():
    collections = {
        kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
        kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 4,
    }
    mongo = _FakeMongo(collections)
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=5,
        low_watermark=2,
        check_interval_sec=0,
        mode="image_priority",
    )

    throttled = await controller.should_throttle()
    status = await controller.status()

    assert throttled is False
    assert status["active"] is False
    assert status["soft_active"] is True
    assert status["soft_high_watermark"] >= 1
    assert "soft_watermark_factor" in status
    assert "soft_throttle_sleep_ms" in status
    assert "soft_micro_batch_min" in status


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_accepts_soft_limit_hot_updates():
    mongo = _FakeMongo(
        {
            kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
            kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 0,
        }
    )
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=100,
        low_watermark=10,
        check_interval_sec=0,
        mode="image_priority",
    )

    status = await controller.configure(
        soft_watermark_factor=0.7,
        soft_throttle_sleep_ms=333,
        soft_micro_batch_min=2,
        force_recheck=False,
    )

    assert status["soft_watermark_factor"] == 0.7
    assert status["soft_throttle_sleep_ms"] == 333
    assert status["soft_micro_batch_min"] == 2


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_uses_wider_default_hysteresis_for_image_priority(monkeypatch):
    collections = {
        kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
        kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 3_100,
    }
    mongo = _FakeMongo(collections)
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=5_000,
        low_watermark=2_500,
        check_interval_sec=0,
        mode="image_priority",
    )

    throttled = await controller.should_throttle()
    status = await controller.status()

    assert throttled is False
    assert status["active"] is False
    assert status["soft_active"] is True
    assert status["soft_high_watermark"] == 2_750
    assert status["low_watermark"] == 2_500


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_activates_soft_limit_when_document_lead_is_too_high():
    status_docs = [
        {
            "core_status": "SUCCESS",
            "image_required": True,
            "image_status": "PENDING",
            "document_status": "SUCCESS",
            "fulltext_required": False,
            "fulltext_status": "NOT_PRESENT",
        }
        for _ in range(3)
    ]
    mongo = _FakeMongo(
        {
            kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
            kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 0,
        },
        status_count=status_docs,
    )
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=100,
        low_watermark=10,
        check_interval_sec=0,
        mode="image_priority",
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        kafka_module.settings,
        "KAFKA_BACKPRESSURE_IMAGE_DOC_LEAD_LIMIT",
        2,
    )
    try:
        throttled = await controller.should_throttle()
        status = await controller.status()
    finally:
        monkeypatch.undo()

    assert throttled is False
    assert status["active"] is False
    assert status["soft_active"] is True
    assert status["lead_limit_active"] is True


@pytest.mark.asyncio
async def test_kafka_backpressure_controller_allows_more_document_lead_before_throttling(monkeypatch):
    status_docs = [
        {
            "core_status": "SUCCESS",
            "image_required": True,
            "image_status": "PENDING",
            "document_status": "SUCCESS",
            "fulltext_required": False,
            "fulltext_status": "NOT_PRESENT",
        }
        for _ in range(4_500)
    ]
    mongo = _FakeMongo(
        {
            kafka_module.settings.MONGO_FULLTEXT_TASK_COLLECTION: 0,
            kafka_module.settings.MONGO_IMAGE_TASK_COLLECTION: 0,
        },
        status_count=status_docs,
    )
    controller = KafkaBackpressureController(
        manager_getter=lambda: mongo,
        enabled=True,
        high_watermark=5_000,
        low_watermark=2_500,
        check_interval_sec=0,
        mode="image_priority",
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        kafka_module.settings,
        "KAFKA_BACKPRESSURE_IMAGE_DOC_LEAD_LIMIT",
        5_000,
    )
    try:
        throttled = await controller.should_throttle()
        status = await controller.status()
    finally:
        monkeypatch.undo()

    assert throttled is False
    assert status["active"] is False
    assert status["soft_active"] is False
    assert status["lead_limit_active"] is False


@pytest.mark.asyncio
async def test_handle_batch_reduces_micro_batch_when_soft_backpressure_is_active(monkeypatch):
    control = RuntimeControl()
    consumer = FakeConsumer()
    seen_sizes = []

    class _FakeBackpressure:
        def snapshot(self):
            return {
                "consumer_micro_batch_size": 4,
                "consumer_time_budget_ms": 30_000,
                "soft_active": True,
            }

        def soft_micro_batch_min(self):
            return 1

    monkeypatch.setattr(
        kafka_module,
        "get_kafka_backpressure_controller",
        lambda: _FakeBackpressure(),
    )

    async def process_batch(batch):
        seen_sizes.append(len(batch))
        return _success_result()

    await kafka_module._handle_batch(
        [_record(1), _record(2), _record(3)],
        consumer,
        process_batch,
        runtime_control=control,
    )

    assert seen_sizes == [1, 1, 1]


def _matches(doc: dict, query: dict) -> bool:
    if "$and" in query:
        return all(_matches(doc, item) for item in query["$and"])
    if "$or" in query:
        return any(_matches(doc, item) for item in query["$or"])
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                if actual not in set(expected["$in"]):
                    return False
            elif "$lte" in expected:
                limit = expected["$lte"]
                if actual is None or actual > limit:
                    return False
            else:
                raise AssertionError(f"unsupported fake filter operator: {expected}")
        elif actual != expected:
            return False
    return True
