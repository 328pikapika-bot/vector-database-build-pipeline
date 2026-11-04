"""Kafka 消费循环与批量聚合。"""
from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from confluent_kafka import Consumer, KafkaError, Message, Producer, TopicPartition

from app.config import settings
from app.models.raw_kafka_message import RawKafkaMessage
from app.models.sync_runtime import BatchProcessingResult, ConsumedRecord, sha256_hex
from app.observability.logger import get_logger
from app.observability.metrics import (
    batch_flush_total,
    idempotency_skip_total,
    kafka_consumer_lag,
    kafka_messages_total,
)
from app.services.kafka_backpressure import get_kafka_backpressure_controller
from app.services.message_utils import extract_message_meta
from app.services.runtime_control import RuntimeControl, get_runtime_control

logger = get_logger(__name__)

_dump_lock = threading.Lock()
T = TypeVar("T")
_assignment_states: dict[int, "_AssignmentState"] = {}
_DOC_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


class _AssignmentState:
    """Track partition ownership so a revoked batch can never commit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._assigned: set[tuple[str, int]] = set()

    def assigned(self, partitions: list[TopicPartition]) -> None:
        with self._lock:
            self._generation += 1
            self._assigned = {(item.topic, item.partition) for item in partitions}

    def revoked(self, partitions: list[TopicPartition]) -> None:
        with self._lock:
            self._generation += 1
            revoked = {(item.topic, item.partition) for item in partitions}
            self._assigned.difference_update(revoked)

    def snapshot_generation(self) -> int:
        with self._lock:
            return self._generation

    def can_commit(self, generation: int, offsets: list[TopicPartition]) -> bool:
        with self._lock:
            return generation == self._generation and all(
                (item.topic, item.partition) in self._assigned for item in offsets
            )


class _KafkaDlqPublisher:
    def __init__(self) -> None:
        self.topic = str(settings.KAFKA_DLQ_TOPIC or "").strip()
        self.enabled = bool(settings.KAFKA_DLQ_ENABLED and self.topic)
        self._producer: Optional[Producer] = None

    def publish(self, records: List[ConsumedRecord], *, reason: str) -> bool:
        if not self.enabled or not records:
            return False
        try:
            if self._producer is None:
                config = _build_kafka_client_security_config()
                config["bootstrap.servers"] = settings.KAFKA_BOOTSTRAP_SERVERS
                self._producer = Producer(config)
            for record in records:
                envelope = {
                    "reason": reason,
                    "source": {
                        "topic": record.topic,
                        "partition": record.partition,
                        "offset": record.offset,
                        "event_fingerprint": record.event_fingerprint,
                    },
                    "payload": record.parsed_payload,
                    "raw_text": record.raw_text,
                    "decode_error": record.decode_error,
                    "validation_error": record.validation_error,
                }
                self._producer.produce(
                    self.topic,
                    key=record.event_fingerprint.encode("utf-8"),
                    value=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
                )
            return self._producer.flush(10) == 0
        except Exception as error:
            logger.error("kafka_dlq_publish_failed", topic=self.topic, error=str(error))
            return False


@dataclass
class _BatchHandleResult:
    """Commit information returned when a batch is processed without committing."""

    commit_offset_groups: list[list[TopicPartition]]
    commit_safe: bool
    assignment_generation: Optional[int] = None


class _UnsafeOrderedPipelineResult(RuntimeError):
    """A pipelined batch was not safe to commit, so consumption must stop."""


class _OrderedBatchPipeline:
    """Bounded FIFO processing with commits delayed until the oldest task finishes.

    Tasks may finish out of order, but Kafka offsets are only committed in their
    dispatch order. This lets metadata embedding overlap the previous Milvus
    write without allowing a later batch to skip an earlier failed batch.
    """

    def __init__(
        self,
        *,
        consumer: Consumer,
        process_batch_fn,
        runtime_control: RuntimeControl,
        inflight_limit: int,
    ) -> None:
        self.consumer = consumer
        self.process_batch_fn = process_batch_fn
        self.runtime_control = runtime_control
        self.inflight_limit = max(int(inflight_limit or 1), 1)
        self._inflight: list[asyncio.Task[_BatchHandleResult]] = []

    async def dispatch(self, batch: List[ConsumedRecord], *, force_commit: bool = True) -> None:
        if not batch:
            return
        self._inflight.append(
            asyncio.create_task(
                _handle_batch(
                    batch,
                    self.consumer,
                    self.process_batch_fn,
                    force_commit=False,
                    runtime_control=self.runtime_control,
                )
            )
        )
        if len(self._inflight) >= self.inflight_limit:
            await self.complete_oldest(force_commit=force_commit)

    async def complete_oldest(self, *, force_commit: bool = True) -> None:
        if not self._inflight:
            return
        task = self._inflight.pop(0)
        result = await self._wait_for_task_with_heartbeats(task)
        if not result.commit_safe:
            raise _UnsafeOrderedPipelineResult("pipelined batch has an unsafe outcome")
        if not force_commit:
            return
        for offsets in result.commit_offset_groups:
            assignment_state = _assignment_states.get(id(self.consumer))
            if (
                assignment_state is not None
                and result.assignment_generation is not None
                and not assignment_state.can_commit(result.assignment_generation, offsets)
            ):
                raise _UnsafeOrderedPipelineResult("partition assignment changed before deferred commit")
            self.consumer.commit(offsets=offsets, asynchronous=False)
            logger.info(
                "offset_committed_ordered_pipeline",
                offsets=[
                    {"topic": offset.topic, "partition": offset.partition, "offset": offset.offset}
                    for offset in offsets
                ],
            )

    async def _wait_for_task_with_heartbeats(
        self,
        task: asyncio.Task[_BatchHandleResult],
    ) -> _BatchHandleResult:
        """Wait for the FIFO head without letting Kafka revoke the assignment.

        The ordered pipeline intentionally waits for the oldest batch before
        committing later batches.  A slow Milvus call can therefore outlive
        ``max.poll.interval.ms`` unless the consumer keeps polling.  Pausing
        assigned partitions prevents a heartbeat poll from consuming and
        dropping a record while the batch is in flight.
        """
        assigned: list[TopicPartition] = []
        paused = False
        try:
            assignment = getattr(self.consumer, "assignment", None)
            if callable(assignment):
                assigned = list(assignment() or [])
            pause = getattr(self.consumer, "pause", None)
            if assigned and callable(pause):
                pause(assigned)
                paused = True
        except Exception as error:
            logger.warning(
                "ordered_pipeline_partition_pause_failed",
                error=str(error),
            )

        heartbeat_interval = max(
            min(float(getattr(settings, "KAFKA_HEARTBEAT_INTERVAL_MS", 10_000)) / 2000.0, 1.0),
            0.1,
        )
        try:
            while not task.done():
                if paused:
                    try:
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: self.consumer.poll(timeout=0.0),
                        )
                    except Exception as error:
                        logger.warning(
                            "ordered_pipeline_heartbeat_poll_failed",
                            error=str(error),
                        )
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    continue
            return await task
        finally:
            if paused:
                try:
                    resume = getattr(self.consumer, "resume", None)
                    if callable(resume):
                        current_assignment = list(
                            self.consumer.assignment() or []
                        ) if callable(getattr(self.consumer, "assignment", None)) else assigned
                        if current_assignment:
                            resume(current_assignment)
                except Exception as error:
                    logger.warning(
                        "ordered_pipeline_partition_resume_failed",
                        error=str(error),
                    )

    async def drain(self, *, force_commit: bool = True) -> None:
        while self._inflight:
            await self.complete_oldest(force_commit=force_commit)

    async def cancel(self) -> None:
        for task in self._inflight:
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        self._inflight.clear()


def _append_jsonl(file_path: str, payload: dict) -> None:
    if not file_path:
        return
    try:
        with _dump_lock:
            with open(file_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as error:
        logger.error("append_jsonl_failed", file_path=file_path, error=str(error))


class BatchAggregator(Generic[T]):
    def __init__(
        self,
        max_size: int = 50,
        max_wait_ms: int = 5000,
        max_bytes: int = 0,
        size_of=None,
    ) -> None:
        self.max_size = max_size
        self.max_wait_ms = max_wait_ms
        self.max_bytes = max(int(max_bytes or 0), 0)
        self.size_of = size_of or (lambda _item: 0)
        self._buffer: List[T] = []
        self._first_event_time: Optional[float] = None
        self._buffer_bytes = 0

    def would_exceed_byte_limit(self, event: T) -> bool:
        """Keep a Kafka processing batch below the configured byte boundary."""
        if self.max_bytes <= 0 or not self._buffer:
            return False
        return self._buffer_bytes + max(int(self.size_of(event) or 0), 0) > self.max_bytes

    def add(self, event: T) -> Optional[List[T]]:
        now = time.monotonic()
        if self._first_event_time is None:
            self._first_event_time = now

        self._buffer.append(event)
        self._buffer_bytes += max(int(self.size_of(event) or 0), 0)
        elapsed_ms = (now - self._first_event_time) * 1000
        if (
            len(self._buffer) >= self.max_size
            or elapsed_ms >= self.max_wait_ms
            or (self.max_bytes > 0 and self._buffer_bytes >= self.max_bytes)
        ):
            return self.flush()
        return None

    def flush(self) -> List[T]:
        batch = self._buffer
        self._buffer = []
        self._first_event_time = None
        self._buffer_bytes = 0
        return batch

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def byte_size(self) -> int:
        return self._buffer_bytes


def _build_consumer_config(*, group_id: Optional[str] = None) -> dict[str, object]:
    config = {
        "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": str(group_id or settings.KAFKA_GROUP_ID).strip(),
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "max.poll.interval.ms": int(settings.KAFKA_MAX_POLL_INTERVAL_MS),
        "session.timeout.ms": int(settings.KAFKA_SESSION_TIMEOUT_MS),
        "heartbeat.interval.ms": int(settings.KAFKA_HEARTBEAT_INTERVAL_MS),
    }

    config.update(_build_kafka_client_security_config())

    return config


def _build_kafka_client_security_config() -> dict[str, object]:
    config: dict[str, object] = {}
    if settings.KAFKA_SECURITY_PROTOCOL in ("SASL_SSL", "SASL_PLAINTEXT"):
        config["security.protocol"] = settings.KAFKA_SECURITY_PROTOCOL
        config["sasl.mechanism"] = settings.KAFKA_SASL_MECHANISM or "PLAIN"
        if settings.KAFKA_SASL_USERNAME:
            config["sasl.username"] = settings.KAFKA_SASL_USERNAME
        if settings.KAFKA_SASL_PASSWORD:
            config["sasl.password"] = settings.KAFKA_SASL_PASSWORD
    else:
        config["security.protocol"] = "PLAINTEXT"
    return config


def _partition_payload(partitions: list[TopicPartition]) -> list[dict[str, int | str]]:
    return [
        {"topic": item.topic, "partition": item.partition, "offset": item.offset}
        for item in partitions
    ]


def create_consumer(
    *,
    group_id: Optional[str] = None,
    topic: Optional[str] = None,
) -> Consumer:
    resolved_topic = str(topic or settings.KAFKA_TOPIC).strip()
    resolved_group_id = str(group_id or settings.KAFKA_GROUP_ID).strip()
    if not resolved_topic:
        raise ValueError("Kafka topic is required")
    if not resolved_group_id:
        raise ValueError("Kafka consumer group id is required")

    config = _build_consumer_config(group_id=resolved_group_id)
    consumer = Consumer(config)
    assignment_state = _AssignmentState()
    _assignment_states[id(consumer)] = assignment_state

    def on_assign(current_consumer: Consumer, partitions: list[TopicPartition]) -> None:
        assignment_state.assigned(partitions)
        current_consumer.assign(partitions)
        logger.info("kafka_partitions_assigned", partitions=_partition_payload(partitions))

    def on_revoke(_current_consumer: Consumer, partitions: list[TopicPartition]) -> None:
        assignment_state.revoked(partitions)
        logger.warning("kafka_partitions_revoked", partitions=_partition_payload(partitions))

    consumer.subscribe([resolved_topic], on_assign=on_assign, on_revoke=on_revoke)
    logger.info(
        "kafka_consumer_created",
        topic=resolved_topic,
        group_id=resolved_group_id,
    )
    return consumer


def parse_message(msg: Message) -> Optional[ConsumedRecord]:
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            return None
        logger.error("kafka_message_error", error=str(msg.error()))
        return None

    value = msg.value()
    if value is None:
        return None

    raw_text = value.decode("utf-8", errors="replace")
    event_fingerprint = _build_event_fingerprint(msg, value)
    record = ConsumedRecord(
        topic=msg.topic(),
        partition=msg.partition(),
        offset=msg.offset(),
        raw_bytes=value,
        raw_text=raw_text,
        event_fingerprint=event_fingerprint,
    )

    max_message_bytes = max(int(getattr(settings, "KAFKA_MAX_MESSAGE_BYTES", 0) or 0), 0)
    if max_message_bytes and len(value) > max_message_bytes:
        record.validation_error = (
            f"KAFKA_MESSAGE_TOO_LARGE: {len(value)} bytes exceeds "
            f"KAFKA_MAX_MESSAGE_BYTES={max_message_bytes}"
        )
        logger.error(
            "kafka_message_too_large",
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            event_fingerprint=record.event_fingerprint,
            message_bytes=len(value),
            max_message_bytes=max_message_bytes,
        )
        return record

    stripped_text = raw_text.strip()
    if _DOC_ID_RE.fullmatch(stripped_text):
        # The three V2 topics intentionally carry only an authoritative Solr
        # document ID.  The role-specific worker resolves the rest of the
        # payload from Solr; preserve the same payload shape for observability.
        payload = {
            "data": [{"id": stripped_text, "sysId": stripped_text}],
            "operation": "UPDATE",
            "source_mode": "solr_doc_id",
        }
        record.parsed_payload = payload
        _append_jsonl(settings.KAFKA_DUMP_FILE, payload)
        logger.info(
            "kafka_document_id_received",
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            event_fingerprint=record.event_fingerprint,
            doc_id=stripped_text,
        )
        return record

    try:
        data = json.loads(raw_text)
    except Exception as error:
        record.decode_error = str(error)
        logger.error(
            "kafka_decode_error",
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            event_fingerprint=record.event_fingerprint,
            error=record.decode_error,
        )
        return record

    try:
        validated = RawKafkaMessage.model_validate(data)
        payload = validated.model_dump()
        record.parsed_payload = payload
        _append_jsonl(settings.KAFKA_DUMP_FILE, payload)
        meta = extract_message_meta(payload)
        logger.info(
            "kafka_message_received",
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            event_fingerprint=record.event_fingerprint,
            subDocType=meta.get("subDocType"),
            dataType=meta.get("dataType"),
            articleCount=meta.get("articleCount"),
            firstDocId=meta.get("firstDocId"),
            sysId=meta.get("sysId"),
            sysDataUpdateTime=meta.get("sysDataUpdateTime"),
            sysSubType=meta.get("sysSubType"),
        )
        return record
    except Exception as error:
        record.validation_error = str(error)
        logger.error(
            "kafka_validate_error",
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            event_fingerprint=record.event_fingerprint,
            error=record.validation_error,
        )
        return record


async def run_consumer(
    consumer: Consumer,
    process_batch_fn,
    batch_max_size: int | None = None,
    batch_max_wait_ms: int | None = None,
    runtime_control: RuntimeControl | None = None,
) -> None:
    batch_max_size = batch_max_size or settings.BATCH_MAX_SIZE
    batch_max_wait_ms = batch_max_wait_ms or settings.BATCH_MAX_WAIT_MS
    runtime_control = runtime_control or get_runtime_control()
    backpressure = get_kafka_backpressure_controller()
    assignment_state = _assignment_states.get(id(consumer))
    partitions_paused = False

    def pause_assigned_partitions() -> bool:
        nonlocal partitions_paused
        if assignment_state is None or partitions_paused or not hasattr(consumer, "assignment"):
            return False
        assigned = list(consumer.assignment() or [])
        if not assigned:
            return False
        consumer.pause(assigned)
        partitions_paused = True
        logger.info("kafka_partitions_paused", partitions=_partition_payload(assigned))
        return True

    def resume_assigned_partitions() -> None:
        nonlocal partitions_paused
        if not partitions_paused or not hasattr(consumer, "assignment"):
            return
        assigned = list(consumer.assignment() or [])
        if assigned:
            consumer.resume(assigned)
            logger.info("kafka_partitions_resumed", partitions=_partition_payload(assigned))
        partitions_paused = False

    async def wait_with_heartbeats(waiter) -> None:
        if not pause_assigned_partitions():
            await waiter
            return
        wait_task = asyncio.create_task(waiter)
        try:
            while not wait_task.done():
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: consumer.poll(timeout=0.0)
                )
                await asyncio.sleep(0.25)
            await wait_task
        finally:
            resume_assigned_partitions()

    aggregator: BatchAggregator[ConsumedRecord] = BatchAggregator(
        max_size=batch_max_size,
        max_wait_ms=batch_max_wait_ms,
        max_bytes=getattr(settings, "KAFKA_MAX_BATCH_BYTES", 0),
        size_of=lambda record: len(record.raw_bytes),
    )
    consume_limit = max(int(settings.KAFKA_CONSUME_LIMIT or 0), 0)
    consumed_records = 0
    metadata_bulk_pipeline_enabled = (
        str(getattr(settings, "SYNC_PIPELINE_MODE", "") or "").strip().lower()
        == "metadata_bulk"
        and int(getattr(settings, "DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES", 1) or 1) > 1
    )
    image_direct_pipeline_enabled = (
        str(getattr(settings, "SYNC_WORKER_ROLE", "") or "").strip().lower() == "image"
        and int(getattr(settings, "IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES", 1) or 1) > 1
    )
    fulltext_pipeline_enabled = (
        str(getattr(settings, "SYNC_WORKER_ROLE", "") or "").strip().lower() == "fulltext"
        and int(getattr(settings, "FULLTEXT_PIPELINE_INFLIGHT_BATCHES", 1) or 1) > 1
    )
    ordered_pipeline_inflight_limit = (
        int(getattr(settings, "DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES", 1) or 1)
        if metadata_bulk_pipeline_enabled
        else int(getattr(settings, "FULLTEXT_PIPELINE_INFLIGHT_BATCHES", 1) or 1)
        if fulltext_pipeline_enabled
        else int(getattr(settings, "IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES", 1) or 1)
    )
    ordered_pipeline = (
        _OrderedBatchPipeline(
            consumer=consumer,
            process_batch_fn=process_batch_fn,
            runtime_control=runtime_control,
            inflight_limit=ordered_pipeline_inflight_limit,
        )
        if metadata_bulk_pipeline_enabled or fulltext_pipeline_enabled or image_direct_pipeline_enabled
        else None
    )

    async def handle_ready_batch(
        batch: List[ConsumedRecord],
        *,
        force_commit: bool = True,
    ) -> None:
        if ordered_pipeline is not None:
            await ordered_pipeline.dispatch(batch, force_commit=force_commit)
            return
        await _handle_batch(
            batch,
            consumer,
            process_batch_fn,
            force_commit=force_commit,
            runtime_control=runtime_control,
        )

    async def drain_ordered_pipeline(*, force_commit: bool = True) -> None:
        if ordered_pipeline is not None:
            await ordered_pipeline.drain(force_commit=force_commit)

    logger.info(
        "consumer_loop_started",
        batch_max_size=batch_max_size,
        batch_max_wait_ms=batch_max_wait_ms,
        consume_limit=consume_limit,
        metadata_bulk_pipeline_enabled=metadata_bulk_pipeline_enabled,
        fulltext_pipeline_enabled=fulltext_pipeline_enabled,
        image_direct_pipeline_enabled=image_direct_pipeline_enabled,
        ordered_pipeline_inflight_batches=(
            ordered_pipeline.inflight_limit if ordered_pipeline is not None else 1
        ),
    )

    while True:
        try:
            if consume_limit and consumed_records >= consume_limit:
                if aggregator.size > 0:
                    batch = aggregator.flush()
                    await runtime_control.set_buffered_records(aggregator.size)
                    await handle_ready_batch(batch)
                await drain_ordered_pipeline()
                logger.info(
                    "consumer_consume_limit_reached",
                    consume_limit=consume_limit,
                    consumed_records=consumed_records,
                )
                break

            if await runtime_control.is_paused_or_draining():
                if aggregator.size > 0:
                    batch = aggregator.flush()
                    await runtime_control.set_buffered_records(aggregator.size)
                    await handle_ready_batch(batch)
                await drain_ordered_pipeline()
                await wait_with_heartbeats(runtime_control.wait_if_paused_or_draining())
                continue

            if await backpressure.should_throttle():
                if aggregator.size > 0:
                    batch = aggregator.flush()
                    await runtime_control.set_buffered_records(aggregator.size)
                    await handle_ready_batch(batch)
                await drain_ordered_pipeline()
                await wait_with_heartbeats(backpressure.wait_for_capacity())
                continue

            if getattr(backpressure, "soft_active", lambda: False)():
                sleep_ms = max(int(getattr(backpressure, "soft_throttle_sleep_ms", lambda: 0)() or 0), 0)
                if sleep_ms > 0:
                    await asyncio.sleep(sleep_ms / 1000.0)

            poll_registered = False
            await runtime_control.enter_poll()
            poll_registered = True
            try:
                msg = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: consumer.poll(timeout=1.0)
                )

                _update_lag(consumer)

                if msg is None:
                    await runtime_control.exit_poll()
                    poll_registered = False
                    batch = aggregator.flush() if aggregator.size > 0 else None
                    if batch:
                        await runtime_control.set_buffered_records(aggregator.size)
                        await handle_ready_batch(batch)
                    await drain_ordered_pipeline()
                    continue

                record = parse_message(msg)
                if record is None:
                    await runtime_control.exit_poll()
                    poll_registered = False
                    continue

                kafka_messages_total.labels(status="received").inc()
                if record.parsed_payload is not None:
                    _append_jsonl(settings.KAFKA_PROCESSED_DUMP_FILE, record.parsed_payload)

                if aggregator.would_exceed_byte_limit(record):
                    byte_limited_batch = aggregator.flush()
                    await runtime_control.set_buffered_records(aggregator.size)
                    await handle_ready_batch(byte_limited_batch)

                batch = aggregator.add(record)
                consumed_records += 1
                await runtime_control.set_buffered_records(aggregator.size)
                await runtime_control.exit_poll()
                poll_registered = False

                if batch is not None:
                    await runtime_control.set_buffered_records(aggregator.size)
                    await handle_ready_batch(batch)
            finally:
                if poll_registered:
                    await runtime_control.exit_poll()

        except asyncio.CancelledError:
            logger.info("consumer_cancelled", message="Consumer loop cancelled, flushing remaining")
            if aggregator.size > 0:
                batch = aggregator.flush()
                await runtime_control.set_buffered_records(aggregator.size)
                await handle_ready_batch(batch, force_commit=False)
            try:
                await drain_ordered_pipeline(force_commit=False)
            except _UnsafeOrderedPipelineResult:
                pass
            break
        except _UnsafeOrderedPipelineResult as error:
            logger.error("ordered_pipeline_stopped_unsafe_batch", error=str(error))
            if ordered_pipeline is not None:
                await ordered_pipeline.cancel()
            break
        except Exception as error:
            logger.error("consumer_loop_error", error=str(error))
            await asyncio.sleep(1.0)

    if ordered_pipeline is not None:
        await ordered_pipeline.cancel()
    resume_assigned_partitions()
    logger.info("consumer_loop_stopped")


async def _handle_batch(
    batch: List[ConsumedRecord],
    consumer: Consumer,
    process_batch_fn,
    force_commit: bool = True,
    runtime_control: RuntimeControl | None = None,
) -> _BatchHandleResult:
    import uuid

    runtime_control = runtime_control or get_runtime_control()
    assignment_state = _assignment_states.get(id(consumer))
    assignment_generation = (
        assignment_state.snapshot_generation() if assignment_state is not None else None
    )
    dlq_publisher = _KafkaDlqPublisher()
    batch_id = str(uuid.uuid4())[:8]
    logger.info("batch_flush", batch_id=batch_id, size=len(batch))
    batch_flush_total.inc()
    controller_snapshot = get_kafka_backpressure_controller().snapshot()
    micro_batch_size = max(
        int(controller_snapshot.get("consumer_micro_batch_size") or settings.KAFKA_CONSUMER_MICRO_BATCH_SIZE or len(batch) or 1),
        1,
    )
    if controller_snapshot.get("soft_active") or controller_snapshot.get("lead_limit_active"):
        soft_min = max(
            int(
                getattr(
                    get_kafka_backpressure_controller(),
                    "soft_micro_batch_min",
                    lambda: 1,
                )()
                or 1
            ),
            1,
        )
        micro_batch_size = min(micro_batch_size, soft_min)
    time_budget_seconds = max(
        float(controller_snapshot.get("consumer_time_budget_ms") or settings.KAFKA_CONSUMER_TIME_BUDGET_MS or 0),
        0.0,
    ) / 1000.0
    started_at = time.perf_counter()

    async with runtime_control.track_inflight(batch_id):
        try:
            aggregate_success = 0
            aggregate_skip = 0
            aggregate_fail = 0
            aggregate_dupes = 0
            last_commit_safe = True
            commit_offset_groups: list[list[TopicPartition]] = []
            sub_batches = _split_micro_batches(batch, micro_batch_size)
            for index, sub_batch in enumerate(sub_batches, start=1):
                result: BatchProcessingResult = await process_batch_fn(sub_batch)
                aggregate_success += result.success
                aggregate_skip += result.skip
                aggregate_fail += result.fail
                aggregate_dupes += result.skip_duplicates
                last_commit_safe = last_commit_safe and bool(result.commit_safe)

                if result.skip_duplicates > 0:
                    idempotency_skip_total.inc(result.skip_duplicates)

                kafka_messages_total.labels(status="success").inc(result.success)
                kafka_messages_total.labels(status="skip").inc(result.skip)
                kafka_messages_total.labels(status="fail").inc(result.fail)

                if result.commit_safe:
                    offsets = _build_commit_offsets(sub_batch)
                    commit_offset_groups.append(offsets)
                    if force_commit:
                        if (
                            assignment_state is not None
                            and assignment_generation is not None
                            and not assignment_state.can_commit(assignment_generation, offsets)
                        ):
                            logger.warning(
                                "offset_not_committed_assignment_changed",
                                batch_id=batch_id,
                                offsets=_partition_payload(offsets),
                            )
                            last_commit_safe = False
                            break
                        consumer.commit(offsets=offsets, asynchronous=False)
                        logger.info(
                            "offset_committed",
                            batch_id=batch_id,
                            sub_batch_index=index,
                            sub_batch_size=len(sub_batch),
                            offsets=[
                                {
                                    "topic": offset.topic,
                                    "partition": offset.partition,
                                    "offset": offset.offset,
                                }
                                for offset in offsets
                            ],
                            success=result.success,
                            skip=result.skip,
                            fail=result.fail,
                        )
                    else:
                        logger.debug(
                            "offset_commit_deferred",
                            batch_id=batch_id,
                            sub_batch_index=index,
                            sub_batch_size=len(sub_batch),
                        )
                else:
                    dlq_persisted = await asyncio.to_thread(
                        dlq_publisher.publish,
                        sub_batch,
                        reason="batch_outcome_not_commit_safe",
                    )
                    if dlq_persisted:
                        offsets = _build_commit_offsets(sub_batch)
                        if (
                            assignment_state is None
                            or assignment_generation is None
                            or assignment_state.can_commit(assignment_generation, offsets)
                        ):
                            commit_offset_groups.append(offsets)
                            if force_commit:
                                consumer.commit(offsets=offsets, asynchronous=False)
                            last_commit_safe = True
                    logger.warning(
                        "offset_not_committed",
                        batch_id=batch_id,
                        sub_batch_index=index,
                        sub_batch_size=len(sub_batch),
                        reason="unsafe_outcome" if not result.commit_safe else "force_commit_false",
                        success=result.success,
                        skip=result.skip,
                        fail=result.fail,
                    )
                    if not result.commit_safe:
                        break

                if time_budget_seconds > 0 and (time.perf_counter() - started_at) >= time_budget_seconds:
                    logger.info(
                        "consumer_time_budget_reached",
                        batch_id=batch_id,
                        processed_sub_batches=index,
                        total_sub_batches=len(sub_batches),
                        elapsed_seconds=round(time.perf_counter() - started_at, 3),
                        time_budget_seconds=round(time_budget_seconds, 3),
                    )
                    # A polled Kafka batch is an ownership unit. Continue to
                    # its end so a later offset can never skip an unprocessed
                    # tail from this batch.
                    time_budget_seconds = 0.0

            if aggregate_dupes > 0 and aggregate_dupes != 0:
                pass
            if not last_commit_safe and force_commit:
                logger.warning(
                    "batch_partial_commit_stopped",
                    batch_id=batch_id,
                    success=aggregate_success,
                    skip=aggregate_skip,
                    fail=aggregate_fail,
                )
            return _BatchHandleResult(
                commit_offset_groups=commit_offset_groups,
                commit_safe=last_commit_safe,
                assignment_generation=assignment_generation,
            )
        except Exception as error:
            logger.error("batch_process_error", batch_id=batch_id, error=str(error))
            return _BatchHandleResult(
                commit_offset_groups=[],
                commit_safe=False,
                assignment_generation=assignment_generation,
            )


def _update_lag(consumer: Consumer) -> None:
    try:
        metrics = consumer.metrics()
        for metric_name, metric_value in metrics.items():
            if "consumer_lag" in str(metric_name):
                kafka_consumer_lag.set(float(metric_value.value))
                return
    except Exception:
        pass


def _build_event_fingerprint(msg: Message, raw_bytes: bytes) -> str:
    return "|".join(
        [
            msg.topic(),
            str(msg.partition()),
            str(msg.offset()),
            sha256_hex(raw_bytes),
        ]
    )


def _build_commit_offsets(batch: List[ConsumedRecord]) -> List[TopicPartition]:
    latest_offsets: dict[tuple[str, int], int] = defaultdict(int)
    for record in batch:
        key = (record.topic, record.partition)
        latest_offsets[key] = max(latest_offsets.get(key, -1), record.offset)

    return [
        TopicPartition(topic, partition, offset + 1)
        for (topic, partition), offset in sorted(latest_offsets.items())
    ]


def _split_micro_batches(
    batch: List[ConsumedRecord],
    micro_batch_size: int,
) -> List[List[ConsumedRecord]]:
    if not batch:
        return []
    size = max(int(micro_batch_size or 1), 1)
    return [
        batch[index : index + size]
        for index in range(0, len(batch), size)
    ]
