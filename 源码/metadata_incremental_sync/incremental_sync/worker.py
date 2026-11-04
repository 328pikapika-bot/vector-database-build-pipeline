"""Single-instance incremental worker with durable Kafka inbox and safe writes."""

from __future__ import annotations

import hashlib
import json
import logging
import signal
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .adapters import EmbeddingReader, MilvusTarget, SafetyPause, SolrReader, TemporaryDependencyError
from .config import Settings
from .contract import PermanentDataError, assert_write_payload, new_record, require_id, source_values, update_payload, vector_inputs
from .store import Ledger, SingleInstanceLock, atomic_json, utc_now
from .status_http import start_status_server
from .failure_reporter import FailureReporter


LOG = logging.getLogger(__name__)


def target_identity(settings: Settings) -> str:
    # Do not include tokens or passwords in either the hash input or report.
    identity = {"topic": settings.kafka_topic, "group": settings.kafka_group,
                "milvus_uri": settings.milvus_uri, "database": settings.milvus_db,
                "collection": settings.milvus_collection}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def parse_message(value: bytes | None) -> str:
    if value is None or len(value) > 256:
        raise PermanentDataError("Kafka message is empty or longer than 256 bytes")
    try:
        return require_id(value.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise PermanentDataError("Kafka message is not UTF-8 text") from error


class IncrementalWorker:
    def __init__(self, settings: Settings, *, consumer: Any = None,
                 solr: Any = None, embedder: Any = None, milvus: Any = None) -> None:
        self.settings = settings
        self._injected_consumer = consumer
        self._injected_solr = solr
        self._injected_embedder = embedder
        self._injected_milvus = milvus
        self.consumer: Any = None
        self.solr: Any = None
        self.embedder: Any = None
        self.milvus: Any = None
        self.ledger: Ledger | None = None
        self.instance_lock: SingleInstanceLock | None = None
        self.stop_event = threading.Event()
        self.status = "starting"
        self.status_lock = threading.Lock()
        self.collection_id = ""
        self.processed_since_check = 0
        self.last_message_at = 0.0
        self.last_report_at = 0.0
        self.latest_counts: dict[str, int] = {}
        self.latest_batch_metrics: dict[str, Any] = {}
        self.reporter = FailureReporter(settings)
        self.mongo_pending = 0
        self.storage_fault = threading.Event()

    def _consumer(self) -> Any:
        if self._injected_consumer is not None:
            return self._injected_consumer
        from confluent_kafka import Consumer
        return Consumer(self.settings.kafka_config())

    def connect(self) -> None:
        self.solr = self._injected_solr or SolrReader(self.settings.solr_url, self.settings.request_timeout)
        self.embedder = self._injected_embedder or EmbeddingReader(
            self.settings.embedding_url, self.settings.embedding_model,
            self.settings.embedding_key, self.settings.request_timeout)
        self.milvus = self._injected_milvus or MilvusTarget(
            self.settings.milvus_uri, self.settings.milvus_db, self.settings.milvus_collection,
            self.settings.milvus_token, self.settings.request_timeout)
        self.consumer = self._consumer()

    def _close_connections(self) -> None:
        for name in ("consumer", "solr", "embedder", "milvus"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    LOG.exception("resource close failed: %s", name)
                setattr(self, name, None)

    def preflight(self) -> dict[str, Any]:
        description = self.milvus.check()
        self.collection_id = description["collection_id"]
        # Topic-specific metadata requests can auto-create a missing Topic.
        metadata = self.consumer.list_topics(timeout=10)
        topic = metadata.topics.get(self.settings.kafka_topic)
        if topic is None or topic.error is not None:
            raise SafetyPause("configured Kafka Topic is absent or unavailable")
        if len(topic.partitions) != 1:
            raise SafetyPause("Topic must have exactly one partition for ordered ID events")
        try:
            response = self.solr.http.get(self.solr.url, params={"q": "*:*", "rows": 0, "wt": "json"}, timeout=10)
            if response.status_code >= 400 or response.json().get("responseHeader", {}).get("status", 0) != 0:
                raise TemporaryDependencyError("Solr preflight failed")
        except (requests.RequestException, ValueError) as error:
            raise TemporaryDependencyError("Solr preflight failed") from error
        result = {"mode": self.settings.mode, "topic": self.settings.kafka_topic,
                  "group": self.settings.kafka_group, "milvus_collection": self.settings.milvus_collection,
                  "milvus_database": self.settings.milvus_db, **description, "checked_at": utc_now(),
                  "mongo_status": self.reporter.check(), "mongo_database": self.settings.mongo_db,
                  "mongo_failure_collection": self.settings.mongo_failure_collection,
                  "batch_size": self.settings.batch_size, "batch_wait_sec": self.settings.batch_wait_sec}
        return result

    def _set_status(self, value: str) -> None:
        with self.status_lock:
            self.status = value

    def heartbeat(self) -> None:
        while not self.stop_event.is_set():
            try:
                with self.status_lock:
                    status = self.status
                atomic_json(self.settings.state_dir / "progress.json", {
                    "status": status, "at": utc_now(), "topic": self.settings.kafka_topic,
                    "collection_id": self.collection_id, "counts": self.latest_counts,
                    "mongo_status": self.reporter.status, "mongo_pending": self.mongo_pending,
                    "latest_batch": self.latest_batch_metrics,
                })
            except Exception:
                LOG.exception("heartbeat or state persistence failed")
                self.storage_fault.set()
                self._set_status("storage_safety_pause")
            self.stop_event.wait(10)

    def _report(self, *, force: bool = False) -> None:
        if not self.ledger or (not force and time.monotonic() - self.last_report_at < 30):
            return
        since = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        reports = self.ledger.daily(self.ledger.report_days(since))
        for item in reports:
            atomic_json(self.settings.state_dir / "daily" / f"{item['received_day']}.json", item)
        from zoneinfo import ZoneInfo
        since_day = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(days=2)).date().isoformat()
        for item in self.ledger.activity_daily(since_day):
            atomic_json(self.settings.state_dir / "daily_activity" / f"{item['day']}.json", item)
            LOG.info("daily_activity day=%s inserted=%s updated=%s unchanged=%s failed=%s source_missing=%s active_seconds=%.2f first=%s last=%s",
                item["day"], item["inserted"], item["updated"], item["unchanged"], item["failed"],
                item["source_missing"], item["active_processing_seconds"], item["first_started_at"], item["last_completed_at"])
        atomic_json(self.settings.state_dir / "failures_recent.json", {
            "generated_at": utc_now(), "records": self.ledger.failures(),
            "note": "The complete failure ledger is checkpoint.db/events; this JSON is the latest 100 only.",
        })
        self.last_report_at = time.monotonic()
        self.latest_counts = self.ledger.counts()

    def _finish(self, event: dict[str, Any], outcome: str, error: str, started: float) -> None:
        self.ledger.finish(event, outcome, error, duration=time.monotonic() - started, target={
            "uri": self.settings.milvus_uri.split("@")[-1], "database": self.settings.milvus_db,
            "collection": self.settings.milvus_collection, "collection_id": self.collection_id})

    def _check_identity(self, count: int = 1) -> None:
        self.processed_since_check += count
        if self.processed_since_check < 100:
            return
        current = self.milvus.check()
        if current["collection_id"] != self.collection_id:
            raise SafetyPause("Milvus collection identity changed during processing")
        self.processed_since_check = 0

    def process_id(self, doc_id: str) -> str:
        doc_id = require_id(doc_id)
        self._check_identity()
        source = self.solr.fetch(doc_id)
        if source is None:
            return "source_missing"
        values = source_values(source, doc_id)
        vectors = self.embedder.embed(vector_inputs(values))
        before = self.milvus.get(doc_id)
        if before is None:
            payload = new_record(doc_id, values, vectors)
            self.milvus.write(doc_id, None, payload, new=True)
            return "inserted"
        payload = update_payload(doc_id, values, vectors, before)
        if len(payload) == 1:
            return "unchanged"
        assert_write_payload(payload, new=False)
        self.milvus.write(doc_id, before, payload, new=False)
        return "updated"

    def _batch_retry(self, events: list[dict[str, Any]], error: Exception,
                     started: float) -> None:
        assert self.ledger is not None
        duration = (time.monotonic() - started) / max(len(events), 1)
        for event in events:
            if int(event["attempts"]) + 1 >= self.settings.max_attempts:
                self._finish_batch_event(event, "failed", "retry_exhausted: " + str(error), duration)
                LOG.error("document_failed id=%s reason=retry_exhausted attempts=%s",
                          event["doc_id"], self.settings.max_attempts)
            else:
                delay = min(self.settings.retry_wait_sec * (2 ** min(int(event["attempts"]), 4)), 300)
                self.ledger.defer(event, str(error), duration=duration, delay=delay)
        self._set_status("waiting_dependency")

    def _finish_batch_event(self, event: dict[str, Any], outcome: str, error: str,
                            duration: float) -> None:
        assert self.ledger is not None
        self.ledger.finish(event, outcome, error, duration=duration, target={
            "uri": self.settings.milvus_uri.split("@")[-1], "database": self.settings.milvus_db,
            "collection": self.settings.milvus_collection, "collection_id": self.collection_id})

    def process_batch(self, events: list[dict[str, Any]]) -> None:
        """Resolve and write one bounded, durable inbox batch."""
        assert self.ledger is not None
        if not events:
            return
        batch_started = time.monotonic()
        by_id: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            self.ledger.begin(event)
            try:
                doc_id = require_id(event["doc_id"])
            except PermanentDataError as error:
                self._finish_batch_event(event, "failed", str(error),
                                         (time.monotonic() - batch_started) / len(events))
                LOG.error("document_failed id=%s reason=%s", event["doc_id"], error)
                continue
            by_id.setdefault(doc_id, []).append(event)
        if not by_id:
            self.latest_counts = self.ledger.counts()
            self._report()
            return

        stage: dict[str, float] = {}
        self._check_identity(len(by_id))
        tick = time.monotonic()
        try:
            ids = list(by_id)
            sources: dict[str, dict[str, Any]] = {}
            for start in range(0, len(ids), 200):
                sources.update(self.solr.fetch_many(ids[start:start + 200]))
        except SafetyPause:
            raise
        except TemporaryDependencyError as error:
            affected = [event for doc_id in by_id for event in by_id[doc_id]]
            self._batch_retry(affected, error, batch_started)
            return
        stage["solr"] = time.monotonic() - tick

        def finish_id(doc_id: str, outcome: str, error: str = "") -> None:
            duration = max((time.monotonic() - batch_started) / len(events), 0.0)
            for event in by_id.get(doc_id, []):
                self._finish_batch_event(event, outcome, error, duration)

        documents: dict[str, dict[str, Any]] = {}
        values_by_id: dict[str, dict[str, Any]] = {}
        for doc_id in by_id:
            source = sources.get(doc_id)
            if source is None:
                finish_id(doc_id, "source_missing")
                continue
            try:
                values_by_id[doc_id] = source_values(source, doc_id)
                documents[doc_id] = vector_inputs(values_by_id[doc_id])
            except PermanentDataError as error:
                finish_id(doc_id, "failed", str(error))
                LOG.error("document_failed id=%s reason=%s", doc_id, error)
        if not values_by_id:
            self.latest_counts = self.ledger.counts()
            self._set_status("consuming")
            self._report()
            return

        tick = time.monotonic()
        try:
            vectors_by_id: dict[str, dict[str, list[float]]] = {}
            document_items = list(documents.items())
            for start in range(0, len(document_items), 200):
                vectors_by_id.update(self.embedder.embed_many(
                    dict(document_items[start:start + 200])))
        except SafetyPause:
            raise
        except TemporaryDependencyError as error:
            affected = [event for doc_id in values_by_id for event in by_id[doc_id]]
            self._batch_retry(affected, error, batch_started)
            return
        except PermanentDataError:
            # A malformed vector can be isolated to one document. Retry each
            # embedding separately only on this exceptional path.
            vectors_by_id = {}
            for doc_id, inputs in documents.items():
                try:
                    vectors_by_id[doc_id] = self.embedder.embed(inputs)
                except PermanentDataError as error:
                    finish_id(doc_id, "failed", str(error))
                    LOG.error("document_failed id=%s reason=%s", doc_id, error)
                except TemporaryDependencyError as error:
                    self._batch_retry(by_id[doc_id], error, batch_started)
            values_by_id = {doc_id: values for doc_id, values in values_by_id.items()
                            if doc_id in vectors_by_id}
        stage["embedding"] = time.monotonic() - tick
        if not values_by_id:
            self.latest_counts = self.ledger.counts()
            self._report()
            return

        tick = time.monotonic()
        try:
            ids = list(values_by_id)
            before_by_id: dict[str, dict[str, Any]] = {}
            for start in range(0, len(ids), 200):
                before_by_id.update(self.milvus.get_many(ids[start:start + 200]))
        except SafetyPause:
            raise
        except TemporaryDependencyError as error:
            affected = [event for doc_id in values_by_id for event in by_id[doc_id]]
            self._batch_retry(affected, error, batch_started)
            return
        stage["strong_read"] = time.monotonic() - tick

        writes: list[dict[str, Any]] = []
        outcomes: dict[str, str] = {}
        for doc_id, values in values_by_id.items():
            before = before_by_id.get(doc_id)
            vectors = vectors_by_id.get(doc_id, {})
            try:
                if before is None:
                    payload = new_record(doc_id, values, vectors)
                    writes.append({"id": doc_id, "before": None, "payload": payload, "new": True})
                    outcomes[doc_id] = "inserted"
                else:
                    payload = update_payload(doc_id, values, vectors, before)
                    if len(payload) == 1:
                        outcomes[doc_id] = "unchanged"
                    else:
                        writes.append({"id": doc_id, "before": before, "payload": payload, "new": False})
                        outcomes[doc_id] = "updated"
            except PermanentDataError as error:
                finish_id(doc_id, "failed", str(error))
                LOG.error("document_failed id=%s reason=%s", doc_id, error)

        tick = time.monotonic()
        if writes:
            try:
                self.milvus.write_batch(writes)
            except SafetyPause:
                raise
            except TemporaryDependencyError as error:
                for doc_id, outcome in outcomes.items():
                    if outcome == "unchanged":
                        finish_id(doc_id, outcome)
                affected = [event for item in writes for event in by_id[item["id"]]]
                self._batch_retry(affected, error, batch_started)
                return
        stage["write_verify"] = time.monotonic() - tick

        for doc_id, outcome in outcomes.items():
            finish_id(doc_id, outcome)
        elapsed = time.monotonic() - batch_started
        batch_outcomes = {key: list(outcomes.values()).count(key)
                          for key in ("inserted", "updated", "unchanged")}
        self.latest_batch_metrics = {
            "events": len(events), "unique_ids": len(by_id), "elapsed_seconds": round(elapsed, 3),
            "rows_per_second": round(len(events) / elapsed, 3) if elapsed else 0,
            "outcomes": batch_outcomes,
            "stages_seconds": {key: round(value, 3) for key, value in stage.items()},
        }
        self.latest_counts = self.ledger.counts()
        self._set_status("consuming")
        LOG.info("batch_done events=%s ids=%s outcomes=%s rows_per_second=%.2f stages=%s",
                 len(events), len(by_id), json.dumps(batch_outcomes, sort_keys=True),
                 self.latest_batch_metrics["rows_per_second"],
                 json.dumps({key: round(value, 3) for key, value in stage.items()}, sort_keys=True))
        self._report()

    def _process_event(self, event: dict[str, Any]) -> None:
        assert self.ledger is not None
        started = time.monotonic()
        self.ledger.begin(event)
        try:
            doc_id = require_id(event["doc_id"])
            outcome = self.process_id(doc_id)
        except PermanentDataError as error:
            outcome = "failed"
            self._finish(event, outcome, str(error), started)
            LOG.error("document_failed id=%s reason=%s", event["doc_id"], error)
        except TemporaryDependencyError as error:
            if int(event["attempts"]) + 1 >= self.settings.max_attempts:
                self._finish(event, "failed", "retry_exhausted: " + str(error), started)
                LOG.error("document_failed id=%s reason=retry_exhausted attempts=%s", event["doc_id"], self.settings.max_attempts)
                self.latest_counts = self.ledger.counts()
                self._report()
                return
            self._set_status("waiting_dependency")
            delay = min(self.settings.retry_wait_sec * (2 ** min(int(event["attempts"]), 4)), 300)
            self.ledger.defer(event, str(error), duration=time.monotonic() - started, delay=delay)
            LOG.warning("dependency_wait id=%s delay=%ss reason=%s", event["doc_id"], delay, error)
            self.stop_event.wait(delay)
            return
        except (SafetyPause, sqlite3.Error, OSError):
            raise
        else:
            self._finish(event, outcome, "", started)
            LOG.info("document_done id=%s outcome=%s seconds=%.2f", event["doc_id"], outcome,
                     time.monotonic() - started)
        self.latest_counts = self.ledger.counts()
        self._set_status("consuming")
        self._report()

    def _consume_one(self, timeout: float = 1.0) -> bool:
        assert self.ledger is not None
        from confluent_kafka import KafkaException
        try:
            message = self.consumer.poll(timeout)
        except KafkaException as error:
            raise TemporaryDependencyError("Kafka poll unavailable") from error
        if message is None:
            if self.last_message_at and time.monotonic() - self.last_message_at >= self.settings.idle_window_sec:
                self._set_status("idle")
                self._report(force=True)
                self.last_message_at = 0.0
            return False
        if message.error():
            raise TemporaryDependencyError("Kafka poll returned an error")
        if message.topic() != self.settings.kafka_topic:
            raise SafetyPause("consumer received an unexpected Topic")
        raw = message.value()
        try:
            doc_id = parse_message(raw)
        except PermanentDataError:
            # Keep malformed messages auditable by Topic/partition/offset.
            doc_id = (raw or b"").decode("utf-8", errors="replace")[:256]
        self.ledger.enqueue(message.topic(), message.partition(), message.offset(), doc_id)
        self.latest_counts = self.ledger.counts()
        # Commit only after the SQLite transaction has completed with FULL sync.
        try:
            self.consumer.commit(message=message, asynchronous=False)
        except KafkaException as error:
            # Inbox is durable; broker redelivery is deduplicated by offset.
            raise TemporaryDependencyError("Kafka offset commit unavailable; durable inbox retained") from error
        self.last_message_at = time.monotonic()
        self._set_status("consuming")
        return True

    def _wait_safely(self, error: Exception) -> None:
        self._set_status("safety_pause")
        try:
            if not (self.settings.state_dir / "safety_pause.json").exists():
                atomic_json(self.settings.state_dir / "safety_pause.json", {
                    "at": utc_now(), "reason": str(error), "collection_id": self.collection_id,
                })
        except Exception:
            LOG.exception("could not persist safety-pause report")
        LOG.critical("writes_paused: %s", error)
        while not self.stop_event.is_set():
            try:
                atomic_json(self.settings.state_dir / "progress.json", {
                    "status": "safety_pause", "at": utc_now(),
                    "topic": self.settings.kafka_topic, "collection_id": self.collection_id,
                    "counts": self.latest_counts,
                })
            except Exception:
                LOG.exception("safety-pause heartbeat could not be persisted")
            self.stop_event.wait(10)

    def run(self) -> None:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.instance_lock = SingleInstanceLock(self.settings.state_dir / "instance.lock")
        heartbeat: threading.Thread | None = None
        status_server = None
        status_thread: threading.Thread | None = None
        try:
            if self.settings.mode == "run" and not self.settings.mongo_uri:
                raise SafetyPause("MONGO_URI must be configured for run mode")
            if (self.settings.state_dir / "safety_pause.json").exists():
                raise SafetyPause("persisted safety pause requires manual investigation before restart")
            if self.settings.mode == "run" and not self.settings.vector_update_verified:
                raise SafetyPause("VECTOR_PARTIAL_UPDATE_VERIFIED=1 is required after isolated vector-update validation")
            heartbeat = threading.Thread(target=self.heartbeat, name="incremental-heartbeat", daemon=True)
            heartbeat.start()
            status_server = start_status_server(self.settings.state_dir)
            status_thread = threading.Thread(target=status_server.serve_forever,
                                             name="incremental-status", daemon=True)
            status_thread.start()
            result = None
            while not self.stop_event.is_set():
                self._set_status("checking_dependencies")
                try:
                    self.connect()
                    result = self.preflight()
                    break
                except SafetyPause:
                    raise
                except Exception as error:
                    self._close_connections()
                    self._set_status("waiting_dependencies")
                    LOG.warning("preflight_wait reason=%s", type(error).__name__)
                    self.stop_event.wait(self.settings.retry_wait_sec)
            if result is None:
                return
            atomic_json(self.settings.state_dir / "preflight.json", result)
            LOG.info("preflight_passed mode=%s topic=%s fields=%s", self.settings.mode,
                     self.settings.kafka_topic, result["fields"])
            if self.settings.mode == "run":
                self.ledger = Ledger(self.settings.state_dir, target_identity(self.settings))
                self.latest_counts = self.ledger.counts()
                self.consumer.subscribe([self.settings.kafka_topic])
                self._set_status("consuming")
            else:
                self._set_status("check_passed")
            if self.settings.mode == "check":
                while not self.stop_event.wait(10):
                    pass
                return
            while not self.stop_event.is_set():
                if self.storage_fault.is_set():
                    raise SafetyPause("state directory cannot persist heartbeat/progress")
                self.reporter.sync(self.ledger)
                self.mongo_pending = self.ledger.outbox_pending()
                events = self.ledger.next_events(self.settings.batch_size)
                if not events:
                    try:
                        self._consume_one()
                    except TemporaryDependencyError as error:
                        self._set_status("waiting_kafka")
                        LOG.warning("kafka_wait: %s", error)
                        self.stop_event.wait(self.settings.retry_wait_sec)
                    continue
                wait_started = time.monotonic()
                while (len(events) < self.settings.batch_size and
                       time.monotonic() - wait_started < self.settings.batch_wait_sec and
                       not self.stop_event.is_set()):
                    try:
                        received = self._consume_one(0.0)
                    except TemporaryDependencyError as error:
                        self._set_status("waiting_kafka")
                        LOG.warning("kafka_wait: %s", error)
                        break
                    events = self.ledger.next_events(self.settings.batch_size)
                    if not received:
                        remaining = self.settings.batch_wait_sec - (time.monotonic() - wait_started)
                        self.stop_event.wait(min(0.01, max(remaining, 0.0)))
                if self.stop_event.is_set():
                    break
                self.process_batch(events)
            self._set_status("stopped")
            self._report(force=True)
            self.reporter.sync(self.ledger, force=True)
        except SafetyPause as error:
            self._wait_safely(error)
        except Exception as error:
            LOG.exception("unexpected worker failure; pausing writes")
            self._wait_safely(SafetyPause(f"unexpected {type(error).__name__}; see local diagnostics"))
        finally:
            self.stop_event.set()
            if status_server is not None:
                status_server.shutdown()
                status_server.server_close()
            if status_thread is not None:
                status_thread.join(timeout=5)
            if heartbeat is not None:
                heartbeat.join(timeout=15)
            self._close_connections()
            self.reporter.close()
            for resource in (self.ledger, self.instance_lock):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        LOG.exception("resource close failed")

    def stop(self, *_: Any) -> None:
        self.stop_event.set()


def install_signals(worker: IncrementalWorker) -> None:
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
