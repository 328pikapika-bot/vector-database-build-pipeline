"""Durable, resumable full backfill. Only BackfillWorker's scalar partial upsert is used.

Business failures are committed alongside batch state before progress advances.
Transient dependencies wait; safety faults latch across container restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from pymongo import MongoClient, UpdateOne
from pymilvus.exceptions import MilvusException

try:
    from .backfill import (BackfillWorker, Checkpoint, FIELD_SPECS, HEX_ID, SHARDS,
                           PARSER_VERSION, SOLR_MAPPING, SOLR_QUERY_FIELDS, TARGET_FIELDS,
                           MilvusClient, Settings, state_lock)
except ImportError:
    from backfill import (BackfillWorker, Checkpoint, FIELD_SPECS, HEX_ID, SHARDS,
                          PARSER_VERSION, SOLR_MAPPING, SOLR_QUERY_FIELDS, TARGET_FIELDS,
                          MilvusClient, Settings, state_lock)

LOGGER = logging.getLogger("solr_field_backfill")

# A transient count drop above this floor is not a write-safety failure. The
# operational high-water mark remains audit-only; identity, schema, index,
# planned-ID, and history checks still latch a safety pause.
LOGICAL_COUNT_SAFETY_FLOOR = 52_500_000


class SafetyPause(RuntimeError):
    pass


class Stopping(RuntimeError):
    pass


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(encoded(value))
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def closed_channel(exc: Exception) -> bool:
    return any(text in str(exc).lower() for text in (
        "closed channel", "channel closed", "channel is closed", "should create connection first"))


def transient(exc: Exception) -> bool:
    if isinstance(exc, (SafetyPause, Stopping)):
        return False
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and (exc.response.status_code >= 500 or exc.response.status_code in {408, 429})
    if isinstance(exc, (TimeoutError, ConnectionError, requests.ConnectionError, requests.Timeout)):
        return True
    if closed_channel(exc):
        return True
    text = str(exc).lower()
    # Milvus reports this while QueryCoord rebuilds a loaded collection after
    # recovery. Wait behind the Strong probe gate and reconcile uncertain
    # writes; other code-106 faults still require a safety pause.
    if isinstance(exc, MilvusException) and exc.code == 106 and "collection on recovering" in text:
        return True
    return any(word in text for word in (
        "timeout", "timed out", "unavailable", "deadline_exceeded", "deadline exceeded", "resource_exhausted",
        "failed to connect", "connection refused", "connection reset", "channel closed",
        "not connected", "temporarily", "rate limit", "too many requests"))


class Status:
    def __init__(self, settings: Settings, stop: threading.Event):
        self.settings, self.stop = settings, stop
        self.lock = threading.RLock()
        self.stage = "starting"
        self.shard = ""
        self.waiters: dict[int, str] = {}
        self.storage_error: str | None = None
        self.storage_waiters: set[int] = set()
        self.pause_lock = threading.Lock()
        self.safety: str | None = None
        self.mongo_error: str | None = None
        self.snapshot: dict[str, Any] = {}
        self.session_processed_start: int | None = None
        self.started = time.time()
        self.last_log = 0.0
        self.last_progress_epoch = self.started
        self.rate_samples: list[tuple[float, int]] = []
        self.backpressure: dict[str, Any] | None = None
        self.active_rpcs: dict[int, dict[str, Any]] = {}
        self.last_rpc: dict[str, Any] | None = None
        self.operation: dict[str, Any] | None = None
        self.last_dependency_error: dict[str, Any] | None = None
        self.connection_rebuilds = 0

    def document(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            processed = self.snapshot.get("processed", 0)
            if self.rate_samples and processed != self.rate_samples[-1][1]:
                self.last_progress_epoch = now
            self.rate_samples.append((now, processed))
            while len(self.rate_samples) > 1 and self.rate_samples[1][0] <= now - 60:
                self.rate_samples.pop(0)
            previous_time, previous_count = self.rate_samples[0]
            session_processed = processed - (self.session_processed_start or 0)
            elapsed = max(0, now - self.started)
            state = ("safety_paused" if self.safety else "waiting_storage" if self.storage_error or self.storage_waiters
                     else "waiting_dependencies" if self.waiters or self.backpressure else self.stage)
            return {**self.snapshot, "state": state, "current_shard": self.shard,
                    "operation": ({**self.operation, "elapsed_seconds": max(0, now - self.operation["started_epoch"])}
                                  if self.operation else None),
                    "last_dependency_error": self.last_dependency_error,
                    "milvus_connection_rebuilds": self.connection_rebuilds,
                    "elapsed_seconds": elapsed, "session_processed": session_processed,
                    "processed_per_second_since_start": session_processed / max(1, elapsed),
                    "processed_per_second_recent": (processed - previous_count) / max(1, now - previous_time),
                    "seconds_without_confirmed_progress": max(0, now - self.last_progress_epoch),
                    "milvus_backpressure": self.backpressure,
                    "max_write_rows_per_second": self.settings.max_write_rows_per_second,
                    "active_milvus_rpcs": [{**rpc, "elapsed_seconds": max(0, now - rpc["started_epoch"])}
                                           for rpc in self.active_rpcs.values()],
                    "last_milvus_rpc": self.last_rpc,
                    "waiting_for": list(self.waiters.values()), "safety_error": self.safety,
                    "storage_error": self.storage_error or ("checkpoint unavailable" if self.storage_waiters else None), "mongo_error": self.mongo_error,
                    "heartbeat_epoch": time.time(), "started_epoch": self.started,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def begin(self, stage: str, **details) -> None:
        with self.lock:
            self.stage = stage
            self.operation = {"name": stage, "started_epoch": time.time(), **details}

    def dependency_error(self, name: str, exc: Exception) -> None:
        # Keep diagnostics bounded and omit exception messages, which can contain URIs/credentials.
        with self.lock:
            self.last_dependency_error = {"stage": name, "type": type(exc).__name__,
                                          "closed_channel": closed_channel(exc), "epoch": time.time()}
        LOGGER.warning("dependency retry: stage=%s type=%s closed_channel=%s",
                       name, type(exc).__name__, closed_channel(exc))

    def publish(self) -> None:
        document = self.document()
        try:
            atomic_json(self.settings.state_dir / "heartbeat.json", document)
            # There is a single publisher, avoiding atomic temporary-file races.
            atomic_json(self.settings.state_dir / "progress.json", document)
            self.storage_error = None
        except OSError as exc:
            self.storage_error = f"heartbeat persistence failed: {type(exc).__name__}"
            LOGGER.error("state directory unavailable; new writes paused")
        if time.time() - self.last_log >= 30:
            LOGGER.info("progress %s", encoded(document))
            self.last_log = time.time()

    def loop(self) -> None:
        while not self.stop.is_set():
            self.publish()
            self.stop.wait(15)

    def storage(self, action: Callable):
        ident = threading.get_ident()
        while True:
            try:
                result = action()
                with self.lock:
                    self.storage_waiters.discard(ident)
                return result
            except (sqlite3.OperationalError, OSError):
                with self.lock:
                    self.storage_waiters.add(ident)
                LOGGER.error("checkpoint persistence unavailable; waiting before any new write")
                if self.stop.wait(self.settings.recovery_initial):
                    raise Stopping()

    def guard(self) -> None:
        if self.stop.is_set():
            raise Stopping()
        if self.safety:
            raise SafetyPause(self.safety)
        while (self.storage_error or self.storage_waiters) and not self.stop.is_set():
            if self.safety:
                raise SafetyPause(self.safety)
            self.stop.wait(1)
        if self.stop.is_set():
            raise Stopping()

    def pause(self, reason: str) -> None:
        # The in-memory latch stops the other writer before persistence retries.
        self.safety = self.safety or reason
        with self.pause_lock:
            self.storage(lambda: atomic_json(self.settings.state_dir / "safety_pause.json",
                                             {"reason": self.safety, "time": time.time()}))
        LOGGER.error("SAFETY PAUSE: %s; no automatic reset", reason)


class WritePacer:
    """Space batches across both writers, with no accumulated catch-up burst.

    This is an admission limit, not a throughput promise. Strong validation and
    backpressure still run after the wait and before an upsert is submitted.
    """

    def __init__(self, status: Status):
        self.status = status
        self.lock = threading.Lock()
        self.next_batch = 0.0

    def ready(self, rows: int) -> None:
        rate = self.status.settings.max_write_rows_per_second
        self.status.guard()
        if not rate or not rows:
            return
        with self.lock:
            while True:
                self.status.guard()
                delay = self.next_batch - time.monotonic()
                if delay <= 0:
                    self.next_batch = time.monotonic() + rows / rate
                    return
                if self.status.stop.wait(min(delay, 0.1)):
                    raise Stopping()


class MilvusBackpressure:
    """One read-only recovery probe for the whole worker, never an upsert probe.

    Already submitted RPCs may drain. No new Milvus action is admitted after a
    transient failure until a Strong read succeeds. Uncertain writes are still
    reconciled by write_batch before any retry.
    """

    def __init__(self, status: Status):
        self.status = status
        self.lock = threading.Lock()
        self.opened = False
        self.probing = False
        self.generation = 0
        self.failures = 0
        self.next_probe = 0.0

    def failed(self, name: str) -> None:
        with self.lock:
            self.generation += 1
            if not self.opened:
                self.opened = True
                self._defer(name)

    def _defer(self, name: str) -> None:
        delay = min(self.status.settings.recovery_max,
                    self.status.settings.recovery_initial * 2 ** min(self.failures, 8))
        self.failures += 1
        self.next_probe = time.monotonic() + delay
        with self.status.lock:
            self.status.backpressure = {"state": "cooldown", "failed_stage": name,
                                        "next_probe_epoch": time.time() + delay,
                                        "probe": "read_only_strong", "delay_seconds": delay}
        LOGGER.warning("Milvus backpressure: stop new writes; one Strong probe in %.1fs (%s)", delay, name)

    def ready(self, probe: Callable) -> None:
        while True:
            self.status.guard()
            with self.lock:
                if not self.opened:
                    return
                claim = not self.probing and time.monotonic() >= self.next_probe
                if claim:
                    self.probing = True
                    generation = self.generation
                    with self.status.lock:
                        self.status.backpressure = {**self.status.backpressure, "state": "probing"}
            if not claim:
                if self.status.stop.wait(0.05):
                    raise Stopping()
                continue
            try:
                probe()
            except Exception as exc:
                self.status.dependency_error("strong_recovery_probe", exc)
                with self.lock:
                    self.probing = False
                    if transient(exc):
                        self._defer("strong_recovery_probe")
                    else:
                        raise
            else:
                with self.lock:
                    self.probing = False
                    if generation == self.generation:
                        self.opened = False
                        self.failures = 0
                        with self.status.lock:
                            self.status.backpressure = None
                        LOGGER.info("Milvus Strong read recovered; resume bounded batches with reconciliation")


class DurableState:
    """Uses new tables in the existing DB; legacy tables/reports remain intact."""
    def __init__(self, checkpoint: Checkpoint, status: Status):
        self.cp, self.status = checkpoint, status
        self.db, self.lock = checkpoint.db, checkpoint.lock
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.execute("CREATE TABLE IF NOT EXISTS bf_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                self.db.execute("CREATE TABLE IF NOT EXISTS bf_shards("
                                "prefix TEXT PRIMARY KEY,planned INTEGER NOT NULL DEFAULT 0,"
                                "size INTEGER NOT NULL DEFAULT 0,status TEXT NOT NULL DEFAULT 'pending')")
                self.db.execute("CREATE TABLE IF NOT EXISTS bf_batches("
                                "prefix TEXT NOT NULL,number INTEGER NOT NULL,ids TEXT NOT NULL,"
                                "phase TEXT NOT NULL DEFAULT 'pending',payload TEXT,result TEXT,"
                                "processed INTEGER NOT NULL DEFAULT 0,updated INTEGER NOT NULL DEFAULT 0,"
                                "unchanged INTEGER NOT NULL DEFAULT 0,failed INTEGER NOT NULL DEFAULT 0,"
                                "PRIMARY KEY(prefix,number))")
                self.db.execute("CREATE TABLE IF NOT EXISTS bf_failures("
                                "failure_id TEXT PRIMARY KEY,document TEXT NOT NULL,sent INTEGER NOT NULL DEFAULT 0)")
                if not self.get_meta("schema_version"):
                    self.db.execute("INSERT INTO bf_meta(key,value) VALUES('legacy_progress',?)", (encoded(self.cp.summary()),))
                    for prefix, old_status, scanned, updated, failed in self.db.execute(
                            "SELECT prefix,status,scanned,updated,failed FROM shards").fetchall():
                        completed = old_status == "completed"
                        self.db.execute("INSERT OR IGNORE INTO bf_shards(prefix,planned,size,status) VALUES(?,?,?,?)",
                                        (prefix, int(completed), scanned if completed else 0,
                                         "completed" if completed else "pending"))
                        if completed:
                            self.db.execute("INSERT INTO bf_batches(prefix,number,ids,phase,processed,updated,unchanged,failed) "
                                            "VALUES(?,0,'[]','done',?,?,?,?)", (prefix, scanned, updated, scanned-updated-failed, failed))
                    self.db.execute("INSERT INTO bf_meta(key,value) VALUES('schema_version','2')")

    def get_meta(self, key: str, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM bf_meta WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO bf_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (key, encoded(value)))

    def shard(self, prefix: str):
        with self.lock:
            return self.db.execute("SELECT planned,size,status FROM bf_shards WHERE prefix=?", (prefix,)).fetchone()

    def plan(self, prefix: str, ids: list[str], batch_size: int) -> None:
        if len(ids) != len(set(ids)) or any(not HEX_ID.fullmatch(key) or key[:2] != prefix for key in ids):
            raise SafetyPause("invalid or duplicate ID in shard manifest")
        with self.lock, self.db:
            if self.shard(prefix)[0]:
                return
            for number, offset in enumerate(range(0, len(ids), batch_size)):
                self.db.execute("INSERT INTO bf_batches(prefix,number,ids) VALUES(?,?,?)",
                                (prefix, number, encoded(ids[offset:offset + batch_size])))
            self.db.execute("UPDATE bf_shards SET planned=1,size=?,status='running' WHERE prefix=?", (len(ids), prefix))

    def pending(self, prefix: str):
        with self.lock:
            return [(n, json.loads(ids)) for n, ids in self.db.execute(
                "SELECT number,ids FROM bf_batches WHERE prefix=? AND phase<>'done' ORDER BY number", (prefix,))]

    def planned_ids(self, prefix: str) -> set[str]:
        with self.lock:
            planned, size, _ = self.shard(prefix)
            if not planned:
                return set()
            ids: set[str] = set()
            for (encoded_ids,) in self.db.execute("SELECT ids FROM bf_batches WHERE prefix=?", (prefix,)):
                batch = json.loads(encoded_ids)
                if len(batch) != len(set(batch)) or any(key in ids for key in batch):
                    raise SafetyPause("duplicate ID in persisted shard manifest")
                ids.update(batch)
            if len(ids) != size:
                raise SafetyPause("persisted shard manifest does not account for every planned ID")
            return ids

    def extend(self, prefix: str, new_ids: list[str], batch_size: int) -> None:
        if not new_ids:
            return
        if (len(new_ids) != len(set(new_ids))
                or any(not HEX_ID.fullmatch(key) or key[:2].lower() != prefix for key in new_ids)):
            raise SafetyPause("invalid or duplicate newly discovered ID")
        with self.lock, self.db:
            if not self.shard(prefix)[0]:
                raise SafetyPause("cannot extend an unplanned shard")
            if set(new_ids) & self.planned_ids(prefix):
                raise SafetyPause("new ID is already present in a shard manifest")
            number = self.db.execute(
                "SELECT coalesce(max(number),-1)+1 FROM bf_batches WHERE prefix=?", (prefix,)
            ).fetchone()[0]
            for offset in range(0, len(new_ids), batch_size):
                self.db.execute("INSERT INTO bf_batches(prefix,number,ids) VALUES(?,?,?)",
                                (prefix, number, encoded(new_ids[offset:offset + batch_size])))
                number += 1
            self.db.execute("UPDATE bf_shards SET size=size+?,status='running' WHERE prefix=?",
                            (len(new_ids), prefix))

    def batch(self, prefix: str, number: int):
        with self.lock:
            row = self.db.execute("SELECT phase,payload,result,ids FROM bf_batches WHERE prefix=? AND number=?",
                                  (prefix, number)).fetchone()
            if row is None:
                raise SafetyPause("batch missing from fixed manifest")
            return {"phase": row[0], "payload": json.loads(row[1]) if row[1] else [],
                    "result": json.loads(row[2]) if row[2] else {}, "ids": json.loads(row[3])}

    def prepare(self, prefix: str, number: int, payload: list[dict], result: dict, failures: list[dict]) -> None:
        with self.lock, self.db:
            if self.batch(prefix, number)["phase"] != "pending":
                return
            target = self.get_meta("target")
            job = self.get_meta("job_id")
            for failure in failures:
                key = hashlib.sha256(encoded([job, failure["id"], failure["stage"], failure["reason"]]).encode()).hexdigest()
                document = {"_id": key, "job_id": job, "target": target, "doc_id": failure["id"],
                            "shard": prefix, "batch": number, "status": "pending", "attempt_count": 1,
                            "first_seen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "last_seen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            **{k: v for k, v in failure.items() if k != "id"}}
                self.db.execute("INSERT OR IGNORE INTO bf_failures(failure_id,document) VALUES(?,?)", (key, encoded(document)))
            self.db.execute("UPDATE bf_batches SET phase='inflight',payload=?,result=?,processed=?,updated=?,unchanged=?,failed=? "
                            "WHERE prefix=? AND number=?", (encoded(payload), encoded(result), result["processed"],
                            result["updated"], result["unchanged"], result["failed"], prefix, number))

    def acknowledge(self, prefix: str, number: int) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE bf_batches SET phase='acked' WHERE prefix=? AND number=? AND phase='inflight'",
                            (prefix, number))

    def confirm_flush(self, prefix: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE bf_batches SET phase='done',payload=NULL WHERE prefix=? AND phase='acked'", (prefix,))

    def finish_shard(self, prefix: str) -> None:
        with self.lock, self.db:
            pending = self.db.execute("SELECT count(*) FROM bf_batches WHERE prefix=? AND phase<>'done'", (prefix,)).fetchone()[0]
            if pending:
                raise SafetyPause("cannot complete a shard with unconfirmed batches")
            self.db.execute("UPDATE bf_shards SET status='completed' WHERE prefix=?", (prefix,))

    def summary(self):
        with self.lock:
            row = self.db.execute("SELECT coalesce(sum(processed),0),coalesce(sum(updated),0),"
                                  "coalesce(sum(unchanged),0),coalesce(sum(failed),0) FROM bf_batches WHERE phase='done'").fetchone()
            shards = dict(self.db.execute("SELECT status,count(*) FROM bf_shards GROUP BY status"))
            scanned = self.db.execute("SELECT coalesce(sum(size),0) FROM bf_shards").fetchone()[0]
            outbox = self.db.execute("SELECT count(*) FROM bf_failures WHERE sent=0").fetchone()[0]
            acked = self.db.execute("SELECT coalesce(sum(processed),0) FROM bf_batches WHERE phase='acked'").fetchone()[0]
            registered = self.db.execute("SELECT count(*) FROM bf_failures").fetchone()[0]
            return {"scanned": scanned, "processed": row[0], "updated": row[1], "unchanged": row[2],
                    "failed": row[3], "registered_failures": registered, "acknowledged_pending_flush": acked, "mongo_pending": outbox, "shards": shards}

    def unsent(self, limit=200):
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute("SELECT document FROM bf_failures WHERE sent=0 LIMIT ?", (limit,))]

    def mark_sent(self, ids: list[str]) -> None:
        with self.lock, self.db:
            self.db.executemany("UPDATE bf_failures SET sent=1 WHERE failure_id=?", [(key,) for key in ids])


def check_mongo(settings: Settings) -> None:
    if not settings.mongo_uri:
        raise SafetyPause("MONGO_URI is required for full backfill")
    with MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000,
                     socketTimeoutMS=10000) as client:
        client.admin.command("ping")
        # Read-only namespace inspection. Do not initialize the existing sync manager.
        client[settings.mongo_db].list_collection_names(filter={"name": settings.mongo_failure_collection})


class MongoReporter:
    def __init__(self, state: DurableState, settings: Settings, status: Status, collection=None):
        self.state, self.settings, self.status = state, settings, status
        self.client = None
        self.collection = collection

    def sync_once(self) -> int:
        docs = self.status.storage(self.state.unsent)
        if not docs:
            return 0
        if self.collection is None:
            self.client = MongoClient(self.settings.mongo_uri, serverSelectionTimeoutMS=10000,
                                      connectTimeoutMS=10000, socketTimeoutMS=10000, retryWrites=False)
            self.collection = self.client[self.settings.mongo_db][self.settings.mongo_failure_collection]
        writes = [UpdateOne({"_id": d["_id"]}, {"$set": {k: v for k, v in d.items() if k != "_id"}}, upsert=True) for d in docs]
        self.collection.bulk_write(writes, ordered=False)
        self.status.storage(lambda: self.state.mark_sent([d["_id"] for d in docs]))
        self.status.mongo_error = None
        return len(docs)

    def loop(self) -> None:
        while not self.status.stop.is_set():
            try:
                count = self.sync_once()
                if count:
                    continue
            except Stopping:
                break
            except Exception as exc:
                # Do not log connection strings, exception text can contain credentials.
                self.status.mongo_error = type(exc).__name__
                LOGGER.warning("Mongo failure sync deferred (%s); durable local outbox retained", type(exc).__name__)
            self.status.stop.wait(15)

    def close(self):
        if self.client is not None:
            self.client.close()


@dataclass
class Prepared:
    ids: list[str]
    rows: list[dict]
    failures: list[dict]


class ResilientWorker(BackfillWorker):
    def __init__(self, settings, *, status, state, **kwargs):
        super().__init__(settings, **kwargs)
        self.status, self.state = status, state
        self.stop_event = status.stop
        self.audit_metrics = {"solr_seconds": 0.0, "write_seconds": 0.0, "flush_seconds": 0.0}
        self.milvus_backpressure = MilvusBackpressure(status)
        self.write_pacer = WritePacer(status)
        self.recovery_probe_id: str | None = None
        self._connection_condition = threading.Condition(threading.RLock())
        self._connection_users: dict[int, int] = {}
        self._connection_resetting = False
        self._connection_broken = False
        self._reconnect_verification_required = False
        self._last_collection_check = 0.0

    def _client_action(self, action: Callable):
        """Lease the current client, including iterator lifetime and nested RPCs."""
        ident = threading.get_ident()
        with self._connection_condition:
            while self._connection_resetting and ident not in self._connection_users:
                self.status.guard()
                self._connection_condition.wait(0.1)
            self.status.guard()
            if self._connection_broken:
                raise ConnectionError("Milvus closed channel requires a fresh connection")
            self._connection_users[ident] = self._connection_users.get(ident, 0) + 1
        try:
            return action()
        except Exception as exc:
            if closed_channel(exc):
                with self._connection_condition:
                    self._connection_broken = True
            raise
        finally:
            with self._connection_condition:
                self._connection_users[ident] -= 1
                if not self._connection_users[ident]:
                    del self._connection_users[ident]
                self._connection_condition.notify_all()

    def _repair_milvus(self):
        # Only the shared read-only probe calls this. Drain leases before closing
        # a client so that reconnect never cancels another writer's active RPC.
        with self._connection_condition:
            if not self._connection_broken:
                return
            self._connection_resetting = True
        try:
            with self._connection_condition:
                while self._connection_users:
                    self.status.guard()
                    self._connection_condition.wait(0.1)
            self.status.guard()
            client = MilvusClient(uri=self.settings.milvus_uri, db_name=self.settings.milvus_db_name,
                                  timeout=min(15, self.settings.milvus_timeout), dedicated=True)
            old = self.milvus
            with self._milvus_lock:
                self._milvus_clients.append(client)
                self.milvus = client
            self._reconnect_verification_required = True
            with self._connection_condition:
                self._connection_broken = False
            with self.status.lock:
                self.status.connection_rebuilds += 1
            try:
                old.close()
            except Exception:
                LOGGER.warning("retired Milvus client cleanup failed; retained for final cleanup")
            else:
                with self._milvus_lock:
                    self._milvus_clients = [item for item in self._milvus_clients if item is not old]
            LOGGER.info("Milvus client rebuilt after draining active RPCs; Strong verification still required")
        finally:
            with self._connection_condition:
                self._connection_resetting = False
                self._connection_condition.notify_all()

    def _write_json(self, path: Path, data: Any) -> None:
        # Only the heartbeat thread publishes progress.json.
        if path != self.progress_path:
            self.status.storage(lambda: atomic_json(path, data))

    def saved(self, prefix, number):
        return self.status.storage(lambda: self.state.batch(prefix, number))

    def meta(self, key, default=None):
        return self.status.storage(lambda: self.state.get_meta(key, default))

    def _query_ids(self, prefix, limit=None):
        ids = super()._query_ids(prefix, limit)
        # The legacy iterator returns early on SIGTERM. Such a partial scan must
        # never become a fixed shard manifest, or the remaining IDs are omitted.
        self.status.guard()
        return ids

    def external(self, name: str, action: Callable):
        cycle = 0
        ident = threading.get_ident()
        milvus_action = name not in {"solr", "solr_check"}
        try:
            while True:
                for attempt in range(self.settings.max_retries + 1):
                    self.status.guard()
                    if milvus_action:
                        self.milvus_backpressure.ready(self._probe_milvus)
                    try:
                        if milvus_action and name not in {"milvus_partial_upsert", "milvus_readiness"}:
                            return self._client_action(action)
                        return action()
                    except (SafetyPause, Stopping):
                        raise
                    except Exception as exc:
                        if not transient(exc):
                            raise SafetyPause(f"{name}: {exc}") from exc
                        self.status.dependency_error(name, exc)
                        with self.status.lock:
                            self.status.waiters[ident] = name
                        if milvus_action:
                            self.milvus_backpressure.failed(name)
                        if attempt < self.settings.max_retries:
                            if self.stop_event.wait(self.settings.retry_backoff * (attempt + 1)):
                                raise Stopping() from exc
                with self.status.lock:
                    self.status.waiters[ident] = name
                delay = min(self.settings.recovery_max, self.settings.recovery_initial * 2 ** min(cycle, 8))
                LOGGER.warning("dependency %s unavailable; waiting %.0fs, batch retained", name, delay)
                if self.stop_event.wait(delay):
                    raise Stopping()
                cycle += 1
        finally:
            with self.status.lock:
                self.status.waiters.pop(ident, None)

    def _probe_milvus(self):
        # An id-only Strong read verifies the time-tick barrier, unlike /healthz
        # or Eventually reads. It neither submits data nor changes consistency.
        self._repair_milvus()
        timeout = min(15, self.settings.milvus_timeout)
        if self.recovery_probe_id:
            rows = self.milvus_call("strong_recovery_probe", lambda: self.milvus.get(
                collection_name=self.settings.collection, ids=[self.recovery_probe_id],
                output_fields=["id"], consistency_level="Strong", timeout=timeout))
            if len(rows) != 1 or rows[0]["id"] != self.recovery_probe_id:
                raise SafetyPause("recovery probe record missing; refusing further writes")
        else:
            self.milvus_call("strong_recovery_probe", lambda: self.milvus.query(
                collection_name=self.settings.collection, filter="", limit=1,
                output_fields=["id"], consistency_level="Strong", timeout=timeout))
        if self._reconnect_verification_required:
            baseline_path = self.settings.state_dir / "full_baseline.json"
            baseline = self.meta("baseline")
            if baseline:
                now = self.milvus_call("reconnect_collection_verify", self._collection_evidence)
                self.validate_evidence(baseline, now)
            history = []
            if baseline_path.exists():
                history.append(json.loads(baseline_path.read_text(encoding="utf-8"))["original_hashes"])
            shard_path = self.settings.state_dir / f"shard_{self.status.shard}_original_hashes.json"
            if shard_path.exists():
                history.append(json.loads(shard_path.read_text(encoding="utf-8")))
            for hashes in history:
                self.milvus_call("reconnect_history_verify", lambda hashes=hashes: self.verify_integrity(hashes))
            self._reconnect_verification_required = False

    def milvus_call(self, phase: str, action: Callable):
        ident = threading.get_ident()
        started = time.monotonic()
        success = False
        with self.status.lock:
            self.status.active_rpcs[ident] = {"phase": phase, "started_epoch": time.time()}
        try:
            result = self._client_action(action)
            success = True
            return result
        finally:
            elapsed = time.monotonic() - started
            with self.status.lock:
                self.status.active_rpcs.pop(ident, None)
                self.status.last_rpc = {"phase": phase, "seconds": elapsed, "success": success}
            with self._metrics_lock:
                self.audit_metrics[phase + "_seconds"] = self.audit_metrics.get(phase + "_seconds", 0) + elapsed
            threshold = self.settings.milvus_slow_request_seconds
            if success and phase != "strong_recovery_probe" and threshold and elapsed >= threshold:
                self.milvus_backpressure.failed("slow_" + phase)

    def guarded(self, action: Callable):
        try:
            return action()
        except Stopping:
            raise
        except Exception as exc:
            # Writers see the latch immediately, before the supervisor drains futures.
            self.status.pause(str(exc))
            raise SafetyPause(str(exc)) from exc

    def _read(self, ids: list[str]) -> Prepared:
        def request():
            response = self._http_session().post(self.settings.solr_url, data={
                "q": "{!terms f=id}" + ",".join(ids), "fl": ",".join(SOLR_QUERY_FIELDS),
                "rows": len(ids), "wt": "json"}, timeout=self.settings.solr_timeout)
            response.raise_for_status()
            body = response.json()
            header = body.get("responseHeader", {})
            if header.get("status", 0) or header.get("partialResults"):
                raise TimeoutError("Solr incomplete response")
            docs = body["response"]["docs"]
            found = [str(d.get("id")) for d in docs]
            if len(set(found)) != len(found) or not set(found).issubset(ids):
                raise SafetyPause("Solr returned duplicate or unrequested IDs")
            return docs
        started = time.monotonic()
        docs = self.external("solr", request)
        rows, failures = [], []
        found = {str(d["id"]) for d in docs}
        for key in ids:
            if key not in found:
                failures.append({"id": key, "stage": "solr_lookup", "reason": "source_missing", "error": "Solr returned no document"})
        for doc in docs:
            try:
                rows.append(self.extract(doc))
            except (ValueError, TypeError) as exc:
                failures.append({"id": str(doc["id"]), "stage": "transform", "reason": "invalid_value", "error": str(exc)})
        if len(rows) + len(failures) != len(ids):
            raise SafetyPause("prepared result does not account for every manifest ID")
        with self._metrics_lock:
            self.audit_metrics["solr_seconds"] += time.monotonic() - started
        return Prepared(ids, rows, failures)

    def strong(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        self.recovery_probe_id = ids[0]
        rows = self.milvus_call("strong_read", lambda: self.milvus.get(
            collection_name=self.settings.collection, ids=ids,
            output_fields=["id", *TARGET_FIELDS], consistency_level="Strong", timeout=self.settings.milvus_timeout))
        if len(rows) != len(ids) or {r["id"] for r in rows} != set(ids):
            raise SafetyPause("source record no longer exists; refusing partial upsert")
        if any(not set(["id", *TARGET_FIELDS]).issubset(r) for r in rows):
            raise SafetyPause("Strong read omitted target fields")
        return {r["id"]: r for r in rows}

    @staticmethod
    def conflicting_fields(row: dict, old: dict) -> list[str]:
        return [name for name in TARGET_FIELDS
                if row[name] is not None and old[name] is not None
                and old[name] != row[name]]

    def write_batch(self, prefix: str, number: int, prepared: Prepared | None) -> dict:
        def perform():
            self.status.guard()
            saved = self.saved(prefix, number)
            if saved["phase"] in {"acked", "done"}:
                return saved["result"]
            if saved["phase"] == "pending":
                if prepared is None or prepared.ids != saved["ids"]:
                    raise SafetyPause("write is outside the fixed batch manifest")
                rows = prepared.rows
                current = self.external("milvus_strong", lambda: self.strong([r["id"] for r in rows]))
                payload = []
                conflicts = []
                for row in rows:
                    self._assert_write_payload(row)
                    old = current[row["id"]]
                    # An earlier parser may have filled a different value.
                    # Preserve it, but do not silently count the row as fully
                    # compatible with the current production parsing rules.
                    differing = self.conflicting_fields(row, old)
                    if differing:
                        conflicts.append({"id": row["id"], "stage": "parser_compare",
                                          "reason": "existing_value_differs",
                                          "error": "existing target differs from v4 parser: "
                                                   + ",".join(differing)})
                        continue
                    merged = {"id": row["id"], **{name: old[name] if old[name] is not None else row[name] for name in TARGET_FIELDS}}
                    self._assert_write_payload(merged)
                    if any(merged[name] != old[name] for name in TARGET_FIELDS):
                        payload.append(merged)
                result = {"processed": len(prepared.ids), "updated": len(payload),
                          "unchanged": len(rows) - len(payload) - len(conflicts),
                          "failed": len(prepared.failures) + len(conflicts),
                          "null_counts": {name: sum(r[name] is None for r in rows) for name in TARGET_FIELDS}}
                self.status.storage(lambda: self.state.prepare(prefix, number, payload, result,
                                                              [*prepared.failures, *conflicts]))
                saved = self.saved(prefix, number)
            payload = saved["payload"]
            for row in payload:
                self._assert_write_payload(row)
                if row["id"] not in saved["ids"]:
                    raise SafetyPause("persisted payload is outside the fixed manifest")

            def submit_or_reconcile():
                self.status.guard()
                self.write_pacer.ready(len(payload))
                self.milvus_backpressure.ready(self._probe_milvus)
                # Reconcile every uncertain request before retrying. Existing records only.
                current = self.strong([r["id"] for r in payload])
                needed = []
                for row in payload:
                    old = current[row["id"]]
                    if any(old[name] is not None and old[name] != row[name] for name in TARGET_FIELDS):
                        raise SafetyPause("target value changed during an uncertain write")
                    if any(old[name] != row[name] for name in TARGET_FIELDS):
                        needed.append(row)
                if needed:
                    self.status.guard()
                    self.milvus_backpressure.ready(self._probe_milvus)
                    for row in needed:
                        self._assert_write_payload(row)
                    response = self.milvus_call("partial_upsert", lambda: self.milvus.upsert(
                        collection_name=self.settings.collection, data=needed,
                        partial_update=True, timeout=self.settings.milvus_timeout))
                    if int(response.get("upsert_count", 0)) != len(needed):
                        raise TimeoutError("uncertain upsert acknowledgement; Strong reconciliation required")
            started = time.monotonic()
            if payload:
                self.external("milvus_partial_upsert", submit_or_reconcile)
            self.status.storage(lambda: self.state.acknowledge(prefix, number))
            with self._metrics_lock:
                self.audit_metrics["write_seconds"] += time.monotonic() - started
            return saved["result"]
        return self.guarded(perform)

    def flush_confirm(self, prefix: str, reason: str) -> None:
        started = time.monotonic()
        self.external("milvus_flush", lambda: self.milvus_call("flush", lambda: self.milvus.flush(
            collection_name=self.settings.collection, timeout=self.settings.milvus_timeout)))
        self.status.storage(lambda: self.state.confirm_flush(prefix))
        self.refresh()
        with self._metrics_lock:
            self.audit_metrics["flush_seconds"] += time.monotonic() - started
        LOGGER.info("flush %s confirmed in %.3fs", reason, time.monotonic() - started)

    def refresh(self):
        snapshot = self.status.storage(self.state.summary)
        snapshot["logical_rows_baseline"] = self.meta("logical_count")
        snapshot["max_logical_count"] = self.meta("max_logical_count")
        snapshot["elapsed_seconds"] = time.time() - self.status.started
        if self.status.session_processed_start is None:
            self.status.session_processed_start = snapshot["processed"]
            with self.status.lock:
                self.status.rate_samples = [(time.time(), snapshot["processed"])]
        snapshot["session_processed"] = snapshot["processed"] - self.status.session_processed_start
        snapshot["processed_per_second_since_start"] = snapshot["session_processed"] / max(1, snapshot["elapsed_seconds"])
        with self.status.lock:
            self.status.snapshot = snapshot

    def audit(self, baseline: dict):
        result = self.external("history_audit", lambda: self.verify_integrity(baseline))
        return result

    def evidence(self, baseline: dict):
        now = self.external("collection_evidence", self._collection_evidence)
        return self.validate_evidence(baseline, now)

    def validate_evidence(self, baseline: dict, now: dict):
        result = self._evidence_comparison(baseline, now)
        old_names = {field["name"] for field in baseline["schema"]["fields"]}
        new_names = {field["name"] for field in now["schema"]["fields"]}
        result["only_allowed_schema_additions"] = not (new_names - old_names - set(TARGET_FIELDS))
        highest = max(baseline["logical_row_count"], self.meta("max_logical_count", 0))
        # Keep small isolated fixtures strict without weakening the production
        # floor for the 52M+ target collection.
        effective_floor = (LOGICAL_COUNT_SAFETY_FLOOR
                           if highest >= LOGICAL_COUNT_SAFETY_FLOOR else highest)
        result["logical_rows_non_decreasing"] = now["logical_row_count"] >= highest
        result["logical_rows_above_safety_floor"] = now["logical_row_count"] >= effective_floor
        result["logical_count_safety_floor"] = effective_floor
        result["configured_logical_count_safety_floor"] = LOGICAL_COUNT_SAFETY_FLOOR
        result["highest_logical_rows_before"] = highest
        result["passed"] = all(result[name] for name in (
            "old_schema_unchanged", "indexes_unchanged", "collection_identity_unchanged",
            "only_allowed_schema_additions", "logical_rows_above_safety_floor"))
        if not result["passed"]:
            failed_checks = [name for name in (
                "old_schema_unchanged", "indexes_unchanged", "collection_identity_unchanged",
                "only_allowed_schema_additions") if not result[name]]
            if not result["logical_rows_above_safety_floor"]:
                # Keep the historical evidence label for existing dashboards.
                failed_checks.append("logical_rows_non_decreasing")
            self.status.storage(lambda: atomic_json(self.settings.state_dir / "safety_collection_evidence.json",
                                                    {**result, "failed_checks": failed_checks, "epoch": time.time()}))
            raise SafetyPause("collection identity/schema/index changed or logical record count decreased below safety floor; "
                              f"checks={','.join(failed_checks)}; count={now['logical_row_count']}; "
                              f"floor={effective_floor}")
        if now["logical_row_count"] > highest:
            self.status.storage(lambda: self.state.set_meta("max_logical_count", now["logical_row_count"]))
        self._last_collection_check = time.monotonic()
        return result

    def reconcile_shard_ids(self, prefix: str) -> int:
        """Append newly visible IDs, but never accept loss of a planned ID."""
        self.status.begin("reconciling_ids", shard=prefix, shards_total=len(SHARDS),
                          shard_index=SHARDS.index(prefix) + 1)
        current = sorted(self.external("milvus_id_scan", lambda: self._query_ids(prefix)))
        planned, _, _ = self.status.storage(lambda: self.state.shard(prefix))
        if not planned:
            self.status.storage(lambda: self.state.plan(prefix, current, self.settings.batch_size))
            return len(current)
        known = self.status.storage(lambda: self.state.planned_ids(prefix))
        missing = known - set(current)
        if missing:
            raise SafetyPause(f"{prefix}: {len(missing)} previously planned IDs disappeared")
        added = sorted(set(current) - known)
        if added:
            self.status.storage(lambda: self.state.extend(prefix, added, self.settings.batch_size))
            LOGGER.info("shard %s: appended %d new IDs to durable batches", prefix, len(added))
        return len(added)

    def canary(self, prefix: str, number: int, ids: list[str]):
        path = self.settings.state_dir / "resilient_canary.json"
        if path.exists():
            before = json.loads(path.read_text(encoding="utf-8"))
            if before["prefix"] != prefix or before["number"] != number or before["ids"] != ids:
                raise SafetyPause("canary manifest changed across restart")
            self.audit(before["hashes"])
        else:
            hashes = self.external("canary_history_read", lambda: self._hash_records(ids))
            before = {"prefix": prefix, "number": number, "ids": ids, "hashes": hashes}
            self.status.storage(lambda: atomic_json(path, before))
        prepared = None
        if "expected" not in before:
            if self.saved(prefix, number)["phase"] != "pending":
                raise SafetyPause("canary expected values are missing after a write")
            prepared = self.guarded(lambda: self._read(ids))
            current = self.external("canary_strong", lambda: self.strong([r["id"] for r in prepared.rows]))
            before["expected"] = [
                {"id": row["id"], **{
                    name: (current[row["id"]][name]
                           if self.conflicting_fields(row, current[row["id"]])
                           or current[row["id"]][name] is not None else row[name])
                    for name in TARGET_FIELDS
                }} for row in prepared.rows
            ]
            before["prepared"] = {"ids": prepared.ids, "rows": prepared.rows, "failures": prepared.failures}
            self.status.storage(lambda: atomic_json(path, before))
        else:
            prepared = Prepared(**before["prepared"])
        expected = before["expected"]
        self.write_batch(prefix, number, prepared)
        self.flush_confirm(prefix, "canary")
        self.audit(before["hashes"])
        after = self.external("canary_target_verify", lambda: self.strong([r["id"] for r in expected]))
        if any(any(after[r["id"]][n] != r[n] for n in TARGET_FIELDS) for r in expected):
            raise SafetyPause("canary target values differ from expected transformation")
        collection = self.meta("baseline")
        checks = self.evidence(collection)
        report = {"passed": True, "sample_size": len(ids), "historical_changed_ids": [],
                  "target_values_verified": len(expected), "collection_checks": checks,
                  "batch_result": self.saved(prefix, number)["result"]}
        self.status.storage(lambda: atomic_json(self.settings.state_dir / "resilient_canary_report.json", report))
        self.status.storage(lambda: self.state.set_meta("canary_passed", True))
        LOGGER.info("CANARY PASSED: %d original records including full vectors verified", len(ids))

    def window(self, prefix: str, batches: list[tuple[int, list[str]]], readers, writers):
        pending, active, ready = {}, {}, []
        cursor = 0
        try:
            while cursor < len(batches) or pending or active or ready:
                self.status.guard()
                for future in list(pending):
                    if future.done():
                        number = pending.pop(future)
                        ready.append((number, future.result()))
                for future in list(active):
                    if future.done():
                        active.pop(future)
                        future.result()
                if not self.status.waiters:
                    while ready and len(active) < self.settings.concurrency:
                        number, prepared = ready.pop(0)
                        active[writers.submit(self.write_batch, prefix, number, prepared)] = number
                    while cursor < len(batches) and len(pending) + len(ready) < self.settings.concurrency:
                        number, ids = batches[cursor]
                        cursor += 1
                        phase = self.saved(prefix, number)["phase"]
                        if phase in {"inflight", "acked"}:
                            ready.append((number, None))
                        else:
                            pending[readers.submit(self.guarded, lambda ids=ids: self._read(ids))] = number
                waiting = [f for f in [*pending, *active] if not f.done()]
                if waiting:
                    wait(waiting, timeout=1, return_when=FIRST_COMPLETED)
                elif ready or cursor < len(batches):
                    self.stop_event.wait(0.01)
        except Stopping:
            raise
        except Exception as exc:
            self.status.pause(str(exc))
            raise
        finally:
            for future in pending:
                future.cancel()
            # In-flight RPCs must finish before an audit or close; do not auto-rollback.
            for future in [*pending, *active]:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass

    def process_shard(self, prefix: str, baseline: dict, readers, writers) -> None:
        planned, _, phase = self.status.storage(lambda: self.state.shard(prefix))
        if not planned:
            raise SafetyPause("cannot process an unplanned shard")
        batches = self.status.storage(lambda: self.state.pending(prefix))
        audit_path = self.settings.state_dir / f"shard_{prefix}_original_hashes.json"
        if phase == "completed" and not batches:
            self.status.begin("verifying_history", shard=prefix)
            if audit_path.exists():
                self.audit(json.loads(audit_path.read_text(encoding="utf-8")))
            return
        if batches:
            # Restart reconciliation can take many minutes. Recheck the floor
            # immediately before resuming writes, not only at the shard end.
            self.evidence(baseline)
        if audit_path.exists():
            originals = json.loads(audit_path.read_text(encoding="utf-8"))
            self.audit(originals)
        else:
            ids = sorted(self.status.storage(lambda: self.state.planned_ids(prefix)))
            count = min(len(ids), self.settings.sample_size)
            selected = [ids[i * len(ids) // count] for i in range(count)] if count else []
            originals = self.external("shard_history_read", lambda: self._hash_records(selected))
            self.status.storage(lambda: atomic_json(audit_path, originals))
        if batches and not self.meta("canary_passed"):
            canary_path = self.settings.state_dir / "resilient_canary.json"
            if canary_path.exists():
                c = json.loads(canary_path.read_text(encoding="utf-8"))
                self.canary(c["prefix"], c["number"], c["ids"])
            else:
                self.canary(prefix, batches[0][0], batches[0][1])
            batches = self.status.storage(lambda: self.state.pending(prefix))
        self.status.begin("running", shard=prefix, pending_batches=len(batches))
        for offset in range(0, len(batches), self.settings.flush_every_batches):
            if time.monotonic() - self._last_collection_check >= 60:
                self.evidence(baseline)
            self.window(prefix, batches[offset:offset + self.settings.flush_every_batches], readers, writers)
            self.flush_confirm(prefix, f"{prefix}_window")
        if batches:
            self.flush_confirm(prefix, f"{prefix}_final")
        self.audit(originals)
        self.evidence(baseline)
        self.status.storage(lambda: self.state.finish_shard(prefix))
        self.refresh()

    def run_resilient(self):
        # Show durable progress even when startup preflight is blocked.
        self.refresh()
        self.external("milvus_readiness", self._probe_milvus)
        self.external("preflight", lambda: self.preflight(mutate_schema=False))
        self.external("solr_check", self.check_solr)
        current = self.external("collection_evidence", self._collection_evidence)
        target = {"uri": self.settings.milvus_uri.rstrip("/"), "database": self.settings.milvus_db_name,
                  "collection": self.settings.collection, "collection_id": current["schema"].get("collection_id"),
                  "field_specs": FIELD_SPECS, "solr_mapping": SOLR_MAPPING,
                  "parser_version": PARSER_VERSION, "solr_url": self.settings.solr_url}
        self.status.storage(lambda: self.checkpoint.bind_target(target))
        existing_target = self.meta("target")
        if existing_target and existing_target != json.loads(encoded(target)):
            raise SafetyPause("resilient checkpoint belongs to another target")
        self.status.storage(lambda: self.state.set_meta("target", target))
        self.status.storage(lambda: self.state.set_meta("job_id", hashlib.sha256(encoded(target).encode()).hexdigest()))
        baseline_path = self.settings.state_dir / "full_baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.evidence(baseline["collection"])
            self.audit(baseline["original_hashes"])
        else:
            legacy = self.meta("legacy_progress", {})
            if legacy.get("shards") != {"pending": len(SHARDS)}:
                raise SafetyPause("legacy progress exists without original baseline")
            hashes = self.external("initial_history_read", self._integrity_snapshot)
            baseline = {"collection": current, "original_hashes": hashes}
            self.status.storage(lambda: atomic_json(baseline_path, baseline))
        self.status.storage(lambda: self.state.set_meta("baseline", baseline["collection"]))
        self.status.storage(lambda: self.state.set_meta("logical_count", baseline["collection"]["logical_row_count"]))
        self.external("schema_addition", lambda: self.preflight(mutate_schema=True))
        # Evidence comparisons permit only the explicitly added target fields.
        self.status.stage = "running"
        self.refresh()
        with ThreadPoolExecutor(max_workers=self.settings.concurrency, thread_name_prefix="solr-prefetch") as readers, \
                ThreadPoolExecutor(max_workers=self.settings.concurrency, thread_name_prefix="partial-writer") as writers:
            for prefix in SHARDS:
                self.status.guard()
                self.status.shard = prefix
                self.reconcile_shard_ids(prefix)
                self.process_shard(prefix, baseline["collection"], readers, writers)
                self.refresh()
            # A second Strong ID pass catches records added after an earlier
            # shard was visited. Repeat only when a pass discovers new IDs.
            last_unreconciled = None
            while True:
                self.status.stage = "verifying"
                added_total = 0
                for prefix in SHARDS:
                    self.status.guard()
                    self.status.shard = prefix
                    added = self.reconcile_shard_ids(prefix)
                    added_total += added
                    planned, _, phase = self.status.storage(lambda: self.state.shard(prefix))
                    if added or phase != "completed":
                        self.process_shard(prefix, baseline["collection"], readers, writers)
                self.external("final_flush", lambda: self.milvus.flush(
                    collection_name=self.settings.collection, timeout=self.settings.milvus_timeout))
                integrity = self.audit(baseline["original_hashes"])
                checks = self.evidence(baseline["collection"])
                self.refresh()
                summary = self.status.storage(self.state.summary)
                count = checks["logical_rows_after"]
                if summary["processed"] != summary["scanned"] or summary["scanned"] > count:
                    raise SafetyPause("durable ID batches do not reconcile with the collection")
                if not added_total and summary["scanned"] == count:
                    break
                unresolved = (summary["scanned"], count)
                if not added_total and unresolved == last_unreconciled:
                    raise SafetyPause("Strong ID scans cannot account for the logical record count")
                last_unreconciled = unresolved
                LOGGER.info("verification found %d new IDs; planned=%d current_count=%d; scanning again",
                            added_total, summary["scanned"], count)
                self.status.stage = "running"
        result = {**summary, "scan_complete": True, "all_records_enriched": not summary["failed"],
                  "historical_sample_check_passed": integrity["passed"], "collection_checks": checks,
                  "legacy_progress": self.meta("legacy_progress"), "stage_seconds_sum": self.audit_metrics}
        self.status.storage(lambda: atomic_json(self.summary_path, result))
        self.status.stage = "completed_with_failures" if summary["failed"] else "completed"
        self.status.shard = ""
        return result


def supervise(settings: Settings, startup_check: Callable | None = None) -> int:
    """Stay alive on dependency outages, latched faults, and successful completion."""
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    status = Status(settings, stop)
    def stopping(_signal, _frame):
        LOGGER.info("stop requested; retaining all state and draining active requests")
        stop.set()
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    worker = reporter = checkpoint = None
    heartbeat = publisher = None
    try:
        settings.validate()
        with state_lock(settings.state_dir):
            status.storage(lambda: (settings.state_dir / "worker.pid").write_text(str(os.getpid())))
            heartbeat = threading.Thread(target=status.loop, daemon=True, name="heartbeat")
            heartbeat.start()
            try:
                pause_path = settings.state_dir / "safety_pause.json"
                if pause_path.exists():
                    raise SafetyPause(json.loads(pause_path.read_text(encoding="utf-8"))["reason"])
                if not settings.mongo_uri:
                    raise SafetyPause("MONGO_URI must be configured for the dedicated failure collection")
                checkpoint = status.storage(lambda: Checkpoint(settings.state_dir / "checkpoint.db"))
                state = status.storage(lambda: DurableState(checkpoint, status))
                reporter = MongoReporter(state, settings, status)
                publisher = threading.Thread(target=reporter.loop, daemon=True, name="mongo-outbox")
                publisher.start()
                cycle = 0
                while not stop.is_set():
                    try:
                        client = MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name,
                                              timeout=settings.milvus_timeout)
                        break
                    except Exception as exc:
                        if not transient(exc):
                            raise SafetyPause("Milvus initialization failed: " + type(exc).__name__) from exc
                        status.waiters[threading.get_ident()] = "milvus_connect"
                        stop.wait(min(settings.recovery_max, settings.recovery_initial * 2 ** min(cycle, 8)))
                        cycle += 1
                else:
                    raise Stopping()
                status.waiters.pop(threading.get_ident(), None)
                worker = ResilientWorker(settings, status=status, state=state, checkpoint=checkpoint, milvus=client)
                worker.refresh()
                if startup_check is not None:
                    status.begin("startup_preflight")
                    worker.external("startup_preflight", startup_check)
                worker.run_resilient()
                while not stop.wait(5):
                    worker.refresh()
            except Stopping:
                pass
            except Exception as exc:
                try:
                    status.pause(str(exc))
                except Stopping:
                    pass
                while not stop.wait(5):
                    if worker is not None:
                        try:
                            worker.refresh()
                        except Stopping:
                            break
            finally:
                stop.set()
                if publisher:
                    publisher.join()
                if heartbeat:
                    heartbeat.join(timeout=20)
                if reporter:
                    reporter.close()
                if worker:
                    worker.close()
                elif checkpoint:
                    checkpoint.close()
    except Exception:
        LOGGER.exception("supervisor could not own its state directory")
        return 2
    return 0
