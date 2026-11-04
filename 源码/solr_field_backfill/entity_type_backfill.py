"""Safely backfill the production-compatible entityType field for fixed IDs.

This task has its own checkpoint and failure outbox. It never edits the
original 16-field checkpoint and only submits ``id`` plus ``entityType``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from pymilvus import DataType, MilvusClient

if __package__:
    from .production_parser import PARSER_VERSION, SOLR_SOURCES, parse_field
else:
    from production_parser import PARSER_VERSION, SOLR_SOURCES, parse_field


LOG = logging.getLogger("entity_type_backfill")
ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
FIELD_NAME = "entityType"
SOLR_FIELD = SOLR_SOURCES[FIELD_NAME]
FIELD_MAX_LENGTH = 50
WRITE_FIELDS = frozenset(("id", FIELD_NAME))


class SafetyError(RuntimeError):
    """A failed invariant that must stop writes until it is investigated."""


class StopRequested(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    milvus_uri: str
    database: str
    collection: str
    collection_id: int
    logical_count: int
    solr_url: str
    source_checkpoint: Path
    state_dir: Path
    batch_size: int = 500
    concurrency: int = 2
    flush_every_batches: int = 8
    max_write_rows_per_second: float = 350.0
    max_retries: int = 3
    solr_timeout: float = 60
    milvus_timeout: float = 120
    mongo_uri: str = ""
    mongo_database: str = "nstl_vector_sync_scientific_prod"
    mongo_collection: str = "solr_field_backfill_failures"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _transient(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and (
            exc.response.status_code >= 500 or exc.response.status_code in {408, 429}
        )
    if isinstance(exc, (TimeoutError, ConnectionError, requests.ConnectionError, requests.Timeout)):
        return True
    text = str(exc).lower()
    return any(term in text for term in (
        "timeout", "timed out", "unavailable", "deadline_exceeded", "deadline exceeded",
        "resource_exhausted", "failed to connect", "connection refused", "connection reset",
        "channel closed", "not connected", "temporarily", "503", "502", "504",
    ))


class DurableCheckpoint:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        with self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS batches(
                    prefix TEXT NOT NULL, number INTEGER NOT NULL, ids TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'pending', payload TEXT NOT NULL DEFAULT '[]',
                    result TEXT NOT NULL DEFAULT '{}', failures TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL, PRIMARY KEY(prefix, number));
                CREATE INDEX IF NOT EXISTS batches_phase_idx ON batches(phase, prefix, number);
                CREATE TABLE IF NOT EXISTS failures(
                    failure_id TEXT PRIMARY KEY, document TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
            """)

    def get_meta(self, key: str) -> Any:
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def bind_manifest(self, source: Path, digest: str, batch_count: int,
                      batches: Iterable[tuple[str, int, str]]) -> None:
        target = {"source_checkpoint": str(source.resolve()), "manifest_sha256": digest,
                  "field": FIELD_NAME, "solr_field": SOLR_FIELD, "max_length": FIELD_MAX_LENGTH,
                  "parser_version": PARSER_VERSION}
        encoded_target = _json(target)
        with self.lock, self.db:
            old = self.db.execute("SELECT value FROM meta WHERE key='target'").fetchone()
            if old and old[0] != encoded_target:
                raise SafetyError("entityType checkpoint belongs to another manifest or target")
            if not old:
                self.db.execute("BEGIN IMMEDIATE")
                inserted = hashlib.sha256()
                inserted_count = 0
                for prefix, number, ids_json in batches:
                    ids = json.loads(ids_json)
                    inserted.update(_json([prefix, number, ids]).encode("utf-8"))
                    self.db.execute(
                        "INSERT INTO batches(prefix,number,ids,updated_at) VALUES(?,?,?,?)",
                        (prefix, number, ids_json, _now()),
                    )
                    inserted_count += 1
                if inserted_count != batch_count or inserted.hexdigest() != digest:
                    raise SafetyError("source checkpoint changed while its entityType manifest was copied")
                self.db.execute("INSERT INTO meta(key,value) VALUES('target',?)", (encoded_target,))
                self.db.commit()
            else:
                actual = self.db.execute("SELECT count(*) FROM batches").fetchone()[0]
                if actual != batch_count:
                    raise SafetyError("entityType checkpoint batch count differs from the source manifest")

    def batch(self, prefix: str, number: int) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute(
                "SELECT ids,phase,payload,result,failures FROM batches WHERE prefix=? AND number=?",
                (prefix, number),
            ).fetchone()
        if row is None:
            raise SafetyError("batch is absent from the fixed entityType manifest")
        return {"ids": json.loads(row[0]), "phase": row[1], "payload": json.loads(row[2]),
                "result": json.loads(row[3]), "failures": json.loads(row[4])}

    def pending(self) -> list[tuple[str, int]]:
        with self.lock:
            return self.db.execute(
                "SELECT prefix,number FROM batches WHERE phase<>'done' ORDER BY prefix,number"
            ).fetchall()

    def phase(self, prefix: str, number: int) -> str:
        return self.batch(prefix, number)["phase"]

    def _failure_doc(self, task_id: str, target: dict[str, Any], prefix: str,
                     number: int, failure: dict[str, str]) -> tuple[str, dict[str, Any]]:
        key = hashlib.sha256(_json([task_id, failure["id"], failure["stage"], failure["reason"]]).encode()).hexdigest()
        now = _now()
        document = {"_id": key, "job_id": task_id, "target": target,
                    "doc_id": failure["id"], "shard": prefix, "batch": number,
                    "status": "pending", "attempt_count": 1,
                    "first_seen_at": now, "last_seen_at": now,
                    "stage": failure["stage"], "reason": failure["reason"],
                    "error": failure["error"][:2000]}
        return key, document

    def prepare(self, task_id: str, target: dict[str, Any], prefix: str, number: int,
                payload: list[dict[str, str]], result: dict[str, int],
                failures: list[dict[str, str]]) -> None:
        if any(set(row) != WRITE_FIELDS for row in payload):
            raise SafetyError("refusing a payload outside {id, entityType}")
        with self.lock, self.db:
            current = self.batch(prefix, number)
            if current["phase"] != "pending":
                return
            docs = [self._failure_doc(task_id, target, prefix, number, failure) for failure in failures]
            self.db.execute(
                "UPDATE batches SET phase='inflight',payload=?,result=?,failures=?,updated_at=? "
                "WHERE prefix=? AND number=? AND phase='pending'",
                (_json(payload), _json(result), _json(failures), _now(), prefix, number),
            )
            self.db.executemany("INSERT OR IGNORE INTO failures(failure_id,document) VALUES(?,?)",
                                [(key, _json(document)) for key, document in docs])

    def conflict(self, task_id: str, target: dict[str, Any], prefix: str, number: int,
                 row_id: str, message: str) -> None:
        failure = {"id": row_id, "stage": "milvus_reconcile", "reason": "entity_type_conflict", "error": message}
        key, document = self._failure_doc(task_id, target, prefix, number, failure)
        with self.lock, self.db:
            current = self.batch(prefix, number)
            if any(item["id"] == row_id and item["reason"] == failure["reason"] for item in current["failures"]):
                return
            payload = [item for item in current["payload"] if item["id"] != row_id]
            result = current["result"]
            result["updated"] = max(0, result.get("updated", 0) - 1)
            result["failed"] = result.get("failed", 0) + 1
            result["unchanged"] = result.get("unchanged", 0)
            failures = current["failures"] + [failure]
            if result["processed"] != result["updated"] + result["unchanged"] + result["failed"]:
                raise SafetyError("batch accounting did not reconcile after a target conflict")
            self.db.execute("UPDATE batches SET payload=?,result=?,failures=?,updated_at=? WHERE prefix=? AND number=?",
                            (_json(payload), _json(result), _json(failures), _now(), prefix, number))
            self.db.execute("INSERT OR IGNORE INTO failures(failure_id,document) VALUES(?,?)", (key, _json(document)))

    def mark_acked(self, prefix: str, number: int) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE batches SET phase='acked',updated_at=? WHERE prefix=? AND number=? AND phase='inflight'",
                            (_now(), prefix, number))

    def mark_done(self, batches: list[tuple[str, int]]) -> None:
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for prefix, number in batches:
                self.db.execute("UPDATE batches SET phase='done',updated_at=? WHERE prefix=? AND number=? AND phase='acked'",
                                (_now(), prefix, number))
            self.db.commit()

    def summary(self) -> dict[str, Any]:
        with self.lock:
            phases = dict(self.db.execute("SELECT phase,count(*) FROM batches GROUP BY phase").fetchall())
            totals = self.db.execute(
                "SELECT coalesce(sum(json_extract(result,'$.processed')),0),"
                "coalesce(sum(json_extract(result,'$.updated')),0),"
                "coalesce(sum(json_extract(result,'$.unchanged')),0),"
                "coalesce(sum(json_extract(result,'$.failed')),0) FROM batches WHERE phase='done'"
            ).fetchone()
            pending_mongo = self.db.execute("SELECT count(*) FROM failures WHERE sent=0").fetchone()[0]
        return {"batches": phases, "confirmed_processed": totals[0], "updated": totals[1],
                "unchanged": totals[2], "failed": totals[3], "mongo_pending": pending_mongo}

    def unsent_failures(self, limit: int = 500) -> list[tuple[str, dict[str, Any]]]:
        with self.lock:
            rows = self.db.execute("SELECT failure_id,document FROM failures WHERE sent=0 LIMIT ?", (limit,)).fetchall()
        return [(key, json.loads(document)) for key, document in rows]

    def mark_failures_sent(self, keys: list[str]) -> None:
        with self.lock, self.db:
            self.db.executemany("UPDATE failures SET sent=1 WHERE failure_id=?", [(key,) for key in keys])

    def close(self) -> None:
        with self.lock:
            self.db.close()


class EntityTypeBackfill:
    def __init__(self, settings: Settings, *, milvus: Any | None = None,
                 mongo: Any | None = None, session_factory: Callable[[], Any] | None = None):
        if settings.mongo_collection != "solr_field_backfill_failures":
            raise ValueError("failure events must use the isolated backfill collection")
        if settings.batch_size != 500 or settings.concurrency != 2:
            raise ValueError("this tested run is fixed to batch_size=500 and concurrency=2")
        self.settings = settings
        self.milvus = milvus or MilvusClient(uri=settings.milvus_uri, db_name=settings.database,
                                              timeout=settings.milvus_timeout)
        self._owns_milvus = milvus is None
        self.mongo = mongo
        self._owns_mongo = False
        self.session_factory = session_factory or self._new_session
        self.local = threading.local()
        self.sessions: list[Any] = []
        self.session_lock = threading.Lock()
        self.stop = threading.Event()
        self.checkpoint = DurableCheckpoint(settings.state_dir / "checkpoint.db")
        self.progress_path = settings.state_dir / "progress.json"
        self.report_path = settings.state_dir / "summary.json"
        self.baseline_path = settings.state_dir / "baseline.json"
        self.stop_path = settings.state_dir / "stop.request"
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = ""
        self.manifest_hash = ""
        self.target: dict[str, Any] = {}
        self.baseline: dict[str, Any] = {}
        self.history_fields: list[str] = []
        self.started = time.time()
        self.session_started = time.time()
        self.session_rows = 0
        self.write_gate = threading.Lock()
        self.progress_gate = threading.Lock()
        self.next_write_at = time.monotonic()
        self.heartbeat_stop = threading.Event()
        self._install_signals()

    @staticmethod
    def _new_session():
        session = requests.Session()
        session.trust_env = False
        return session

    def _http(self):
        session = getattr(self.local, "session", None)
        if session is None:
            session = self.session_factory()
            self.local.session = session
            with self.session_lock:
                self.sessions.append(session)
        return session

    def _install_signals(self):
        def request_stop(_signum, _frame):
            LOG.info("graceful stop requested; current batch window will drain and checkpoint will be kept")
            self.stop.set()
        try:
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
        except ValueError:
            pass

    def _check_stop_request(self):
        if self.stop_path.exists():
            self.stop.set()

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, default=str))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def _collection_info(self) -> dict[str, Any]:
        return self.milvus.describe_collection(collection_name=self.settings.collection)

    def _evidence(self) -> dict[str, Any]:
        logical = self.milvus.query(collection_name=self.settings.collection, filter="",
                                    output_fields=["count(*)"], consistency_level="Strong",
                                    timeout=self.settings.milvus_timeout)
        indexes = {}
        for name in self.milvus.list_indexes(collection_name=self.settings.collection):
            item = self.milvus.describe_index(collection_name=self.settings.collection, index_name=name)
            indexes[name] = {key: value for key, value in item.items()
                             if key not in {"indexed_rows", "total_rows", "pending_index_rows", "state"}}
        return {"schema": self._collection_info(), "logical_row_count": int(logical[0]["count(*)"]),
                "index_definitions": indexes}

    @staticmethod
    def _plain(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return EntityTypeBackfill._plain(value.tolist())
        if isinstance(value, dict):
            return {key: EntityTypeBackfill._plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [EntityTypeBackfill._plain(item) for item in value]
        if (not isinstance(value, (str, bytes, bytearray))
                and (hasattr(value, "__iter__")
                     or (hasattr(value, "__len__") and hasattr(value, "__getitem__")))):
            try:
                return [EntityTypeBackfill._plain(item) for item in value]
            except TypeError:
                pass
        return value

    def _hash_records(self, ids: list[str], fields: list[str]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for offset in range(0, len(ids), 50):
            current = ids[offset:offset + 50]
            rows = self.milvus.get(collection_name=self.settings.collection, ids=current,
                                   output_fields=fields, consistency_level="Strong",
                                   timeout=self.settings.milvus_timeout)
            if len(rows) != len(current) or {row["id"] for row in rows} != set(current):
                raise SafetyError("historical integrity sample is missing or duplicated")
            for row in rows:
                values = {}
                for field in fields:
                    raw = json.dumps(self._plain(row[field]), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")
                    values[field] = hashlib.sha256(raw).hexdigest()
                result[row["id"]] = values
        return result

    def _setup_schema_and_baseline(self) -> None:
        evidence = self._evidence()
        schema = evidence["schema"]
        if schema.get("collection_id") != self.settings.collection_id:
            raise SafetyError("target collection identity changed")
        if evidence["logical_row_count"] != self.settings.logical_count:
            raise SafetyError("logical collection count changed before the entityType task")
        old_fields = {item["name"]: item for item in schema.get("fields", [])}
        field = old_fields.get(FIELD_NAME)
        if field is not None:
            self._validate_entity_field(field)
        archived = json.loads((self.settings.source_checkpoint.parent / "full_baseline.json").read_text(encoding="utf-8"))
        old_baseline = archived["collection"]
        old_schema = {item["name"]: item for item in old_baseline["schema"]["fields"]}
        if any(old_fields.get(name) != definition for name, definition in old_schema.items()):
            raise SafetyError("a pre-existing field definition differs from the saved baseline")
        if evidence["index_definitions"] != old_baseline["index_definitions"]:
            raise SafetyError("index definitions changed since the saved baseline")
        if evidence["logical_row_count"] != old_baseline["logical_row_count"]:
            raise SafetyError("logical row count differs from the saved baseline")
        archived_hashes = archived["original_hashes"]
        if not archived_hashes:
            raise SafetyError("saved historical baseline has no protected sample")
        hash_field_sets = {tuple(sorted(value)) for value in archived_hashes.values()}
        if len(hash_field_sets) != 1:
            raise SafetyError("saved historical baseline has inconsistent field sets")
        self.history_fields = list(next(iter(hash_field_sets)))
        if not set(self.history_fields).issubset(old_schema):
            raise SafetyError("saved historical audit fields do not match the original schema")
        if self.baseline_path.exists():
            self.baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
            before = self.baseline["before"]
            if (before["collection_id"] != self.settings.collection_id
                    or before["logical_row_count"] != self.settings.logical_count
                    or before["index_definitions"] != evidence["index_definitions"]):
                raise SafetyError("saved entityType baseline belongs to a different collection state")
            saved_fields = {item["name"]: item for item in before["schema"]["fields"]}
            if any(old_fields.get(name) != definition for name, definition in saved_fields.items()
                   if name != FIELD_NAME):
                raise SafetyError("existing schema differs from the saved entityType baseline")
            allowed_names = set(saved_fields)
            if FIELD_NAME in old_fields:
                allowed_names.add(FIELD_NAME)
            if set(old_fields) != allowed_names:
                raise SafetyError("unexpected schema change since the entityType task began")
            if self.baseline.get("after_schema") is not None and FIELD_NAME in old_fields:
                expected_fields = {item["name"]: item for item in self.baseline["after_schema"]["fields"]}
                if old_fields != expected_fields:
                    raise SafetyError("current Schema differs from the saved post-addition schema")
            current_hashes = self._hash_records(before["history_sample_ids"], self.history_fields)
            if current_hashes != before["history_hashes"]:
                raise SafetyError("historical scalar/vector sample changed during this task")
        else:
            sample_before = archived["original_hashes"]
            sample_ids = list(sample_before)
            current_hashes = self._hash_records(sample_ids, self.history_fields)
            if current_hashes != sample_before:
                raise SafetyError("historical scalar/vector sample changed before this task")
            before = {"collection_id": schema["collection_id"],
                      "logical_row_count": evidence["logical_row_count"], "schema": schema,
                      "index_definitions": evidence["index_definitions"],
                      "history_sample_ids": sample_ids, "history_hashes": current_hashes}
            self.baseline = {"before": before, "after_schema": None,
                             "index_definitions": evidence["index_definitions"]}
            self._write_json(self.baseline_path, self.baseline)
        if field is None:
            self.milvus.add_collection_field(collection_name=self.settings.collection,
                                             field_name=FIELD_NAME, data_type=DataType.VARCHAR,
                                             max_length=FIELD_MAX_LENGTH, nullable=True,
                                             timeout=self.settings.milvus_timeout)
            self.milvus.load_collection(collection_name=self.settings.collection, refresh=True,
                                        timeout=self.settings.milvus_timeout)
            LOG.info("added nullable VARCHAR(%d) field %s", FIELD_MAX_LENGTH, FIELD_NAME)
        after = self._evidence()
        after_fields = {item["name"]: item for item in after["schema"]["fields"]}
        if after["schema"].get("collection_id") != self.settings.collection_id:
            raise SafetyError("collection identity changed during schema addition")
        if after["logical_row_count"] != self.settings.logical_count:
            raise SafetyError("logical count changed during schema addition")
        if after["index_definitions"] != evidence["index_definitions"]:
            raise SafetyError("index definitions changed during schema addition")
        if set(after_fields) - set(old_fields) not in (set(), {FIELD_NAME}):
            raise SafetyError("schema gained an unexpected field")
        if any(after_fields.get(name) != definition for name, definition in old_fields.items()):
            raise SafetyError("a pre-existing schema field changed")
        field = after_fields.get(FIELD_NAME)
        if field is None:
            raise SafetyError("entityType schema verification failed")
        self._validate_entity_field(field)
        self.baseline = {"before": before, "after_schema": after["schema"],
                         "index_definitions": after["index_definitions"]}
        self._write_json(self.baseline_path, self.baseline)
        self._write_json(self.settings.state_dir / "schema_addition.json", {
            "passed": True, "field": field, "collection_id": self.settings.collection_id,
            "logical_row_count_before": evidence["logical_row_count"],
            "logical_row_count_after": after["logical_row_count"],
            "indexes_unchanged": True, "existing_fields_unchanged": True,
        })

    @staticmethod
    def _validate_entity_field(field: dict[str, Any]) -> None:
        if (field.get("type") != DataType.VARCHAR
                or int(field.get("params", {}).get("max_length", 0)) != FIELD_MAX_LENGTH
                or not field.get("nullable", False)):
            raise SafetyError("entityType field is not nullable VARCHAR(50)")

    def _source_manifest(self) -> tuple[str, int]:
        source_uri = self.settings.source_checkpoint.resolve().as_uri() + "?mode=ro"
        source = sqlite3.connect(source_uri, uri=True, timeout=30)
        digest = hashlib.sha256()
        try:
            cursor = source.execute(
                "SELECT prefix,number,ids,processed FROM bf_batches WHERE phase='done' ORDER BY prefix,number"
            )
            count = total = 0
            for prefix, number, raw_ids, processed in cursor:
                ids = json.loads(raw_ids)
                if (not ids or len(ids) != processed or len(ids) != len(set(ids))
                        or any(not isinstance(value, str) or not ID_RE.fullmatch(value)
                               or value[:2].lower() != prefix for value in ids)):
                    raise SafetyError(f"invalid confirmed source manifest at {prefix}/{number}")
                canonical = _json([prefix, number, ids])
                digest.update(canonical.encode("utf-8"))
                count += 1
                total += len(ids)
        finally:
            source.close()
        if total != 5_414_828 or count != 10_838:
            raise SafetyError(f"confirmed source manifest changed: {count} batches, {total} IDs")
        return digest.hexdigest(), count

    def _iter_source_manifest(self) -> Iterable[tuple[str, int, str]]:
        source_uri = self.settings.source_checkpoint.resolve().as_uri() + "?mode=ro"
        source = sqlite3.connect(source_uri, uri=True, timeout=30)
        try:
            cursor = source.execute(
                "SELECT prefix,number,ids,processed FROM bf_batches WHERE phase='done' ORDER BY prefix,number"
            )
            for prefix, number, raw_ids, processed in cursor:
                ids = json.loads(raw_ids)
                if (not ids or len(ids) != processed or len(ids) != len(set(ids))
                        or any(not isinstance(value, str) or not ID_RE.fullmatch(value)
                               or value[:2].lower() != prefix for value in ids)):
                    raise SafetyError(f"invalid confirmed source manifest at {prefix}/{number}")
                yield prefix, number, _json(ids)
        finally:
            source.close()

    def _bind_task(self) -> None:
        self.manifest_hash, batch_count = self._source_manifest()
        self.target = {"milvus_uri": self.settings.milvus_uri, "database": self.settings.database,
                       "collection": self.settings.collection, "collection_id": self.settings.collection_id,
                       "field": FIELD_NAME, "solr_field": SOLR_FIELD, "max_length": FIELD_MAX_LENGTH,
                       "parser_version": PARSER_VERSION}
        self.task_id = hashlib.sha256(_json([self.target, self.manifest_hash]).encode()).hexdigest()
        self.checkpoint.bind_manifest(self.settings.source_checkpoint, self.manifest_hash,
                                      batch_count, self._iter_source_manifest())

    def _retry(self, name: str, action: Callable[[], Any]) -> Any:
        delay_cycle = 0
        while True:
            self._check_stop_request()
            if self.stop.is_set():
                raise StopRequested()
            for attempt in range(self.settings.max_retries + 1):
                try:
                    return action()
                except Exception as exc:
                    if isinstance(exc, (SafetyError, StopRequested)) or not _transient(exc):
                        raise
                    if attempt < self.settings.max_retries:
                        time.sleep(attempt + 1)
            delay = min(300, 30 * (2 ** min(delay_cycle, 4)))
            LOG.warning("dependency %s unavailable; retaining this batch and retrying in %ss", name, delay)
            for _ in range(delay):
                self._check_stop_request()
                if self.stop.wait(1):
                    raise StopRequested()
            delay_cycle += 1

    def _query_solr(self, ids: list[str]) -> list[dict[str, Any]]:
        if any(not ID_RE.fullmatch(value) for value in ids):
            raise SafetyError("unsafe ID in fixed manifest")

        def request():
            response = self._http().post(self.settings.solr_url, data={
                "q": "{!terms f=id}" + ",".join(ids), "fl": f"id,{SOLR_FIELD}",
                "rows": len(ids), "wt": "json",
            }, timeout=self.settings.solr_timeout)
            response.raise_for_status()
            body = response.json()
            header = body.get("responseHeader", {})
            if header.get("status", 0) or header.get("partialResults"):
                raise TimeoutError("Solr returned an incomplete entityType response")
            docs = body["response"]["docs"]
            found = [str(doc.get("id")) for doc in docs]
            if len(set(found)) != len(found) or not set(found).issubset(ids):
                raise SafetyError("Solr returned duplicate or unrequested IDs")
            return docs

        return self._retry("Solr query", request)

    def _strong(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        rows = self._retry("Milvus Strong read", lambda: self.milvus.get(
            collection_name=self.settings.collection, ids=ids, output_fields=["id", FIELD_NAME],
            consistency_level="Strong", timeout=self.settings.milvus_timeout))
        if len(rows) != len(ids) or {str(row.get("id")) for row in rows} != set(ids):
            raise SafetyError("Strong read found a missing or duplicated target record")
        if any(FIELD_NAME not in row for row in rows):
            raise SafetyError("Strong read did not return entityType")
        return {str(row["id"]): row for row in rows}

    def _failure(self, row_id: str, stage: str, reason: str, error: str) -> dict[str, str]:
        return {"id": row_id, "stage": stage, "reason": reason, "error": error}

    def _prepare_batch(self, prefix: str, number: int, ids: list[str]) -> None:
        docs = self._query_solr(ids)
        by_id = {str(doc["id"]): doc for doc in docs}
        failures: list[dict[str, str]] = []
        for row_id in ids:
            if row_id not in by_id:
                failures.append(self._failure(row_id, "solr_lookup", "source_missing", "Solr returned no document"))
        values: dict[str, str] = {}
        for row_id, doc in by_id.items():
            value = parse_field(doc, FIELD_NAME) or ""
            if not value:
                failures.append(self._failure(row_id, "transform", "entity_type_missing",
                                              f"Solr field {SOLR_FIELD} is empty"))
            elif len(value.encode("utf-8")) > FIELD_MAX_LENGTH:
                failures.append(self._failure(row_id, "transform", "value_too_long",
                                              f"{SOLR_FIELD} exceeds VARCHAR({FIELD_MAX_LENGTH})"))
            else:
                values[row_id] = value
        current = self._strong(ids)
        payload: list[dict[str, str]] = []
        unchanged = 0
        for row_id, value in values.items():
            existing = current[row_id][FIELD_NAME]
            if existing is None:
                payload.append({"id": row_id, FIELD_NAME: value})
            elif str(existing) == value:
                unchanged += 1
            else:
                failures.append(self._failure(row_id, "milvus_read", "existing_value_conflict",
                                              "existing non-null entityType differs from Solr; preserved"))
        result = {"processed": len(ids), "updated": len(payload), "unchanged": unchanged,
                  "failed": len(failures)}
        if result["processed"] != result["updated"] + result["unchanged"] + result["failed"]:
            raise SafetyError("batch accounting does not reconcile")
        self.checkpoint.prepare(self.task_id, self.target, prefix, number, payload, result, failures)

    @staticmethod
    def _assert_payload(payload: list[dict[str, str]]) -> None:
        for row in payload:
            if (set(row) != WRITE_FIELDS or not ID_RE.fullmatch(row.get("id", ""))
                    or not isinstance(row.get(FIELD_NAME), str)
                    or len(row[FIELD_NAME].encode("utf-8")) > FIELD_MAX_LENGTH):
                raise SafetyError("entityType payload whitelist or VARCHAR length check failed")

    def _pace_write(self, count: int) -> None:
        if count <= 0:
            return
        rate = self.settings.max_write_rows_per_second
        if rate <= 0:
            raise SafetyError("write pacing must be enabled for the live task")
        with self.write_gate:
            now = time.monotonic()
            delay = max(0.0, self.next_write_at - now)
            self.next_write_at = max(now, self.next_write_at) + count / rate
        end = time.monotonic() + delay
        while time.monotonic() < end:
            self._check_stop_request()
            if self.stop.wait(min(0.25, end - time.monotonic())):
                raise StopRequested()

    def _write_inflight(self, prefix: str, number: int) -> None:
        batch = self.checkpoint.batch(prefix, number)
        if batch["phase"] == "acked" or batch["phase"] == "done":
            return
        if batch["phase"] == "pending":
            self._prepare_batch(prefix, number, batch["ids"])
            batch = self.checkpoint.batch(prefix, number)
        if batch["phase"] != "inflight":
            raise SafetyError(f"unexpected entityType batch phase: {batch['phase']}")
        retries = 0
        recovery_cycle = 0
        while True:
            batch = self.checkpoint.batch(prefix, number)
            payload = batch["payload"]
            self._assert_payload(payload)
            if not payload:
                break
            current = self._strong([row["id"] for row in payload])
            needed: list[dict[str, str]] = []
            for row in payload:
                existing = current[row["id"]][FIELD_NAME]
                if existing is None:
                    needed.append(row)
                elif str(existing) != row[FIELD_NAME]:
                    self.checkpoint.conflict(
                        self.task_id, self.target, prefix, number, row["id"],
                        "a different non-null value appeared after the durable payload was prepared",
                    )
            if not needed:
                break
            self._assert_payload(needed)
            self._pace_write(len(needed))
            try:
                response = self.milvus.upsert(
                    collection_name=self.settings.collection, data=needed, partial_update=True,
                    timeout=self.settings.milvus_timeout,
                )
            except Exception as exc:
                if not _transient(exc):
                    raise
                retries += 1
                LOG.warning("partial update result uncertain for %s/%d; Strong reconciliation before retry (%s)",
                            prefix, number, type(exc).__name__)
                if retries > self.settings.max_retries:
                    delay = min(300, 30 * (2 ** min(recovery_cycle, 4)))
                    recovery_cycle += 1
                    retries = 0
                    for _ in range(delay):
                        self._check_stop_request()
                        if self.stop.wait(1):
                            raise StopRequested()
                continue
            if int(response.get("upsert_count", 0)) != len(needed):
                retries += 1
                LOG.warning("partial update count uncertain for %s/%d; Strong reconciliation before retry",
                            prefix, number)
            verified = self._strong([row["id"] for row in needed])
            unresolved = []
            for row in needed:
                existing = verified[row["id"]][FIELD_NAME]
                if existing is None:
                    unresolved.append(row)
                elif str(existing) != row[FIELD_NAME]:
                    self.checkpoint.conflict(
                        self.task_id, self.target, prefix, number, row["id"],
                        "Strong read-back found a different non-null value; preserved",
                    )
            if not unresolved:
                break
            retries += 1
            if retries > self.settings.max_retries:
                delay = min(300, 30 * (2 ** min(recovery_cycle, 4)))
                recovery_cycle += 1
                retries = 0
                LOG.warning("waiting %ss for Milvus to apply uncertain entityType writes", delay)
                for _ in range(delay):
                    self._check_stop_request()
                    if self.stop.wait(1):
                        raise StopRequested()
        self.checkpoint.mark_acked(prefix, number)

    def _flush(self) -> None:
        self._retry("Milvus flush", lambda: self.milvus.flush(
            collection_name=self.settings.collection, timeout=self.settings.milvus_timeout))

    def _sync_mongo(self) -> None:
        events = self.checkpoint.unsent_failures()
        if not events or not self.settings.mongo_uri:
            return
        if self.mongo is None:
            try:
                from pymongo import MongoClient
                self.mongo = MongoClient(self.settings.mongo_uri, serverSelectionTimeoutMS=5000,
                                         socketTimeoutMS=5000, retryWrites=False)
                self._owns_mongo = True
            except Exception:
                LOG.exception("could not create MongoDB client; failure events remain in SQLite")
                return
        try:
            from pymongo import UpdateOne
            collection = self.mongo[self.settings.mongo_database][self.settings.mongo_collection]
            collection.bulk_write([UpdateOne({"_id": key}, {"$setOnInsert": document}, upsert=True)
                                   for key, document in events], ordered=False)
            self.checkpoint.mark_failures_sent([key for key, _ in events])
        except Exception:
            LOG.exception("MongoDB failure sync unavailable; failure events remain in SQLite")

    def _write_progress(self, state: str) -> None:
        with self.progress_gate:
            summary = self.checkpoint.summary()
            elapsed = max(1.0, time.time() - self.session_started)
            summary.update({"task_id": self.task_id, "field": FIELD_NAME, "solr_field": SOLR_FIELD,
                            "state": state, "source_manifest_sha256": self.manifest_hash,
                            "source_confirmed_ids": 5_414_828,
                            "session_processed_per_second": self.session_rows / elapsed,
                            "session_processed": self.session_rows,
                            "updated_at": _now(), "collection_id": self.settings.collection_id})
            self._write_json(self.progress_path, summary)

    def _heartbeat(self) -> None:
        while not self.heartbeat_stop.wait(30):
            self._check_stop_request()
            self._write_progress("stopping" if self.stop.is_set() else "running")

    def _verify_final(self) -> dict[str, Any]:
        current = self._evidence()
        before = self.baseline["before"]
        old_fields = {item["name"]: item for item in before["schema"]["fields"]}
        new_fields = {item["name"]: item for item in current["schema"]["fields"]}
        if any(new_fields.get(name) != item for name, item in old_fields.items()):
            raise SafetyError("a pre-existing schema field changed")
        additions = set(new_fields) - set(old_fields)
        if additions != {FIELD_NAME}:
            raise SafetyError(f"unexpected schema additions at finish: {sorted(additions)}")
        if current["schema"].get("collection_id") != self.settings.collection_id:
            raise SafetyError("collection identity changed during backfill")
        if current["logical_row_count"] != self.settings.logical_count:
            raise SafetyError("logical row count changed during backfill")
        if current["index_definitions"] != before["index_definitions"]:
            raise SafetyError("index definitions changed during backfill")
        field = new_fields[FIELD_NAME]
        self._validate_entity_field(field)
        after_hashes = self._hash_records(before["history_sample_ids"], self.history_fields)
        if after_hashes != before["history_hashes"]:
            raise SafetyError("original fields or vectors changed during entityType backfill")
        return {"passed": True, "collection_id": self.settings.collection_id,
                "logical_row_count": current["logical_row_count"], "field": field,
                "only_schema_addition": FIELD_NAME, "indexes_unchanged": True,
                "original_fields_and_vectors_unchanged": True,
                "historical_sample_size": len(after_hashes)}

    def run(self, *, max_batches: int | None = None) -> dict[str, Any]:
        self.started = self.session_started = time.time()
        if (self.settings.state_dir / "safety_pause.json").exists():
            raise SafetyError("a persisted safety pause exists; review it before restarting")
        self._bind_task()
        self._setup_schema_and_baseline()
        pending = self.checkpoint.pending()
        total = len(pending) if max_batches is None else min(len(pending), max_batches)
        selected = pending[:total]
        self._write_progress("running")
        heartbeat = threading.Thread(target=self._heartbeat, daemon=True, name="entityType-heartbeat")
        heartbeat.start()
        LOG.info("starting entityType backfill for %d confirmed batches / 5,414,828 IDs", len(selected))
        try:
            with ThreadPoolExecutor(max_workers=self.settings.concurrency, thread_name_prefix="entity-type") as pool:
                for start in range(0, len(selected), self.settings.flush_every_batches):
                    self._check_stop_request()
                    if self.stop.is_set():
                        break
                    window = selected[start:start + self.settings.flush_every_batches]
                    active = [(prefix, number) for prefix, number in window
                              if self.checkpoint.phase(prefix, number) not in {"acked", "done"}]
                    futures = [pool.submit(self._write_inflight, prefix, number) for prefix, number in active]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception:
                            self.stop.set()
                            for queued in futures:
                                queued.cancel()
                            raise
                    self._flush()
                    self.checkpoint.mark_done(window)
                    self.session_rows += sum(len(self.checkpoint.batch(prefix, number)["ids"])
                                             for prefix, number in window)
                    self._sync_mongo()
                    self._write_progress("running")
                    LOG.info("confirmed entityType window: %d/%d IDs", self.checkpoint.summary()["confirmed_processed"],
                             5_414_828)
                    self._check_stop_request()
                    if self.stop.is_set():
                        break
        except StopRequested:
            self.stop.set()
        except Exception as exc:
            self._write_json(self.settings.state_dir / "safety_pause.json",
                             {"timestamp": _now(), "reason": str(exc), "task_id": self.task_id})
            self._write_progress("safety_paused")
            raise
        finally:
            self.heartbeat_stop.set()
            heartbeat.join(timeout=2)
            try:
                self._sync_mongo()
            except Exception:
                LOG.exception("failure queue sync failed")
        summary = self.checkpoint.summary()
        all_done = summary["batches"].get("done", 0) == 10_838
        if all_done:
            self._flush()
            verification = self._verify_final()
            result = {**summary, "state": "completed_with_failures" if summary["failed"] else "completed",
                      "scan_complete": True, "verification": verification,
                      "elapsed_seconds": time.time() - self.started,
                      "finished_at": _now(), "task_id": self.task_id}
            self._write_json(self.report_path, result)
            self._write_progress(result["state"])
            LOG.info("entityType task finished: processed=%d updated=%d failed=%d",
                     summary["confirmed_processed"], summary["updated"], summary["failed"])
            return result
        state = "paused" if self.stop.is_set() else "running"
        if max_batches is not None:
            verification = self._verify_final()
            self._write_json(self.settings.state_dir / "canary_verification.json", verification)
        result = {**summary, "state": state, "scan_complete": False,
                  "updated_at": _now(), "task_id": self.task_id}
        self._write_json(self.report_path, result)
        self._write_progress(state)
        return result

    def close(self) -> None:
        for session in self.sessions:
            try:
                session.close()
            except Exception:
                pass
        if self._owns_mongo and self.mongo is not None:
            self.mongo.close()
        if self._owns_milvus:
            self.milvus.close()
        self.checkpoint.close()

