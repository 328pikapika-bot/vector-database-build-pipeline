"""安全的 Solr -> Milvus 新字段补全 Worker。

这个程序有意只使用 Milvus partial upsert，并在写入前强制校验 payload。
原集合中的旧字段和向量永远不会进入写入 payload，也没有任何删除 API。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import signal
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Iterable

import requests
from pymilvus import DataType, MilvusClient

if __package__:
    from .production_parser import PARSER_VERSION, SOLR_QUERY_FIELDS, SOLR_SOURCES, parse_all
else:  # The container executes backfill.py as a top-level module.
    from production_parser import PARSER_VERSION, SOLR_QUERY_FIELDS, SOLR_SOURCES, parse_all


LOGGER = logging.getLogger("solr_field_backfill")
HEX_ID = re.compile(r"^[0-9a-fA-F]{32}$")
SHARDS = tuple(f"{i:02x}" for i in range(256))
ORIGINAL_VECTOR_FIELDS = ("title_embed", "abstract_embed", "keyword_embed")
ORIGINAL_REQUIRED_FIELDS = ("id",) + ORIGINAL_VECTOR_FIELDS
TARGET_FIELDS = (
    "hasSource", "hasFile", "volume", "issue", "startPage", "endPage",
    "pageCount", "createTime", "publisher", "labelEvaluateGrade",
    "sysProcessingUnit", "hasHoldinglibraryCode", "hasAuthor_str",
    "hasSource_str", "hasFile_str", "hasCreateOrganization_str", "entityType",
)

# 与生产 schema 对齐；值来自 nstl_paper_abstract_ec_expand_202510.json。
FIELD_SPECS: dict[str, int] = {
    "hasSource": 20000,
    "hasFile": 500,
    "volume": 50,
    "issue": 50,
    "startPage": 50,
    "endPage": 50,
    "pageCount": 50,
    "createTime": 50,
    "publisher": 100,
    "labelEvaluateGrade": 100,
    "sysProcessingUnit": 100,
    "hasHoldinglibraryCode": 100,
    "hasAuthor_str": 20000,
    "hasSource_str": 20000,
    "hasFile_str": 500,
    "hasCreateOrganization_str": 20000,
    "entityType": 50,
}

SOLR_MAPPING = SOLR_SOURCES
WRITE_FIELDS = frozenset(("id",) + TARGET_FIELDS)


@contextmanager
def timed(metrics: dict[str, Any], name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics[name] = metrics.get(name, 0.0) + time.perf_counter() - started


@dataclass(frozen=True)
class Settings:
    milvus_uri: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19533")
    milvus_db_name: str = os.getenv("MILVUS_DB_NAME", "default")
    collection: str = os.getenv("MILVUS_COLLECTION", "scientific_documents_v2")
    solr_url: str = os.getenv(
        "SOLR_URL", "http://solr-bak1.example.invalid:8984/solr/paper6.6/select"
    )
    mode: str = os.getenv("BACKFILL_MODE", "check").lower()
    concurrency: int = int(os.getenv("BACKFILL_CONCURRENCY", "2"))
    batch_size: int = int(os.getenv("BACKFILL_BATCH_SIZE", "200"))
    solr_timeout: float = float(os.getenv("SOLR_TIMEOUT_SEC", "60"))
    milvus_timeout: float = float(os.getenv("MILVUS_TIMEOUT_SEC", "120"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_backoff: float = float(os.getenv("RETRY_BACKOFF_SEC", "1.0"))
    flush_every_batches: int = int(os.getenv("FLUSH_EVERY_BATCHES", "20"))
    milvus_thread_clients: bool = os.getenv("MILVUS_THREAD_CLIENTS", "0").lower() in {"1", "true", "yes", "on"}
    max_records: int = int(os.getenv("BACKFILL_MAX_RECORDS", "0"))
    sample_start_prefix: str = os.getenv("BACKFILL_SAMPLE_START_PREFIX", "").lower()
    exclude_sample_manifest: str = os.getenv("BACKFILL_EXCLUDE_SAMPLE_MANIFEST", "")
    state_dir: Path = Path(os.getenv("BACKFILL_STATE_DIR", "/app/state"))
    log_dir: Path = Path(os.getenv("BACKFILL_LOG_DIR", "/app/logs"))
    sample_size: int = int(os.getenv("INTEGRITY_SAMPLE_SIZE", "3"))
    mongo_uri: str = os.getenv("MONGO_URI", "")
    mongo_db: str = os.getenv("MONGO_DB", "nstl_vector_sync_scientific_prod")
    mongo_failure_collection: str = os.getenv("MONGO_FAILURE_COLLECTION", "solr_field_backfill_failures")
    recovery_initial: float = float(os.getenv("RECOVERY_INITIAL_SEC", "30"))
    recovery_max: float = float(os.getenv("RECOVERY_MAX_SEC", "300"))
    milvus_slow_request_seconds: float = float(os.getenv("MILVUS_SLOW_REQUEST_SEC", "15"))
    max_write_rows_per_second: float = float(os.getenv("BACKFILL_MAX_WRITE_ROWS_PER_SEC", "0"))

    def validate(self) -> None:
        if self.mode not in {"check", "run"}:
            raise ValueError("BACKFILL_MODE must be check or run")
        if not 1 <= self.concurrency <= 2:
            raise ValueError("BACKFILL_CONCURRENCY must be 1 or 2 for safety")
        if not 1 <= self.batch_size <= 1000:
            raise ValueError("BACKFILL_BATCH_SIZE must be between 1 and 1000")
        if self.max_retries < 0:
            raise ValueError("MAX_RETRIES must be non-negative")
        if self.flush_every_batches < 1:
            raise ValueError("FLUSH_EVERY_BATCHES must be positive")
        if self.max_records < 0:
            raise ValueError("BACKFILL_MAX_RECORDS must be zero or positive")
        if self.sample_start_prefix and self.sample_start_prefix not in SHARDS:
            raise ValueError("BACKFILL_SAMPLE_START_PREFIX must be two hexadecimal digits")
        if self.sample_size < 0:
            raise ValueError("INTEGRITY_SAMPLE_SIZE must be non-negative")
        if self.recovery_initial <= 0 or self.recovery_max < self.recovery_initial:
            raise ValueError("invalid dependency recovery intervals")
        if self.milvus_slow_request_seconds < 0:
            raise ValueError("MILVUS_SLOW_REQUEST_SEC must be non-negative")
        if not math.isfinite(self.max_write_rows_per_second) or self.max_write_rows_per_second < 0:
            raise ValueError("BACKFILL_MAX_WRITE_ROWS_PER_SEC must be finite and non-negative")
        if self.mongo_failure_collection != "solr_field_backfill_failures":
            raise ValueError("failure collection must be isolated from existing compensators")


@dataclass(frozen=True)
class PreparedBatch:
    ids: tuple[str, ...]
    rows: list[dict[str, Any]]
    missing: list[str]
    error: str | None
    metrics: dict[str, Any]
    started: float
    finished: float


class Checkpoint:
    """小型 SQLite checkpoint，避免保存全量 ID。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS shards ("
                "prefix TEXT PRIMARY KEY, status TEXT NOT NULL, scanned INTEGER DEFAULT 0, "
                "updated INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, attempts INTEGER DEFAULT 0, "
                "last_error TEXT, updated_at TEXT NOT NULL)"
            )
            for prefix in SHARDS:
                self.db.execute(
                    "INSERT OR IGNORE INTO shards(prefix,status,updated_at) VALUES(?,?,datetime('now'))",
                    (prefix, "pending"),
                )
            self.db.commit()

    def bind_target(self, target: dict[str, Any]) -> None:
        """Never reuse completed shards for another collection or write mapping."""
        encoded = json.dumps(target, sort_keys=True, default=str)
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = self.db.execute("SELECT value FROM metadata WHERE key='target'").fetchone()
            if row is not None and row[0] != encoded:
                raise RuntimeError("checkpoint belongs to a different target or field mapping")
            if row is None:
                used = self.db.execute("SELECT count(*) FROM shards WHERE status <> 'pending'").fetchone()[0]
                if used:
                    raise RuntimeError("unbound checkpoint already contains progress; use a dedicated state volume")
                self.db.execute("INSERT INTO metadata(key,value) VALUES('target',?)", (encoded,))
            self.db.commit()

    def pending(self) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute(
                "SELECT prefix FROM shards WHERE status <> 'completed' ORDER BY prefix"
            )]

    def start(self, prefix: str) -> None:
        self.set(prefix, "running", attempts_delta=1)

    def set(self, prefix: str, status: str, *, scanned: int = 0, updated: int = 0,
            failed: int = 0, error: str = "", attempts_delta: int = 0) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE shards SET status=?, scanned=?, updated=?, failed=?, "
                "attempts=attempts+?, last_error=?, updated_at=datetime('now') WHERE prefix=?",
                (status, scanned, updated, failed, attempts_delta, error[:2000], prefix),
            )
            self.db.commit()

    def summary(self) -> dict[str, Any]:
        with self.lock:
            rows = self.db.execute(
                "SELECT status, count(*), coalesce(sum(scanned),0), coalesce(sum(updated),0), "
                "coalesce(sum(failed),0) FROM shards GROUP BY status"
            ).fetchall()
        result = {"shards": {}, "scanned": 0, "updated": 0, "failed": 0}
        for status, count, scanned, updated, failed in rows:
            result["shards"][status] = count
            result["scanned"] += scanned
            result["updated"] += updated
            result["failed"] += failed
        return result

    def close(self) -> None:
        with self.lock:
            self.db.close()


class BackfillWorker:
    def __init__(self, settings: Settings, *, milvus: Any | None = None,
                 http: requests.Session | None = None, checkpoint: Checkpoint | None = None) -> None:
        settings.validate()
        self.settings = settings
        self.milvus = milvus or MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name)
        self._injected_milvus = milvus is not None
        self._milvus_clients: list[Any] = [self.milvus]
        self._milvus_lock = threading.Lock()
        self.http = http or requests.Session()
        if http is None:
            # Internal Solr is reached directly; host proxy settings can
            # otherwise turn a healthy endpoint into a proxy-side HTTP 503.
            self.http.trust_env = False
        self._root_http = self.http
        self._injected_http = http is not None
        self._thread_local = threading.local()
        self._http_lock = threading.Lock()
        self._http_sessions = [self.http]
        self.checkpoint = checkpoint or Checkpoint(settings.state_dir / "checkpoint.db")
        self.stop_event = threading.Event()
        self.failures_path = settings.state_dir / "failures.jsonl"
        self.progress_path = settings.state_dir / "progress.json"
        self.summary_path = settings.state_dir / "summary.json"
        self.original_fields: list[str] = []
        self._sample_allowlist: frozenset[str] | None = None
        self._metrics_lock = threading.Lock()
        self.batch_metrics: list[dict[str, Any]] = []
        self.flush_metrics: list[dict[str, Any]] = []
        self.preparation_metrics: list[dict[str, Any]] = []
        self.pipeline_metrics = {"peak_pending_preparations": 0, "peak_active_writes": 0}
        self._sample_expected: dict[str, dict[str, Any]] | None = None
        self._reader_pool: ThreadPoolExecutor | None = None
        self._writer_pool: ThreadPoolExecutor | None = None
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        settings.log_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        # Pools span shards so that connections and thread-local clients stay bounded.
        for pool in (self._reader_pool, self._writer_pool):
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        with self._http_lock:
            sessions = {id(session): session for session in [*self._http_sessions, self.http]}
            self._http_sessions.clear()
        for session in sessions.values():
            session.close()
        with self._milvus_lock:
            clients = {id(client): client for client in self._milvus_clients}
            self._milvus_clients.clear()
        for client in clients.values():
            if hasattr(client, "close"):
                client.close()
        self.checkpoint.close()

    @contextmanager
    def _pipeline_executors(self):
        if self._reader_pool is None:
            self._reader_pool = ThreadPoolExecutor(max_workers=self.settings.concurrency,
                                                   thread_name_prefix="solr-prepare")
            self._writer_pool = ThreadPoolExecutor(max_workers=self.settings.concurrency,
                                                   thread_name_prefix="milvus-write")
        try:
            yield self._reader_pool, self._writer_pool
        except BaseException:
            self.stop_event.set()
            # Never audit while a write is in flight after an exceptional exit.
            for pool in (self._reader_pool, self._writer_pool):
                pool.shutdown(wait=True, cancel_futures=True)
            self._reader_pool = self._writer_pool = None
            raise

    def _milvus_for_thread(self) -> Any:
        """Return a writer-thread-local client when explicitly enabled.

        The injected client path is kept shared for deterministic unit tests and
        fake Milvus implementations. Production runs can opt into one client
        per writer thread to avoid contention in the client connection/cache.
        """
        if not self.settings.milvus_thread_clients or self._injected_milvus:
            return self.milvus
        if not hasattr(self._thread_local, "milvus"):
            client = MilvusClient(uri=self.settings.milvus_uri, db_name=self.settings.milvus_db_name)
            self._thread_local.milvus = client
            with self._milvus_lock:
                self._milvus_clients.append(client)
        return self._thread_local.milvus

    def _http_session(self) -> requests.Session:
        # Preserve explicitly injected test clients; real worker threads each
        # own a Session, avoiding concurrent use of mutable Session state.
        if self._injected_http or self.http is not self._root_http:
            return self.http
        if not hasattr(self._thread_local, "http"):
            session = requests.Session()
            session.trust_env = False
            self._thread_local.http = session
            with self._http_lock:
                self._http_sessions.append(session)
        return self._thread_local.http

    def _flush(self, reason: str) -> None:
        started = time.perf_counter()
        error = None
        try:
            self.milvus.flush(collection_name=self.settings.collection,
                              timeout=self.settings.milvus_timeout)
        except Exception as exc:
            self.stop_event.set()
            error = str(exc)
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.flush_metrics.append({"reason": reason, "seconds": elapsed, "error": error})
            LOGGER.info("flush %s took %.3fs", reason, elapsed)

    def collection_info(self) -> dict[str, Any]:
        return self.milvus.describe_collection(self.settings.collection)

    def check_solr(self) -> dict[str, Any]:
        """Perform a zero-row, read-only Solr connectivity check."""
        response = self.http.get(
            self.settings.solr_url,
            params={"q": "*:*", "rows": 0, "wt": "json"},
            timeout=self.settings.solr_timeout,
        )
        response.raise_for_status()
        body = response.json()
        found = int(body.get("response", {}).get("numFound", 0))
        return {"url": self.settings.solr_url, "num_found": found}

    def _validate_id_format(self) -> None:
        iterator = self.milvus.query_iterator(
            collection_name=self.settings.collection,
            filter="id like '00%'",
            output_fields=["id"],
            batch_size=16,
            limit=16,
            timeout=self.settings.milvus_timeout,
        )
        try:
            batch = iterator.next() or []
        finally:
            iterator.close()
        for row in batch:
            value = str(row.get("id", ""))
            if not HEX_ID.fullmatch(value) or value[:2].lower() != "00":
                raise RuntimeError(f"unsafe ID format; expected 32 hex characters: {value}")

    @staticmethod
    def _field_type(field: dict[str, Any]) -> Any:
        return field.get("type")

    def preflight(self, *, mutate_schema: bool = False) -> dict[str, Any]:
        if mutate_schema and self.settings.mode != "run":
            raise RuntimeError("schema changes require BACKFILL_MODE=run")
        if not self.milvus.has_collection(self.settings.collection):
            raise RuntimeError(f"target collection does not exist: {self.settings.collection}")
        info = self.collection_info()
        fields = {f["name"]: f for f in info.get("fields", [])}
        self.original_fields = [name for name in fields if name not in TARGET_FIELDS]
        for required in ORIGINAL_REQUIRED_FIELDS:
            if required not in fields:
                raise RuntimeError(f"required original field is missing: {required}")
        if not fields["id"].get("is_primary"):
            raise RuntimeError("id must remain the primary key")
        if self._field_type(fields["id"]) != DataType.VARCHAR:
            raise RuntimeError("id must be a VARCHAR primary key")
        if info.get("auto_id", False):
            raise RuntimeError("auto_id must be false for safe partial updates")
        for vector in ORIGINAL_VECTOR_FIELDS:
            if self._field_type(fields[vector]) != DataType.FLOAT_VECTOR:
                raise RuntimeError(f"{vector} is not FLOAT_VECTOR")
            if int(fields[vector].get("params", {}).get("dim", 0)) != 2560:
                raise RuntimeError(f"{vector} must have dim=2560")
        self._validate_id_format()
        added: list[str] = []
        for name in TARGET_FIELDS:
            field = fields.get(name)
            if field is None:
                added.append(name)
                continue
            if self._field_type(field) != DataType.VARCHAR:
                raise RuntimeError(f"{name} must be VARCHAR")
            max_length = int(field.get("params", {}).get("max_length", 0))
            if max_length != FIELD_SPECS[name] or not field.get("nullable", False):
                raise RuntimeError(f"{name} has incompatible schema: {field}")
        # Validate the entire schema before performing any schema mutation.
        if mutate_schema:
            for name in added:
                self.milvus.add_collection_field(
                    collection_name=self.settings.collection,
                    field_name=name, data_type=DataType.VARCHAR,
                    max_length=FIELD_SPECS[name], nullable=True,
                    timeout=self.settings.milvus_timeout,
                )
                LOGGER.info("added nullable field %s", name)
            if added:
                # Refresh the loaded schema; this does not recreate vector indexes.
                self.milvus.load_collection(collection_name=self.settings.collection,
                                           refresh=True, timeout=self.settings.milvus_timeout)
        stats = self.milvus.get_collection_stats(self.settings.collection)
        row_count = int(stats.get("row_count", 0))
        result = {"collection": self.settings.collection, "row_count": row_count,
                  "existing_fields": len(fields), "missing_fields": added,
                  "target_fields": list(TARGET_FIELDS), "mode": self.settings.mode}
        self._write_json(self.progress_path, result)
        return result

    @staticmethod
    def _plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: BackfillWorker._plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [BackfillWorker._plain(v) for v in value]
        if hasattr(value, "tolist"):
            return BackfillWorker._plain(value.tolist())
        # pymilvus may return protobuf repeated scalar containers for vector
        # fields.  They are iterable but are not JSON serializable directly.
        # Convert only non-text iterables so scalar strings/bytes keep their
        # original representation.
        if (hasattr(value, "__iter__") or
                (hasattr(value, "__len__") and hasattr(value, "__getitem__"))) \
                and not isinstance(value, (str, bytes, bytearray)):
            try:
                return [BackfillWorker._plain(v) for v in value]
            except TypeError:
                pass
        return value

    def _hash_records(self, ids: list[str]) -> dict[str, dict[str, str]]:
        """Hash every original field, including complete vector arrays, by exact ID."""
        result = {}
        for offset in range(0, len(ids), 100):
            batch = ids[offset:offset + 100]
            rows = self.milvus.get(collection_name=self.settings.collection, ids=batch,
                                   output_fields=self.original_fields,
                                   consistency_level="Strong", timeout=self.settings.milvus_timeout)
            if len(rows) != len(batch) or {r["id"] for r in rows} != set(batch):
                raise RuntimeError("historical record missing or duplicated during integrity read")
            for row in rows:
                hashes = {}
                for name in self.original_fields:
                    if name not in row:
                        raise RuntimeError(f"original field not returned: {name}")
                    value = self._plain(row[name])
                    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False).encode("utf-8")
                    hashes[name] = hashlib.sha256(encoded).hexdigest()
                result[str(row["id"])] = hashes
            if len(ids) >= 1000 and (offset + len(batch)) % 1000 == 0:
                LOGGER.info("historical fields/vector audit read %d/%d", offset + len(batch), len(ids))
        return result

    def _integrity_snapshot(self) -> dict[str, Any]:
        """Hash a small sample of the original fields and vectors.

        The snapshot deliberately reads only the original primary key, scalar
        fields and vectors.  It is used before and after a run to prove that
        the partial updates did not replace historical values.
        """
        if self.settings.sample_size == 0:
            return {}
        iterator = self.milvus.query_iterator(
            collection_name=self.settings.collection,
            filter="id like '00%'",
            output_fields=list(ORIGINAL_REQUIRED_FIELDS),
            batch_size=self.settings.sample_size,
            limit=self.settings.sample_size,
            timeout=self.settings.milvus_timeout,
        )
        rows: list[dict[str, Any]] = []
        try:
            while len(rows) < self.settings.sample_size:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()
        ids = [str(row["id"]) for row in rows[:self.settings.sample_size]]
        return self._hash_records(ids)

    def verify_integrity(self, before: dict[str, Any]) -> dict[str, Any]:
        after = self._hash_records(list(before))
        changed = sorted(
            row_id for row_id in set(before) | set(after)
            if before.get(row_id) != after.get(row_id)
        )
        result = {"sample_size": len(before), "before": before, "after": after,
                  "changed_ids": changed, "passed": not changed}
        if changed:
            raise RuntimeError(f"original fields/vector changed for sample IDs: {changed}")
        return result

    def _write_json(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def _record_failure(self, prefix: str, ids: list[str], error: str) -> None:
        with self.failures_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"prefix": prefix, "ids": ids, "error": error,
                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False) + "\n")

    def _query_ids(self, prefix: str, limit: int | None = None) -> list[str]:
        iterator = self.milvus.query_iterator(
            collection_name=self.settings.collection,
            filter=f"id like '{prefix}%'",
            output_fields=["id"],
            batch_size=1000,
            limit=limit if limit is not None else -1,
            consistency_level="Strong",
            timeout=self.settings.milvus_timeout,
        )
        ids: list[str] = []
        try:
            while not self.stop_event.is_set():
                batch = iterator.next()
                if not batch:
                    break
                ids.extend(str(row["id"]) for row in batch)
                if limit is not None and len(ids) >= limit:
                    ids = ids[:limit]
                    break
        finally:
            iterator.close()
        invalid = [value for value in ids if not HEX_ID.fullmatch(value) or value[:2].lower() != prefix]
        if invalid:
            raise RuntimeError(f"unsafe ID format in shard {prefix}: {invalid[:3]}")
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"duplicate source IDs in shard {prefix}")
        return ids

    def extract(self, doc: dict[str, Any]) -> dict[str, Any]:
        if not doc.get("id") or not HEX_ID.fullmatch(str(doc["id"])):
            raise ValueError("Solr document has an invalid id")
        row: dict[str, Any] = {"id": str(doc["id"])}
        parsed = parse_all(doc)
        for field in TARGET_FIELDS:
            value = parsed[field]
            if value is not None and len(value.encode("utf-8")) > FIELD_SPECS[field]:
                raise ValueError(f"{field} exceeds max_length={FIELD_SPECS[field]}")
            row[field] = value
        self._assert_write_payload(row)
        return row

    @staticmethod
    def _assert_write_payload(row: dict[str, Any]) -> None:
        if set(row) != WRITE_FIELDS:
            raise AssertionError(f"unsafe payload fields: {sorted(row)}")
        if not isinstance(row["id"], str) or not HEX_ID.fullmatch(row["id"]):
            raise AssertionError("invalid payload ID")
        for name in TARGET_FIELDS:
            value = row[name]
            if value is not None and (not isinstance(value, str)
                                      or len(value.encode("utf-8")) > FIELD_SPECS[name]):
                raise AssertionError(f"invalid payload value: {name}")

    def _solr_batch(self, ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        if len(set(ids)) != len(ids) or any(not HEX_ID.fullmatch(value) for value in ids):
            raise ValueError("unsafe or duplicated query IDs")
        params = {
            "q": "{!terms f=id}" + ",".join(ids),
            "fl": ",".join(SOLR_QUERY_FIELDS),
            "rows": len(ids), "wt": "json",
        }
        last_error: Exception | None = None
        metrics = getattr(self._thread_local, "metrics", {})
        for attempt in range(self.settings.max_retries + 1):
            try:
                metrics["solr_attempts"] = metrics.get("solr_attempts", 0) + 1
                with timed(metrics, "solr_query_seconds"):
                    response = self._http_session().post(self.settings.solr_url, data=params,
                                                       timeout=self.settings.solr_timeout)
                    response.raise_for_status()
                    body = response.json()
                    if body.get("responseHeader", {}).get("status", 0) != 0:
                        raise ValueError("Solr returned nonzero status")
                    if body.get("responseHeader", {}).get("partialResults"):
                        raise ValueError("Solr returned partial results")
                    docs = body["response"]["docs"]
                    found = {str(doc.get("id")) for doc in docs}
                    if len(found) != len(docs) or not found.issubset(set(ids)):
                        raise ValueError("Solr returned duplicate or unrequested IDs")
                missing = [value for value in ids if value not in found]
                with timed(metrics, "transform_seconds"):
                    rows = [self.extract(doc) for doc in docs]
                return rows, missing
            except Exception as exc:  # transport, JSON and mapping errors are retryable.
                last_error = exc
                if attempt < self.settings.max_retries:
                    with timed(metrics, "retry_wait_seconds"):
                        if self.stop_event.wait(self.settings.retry_backoff * (attempt + 1)):
                            raise RuntimeError("stopped during Solr retry wait") from exc
        raise RuntimeError(f"Solr batch failed after retries: {last_error}")

    def _assert_batch_scope(self, ids: list[str]) -> None:
        if self._sample_allowlist is not None and not set(ids).issubset(self._sample_allowlist):
            raise RuntimeError("write outside the fixed sample ID allowlist")
        if len(set(ids)) != len(ids) or any(not HEX_ID.fullmatch(value) for value in ids):
            raise ValueError("unsafe or duplicated batch IDs")

    def _validate_prepared_rows(self, ids: list[str], rows: list[dict[str, Any]],
                                missing: list[str]) -> None:
        self._assert_batch_scope(ids)
        if len(set(missing)) != len(missing) or not set(missing).issubset(ids):
            raise RuntimeError("invalid missing Solr IDs")
        for row in rows:
            self._assert_write_payload(row)
        row_ids = [r["id"] for r in rows]
        if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(ids) - set(missing):
            raise RuntimeError("duplicate or unrequested payload IDs")

    def _prepare_batch(self, ids: list[str]) -> PreparedBatch:
        """Read and convert only. No Milvus mutation or checkpoint advancement."""
        started = time.perf_counter()
        metrics: dict[str, Any] = {name: 0.0 for name in (
            "solr_query_seconds", "transform_seconds", "audit_write_seconds",
            "strong_check_seconds", "partial_upsert_seconds", "retry_wait_seconds")}
        metrics.update(requested=len(ids), updated=0, solr_attempts=0, upsert_attempts=0,
                       solr_seconds=0.0, write_seconds=0.0, error=None,
                       queue_wait_seconds=0.0, prepared_wait_seconds=0.0)
        self._thread_local.metrics = metrics
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        error = None
        try:
            if self.settings.mode != "run":
                raise RuntimeError("batch preparation requires BACKFILL_MODE=run")
            self._assert_batch_scope(ids)
            if self.stop_event.is_set():
                raise RuntimeError("stopped before batch preparation")
            with timed(metrics, "solr_seconds"):
                rows, missing = self._solr_batch(ids)
            self._validate_prepared_rows(ids, rows, missing)
            if missing:
                self.stop_event.set()
        except Exception as exc:
            error = str(exc)
            metrics["error"] = error
            self.stop_event.set()
        finally:
            finished = time.perf_counter()
            metrics["prepare_seconds"] = finished - started
            with self._metrics_lock:
                self.preparation_metrics.append({**metrics, "missing_count": len(missing)})
            del self._thread_local.metrics
        return PreparedBatch(tuple(ids), rows, missing, error, metrics, started, finished)

    def _update_batch(self, ids: list[str], *, submitted_at: float | None = None,
                      prepared: PreparedBatch | None = None) -> tuple[int, list[str], str | None]:
        started = time.perf_counter()
        if self.settings.mode != "run":
            return 0, [], "data writes require BACKFILL_MODE=run"
        if self._sample_allowlist is not None and not set(ids).issubset(self._sample_allowlist):
            raise RuntimeError("write outside the fixed sample ID allowlist")
        batch = prepared if prepared is not None else self._prepare_batch(ids)
        metrics = dict(batch.metrics)
        metrics["queue_wait_seconds"] = max(0.0, started - submitted_at) if submitted_at else 0.0
        metrics["prepared_wait_seconds"] = max(0.0, started - batch.finished) if prepared is not None else 0.0
        write_start = None
        try:
            milvus = self._milvus_for_thread()
            if batch.ids != tuple(ids):
                raise RuntimeError("prepared batch IDs do not match write IDs")
            if batch.error:
                raise RuntimeError(batch.error)
            rows, missing = batch.rows, batch.missing
            write_start = time.perf_counter()
            # Revalidate at the write boundary even after successful prefetch.
            self._validate_prepared_rows(ids, rows, missing)
            row_ids = [r["id"] for r in rows]
            if missing:
                self.stop_event.set()
                return 0, missing, None
            if self.stop_event.is_set():
                raise RuntimeError("stopped before batch submission")
            if rows:
                if self._sample_expected is not None:
                    with timed(metrics, "audit_write_seconds"):
                        with self._metrics_lock:
                            with (self.settings.state_dir / "sample_payloads.jsonl").open("a", encoding="utf-8") as file:
                                for row in rows:
                                    self._sample_expected[row["id"]] = row
                                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                expected = {r["id"]: r for r in rows}
                for attempt in range(self.settings.max_retries + 1):
                    if self.stop_event.is_set():
                        raise RuntimeError("stopped before partial upsert")
                    # Recheck existence even on a retry. Never send originals
                    # or vectors to upsert. Preserve concurrently filled targets.
                    with timed(metrics, "strong_check_seconds"):
                        existing = milvus.get(
                            collection_name=self.settings.collection, ids=row_ids,
                            output_fields=["id", *TARGET_FIELDS],
                            consistency_level="Strong", timeout=self.settings.milvus_timeout)
                        if len(existing) != len(row_ids) or {r["id"] for r in existing} != set(row_ids):
                            raise RuntimeError("source record no longer exists; refusing partial upsert")
                        current_by_id = {r["id"]: r for r in existing}
                        for current in existing:
                            if not WRITE_FIELDS.issubset(current):
                                raise RuntimeError("target fields absent from Strong pre-write read")
                            if self._sample_allowlist is not None:
                                for name in TARGET_FIELDS:
                                    if current.get(name) is not None and (
                                            attempt == 0 or current[name] != expected[current["id"]][name]):
                                        raise RuntimeError(f"target field changed before write: {current['id']} {name}")
                        # Full runs fill null cells and preserve all existing target values,
                        # including empty strings. Re-merge on every uncertain RPC retry.
                        payload = rows if self._sample_allowlist is not None else [
                            {"id": row["id"], **{name: current_by_id[row["id"]][name]
                             if current_by_id[row["id"]][name] is not None else row[name]
                             for name in TARGET_FIELDS}} for row in rows]
                        for row in payload:
                            self._assert_write_payload(row)
                    try:
                        if self.stop_event.is_set():
                            raise RuntimeError("stopped after Strong check; refusing partial upsert")
                        metrics["upsert_attempts"] += 1
                        with timed(metrics, "partial_upsert_seconds"):
                            response = milvus.upsert(
                                collection_name=self.settings.collection, data=payload,
                                partial_update=True, timeout=self.settings.milvus_timeout,
                            )
                        if int(response.get("upsert_count", 0)) != len(rows):
                            raise RuntimeError(f"unexpected upsert_count: {response}")
                        break
                    except Exception:
                        if attempt == self.settings.max_retries:
                            raise
                        with timed(metrics, "retry_wait_seconds"):
                            if self.stop_event.wait(self.settings.retry_backoff * (attempt + 1)):
                                raise RuntimeError("stopped during Milvus retry wait")
            metrics["updated"] = len(rows)
            metrics["null_counts"] = {name: sum(r[name] is None for r in payload) for name in TARGET_FIELDS} if rows else {}
            return len(rows), missing, None
        except Exception as exc:
            metrics["error"] = str(exc)
            self.stop_event.set()
            return 0, [], str(exc)
        finally:
            if write_start is not None:
                metrics["write_seconds"] = time.perf_counter() - write_start
            metrics["total_seconds"] = time.perf_counter() - batch.started
            with self._metrics_lock:
                self.batch_metrics.append(metrics)

    def _run_pipeline(self, batches: list[list[str]], accept: Callable,
                      *, completed_batches: int, flush_reason: Callable[[int], str],
                      failure_prefix: str) -> list[str]:
        """At most two read-only preparations ahead of at most two writers.

        Flush forms a write barrier: drain the current write window, keep
        bounded Solr preparation running, then resume writes only on success.
        """
        capacity = self.settings.concurrency
        next_flush = (completed_batches // self.settings.flush_every_batches + 1) * self.settings.flush_every_batches
        cursor = 0
        preparing: dict[Any, list[str]] = {}
        active: dict[Any, list[str]] = {}
        flush_errors: list[str] = []
        with self._pipeline_executors() as (readers, writers):
            def fill_preparations() -> None:
                nonlocal cursor
                while len(preparing) < capacity and cursor < len(batches) and not self.stop_event.is_set():
                    batch_ids = batches[cursor]
                    preparing[readers.submit(self._prepare_batch, batch_ids)] = batch_ids
                    cursor += 1
                    self.pipeline_metrics["peak_pending_preparations"] = max(
                        self.pipeline_metrics["peak_pending_preparations"], len(preparing))

            while active or preparing or (cursor < len(batches) and not self.stop_event.is_set()):
                fill_preparations()
                # Observe preparation failures before releasing any ready write.
                for future in list(preparing):
                    if not future.done():
                        continue
                    if future.cancelled():
                        preparing.pop(future)
                        continue
                    prepared_batch = future.result()
                    if prepared_batch.error or prepared_batch.missing:
                        batch_ids = preparing.pop(future)
                        accept(batch_ids, self._update_batch(batch_ids, prepared=prepared_batch))
                    elif self.stop_event.is_set():
                        preparing.pop(future)  # read-only result; never replay it

                for future in list(active):
                    if future.done():
                        batch_ids = active.pop(future)
                        accept(batch_ids, future.result())
                        completed_batches += 1

                # Replenish before blocking in flush so its waiting time can
                # overlap the next read-only Solr queries and conversions.
                fill_preparations()
                if completed_batches >= next_flush and not active:
                    try:
                        self._flush(flush_reason(completed_batches))
                    except Exception as exc:
                        flush_errors.append(str(exc))
                        self._record_failure(failure_prefix, [], f"flush failed: {exc}")
                    next_flush += self.settings.flush_every_batches

                if not self.stop_event.is_set():
                    for future in list(preparing):
                        if len(active) >= capacity or completed_batches + len(active) >= next_flush:
                            break
                        if not future.done() or future.cancelled():
                            continue
                        prepared_batch = future.result()
                        # A preparation can finish while flush is running.
                        if prepared_batch.error or prepared_batch.missing or self.stop_event.is_set():
                            break
                        batch_ids = preparing.pop(future)
                        active[writers.submit(self._update_batch, batch_ids,
                                              submitted_at=time.perf_counter(), prepared=prepared_batch)] = batch_ids
                        self.pipeline_metrics["peak_active_writes"] = max(
                            self.pipeline_metrics["peak_active_writes"], len(active))
                    fill_preparations()
                else:
                    for future in preparing:
                        future.cancel()

                pending = [future for future in (*active, *preparing) if not future.done()]
                if pending:
                    wait(pending, return_when=FIRST_COMPLETED)
        return flush_errors

    def _collection_evidence(self) -> dict[str, Any]:
        """Read schema, logical row count and existing index definitions."""
        logical = self.milvus.query(collection_name=self.settings.collection, filter="",
                                    output_fields=["count(*)"], consistency_level="Strong",
                                    timeout=self.settings.milvus_timeout)
        indexes = {}
        for name in self.milvus.list_indexes(collection_name=self.settings.collection):
            item = self.milvus.describe_index(collection_name=self.settings.collection, index_name=name)
            indexes[name] = {k: v for k, v in item.items()
                             if k not in {"indexed_rows", "total_rows", "pending_index_rows", "state"}}
        return {"schema": self.collection_info(),
                "logical_row_count": int(logical[0]["count(*)"]),
                "stats": self.milvus.get_collection_stats(self.settings.collection),
                "index_definitions": indexes}

    def _excluded_sample_ids(self) -> set[str]:
        source = self.settings.exclude_sample_manifest
        if not source:
            return set()
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if (str(data.get("milvus_uri", "")).rstrip("/") != self.settings.milvus_uri.rstrip("/")
                or data.get("database") != self.settings.milvus_db_name
                or data.get("collection") != self.settings.collection):
            raise RuntimeError("excluded sample manifest target does not match current collection")
        excluded = []
        for name in ("ids", "control_ids"):
            values = data.get(name)
            if not isinstance(values, list) or any(
                    not isinstance(value, str) or not HEX_ID.fullmatch(value) for value in values):
                raise ValueError(f"invalid excluded manifest {name}")
            excluded.extend(values)
        if len(excluded) != len(set(excluded)):
            raise ValueError("duplicate IDs in excluded sample manifest")
        return set(excluded)

    def _select_sample(self, excluded: set[str]) -> tuple[list[str], dict[str, int]]:
        needed = self.settings.max_records + 20
        present = {f["name"] for f in self.collection_info()["fields"]}
        target_fields = [name for name in TARGET_FIELDS if name in present]
        ids: list[str] = []
        seen: set[str] = set()
        counts = {"scanned": 0, "excluded": 0, "already_populated": 0}
        sample_shards = (SHARDS[SHARDS.index(self.settings.sample_start_prefix):]
                         if self.settings.sample_start_prefix else SHARDS)
        for prefix in sample_shards:
            if self.stop_event.is_set():
                raise RuntimeError("stopped before any write")
            iterator = self.milvus.query_iterator(
                collection_name=self.settings.collection, filter=f"id like '{prefix}%'",
                output_fields=["id", *target_fields], batch_size=1000,
                consistency_level="Strong", timeout=self.settings.milvus_timeout)
            try:
                while not self.stop_event.is_set() and len(ids) < needed:
                    rows = iterator.next() or []
                    if not rows:
                        break
                    for row in rows:
                        if not {"id", *target_fields}.issubset(row):
                            raise RuntimeError("target fields absent from sample selection read")
                        key = row.get("id")
                        if (not isinstance(key, str) or not HEX_ID.fullmatch(key)
                                or not key.startswith(prefix) or key in seen):
                            raise RuntimeError(f"unsafe or duplicate source ID during selection: {key}")
                        seen.add(key)
                        counts["scanned"] += 1
                        if key in excluded:
                            counts["excluded"] += 1
                            continue
                        # Empty string also counts as an existing value.
                        if any(row.get(name) is not None for name in target_fields):
                            counts["already_populated"] += 1
                            continue
                        ids.append(key)
                        if len(ids) == needed:
                            break
            finally:
                iterator.close()
            if len(ids) == needed:
                return ids, counts
        raise RuntimeError(f"not enough unique empty-target IDs: need {needed}, found {len(ids)}")

    def _verify_new_values(self, ids: list[str], expected: dict[str, dict[str, Any]]) -> list[str]:
        mismatches = []
        for offset in range(0, len(ids), 200):
            batch = ids[offset:offset + 200]
            rows = self.milvus.get(collection_name=self.settings.collection, ids=batch,
                                   output_fields=["id", *TARGET_FIELDS], consistency_level="Strong",
                                   timeout=self.settings.milvus_timeout)
            returned = {r["id"]: self._plain(r) for r in rows}
            if len(returned) != len(rows) or not set(returned).issubset(batch):
                raise RuntimeError("duplicate or unrequested ID during new-field verification")
            mismatches.extend(key for key in batch if key not in expected or returned.get(key) != expected[key])
        return mismatches

    def _run_sample(self) -> dict[str, Any]:
        # Persist a failure report even if canary, flush or audit raises. Never
        # attempt rollback or replay after an uncertain RPC result.
        self._sample_run_summary: dict[str, Any] = {"mode": "bounded_real_sample", "passed": False}
        started = time.perf_counter()
        try:
            return self._execute_sample()
        except Exception as exc:
            self.stop_event.set()
            self._sample_run_summary.update(
                passed=False, error=str(exc), batches=self.batch_metrics, flushes=self.flush_metrics,
                preparations=self.preparation_metrics, pipeline=self.pipeline_metrics,
                acknowledged_updated=sum(m["updated"] for m in self.batch_metrics),
                total_seconds_including_audits=time.perf_counter() - started)
            # Do not overwrite an earlier completed run's summary on replay.
            if not self.summary_path.exists():
                self._write_json(self.summary_path, self._sample_run_summary)
            self._record_failure("sample", [], str(exc))
            raise

    def _execute_sample(self) -> dict[str, Any]:
        """Bounded real test. A persisted ID allowlist is the write boundary."""
        if self.settings.mode != "run" or not 1 <= self.settings.max_records <= 10000:
            raise ValueError("sample run requires mode=run and max_records between 1 and 10000")
        manifest = self.settings.state_dir / "sample_ids.json"
        if manifest.exists():
            raise RuntimeError("sample directory already used; refusing automatic sample replay")
        started = time.perf_counter()
        excluded = self._excluded_sample_ids()
        before_evidence = self._collection_evidence()
        self._write_json(self.settings.state_dir / "collection_before.json", before_evidence)
        ids, selection_counts = self._select_sample(excluded)
        selected, controls = ids[:self.settings.max_records], ids[self.settings.max_records:]
        self._sample_allowlist = frozenset(selected)
        self._write_json(manifest, {"milvus_uri": self.settings.milvus_uri,
                                   "database": self.settings.milvus_db_name,
                                   "collection": self.settings.collection,
                                   "ids": selected, "control_ids": controls,
                                   "excluded_sample_manifest": self.settings.exclude_sample_manifest,
                                   "excluded_id_count": len(excluded), "selection_counts": selection_counts})
        self._sample_run_summary.update(max_records=len(selected), selection_counts=selection_counts,
                                        concurrency=self.settings.concurrency, batch_size=self.settings.batch_size)
        LOGGER.info("fixed sample ready: %d writable IDs, %d read-only controls", len(selected), len(controls))
        before = self._hash_records(ids)
        self._write_json(self.settings.state_dir / "original_hashes_before.json", before)
        self._sample_expected = {}
        # No collection mutation has occurred before the complete baseline is persisted.
        if self.stop_event.is_set():
            raise RuntimeError("stopped before any write")
        self.preflight(mutate_schema=True)
        self.preflight(mutate_schema=False)
        batch_size = self.settings.batch_size
        batches = [selected[i:i + batch_size] for i in range(0, len(selected), batch_size)]
        updated = failed = 0
        write_started = time.perf_counter()

        def accept(batch: list[str], result: tuple[int, list[str], str | None]) -> None:
            nonlocal updated, failed
            count, missing, error = result
            updated += count
            failed += len(batch) if error else len(missing)
            if error or missing:
                self._record_failure("sample", batch if error else missing, error or "Solr returned no document")
                self.stop_event.set()
            self._write_json(self.progress_path, {"mode": "run", "sample_limit": len(selected),
                                                  "updated": updated, "failed": failed})
            LOGGER.info("sample progress: updated=%d failed=%d limit=%d", updated, failed, len(selected))

        # Verify a single batch immediately before releasing the rest of the sample.
        accept(batches[0], self._update_batch(batches[0]))
        if self.stop_event.is_set():
            raise RuntimeError("canary batch failed or stopped; no further writes")
        canary_audit_started = time.perf_counter()
        first_after = self._hash_records(batches[0])
        canary_changed = [k for k in batches[0] if before[k] != first_after[k]]
        canary_value_mismatches = self._verify_new_values(batches[0], self._sample_expected)
        canary = {"records": len(batches[0]), "changed_original_ids": canary_changed,
                  "new_field_mismatch_ids": canary_value_mismatches,
                  "seconds": time.perf_counter() - canary_audit_started,
                  "passed": not canary_changed and not canary_value_mismatches}
        self._write_json(self.settings.state_dir / "canary_audit.json", canary)
        self._sample_run_summary["canary"] = canary
        if not canary["passed"]:
            self.stop_event.set()
            raise RuntimeError("canary historical/new-field integrity mismatch; no further writes")
        LOGGER.info("first %d records passed historical fields, vectors and new-field checks", len(batches[0]))
        pipeline_started = time.perf_counter()
        flush_errors = self._run_pipeline(
            batches[1:], accept, completed_batches=1,
            flush_reason=lambda count: f"periodic_batch_{count}", failure_prefix="sample")
        pipeline_seconds = time.perf_counter() - pipeline_started
        write_seconds = time.perf_counter() - write_started
        self._flush("sample_final")
        pipeline_with_flush = time.perf_counter() - pipeline_started
        after = self._hash_records(ids)
        self._write_json(self.settings.state_dir / "original_hashes_after.json", after)
        changed = {key: [name for name in self.original_fields if before[key][name] != after[key][name]]
                   for key in ids if before[key] != after[key]}
        value_mismatches = self._verify_new_values(selected, self._sample_expected)
        control_expected = {key: {"id": key, **{name: None for name in TARGET_FIELDS}} for key in controls}
        control_mismatches = self._verify_new_values(controls, control_expected)
        after_evidence = self._collection_evidence()
        self._write_json(self.settings.state_dir / "collection_after.json", after_evidence)
        old_schema = {f["name"]: f for f in before_evidence["schema"]["fields"]}
        new_schema = {f["name"]: f for f in after_evidence["schema"]["fields"]}
        old_schema_unchanged = all(new_schema.get(k) == v for k, v in old_schema.items())
        indexes_unchanged = before_evidence["index_definitions"] == after_evidence["index_definitions"]
        identity_unchanged = (before_evidence["schema"].get("collection_id")
                              == after_evidence["schema"].get("collection_id"))
        row_count_delta = after_evidence["logical_row_count"] - before_evidence["logical_row_count"]
        passed = (updated == len(selected) and not failed and not changed and not value_mismatches
                  and not control_mismatches and not flush_errors and not self.stop_event.is_set()
                  and identity_unchanged and old_schema_unchanged and indexes_unchanged and row_count_delta == 0)
        pipeline_records = max(0, updated - len(batches[0]))
        backfill_with_flush = write_seconds - canary["seconds"] + self.flush_metrics[-1]["seconds"]
        summary = {"mode": "bounded_real_sample", "passed": passed,
                   "max_records": len(selected), "updated": updated, "failed": failed,
                   "concurrency": self.settings.concurrency, "batch_size": batch_size,
                   "original_fields_checked": self.original_fields,
                   "historical_records_checked": len(ids), "control_records_checked": len(controls),
                   "changed_original_fields": changed, "new_field_mismatch_ids": value_mismatches,
                   "control_new_field_mismatch_ids": control_mismatches,
                   "collection_identity_unchanged": identity_unchanged,
                   "old_schema_unchanged": old_schema_unchanged, "indexes_unchanged": indexes_unchanged,
                   "logical_rows_before": before_evidence["logical_row_count"],
                   "logical_rows_after": after_evidence["logical_row_count"], "row_count_delta": row_count_delta,
                   "write_seconds_including_canary_audit": write_seconds,
                   "pipeline_seconds": pipeline_seconds,
                   "pipeline_records": pipeline_records,
                   "pipeline_rows_per_sec": pipeline_records / pipeline_seconds,
                   "pipeline_seconds_including_final_flush": pipeline_with_flush,
                   "pipeline_rows_per_sec_including_final_flush": pipeline_records / pipeline_with_flush,
                   "backfill_seconds_including_final_flush_excluding_canary_audit": backfill_with_flush,
                   "backfill_rows_per_sec_including_final_flush": updated / backfill_with_flush,
                   "canary": canary, "flushes": self.flush_metrics, "flush_errors": flush_errors,
                   "pipeline": {"strategy": "bounded_solr_prefetch_with_flush_write_barrier",
                                **self.pipeline_metrics}, "preparations": self.preparation_metrics,
                   "excluded_id_count": len(excluded), "selection_counts": selection_counts,
                   "stage_seconds_sum": {name: sum(m.get(name, 0) for m in self.batch_metrics)
                                         for name in ("solr_query_seconds", "transform_seconds", "audit_write_seconds",
                                                      "strong_check_seconds", "partial_upsert_seconds", "retry_wait_seconds",
                                                      "queue_wait_seconds", "prepared_wait_seconds")},
                   "total_seconds_including_audits": time.perf_counter() - started,
                   "nonempty_counts": {name: sum(bool(r[name]) for r in self._sample_expected.values())
                                       for name in TARGET_FIELDS}, "batches": self.batch_metrics}
        self._write_json(self.summary_path, summary)
        LOGGER.info("bounded sample finished: passed=%s updated=%d failed=%d pipeline_rows_per_sec=%.1f",
                    passed, updated, failed, summary["pipeline_rows_per_sec"])
        return summary

    @staticmethod
    def _evidence_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        old_fields = {f["name"]: f for f in before["schema"]["fields"]}
        new_fields = {f["name"]: f for f in after["schema"]["fields"]}
        result = {
            "old_schema_unchanged": all(new_fields.get(k) == v for k, v in old_fields.items()),
            "indexes_unchanged": before["index_definitions"] == after["index_definitions"],
            "collection_identity_unchanged": before["schema"].get("collection_id") == after["schema"].get("collection_id"),
            "logical_rows_before": before["logical_row_count"],
            "logical_rows_after": after["logical_row_count"],
            "row_count_delta": after["logical_row_count"] - before["logical_row_count"],
        }
        result["passed"] = (result["old_schema_unchanged"] and result["indexes_unchanged"]
                            and result["collection_identity_unchanged"] and result["row_count_delta"] == 0)
        return result

    def _full_progress(self, prefix: str, stage: str) -> None:
        self._write_json(self.progress_path, {**self.checkpoint.summary(), "mode": "run",
                         "current_shard": prefix, "stage": stage,
                         "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    def _process_shard(self, prefix: str, limit: int | None = None) -> bool:
        self.checkpoint.start(prefix)
        self._full_progress(prefix, "reading_ids")
        try:
            ids = self._query_ids(prefix, limit=limit)
            updated = failed = completed_batches = 0
            if self.stop_event.is_set():
                self.checkpoint.set(prefix, "pending", scanned=len(ids))
                return False
            # Persist the original baseline so restart cannot hide an earlier change.
            audit_path = self.settings.state_dir / f"shard_{prefix}_original_hashes.json"
            if audit_path.exists():
                before = json.loads(audit_path.read_text(encoding="utf-8"))
                self.verify_integrity(before)
            else:
                count = min(len(ids), self.settings.sample_size)
                audit_ids = [ids[i * len(ids) // count] for i in range(count)] if count else []
                before = self._hash_records(audit_ids)
                self._write_json(audit_path, before)
            self._full_progress(prefix, "backfilling")

            def accept(batch: list[str], result: tuple[int, list[str], str | None]) -> None:
                nonlocal updated, failed, completed_batches
                count, missing, error = result
                completed_batches += 1
                updated += count
                if missing or error:
                    failed += len(batch) if error else len(missing)
                    self._record_failure(prefix, batch if error else missing, error or "Solr returned no document")
                    self.stop_event.set()
                self.checkpoint.set(prefix, "running", scanned=len(ids), updated=updated, failed=failed,
                                    error=error or ("Solr missing IDs" if missing else ""))
                self._full_progress(prefix, "backfilling")
                LOGGER.info("shard %s progress: updated=%d/%d failed=%d", prefix, updated, len(ids), failed)

            batches = [ids[offset:offset + self.settings.batch_size]
                       for offset in range(0, len(ids), self.settings.batch_size)]
            flush_errors = self._run_pipeline(batches, accept, completed_batches=0,
                flush_reason=lambda count: f"shard_{prefix}_batch_{count}", failure_prefix=prefix)
            if completed_batches:
                self._flush(f"shard_{prefix}_final")
            integrity = self.verify_integrity(before)
            status = ("completed" if not failed and not flush_errors and not self.stop_event.is_set()
                      and updated == len(ids) else "failed" if failed or flush_errors else "pending")
            self.checkpoint.set(prefix, status, scanned=len(ids), updated=updated, failed=failed,
                                error="; ".join(flush_errors))
            result = {"prefix": prefix, "status": status, "scanned": len(ids), "updated": updated,
                      "failed": failed, "integrity": integrity, "flush_errors": flush_errors,
                      "null_counts": {name: sum(m.get("null_counts", {}).get(name, 0) for m in self.batch_metrics)
                                      for name in TARGET_FIELDS},
                      "batches": self.batch_metrics, "preparations": self.preparation_metrics,
                      "flushes": self.flush_metrics}
            self._write_json(self.settings.state_dir / f"shard_{prefix}_result.json", result)
            self._full_progress(prefix, status)
            return False
        finally:
            # One shard's timing details are persisted; don't accumulate 100k batch dictionaries.
            self.batch_metrics.clear()
            self.preparation_metrics.clear()
            self.flush_metrics.clear()

    def run(self) -> dict[str, Any]:
        if self.settings.mode != "run":
            raise RuntimeError("run requires BACKFILL_MODE=run")
        solr_info = self.check_solr()
        LOGGER.info("Solr connectivity passed: %s", json.dumps(solr_info, ensure_ascii=False))
        self.preflight(mutate_schema=False)
        if self.settings.max_records:
            return self._run_sample()
        if (self.settings.state_dir / "sample_ids.json").exists():
            raise RuntimeError("bounded sample state cannot be used for a full backfill")
        try:
            return self._run_full()
        except Exception as exc:
            self.stop_event.set()
            self._record_failure("full", [], str(exc))
            self._write_json(self.summary_path, {**self.checkpoint.summary(), "passed": False, "error": str(exc)})
            LOGGER.exception("full backfill stopped")
            raise

    def _run_full(self) -> dict[str, Any]:
        started = time.perf_counter()
        current = self._collection_evidence()
        self.checkpoint.bind_target({"uri": self.settings.milvus_uri.rstrip("/"),
            "database": self.settings.milvus_db_name, "collection": self.settings.collection,
            "collection_id": current["schema"].get("collection_id"),
            "field_specs": FIELD_SPECS, "solr_mapping": SOLR_MAPPING,
            "parser_version": PARSER_VERSION, "solr_url": self.settings.solr_url})
        baseline_path = self.settings.state_dir / "full_baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            if not self._evidence_comparison(baseline["collection"], current)["passed"]:
                raise RuntimeError("collection identity/schema/index/count changed since baseline; stopping before writes")
            before = baseline["original_hashes"]
            self.verify_integrity(before)
        else:
            if self.checkpoint.summary()["shards"] != {"pending": len(SHARDS)}:
                raise RuntimeError("checkpoint has progress but its original baseline is missing")
            before = self._integrity_snapshot()
            baseline = {"collection": current, "original_hashes": before}
            self._write_json(baseline_path, baseline)
        self._write_json(self.settings.state_dir / "collection_run_start.json", current)
        if not self.stop_event.is_set():
            self.preflight(mutate_schema=True)
        for prefix in self.checkpoint.pending():
            if self.stop_event.is_set():
                break
            try:
                LOGGER.info("processing shard %s", prefix)
                self._process_shard(prefix)
            except Exception as exc:
                # Never move on to another shard after an uncertain failure.
                self.stop_event.set()
                LOGGER.exception("shard %s failed", prefix)
                self._record_failure(prefix, [], str(exc))
                with self.checkpoint.lock:
                    self.checkpoint.db.execute("UPDATE shards SET status='failed',last_error=? WHERE prefix=?",
                                               (str(exc)[:2000], prefix))
                    self.checkpoint.db.commit()
                break
        self._flush("full_final")
        integrity = self.verify_integrity(before)
        after = self._collection_evidence()
        self._write_json(self.settings.state_dir / "collection_run_end.json", after)
        evidence = self._evidence_comparison(baseline["collection"], after)
        summary = {**self.checkpoint.summary(), "mode": "full", "integrity": integrity,
                   "collection_checks": evidence, "stopped": self.stop_event.is_set(),
                   "seconds_this_run": time.perf_counter() - started,
                   "concurrency": self.settings.concurrency, "batch_size": self.settings.batch_size,
                   "flush_every_batches": self.settings.flush_every_batches,
                   "strategy": "bounded_solr_prefetch_with_flush_write_barrier",
                   "audit_scope": "startup sample and distributed sample in every shard; original fields including full vectors"}
        summary["scanned_matches_logical_count"] = summary["scanned"] == baseline["collection"]["logical_row_count"]
        summary["passed"] = (summary["shards"] == {"completed": len(SHARDS)} and not summary["failed"]
                             and not self.stop_event.is_set() and evidence["passed"] and integrity["passed"]
                             and summary["scanned_matches_logical_count"])
        self._write_json(self.summary_path, summary)
        self._full_progress("", "completed" if summary["passed"] else "stopped")
        return summary

    def install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: Any) -> None:
            LOGGER.warning("stop requested; finishing active batches")
            self.stop_event.set()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[logging.StreamHandler(), RotatingFileHandler(log_dir / "backfill.log", encoding="utf-8",
                                                              maxBytes=20 * 1024 * 1024, backupCount=5)],
    )


@contextmanager
def state_lock(state_dir: Path):
    """OS releases the lock on exit/crash. Shared state permits only one worker."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "worker.lock").open("a+b") as file:
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another worker owns this state volume; only one replica is allowed") from exc
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safely enrich Milvus documents from Solr")
    parser.add_argument("--healthcheck", action="store_true", help="container health check")
    args = parser.parse_args(argv)
    settings = Settings()
    if args.healthcheck:
        try:
            # This worker exposes no HTTP port. Check its process, never write data.
            pid = int((settings.state_dir / "worker.pid").read_text())
            if os.name != "posix":
                return 1
            os.kill(pid, 0)
            heartbeat = settings.state_dir / "heartbeat.json"
            if settings.mode == "run" and settings.max_records == 0:
                if time.time() - json.loads(heartbeat.read_text())["heartbeat_epoch"] > 90:
                    return 1
            return 0
        except (OSError, ValueError, KeyError, TypeError):
            return 1
    configure_logging(settings.log_dir)
    if settings.mode == "run" and settings.max_records == 0:
        try:
            from .resilient import supervise
        except ImportError:
            from resilient import supervise
        return supervise(settings)
    try:
        settings.validate()
        with state_lock(settings.state_dir):
            (settings.state_dir / "worker.pid").write_text(str(os.getpid()))
            worker = BackfillWorker(settings)
            worker.install_signal_handlers()
            try:
                if settings.mode == "check":
                    solr_info = worker.check_solr()
                    LOGGER.info("Solr connectivity passed: %s", json.dumps(solr_info, ensure_ascii=False))
                    result = worker.preflight(mutate_schema=False)
                    if settings.mongo_uri:
                        try:
                            from .resilient import check_mongo
                        except ImportError:
                            from resilient import check_mongo
                        check_mongo(settings)
                    LOGGER.info("preflight passed: %s", json.dumps(result, ensure_ascii=False))
                    LOGGER.info("BACKFILL_MODE=check: no schema or data write was performed")
                    return 0
                summary = worker.run()
                LOGGER.info("backfill finished: %s", json.dumps(summary, ensure_ascii=False))
                return 0 if summary.get("failed", 0) == 0 and summary.get("passed", False) else 2
            finally:
                worker.close()
    except Exception:
        LOGGER.exception("worker stopped; review summary.json and failures.jsonl before restarting")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
