"""Build the independent 41-field metadata v3 collection from source Milvus.

The 2026, 2025 and 2024 journal stages are resumable. The retained Zijian
Kafka IDs are snapshotted separately and scheduled after those journal stages.
"""

from __future__ import annotations

import json
import faulthandler
import math
import msvcrt
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from confluent_kafka import Consumer, TopicPartition
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_merged_2026 import (
    DIM, FILTER, SOURCE, SOURCE_ID, URI,
    connect, embed, first_text, model_key,
)


TARGET = "scientific_documents_v3"
OLD_VECTORS = {"title_embed", "abstract_embed", "keyword_embed"}
MERGED_VECTOR = "text_embedding"
NULL_FLAG = "text_embedding_is_null"
ZJ_VERSION = "zijian_version"
NEAR_ZERO_NORM = 1e-6
ZERO_VECTOR = [0.0] * DIM
TOPIC = "VECTOR_JOURNAL_ARTICLE_FULLTEXT"
YEARS = ("2026", "2025", "2024")
BATCH_SIZE = 128
WORKERS = 6
# Source collection previously experienced QueryNode recovery; serialize source
# queries initially even though six embedding/write workers may run in parallel.
SOURCE_QUERY_CONCURRENCY = 1
QUERY_RETRIES = 4
BATCH_RETRIES = 4
RECONNECT_RETRIES = 6
RECOVERY_RETRIES = 20
AUTO_RECOVERY_MAX_DELAY = 300
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "out" / "metadata_v3_2024_2026"
DB_PATH = STATE / "checkpoint.db"
PROGRESS_PATH = STATE / "progress.json"
SUMMARY_PATH = STATE / "summary.json"
LOG_PATH = STATE / "worker.log"
LOCK_PATH = STATE / "worker.lock"
TOPIC_LOCK_PATH = STATE / "topic_scan.lock"


def validate_source(client: MilvusClient) -> list[str]:
    description = client.describe_collection(SOURCE)
    if str(description["collection_id"]) != SOURCE_ID:
        raise RuntimeError("source collection identity changed")
    all_fields = [field["name"] for field in description["fields"]]
    scalar_fields = [field for field in all_fields if field not in OLD_VECTORS]
    if len(all_fields) != 41 or len(scalar_fields) != 38 or "id" not in scalar_fields:
        raise RuntimeError("source schema is not the expected 41 fields")
    return scalar_fields


def normalize(row: dict[str, Any], fields: list[str], stage: str) -> dict[str, Any]:
    record = {}
    for field in fields:
        value = row.get(field)
        if value is not None and not isinstance(value, (str, int, float, bool, dict, list)):
            value = list(value)
        record[field] = value
    if stage in YEARS and (record["year"] != stage or record["sysSubType"] not in {"J02", "Z01"} or record["cls"] != "Article"):
        raise RuntimeError("source journal filter changed during backfill")
    record[ZJ_VERSION] = "v1" if stage == "zijian" else ""
    return record


def ensure_target(client: MilvusClient) -> None:
    expected = set(validate_source(client)) | {MERGED_VECTOR, NULL_FLAG, ZJ_VERSION}
    if not client.has_collection(TARGET):
        connections.connect(alias="metadata_v3_builder", uri=URI, db_name="default")
        source = Collection(SOURCE, using="metadata_v3_builder")
        fields = [field for field in source.schema.fields if field.name not in OLD_VECTORS]
        fields.extend((
            FieldSchema(name=MERGED_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name=NULL_FLAG, dtype=DataType.BOOL, default_value=False),
            FieldSchema(name=ZJ_VERSION, dtype=DataType.VARCHAR, max_length=16, default_value=""),
        ))
        Collection(TARGET, schema=CollectionSchema(
            fields=fields,
            description="V3 metadata: source ordinary fields, one merged text vector, null-vector flag",
            enable_dynamic_field=source.schema.enable_dynamic_field,
        ), using="metadata_v3_builder")
        log("collection_created", target=TARGET, field_count=len(fields))
    description = client.describe_collection(TARGET)
    names = {field["name"] for field in description["fields"]}
    if names != expected or len(names) != 41:
        raise RuntimeError("target collection schema mismatch")
    flag = next(field for field in description["fields"] if field["name"] == NULL_FLAG)
    if int(flag["type"]) != int(DataType.BOOL):
        raise RuntimeError("target null-vector flag is not Boolean")
    vector = next(field for field in description["fields"] if field["name"] == MERGED_VECTOR)
    if int(vector["params"]["dim"]) != DIM:
        raise RuntimeError("target merged vector dimension mismatch")
    existing = set(client.list_indexes(TARGET))
    for index_name in client.list_indexes(SOURCE):
        index = client.describe_index(SOURCE, index_name)
        if index["field_name"] in OLD_VECTORS or index_name in existing:
            continue
        params = client.prepare_index_params()
        params.add_index(field_name=index["field_name"], index_name=index_name,
                         index_type=index["index_type"])
        client.create_index(TARGET, params)
        existing.add(index_name)
        log("index_created", field=index["field_name"], index_type=index["index_type"])
    if MERGED_VECTOR not in existing:
        params = client.prepare_index_params()
        params.add_index(field_name=MERGED_VECTOR, index_name=MERGED_VECTOR,
                         index_type="IVF_PQ", metric_type="COSINE",
                         params={"nlist": 4096, "m": 64, "nbits": 8})
        client.create_index(TARGET, params)
        log("index_created", field=MERGED_VECTOR, index_type="IVF_PQ")
    for index_name in client.list_indexes(SOURCE):
        source_index = client.describe_index(SOURCE, index_name)
        if source_index["field_name"] in OLD_VECTORS:
            continue
        target_index = client.describe_index(TARGET, index_name)
        if (target_index["field_name"], target_index["index_type"]) != (
            source_index["field_name"], source_index["index_type"]
        ):
            raise RuntimeError(f"target scalar index differs from source: {index_name}")
    vector_index = client.describe_index(TARGET, MERGED_VECTOR)
    if (vector_index["index_type"] != "IVF_PQ" or vector_index["metric_type"] != "COSINE"
            or any(int(vector_index[key]) != expected_value for key, expected_value in
                   (("nlist", 4096), ("m", 64), ("nbits", 8)))):
        raise RuntimeError("merged vector index differs from source IVF_PQ settings")
    if client.get_load_state(TARGET).get("state").name != "Loaded":
        client.load_collection(TARGET, timeout=600)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(event: str, **values: Any) -> None:
    record = {"event": event, "at": now(), **values}
    line = json.dumps(record, ensure_ascii=False)
    try:
        print(line, flush=True)
    except OSError:
        pass
    try:
        with LOG_PATH.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
    except OSError as exc:
        # Logging is telemetry; a temporary log-file lock must not stop writes.
        try:
            print(json.dumps({"event": "worker_log_write_failed", "at": now(),
                              "error_type": type(exc).__name__,
                              "winerror": getattr(exc, "winerror", None)}),
                  file=sys.stderr, flush=True)
        except OSError:
            pass


def checkpoint_wal() -> None:
    """Bound the shared SQLite WAL during concurrent ID scans and writes."""
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as maintenance:
            maintenance.execute("PRAGMA busy_timeout=5000")
            busy, frames, checkpointed = maintenance.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
        if busy:
            log("wal_checkpoint_busy", frames=frames, checkpointed=checkpointed)
    except sqlite3.Error as exc:
        log("wal_checkpoint_retry_later", error_type=type(exc).__name__)


def acquire_lock():
    handle = LOCK_PATH.open("a+b")
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError("another merged-journal worker holds the lock")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode("ascii"))
    handle.flush()
    return handle


def open_db() -> sqlite3.Connection:
    STATE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA busy_timeout=60000")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            year TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'selected',
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS records_year_status ON records(year,status);
        CREATE TABLE IF NOT EXISTS stages (
            year TEXT PRIMARY KEY,
            source_count INTEGER,
            selected INTEGER NOT NULL DEFAULT 0,
            selection_complete INTEGER NOT NULL DEFAULT 0,
            written INTEGER NOT NULL DEFAULT 0,
            missing INTEGER NOT NULL DEFAULT 0,
            null_rows INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS errors (
            id TEXT NOT NULL,
            year TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            error TEXT NOT NULL,
            at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS topic_offsets (
            partition_id INTEGER PRIMARY KEY,
            low_offset INTEGER NOT NULL,
            high_offset INTEGER NOT NULL,
            next_offset INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS topic_manifest (
            id TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS topic_overlap (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'selected',
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
    """)
    for year in (*YEARS, "zijian"):
        db.execute("INSERT OR IGNORE INTO stages(year) VALUES (?)", (year,))
    for meta, expected in (("source_id", SOURCE_ID), ("target", TARGET),
                           ("schema_version", "metadata_v3_41_text_embedding"), ("topic", TOPIC)):
        db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES (?,?)", (meta, expected))
    db.commit()
    stored = dict(db.execute("SELECT key,value FROM meta"))
    assert stored == {"source_id": SOURCE_ID, "target": TARGET,
                       "schema_version": "metadata_v3_41_text_embedding", "topic": TOPIC}, "checkpoint identity mismatch"
    db.execute("UPDATE records SET status='selected' WHERE status='inflight'")
    db.execute("UPDATE records SET status='selected', error=NULL WHERE status='failed'")
    db.execute("UPDATE topic_overlap SET status='selected' WHERE status IN ('inflight','failed')")
    db.commit()
    return db


def count_source(client: MilvusClient, year: str) -> int:
    expression = f'year == "{year}" and sysSubType in ["J02", "Z01"] and cls == "Article"'
    active_client = client
    for attempt in range(RECOVERY_RETRIES):
        try:
            return active_client.query(SOURCE, filter=expression, output_fields=['count(*)'], consistency_level='Strong', timeout=180)[0]['count(*)']
        except Exception as exc:
            recovery = is_recovery_error(exc)
            max_attempts = RECOVERY_RETRIES if recovery else QUERY_RETRIES
            if not retryable_query_error(exc) or attempt + 1 >= max_attempts:
                raise
            delay = retry_delay(attempt, recovery)
            log("source_count_retry", year=year, attempt=attempt + 1, delay_seconds=delay,
                error_type=type(exc).__name__, message=str(exc)[:160])
            time.sleep(delay)
            if recovery:
                try:
                    close = getattr(active_client, "close", None)
                    if close:
                        close()
                except Exception:
                    pass
                active_client = connect()


def retryable_query_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "cannot allocate memory", "failed to map", "map_failed", "on recovering",
        "no available channel", "do not found any channel", "channel not found",
        "no channel in target", "collection is not loaded",
        "deadline exceeded", "timed out", "temporarily unavailable", "unavailable", "503",
        "closed channel", "cannot invoke rpc", "server unavailable", "fail connecting",
    ))


def is_recovery_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "on recovering", "no available channel", "do not found any channel",
        "server unavailable", "fail connecting", "closed channel",
        "cannot invoke rpc", "temporarily unavailable", "cannot allocate memory",
        "failed to map", "map_failed", "channel not found", "no channel in target",
        "collection is not loaded",
    ))


def retry_delay(attempt: int, recovery: bool) -> int:
    if recovery:
        return min(5 + attempt * 5, 30)
    return min(2 ** attempt, 15)


def reconnect_client(local: threading.local, rows: int) -> None:
    try:
        close = getattr(local.client, "close", None)
        if close:
            close()
    except Exception:
        pass
    for attempt in range(RECONNECT_RETRIES):
        try:
            local.client = connect()
            return
        except Exception as exc:
            if attempt + 1 == RECONNECT_RETRIES:
                raise
            delay = min(2 ** attempt, 20)
            log("milvus_reconnect_retry", rows=rows, attempt=attempt + 1,
                delay_seconds=delay, error_type=type(exc).__name__,
                message=str(exc)[:160])
            time.sleep(delay)


def query_source_ids(local: threading.local, ids: list[str], fields: list[str],
                     query_slots: threading.Semaphore) -> list[dict[str, Any]]:
    expression = 'id in ' + json.dumps(ids, ensure_ascii=False)
    for attempt in range(RECOVERY_RETRIES):
        try:
            with query_slots:
                return local.client.query(SOURCE, filter=expression, output_fields=fields,
                                    consistency_level='Strong', timeout=180)
        except Exception as exc:
            recovery = is_recovery_error(exc)
            max_attempts = RECOVERY_RETRIES if recovery else QUERY_RETRIES
            if not retryable_query_error(exc) or attempt + 1 >= max_attempts:
                raise
            delay = retry_delay(attempt, recovery)
            log("source_query_retry", rows=len(ids), attempt=attempt + 1,
                delay_seconds=delay, error_type=type(exc).__name__, message=str(exc)[:160])
            time.sleep(delay)
            reconnect_client(local, len(ids))


def upsert_with_retry(local: threading.local, records: list[dict[str, Any]],
                      rows: int) -> None:
    # Recovery-class outages need the full recovery budget. Never fall through
    # without an acknowledged upsert, or checkpoint rows could be marked done.
    for attempt in range(RECOVERY_RETRIES):
        try:
            local.client.upsert(TARGET, data=records, timeout=180)
            return
        except Exception as exc:
            recovery = is_recovery_error(exc)
            max_attempts = RECOVERY_RETRIES if recovery else BATCH_RETRIES
            if not retryable_query_error(exc) or attempt + 1 >= max_attempts:
                raise
            delay = retry_delay(attempt, recovery)
            log("upsert_retry", rows=rows, attempt=attempt + 1,
                delay_seconds=delay, error_type=type(exc).__name__,
                message=str(exc)[:160])
            time.sleep(delay)
            reconnect_client(local, rows)


def select_stage(client: MilvusClient, db_path: Path, year: str, stop: threading.Event, started: float) -> None:
    db = sqlite3.connect(db_path, timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    expression = f'year == "{year}" and sysSubType in ["J02", "Z01"] and cls == "Article"'
    for attempt in range(3):
        initial_count = count_source(client, year)
        resume_expression = expression
        checkpoint_max_id = db.execute(
            "SELECT MAX(id) FROM records WHERE year=?", (year,)
        ).fetchone()[0]
        if checkpoint_max_id is not None:
            prefix_filter = f'({expression}) and id <= "{checkpoint_max_id}"'
            prefix_source_count = client.query(
                SOURCE, filter=prefix_filter, output_fields=['count(*)'],
                consistency_level='Strong', timeout=180,
            )[0]['count(*)']
            prefix_checkpoint_count = db.execute(
                "SELECT COUNT(*) FROM records WHERE year=? AND id<=?",
                (year, checkpoint_max_id),
            ).fetchone()[0]
            if prefix_source_count == prefix_checkpoint_count:
                resume_expression = f'({expression}) and id > "{checkpoint_max_id}"'
                log("selection_resume_from_checkpoint", year=year,
                    checkpoint_max_id=checkpoint_max_id,
                    covered_source_count=prefix_source_count)
            else:
                log("selection_prefix_mismatch", year=year,
                    checkpoint_max_id=checkpoint_max_id,
                    covered_source_count=prefix_source_count,
                    checkpoint_prefix_count=prefix_checkpoint_count)
        db.execute("UPDATE stages SET source_count=? WHERE year=?", (initial_count, year))
        db.commit()
        pages = 0
        iterator = client.query_iterator(
            collection_name=SOURCE, filter=resume_expression, output_fields=['id'],
            batch_size=256, consistency_level='Strong', timeout=180,
        )
        try:
            while not stop.is_set():
                rows = iterator.next()
                if not rows:
                    break
                db.executemany("INSERT OR IGNORE INTO records(id,year,status) VALUES (?,?, 'selected')",
                               [(row['id'], year) for row in rows])
                db.commit()
                pages += 1
                if pages % 100 == 0:
                    selected_count = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (year,)).fetchone()[0]
                    db.execute("UPDATE stages SET selected=? WHERE year=?", (selected_count, year))
                    db.commit()
                    log("ids_selected", year=year,
                        selected=selected_count,
                        source_count=initial_count)
        finally:
            iterator.close()
        if stop.is_set():
            return
        selected = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (year,)).fetchone()[0]
        final_count = count_source(client, year)
        if selected == final_count:
            db.execute("UPDATE stages SET source_count=?,selected=?,selection_complete=1 WHERE year=?", (final_count, selected, year))
            db.commit()
            log("year_selected", year=year, source_count=final_count, checkpoint_count=selected, pass_number=attempt + 1)
            return
        log("selection_reconcile", year=year, selected=selected, source_before=initial_count,
            source_after=final_count, pass_number=attempt + 1)
    db.close()
    raise RuntimeError(f"source count and selected ID count did not converge for {year}")


def select_stage_guarded(client: MilvusClient, year: str, stop: threading.Event,
                         started: float, selector: threading.Thread) -> None:
    try:
        select_stage(client, DB_PATH, year, stop, started)
    except BaseException as exc:
        selector.failure = exc
        log("selector_error", year=year, error_type=type(exc).__name__, message=str(exc)[:250])
        stop.set()


def kafka_config() -> dict[str, Any]:
    env_path = ROOT / "vector_sync_service" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())
    config = {
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""),
        "group.id": "metadata_v3_read_only_topic_scan",
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "security.protocol": os.environ.get("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
        "sasl.mechanisms": os.environ.get("KAFKA_SASL_MECHANISM", "PLAIN"),
        "sasl.username": os.environ.get("KAFKA_SASL_USERNAME", ""),
        "sasl.password": os.environ.get("KAFKA_SASL_PASSWORD", ""),
    }
    if not config["bootstrap.servers"] or not config["sasl.username"]:
        raise RuntimeError("Kafka configuration is incomplete")
    return config


def topic_id(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"[0-9a-fA-F]{24}|[0-9a-fA-F]{32}", value):
            return value.lower()
        if value.startswith(("{", "[")):
            try:
                return topic_id(json.loads(value))
            except ValueError:
                return None
    if isinstance(value, list):
        return next((identifier for item in value if (identifier := topic_id(item))), None)
    if isinstance(value, dict):
        for key in ("id", "doc_id", "docId", "sourceId", "sysId", "articleId",
                    "data", "payload", "document", "record", "item"):
            if key in value and (identifier := topic_id(value[key])):
                return identifier
    return None


def save_topic_batch(db: sqlite3.Connection, partition: int, batch: list[tuple[int, Any]]) -> None:
    with db:
        for offset, raw in batch:
            identifier = topic_id(raw)
            if identifier is None:
                db.execute("INSERT INTO errors(id,year,attempts,error,at) VALUES (?,?,?,?,?)",
                           (f"{TOPIC}:{partition}:{offset}", "zijian", 0, "invalid Kafka ID", now()))
                continue
            db.execute("INSERT OR IGNORE INTO topic_manifest(id) VALUES (?)", (identifier,))
        db.execute("UPDATE topic_offsets SET next_offset=? WHERE partition_id=?",
                   (batch[-1][0] + 1, partition))
    if batch[-1][0] >= 500_000 and batch[-1][0] % 500_000 < len(batch):
        checkpoint_wal()


def materialize_topic(db: sqlite3.Connection) -> None:
    with db:
        db.execute("INSERT OR IGNORE INTO topic_overlap(id) "
                   "SELECT m.id FROM topic_manifest m JOIN records r ON r.id=m.id WHERE r.year IN ('2026','2025','2024')")
        db.execute("INSERT OR IGNORE INTO records(id,year,status) "
                   "SELECT m.id,'zijian','selected' FROM topic_manifest m")
        selected = db.execute("SELECT COUNT(*) FROM records WHERE year='zijian'").fetchone()[0]
        messages = db.execute("SELECT SUM(high_offset-low_offset) FROM topic_offsets").fetchone()[0]
        db.execute("UPDATE stages SET source_count=?,selected=?,selection_complete=1 WHERE year='zijian'",
                   (messages, selected))
    overlaps = db.execute("SELECT COUNT(*) FROM topic_overlap").fetchone()[0]
    log("topic_ids_selected", retained_messages=messages, new_unique_ids=selected,
        journal_overlap_ids=overlaps)


def _select_topic_unlocked(stop: threading.Event, materialize: bool = True) -> None:
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("CREATE TABLE IF NOT EXISTS topic_manifest(id TEXT PRIMARY KEY)")
    db.commit()
    consumer = Consumer(kafka_config())
    try:
        metadata = consumer.list_topics(TOPIC, timeout=30)
        info = metadata.topics.get(TOPIC)
        if info is None or info.error:
            raise RuntimeError(f"Kafka topic unavailable: {TOPIC}")
        partitions = sorted(info.partitions)
        existing = db.execute("SELECT COUNT(*) FROM topic_offsets").fetchone()[0]
        if not existing:
            for partition in partitions:
                low, high = consumer.get_watermark_offsets(TopicPartition(TOPIC, partition), timeout=30)
                db.execute("INSERT INTO topic_offsets VALUES (?,?,?,?)", (partition, low, high, low))
            db.commit()
            log("topic_snapshot_created", topic=TOPIC,
                partitions=len(partitions), retained_messages=sum(row[0] for row in db.execute(
                    "SELECT high_offset-low_offset FROM topic_offsets")))
        for partition, initial_low, high, next_offset in db.execute(
            "SELECT partition_id,low_offset,high_offset,next_offset FROM topic_offsets ORDER BY partition_id"
        ).fetchall():
            current_low, current_high = consumer.get_watermark_offsets(
                TopicPartition(TOPIC, partition), timeout=30)
            if next_offset < current_low or current_high < high:
                raise RuntimeError(f"Kafka retention or partition changed at {partition}: "
                                   f"next={next_offset}, available=[{current_low},{current_high})")
            consumer.assign([TopicPartition(TOPIC, partition, next_offset)])
            batch: list[tuple[int, Any]] = []
            idle_since = time.monotonic()
            while next_offset < high and not stop.is_set():
                message = consumer.poll(1)
                if message is None:
                    if time.monotonic() - idle_since > 90:
                        raise RuntimeError(f"Kafka read stalled at partition {partition}, offset {next_offset}")
                    continue
                if message.error():
                    raise RuntimeError(f"Kafka read error at partition {partition}: {message.error()}")
                if message.offset() >= high:
                    break
                batch.append((message.offset(), message.value()))
                next_offset = message.offset() + 1
                idle_since = time.monotonic()
                if len(batch) >= 5000:
                    save_topic_batch(db, partition, batch)
                    batch.clear()
                    if next_offset % 100000 < 5000:
                        log("topic_scan_progress", partition=partition, offset=next_offset,
                            high_offset=high)
            if batch:
                save_topic_batch(db, partition, batch)
            if stop.is_set():
                return
            if next_offset < high:
                raise RuntimeError(f"Kafka partition {partition} ended before snapshot high {high}")
            log("topic_partition_scanned", partition=partition,
                retained_messages=high - initial_low)
        messages = db.execute("SELECT SUM(high_offset-low_offset) FROM topic_offsets").fetchone()[0]
        manifest_ids = db.execute("SELECT COUNT(*) FROM topic_manifest").fetchone()[0]
        log("topic_manifest_ready", retained_messages=messages, unique_ids=manifest_ids,
            materialize=materialize)
        if materialize:
            materialize_topic(db)
    finally:
        consumer.close()
        db.close()


def select_topic(stop: threading.Event, materialize: bool = True) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    handle = TOPIC_LOCK_PATH.open("a+b")
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        _select_topic_unlocked(stop, materialize=materialize)
    finally:
        handle.close()


def select_topic_guarded(stop: threading.Event, selector: threading.Thread) -> None:
    try:
        select_topic(stop)
    except BaseException as exc:
        selector.failure = exc
        log("selector_error", year="zijian", error_type=type(exc).__name__, message=str(exc)[:250])
        stop.set()


def progress(db: sqlite3.Connection, year: str, started: float) -> dict[str, Any]:
    stages = {}
    for item in db.execute("SELECT year,source_count,selection_complete,written,missing,null_rows FROM stages ORDER BY year DESC"):
        selected = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (item[0],)).fetchone()[0]
        stages[item[0]] = {"source_count": item[1], "selected": selected,
                           "selection_complete": bool(item[2]), "written": item[3],
                           "missing": item[4], "null_rows": item[5]}
    total_done = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
    failed = db.execute("SELECT COUNT(*) FROM records WHERE status='failed'").fetchone()[0]
    missing = db.execute("SELECT COUNT(*) FROM records WHERE status='missing'").fetchone()[0]
    return {"state": "running", "current_year": year, "stages": stages,
            "processed": total_done, "failed": failed, "missing_source": missing,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "updated_at": now(), "target_collection": TARGET}


def publish_progress(snapshot: dict[str, Any]) -> bool:
    """Publish optional monitoring data without stopping a healthy data pipeline.

    The SQLite checkpoint is authoritative. On Windows a reader can briefly
    prevent replacing progress.json, so retry that access error and let the
    next reporting interval try again if the file remains occupied.
    """
    tmp = PROGRESS_PATH.with_name(f"{PROGRESS_PATH.name}.{os.getpid()}.tmp")
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    delays = (0.1, 0.2, 0.4, 0.8, 1.6)
    for attempt in range(len(delays) + 1):
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(PROGRESS_PATH)
            return True
        except OSError as exc:
            if not (isinstance(exc, PermissionError)
                    or getattr(exc, "winerror", None) in (5, 32, 33)):
                raise
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            log("progress_write_deferred", attempts=attempt + 1,
                error_type=type(exc).__name__, winerror=getattr(exc, "winerror", None))
            return False


def write_progress(db: sqlite3.Connection, year: str, started: float) -> None:
    publish_progress(progress(db, year, started))


def meaningful_text(row: dict[str, Any]) -> bool:
    title = first_text(row.get("title")).strip()
    abstract = first_text(row.get("abstract")).strip()
    keywords = row.get("keyword") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    parts = [title, abstract, *(first_text(item).strip() for item in keywords if item)]
    return any(part and not re.fullmatch(r"[+-]?0+(?:\.0+)?", part) for part in parts)


def write_batch(ids: list[str], fields: list[str], key: str, stage: str,
                local: threading.local, query_slots: threading.Semaphore) -> tuple[list[str], list[str], int, dict[str, float]]:
    if not hasattr(local, 'session'):
        local.session = requests.Session()
    if not hasattr(local, 'client'):
        local.client = connect()
    phase_started = time.monotonic()
    fetched = query_source_ids(local, ids, fields, query_slots)
    query_seconds = time.monotonic() - phase_started
    mapped = {row['id']: normalize(row, fields, stage) for row in fetched}
    if len(mapped) != len(fetched) or not set(mapped).issubset(ids):
        raise RuntimeError("source query returned duplicate or unrequested IDs")
    missing = [identifier for identifier in ids if identifier not in mapped]
    if stage in YEARS and missing:
        raise RuntimeError(f"selected journal IDs disappeared from source: {len(missing)}")
    records = [mapped[identifier] for identifier in ids if identifier in mapped]
    phase_started = time.monotonic()
    embedding_rows = [row for row in records if meaningful_text(row)]
    vectors = embed(local.session, key, embedding_rows) if embedding_rows else []
    embedding_seconds = time.monotonic() - phase_started
    null_count = len(records) - len(embedding_rows)
    for record in records:
        if not meaningful_text(record):
            record[MERGED_VECTOR] = ZERO_VECTOR.copy()
            record[NULL_FLAG] = True
    for record, vector in zip(embedding_rows, vectors):
        norm_sq = sum(float(value) ** 2 for value in vector)
        if not math.isfinite(norm_sq) or len(vector) != DIM:
            raise RuntimeError("embedding contains invalid values or dimensions")
        record[MERGED_VECTOR] = vector
        record[NULL_FLAG] = norm_sq <= NEAR_ZERO_NORM ** 2
        null_count += int(record[NULL_FLAG])
    phase_started = time.monotonic()
    if records:
        upsert_with_retry(local, records, len(records))
    upsert_seconds = time.monotonic() - phase_started
    return [row["id"] for row in records], missing, null_count, {
        "source_query_seconds": query_seconds,
        "embedding_seconds": embedding_seconds,
        "upsert_seconds": upsert_seconds,
    }


def write_stage(client: MilvusClient, db: sqlite3.Connection, fields: list[str], year: str,
                key: str, started: float, stop: threading.Event) -> None:
    local = threading.local()
    done = db.execute("SELECT COUNT(*) FROM records WHERE year=? AND status='done'", (year,)).fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (year,)).fetchone()[0]
    first_error = None
    stage_started = time.monotonic()
    stage_base_done = done
    phase_seconds = {"source_query_seconds": 0.0, "embedding_seconds": 0.0, "upsert_seconds": 0.0}
    query_slots = threading.Semaphore(SOURCE_QUERY_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        active: dict[Any, list[str]] = {}

        def enqueue() -> bool:
            if stop.is_set() or first_error:
                return False
            rows = db.execute("SELECT id FROM records WHERE year=? AND status='selected' ORDER BY rowid LIMIT ?",
                              (year, BATCH_SIZE)).fetchall()
            if not rows:
                return False
            ids = [item[0] for item in rows]
            db.executemany("UPDATE records SET status='inflight',attempts=attempts+1 WHERE id=?", [(identifier,) for identifier in ids])
            db.commit()
            active[pool.submit(write_batch, ids, fields, key, year, local, query_slots)] = ids
            return True

        while len(active) < WORKERS * 2 and enqueue():
            pass
        last_report = time.monotonic()
        last_checkpoint = time.monotonic()
        while active or not stop.is_set():
            if time.monotonic() - last_checkpoint >= 60:
                checkpoint_wal()
                last_checkpoint = time.monotonic()
            if not active:
                if stop.is_set():
                    break
                remaining = db.execute("SELECT COUNT(*) FROM records WHERE year=? AND status='selected'", (year,)).fetchone()[0]
                if remaining:
                    while len(active) < WORKERS * 2 and enqueue():
                        pass
                    continue
                stage_done = db.execute("SELECT selection_complete FROM stages WHERE year=?", (year,)).fetchone()[0]
                if stage_done:
                    break
                time.sleep(2)
                if time.monotonic() - last_report >= 10:
                    write_progress(db, year, started)
                    last_report = time.monotonic()
                continue
            completed, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
            for future in completed:
                ids = active.pop(future)
                try:
                    successful, missing_ids, null_count, phases = future.result()
                    for name, seconds in phases.items():
                        phase_seconds[name] += seconds
                    db.executemany("UPDATE records SET status='done',error=NULL WHERE id=?", [(identifier,) for identifier in successful])
                    db.executemany("UPDATE records SET status='missing',error='not found in source' WHERE id=?",
                                   [(identifier,) for identifier in missing_ids])
                    db.executemany("INSERT INTO errors(id,year,attempts,error,at) VALUES (?,?,?,?,?)",
                                   [(identifier, year, db.execute('SELECT attempts FROM records WHERE id=?', (identifier,)).fetchone()[0],
                                     'not found in source', now()) for identifier in missing_ids])
                    db.commit()
                    done += len(successful)
                    db.execute("UPDATE stages SET written=?,missing=missing+?,null_rows=null_rows+? WHERE year=?",
                               (done, len(missing_ids), null_count, year))
                    db.commit()
                    total = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (year,)).fetchone()[0]
                    if done % 10000 < len(successful) or done == total:
                        log("written", year=year, year_done=done, year_total=total,
                            total_done=db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0],
                            stage_rows_per_second=round((done - stage_base_done) / max(time.monotonic() - stage_started, 1), 2),
                            phase_seconds={name: round(seconds, 2) for name, seconds in phase_seconds.items()})
                except Exception as exc:
                    safe_message = f"{type(exc).__name__}: {str(exc)[:180]}"
                    db.executemany("UPDATE records SET status='failed',error=? WHERE id=?", [(safe_message, identifier) for identifier in ids])
                    db.executemany("INSERT INTO errors(id,year,attempts,error,at) VALUES (?,?,?,?,?)",
                                   [(identifier, year, db.execute('SELECT attempts FROM records WHERE id=?', (identifier,)).fetchone()[0], safe_message, now()) for identifier in ids])
                    db.commit()
                    first_error = exc
                    stop.set()
            while first_error is None and len(active) < WORKERS * 2 and enqueue():
                pass
            if time.monotonic() - last_report >= 10:
                write_progress(db, year, started)
                last_report = time.monotonic()
    if first_error:
        raise first_error
    if stop.is_set():
        raise RuntimeError("worker stopped by safety guard")
    remaining = db.execute("SELECT COUNT(*) FROM records WHERE year=? AND status NOT IN ('done','missing')", (year,)).fetchone()[0]
    assert remaining == 0, f"{year}: {remaining} IDs remain unwritten"
    client.flush(TARGET)
    if year in YEARS:
        expression = f'year == "{year}" and sysSubType in ["J02", "Z01"] and cls == "Article"'
        target_year = client.query(TARGET, filter=expression, output_fields=['count(*)'], consistency_level='Strong', timeout=180)[0]['count(*)']
        assert target_year == total, f"{year}: target count {target_year}, expected {total}"
        log("year_verified", year=year, strong_target_count=target_year)
    else:
        log("topic_stage_written", written=done,
            missing=db.execute("SELECT COUNT(*) FROM records WHERE year='zijian' AND status='missing'").fetchone()[0])


def write_topic_overlaps(client: MilvusClient, db: sqlite3.Connection) -> int:
    written = db.execute("SELECT COUNT(*) FROM topic_overlap WHERE status='done'").fetchone()[0]
    target_fields = [field["name"] for field in client.describe_collection(TARGET)["fields"]]
    while True:
        ids = [row[0] for row in db.execute(
            "SELECT id FROM topic_overlap WHERE status='selected' ORDER BY rowid LIMIT 128"
        )]
        if not ids:
            break
        with db:
            db.executemany("UPDATE topic_overlap SET status='inflight',attempts=attempts+1 WHERE id=?",
                           [(identifier,) for identifier in ids])
        for attempt in range(RECOVERY_RETRIES):
            try:
                rows = client.get(TARGET, ids=ids, output_fields=target_fields,
                                  consistency_level="Strong", timeout=180)
                if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
                    raise RuntimeError("topic overlap ID missing or duplicated in target")
                for row in rows:
                    row[ZJ_VERSION] = "v1"
                client.upsert(TARGET, data=rows, timeout=180)
                break
            except Exception as exc:
                if not retryable_query_error(exc) or attempt + 1 >= RECOVERY_RETRIES:
                    with db:
                        db.executemany("UPDATE topic_overlap SET status='failed',error=? WHERE id=?",
                                       [(f"{type(exc).__name__}: {str(exc)[:180]}", identifier) for identifier in ids])
                    raise
                time.sleep(retry_delay(attempt, is_recovery_error(exc)))
                client = connect()
        with db:
            db.executemany("UPDATE topic_overlap SET status='done',error=NULL WHERE id=?",
                           [(identifier,) for identifier in ids])
        written += len(ids)
        if written % 10000 < len(ids):
            log("topic_overlap_versioned", count=written)
    remaining = db.execute("SELECT COUNT(*) FROM topic_overlap WHERE status!='done'").fetchone()[0]
    if remaining:
        raise RuntimeError(f"{remaining} topic overlap IDs remain unversioned")
    return written


def assert_collections_loaded(client: MilvusClient) -> None:
    state = client.get_load_state(SOURCE).get("state")
    if getattr(state, "name", None) != "Loaded":
        raise RuntimeError(f"{SOURCE} is not loaded; current state: {state}")


def run_locked() -> int:
    stop = threading.Event()
    started = time.monotonic()
    client = connect()
    assert_collections_loaded(client)
    fields = validate_source(client)
    db = open_db()
    ensure_target(client)
    expected_indexes = len([name for name in client.list_indexes(SOURCE)
                            if client.describe_index(SOURCE, name)["field_name"] not in OLD_VECTORS]) + 1
    if len(client.list_indexes(TARGET)) != expected_indexes:
        raise RuntimeError("target scalar/vector index count mismatch")
    checkpoint_done = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
    target_count = client.query(TARGET, filter='', output_fields=['count(*)'],
                                consistency_level='Strong', timeout=180)[0]['count(*)']
    if target_count < checkpoint_done or (target_count and not db.execute(
        "SELECT 1 FROM records LIMIT 1").fetchone()):
        raise RuntimeError("target count and checkpoint identity mismatch")
    key = model_key()
    log("worker_started", years=list(YEARS), filter=FILTER, target=TARGET, batch_size=BATCH_SIZE,
        workers=WORKERS, source_query_concurrency=SOURCE_QUERY_CONCURRENCY, source_id=SOURCE_ID,
        field_count=41, index_count=expected_indexes, topic=TOPIC)

    for year in YEARS:
        stage = db.execute("SELECT selection_complete FROM stages WHERE year=?", (year,)).fetchone()
        if not stage[0]:
            selector = threading.Thread(target=lambda: select_stage_guarded(connect(), year, stop, started, selector), daemon=True)
            selector.start()
        else:
            selector = None
        try:
            write_stage(client, db, fields, year, key, started, stop)
        except Exception:
            if selector:
                selector.join(timeout=30)
                if getattr(selector, 'failure', None):
                    raise selector.failure
            raise
        if selector:
            selector.join()
            if selector.is_alive():
                raise RuntimeError(f"ID selection did not finish for {year}")
            # A selector exception is forwarded by the event field in the wrapper below.
            error = getattr(selector, 'failure', None)
            if error:
                raise error
        write_progress(db, year, started)

    topic_stage = db.execute("SELECT selection_complete FROM stages WHERE year='zijian'").fetchone()
    if not topic_stage[0]:
        topic_selector = threading.Thread(target=lambda: select_topic_guarded(stop, topic_selector), daemon=True)
        topic_selector.start()
    else:
        topic_selector = None
    try:
        write_stage(client, db, fields, "zijian", key, started, stop)
    except Exception:
        if topic_selector:
            topic_selector.join(timeout=30)
            if getattr(topic_selector, 'failure', None):
                raise topic_selector.failure
        raise
    if topic_selector:
        topic_selector.join()
        if getattr(topic_selector, 'failure', None):
            raise topic_selector.failure
    overlap_versioned = write_topic_overlaps(client, db)
    client.flush(TARGET)
    final_count = client.query(TARGET, filter='', output_fields=['count(*)'], consistency_level='Strong', timeout=180)[0]['count(*)']
    written = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
    missing = db.execute("SELECT COUNT(*) FROM records WHERE status='missing'").fetchone()[0]
    expected = db.execute("SELECT SUM(selected) FROM stages").fetchone()[0]
    assert written + missing == expected and final_count == written, (
        f"final count mismatch: written={written}, missing={missing}, target={final_count}, expected={expected}"
    )
    assert len(client.describe_collection(TARGET)['fields']) == 41
    assert len(client.list_indexes(TARGET)) == expected_indexes
    failures = db.execute("SELECT COUNT(*) FROM records WHERE status='failed'").fetchone()[0]
    assert failures == 0
    topic_messages = db.execute("SELECT SUM(high_offset-low_offset) FROM topic_offsets").fetchone()[0]
    invalid_topic_ids = db.execute("SELECT COUNT(*) FROM errors WHERE year='zijian' AND error='invalid Kafka ID'").fetchone()[0]
    summary = {"state": "completed", "years_order": list(YEARS), "filter": FILTER,
               "source_collection_id": SOURCE_ID, "target_collection": TARGET,
               "stages": {row[0]: {"source_count": row[1], "selected": row[2],
                                   "written": row[3], "missing": row[4], "null_rows": row[5]}
                          for row in db.execute("SELECT year,source_count,selected,written,missing,null_rows FROM stages")},
               "zijian_topic_retained_messages": topic_messages,
               "zijian_journal_overlap_versioned": overlap_versioned,
               "invalid_topic_ids": invalid_topic_ids,
               "written_count": written, "missing_source": missing,
               "strong_target_count": final_count, "failed": failures,
               "field_count": 41, "index_count": expected_indexes, "completed_at": now()}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    final_progress = progress(db, "completed", started)
    final_progress["state"] = "completed"
    publish_progress(final_progress)
    log("completed", **summary)
    return 0


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock()
    try:
        return run_locked()
    finally:
        lock.close()


def run_until_complete() -> int:
    recovery_attempt = 0
    while True:
        try:
            return main()
        except Exception as error:
            if not retryable_query_error(error):
                raise
            delay = min(30 * (2 ** min(recovery_attempt, 4)), AUTO_RECOVERY_MAX_DELAY)
            try:
                db = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10)
                processed, failed = db.execute(
                    "SELECT SUM(status='done'), SUM(status='failed') FROM records"
                ).fetchone()
                db.close()
            except Exception:
                processed, failed = None, None
            log("automatic_recovery_wait", attempt=recovery_attempt + 1,
                delay_seconds=delay, processed=processed, failed=failed,
                error_type=type(error).__name__, message=str(error)[:200])
            progress_snapshot = {
                "state": "waiting_for_milvus", "error_type": type(error).__name__,
                "message": str(error)[:250], "retry_in_seconds": delay,
                "recovery_attempt": recovery_attempt + 1,
                "processed": processed, "failed": failed, "updated_at": now(),
            }
            publish_progress(progress_snapshot)
            recovery_attempt += 1
            time.sleep(delay)


if __name__ == '__main__':
    # Emit Python/native crash stacks to the launcher's redirected stderr.
    faulthandler.enable(all_threads=True)
    try:
        sys.exit(run_until_complete())
    except Exception as error:
        STATE.mkdir(parents=True, exist_ok=True)
        log('paused_error', error_type=type(error).__name__, message=str(error)[:250])
        try:
            if DB_PATH.exists():
                c = sqlite3.connect(DB_PATH)
                c.execute("UPDATE records SET status='selected' WHERE status='inflight'")
                c.commit()
                c.close()
        except Exception:
            pass
        progress_snapshot = {"state": "safety_paused", "error_type": type(error).__name__,
                             "message": str(error)[:250], "updated_at": now()}
        publish_progress(progress_snapshot)
        raise
