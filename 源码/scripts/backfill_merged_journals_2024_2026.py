"""Extend the merged-vector test collection with 2026, 2025, then 2024 journals."""

from __future__ import annotations

import json
import msvcrt
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_merged_2026 import (
    DIM, FILTER, INSTRUCTION, MODEL_URL, SOURCE, SOURCE_ID, TARGET,
    connect, embed, first_text, model_key, normalize, validate_source,
)


YEARS = ("2026", "2025", "2024")
# Keep source queries small while the remote QueryNode recovers from the prior
# mmap allocation failure. The checkpoint remains the source of truth for resume.
BATCH_SIZE = 16
WORKERS = 4
SOURCE_QUERY_CONCURRENCY = 1
QUERY_RETRIES = 4
BATCH_RETRIES = 4
RECONNECT_RETRIES = 6
RECOVERY_RETRIES = 20
AUTO_RECOVERY_MAX_DELAY = 300
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "out" / "merged_2024_2026_journal"
DB_PATH = STATE / "checkpoint.db"
PROGRESS_PATH = STATE / "progress.json"
SUMMARY_PATH = STATE / "summary.json"
ZJ_IDS_PATH = STATE / "zijian_topic_ids.txt"
OLD_DB_PATH = ROOT / "out" / "merged_2026_100k" / "checkpoint.db"
LOG_PATH = STATE / "worker.log"
LOCK_PATH = STATE / "worker.lock"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(event: str, **values: Any) -> None:
    record = {"event": event, "at": now(), **values}
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


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
            written INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS errors (
            id TEXT NOT NULL,
            year TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            error TEXT NOT NULL,
            at TEXT NOT NULL
        );
    """)
    for year in YEARS:
        db.execute("INSERT OR IGNORE INTO stages(year) VALUES (?)", (year,))
    for meta in ("source_id", "target"):
        expected = SOURCE_ID if meta == "source_id" else TARGET
        db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES (?,?)", (meta, expected))
    db.commit()
    stored = dict(db.execute("SELECT key,value FROM meta"))
    assert stored == {"source_id": SOURCE_ID, "target": TARGET}, "checkpoint identity mismatch"
    db.execute("UPDATE records SET status='selected' WHERE status='inflight'")
    db.execute("UPDATE records SET status='selected', error=NULL WHERE status='failed'")
    db.commit()
    return db


def seed_existing_sample(db: sqlite3.Connection, client: MilvusClient) -> None:
    old = sqlite3.connect(f"file:{OLD_DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        legacy_count = old.execute("SELECT COUNT(*) FROM sample WHERE status='done'").fetchone()[0]
        rows = old.execute("SELECT id FROM sample WHERE status='done' ORDER BY seq")
        while batch := rows.fetchmany(5000):
            db.executemany("INSERT OR IGNORE INTO records(id,year,status,attempts) VALUES (?, '2026','done',1)", batch)
            db.commit()
    finally:
        old.close()
    checkpoint_done = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
    target_count = client.query(TARGET, filter='', output_fields=['count(*)'], consistency_level='Strong', timeout=120)[0]['count(*)']
    assert legacy_count == 100_000 and checkpoint_done >= legacy_count and target_count >= checkpoint_done, (
        f"seed/target mismatch: legacy={legacy_count}, checkpoint_done={checkpoint_done}, target={target_count}"
    )
    log("seeded_existing_100k", count=legacy_count, checkpoint_done=checkpoint_done,
        target_count=target_count)


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
    for attempt in range(BATCH_RETRIES):
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


def progress(db: sqlite3.Connection, year: str, started: float) -> dict[str, Any]:
    stages = {}
    for item in db.execute("SELECT year,source_count,selection_complete,written FROM stages ORDER BY year DESC"):
        selected = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (item[0],)).fetchone()[0]
        stages[item[0]] = {"source_count": item[1], "selected": selected,
                           "selection_complete": bool(item[2]), "written": item[3]}
    total_done = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
    failed = db.execute("SELECT COUNT(*) FROM records WHERE status='failed'").fetchone()[0]
    return {"state": "running", "current_year": year, "stages": stages,
            "processed": total_done, "failed": failed, "elapsed_seconds": round(time.monotonic() - started, 1),
            "updated_at": now(), "target_collection": TARGET}


def write_progress(db: sqlite3.Connection, year: str, started: float) -> None:
    tmp = PROGRESS_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(progress(db, year, started), ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(PROGRESS_PATH)


def write_batch(ids: list[str], fields: list[str], key: str,
                local: threading.local, query_slots: threading.Semaphore) -> tuple[list[str], dict[str, float]]:
    if not hasattr(local, 'session'):
        local.session = requests.Session()
    if not hasattr(local, 'client'):
        local.client = connect()
    phase_started = time.monotonic()
    fetched = query_source_ids(local, ids, fields, query_slots)
    query_seconds = time.monotonic() - phase_started
    mapped = {row['id']: normalize(row, fields, YEARS) for row in fetched}
    assert set(mapped) == set(ids), 'selected source IDs missing or duplicated'
    records = [mapped[identifier] for identifier in ids]
    phase_started = time.monotonic()
    vectors = embed(local.session, key, records)
    embedding_seconds = time.monotonic() - phase_started
    for record, vector in zip(records, vectors):
        record['text_embedding'] = vector
    phase_started = time.monotonic()
    upsert_with_retry(local, records, len(ids))
    upsert_seconds = time.monotonic() - phase_started
    return ids, {"source_query_seconds": query_seconds,
                 "embedding_seconds": embedding_seconds,
                 "upsert_seconds": upsert_seconds}


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
            active[pool.submit(write_batch, ids, fields, key, local, query_slots)] = ids
            return True

        while len(active) < WORKERS * 2 and enqueue():
            pass
        last_report = time.monotonic()
        while active or not stop.is_set():
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
                    _, phases = future.result()
                    for name, seconds in phases.items():
                        phase_seconds[name] += seconds
                    db.executemany("UPDATE records SET status='done',error=NULL WHERE id=?", [(identifier,) for identifier in ids])
                    db.commit()
                    done += len(ids)
                    db.execute("UPDATE stages SET written=? WHERE year=?", (done, year))
                    db.commit()
                    total = db.execute("SELECT COUNT(*) FROM records WHERE year=?", (year,)).fetchone()[0]
                    if done % 10000 < len(ids) or done == total:
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
    remaining = db.execute("SELECT COUNT(*) FROM records WHERE year=? AND status!='done'", (year,)).fetchone()[0]
    assert remaining == 0, f"{year}: {remaining} IDs remain unwritten"
    client.flush(TARGET)
    target_year = client.query(TARGET, filter=f'year == "{year}"', output_fields=['count(*)'], consistency_level='Strong', timeout=180)[0]['count(*)']
    assert target_year == total, f"{year}: target count {target_year}, expected {total}"
    log("year_verified", year=year, strong_target_count=target_year)


def verify_zijian(client: MilvusClient, fields: list[str], key: str) -> int:
    ids = [line.strip() for line in ZJ_IDS_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
    missing = []
    for offset in range(0, len(ids), 200):
        batch = ids[offset:offset + 200]
        rows = client.query(TARGET, filter='id in ' + json.dumps(batch), output_fields=['id'], consistency_level='Strong', timeout=120)
        missing.extend(identifier for identifier in batch if identifier not in {row['id'] for row in rows})
    if missing:
        source_rows = {}
        for offset in range(0, len(missing), 64):
            batch = missing[offset:offset + 64]
            rows = client.query(SOURCE, filter='id in ' + json.dumps(batch), output_fields=fields,
                                consistency_level='Strong', timeout=180)
            source_rows.update({row['id']: normalize(row, fields, YEARS) for row in rows})
            ordered = [source_rows[identifier] for identifier in batch if identifier in source_rows]
            vectors = embed(requests.Session(), key, ordered)
            for row, vector in zip(ordered, vectors):
                row['text_embedding'] = vector
            if ordered:
                client.upsert(TARGET, data=ordered, timeout=180)
        if len(source_rows) != len(missing):
            raise RuntimeError(f"{len(missing)-len(source_rows)} Zijian topic IDs are absent from source")
    log("zijian_ids_verified", count=len(ids), missing_before_repair=len(missing))
    return len(ids)


def assert_collections_loaded(client: MilvusClient) -> None:
    for collection in (SOURCE, TARGET):
        state = client.get_load_state(collection).get("state")
        if getattr(state, "name", None) != "Loaded":
            raise RuntimeError(f"{collection} is not loaded; current state: {state}")


def run_locked() -> int:
    stop = threading.Event()
    started = time.monotonic()
    client = connect()
    assert_collections_loaded(client)
    fields = validate_source(client)
    assert str(client.describe_collection(SOURCE)['collection_id']) == SOURCE_ID
    target = client.describe_collection(TARGET)
    assert len(target['fields']) == 42 and len(client.list_indexes(TARGET)) == 19
    db = open_db()
    seed_existing_sample(db, client)
    key = model_key()
    log("worker_started", years=list(YEARS), filter=FILTER, target=TARGET, batch_size=BATCH_SIZE,
        workers=WORKERS, source_query_concurrency=SOURCE_QUERY_CONCURRENCY, source_id=SOURCE_ID)

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

    zijian = verify_zijian(client, fields, key)
    client.flush(TARGET)
    final_count = client.query(TARGET, filter='', output_fields=['count(*)'], consistency_level='Strong', timeout=180)[0]['count(*)']
    written = db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0]
    expected = db.execute("SELECT SUM(selected) FROM stages").fetchone()[0]
    assert written == expected and final_count == expected, f"final count mismatch: written={written}, target={final_count}, expected={expected}"
    assert len(client.describe_collection(TARGET)['fields']) == 42
    assert len(client.list_indexes(TARGET)) == 19
    failures = db.execute("SELECT COUNT(*) FROM records WHERE status='failed'").fetchone()[0]
    assert failures == 0
    summary = {"state": "completed", "years_order": list(YEARS), "filter": FILTER,
               "source_collection_id": SOURCE_ID, "target_collection": TARGET,
               "per_year": {row[0]: {"source_count": row[1], "selected": row[2], "written": row[4]}
                            for row in db.execute("SELECT year,source_count,selected,selection_complete,written FROM stages")},
               "zijian_topic_ids": zijian, "written_count": written, "strong_target_count": final_count,
               "failed": failures, "field_count": 42, "index_count": 19, "completed_at": now()}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_progress(db, "completed", started)
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
            tmp = PROGRESS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(progress_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(PROGRESS_PATH)
            recovery_attempt += 1
            time.sleep(delay)


if __name__ == '__main__':
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
        PROGRESS_PATH.write_text(json.dumps(progress_snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
