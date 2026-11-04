"""Crash-safe local inbox and auditable per-message outcome ledger."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        self.handle.seek(0)
        self.handle.write(b"0")
        self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError("another incremental-sync instance owns this state directory") from error

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


class Ledger:
    """An event is fsynced before Kafka offset commit; outcome is durable too."""

    def __init__(self, state_dir: Path, identity: str) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "checkpoint.db"
        self.conn = sqlite3.connect(self.path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                topic TEXT NOT NULL, partition_id INTEGER NOT NULL, offset_id INTEGER NOT NULL,
                doc_id TEXT NOT NULL, received_at TEXT NOT NULL, received_day TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', outcome TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                completed_at TEXT, started_at TEXT,
                retry_after REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(topic, partition_id, offset_id)
            );
            CREATE INDEX IF NOT EXISTS events_work ON events(status, received_at);
            CREATE INDEX IF NOT EXISTS events_day ON events(received_day);
            CREATE INDEX IF NOT EXISTS events_received ON events(received_at);
            CREATE INDEX IF NOT EXISTS events_completed ON events(completed_at);
            CREATE TABLE IF NOT EXISTS failure_outbox (
                failure_id TEXT PRIMARY KEY, document TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS activity (
                day TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL, outcome TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS activity_day ON activity(day);
        """)
        self.identity = identity
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "duration_seconds" not in columns:
            self.conn.execute("ALTER TABLE events ADD COLUMN duration_seconds REAL NOT NULL DEFAULT 0")
        if "retry_after" not in columns:
            self.conn.execute("ALTER TABLE events ADD COLUMN retry_after REAL NOT NULL DEFAULT 0")
        prior = self.conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if prior and prior["value"] != identity:
            raise RuntimeError("state directory belongs to another Kafka/Milvus target identity")
        self.conn.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('identity',?)", (identity,))
        self.conn.commit()

    def enqueue(self, topic: str, partition: int, offset: int, doc_id: str) -> bool:
        received = datetime.now(timezone.utc)
        day = received.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
        before = self.conn.total_changes
        with self.conn:
            self.conn.execute("""INSERT OR IGNORE INTO events
                (topic,partition_id,offset_id,doc_id,received_at,received_day)
                VALUES(?,?,?,?,?,?)""", (topic, partition, offset, doc_id[:256], received.isoformat(), day))
        return self.conn.total_changes > before

    def next_event(self) -> dict[str, Any] | None:
        row = self.conn.execute("""SELECT * FROM events WHERE status IN ('queued','processing')
            AND (status='processing' OR retry_after <= ?)
            ORDER BY received_at, partition_id, offset_id LIMIT 1""", (time.time(),)).fetchone()
        return dict(row) if row else None

    def next_events(self, limit: int) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("event batch limit must be positive")
        rows = self.conn.execute("""SELECT * FROM events WHERE status IN ('queued','processing')
            AND (status='processing' OR retry_after <= ?)
            ORDER BY received_at, partition_id, offset_id LIMIT ?""",
            (time.time(), limit)).fetchall()
        return [dict(row) for row in rows]

    def begin(self, event: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("""UPDATE events SET status='processing',
                started_at=COALESCE(started_at,?), attempts=attempts+1
                WHERE topic=? AND partition_id=? AND offset_id=?""",
                (utc_now(), event["topic"], event["partition_id"], event["offset_id"]))

    def finish(self, event: dict[str, Any], outcome: str, error: str = "", *,
               duration: float = 0, target: dict[str, Any] | None = None,
               validation_run_id: str = "") -> None:
        if outcome not in {"inserted", "updated", "unchanged", "source_missing", "failed"}:
            raise ValueError("invalid event outcome")
        with self.conn:
            current = self.conn.execute("SELECT * FROM events WHERE topic=? AND partition_id=? AND offset_id=?",
                (event["topic"], event["partition_id"], event["offset_id"])).fetchone()
            if current is None or current["status"] == "done":
                return
            completed = utc_now()
            self.conn.execute("""UPDATE events SET status='done', outcome=?, last_error=?, completed_at=?
                ,duration_seconds=duration_seconds+?
                WHERE topic=? AND partition_id=? AND offset_id=?""",
                (outcome, error[:2000], completed, duration, event["topic"], event["partition_id"], event["offset_id"]))
            self._activity(completed, duration, outcome)
            if outcome in {"source_missing", "failed"}:
                key = hashlib.sha256(json.dumps([self.identity, event["topic"], event["partition_id"],
                    event["offset_id"]]).encode()).hexdigest()
                document = {"_id": key, "task": "metadata_incremental_sync", "task_identity": self.identity,
                    "target": target or {}, "topic": event["topic"], "partition": event["partition_id"],
                    "offset": event["offset_id"], "document_id": event["doc_id"], "stage": outcome,
                    "reason": error or "Solr has no matching document", "status": "pending",
                    "first_seen_at": current["received_at"], "last_seen_at": completed,
                    "attempts": current["attempts"]}
                if validation_run_id:
                    document.update(validation_test=True, validation_run_id=validation_run_id,
                                    status="validation_test")
                self.conn.execute("INSERT OR IGNORE INTO failure_outbox(failure_id,document) VALUES(?,?)",
                    (key, json.dumps(document, ensure_ascii=False)))

    def _activity(self, completed: str, duration: float, outcome: str) -> None:
        from datetime import timedelta
        end = datetime.fromisoformat(completed)
        start = end - timedelta(seconds=max(duration, 0))
        # Split attempts across Shanghai midnight; reports describe actual processing day.
        while start.date() <= end.date():
            local = start.astimezone(ZoneInfo("Asia/Shanghai"))
            midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(),
                tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(timezone.utc)
            stop = min(midnight, end)
            self.conn.execute("INSERT INTO activity VALUES(?,?,?,?,?)", (local.date().isoformat(),
                start.isoformat(), stop.isoformat(), max((stop-start).total_seconds(), 0),
                outcome if stop == end else "continuing"))
            if stop == end:
                break
            start = stop

    def defer(self, event: dict[str, Any], error: str, *, duration: float = 0, delay: float = 0) -> None:
        with self.conn:
            self.conn.execute("""UPDATE events SET status='queued', last_error=?, duration_seconds=duration_seconds+?
                ,retry_after=?
                WHERE topic=? AND partition_id=? AND offset_id=?""",
                (error[:2000], duration, time.time() + max(delay, 0), event["topic"], event["partition_id"], event["offset_id"]))
            self._activity(utc_now(), duration, "retry")

    def unsent(self) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self.conn.execute(
            "SELECT document FROM failure_outbox WHERE sent=0 LIMIT 100")]

    def mark_sent(self, keys: list[str]) -> None:
        with self.conn:
            self.conn.executemany("UPDATE failure_outbox SET sent=1 WHERE failure_id=?", [(key,) for key in keys])

    def outbox_pending(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM failure_outbox WHERE sent=0").fetchone()[0])

    def activity_daily(self, since_day: str = "0001-01-01") -> list[dict[str, Any]]:
        rows = self.conn.execute("""SELECT day, MIN(started_at) AS first_started_at,
            MAX(completed_at) AS last_completed_at, SUM(duration_seconds) AS active_processing_seconds,
            COUNT(*) AS attempts,
            SUM(outcome='inserted') AS inserted, SUM(outcome='updated') AS updated,
            SUM(outcome='unchanged') AS unchanged, SUM(outcome='failed') AS failed,
            SUM(outcome='source_missing') AS source_missing, SUM(outcome='retry') AS retries
            FROM activity WHERE day >= ? GROUP BY day ORDER BY day""", (since_day,))
        output = []
        for row in rows:
            item = dict(row)
            first, last = (datetime.fromisoformat(item[name]) for name in ("first_started_at", "last_completed_at"))
            item["elapsed_seconds"] = (last-first).total_seconds()
            for name, value in (("first_started_at", first), ("last_completed_at", last)):
                item[name] = value.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
            output.append(item)
        return output

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT COALESCE(outcome,status) AS name, COUNT(*) AS n FROM events GROUP BY 1")
        return {str(row["name"]): int(row["n"]) for row in rows}

    def report_days(self, since_utc: str) -> list[str]:
        rows = self.conn.execute("""SELECT received_day FROM events WHERE received_at >= ?
            UNION SELECT received_day FROM events WHERE completed_at >= ?
            UNION SELECT received_day FROM events WHERE status IN ('queued','processing')""",
            (since_utc, since_utc))
        return [str(row[0]) for row in rows]

    def daily(self, days: list[str] | None = None) -> list[dict[str, Any]]:
        if days is not None and not days:
            return []
        where = " WHERE received_day IN (" + ",".join("?" for _ in days) + ")" if days is not None else ""
        rows = self.conn.execute("""SELECT received_day, COUNT(*) AS received,
            MIN(received_at) AS first_received_at, MAX(completed_at) AS last_completed_at,
            MIN(started_at) AS first_started_at,
            SUM(CASE WHEN outcome='inserted' THEN 1 ELSE 0 END) AS inserted,
            SUM(CASE WHEN outcome='updated' THEN 1 ELSE 0 END) AS updated,
            SUM(CASE WHEN outcome='unchanged' THEN 1 ELSE 0 END) AS unchanged,
            SUM(CASE WHEN outcome='source_missing' THEN 1 ELSE 0 END) AS source_missing,
            SUM(CASE WHEN outcome='failed' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status!='done' THEN 1 ELSE 0 END) AS pending
            FROM events""" + where + " GROUP BY received_day ORDER BY received_day", days or [])
        output = []
        for row in rows:
            item = dict(row)
            first, last = item["first_started_at"], item["last_completed_at"]
            seconds = max((datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds(), 0) if first and last else 0
            item["processing_seconds"] = round(seconds, 1)
            item["completed_per_second"] = round((item["inserted"] + item["updated"] + item["unchanged"]) / seconds, 2) if seconds else 0
            output.append(item)
        return output

    def failures(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute("""SELECT topic, partition_id, offset_id, doc_id, outcome,
            last_error, attempts, received_at, completed_at FROM events
            WHERE outcome IN ('failed','source_missing') ORDER BY received_at DESC LIMIT ?""", (limit,))
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
