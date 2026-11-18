"""Durable three-replica, field-only backfill; no collection/service operations.

Run in a dedicated directory on 231. SQLite commits a batch before its first
write, then advances the primary-key cursor only after all replicas are verified.
An interrupted batch is replayed idempotently before enumeration resumes.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import paramiko
import requests

from benchmark_fulltext_source_id_100 import (
    COLLECTION, FIELD, HOSTS, SOLR, SOLR_FIELD, connect, digest, plain, snapshot,
)

TZ = timezone(timedelta(hours=8))
BATCH = 500
PAGE = 20000


class IntegrityError(RuntimeError):
    pass


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(event, **kwargs):
    print(json.dumps({"at": now(), "event": event, **kwargs}, ensure_ascii=False), flush=True)


def encode(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), 1)


def decode(value):
    return json.loads(zlib.decompress(value))


def original_hash(row, fields):
    # All original scalar fields plus exact FLOAT_VECTOR float32 bytes.
    scalars = {k: row[k] for k in fields if k != "text_embed"}
    h = hashlib.sha256(json.dumps(scalars, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode())
    h.update(np.asarray(row["text_embed"], dtype="<f4").tobytes())
    return h.hexdigest()


def valid_source(value):
    return (isinstance(value, list) and 0 < len(value) <= 4096
            and all(isinstance(v, str) and v.strip() and len(v.encode()) <= 65535 for v in value))


class Runner:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.folder / "runner.lock", "a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(str(os.getpid()))
        self.lock.flush()
        self.db = sqlite3.connect(self.folder / "progress.sqlite", timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS batches(
          seq INTEGER PRIMARY KEY, status TEXT NOT NULL, payload BLOB NOT NULL,
          result TEXT, created_at TEXT NOT NULL, completed_at TEXT);
        CREATE TABLE IF NOT EXISTS source_cache(
          doc_id TEXT PRIMARY KEY, result TEXT NOT NULL, reason TEXT, checked_at TEXT NOT NULL);
        """)
        self.db.commit()
        self.clients = {h: connect(h) for h in HOSTS}
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.session = requests.Session()
        self.session.trust_env = False
        self.fields = None
        self.original_fields = None
        self.last_disk = 0.0
        self.last_counts = time.monotonic()
        self.last_batch_seconds = None
        self.runtime_start = time.monotonic()
        self.scan_start = int(self.state("scanned", 0))
        self.start_verified = int(self.state("eligible_verified", 0))
        self.start_retries = int(self.state("retry_events", 0))
        self.start_integrity_failures = int(self.state("integrity_failure_events", 0))

    def state(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value, ensure_ascii=False)))

    def parallel(self, action, record=None):
        futures = {h: self.pool.submit(action, h) for h in HOSTS}
        results, errors = {}, {}
        # Await every host even if another failed: never leave hidden writes running.
        for h, f in futures.items():
            try:
                results[h] = f.result()
            except Exception as exc:
                errors[h] = repr(exc)
            if record is not None:
                record(results, errors)
        if errors:
            raise RuntimeError(json.dumps(errors))
        return results

    def rows(self, host, ids):
        rows = self.clients[host].get(COLLECTION, ids=ids, output_fields=self.fields,
                                     consistency_level="Strong", timeout=90)
        # SDK ARRAY values may be protobuf RepeatedScalarContainer, which JSON
        # cannot encode. Normalize scalars/arrays; retain vectors for fast hashing.
        result = {r["id"]: {k: (v if k == "text_embed" else plain(v)) for k, v in r.items()}
                  for r in rows}
        if len(result) != len(ids) or set(result) != set(ids):
            raise IntegrityError("Existing primary key missing/duplicated on " + host)
        return result

    def status(self, status="running", **extra):
        elapsed = time.monotonic() - self.runtime_start
        scanned = int(self.state("scanned", 0))
        verified = int(self.state("eligible_verified", 0))
        data = {"updated_at": now(), "pid": os.getpid(), "status": status,
                "collection": COLLECTION, "field": FIELD, "hosts": list(HOSTS),
                "batch_size": BATCH, "initial_count_per_host": self.state("initial_count"),
                "started_at": self.state("started_at"),
                "backfill_started_at": self.state("backfill_started_at"),
                "current_run_started_at": self.state("current_run_started_at"),
                "document_id_bucket": self.state("document_id_bucket", 0),
                "scanned_chunks": scanned, "eligible_verified_per_host": verified,
                "newly_filled_per_host": self.state("newly_filled", {h: 0 for h in HOSTS}),
                "no_source_chunks": self.state("no_source", 0),
                "duplicate_keys_skipped": self.state("duplicate_keys_skipped", 0),
                "preserved_duplicate_extra_versions": self.state("preserved_duplicate_extra_versions", 0),
                "expected_current_count_per_host": self.state("expected_count", self.state("initial_count")),
                "initial_verified_10100": self.state("prior_verified", False),
                "completed_batches": self.state("completed_batches", 0),
                "cursor": self.state("cursor"), "retry_events": self.state("retry_events", 0),
                "current_run_retry_events": int(self.state("retry_events", 0))-self.start_retries,
                "integrity_failure_events": self.state("integrity_failure_events", 0),
                "current_run_integrity_failure_events": int(self.state("integrity_failure_events", 0))-self.start_integrity_failures,
                "runtime_seconds": round(elapsed, 2),
                "verified_rows_per_second": round((verified-self.start_verified) / max(elapsed, 1), 2),
                "scan_rows_per_second": round((scanned-self.scan_start) / max(elapsed, 1), 2),
                "last_batch_seconds": self.last_batch_seconds,
                "disk_checks": self.state("disk_checks"), **extra}
        if self.state("backfill_started_epoch"):
            run_s = time.time() - self.state("backfill_started_epoch")
            data["backfill_elapsed_seconds"] = round(run_s, 1)
            data["sustained_scanned_rows_per_second"] = round(scanned / max(run_s, 1), 2)
        tmp = self.folder / "status.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.folder / "status.json")
        return data

    def disk_guard(self, force=False):
        if not force and time.monotonic()-self.last_disk < 120:
            return
        password = os.environ["BACKFILL_SSH_PASSWORD"]
        def check(h):
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(h, username="root", password=password, timeout=10,
                      auth_timeout=10, banner_timeout=10)
            try:
                _, out, err = c.exec_command(
                    "python3 -c 'import os,json; print(json.dumps({p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in [\"/\",\"/data\"]}))'",
                    timeout=20)
                data = json.loads(out.read().decode())
                if out.channel.recv_exit_status() != 0:
                    raise RuntimeError("Disk guard failed: " + err.read().decode())
                if data["/"] < 40*1024**3 or data["/data"] < 1024**4:
                    raise RuntimeError("Free-space safety threshold reached on " + h)
                return {p: round(v/1024**3, 2) for p, v in data.items()}
            finally:
                c.close()
        checked = self.parallel(check)
        self.put("disk_checks", {"at": now(), "free_gib": checked})
        self.db.commit()
        self.last_disk = time.monotonic()

    def preflight(self):
        self.status("preflight")
        self.disk_guard(force=True)
        baselines = self.parallel(lambda h: snapshot(self.clients[h]))
        if len({digest(b) for b in baselines.values()}) != 1:
            raise IntegrityError("Collection counts/schema/index inventory differ across replicas")
        schema = baselines[HOSTS[0]][COLLECTION]
        self.fields = [f["name"] for f in schema["fields"]]
        self.original_fields = [f for f in self.fields if f != FIELD]
        target = next(f for f in schema["fields"] if f["name"] == FIELD)
        if (len(self.fields) != 19 or not target.get("nullable") or target["type"] != 22
                or target.get("element_type") != 21):
            raise IntegrityError("Unexpected schema; no alterations will be performed")
        previous = self.state("baseline")
        if previous is not None and baselines != previous:
            # This narrowly scoped recovery requires the recorded, read-only
            # historical audit, never an arbitrary count tolerance/rebaseline.
            path = self.folder / "count_diagnosis.json"
            diagnosis = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            normalized = copy.deepcopy(baselines)
            for h in HOSTS:
                normalized[h][COLLECTION]["count"] = previous[h][COLLECTION]["count"]
            delta = self.state("initial_count")-schema["count"]
            if not (normalized == previous and delta == 37
                    and diagnosis.get("original_baseline_count") == self.state("initial_count")
                    and diagnosis.get("global_count_now") == schema["count"]
                    and diagnosis.get("duplicate_extra_versions") == delta
                    and diagnosis.get("count_delta_exactly_duplicate_extras") is True
                    and diagnosis.get("historical_duplicate_visible_originals_match_journal") is True
                    and diagnosis.get("missing_keys") == []):
                raise IntegrityError("Baseline changed since the prior run; no verified recovery certificate")
            self.put("original_baseline_before_duplicate_diagnosis", previous)
            self.put("baseline", baselines)
            self.put("expected_count", schema["count"])
            self.put("duplicate_count_diagnosis", diagnosis)
            self.put("count_diagnosis_applied_at", now())
            self.db.commit()
            log("historical_duplicate_count_diagnosis_accepted", extra_versions=delta,
                expected_current_count=schema["count"], duplicate_keys_will_be_skipped=True)
        if previous is None:
            self.put("baseline", baselines)
            self.put("initial_count", schema["count"])
            self.put("started_at", now())
            self.db.commit()
        if self.state("prior_verified"):
            log("preflight_baseline_passed", count=schema["count"], resumed=True)
            return
        spec = json.loads((self.folder / "prior_10100.json").read_text(encoding="utf-8"))
        patches, hashes = spec["patches"], spec["original_hashes"]
        if len(patches) != 10100 or len({p["id"] for p in patches}) != 10100:
            raise IntegrityError("Invalid earlier-test inventory")
        for offset in range(0, len(patches), BATCH):
            group = patches[offset:offset+BATCH]
            ids = [p["id"] for p in group]
            expected = {p["id"]: p[FIELD] for p in group}
            def check(h):
                rows = self.rows(h, ids)
                for i, row in rows.items():
                    if row.get(FIELD) != expected[i]:
                        raise IntegrityError("Earlier source value differs: " + h + "/" + i)
                    if digest({k: row[k] for k in self.original_fields}) != hashes[i]:
                        raise IntegrityError("Earlier original fields/vector differ: " + h + "/" + i)
                return len(rows)
            self.parallel(check)
            self.status("preflight", prior_records_checked_per_host=offset+len(group))
            log("prior_records_verified", per_host=offset+len(group))
        self.put("prior_verified", True)
        self.put("prior_verified_at", now())
        self.db.commit()
        log("preflight_passed", prior_per_host=10100, total_count=schema["count"])

    def source_values(self, doc_ids):
        values, reasons, missing = {}, {}, []
        for doc_id in sorted(set(doc_ids)):
            row = self.db.execute("SELECT result,reason FROM source_cache WHERE doc_id=?", (doc_id,)).fetchone()
            if row:
                values[doc_id], reasons[doc_id] = json.loads(row[0]), row[1]
            else:
                missing.append(doc_id)
        for offset in range(0, len(missing), 200):
            wanted = missing[offset:offset+200]
            if any(not isinstance(i, str) or not i or any(c in i for c in ',\n\r') for i in wanted):
                raise IntegrityError("Unsafe/unexpected document ID for Solr terms query")
            response = self.session.post(SOLR, data={"q": "{!terms f=id}"+",".join(wanted),
                "fl": "id,"+SOLR_FIELD, "rows": len(wanted), "wt": "json", "timeAllowed": 15000},
                timeout=(10, 35))
            response.raise_for_status()
            body = response.json()
            if body.get("responseHeader", {}).get("partialResults"):
                raise RuntimeError("Incomplete Solr response")
            docs = body["response"]["docs"]
            if body["response"]["numFound"] != len(docs) or len({d["id"] for d in docs}) != len(docs):
                raise IntegrityError("Solr IDs are ambiguous/truncated")
            found = {d["id"]: d for d in docs}
            for doc_id in wanted:
                value = found.get(doc_id, {}).get(SOLR_FIELD)
                reason = None if valid_source(value) else ("solr_document_missing" if doc_id not in found else "source_id_missing_or_invalid")
                if reason:
                    value = None
                values[doc_id], reasons[doc_id] = value, reason
                self.db.execute("INSERT INTO source_cache VALUES(?,?,?,?)",
                                (doc_id, json.dumps(value, ensure_ascii=False), reason, now()))
        self.db.commit()
        return values, reasons

    def prepare(self, selected):
        ids = [r["id"] for r in selected]
        if any(not r["id"].startswith(r["doc_id"]+"_") for r in selected):
            raise IntegrityError("Unexpected chunk primary-key/document-ID relationship")
        if len(ids) != len(set(ids)) or ids != sorted(ids):
            raise IntegrityError("Iterator primary-key ordering/uniqueness violated")
        cursor = self.state("cursor")
        if cursor and any(i <= cursor for i in ids):
            raise IntegrityError("Iterator returned already committed keys")
        # An upsert can coalesce pre-existing duplicate primary-key versions.
        # Preserve those versions by excluding their keys from this backfill.
        physical_counts = self.parallel(lambda h: self.key_count(h, ids))
        if len(set(physical_counts.values())) != 1 or min(physical_counts.values()) < len(ids):
            raise IntegrityError("Primary-key multiplicity differs across hosts or a key is missing")
        duplicate_keys = {}
        if physical_counts[HOSTS[0]] > len(ids):
            duplicates = self.parallel(lambda h: self.find_duplicate_keys(h, ids))
            if len({digest(d) for d in duplicates.values()}) != 1:
                raise IntegrityError("Duplicate-key identities differ across hosts")
            duplicate_keys = duplicates[HOSTS[0]]
        values, reasons = self.source_values([r["doc_id"] for r in selected])
        expected = {r["id"]: values[r["doc_id"]] for r in selected}
        def check(h):
            rows = self.rows(h, ids)
            for selected_row in selected:
                i = selected_row["id"]
                if rows[i]["doc_id"] != selected_row["doc_id"]:
                    raise IntegrityError("Document ID differs on " + h)
                current, target = rows[i].get(FIELD), expected[i]
                if current is not None and target is not None and current != target:
                    raise IntegrityError("Refusing to replace a conflicting existing value on " + h + "/" + i)
            return {"hashes": {i: original_hash(r, self.original_fields) for i, r in rows.items()},
                    "values": {i: r.get(FIELD) for i, r in rows.items()}, "rows": rows}
        originals = self.parallel(check)
        if len({digest(v["hashes"]) for v in originals.values()}) != 1:
            raise IntegrityError("Original records/vectors differ across hosts before write")
        # Rows without a source must already match; never erase or fabricate IDs.
        for i in ids:
            if expected[i] is None and len({digest(originals[h]["values"][i]) for h in HOSTS}) != 1:
                raise IntegrityError("Existing source values differ for a source-missing row")
        patches = [{"id": i, FIELD: expected[i]} for i in ids
                   if expected[i] is not None and i not in duplicate_keys]
        payload = {"ids": ids, "cursor": ids[-1], "patches": patches,
                   "duplicate_keys": duplicate_keys,
                   "original_hashes": originals[HOSTS[0]]["hashes"],
                   "original_values": {h: originals[h]["values"] for h in HOSTS},
                   "no_source": [{"id": r["id"], "doc_id": r["doc_id"], "reason": reasons[r["doc_id"]]}
                                 for r in selected if expected[r["id"]] is None]}
        with self.db:
            seq = self.db.execute("INSERT INTO batches(status,payload,created_at) VALUES('pending',?,?)",
                                  (encode(payload), now())).lastrowid
        # Reuse this fresh pre-write read once. A retry or process restart must
        # read again; the original hashes remain durable in the pending journal.
        self.fresh_before = (seq, {h: originals[h]["rows"] for h in HOSTS})
        return seq, payload

    def key_count(self, host, ids):
        return int(self.clients[host].query(COLLECTION, filter="id in "+json.dumps(ids),
            output_fields=["count(*)"], consistency_level="Strong", timeout=90)[0]["count(*)"])

    def find_duplicate_keys(self, host, ids):
        count = self.key_count(host, ids)
        if count < len(ids):
            raise IntegrityError("Missing primary key during duplicate detection")
        if count == len(ids):
            return {}
        if len(ids) == 1:
            return {ids[0]: count}
        half = len(ids)//2
        return {**self.find_duplicate_keys(host, ids[:half]),
                **self.find_duplicate_keys(host, ids[half:])}

    def apply_and_verify(self, seq, payload):
        ids = payload["ids"]
        expected = {r["id"]: r[FIELD] for r in payload["patches"]}
        fresh = getattr(self, "fresh_before", None)
        before_rows = fresh[1] if fresh and fresh[0] == seq else None
        self.fresh_before = None
        def apply(h):
            before = before_rows[h] if before_rows is not None else self.rows(h, ids)
            updates = []
            for i, row in before.items():
                if original_hash(row, self.original_fields) != payload["original_hashes"][i]:
                    raise IntegrityError("Original record changed before/retrying patch on " + h + "/" + i)
                value = row.get(FIELD)
                if i in expected:
                    if value is None:
                        updates.append({"id": i, FIELD: expected[i]})
                    elif value != expected[i]:
                        raise IntegrityError("Conflicting source value on " + h + "/" + i)
                elif value != payload["original_values"][h][i]:
                    raise IntegrityError("Unexpected change to source-missing row on " + h)
            started = time.monotonic()
            if updates:
                if self.key_count(h, [r["id"] for r in updates]) != len(updates):
                    raise IntegrityError("Refusing to update duplicate primary-key versions on "+h)
                result = self.clients[h].upsert(COLLECTION, data=updates, partial_update=True, timeout=120)
                if int(result.get("upsert_count", -1)) != len(updates):
                    raise RuntimeError("Unexpected update count on " + h)
            write_s = time.monotonic()-started
            after = self.rows(h, ids)
            for i, row in after.items():
                if original_hash(row, self.original_fields) != payload["original_hashes"][i]:
                    raise IntegrityError("Original fields/vector changed after patch on " + h + "/" + i)
                wanted = expected.get(i, payload["original_values"][h][i])
                if row.get(FIELD) != wanted:
                    raise IntegrityError("Source value readback mismatch on " + h + "/" + i)
            return {"written_this_attempt": len(updates), "write_seconds": round(write_s, 6),
                    "newly_filled": sum(payload["original_values"][h][i] is None for i in expected),
                    "verified": len(ids), "eligible": len(expected)}
        def record_attempt(results, errors):
            with self.db:
                self.db.execute("UPDATE batches SET result=? WHERE seq=?",
                    (json.dumps({"verified_hosts": results, "errors": errors, "at": now()}), seq))
        result = self.parallel(apply, record=record_attempt)
        # Atomically mark all-replica verification and the enumeration cursor.
        with self.db:
            self.db.execute("UPDATE batches SET status='completed',result=?,completed_at=? WHERE seq=?",
                            (json.dumps(result), now(), seq))
            self.put("cursor", payload["cursor"])
            self.put("scanned", int(self.state("scanned", 0))+len(ids))
            self.put("eligible_verified", int(self.state("eligible_verified", 0))+len(expected))
            self.put("no_source", int(self.state("no_source", 0))+len(payload["no_source"]))
            duplicates = payload.get("duplicate_keys", {})
            self.put("duplicate_keys_skipped", int(self.state("duplicate_keys_skipped", 0))+len(duplicates))
            self.put("preserved_duplicate_extra_versions", int(self.state("preserved_duplicate_extra_versions", 0))
                     +sum(n-1 for n in duplicates.values()))
            filled = self.state("newly_filled", {h: 0 for h in HOSTS})
            for h in HOSTS:
                filled[h] += result[h]["newly_filled"]
            self.put("newly_filled", filled)
            self.put("completed_batches", int(self.state("completed_batches", 0))+1)
        return result

    def count_guard(self):
        initial_count = self.state("expected_count", self.state("initial_count"))
        def check(h):
            actual = int(self.clients[h].query(COLLECTION, filter="", output_fields=["count(*)"],
                consistency_level="Strong", timeout=90)[0]["count(*)"])
            if actual != initial_count:
                raise IntegrityError("Collection count changed on " + h)
            return actual
        result = self.parallel(check)
        self.put("last_count_check", {"at": now(), "counts": result})
        self.db.commit()
        self.last_counts = time.monotonic()

    def retry(self, action, phase):
        for attempt in range(1, 6):
            try:
                return action()
            except IntegrityError:
                raise
            except Exception as exc:
                # parallel() collects failures; integrity failures must still halt.
                if "IntegrityError(" in str(exc):
                    raise IntegrityError(str(exc)) from exc
                self.put("retry_events", int(self.state("retry_events", 0))+1)
                self.db.commit()
                self.status("retrying", phase=phase, error=repr(exc), attempt=attempt)
                log("retry", phase=phase, attempt=attempt, error=repr(exc))
                if attempt == 5:
                    raise
                time.sleep(min(5*2**(attempt-1), 60))

    def run(self):
        self.retry(self.preflight, "preflight")
        # The one-hour observation window begins only after preflight succeeds.
        if not self.state("backfill_started_at") or (
                not self.state("scanned", 0)
                and not self.db.execute("SELECT 1 FROM batches WHERE status='pending' LIMIT 1").fetchone()):
            self.put("backfill_started_at", now())
            self.put("backfill_started_epoch", time.time())
            self.db.commit()
        self.runtime_start = time.monotonic()
        self.put("current_run_started_at", now())
        self.db.commit()
        for seq, blob in self.db.execute("SELECT seq,payload FROM batches WHERE status='pending' ORDER BY seq").fetchall():
            self.retry(lambda: self.apply_and_verify(seq, decode(blob)), "recover_pending_batch")
            log("pending_batch_recovered", seq=seq)
        self.status()
        log("backfill_started", cursor=self.state("cursor"), batch_size=BATCH)
        if (self.folder / "STOP_REQUESTED").exists():
            self.put("stopped_at", now())
            self.db.commit()
            self.status("stopped_by_user")
            log("stopped_by_user", reason="Persistent user stop marker; no new batches")
            return
        # Bound every iterator by the EXISTING doc_id INVERTED index. An
        # unbounded primary-key-only continuation timed out after 10,000 rows.
        # 4096 disjoint lexical ranges cover the stored MD5 document IDs.
        while int(self.state("document_id_bucket", 0)) < 4096:
            if (self.folder / "STOP_REQUESTED").exists():
                self.put("stopped_at", now())
                self.db.commit()
                self.status("stopped_by_user")
                return
            self.retry(self.disk_guard, "disk_guard")
            bucket = int(self.state("document_id_bucket", 0))
            lower = format(bucket, "03x")
            upper = format(bucket+1, "03x") if bucket < 4095 else "g"
            cursor = self.state("cursor")
            expr = "doc_id >= " + json.dumps(lower) + " and doc_id < " + json.dumps(upper)
            if cursor:
                expr += " and id > " + json.dumps(cursor)
            iterator = None
            page_rows = 0
            try:
                iterator = self.retry(lambda: self.clients[HOSTS[0]].query_iterator(
                    COLLECTION, filter=expr, batch_size=BATCH, limit=PAGE,
                    output_fields=["id", "doc_id"], consistency_level="Strong", timeout=90), "iterator_open")
                while page_rows < PAGE:
                    if (self.folder / "STOP_REQUESTED").exists():
                        self.put("stopped_at", now())
                        self.db.commit()
                        self.status("stopped_by_user")
                        return
                    # A failed iterator.next() is reopened from the committed cursor.
                    selected = iterator.next()
                    if not selected:
                        break
                    started = time.monotonic()
                    seq, payload = self.retry(lambda: self.prepare(selected), "prepare")
                    results = self.retry(lambda: self.apply_and_verify(seq, payload), "apply_and_verify")
                    page_rows += len(selected)
                    self.last_batch_seconds = round(time.monotonic()-started, 3)
                    self.status()
                    log("batch_verified", seq=seq, rows=len(selected), eligible=len(payload["patches"]),
                        no_source=len(payload["no_source"]), seconds=self.last_batch_seconds,
                        scanned=self.state("scanned"), new_per_host=self.state("newly_filled"), hosts=results)
                    self.retry(self.disk_guard, "disk_guard")
                    if time.monotonic()-self.last_counts >= 600:
                        self.retry(self.count_guard, "count_guard")
            except IntegrityError:
                raise
            except Exception:
                # Never advance around a pending batch or a preparation failure.
                raise
            finally:
                if iterator:
                    iterator.close()
            if page_rows < PAGE:
                self.put("document_id_bucket", bucket+1)
                self.db.commit()
                self.status()
                log("document_bucket_completed", bucket=lower, scanned=self.state("scanned"))
        self.count_guard()
        if self.state("scanned")+self.state("preserved_duplicate_extra_versions", 0) != self.state("expected_count", self.state("initial_count")):
            raise IntegrityError("Enumeration ended before the expected total; inspect nonstandard document IDs")
        status = "completed_with_unresolved_records" if (self.state("no_source", 0)
            or self.state("duplicate_keys_skipped", 0)) else "completed"
        self.put("finished_at", now())
        self.db.commit()
        self.status(status)
        log(status, scanned=self.state("scanned"), no_source=self.state("no_source", 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if not args.apply:
        p.error("--apply required for the explicitly authorized field-only backfill")
    r = Runner(args.workdir)
    try:
        r.run()
    except Exception as exc:
        if isinstance(exc, IntegrityError) or "IntegrityError(" in str(exc):
            r.put("integrity_failure_events", int(r.state("integrity_failure_events", 0))+1)
            r.db.commit()
        r.status("paused_error", error=repr(exc), traceback=traceback.format_exc())
        log("paused_error", error=repr(exc))
        raise
    finally:
        r.pool.shutdown(wait=True)
        for client in r.clients.values():
            client.close()
        r.db.close()


if __name__ == "__main__":
    main()
