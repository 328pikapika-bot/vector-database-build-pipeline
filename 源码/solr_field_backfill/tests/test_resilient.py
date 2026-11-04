from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import requests
from pymilvus import DataType
from pymilvus.exceptions import MilvusException

from solr_field_backfill.backfill import Checkpoint, Settings, TARGET_FIELDS, SOLR_MAPPING, state_lock
from solr_field_backfill import resilient as r
from solr_field_backfill.tests.test_backfill import FakeMilvus


def key(n):
    return f"{n:032x}"


class Database(FakeMilvus):
    def __init__(self, count=4):
        super().__init__()
        for n in range(20):
            self.fields[f"original_{n}"] = {"name": f"original_{n}", "type": DataType.VARCHAR}
        self.iterator_rows = [{"id": key(n), "title_embed": [0.1, float(n)],
                               "abstract_embed": [0.2], "keyword_embed": [0.3],
                               **{f"original_{i}": f"original-{n}-{i}" for i in range(20)},
                               **{name: None for name in TARGET_FIELDS}} for n in range(count)]
        self.timeout_before = self.timeout_after = self.flush_failures = 0

    def describe_collection(self, name):
        return {**super().describe_collection(name), "collection_id": 123}

    def query(self, **kwargs):
        return [{"count(*)": len(self.iterator_rows)}]

    def get_collection_stats(self, name):
        return {"row_count": len(self.iterator_rows)}

    def list_indexes(self, **kwargs):
        return []

    def upsert(self, **kwargs):
        if self.timeout_before:
            self.timeout_before -= 1
            raise TimeoutError("not applied")
        result = super().upsert(**kwargs)
        if self.timeout_after:
            self.timeout_after -= 1
            raise TimeoutError("applied without acknowledgement")
        return result

    def flush(self, **kwargs):
        if self.flush_failures:
            self.flush_failures -= 1
            raise TimeoutError("flush unavailable")


class HTTP:
    def __init__(self, docs):
        self.docs, self.failures, self.calls = docs, 0, 0

    def post(self, url, **kwargs):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise TimeoutError("Solr unavailable")
        ids = kwargs["data"]["q"].split("}", 1)[1].split(",")
        docs = [copy.deepcopy(d) for d in self.docs if d["id"] in ids]
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"responseHeader": {"status": 0}, "response": {"docs": docs}}
        return Response()

    def get(self, *args, **kwargs):
        return self.post(*args, data={"q": "{}"}, **kwargs)

    def close(self):
        pass


@pytest.fixture
def make(tmp_path):
    workers = []
    def factory(count=4, docs=None, folder="state"):
        settings = Settings(mode="run", state_dir=tmp_path / folder, log_dir=tmp_path / "logs",
                            batch_size=2, concurrency=2, sample_size=2, max_retries=1,
                            retry_backoff=0, recovery_initial=0.01, recovery_max=0.02,
                            flush_every_batches=8, mongo_uri="mongodb://unused")
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        status = r.Status(settings, threading.Event())
        cp = Checkpoint(settings.state_dir / "checkpoint.db")
        state = r.DurableState(cp, status)
        db = Database(count)
        docs = docs if docs is not None else [{"id": key(i), SOLR_MAPPING["volume"]: str(i + 1)} for i in range(count)]
        http = HTTP(docs)
        w = r.ResilientWorker(settings, status=status, state=state, checkpoint=cp, milvus=db, http=http)
        w.preflight(mutate_schema=True)
        state.set_meta("target", {"collection_id": 123})
        state.set_meta("job_id", "test")
        workers.append(w)
        return w, db, http
    yield factory
    for w in workers:
        w.close()


def plan(w, count=4):
    w.state.plan("00", [key(i) for i in range(count)], w.settings.batch_size)


def run_batch(w, number=0):
    ids = w.state.batch("00", number)["ids"]
    return w.write_batch("00", number, w._read(ids))


def test_mixed_missing_and_normal_preserves_history(make):
    w, db, _ = make(docs=[{"id": key(0), SOLR_MAPPING["volume"]: "7"}])
    before = copy.deepcopy(db.iterator_rows)
    plan(w)
    result = run_batch(w)
    assert result == {**result, "processed": 2, "updated": 1, "failed": 1, "unchanged": 0}
    assert w.state.summary()["processed"] == 0
    assert w.state.summary()["acknowledged_pending_flush"] == 2
    failures = w.state.unsent()
    assert [f["doc_id"] for f in failures] == [key(1)]
    assert failures[0]["stage"] == "solr_lookup"
    assert db.iterator_rows[1] == before[1]
    w.flush_confirm("00", "test")
    assert w.state.summary()["processed"] == 2
    for previous, current in zip(before, db.iterator_rows):
        assert all(previous[n] == current[n] for n in w.original_fields)
    assert all(set(p) == {"id", *TARGET_FIELDS} for call in db.upserts for p in call["data"])
    assert not w.stop_event.is_set() and not w.status.safety


def test_oversized_row_isolated_without_truncating(make):
    w, db, _ = make(docs=[{"id": key(0), SOLR_MAPPING["volume"]: "x" * 51},
                          {"id": key(1), SOLR_MAPPING["volume"]: "12"}])
    plan(w)
    result = run_batch(w)
    assert (result["updated"], result["failed"]) == (1, 1)
    assert db.iterator_rows[0]["volume"] is None
    assert "volume" in w.state.unsent()[0]["error"]


def test_existing_empty_string_and_nonempty_conflicts_are_preserved_and_recorded(make):
    w, db, _ = make()
    db.iterator_rows[0]["volume"] = ""
    db.iterator_rows[1]["volume"] = "old"
    plan(w)
    result = run_batch(w)
    assert result["failed"] == 2 and result["unchanged"] == 0 and result["updated"] == 0
    assert not db.upserts
    assert [row["volume"] for row in db.iterator_rows[:2]] == ["", "old"]
    assert {f["doc_id"] for f in w.state.unsent()} == {key(0), key(1)}
    assert {f["reason"] for f in w.state.unsent()} == {"existing_value_differs"}


def test_parser_conflict_does_not_block_other_records(make):
    w, db, _ = make()
    db.iterator_rows[0]["volume"] = "old"
    plan(w)
    result = run_batch(w)
    assert (result["processed"], result["updated"], result["unchanged"], result["failed"]) == (2, 1, 0, 1)
    assert db.iterator_rows[0]["volume"] == "old"
    assert db.iterator_rows[1]["volume"] == "2"
    assert [f["doc_id"] for f in w.state.unsent()] == [key(0)]
    assert all(set(row) == {"id", *TARGET_FIELDS} for call in db.upserts for row in call["data"])


def test_all_null_source_is_unchanged(make):
    w, db, _ = make(docs=[{"id": key(0)}, {"id": key(1)}])
    plan(w)
    assert run_batch(w)["unchanged"] == 2
    assert not db.upserts


@pytest.mark.parametrize("after", [True, False])
def test_uncertain_upsert_reconciles_before_retry_without_double_count(make, after):
    w, db, _ = make()
    plan(w)
    if after:
        db.timeout_after = 1
    else:
        db.timeout_before = 1
    run_batch(w)
    w.flush_confirm("00", "test")
    assert len(db.upserts) == 1
    assert w.state.summary()["updated"] == 2
    assert w.write_batch("00", 0, None)["updated"] == 2
    w.flush_confirm("00", "duplicate")
    assert w.state.summary()["updated"] == 2


def recovery_error():
    return MilvusException(
        code=106, message="failed to query: loaded collection do not found any channel in target, "
                          "may be in recovery: collection on recovering[collection=123]")


def test_only_explicit_milvus_collection_recovery_is_transient():
    assert r.transient(recovery_error())
    for error in (
        MilvusException(code=106, message="collection not found"),
        MilvusException(code=106, message="collection schema mismatch"),
        MilvusException(code=106, message="loaded collection do not found any channel in target"),
        MilvusException(code=100, message="collection on recovering"),
        RuntimeError("collection on recovering"),
        RuntimeError("original fields/vector changed for sample IDs: ['00000000000000000000000000000503']"),
        r.SafetyPause("historical record missing after timeout"),
        r.Stopping("timeout"),
    ):
        assert not r.transient(error)


def closed_channel_error():
    return MilvusException(code=1, message="Unexpected error, message=<Cannot invoke RPC on closed channel!>")


@pytest.mark.parametrize("after", [True, False])
def test_closed_channel_rebuilds_and_reconciles_uncertain_write(make, monkeypatch, after):
    w, db, _ = make()
    plan(w)
    before = copy.deepcopy(db.iterator_rows)
    original = db.upsert
    fresh = Database()
    fresh.iterator_rows = db.iterator_rows
    calls = factories = 0

    def upsert(**kwargs):
        nonlocal calls
        calls += 1
        if after:
            original(**kwargs)
        raise closed_channel_error()

    def factory(**kwargs):
        nonlocal factories
        factories += 1
        assert kwargs["dedicated"] is True
        return fresh

    db.upsert = upsert
    monkeypatch.setattr(r, "MilvusClient", factory)
    run_batch(w)
    w.flush_confirm("00", "fresh_connection")
    assert calls == factories == 1
    assert len(db.upserts) + len(fresh.upserts) == 1
    assert w.state.summary()["processed"] == w.state.summary()["updated"] == 2
    assert w.status.connection_rebuilds == 1 and not w.status.safety
    assert all(all(old[n] == new[n] for n in w.original_fields)
               for old, new in zip(before, fresh.iterator_rows))


def test_closed_channel_in_probe_rebuilds_instead_of_latching(make, monkeypatch):
    w, db, _ = make()
    w.recovery_probe_id = key(0)
    db.get = lambda **kwargs: (_ for _ in ()).throw(closed_channel_error())
    fresh = Database()
    monkeypatch.setattr(r, "MilvusClient", lambda **kwargs: fresh)
    w.milvus_backpressure.failed("milvus_flush")
    assert w.external("milvus_strong", lambda: "recovered") == "recovered"
    assert w.status.connection_rebuilds == 1
    assert not w.status.safety and not w.status.backpressure
    assert not db.upserts and not fresh.upserts


def test_two_writers_share_reconnect_and_drain_active_request(make, monkeypatch):
    w, db, _ = make()
    active, release = threading.Event(), threading.Event()
    closed, rebuilt = [], []
    db.close = lambda: closed.append(True)
    fresh = Database()
    def factory(**kwargs):
        assert release.is_set()
        rebuilt.append(True)
        return fresh
    monkeypatch.setattr(r, "MilvusClient", factory)
    def held_request():
        active.set()
        assert release.wait(2)
    with ThreadPoolExecutor(3) as pool:
        held = pool.submit(w.milvus_call, "partial_upsert", held_request)
        assert active.wait(1)
        with pytest.raises(MilvusException):
            w.milvus_call("flush", lambda: (_ for _ in ()).throw(closed_channel_error()))
        w.milvus_backpressure.failed("milvus_flush")
        pending = [pool.submit(w.external, "milvus_strong", lambda: "ok") for _ in range(2)]
        time.sleep(0.06)
        assert not rebuilt and not closed and not any(f.done() for f in pending)
        release.set()
        held.result(timeout=1)
        assert [f.result(timeout=2) for f in pending] == ["ok", "ok"]
    assert len(rebuilt) == len(closed) == 1


def test_reconnect_failure_waits_and_stop_interrupts_drain(make, monkeypatch):
    w, _, _ = make()
    w._connection_broken = True
    w._connection_users[123] = 1
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(w._repair_milvus)
        deadline = time.monotonic() + 1
        while not w._connection_resetting and time.monotonic() < deadline:
            time.sleep(0.005)
        w.stop_event.set()
        with pytest.raises(r.Stopping):
            pending.result(timeout=1)
    assert not w._connection_resetting and w._connection_broken


def test_reconnect_retries_outage_without_writes(make, monkeypatch):
    w, db, _ = make()
    w._connection_broken = True
    fresh = Database()
    attempts = 0
    def factory(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("dependency still restarting")
        return fresh
    monkeypatch.setattr(r, "MilvusClient", factory)
    w.milvus_backpressure.failed("milvus_flush")
    assert w.external("milvus_strong", lambda: "ok") == "ok"
    assert attempts == 3 and w.status.connection_rebuilds == 1
    assert not db.upserts and not fresh.upserts and not w.status.safety


def test_permanent_reconnect_error_still_latches(make, monkeypatch):
    w, db, _ = make()
    w._connection_broken = True
    def factory(**kwargs):
        raise MilvusException(code=1400, message="authentication failed")
    monkeypatch.setattr(r, "MilvusClient", factory)
    w.milvus_backpressure.failed("milvus_flush")
    with pytest.raises(r.SafetyPause, match="authentication failed"):
        w.guarded(lambda: w.external("milvus_strong", lambda: pytest.fail("must remain blocked")))
    assert w.status.safety and not db.upserts


def test_reconnected_client_does_not_bypass_missing_id_guard(make, monkeypatch):
    w, _, _ = make()
    w._connection_broken = True
    w.recovery_probe_id = key(0)
    fresh = Database(count=0)
    monkeypatch.setattr(r, "MilvusClient", lambda **kwargs: fresh)
    w.milvus_backpressure.failed("milvus_flush")
    with pytest.raises(r.SafetyPause, match="probe record missing"):
        w.guarded(lambda: w.external("milvus_strong", lambda: pytest.fail("must stay paused")))
    assert w.status.safety and not fresh.upserts


@pytest.mark.parametrize("fault", ["count", "identity", "history"])
def test_reconnect_revalidates_collection_and_history_before_admission(make, monkeypatch, fault):
    w, db, _ = make()
    w.state.set_meta("baseline", w._collection_evidence())
    r.atomic_json(w.settings.state_dir / "full_baseline.json",
                  {"original_hashes": w._hash_records([key(0)])})
    fresh = Database()
    if fault == "count":
        fresh.iterator_rows.pop()
    elif fault == "identity":
        original = fresh.describe_collection
        fresh.describe_collection = lambda name: {**original(name), "collection_id": 456}
    else:
        fresh.iterator_rows[0]["title_embed"] = [99]
    monkeypatch.setattr(r, "MilvusClient", lambda **kwargs: fresh)
    w._connection_broken = True
    w.milvus_backpressure.failed("milvus_flush")
    with pytest.raises(r.SafetyPause):
        w.guarded(lambda: w.external("milvus_strong", lambda: pytest.fail("must remain blocked")))
    assert w.status.safety and not fresh.upserts and w.status.backpressure


def test_status_exposes_reconciliation_without_claiming_processed_progress(make):
    w, _, _ = make()
    plan(w)
    w.refresh()
    w.reconcile_shard_ids("00")
    document = w.status.document()
    assert document["state"] == "reconciling_ids"
    assert document["operation"]["shard"] == "00"
    assert document["operation"]["elapsed_seconds"] >= 0
    assert document["processed"] == document["processed_per_second_recent"] == 0


def test_count_loss_during_restart_scan_is_caught_before_first_write(make):
    w, db, _ = make()
    baseline = w._collection_evidence()
    plan(w)
    w.state.set_meta("max_logical_count", 4)
    db.iterator_rows.pop()
    with ThreadPoolExecutor(2) as readers, ThreadPoolExecutor(2) as writers:
        with pytest.raises(r.SafetyPause, match="count=3; floor=4"):
            w.process_shard("00", baseline, readers, writers)
    assert not db.upserts
    evidence = json.loads((w.settings.state_dir / "safety_collection_evidence.json").read_text())
    assert evidence["failed_checks"] == ["logical_rows_non_decreasing"]
    assert w.state.get_meta("max_logical_count") == 4


@pytest.mark.parametrize("integrity_failure", [False, True])
def test_supervised_startup_retries_outages_but_latches_integrity(tmp_path, monkeypatch, integrity_failure):
    settings = Settings(mode="run", state_dir=tmp_path / "state", log_dir=tmp_path / "logs",
                        mongo_uri="mongodb://unused", max_retries=1, retry_backoff=0,
                        recovery_initial=0.01, recovery_max=0.02)
    monkeypatch.setattr(r, "MilvusClient", lambda **kwargs: Database())
    monkeypatch.setattr(r.MongoReporter, "loop", lambda self: None)
    ran, checked = [], []
    def run(self):
        ran.append(True)
        self.stop_event.set()
    monkeypatch.setattr(r.ResilientWorker, "run_resilient", run)
    def startup():
        checked.append(True)
        if integrity_failure:
            # End the supervisor's latched wait after confirming the fault.
            worker_status = next(t for t in threading.enumerate() if t.name == "heartbeat")
            worker_status._target.__self__.stop.set()
            raise r.SafetyPause("historical vector changed")
        if len(checked) == 1:
            raise TimeoutError("Solr temporarily unavailable")
    assert r.supervise(settings, startup_check=startup) == 0
    pause = settings.state_dir / "safety_pause.json"
    if integrity_failure:
        assert not ran and "historical vector changed" in pause.read_text()
    else:
        assert ran == [True] and len(checked) == 2 and not pause.exists()


@pytest.mark.parametrize("after", [True, False])
def test_collection_recovery_waits_for_strong_probe_and_reconciles_write(make, after):
    w, db, _ = make()
    plan(w)
    before = copy.deepcopy(db.iterator_rows)
    original_upsert, original_get = db.upsert, db.get
    calls = probes = 0

    def upsert(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if after:
                original_upsert(**kwargs)
            raise recovery_error()
        assert probes == 2  # No resubmission until a Strong probe succeeds.
        return original_upsert(**kwargs)

    def get(**kwargs):
        nonlocal probes
        if kwargs["output_fields"] == ["id"]:
            assert kwargs["consistency_level"] == "Strong"
            probes += 1
            if probes == 1:
                raise recovery_error()
        return original_get(**kwargs)

    db.upsert, db.get = upsert, get
    run_batch(w)
    w.flush_confirm("00", "recovery")
    assert probes == 2 and len(db.upserts) == 1
    assert calls == (1 if after else 2)
    assert w.state.summary()["processed"] == w.state.summary()["updated"] == 2
    run_batch(w)
    w.flush_confirm("00", "duplicate")
    assert w.state.summary()["processed"] == 2 and len(db.upserts) == 1
    assert not w.status.safety and not w.status.backpressure
    for old, current in zip(before, db.iterator_rows):
        assert all(old[name] == current[name] for name in w.original_fields)
    assert all(call["partial_update"] for call in db.upserts)
    assert all(set(row) == {"id", *TARGET_FIELDS} for call in db.upserts for row in call["data"])


@pytest.mark.parametrize("phase", ["inflight", "acked", "done"])
def test_restart_after_write_before_checkpoint_is_idempotent(make, phase):
    w, db, _ = make()
    plan(w)
    rows = w._read([key(0), key(1)]).rows
    result = {"processed": 2, "updated": 2, "unchanged": 0, "failed": 0}
    w.state.prepare("00", 0, rows, result, [])
    db.upsert(data=rows, partial_update=True)
    if phase in {"acked", "done"}:
        w.state.acknowledge("00", 0)
    if phase == "done":
        w.state.confirm_flush("00")
    # Reload the SQLite state, keeping the remote collection contents.
    w.checkpoint.close()
    w.checkpoint = Checkpoint(w.settings.state_dir / "checkpoint.db")
    state = r.DurableState(w.checkpoint, w.status)
    w.state = state
    w.write_batch("00", 0, None)
    w.flush_confirm("00", "recovered")
    assert state.summary()["updated"] == 2 and len(db.upserts) == 1
    assert state.batch("00", 0)["phase"] == "done"
    assert [n for n, _ in state.pending("00")] == [1]


def test_flush_outage_keeps_unconfirmed_batches_and_recovers(make):
    w, db, _ = make()
    plan(w)
    run_batch(w)
    db.flush_failures = 3
    w.flush_confirm("00", "recover")
    assert w.state.summary()["processed"] == 2
    assert not w.status.waiters


def test_solr_outage_waits_without_registering_missing_ids(make):
    w, _, http = make()
    http.failures = 4
    prepared = w._read([key(0), key(1)])
    assert len(prepared.rows) == 2 and not prepared.failures
    assert not w.state.unsent() and http.calls == 5


def test_missing_record_cannot_be_recreated(make):
    w, db, _ = make()
    plan(w)
    prepared = w._read([key(0), key(1)])
    db.iterator_rows.pop(0)
    with pytest.raises(r.SafetyPause, match="no longer exists"):
        w.write_batch("00", 0, prepared)
    assert not db.upserts
    assert json.loads((w.settings.state_dir / "safety_pause.json").read_text())["reason"]


def test_foreign_or_historical_field_payload_pauses(make):
    w, db, _ = make()
    plan(w)
    prepared = w._read([key(0), key(1)])
    prepared.rows[0]["title_embed"] = [99]
    with pytest.raises(r.SafetyPause):
        w.write_batch("00", 0, prepared)
    assert not db.upserts and w.status.document()["state"] == "safety_paused"


def test_fixed_manifest_rejects_different_ids(make):
    w, db, _ = make()
    plan(w)
    with pytest.raises(r.SafetyPause):
        w.write_batch("00", 0, w._read([key(2), key(3)]))
    assert not db.upserts


def test_mongo_offline_outbox_and_repeated_delivery_are_idempotent(make):
    w, _, _ = make(docs=[])
    plan(w)
    run_batch(w)
    class Collection:
        offline = True
        docs = {}
        def bulk_write(self, writes, ordered):
            if self.offline:
                raise TimeoutError("offline")
            for op in writes:
                self.docs[op._filter["_id"]] = op._doc["$set"]
    collection = Collection()
    reporter = r.MongoReporter(w.state, w.settings, w.status, collection=collection)
    with pytest.raises(TimeoutError):
        reporter.sync_once()
    assert len(w.state.unsent()) == 2
    collection.offline = False
    original = w.state.mark_sent
    w.state.mark_sent = lambda ids: (_ for _ in ()).throw(RuntimeError("crash after Mongo ack"))
    with pytest.raises(RuntimeError):
        reporter.sync_once()
    w.state.mark_sent = original
    assert reporter.sync_once() == 2 and reporter.sync_once() == 0
    assert len(collection.docs) == 2
    assert all("vector" not in json.dumps(d) for d in collection.docs.values())
    assert w.settings.mongo_failure_collection == "solr_field_backfill_failures"


def test_failure_and_batch_are_atomic_on_storage_error(make):
    w, _, _ = make(docs=[])
    plan(w)
    w.state.db.execute("CREATE TRIGGER fail_batch BEFORE UPDATE ON bf_batches BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError):
        w.state.prepare("00", 0, [], {"processed": 2, "updated": 0, "unchanged": 0, "failed": 2},
                        [{"id": key(0), "stage": "solr_lookup", "reason": "source_missing"}])
    assert not w.state.unsent() and w.state.batch("00", 0)["phase"] == "pending"


def test_legacy_migration_keeps_47000_as_history_and_replans_unfinished(tmp_path):
    settings = Settings(state_dir=tmp_path, log_dir=tmp_path)
    cp = Checkpoint(tmp_path / "checkpoint.db")
    cp.set("00", "failed", scanned=47000, updated=47000)
    cp.set("01", "completed", scanned=10, updated=8, failed=1)
    status = r.Status(settings, threading.Event())
    state = r.DurableState(cp, status)
    assert state.get_meta("legacy_progress")["updated"] == 47008
    assert state.shard("00") == (0, 0, "pending")
    assert state.shard("01") == (1, 10, "completed")
    assert state.summary()["unchanged"] == 1
    # Idempotent migration does not duplicate legacy counts.
    assert r.DurableState(cp, status).summary()["processed"] == 10
    cp.close()


def test_migration_rolls_back_version_marker_when_interrupted(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.db")
    status = r.Status(Settings(state_dir=tmp_path), threading.Event())
    # First create the v2 tables then simulate a pre-version migration DB.
    state = r.DurableState(cp, status)
    with cp.db:
        cp.db.execute("DELETE FROM bf_meta")
        cp.db.execute("DELETE FROM bf_shards")
        cp.db.execute("CREATE TRIGGER fail_migration BEFORE INSERT ON bf_shards BEGIN SELECT RAISE(ABORT,'interrupted'); END")
    with pytest.raises(sqlite3.IntegrityError):
        r.DurableState(cp, status)
    assert state.get_meta("schema_version") is None
    assert state.get_meta("legacy_progress") is None
    cp.close()


def test_stop_interrupts_dependency_wait(make):
    w, _, _ = make()
    w.settings = replace(w.settings, recovery_initial=30, recovery_max=300)
    def action():
        raise TimeoutError()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(w.external, "milvus", action)
        deadline = time.monotonic() + 2
        while not w.status.waiters and time.monotonic() < deadline:
            time.sleep(0.005)
        assert w.status.waiters
        assert w.status.document()["state"] == "waiting_dependencies"
        w.stop_event.set()
        with pytest.raises(r.Stopping):
            future.result(timeout=1)


def test_backpressure_uses_one_readonly_probe_for_two_writers(make):
    w, db, _ = make()
    w.recovery_probe_id = key(0)
    original_get = db.get
    lock = threading.Lock()
    concurrent = maximum = calls = 0
    def get(**kwargs):
        nonlocal concurrent, maximum, calls
        assert kwargs["consistency_level"] == "Strong"
        assert kwargs["output_fields"] == ["id"]
        assert kwargs["ids"] == [key(0)]
        with lock:
            calls += 1
            concurrent += 1
            maximum = max(maximum, concurrent)
        try:
            time.sleep(0.02)
            if calls < 3:
                raise TimeoutError("time tick blocked")
            return original_get(**kwargs)
        finally:
            with lock:
                concurrent -= 1
    db.get = get
    w.milvus_backpressure.failed("milvus_partial_upsert")
    actions = []
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(w.external, "milvus_strong", lambda: actions.append(1)) for _ in range(2)]
        for future in futures:
            future.result(timeout=2)
    assert len(actions) == 2 and maximum == 1 and calls == 3
    assert not db.upserts and not w.status.backpressure and not w.status.waiters


def test_strong_probe_cannot_bypass_missing_history(make):
    w, db, _ = make()
    w.recovery_probe_id = key(0)
    db.iterator_rows.pop(0)
    w.milvus_backpressure.failed("milvus_strong")
    with pytest.raises(r.SafetyPause, match="probe record missing"):
        w.guarded(lambda: w.external("milvus_strong", lambda: pytest.fail("must remain blocked")))
    assert w.status.safety and not db.upserts


def test_dependency_failure_blocks_other_milvus_actions_until_strong_recovers(make):
    w, db, _ = make()
    plan(w)
    db.timeout_after = 1
    run_batch(w)
    w.flush_confirm("00", "reconcile")
    assert w.state.summary()["updated"] == 2 and len(db.upserts) == 1
    assert not w.status.backpressure


def test_live_status_does_not_keep_old_speed_when_work_is_blocked(make, monkeypatch):
    w, _, _ = make()
    now = [1000.0]
    monkeypatch.setattr(r.time, "time", lambda: now[0])
    w.status.started = 900
    w.status.session_processed_start = 10
    w.status.snapshot = {"processed": 110, "elapsed_seconds": 1,
                         "processed_per_second_since_start": 999}
    w.status.document()
    now[0] += 120
    result = w.status.document()
    assert result["elapsed_seconds"] == 220
    assert result["processed_per_second_since_start"] == pytest.approx(100 / 220)
    assert result["processed_per_second_recent"] == 0


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_write_rate_rejects_invalid_config(rate):
    with pytest.raises(ValueError, match="BACKFILL_MAX_WRITE_ROWS_PER_SEC"):
        Settings(max_write_rows_per_second=rate).validate()


def test_shared_pacer_spaces_concurrent_batches_without_catchup(make):
    w, _, _ = make()
    w.status.settings = replace(w.settings, max_write_rows_per_second=200)
    pacer = r.WritePacer(w.status)
    def admitted():
        pacer.ready(10)
        return time.monotonic()
    with ThreadPoolExecutor(2) as pool:
        times = sorted(pool.map(lambda _: admitted(), range(3)))
    assert all(b - a >= 0.045 for a, b in zip(times, times[1:]))
    time.sleep(0.08)
    first = admitted()
    second = admitted()
    assert second - first >= 0.045
    assert w.status.document()["max_write_rows_per_second"] == 200


def test_pacer_wait_is_interruptible_and_cannot_bypass_safety_pause(make):
    w, _, _ = make()
    w.status.settings = replace(w.settings, max_write_rows_per_second=1)
    pacer = r.WritePacer(w.status)
    pacer.ready(100)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(pacer.ready, 100)
        w.status.stop.set()
        with pytest.raises(r.Stopping):
            pending.result(timeout=0.5)
    w.status.stop.clear()
    w.status.safety = "historical mismatch"
    with pytest.raises(r.SafetyPause):
        r.WritePacer(w.status).ready(1)


def test_write_pacing_keeps_strong_checks_and_partial_payload(make):
    w, db, _ = make()
    w.status.settings = replace(w.settings, max_write_rows_per_second=200)
    history = w._hash_records([key(i) for i in range(4)])
    calls = []
    original = db.upsert
    def upsert(**kwargs):
        calls.append(time.monotonic())
        return original(**kwargs)
    db.upsert = upsert
    plan(w)
    run_batch(w, 0)
    run_batch(w, 1)
    w.flush_confirm("00", "paced")
    assert len(calls) == 2 and calls[1] - calls[0] >= 0.008
    assert w.state.summary()["updated"] == 4
    assert w._hash_records(list(history)) == history
    assert all(c["partial_update"] for c in db.upserts)
    assert all(set(row) == {"id", *TARGET_FIELDS} for c in db.upserts for row in c["data"])


def test_slow_successful_request_applies_backpressure_without_resubmitting(make):
    w, db, _ = make()
    w.settings = replace(w.settings, milvus_slow_request_seconds=0.001)
    w.recovery_probe_id = key(0)
    # Windows monotonic clocks may have a 15.6ms tick; exceed it so the real
    # request duration cannot round to zero and hide the slow-success path.
    assert w.milvus_call("strong_read", lambda: time.sleep(0.05)) is None
    assert w.status.backpressure["failed_stage"] == "slow_strong_read"
    assert w.status.document()["last_milvus_rpc"]["success"]
    w.external("milvus_strong", lambda: "ok")
    assert not w.status.backpressure and not db.upserts


def test_storage_wait_is_not_cleared_by_heartbeat(make):
    w, _, _ = make()
    restored = threading.Event()
    def save():
        if not restored.is_set():
            raise sqlite3.OperationalError("disk full")
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(w.status.storage, save)
        deadline = time.monotonic() + 2
        while not w.status.storage_waiters and time.monotonic() < deadline:
            time.sleep(0.005)
        w.status.publish()
        assert w.status.document()["state"] == "waiting_storage"
        restored.set()
        future.result(timeout=1)
    assert w.status.document()["state"] != "waiting_storage"


def test_duplicate_process_is_rejected(tmp_path):
    with state_lock(tmp_path):
        with pytest.raises((OSError, RuntimeError)):
            with state_lock(tmp_path):
                pytest.fail("second process acquired lock")


def test_canary_keeps_expected_and_full_vectors_across_resume(make):
    w, db, _ = make()
    plan(w)
    w.state.set_meta("baseline", w._collection_evidence())
    w.canary("00", 0, [key(0), key(1)])
    expected = json.loads((w.settings.state_dir / "resilient_canary.json").read_text())["expected"]
    assert len(expected) == 2
    assert len(json.loads((w.settings.state_dir / "resilient_canary.json").read_text())["hashes"][key(0)]) == 24
    w.canary("00", 0, [key(0), key(1)])
    assert len(db.upserts) == 1
    db.iterator_rows[0]["volume"] = "corrupt"
    with pytest.raises(r.SafetyPause, match="target values differ"):
        w.canary("00", 0, [key(0), key(1)])


def test_canary_records_old_parser_conflict_without_changing_the_row(make):
    w, db, _ = make()
    db.iterator_rows[0]["volume"] = "old"
    plan(w)
    w.state.set_meta("baseline", w._collection_evidence())
    original = copy.deepcopy(db.iterator_rows[0])
    w.canary("00", 0, [key(0), key(1)])
    result = w.state.batch("00", 0)["result"]
    assert (result["updated"], result["failed"]) == (1, 1)
    assert db.iterator_rows[0] == original
    assert w.state.unsent()[0]["reason"] == "existing_value_differs"


def test_pipeline_is_bounded_and_history_unchanged(make):
    w, db, _ = make(count=20)
    before = w._hash_records([key(i) for i in range(20)])
    plan(w, 20)
    lock = threading.Lock()
    counts = {"read": 0, "write": 0, "max_read": 0, "max_write": 0}
    read, write = w._read, w.write_batch
    def measured(name, fn, *args):
        with lock:
            counts[name] += 1
            counts[f"max_{name}"] = max(counts[f"max_{name}"], counts[name])
        try:
            time.sleep(0.01)
            return fn(*args)
        finally:
            with lock:
                counts[name] -= 1
    w._read = lambda ids: measured("read", read, ids)
    w.write_batch = lambda *args: measured("write", write, *args)
    with ThreadPoolExecutor(2) as readers, ThreadPoolExecutor(2) as writers:
        w.window("00", w.state.pending("00"), readers, writers)
    w.flush_confirm("00", "test")
    assert counts["max_read"] == counts["max_write"] == 2
    assert w.state.summary()["processed"] == 20
    assert w._hash_records(list(before)) == before


def test_full_scan_reports_failures_and_stays_distinct_from_completion(make):
    w, db, _ = make(docs=[{"id": key(0), SOLR_MAPPING["volume"]: "3"}])
    w.check_solr = lambda: {}
    w.state.set_meta("target", None)
    result = w.run_resilient()
    assert result["scan_complete"] and not result["all_records_enriched"]
    assert result["processed"] == result["scanned"] == 4
    assert result["updated"] == 1 and result["failed"] == 3
    assert w.status.stage == "completed_with_failures"
    assert result["historical_sample_check_passed"]


def test_completed_shard_appends_new_ids_without_replaying_old_batches(make):
    w, db, http = make()
    baseline = w._collection_evidence()
    original_hashes = w._hash_records([key(i) for i in range(4)])
    plan(w)
    for number in range(2):
        run_batch(w, number)
    w.flush_confirm("00", "test")
    w.state.finish_shard("00")
    before = w.state.summary()
    new_row = copy.deepcopy(db.iterator_rows[0])
    new_row["id"] = key(4)
    new_row.update({name: None for name in TARGET_FIELDS})
    db.iterator_rows.append(new_row)
    http.docs.append({"id": key(4), SOLR_MAPPING["volume"]: "5"})

    assert w.evidence(baseline)["passed"]
    assert w.state.get_meta("max_logical_count") == 5
    assert w.reconcile_shard_ids("00") == 1
    assert w.reconcile_shard_ids("00") == 0
    assert w.state.shard("00") == (1, 5, "running")
    assert [number for number, _ in w.state.pending("00")] == [2]
    run_batch(w, 2)
    w.flush_confirm("00", "new_id")
    w.state.finish_shard("00")
    after = w.state.summary()
    assert after["processed"] == before["processed"] + 1
    assert after["updated"] == before["updated"] + 1
    assert w._hash_records([key(i) for i in range(4)]) == original_hashes
    assert all(call["partial_update"] and all(set(row) == {"id", *TARGET_FIELDS}
                                                for row in call["data"]) for call in db.upserts)


def test_known_id_disappearance_pauses_even_if_record_count_is_unchanged(make):
    w, db, _ = make()
    plan(w)
    db.iterator_rows[0]["id"] = key(9)
    with pytest.raises(r.SafetyPause, match="previously planned IDs disappeared"):
        w.reconcile_shard_ids("00")
    assert w.state.shard("00")[1] == 4


def test_logical_record_count_drop_is_rejected_after_growth(make):
    w, db, _ = make()
    baseline = w._collection_evidence()
    db.iterator_rows.append({**copy.deepcopy(db.iterator_rows[0]), "id": key(4)})
    assert w.evidence(baseline)["passed"]
    db.iterator_rows.pop()
    with pytest.raises(r.SafetyPause, match="logical record count decreased"):
        w.evidence(baseline)


def test_final_verification_finds_id_added_after_its_shard_was_processed(make):
    w, db, http = make()
    w.check_solr = lambda: {}
    w.state.set_meta("target", None)
    original_hashes = w._hash_records([key(i) for i in range(4)])
    original_reconcile = w.reconcile_shard_ids
    injected = False

    def reconcile(prefix):
        nonlocal injected
        if prefix == "01" and not injected:
            new_row = copy.deepcopy(db.iterator_rows[0])
            new_row["id"] = key(4)
            new_row.update({name: None for name in TARGET_FIELDS})
            db.iterator_rows.append(new_row)
            http.docs.append({"id": key(4), SOLR_MAPPING["volume"]: "5"})
            injected = True
        return original_reconcile(prefix)

    w.reconcile_shard_ids = reconcile
    result = w.run_resilient()
    assert injected
    assert result["scan_complete"] and result["processed"] == result["scanned"] == 5
    assert result["collection_checks"]["logical_rows_after"] == 5
    assert w._hash_records([key(i) for i in range(4)]) == original_hashes
    upserts_before_restart = len(db.upserts)
    assert w.run_resilient()["processed"] == 5
    assert len(db.upserts) == upserts_before_restart


def test_permanent_http_error_is_not_retried_forever():
    response = requests.Response()
    response.status_code = 401
    assert not r.transient(requests.HTTPError(response=response))
    response.status_code = 503
    assert r.transient(requests.HTTPError(response=response))


def test_stop_mid_id_scan_cannot_persist_incomplete_manifest(make):
    w, db, _ = make()
    original = db.query_iterator
    def iterator(**kwargs):
        result = original(**kwargs)
        next_batch = result.next
        def stopped_next():
            batch = next_batch()
            w.stop_event.set()
            return batch
        result.next = stopped_next
        return result
    db.query_iterator = iterator
    with pytest.raises(r.Stopping):
        w._query_ids("00")
    assert w.state.shard("00")[0] == 0


def test_vector_corruption_triggers_safety_audit(make):
    w, db, _ = make()
    baseline = w._hash_records([key(0)])
    db.iterator_rows[0]["title_embed"][0] = 99
    with pytest.raises(r.SafetyPause):
        w.guarded(lambda: w.audit(baseline))
    assert w.status.safety and (w.settings.state_dir / "safety_pause.json").exists()


def test_sticky_safety_pause_survives_restart_without_connecting(tmp_path, monkeypatch):
    settings = Settings(mode="run", state_dir=tmp_path, log_dir=tmp_path)
    r.atomic_json(tmp_path / "safety_pause.json", {"reason": "historical mismatch"})
    callbacks = {}
    monkeypatch.setattr(r.signal, "signal", lambda sig, fn: callbacks.update({sig: fn}))
    monkeypatch.setattr(r, "MilvusClient", lambda **kw: pytest.fail("must not connect through safety latch"))
    thread = threading.Thread(target=r.supervise, args=(settings,))
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if (tmp_path / "heartbeat.json").exists():
                # Force a fresh heartbeat; normal interval is 15 seconds.
                if (tmp_path / "safety_pause.json").exists() and callbacks:
                    break
            time.sleep(0.01)
        assert thread.is_alive()
        assert json.loads((tmp_path / "safety_pause.json").read_text())["reason"] == "historical mismatch"
    finally:
        callbacks[r.signal.SIGTERM](None, None)
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_initial_dependency_outage_recovers_and_completed_worker_stays_alive(tmp_path, monkeypatch):
    settings = Settings(mode="run", state_dir=tmp_path, log_dir=tmp_path, mongo_uri="mongodb://unused",
                        recovery_initial=0.01, recovery_max=0.02)
    callbacks, connected, completed = {}, [], threading.Event()
    monkeypatch.setattr(r.signal, "signal", lambda sig, fn: callbacks.update({sig: fn}))
    def connect(**kwargs):
        connected.append(1)
        if len(connected) < 3:
            raise TimeoutError("network unavailable")
        return object()
    monkeypatch.setattr(r, "MilvusClient", connect)
    class Worker:
        def __init__(self, *a, **kw):
            self.cp = kw["checkpoint"]
        def run_resilient(self):
            completed.set()
        def refresh(self):
            pass
        def close(self):
            self.cp.close()
    monkeypatch.setattr(r, "ResilientWorker", Worker)
    thread = threading.Thread(target=r.supervise, args=(settings,))
    thread.start()
    try:
        assert completed.wait(3)
        assert len(connected) == 3 and thread.is_alive()
        assert not (tmp_path / "safety_pause.json").exists()
    finally:
        callbacks[r.signal.SIGTERM](None, None)
        thread.join(timeout=3)
    assert not thread.is_alive()
