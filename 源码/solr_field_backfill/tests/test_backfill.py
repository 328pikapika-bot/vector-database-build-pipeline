from __future__ import annotations

import json
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from pymilvus import DataType

from solr_field_backfill.backfill import (
    FIELD_SPECS,
    ORIGINAL_REQUIRED_FIELDS,
    SHARDS,
    SOLR_MAPPING,
    TARGET_FIELDS,
    BackfillWorker,
    Checkpoint,
    Settings,
)


VALID_ID = "0" * 32


class FakeIterator:
    def __init__(self, rows):
        self.rows = list(rows)
        self.done = False

    def next(self):
        if self.done:
            return []
        self.done = True
        return self.rows

    def close(self):
        pass


class FakeMilvus:
    def __init__(self):
        self.fields = {
            "id": {"name": "id", "type": DataType.VARCHAR, "is_primary": True,
                    "params": {"max_length": 32}},
            "title_embed": {"name": "title_embed", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2560}},
            "abstract_embed": {"name": "abstract_embed", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2560}},
            "keyword_embed": {"name": "keyword_embed", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2560}},
        }
        self.added = []
        self.upserts = []
        self.iterator_rows = [{"id": VALID_ID, "title_embed": [0.1], "abstract_embed": [0.2], "keyword_embed": [0.3]}]

    def has_collection(self, name):
        return True

    def describe_collection(self, name):
        return {"fields": list(self.fields.values())}

    def get_collection_stats(self, name):
        return {"row_count": 1}

    def add_collection_field(self, **kwargs):
        self.added.append(kwargs)
        self.fields[kwargs["field_name"]] = {
            "name": kwargs["field_name"], "type": kwargs["data_type"],
            "params": {"max_length": kwargs["max_length"]}, "nullable": kwargs["nullable"],
        }

    def query_iterator(self, **kwargs):
        rows = self.iterator_rows
        if kwargs.get("filter"):
            prefix = kwargs["filter"].split("'")[1].rstrip("%")
            rows = [r for r in rows if r["id"].startswith(prefix)]
        if kwargs.get("limit", -1) > 0:
            rows = rows[:kwargs["limit"]]
        return FakeIterator([{f: row.get(f) for f in kwargs["output_fields"]} for row in rows])

    def load_collection(self, **kwargs):
        assert kwargs["refresh"] is True

    def get(self, **kwargs):
        return [{f: row.get(f) for f in kwargs["output_fields"]}
                for row in self.iterator_rows if row["id"] in kwargs["ids"]]

    def flush(self, **kwargs):
        pass

    def upsert(self, **kwargs):
        assert kwargs["partial_update"] is True
        self.upserts.append(kwargs)
        for update in kwargs["data"]:
            row = next(r for r in self.iterator_rows if r["id"] == update["id"])
            row.update(update)
        return {"upsert_count": len(kwargs["data"])}


def settings(tmp_path: Path, mode="check"):
    return Settings(
        mode=mode, state_dir=tmp_path / "state", log_dir=tmp_path / "logs",
        concurrency=1, batch_size=2, sample_size=0,
    )


def worker(tmp_path, mode="check"):
    fake = FakeMilvus()
    return BackfillWorker(settings(tmp_path, mode), milvus=fake), fake


def test_reference_mapping_reads_structured_fields_and_leaves_missing_sources_null(tmp_path):
    w, _ = worker(tmp_path)
    doc = {"id": VALID_ID, SOLR_MAPPING["hasSource"]: ["source"], SOLR_MAPPING["hasFile"]: ["fileType=pdf"],
           SOLR_MAPPING["volume"]: "7"}
    row = w.extract(doc)
    assert row["hasSource"] == "source"
    assert row["hasFile"] == "pdf"
    assert row["volume"] == "7"
    assert row["hasSource_str"] == '["source"]'
    assert row["hasFile_str"] == '["fileType=pdf"]'
    for name in ("hasAuthor_str", "hasCreateOrganization_str", "entityType"):
        assert row[name] is None
    assert set(row) == {"id", *TARGET_FIELDS}


def test_internal_solr_sessions_do_not_inherit_host_proxy_settings(tmp_path):
    w, _ = worker(tmp_path)
    try:
        assert w.http.trust_env is False
        assert w._http_session().trust_env is False
    finally:
        w.close()


def test_oversized_value_is_rejected(tmp_path):
    w, _ = worker(tmp_path)
    with pytest.raises(ValueError, match="exceeds max_length"):
        w.extract({"id": VALID_ID, SOLR_MAPPING["volume"]: "x" * (FIELD_SPECS["volume"] + 1)})


def test_preflight_check_does_not_mutate_schema(tmp_path):
    w, fake = worker(tmp_path)
    result = w.preflight(mutate_schema=False)
    assert len(fake.added) == 0
    assert set(result["missing_fields"]) == set(TARGET_FIELDS)


def test_preflight_run_adds_only_nullable_varchar_fields(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    result = w.preflight(mutate_schema=True)
    assert set(result["missing_fields"]) == set(TARGET_FIELDS)
    assert len(fake.added) == len(TARGET_FIELDS)
    assert all(item["nullable"] is True for item in fake.added)
    assert all(item["max_length"] == FIELD_SPECS[item["field_name"]] for item in fake.added)


def test_update_uses_partial_payload_only(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [])
    w._update_batch([VALID_ID])
    assert len(fake.upserts) == 1
    call = fake.upserts[0]
    assert call["partial_update"] is True
    assert set(call["data"][0]) == {"id", *TARGET_FIELDS}
    assert not set(ORIGINAL_REQUIRED_FIELDS) - {"id"} & set(call["data"][0])


def test_checkpoint_skips_completed_shard(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.db")
    cp.set("00", "completed", scanned=2, updated=2)
    assert "00" not in cp.pending()
    assert set(cp.pending()) == set(SHARDS) - {"00"}
    cp.close()


def test_solr_error_retries(tmp_path):
    class FailingHTTP:
        calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError("timeout")

    w, _ = worker(tmp_path)
    w.settings = Settings(mode="check", state_dir=tmp_path / "state", log_dir=tmp_path / "logs",
                          concurrency=1, batch_size=2, max_retries=2, sample_size=0)
    w.http = FailingHTTP()
    with pytest.raises(RuntimeError, match="after retries"):
        w._solr_batch([VALID_ID])
    assert w.http.calls == 3


def test_no_delete_api_is_used(tmp_path):
    w, fake = worker(tmp_path)
    assert not any(name.startswith("delete") or name.startswith("drop") for name in dir(fake))
    assert not hasattr(w.milvus, "drop_collection")


def test_check_mode_rejects_mutations_even_on_direct_call(tmp_path):
    w, fake = worker(tmp_path)
    with pytest.raises(RuntimeError, match="require"):
        w.preflight(mutate_schema=True)
    assert w._update_batch([VALID_ID])[2]
    with pytest.raises(RuntimeError, match="requires"):
        w.run()
    assert not fake.added and not fake.upserts


def test_invalid_late_schema_field_does_not_add_earlier_fields(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    fake.fields["hasCreateOrganization_str"] = {"name": "hasCreateOrganization_str", "type": DataType.INT64}
    with pytest.raises(RuntimeError, match="VARCHAR"):
        w.preflight(mutate_schema=True)
    assert not fake.added


def test_missing_record_does_not_upsert(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [])
    fake.iterator_rows = []
    assert "no longer exists" in w._update_batch([VALID_ID])[2]
    assert not fake.upserts


def test_unrequested_solr_id_does_not_upsert(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": {"docs": [{"id": "f" * 32}]}}
    class HTTP:
        def post(self, *args, **kwargs):
            return Response()
    w.http = HTTP()
    w.settings = replace(w.settings, max_retries=0)
    result = w._update_batch([VALID_ID])
    assert "unrequested" in result[2]
    assert not fake.upserts


def test_milvus_timeout_retries_partial_update(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w.settings = replace(w.settings, retry_backoff=0)
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [])
    original = fake.upsert
    calls = []
    def transient(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise TimeoutError("timeout")
        return original(**kwargs)
    fake.upsert = transient
    assert w._update_batch([VALID_ID]) == (1, [], None)
    assert len(calls) == 3 and all(call["partial_update"] for call in calls)


def sample_worker(tmp_path, count=7):
    w, fake = worker(tmp_path, mode="run")
    w.settings = replace(w.settings, max_records=count, concurrency=2)
    # Records span shards, exercising the global sample limit across shard boundaries.
    template = fake.iterator_rows[0]
    fake.iterator_rows = [{**copy.deepcopy(template), "id": f"{i % 3:02x}{i:030x}"} for i in range(40)]
    w._solr_batch = lambda ids: ([w.extract({"id": key, SOLR_MAPPING["volume"]: "5"}) for key in ids], [])
    w.check_solr = lambda: {"num_found": 40}
    def evidence():
        return {"schema": copy.deepcopy(fake.describe_collection("test")),
                "logical_row_count": len(fake.iterator_rows), "stats": {}, "index_definitions": {}}
    w._collection_evidence = evidence
    return w, fake


def test_real_sample_path_hard_cap_and_full_old_field_integrity(tmp_path):
    w, fake = sample_worker(tmp_path)
    original = copy.deepcopy(fake.iterator_rows)
    summary = w.run()
    assert summary["passed"] is True
    assert summary["updated"] == 7
    assert summary["historical_records_checked"] == 27
    assert sum(len(c["data"]) for c in fake.upserts) == 7
    for before, after in zip(original, fake.iterator_rows):
        assert all(before[k] == after[k] for k in before)
    with pytest.raises(RuntimeError, match="already used"):
        w.run()
    assert sum(len(c["data"]) for c in fake.upserts) == 7


def test_canary_corruption_stops_remaining_sample(tmp_path):
    w, fake = sample_worker(tmp_path)
    original_upsert = fake.upsert
    def corrupt(**kwargs):
        result = original_upsert(**kwargs)
        key = kwargs["data"][0]["id"]
        next(r for r in fake.iterator_rows if r["id"] == key)["title_embed"] = [999.0]
        return result
    fake.upsert = corrupt
    with pytest.raises(RuntimeError, match="canary"):
        w.run()
    assert sum(len(c["data"]) for c in fake.upserts) == 2


def test_sample_allowlist_blocks_outside_id(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w._sample_allowlist = frozenset([VALID_ID])
    with pytest.raises(RuntimeError, match="allowlist"):
        w._update_batch(["f" * 32])
    assert not fake.upserts


def test_has_file_without_explicit_type_stays_null(tmp_path):
    w, _ = worker(tmp_path)
    assert w.extract({"id": VALID_ID, SOLR_MAPPING["hasFile"]: "http://example.org/document.pdf"})["hasFile"] is None


def write_exclusion(w, tmp_path, old_ids, controls=(), **overrides):
    path = tmp_path / "previous_sample.json"
    path.write_text(json.dumps({"milvus_uri": w.settings.milvus_uri,
                               "database": w.settings.milvus_db_name,
                               "collection": w.settings.collection,
                               "ids": list(old_ids), "control_ids": list(controls), **overrides}), encoding="utf-8")
    w.settings = replace(w.settings, exclude_sample_manifest=str(path))
    return path


def test_excludes_old_sample_and_controls_and_populated_targets(tmp_path):
    w, fake = sample_worker(tmp_path)
    w.preflight(mutate_schema=True)
    old = [r["id"] for r in fake.iterator_rows[:2]]
    write_exclusion(w, tmp_path, old[:1], old[1:])
    fake.iterator_rows[2]["volume"] = ""  # even an empty string is protected
    fake.iterator_rows[3]["publisher"] = "existing"
    populated = {r["id"] for r in fake.iterator_rows[2:4]}
    summary = w.run()
    manifest = json.loads((w.settings.state_dir / "sample_ids.json").read_text())
    assert not set(manifest["ids"] + manifest["control_ids"]) & (set(old) | populated)
    assert summary["passed"] and summary["excluded_id_count"] == 2
    assert summary["selection_counts"]["already_populated"] == 2
    assert fake.iterator_rows[2]["volume"] == "" and fake.iterator_rows[3]["publisher"] == "existing"


def test_sample_start_prefix_skips_populated_earlier_shards(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w.preflight(mutate_schema=True)
    fake.iterator_rows = [
        {"id": VALID_ID, **{name: None for name in TARGET_FIELDS}},
        *({"id": f"ff{index:030x}", **{name: None for name in TARGET_FIELDS}}
          for index in range(21)),
    ]
    w.settings = replace(w.settings, max_records=1, sample_start_prefix="ff")
    ids, counts = w._select_sample(set())
    assert len(ids) == 21 and all(value.startswith("ff") for value in ids)
    assert counts["scanned"] == 21
    assert not fake.upserts


@pytest.mark.parametrize("overrides", [{"database": "other"}, {"collection": "other"},
                                       {"milvus_uri": "http://other:19530"}, {"ids": ["bad"]},
                                       {"ids": [VALID_ID, VALID_ID]}])
def test_invalid_exclusion_fails_before_write(tmp_path, overrides):
    w, fake = sample_worker(tmp_path)
    write_exclusion(w, tmp_path, [], **overrides)
    with pytest.raises((ValueError, RuntimeError)):
        w.run()
    assert not fake.upserts and not fake.added
    assert not json.loads(w.summary_path.read_text())["passed"]


def test_requires_all_twenty_controls(tmp_path):
    w, fake = sample_worker(tmp_path)
    fake.iterator_rows = fake.iterator_rows[:26]
    with pytest.raises(RuntimeError, match="need 27"):
        w.run()
    assert not fake.upserts and not fake.added


def test_duplicate_selection_id_rejected(tmp_path):
    w, fake = sample_worker(tmp_path)
    fake.iterator_rows.insert(0, copy.deepcopy(fake.iterator_rows[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        w.run()
    assert not fake.upserts and not fake.added


def test_canary_new_field_corruption_stops_and_saves_report(tmp_path):
    w, fake = sample_worker(tmp_path)
    original = fake.upsert
    def corrupt(**kwargs):
        response = original(**kwargs)
        key = kwargs["data"][0]["id"]
        next(r for r in fake.iterator_rows if r["id"] == key)["volume"] = "wrong"
        return response
    fake.upsert = corrupt
    with pytest.raises(RuntimeError, match="canary"):
        w.run()
    assert len(fake.upserts) == 1
    assert not json.loads(w.summary_path.read_text())["passed"]
    assert json.loads((w.settings.state_dir / "canary_audit.json").read_text())["new_field_mismatch_ids"]


def test_stop_after_baseline_prevents_all_mutations(tmp_path):
    w, fake = sample_worker(tmp_path)
    original = w._hash_records
    def stop(ids):
        result = original(ids)
        w.stop_event.set()
        return result
    w._hash_records = stop
    with pytest.raises(RuntimeError, match="stopped"):
        w.run()
    assert not fake.upserts and not fake.added


def test_logical_count_growth_is_not_a_pass(tmp_path):
    w, _ = sample_worker(tmp_path)
    original = w._collection_evidence
    calls = []
    def evidence():
        result = original()
        if calls:
            result["logical_row_count"] += 1
        calls.append(True)
        return result
    w._collection_evidence = evidence
    result = w.run()
    assert not result["passed"] and result["row_count_delta"] == 1


@pytest.mark.parametrize("batch_size,flush_every", [(500, 8), (1000, 4)])
def test_large_batch_canary_then_at_most_two_active_batches(tmp_path, batch_size, flush_every):
    count = 3 * batch_size
    w, fake = sample_worker(tmp_path, count=count)
    template = copy.deepcopy(fake.iterator_rows[0])
    fake.iterator_rows = [{**copy.deepcopy(template), "id": f"00{i:030x}"} for i in range(count + 20)]
    w.settings = replace(w.settings, batch_size=batch_size, flush_every_batches=flush_every)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0
    original = fake.upsert
    def concurrent(**kwargs):
        nonlocal active, peak
        if fake.upserts:  # canary has completed
            assert json.loads((w.settings.state_dir / "canary_audit.json").read_text())["passed"]
            with lock:
                active += 1
                peak = max(active, peak)
            barrier.wait(timeout=10)
            result = original(**kwargs)
            with lock:
                active -= 1
            return result
        return original(**kwargs)
    fake.upsert = concurrent
    result = w.run()
    assert result["passed"] and peak == 2
    assert [len(call["data"]) for call in fake.upserts] == [batch_size] * 3
    assert [m["requested"] for m in result["batches"]] == [batch_size] * 3
    assert result["pipeline_records"] == 2 * batch_size
    assert result["pipeline_seconds_including_final_flush"] >= result["pipeline_seconds"]


def test_thread_sessions_reused_and_closed_once(tmp_path, monkeypatch):
    sessions = []
    class HTTP:
        def __init__(self):
            self.closed = 0
            sessions.append(self)
        def close(self):
            self.closed += 1
    monkeypatch.setattr("solr_field_backfill.backfill.requests.Session", HTTP)
    w, _ = worker(tmp_path)
    barrier = threading.Barrier(2)
    def get_twice(_):
        first = w._http_session()
        barrier.wait(timeout=5)
        assert w._http_session() is first
        return first
    with ThreadPoolExecutor(max_workers=2) as pool:
        found = list(pool.map(get_twice, range(2)))
    assert found[0] is not found[1] and len(sessions) == 3
    w.close()
    assert all(s.closed == 1 for s in sessions)


def test_retry_after_applied_timeout_keeps_originals_and_measures_stages(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w._sample_allowlist = frozenset([VALID_ID])
    w.settings = replace(w.settings, retry_backoff=0)
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID, SOLR_MAPPING["volume"]: "8"})], [])
    original = fake.upsert
    calls = []
    def timeout_after_write(**kwargs):
        calls.append(True)
        result = original(**kwargs)
        if len(calls) == 1:
            raise TimeoutError("response lost after apply")
        return result
    fake.upsert = timeout_after_write
    before = copy.deepcopy(fake.iterator_rows[0])
    assert w._update_batch([VALID_ID]) == (1, [], None)
    assert all(fake.iterator_rows[0][key] == value for key, value in before.items())
    metrics = w.batch_metrics[0]
    assert metrics["upsert_attempts"] == 2
    assert metrics["strong_check_seconds"] > 0 and metrics["partial_upsert_seconds"] > 0
    assert metrics["retry_wait_seconds"] >= 0


def test_existing_target_changed_after_selection_rejected(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w._sample_allowlist = frozenset([VALID_ID])
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [])
    fake.iterator_rows[0]["volume"] = "protect"
    assert "target field changed" in w._update_batch([VALID_ID])[2]
    assert not fake.upserts and w.stop_event.is_set()


def test_missing_solr_document_stops_without_partial_batch(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    other = "f" * 32
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [other])
    assert w._update_batch([VALID_ID, other]) == (0, [other], None)
    assert not fake.upserts and w.stop_event.is_set()


def test_payload_whitelist_blocks_original_field(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    unsafe = {**w.extract({"id": VALID_ID}), "title_embed": [1]}
    w._solr_batch = lambda ids: ([unsafe], [])
    assert "unsafe payload" in w._update_batch([VALID_ID])[2]
    assert not fake.upserts


def test_exhausted_milvus_retries_stop_and_keep_failure_metrics(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w.settings = replace(w.settings, max_retries=1, retry_backoff=0)
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [])
    def fail(**kwargs):
        raise TimeoutError("failed")
    fake.upsert = fail
    assert w._update_batch([VALID_ID])[2] == "failed"
    assert w.stop_event.is_set()
    assert w.batch_metrics[0]["upsert_attempts"] == 2
    assert w.batch_metrics[0]["updated"] == 0 and w.batch_metrics[0]["error"] == "failed"


def test_stop_interrupts_retry_wait(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w.settings = replace(w.settings, retry_backoff=60)
    w._solr_batch = lambda ids: ([w.extract({"id": VALID_ID})], [])
    def fail(**kwargs):
        w.stop_event.set()
        raise TimeoutError("stopping")
    fake.upsert = fail
    assert "stopped during Milvus retry wait" in w._update_batch([VALID_ID])[2]
    assert w.batch_metrics[0]["upsert_attempts"] == 1


def test_periodic_and_final_flush_are_retained_and_timed(tmp_path):
    w, fake = sample_worker(tmp_path)
    w.settings = replace(w.settings, flush_every_batches=2)
    flushed = []
    fake.flush = lambda **kwargs: flushed.append(kwargs)
    result = w.run()
    assert len(flushed) == 3  # batch 2, batch 4, final
    assert [m["reason"] for m in result["flushes"]] == ["periodic_batch_2", "periodic_batch_4", "sample_final"]
    assert all(m["seconds"] >= 0 for m in result["flushes"])


def test_plain_handles_protobuf_style_sequence_without_iter():
    class Sequence:
        def __len__(self):
            return 2
        def __getitem__(self, key):
            return ["one", "two"][key]
    assert BackfillWorker._plain({"array": Sequence()}) == {"array": ["one", "two"]}


def test_stop_during_active_batch_prevents_new_submissions(tmp_path):
    w, fake = sample_worker(tmp_path, count=15)
    original = fake.upsert
    def stop_after_canary(**kwargs):
        result = original(**kwargs)
        if len(fake.upserts) >= 2:
            w.stop_event.set()
        return result
    fake.upsert = stop_after_canary
    result = w.run()
    assert not result["passed"]
    assert len(fake.upserts) <= 3  # canary + at most two already active
    assert sum(m["updated"] for m in result["batches"]) <= 6


def test_final_flush_failure_saves_report_without_replay(tmp_path):
    w, fake = sample_worker(tmp_path)
    def fail(**kwargs):
        raise TimeoutError("flush timeout")
    fake.flush = fail
    with pytest.raises(TimeoutError, match="flush timeout"):
        w.run()
    report = json.loads(w.summary_path.read_text())
    assert not report["passed"] and report["acknowledged_updated"] == 7
    assert report["flushes"][-1]["error"] == "flush timeout"


def test_solr_query_and_conversion_metrics_on_retry(tmp_path):
    w, fake = worker(tmp_path, mode="run")
    w.settings = replace(w.settings, retry_backoff=0)
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": {"docs": [{"id": VALID_ID, SOLR_MAPPING["volume"]: ["5"]}]}}
    class HTTP:
        calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient Solr timeout")
            return Response()
    w.http = HTTP()
    assert w._update_batch([VALID_ID]) == (1, [], None)
    metrics = w.batch_metrics[0]
    assert metrics["solr_attempts"] == 2 and metrics["upsert_attempts"] == 1
    assert metrics["solr_query_seconds"] > 0 and metrics["transform_seconds"] > 0


# Full-run and deployment protections (no real service writes).
def full_worker(tmp_path):
    w, fake = sample_worker(tmp_path)
    w.settings = replace(w.settings, max_records=0, sample_size=2)
    return w, fake


def test_full_run_preserves_history_and_preexisting_targets(tmp_path):
    w, fake = full_worker(tmp_path)
    fake.iterator_rows[0].update(volume="already-filled", publisher="")
    original = copy.deepcopy(fake.iterator_rows)
    result = w.run()
    assert result["passed"] and result["updated"] == len(original)
    assert result["shards"] == {"completed": 256}
    for before, after in zip(original, fake.iterator_rows):
        assert all(after[k] == v for k, v in before.items())
    for call in fake.upserts:
        assert call["partial_update"] is True
        assert all(set(row) == {"id", *TARGET_FIELDS} for row in call["data"])
    assert not w.batch_metrics and not w.preparation_metrics
    calls = len(fake.upserts)
    assert w.run()["passed"]
    assert len(fake.upserts) == calls  # completed shards are not replayed
    w.close()


def test_full_run_solr_missing_stops_following_shards(tmp_path):
    w, fake = full_worker(tmp_path)
    w._solr_batch = lambda ids: ([], ids)
    result = w.run()
    assert not result["passed"] and result["stopped"]
    assert result["shards"]["failed"] == 1
    assert result["shards"]["pending"] == 255
    assert not fake.upserts
    w.close()


def test_full_run_iterator_error_stops_without_advancing(tmp_path):
    w, fake = full_worker(tmp_path)
    w._query_ids = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("iterator unavailable"))
    result = w.run()
    assert not result["passed"] and result["shards"]["failed"] == 1
    assert not fake.upserts
    assert "iterator unavailable" in w.failures_path.read_text()
    w.close()


def test_full_shard_corruption_stops_following_shards(tmp_path):
    w, fake = full_worker(tmp_path)
    original_upsert = fake.upsert
    def corrupt(**kwargs):
        result = original_upsert(**kwargs)
        key = kwargs["data"][0]["id"]
        next(r for r in fake.iterator_rows if r["id"] == key)["title_embed"] = [999.0]
        return result
    fake.upsert = corrupt
    with pytest.raises(RuntimeError, match="changed"):
        w.run()
    result = json.loads(w.summary_path.read_text())
    assert result["passed"] is False and result["shards"]["pending"] == 255
    w.close()


def test_full_count_change_is_reported_as_failure(tmp_path):
    w, _ = full_worker(tmp_path)
    original = w._collection_evidence
    calls = 0
    def evidence():
        nonlocal calls
        calls += 1
        result = original()
        if calls > 1:
            result["logical_row_count"] += 1
        return result
    w._collection_evidence = evidence
    result = w.run()
    assert not result["passed"] and result["collection_checks"]["row_count_delta"] == 1
    w.close()


def test_full_resume_rechecks_baseline_before_writes(tmp_path):
    w, fake = full_worker(tmp_path)
    assert w.run()["passed"]
    count = len(fake.upserts)
    fake.iterator_rows[0]["title_embed"] = [999.0]
    with pytest.raises(RuntimeError, match="changed"):
        w.run()
    assert len(fake.upserts) == count
    w.close()


def test_checkpoint_target_binding_rejects_reuse(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.db")
    cp.bind_target({"collection_id": "123"})
    cp.set("00", "completed", updated=10)
    cp.bind_target({"collection_id": "123"})
    with pytest.raises(RuntimeError, match="different target"):
        cp.bind_target({"collection_id": "456"})
    assert "00" not in cp.pending()
    cp.close()


def test_unbound_old_checkpoint_is_not_silently_adopted(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.db")
    cp.set("00", "completed", updated=10)
    with pytest.raises(RuntimeError, match="unbound checkpoint"):
        cp.bind_target({"collection_id": "123"})
    cp.close()


def test_state_lock_excludes_other_worker_and_releases(tmp_path):
    from solr_field_backfill.backfill import state_lock
    with state_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another worker"):
            with state_lock(tmp_path):
                pytest.fail("second worker entered")
    with state_lock(tmp_path):
        pass


def test_full_run_reuses_pools_across_shards(tmp_path):
    w, _ = full_worker(tmp_path)
    pools = []
    original = w._process_shard
    def process(*args, **kwargs):
        result = original(*args, **kwargs)
        pools.append((id(w._reader_pool), id(w._writer_pool)))
        return result
    w._process_shard = process
    assert w.run()["passed"]
    assert len(pools) == 256 and len(set(pools)) == 1
    w.close()


def test_full_mode_refuses_bounded_sample_checkpoint(tmp_path):
    w, fake = full_worker(tmp_path)
    (w.settings.state_dir / "sample_ids.json").write_text("{}")
    with pytest.raises(RuntimeError, match="sample state"):
        w.run()
    assert not fake.upserts and not fake.added
    w.close()


def test_duplicate_full_shard_ids_are_rejected(tmp_path):
    w, fake = worker(tmp_path, "run")
    fake.iterator_rows *= 2
    with pytest.raises(RuntimeError, match="duplicate source IDs"):
        w._query_ids("00")
    assert not fake.upserts
    w.close()


def test_full_resume_after_interruption_preserves_already_filled_values(tmp_path):
    w, fake = full_worker(tmp_path)
    original = copy.deepcopy(fake.iterator_rows)
    original_upsert = fake.upsert
    def stop_after_commit(**kwargs):
        result = original_upsert(**kwargs)
        w.stop_event.set()
        return result
    fake.upsert = stop_after_commit
    first = w.run()
    assert not first["passed"] and first["shards"].get("pending", 0) > 0
    fake.upsert = original_upsert
    w.stop_event.clear()
    assert w.run()["passed"]
    for before, after in zip(original, fake.iterator_rows):
        assert all(after[k] == v for k, v in before.items())
    assert all(r["volume"] == "5" for r in fake.iterator_rows)
    w.close()


def test_full_flush_failure_never_marks_shard_completed(tmp_path):
    w, fake = full_worker(tmp_path)
    fake.flush = lambda **kw: (_ for _ in ()).throw(TimeoutError("flush unavailable"))
    with pytest.raises(TimeoutError, match="flush unavailable"):
        w.run()
    result = json.loads(w.summary_path.read_text())
    assert result["passed"] is False and not result["shards"].get("completed")
    w.close()
