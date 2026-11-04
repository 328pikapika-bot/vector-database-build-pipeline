from __future__ import annotations

from pathlib import Path

import pytest

from incremental_sync.adapters import MilvusTarget, SafetyPause, TemporaryDependencyError, check_schema
from incremental_sync.config import Settings
from incremental_sync.contract import (
    FIELDS, PermanentDataError, SOLR_FIELDS, new_record, source_values,
    update_payload, vector_inputs,
)
from incremental_sync.store import Ledger, SingleInstanceLock
from incremental_sync.failure_reporter import FailureReporter
from incremental_sync.worker import IncrementalWorker, parse_message, target_identity


DOC_ID = "0000019bdeaccd691ec2a327efbbc732"
VECTOR = [1e-6, *([0.0] * 2559)]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        mode="run", kafka_brokers="kafka:9092", kafka_topic="test_incremental",
        kafka_group="test_group", kafka_security_protocol="PLAINTEXT", kafka_sasl_mechanism="PLAIN",
        kafka_username="", kafka_password="", solr_url="http://solr/solr/paper",
        embedding_url="http://embedding/v1/embeddings", embedding_model="qwen3-embed",
        embedding_key="", milvus_uri="http://milvus:19530", milvus_db="default",
        milvus_collection="scientific_documents_v2", milvus_token="",
        state_dir=tmp_path / "state", log_dir=tmp_path / "logs", request_timeout=1,
        retry_wait_sec=0.01, idle_window_sec=2, vector_update_verified=True,
    )


def test_exact_41_field_contract_and_source_mapping() -> None:
    assert len(FIELDS) == 41
    document = {
        "id": DOC_ID, "type": "article", "auto_stringITMS_title": ["en|New title"],
        "auto_stringITMS_abstract": ["New abstract"], "auto_stringITMS_keyword": ["key"],
        "stringIMS_entityType": ["Article", "Other"],
        "stringOM_hasFile": ["id≡file-a∷fileType≡fileType≡pdf∷url≡/a.pdf"],
        "stringOM_hasSource": ["source-a∷title≡Journal"],
        "stringOM_hasFullTextFile": ["id≡file-a∷name≡A.mmd∷url≡/A.mmd"],
    }
    values = source_values(document, DOC_ID)
    assert values["title"] == ["New title"]
    assert values["entityType"] == "Article"
    assert values["hasFile"] == "pdf"
    assert values["hasSource"] == "Journal"
    assert values["hasFullTextFileid"] == ["file-a"]
    assert values["hasFile_str"] == '["id≡file-a∷fileType≡fileType≡pdf∷url≡/a.pdf"]'
    assert values["hasSource_str"] == '["source-a∷title≡Journal"]'
    assert vector_inputs(values) == {"title": "New title", "abstract": "New abstract", "keyword": "key"}
    assert "entityType" not in source_values({"id": DOC_ID, "stringIMS_entityType": ["", "Other"]}, DOC_ID)


def test_batch_upgrade_preserves_completed_offsets_and_accepts_new_id_event(tmp_path: Path) -> None:
    from dataclasses import replace

    old = make_settings(tmp_path)
    ledger = Ledger(old.state_dir, target_identity(old))
    ledger.enqueue(old.kafka_topic, 0, 5766, DOC_ID)
    event = ledger.next_event()
    ledger.begin(event)
    ledger.finish(event, "updated")
    ledger.conn.close()

    new = replace(old, batch_size=1000, batch_wait_sec=0.25)
    assert target_identity(new) == target_identity(old)
    resumed = Ledger(new.state_dir, target_identity(new))
    try:
        assert not resumed.enqueue(new.kafka_topic, 0, 5766, DOC_ID)
        assert resumed.next_events(new.batch_size) == []
        assert resumed.counts() == {"updated": 1}
        assert resumed.enqueue(new.kafka_topic, 0, 5767, DOC_ID)
        assert [e["offset_id"] for e in resumed.next_events(new.batch_size)] == [5767]
    finally:
        resumed.conn.close()


def test_heartbeat_publishes_batch_timings(tmp_path: Path) -> None:
    import json

    worker = IncrementalWorker(make_settings(tmp_path))
    worker.latest_batch_metrics = {"events": 20, "rows_per_second": 15.0,
                                   "stages_seconds": {"solr": 0.2}}

    class OneHeartbeat:
        stopped = False
        def is_set(self): return self.stopped
        def wait(self, seconds): self.stopped = True

    worker.stop_event = OneHeartbeat()
    worker.heartbeat()
    report = json.loads((worker.settings.state_dir / "progress.json").read_text())
    assert report["latest_batch"] == worker.latest_batch_metrics


def test_v4_rules_for_all_17_added_fields() -> None:
    document = {
        "id": DOC_ID,
        "stringIMS_volume": ["en|807", "ignore"],
        "stringIMS_issue": ["en|1", "ignore"],
        "stringIS_startPage": "1",
        "stringIS_endPage": "3",
        "stringIS_pageCount": "24",
        "dateIS_createTime": "2016-12-14T18:44:06Z",
        "auto_stringITMS_publisher": ["en|Publisher", "ignore"],
        "stringIMS_labelEvaluateGrade": ["en|SCOPUS", "ignore"],
        "stringIS_sysProcessingUnit": "CN311001",
        "stringIMS_hasHoldinglibraryCode": ["en|CN311001", "ignore"],
        "stringIMS_entityType": ["en|Article", "ignore"],
        "stringOM_hasAuthor": ["author-a∷name≡Alice"],
        "stringOM_hasSource": ["source-a∷title≡Journal A", "source-b∷title≡Journal B"],
        "stringOM_hasFile": ["file-a∷fileType≡fileType=pdf", "file-b∷fileType≡xml"],
        "stringOM_hasCreateOrganization": ["org-a∷name≡Institute"],
    }
    values = source_values(document, DOC_ID)
    assert {name: values[name] for name in (
        "hasSource", "hasFile", "volume", "issue", "startPage", "endPage",
        "pageCount", "createTime", "publisher", "labelEvaluateGrade",
        "sysProcessingUnit", "hasHoldinglibraryCode", "hasAuthor_str",
        "hasSource_str", "hasFile_str", "hasCreateOrganization_str", "entityType",
    )} == {
        "hasSource": "Journal A;Journal B",
        "hasFile": "pdf;xml",
        "volume": "807", "issue": "1", "startPage": "1", "endPage": "3",
        "pageCount": "24", "createTime": "2016-12-14T18:44:06Z",
        "publisher": "en|Publisher", "labelEvaluateGrade": "SCOPUS",
        "sysProcessingUnit": "CN311001", "hasHoldinglibraryCode": "CN311001",
        "hasAuthor_str": '["author-a∷name≡Alice"]',
        "hasSource_str": '["source-a∷title≡Journal A", "source-b∷title≡Journal B"]',
        "hasFile_str": '["file-a∷fileType≡fileType=pdf", "file-b∷fileType≡xml"]',
        "hasCreateOrganization_str": '["org-a∷name≡Institute"]',
        "entityType": "Article",
    }
    assert "stringOM_hasSource" in SOLR_FIELDS


def test_v4_first_element_missing_and_structured_source_fallback() -> None:
    values = source_values({
        "id": DOC_ID,
        "stringIMS_volume": ["", "999"],
        "stringIMS_entityType": ["", "Other"],
        "auto_stringITMS_hasSourcetitle": ["en|Fallback Journal"],
        "stringIMS_idSet": ["alternate-id"],
    }, DOC_ID)
    assert "volume" not in values and "entityType" not in values
    assert values["hasSource"] == "Fallback Journal"
    assert values["hasSourceidSet"] == ["alternate-id"]
    assert "hasSource_str" not in values


@pytest.mark.parametrize("raw, expected", [
    (["file-a∷fileType≡pdf"], "pdf"),
    (["file-a∷fileType≡fileType≡pdf"], "pdf"),
    (["file-a∷fileType≡fileType=pdf"], "pdf"),
    ([{"id": "file-a", "fileType": "pdf"}], "pdf"),
    (["pdf", "xml"], "pdf;xml"),
])
def test_structured_file_formats(raw, expected) -> None:
    values = source_values({"id": DOC_ID, "stringOM_hasFile": raw}, DOC_ID)
    assert values["hasFile"] == expected


def test_structured_raw_value_over_target_limit_is_not_truncated() -> None:
    with pytest.raises(PermanentDataError, match="hasFile_str"):
        source_values({"id": DOC_ID, "stringOM_hasFile": [
            "file-a∷fileType≡pdf∷url≡/" + "x" * 500,
        ]}, DOC_ID)


def test_solr_logical_delete_marker_is_metadata_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    target = FakeMilvus({"id": DOC_ID, "sysDataStatus": "enable", "doi": "old"})
    worker = IncrementalWorker(settings)
    worker.solr = FakeSolr({"id": DOC_ID, "stringIS_sysDataStatus": "delete"})
    worker.embedder, worker.milvus = FakeEmbed(), target
    assert worker.process_id(DOC_ID) == "updated"
    assert target.calls[0][2] == {"id": DOC_ID, "sysDataStatus": "delete"}
    assert target.calls[0][3] is False


def test_missing_values_preserve_old_fields_and_existing_status() -> None:
    values = source_values({"id": DOC_ID, "auto_stringITMS_title": ["New"]}, DOC_ID)
    old = {"id": DOC_ID, "title": ["Old"], "doi": "old-doi", "entityType": "old-type",
           "hasAuthor_str": "legacy", "sync_status": "CORE_SUCCESS"}
    payload = update_payload(DOC_ID, values, {}, old)
    assert payload == {"id": DOC_ID, "title": ["New"]}
    assert "doi" not in payload and "entityType" not in payload
    assert "hasAuthor_str" not in payload and "sync_status" not in payload


def test_structured_raw_fields_update_only_when_present() -> None:
    values = source_values({"id": DOC_ID, "stringOM_hasAuthor": ["a∷name≡Alice"]}, DOC_ID)
    old = {"id": DOC_ID, "hasAuthor_str": "legacy", "doi": "old-doi",
           "abstract_embed": VECTOR}
    payload = update_payload(DOC_ID, values, {}, old)
    assert payload == {"id": DOC_ID, "hasAuthor_str": '["a∷name≡Alice"]',
                       "hasAuthorname": ["Alice"]}
    assert "abstract_embed" not in payload


def test_new_record_has_all_fields_and_rejects_overlong_value() -> None:
    row = new_record(DOC_ID, {"entityType": "Article"}, {})
    assert set(row) == set(FIELDS)
    assert row["entityType"] == "Article"
    assert row["hasAuthor_str"] is None
    assert row["title_embed"] == VECTOR
    with pytest.raises(PermanentDataError):
        source_values({"id": DOC_ID, "stringIMS_entityType": ["x" * 51]}, DOC_ID)


def test_schema_rejects_missing_field_or_wrong_vector_dimension() -> None:
    fields = []
    for name, spec in FIELDS.items():
        kind = {"varchar": 21, "array": 22, "vector": 101}[spec.kind]
        params = {"max_length": spec.max_length} if spec.kind != "vector" else {"dim": spec.dim}
        if spec.kind == "array":
            params["max_capacity"] = spec.max_capacity
        fields.append({"name": name, "type": kind, "params": params,
                       "nullable": spec.nullable, "is_primary": name == "id"})
    description = {"fields": fields, "enable_dynamic_field": False}
    check_schema(description)
    with pytest.raises(SafetyPause):
        check_schema({"fields": fields[:-1]})
    fields[-4]["params"]["max_length"] = 123  # length mismatch
    with pytest.raises(SafetyPause):
        check_schema(description)


def test_id_only_message_and_durable_checkpoint_replay(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert parse_message(DOC_ID.encode()) == DOC_ID
    with pytest.raises(PermanentDataError):
        parse_message(b'{"id":"' + DOC_ID.encode() + b'"}')
    identity = target_identity(settings)
    first = Ledger(settings.state_dir, identity)
    assert first.enqueue("test_incremental", 0, 1, DOC_ID)
    assert not first.enqueue("test_incremental", 0, 1, DOC_ID)
    event = first.next_event()
    first.begin(event)
    first.close()
    second = Ledger(settings.state_dir, identity)
    replay = second.next_event()
    assert replay["doc_id"] == DOC_ID and replay["status"] == "processing"
    second.finish(replay, "updated")
    assert second.next_event() is None
    assert second.counts()["updated"] == 1
    second.close()
    with pytest.raises(RuntimeError):
        Ledger(settings.state_dir, "different-target")


def test_single_instance_state_lock(tmp_path: Path) -> None:
    first = SingleInstanceLock(tmp_path / "instance.lock")
    try:
        with pytest.raises(RuntimeError):
            SingleInstanceLock(tmp_path / "instance.lock")
    finally:
        first.close()


class FakeSolr:
    def __init__(self, doc=None):
        self.doc = doc
    def fetch(self, _):
        return self.doc


class FakeEmbed:
    def embed(self, inputs):
        return {name + "_embed": VECTOR for name in inputs}


class FakeMilvus:
    def __init__(self, existing=None):
        self.existing = existing
        self.calls = []
    def get(self, _):
        return self.existing
    def write(self, doc_id, before, payload, *, new):
        self.calls.append((doc_id, before, payload, new))


def test_source_missing_never_writes_and_existing_uses_partial_payload(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    target = FakeMilvus({"id": DOC_ID, "title": ["Old"], "doi": "old", "hasFile_str": "legacy"})
    worker = IncrementalWorker(settings, solr=FakeSolr(None), embedder=FakeEmbed(), milvus=target)
    worker.solr, worker.embedder, worker.milvus = worker._injected_solr, worker._injected_embedder, target
    assert worker.process_id(DOC_ID) == "source_missing"
    assert not target.calls
    worker.solr.doc = {"id": DOC_ID, "auto_stringITMS_title": ["New"]}
    assert worker.process_id(DOC_ID) == "updated"
    _, before, payload, new = target.calls[0]
    assert not new and before is target.existing
    assert set(payload) == {"id", "title", "title_embed"}
    assert "doi" not in payload and "hasFile_str" not in payload


def test_new_id_insert_is_41_fields_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    target = FakeMilvus()
    worker = IncrementalWorker(settings, solr=FakeSolr({"id": DOC_ID, "stringIMS_entityType": ["article"]}),
                               embedder=FakeEmbed(), milvus=target)
    worker.solr, worker.embedder, worker.milvus = worker._injected_solr, worker._injected_embedder, target
    assert worker.process_id(DOC_ID) == "inserted"
    _, _, payload, new = target.calls[0]
    assert new and set(payload) == set(FIELDS)


def test_kafka_offset_is_committed_after_durable_enqueue(tmp_path: Path) -> None:
    class Message:
        def value(self): return DOC_ID.encode()
        def error(self): return None
        def topic(self): return "test_incremental"
        def partition(self): return 0
        def offset(self): return 9

    class Consumer:
        def __init__(self, ledger): self.ledger = ledger
        def poll(self, _): return Message()
        def commit(self, *, message, asynchronous):
            assert not asynchronous and message.offset() == 9
            assert self.ledger.next_event()["doc_id"] == DOC_ID

    settings = make_settings(tmp_path)
    worker = IncrementalWorker(settings)
    worker.ledger = Ledger(settings.state_dir, target_identity(settings))
    worker.consumer = Consumer(worker.ledger)
    worker._consume_one()
    assert worker.ledger.counts()["queued"] == 1
    worker.ledger.close()


def test_milvus_existing_write_is_partial_and_checks_historical_fields(monkeypatch) -> None:
    class Client:
        def __init__(self):
            self.current = {"id": DOC_ID, "title": ["Old"], "doi": "original"}
            self.calls = []
        def upsert(self, **kwargs):
            self.calls.append(kwargs)
            self.current.update(kwargs["data"][0])
        def get(self, *_, **__):
            return [dict(self.current)]

    client = Client()
    target = object.__new__(MilvusTarget)
    target.client, target.collection, target.timeout = client, "scientific_documents_v2", 1
    before = dict(client.current)
    target.write(DOC_ID, before, {"id": DOC_ID, "title": ["New"]}, new=False)
    assert client.calls[0]["partial_update"] is True
    assert set(client.calls[0]["data"][0]) == {"id", "title"}
    assert client.current["doi"] == "original"


def test_uncertain_milvus_write_is_verified_before_any_retry(monkeypatch) -> None:
    class Client:
        def __init__(self): self.current = {"id": DOC_ID, "title": ["Old"]}
        def upsert(self, **kwargs):
            self.current.update(kwargs["data"][0])
            raise TimeoutError("request timed out after server accepted it")
        def get(self, *_, **__): return [dict(self.current)]

    monkeypatch.setattr("incremental_sync.adapters.time.sleep", lambda _: None)
    target = object.__new__(MilvusTarget)
    target.client, target.collection, target.timeout = Client(), "scientific_documents_v2", 1
    target.write(DOC_ID, {"id": DOC_ID, "title": ["Old"]},
                 {"id": DOC_ID, "title": ["New"]}, new=False)


def test_batch_recovery_never_recreates_an_existing_record() -> None:
    class Client:
        def __init__(self): self.upserts = 0; self.inserts = 0
        def upsert(self, **_):
            self.upserts += 1
            raise TimeoutError("ambiguous update")
        def insert(self, **_): self.inserts += 1
        def get(self, *_, **__): return []

    target = object.__new__(MilvusTarget)
    target.client, target.collection, target.timeout = Client(), "scientific_documents_v2", 1
    with pytest.raises(SafetyPause, match="refusing to recreate"):
        target.write_batch([{"id": DOC_ID, "before": {"id": DOC_ID, "title": ["Old"]},
                             "payload": {"id": DOC_ID, "title": ["New"]}, "new": False}])
    assert target.client.upserts == 1
    assert target.client.inserts == 0


def test_batch_writer_caps_each_milvus_request_at_200_rows() -> None:
    class Client:
        def __init__(self): self.rows = {}; self.sizes = []; self.inserts = 0
        def upsert(self, **kwargs):
            self.sizes.append(len(kwargs["data"]))
            assert kwargs["partial_update"] is True
            for row in kwargs["data"]: self.rows[row["id"]].update(row)
        def get(self, _, *, ids, **__): return [dict(self.rows[doc_id]) for doc_id in ids]

    target = object.__new__(MilvusTarget)
    target.client, target.collection, target.timeout = Client(), "scientific_documents_v2", 1
    records = []
    for number in range(401):
        doc_id = f"{number + 1:032x}"
        before = {"id": doc_id, "title": ["old"]}
        target.client.rows[doc_id] = dict(before)
        records.append({"id": doc_id, "before": before,
                        "payload": {"id": doc_id, "title": ["new"]}, "new": False})
    target.write_batch(records)
    assert target.client.sizes == [200, 200, 1]
    assert all(row["title"] == ["new"] for row in target.client.rows.values())


def test_unrelated_historical_change_forces_safety_pause() -> None:
    class Client:
        def __init__(self): self.current = {"id": DOC_ID, "title": ["Old"], "doi": "old"}
        def upsert(self, **kwargs):
            self.current.update(kwargs["data"][0])
            self.current["doi"] = "unexpected-change"
        def get(self, *_, **__): return [dict(self.current)]

    target = object.__new__(MilvusTarget)
    target.client, target.collection, target.timeout = Client(), "scientific_documents_v2", 1
    with pytest.raises(SafetyPause):
        target.write(DOC_ID, {"id": DOC_ID, "title": ["Old"], "doi": "old"},
                     {"id": DOC_ID, "title": ["New"]}, new=False)


def test_source_missing_and_transient_failure_are_ledgered_separately(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    worker = IncrementalWorker(settings, solr=FakeSolr(None), embedder=FakeEmbed(), milvus=FakeMilvus())
    worker.solr, worker.embedder, worker.milvus = worker._injected_solr, worker._injected_embedder, worker._injected_milvus
    worker.ledger = Ledger(settings.state_dir, target_identity(settings))
    worker.ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    worker._process_event(worker.ledger.next_event())
    assert worker.ledger.counts()["source_missing"] == 1
    assert worker.ledger.failures()[0]["doc_id"] == DOC_ID
    assert not worker.milvus.calls

    class UnavailableSolr:
        def fetch(self, _): raise TemporaryDependencyError("Solr HTTP 503")

    worker.solr = UnavailableSolr()
    worker.ledger.enqueue("test_incremental", 0, 2, DOC_ID)
    worker._process_event(worker.ledger.next_event())
    pending = worker.ledger.next_event()
    assert pending["offset_id"] == 2 and pending["status"] == "queued"
    assert worker.ledger.counts()["source_missing"] == 1
    assert worker.ledger.daily()[0]["pending"] == 1
    worker.ledger.close()


def test_worker_batch_uses_bulk_adapters_and_ledgers_missing_sources(tmp_path: Path) -> None:
    class BatchSolr:
        def __init__(self): self.calls = []
        def fetch_many(self, doc_ids):
            self.calls.append(doc_ids)
            return {DOC_ID: {"id": DOC_ID, "auto_stringITMS_title": ["New title"]}}

    class BatchEmbed:
        def __init__(self): self.calls = []
        def embed_many(self, documents):
            self.calls.append(documents)
            return {DOC_ID: {"title_embed": VECTOR}}

    class BatchMilvus:
        def __init__(self): self.reads = []; self.writes = []
        def get_many(self, doc_ids):
            self.reads.append(doc_ids)
            return {DOC_ID: {"id": DOC_ID, "title": ["Old title"], "doi": "keep"}}
        def write_batch(self, writes): self.writes.extend(writes)

    settings = make_settings(tmp_path)
    missing = "0000019bdeaccd691ec2a327efbbc733"
    solr, embedder, milvus = BatchSolr(), BatchEmbed(), BatchMilvus()
    worker = IncrementalWorker(settings, solr=solr, embedder=embedder, milvus=milvus)
    worker.solr, worker.embedder, worker.milvus = solr, embedder, milvus
    worker.ledger = Ledger(settings.state_dir, target_identity(settings))
    worker.ledger.enqueue(settings.kafka_topic, 0, 1, DOC_ID)
    worker.ledger.enqueue(settings.kafka_topic, 0, 2, missing)
    worker.process_batch(worker.ledger.next_events(10))

    assert solr.calls == [[DOC_ID, missing]]
    assert embedder.calls == [{DOC_ID: {"title": "New title"}}]
    assert milvus.reads == [[DOC_ID]]
    assert len(milvus.writes) == 1
    assert milvus.writes[0]["payload"] == {"id": DOC_ID, "title": ["New title"],
                                           "title_embed": VECTOR}
    assert worker.ledger.counts() == {"source_missing": 1, "updated": 1}
    worker.ledger.close()


def test_batch_solr_failure_does_not_requeue_invalid_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    worker = IncrementalWorker(settings)
    worker.ledger = Ledger(settings.state_dir, target_identity(settings))
    worker.ledger.enqueue(settings.kafka_topic, 0, 1, "malformed")
    worker.ledger.enqueue(settings.kafka_topic, 0, 2, DOC_ID)
    class UnavailableSolr:
        def fetch_many(self, _): raise TemporaryDependencyError("Solr unavailable")
    worker.solr = UnavailableSolr()
    worker.process_batch(worker.ledger.next_events(10))
    assert worker.ledger.counts() == {"failed": 1, "queued": 1}
    queued = worker.ledger.conn.execute("SELECT doc_id FROM events WHERE status='queued'").fetchall()
    assert [row[0] for row in queued] == [DOC_ID]
    worker.ledger.close()


def test_failure_outbox_is_atomic_idempotent_and_replayable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ledger = Ledger(settings.state_dir, target_identity(settings))
    ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    event = ledger.next_event()
    ledger.begin(event)
    ledger.finish(event, "source_missing", duration=1.25, target={"collection": settings.milvus_collection})
    ledger.finish(event, "source_missing")
    assert ledger.counts()["source_missing"] == 1
    assert ledger.outbox_pending() == 1
    failure = ledger.unsent()[0]
    assert failure["document_id"] == DOC_ID
    assert failure["target"]["collection"] == settings.milvus_collection

    class Collection:
        def __init__(self): self.writes = []
        def bulk_write(self, writes, ordered):
            self.writes.extend(writes)

    mongo = Collection()
    reporter = FailureReporter(settings, collection=mongo)
    assert reporter.sync(ledger, force=True) == 1
    assert ledger.outbox_pending() == 0
    assert reporter.sync(ledger, force=True) == 0
    assert len(mongo.writes) == 1
    ledger.close()


def test_temporary_failure_does_not_block_later_event(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ledger = Ledger(settings.state_dir, target_identity(settings))
    ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    ledger.enqueue("test_incremental", 0, 2, "0000019bdeaccd691ec2a327efbbc733")
    first = ledger.next_event()
    ledger.begin(first)
    ledger.defer(first, "Solr unavailable", duration=0.1, delay=60)
    assert ledger.next_event()["offset_id"] == 2
    ledger.close()


def test_daily_activity_has_actual_processing_interval(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ledger = Ledger(settings.state_dir, target_identity(settings))
    ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    event = ledger.next_event()
    ledger.begin(event)
    ledger.finish(event, "updated", duration=1.5)
    report = ledger.activity_daily()[0]
    assert report["updated"] == 1
    assert report["active_processing_seconds"] == pytest.approx(1.5)
    assert report["first_started_at"].endswith("+08:00")
    ledger.close()


def test_mongo_offline_outbox_survives_and_catches_up(tmp_path: Path) -> None:
    class Offline:
        def bulk_write(self, *args, **kwargs): raise ConnectionError("offline")
    class Online:
        def bulk_write(self, writes, ordered): assert len(writes) == 1
    settings = make_settings(tmp_path)
    ledger = Ledger(settings.state_dir, target_identity(settings))
    ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    event = ledger.next_event()
    ledger.begin(event)
    ledger.finish(event, "failed", "overlong field")
    reporter = FailureReporter(settings, collection=Offline())
    assert reporter.sync(ledger, force=True) == 0
    assert ledger.outbox_pending() == 1
    ledger.close()
    ledger = Ledger(settings.state_dir, target_identity(settings))
    reporter.collection = Online()
    assert reporter.sync(ledger, force=True) == 1
    assert ledger.outbox_pending() == 0
    assert ledger.counts()["failed"] == 1
    ledger.close()


def test_exhausted_document_retries_continue_to_next_id(tmp_path: Path) -> None:
    from dataclasses import replace
    settings = replace(make_settings(tmp_path), retry_wait_sec=0)
    worker = IncrementalWorker(settings)
    worker.ledger = Ledger(settings.state_dir, target_identity(settings))
    other = "0000019bdeaccd691ec2a327efbbc733"
    worker.ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    worker.ledger.enqueue("test_incremental", 0, 2, other)
    calls = []
    def process(doc_id):
        calls.append(doc_id)
        if doc_id == DOC_ID: raise TemporaryDependencyError("Solr HTTP 503")
        return "updated"
    worker.process_id = process
    while event := worker.ledger.next_event(): worker._process_event(event)
    assert calls == [DOC_ID] * 3 + [other]
    assert worker.ledger.counts() == {"failed": 1, "updated": 1}
    assert worker.ledger.unsent()[0]["attempts"] == 3
    worker.ledger.close()


def test_integrity_exception_is_not_quarantined_as_a_document_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    worker = IncrementalWorker(settings)
    worker.ledger = Ledger(settings.state_dir, target_identity(settings))
    worker.ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    def process(_): raise SafetyPause("historical field changed")
    worker.process_id = process
    with pytest.raises(SafetyPause): worker._process_event(worker.ledger.next_event())
    assert worker.ledger.counts() == {"processing": 1}
    assert worker.ledger.outbox_pending() == 0
    worker.ledger.close()


def test_daily_activity_splits_shanghai_midnight(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ledger = Ledger(settings.state_dir, target_identity(settings))
    with ledger.conn:
        ledger._activity("2026-10-20T16:00:01+00:00", 2, "updated")
    reports = ledger.activity_daily()
    assert [r["day"] for r in reports] == ["2026-10-20", "2026-10-21"]
    assert [r["active_processing_seconds"] for r in reports] == [1, 1]
    assert [r["updated"] for r in reports] == [0, 1]
    ledger.close()


def test_failure_and_outcome_rollback_together(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ledger = Ledger(settings.state_dir, target_identity(settings))
    ledger.enqueue("test_incremental", 0, 1, DOC_ID)
    event = ledger.next_event()
    ledger.begin(event)
    with pytest.raises(TypeError):
        ledger.finish(event, "failed", "bad", target={"cannot_serialize": object()})
    assert ledger.counts() == {"processing": 1}
    assert ledger.outbox_pending() == 0
    ledger.close()


def test_milvus_array_container_normalized_for_audit_and_comparison() -> None:
    class ProtobufArray:
        def __getitem__(self, index): return ["title"][index]
        def __len__(self): return 1
    class Client:
        def get(self, *args, **kwargs): return [{"id": DOC_ID, "title": ProtobufArray()}]
    target = object.__new__(MilvusTarget)
    target.client, target.collection, target.timeout = Client(), "test", 1
    assert target.get(DOC_ID) == {"id": DOC_ID, "title": ["title"]}


def test_run_loop_keeps_consuming_after_failures_and_mongo_outage(tmp_path: Path, monkeypatch) -> None:
    """Exercise the real lifecycle/ledger, with no external broker or database."""
    from dataclasses import replace
    from types import SimpleNamespace
    import incremental_sync.worker as worker_module

    settings = replace(make_settings(tmp_path), mongo_uri="mongodb://fake", retry_wait_sec=0,
                       idle_window_sec=0, batch_size=1, batch_wait_sec=0)
    missing, unavailable, good = (f"{n:032x}" for n in (1, 2, 3))
    values = [b"malformed", missing.encode(), unavailable.encode(), good.encode()]
    snapshots = []

    class Consumer:
        def __init__(self): self.position = 0; self.committed = []; self.closed = False; self.empty = 0
        def subscribe(self, topics): assert topics == [settings.kafka_topic]
        def poll(self, _):
            if self.position < len(values):
                self.position += 1
                n = self.position
                return SimpleNamespace(value=lambda: values[n-1], error=lambda: None,
                    topic=lambda: settings.kafka_topic, partition=lambda: 0, offset=lambda: n)
            self.empty += 1
            if self.empty == 2:
                snapshots.append((worker.status, worker.ledger.counts(), worker.ledger.outbox_pending()))
                worker.stop()
            return None
        def commit(self, *, message, asynchronous):
            assert not asynchronous
            assert worker.ledger.next_event()["offset_id"] == message.offset()
            self.committed.append(message.offset())
        def close(self): self.closed = True

    class Reporter:
        status = "unavailable:ConnectionError"
        def sync(self, ledger, **_): return 0
        def close(self): pass

    consumer = Consumer()
    resource = SimpleNamespace(close=lambda: None)
    worker = IncrementalWorker(settings, consumer=consumer, solr=resource, embedder=resource, milvus=resource)
    worker.reporter = Reporter()
    worker.preflight = lambda: {"fields": 41}
    calls = []
    def process(doc_id):
        calls.append(doc_id)
        if doc_id == missing: return "source_missing"
        if doc_id == unavailable: raise TemporaryDependencyError("Solr unavailable")
        return "updated"
    worker.process_id = process
    worker.process_batch = lambda events: [worker._process_event(event) for event in events]
    server = SimpleNamespace(serve_forever=lambda: worker.stop_event.wait(),
                             shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(worker_module, "start_status_server", lambda _: server)
    import threading
    watchdog = threading.Timer(5, worker.stop)
    watchdog.start()
    try:
        worker.run()
    finally:
        watchdog.cancel()
    assert consumer.committed == [1, 2, 3, 4] and consumer.closed
    assert calls.count(unavailable) == 3 and calls[-1] == good
    assert snapshots == [("idle", {"failed": 2, "source_missing": 1, "updated": 1}, 3)]
    assert not (settings.state_dir / "safety_pause.json").exists()
    ledger = Ledger(settings.state_dir, target_identity(settings))
    assert ledger.outbox_pending() == 3
    ledger.close()


def test_persisted_safety_pause_blocks_restart_writes(tmp_path: Path) -> None:
    from dataclasses import replace
    from incremental_sync.store import atomic_json

    settings = replace(make_settings(tmp_path), mongo_uri="mongodb://fake")
    marker = {"reason": "historical integrity failure", "at": "2026-10-20T00:00:00+00:00"}
    atomic_json(settings.state_dir / "safety_pause.json", marker)
    worker = IncrementalWorker(settings)
    worker.connect = lambda: pytest.fail("must not connect/consume through a persisted safety pause")
    class StopOnWait:
        stopped = False
        def is_set(self): return self.stopped
        def set(self): self.stopped = True
        def wait(self, _): self.set(); return True
    worker.stop_event = StopOnWait()
    worker.run()
    import json
    assert worker.status == "safety_pause"
    assert json.loads((settings.state_dir / "safety_pause.json").read_text()) == marker
    assert not (settings.state_dir / "checkpoint.db").exists()


def test_validation_failure_is_clearly_separate_from_pending_business_failure(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "validation-run")
    ledger.enqueue("__local_validation_only__", 0, 0, "__validation__")
    event = ledger.next_event()
    ledger.begin(event)
    ledger.finish(event, "failed", "synthetic validation only", validation_run_id="acceptance-100")
    document = ledger.unsent()[0]
    assert document["validation_test"] is True
    assert document["status"] == "validation_test" and document["validation_run_id"] == "acceptance-100"
    ledger.close()


def test_index_definition_audit_ignores_only_runtime_statistics() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_live_100", Path(__file__).parents[1] / "tools/verify_live_100.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    description = {"index_type": "IVF_PQ", "metric_type": "COSINE", "nlist": "4096",
        "total_rows": 100, "indexed_rows": 99, "pending_index_rows": 1, "state": "InProgress"}
    after = {**description, "indexed_rows": 100, "pending_index_rows": 0, "state": "Finished"}
    assert helpers.index_definition(description) == helpers.index_definition(after)
    after["nlist"] = "1024"
    assert helpers.index_definition(description) != helpers.index_definition(after)
