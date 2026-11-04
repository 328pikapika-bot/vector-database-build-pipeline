import json
import hashlib

from solr_field_backfill.entity_type_backfill import (
    DurableCheckpoint,
    EntityTypeBackfill,
    FIELD_NAME,
    Settings,
    SOLR_FIELD,
)


class FakeResponse:
    def __init__(self, docs):
        self.docs = docs

    def raise_for_status(self):
        return None

    def json(self):
        return {"responseHeader": {"status": 0}, "response": {"docs": self.docs}}


class FakeSession:
    def __init__(self, docs):
        self.docs = docs
        self.params = None

    def post(self, _url, *, data, timeout):
        self.params = data
        requested = data["q"].split("}", 1)[1].split(",")
        return FakeResponse([doc for doc in self.docs if doc["id"] in requested])

    def close(self):
        pass


class FakeMilvus:
    def __init__(self, ids):
        self.rows = {row_id: {"id": row_id, FIELD_NAME: None, "title_embed": [1.0]}
                     for row_id in ids}
        self.calls = []
        self.timeout_after_apply = False

    def get(self, *, collection_name, ids, output_fields, consistency_level, timeout):
        assert output_fields == ["id", FIELD_NAME]
        assert consistency_level == "Strong"
        return [{key: self.rows[row_id][key] for key in output_fields} for row_id in ids]

    def upsert(self, *, collection_name, data, partial_update, timeout):
        assert partial_update is True
        assert all(set(row) == {"id", FIELD_NAME} for row in data)
        self.calls.append([dict(row) for row in data])
        for row in data:
            self.rows[row["id"]][FIELD_NAME] = row[FIELD_NAME]
        if self.timeout_after_apply:
            self.timeout_after_apply = False
            raise TimeoutError("simulated uncertain write")
        return {"upsert_count": len(data)}


def build_worker(tmp_path, ids, docs):
    state_dir = tmp_path / "entity_type"
    settings = Settings(
        milvus_uri="http://milvus", database="default", collection="docs",
        collection_id=1, logical_count=10, solr_url="http://solr/select",
        source_checkpoint=tmp_path / "source.db", state_dir=state_dir,
    )
    milvus = FakeMilvus(ids)
    session = FakeSession(docs)
    worker = EntityTypeBackfill(settings, milvus=milvus, session_factory=lambda: session)
    worker.task_id = "task"
    worker.target = {"collection_id": 1}
    with worker.checkpoint.db:
        worker.checkpoint.db.executemany(
            "INSERT INTO batches(prefix,number,ids,updated_at) VALUES(?,?,?,?)",
            [(row_id[:2], index, json.dumps([row_id]), "now") for index, row_id in enumerate(ids)],
        )
    return worker, milvus, session


def test_uses_first_solr_entity_type_and_only_partial_updates_entity_type(tmp_path):
    ids = ["00000000000000000000000000000001"]
    docs = [{"id": ids[0], SOLR_FIELD: ["JournalPaper", "OtherType"]}]
    worker, milvus, session = build_worker(tmp_path, ids, docs)
    try:
        worker._prepare_batch("00", 0, ids)
        prepared = worker.checkpoint.batch("00", 0)
        assert prepared["payload"] == [{"id": ids[0], FIELD_NAME: "JournalPaper"}]
        assert session.params["fl"] == f"id,{SOLR_FIELD}"
        worker._write_inflight("00", 0)
        assert worker.checkpoint.phase("00", 0) == "acked"
        assert milvus.rows[ids[0]][FIELD_NAME] == "JournalPaper"
        assert milvus.calls == [[{"id": ids[0], FIELD_NAME: "JournalPaper"}]]
        assert milvus.rows[ids[0]]["title_embed"] == [1.0]
    finally:
        worker.close()


def test_missing_and_oversize_source_values_are_recorded_without_writing(tmp_path):
    ids = ["00000000000000000000000000000001", "00000000000000000000000000000002"]
    docs = [{"id": ids[0], SOLR_FIELD: []}, {"id": ids[1], SOLR_FIELD: ["X" * 51]}]
    worker, milvus, _ = build_worker(tmp_path, ids, docs)
    try:
        worker._prepare_batch("00", 0, ids)
        prepared = worker.checkpoint.batch("00", 0)
        assert prepared["payload"] == []
        assert prepared["result"] == {"processed": 2, "updated": 0, "unchanged": 0, "failed": 2}
        worker._write_inflight("00", 0)
        assert milvus.calls == []
        failures = worker.checkpoint.unsent_failures()
        assert {item["reason"] for _, item in failures} == {"entity_type_missing", "value_too_long"}
    finally:
        worker.close()


def test_empty_first_element_does_not_fall_back_and_language_prefix_is_stripped(tmp_path):
    ids = [f"{index:032x}" for index in range(1, 6)]
    docs = [
        {"id": ids[0], SOLR_FIELD: ["", "Patent"]},
        {"id": ids[1], SOLR_FIELD: [None, "Patent"]},
        {"id": ids[2], SOLR_FIELD: [" JournalPaper ", "Patent"]},
        {"id": ids[3], SOLR_FIELD: [123, "Patent"]},
        {"id": ids[4], SOLR_FIELD: ["en|Article", "Patent"]},
    ]
    worker, milvus, _ = build_worker(tmp_path, ids, docs)
    try:
        worker._prepare_batch("00", 0, ids)
        prepared = worker.checkpoint.batch("00", 0)
        assert prepared["payload"] == [
            {"id": ids[2], FIELD_NAME: " JournalPaper "},
            {"id": ids[3], FIELD_NAME: "123"},
            {"id": ids[4], FIELD_NAME: "Article"},
        ]
        assert prepared["result"] == {"processed": 5, "updated": 3, "unchanged": 0, "failed": 2}
        worker._write_inflight("00", 0)
        assert len(milvus.calls) == 1
        assert all(milvus.rows[row_id][FIELD_NAME] is None for row_id in (ids[0], ids[1]))
    finally:
        worker.close()


def test_uncertain_timeout_is_strongly_reconciled_before_resubmit(tmp_path):
    ids = ["00000000000000000000000000000001"]
    docs = [{"id": ids[0], SOLR_FIELD: ["Patent"]}]
    worker, milvus, _ = build_worker(tmp_path, ids, docs)
    milvus.timeout_after_apply = True
    try:
        worker._prepare_batch("00", 0, ids)
        worker._write_inflight("00", 0)
        assert len(milvus.calls) == 1
        assert milvus.rows[ids[0]][FIELD_NAME] == "Patent"
        assert worker.checkpoint.phase("00", 0) == "acked"
    finally:
        worker.close()


def test_checkpoint_rejects_a_different_manifest(tmp_path):
    checkpoint = DurableCheckpoint(tmp_path / "checkpoint.db")
    try:
        ids = ["00000000000000000000000000000001"]
        digest = hashlib.sha256(json.dumps(["00", 0, ids], separators=(",", ":")).encode()).hexdigest()
        checkpoint.bind_manifest(tmp_path / "source.db", digest, 1,
                                 [("00", 0, json.dumps(ids))])
        try:
            checkpoint.bind_manifest(tmp_path / "source.db", "b" * 64, 1, [])
        except RuntimeError as exc:
            assert "another manifest" in str(exc)
        else:
            raise AssertionError("checkpoint accepted a different source manifest")
    finally:
        checkpoint.close()


def test_milvus_vector_container_is_converted_to_json_safe_values():
    class RepeatedLike:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return [1.25, 2.5][index]

    assert EntityTypeBackfill._plain(RepeatedLike()) == [1.25, 2.5]
