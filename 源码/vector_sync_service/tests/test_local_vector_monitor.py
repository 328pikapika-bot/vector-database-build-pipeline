from __future__ import annotations

import http.client
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "local_vector_monitor.py"
SPEC = importlib.util.spec_from_file_location("local_vector_monitor", SCRIPT_PATH)
local_vector_monitor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(local_vector_monitor)


def test_default_three_lane_configuration(monkeypatch):
    monkeypatch.setattr(local_vector_monitor, "_PROJECT_ENV", {}, raising=False)
    config = local_vector_monitor.get_config()

    assert config["collections"]["documents"]["name"] == "scientific_documents_v2"
    assert config["collections"]["fulltext_chunks"]["name"] == "scientific_fulltext_chunks_v2"
    assert config["collections"]["images"]["name"] == "scientific_images_v2"
    assert config["kafka"]["lanes"]["documents"]["topic"] == "VECTOR_JOURNAL_ARTICLE_METADATA_BASE"
    assert config["kafka"]["lanes"]["images"]["group_id"] == "vector_sync_image_v2--target-e72d233f5c35"


def test_mongo_config_uses_explicit_monitor_overrides(monkeypatch):
    monkeypatch.setattr(local_vector_monitor, "_PROJECT_ENV", {}, raising=False)
    monkeypatch.setenv("VECTOR_MONITOR_MONGO_URI", "mongodb://readonly@example:27017")
    monkeypatch.setenv("VECTOR_MONITOR_MONGO_DB", "nstl_vector_sync_scientific_prod")
    monkeypatch.setenv("VECTOR_MONITOR_MONGO_IMAGE_TASK_COLLECTION", "custom_image_tasks")

    config = local_vector_monitor.get_config()

    assert config["mongo_uri"] == "mongodb://readonly@example:27017"
    assert config["mongo_database"] == "nstl_vector_sync_scientific_prod"
    assert config["mongo_collections"]["image_tasks"] == "custom_image_tasks"


def test_mongo_summary_is_read_only_and_groups_failures(monkeypatch):
    calls = []

    class FakeCollection:
        def __init__(self, statuses, errors=None, stages=None):
            self.statuses = statuses
            self.errors = errors or []
            self.stages = stages or []

        def count_documents(self, query, **_kwargs):
            calls.append(("count", query))
            if query == {"status": "FAILURE", "retryable": True}:
                return sum(1 for row in self.errors if row[1])
            if query == {"error_type": "RETRYABLE"}:
                return 2
            status = query.get("status")
            if isinstance(status, str):
                return self.statuses.get(status, 0)
            return sum(self.statuses.values())

        def aggregate(self, pipeline, **_kwargs):
            calls.append(("aggregate", pipeline))
            field = pipeline[1]["$group"]["_id"][1:]
            rows = self.stages if field == "failure_stage" else self.errors
            return [{"_id": value, "count": count} for value, count in rows]

        def insert_one(self, *_args, **_kwargs):
            raise AssertionError("monitor must not insert")

        def update_one(self, *_args, **_kwargs):
            raise AssertionError("monitor must not update")

        def delete_one(self, *_args, **_kwargs):
            raise AssertionError("monitor must not delete")

        def create_index(self, *_args, **_kwargs):
            raise AssertionError("monitor must not create indexes")

    failure = FakeCollection(
        {"PENDING": 3, "RETRYING": 1, "DONE": 2, "GAVE_UP": 4, "HELD_HISTORY": 0, "SKIPPED_STALE": 0},
        errors=[("IMAGE_TASK_PROCESS_ERROR", 1), ("MILVUS_WRITE_ERROR", 0)],
        stages=[("IMAGE_EMBED_CAPTION", 4), ("MILVUS_WRITE_DOCUMENT", 2)],
    )
    task = FakeCollection(
        {"PENDING": 1, "LEASED": 2, "SUCCESS": 8, "FAILURE": 3, "SKIPPED": 0, "HELD_HISTORY": 0},
        errors=[("IMAGE_FEATURE_PIPELINE_ERROR", 1)],
    )

    class FakeDb:
        def __getitem__(self, name):
            return {"doc_sync_failures": failure, "doc_fulltext_tasks": task, "doc_image_tasks": task}[name]

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.admin = self

        def command(self, name):
            assert name == "ping"
            return {"ok": 1}

        def __getitem__(self, _name):
            return FakeDb()

        def close(self):
            calls.append(("close",))

    monkeypatch.setitem(sys.modules, "pymongo", SimpleNamespace(MongoClient=FakeClient))
    config = local_vector_monitor.get_config()
    config["mongo_uri"] = "mongodb://readonly@example:27017"

    result = local_vector_monitor.read_mongo(config)

    assert result["ready"] is True
    assert result["collections"]["failure_records"]["pending_count"] == 4
    assert result["collections"]["failure_records"]["top_stages"][0]["value"] == "IMAGE_EMBED_CAPTION"
    assert result["collections"]["image_tasks"]["failure_count"] == 3
    assert not {"insert_one", "update_one", "delete_one", "create_index"} & {call[0] for call in calls}


def test_kafka_query_does_not_consume_or_commit(monkeypatch):
    calls = []

    class FakeTopicPartition:
        def __init__(self, topic, partition):
            self.topic = topic
            self.partition = partition

    class FakeConsumer:
        def __init__(self, config):
            calls.append(("init", config))

        def list_topics(self, topic, timeout):
            calls.append(("list_topics", topic))
            return SimpleNamespace(topics={topic: SimpleNamespace(error=None, partitions={0: object()})})

        def committed(self, partitions, timeout):
            calls.append(("committed", len(partitions)))
            return [SimpleNamespace(partition=0, offset=7)]

        def get_watermark_offsets(self, partition, timeout, cached):
            calls.append(("watermark", partition.partition))
            return (0, 10)

        def close(self):
            calls.append(("close",))

        def subscribe(self, *_args):
            raise AssertionError("monitor must not subscribe")

        def poll(self, *_args):
            raise AssertionError("monitor must not poll")

        def commit(self, *_args):
            raise AssertionError("monitor must not commit")

    monkeypatch.setitem(sys.modules, "confluent_kafka", SimpleNamespace(Consumer=FakeConsumer, TopicPartition=FakeTopicPartition))
    config = local_vector_monitor.get_config()
    config["kafka"]["bootstrap_servers"] = "kafka:9092"

    result = local_vector_monitor.read_kafka(config)

    assert result["connected"] is True
    assert result["total_lag"] == 9
    assert all(lane["committed_known"] for lane in result["lanes"].values())
    assert not {"subscribe", "poll", "commit"} & {call[0] for call in calls}


def test_milvus_query_is_read_only_and_keeps_schema(monkeypatch):
    calls = []

    class FakeMilvusClient:
        def __init__(self, **_kwargs):
            calls.append(("init",))

        def has_collection(self, name):
            calls.append(("has_collection", name))
            return True

        def get_collection_stats(self, *, collection_name):
            calls.append(("stats", collection_name))
            return {"row_count": 12}

        def describe_collection(self, name):
            calls.append(("describe", name))
            return {
                "schema": {
                    "fields": [
                        {"name": "id", "type": "INT64", "is_primary": True},
                        {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 4}},
                    ]
                }
            }

        def close(self):
            calls.append(("close",))

        def insert(self, *_args, **_kwargs):
            raise AssertionError("monitor must not insert")

        def delete(self, *_args, **_kwargs):
            raise AssertionError("monitor must not delete")

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient))
    config = local_vector_monitor.get_config()
    result = local_vector_monitor.read_milvus(config)

    assert result["connected"] is True
    assert result["total_rows"] == 36
    assert result["collections"]["documents"]["schema"]["vector_fields"][0]["dim"] == 4
    assert not {"insert", "delete"} & {call[0] for call in calls}


def test_daily_kafka_rate_and_eta_require_full_day_history():
    snapshots = [
        {"ts": "2026-09-07T00:00:00+00:00", "offsets": {"documents": {"committed": 100}}},
        {"ts": "2026-09-07T12:00:00+00:00", "offsets": {"documents": {"committed": 250}}},
        {"ts": "2026-09-08T00:00:00+00:00", "offsets": {"documents": {"committed": 400}}},
    ]

    warming = local_vector_monitor._daily_rate("documents", snapshots[:1], 86400)
    rough = local_vector_monitor._daily_rate("documents", snapshots[1:], 86400)
    ready = local_vector_monitor._daily_rate("documents", snapshots, 86400)

    assert warming["status"] == "warming_up"
    assert rough["status"] == "rough"
    assert rough["messages_per_day"] == 300.0
    assert ready == {
        "status": "ready",
        "messages_per_day": 300.0,
        "elapsed_seconds": 86400,
        "delta_messages": 300,
    }
    assert local_vector_monitor._estimate_days({"lag": 7200}, ready) == {"status": "ready", "days": 24.0}
    assert local_vector_monitor._estimate_days({"lag": None}, ready) == {"status": "offset_unknown", "days": None}


def test_snapshot_retention_removes_expired_samples(tmp_path):
    config = {"snapshot_file": str(tmp_path / "snapshots.jsonl"), "retention_days": 7}
    Path(config["snapshot_file"]).write_text('{"ts":"2020-01-01T00:00:00+00:00"}\n', encoding="utf-8")

    local_vector_monitor._append_snapshot(config, {"ts": local_vector_monitor.utc_now(), "offsets": {}})

    assert len(local_vector_monitor._read_snapshots(config)) == 1


def test_summary_includes_mongo_without_changing_milvus_or_kafka(monkeypatch):
    milvus = {
        "connected": True,
        "collections": {key: {"row_count": 10, "exists": True} for key in local_vector_monitor.COLLECTIONS},
        "total_rows": 30,
    }
    kafka = {
        "connected": True,
        "lanes": {
            key: {"topic": key, "committed": 10, "tail": 12, "lag": 2, "committed_known": True}
            for key in local_vector_monitor.COLLECTIONS
        },
        "total_lag": 6,
        "unknown_lanes": 0,
    }
    monkeypatch.setattr(local_vector_monitor, "read_milvus", lambda _config: milvus)
    monkeypatch.setattr(local_vector_monitor, "read_kafka", lambda _config: kafka)
    mongo = {"connected": True, "ready": True, "database": "prod", "collections": {}}
    monkeypatch.setattr(local_vector_monitor, "read_mongo", lambda _config: mongo)
    monkeypatch.setattr(local_vector_monitor, "_append_snapshot", lambda *_args: None)

    summary = local_vector_monitor.build_summary(force=True)

    assert summary["mongo"] == mongo
    assert summary["milvus"]["total_rows"] == 30
    assert summary["lanes"]["documents"]["lag"] == 2


def test_health_and_http_endpoints(monkeypatch):
    fake_summary = {
        "ts": "2026-09-07T00:00:00+00:00",
        "config": {"refresh_seconds": 60},
        "milvus": {"connected": True},
        "kafka": {"connected": True},
        "mongo": {"ready": True},
        "history": {"sample_count": 1},
    }
    monkeypatch.setattr(local_vector_monitor, "build_summary", lambda: fake_summary)
    monkeypatch.setattr(local_vector_monitor, "build_health", lambda: (200, {"status": "ok"}))
    server = local_vector_monitor.ThreadingHTTPServer(("127.0.0.1", 0), local_vector_monitor.MonitorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/")
        assert "三路向量同步监控" in connection.getresponse().read().decode("utf-8")
        connection.request("GET", "/health")
        assert connection.getresponse().status == 200
        connection.request("GET", "/api/summary")
        assert connection.getresponse().status == 200
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_is_degraded_when_mongo_is_unavailable(monkeypatch):
    fake_summary = {
        "ts": local_vector_monitor.utc_now(),
        "config": {"refresh_seconds": 60},
        "milvus": {"connected": True},
        "kafka": {"connected": True},
        "mongo": {"connected": False, "ready": False},
        "history": {"sample_count": 0},
    }
    monkeypatch.setattr(local_vector_monitor, "build_summary", lambda: fake_summary)

    status, payload = local_vector_monitor.build_health()

    assert status == 503
    assert payload["status"] == "degraded"
    assert payload["dependencies"]["mongo"] is False
