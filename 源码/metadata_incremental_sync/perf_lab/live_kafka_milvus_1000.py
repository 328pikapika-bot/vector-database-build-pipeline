"""Run one bounded Kafka -> live Solr/embedding -> isolated 231 Milvus test."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from pymilvus import DataType, MilvusClient

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
OUT = ROOT / "out" / "metadata_incremental_live_kafka_20261021"
SAMPLE_TOPIC = "VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS"
SOURCE_URI = "http://127.0.0.1:19533"
SOURCE_DB = "default"
SOURCE_COLLECTION = "scientific_documents_v2"
SAMPLE_LIMIT = 1000

os.environ["TZ"] = "Asia/Shanghai"
from incremental_sync.adapters import MilvusTarget, SafetyPause, _equal  # noqa: E402
from incremental_sync.config import Settings  # noqa: E402
from incremental_sync.contract import FIELDS, VECTOR_FIELDS  # noqa: E402
from incremental_sync.worker import IncrementalWorker  # noqa: E402


def load_credentials() -> None:
    merged = dict(dotenv_values(ROOT / ".env"))
    # The live embedding credentials are maintained with the vector service.
    # Keep the project root as a fallback, but prefer the service environment.
    for service_env in (ROOT / "vector_sync_service" / ".env", PROJECT / ".env"):
        if service_env.exists():
            merged.update({key: value for key, value in dotenv_values(service_env).items() if value is not None})
    for key, value in merged.items():
        if value is not None and not os.environ.get(key):
            os.environ[key] = value
    values = {
        "KAFKA_BOOTSTRAP_SERVERS": os.getenv("KAFKA_BOOTSTRAP_SERVERS", ""),
        "KAFKA_SECURITY_PROTOCOL": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
        "KAFKA_SASL_MECHANISM": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
        "SOLR_PAPER_URL": "http://solr-bak1.example.invalid:8984/solr/paper6.6",
        "EMBEDDING_API_URL": "http://127.0.0.1:16703/v1/embeddings",
        "EMBEDDING_MODEL": "qwen3-embed",
        "MILVUS_URI": SOURCE_URI,
        "MILVUS_DATABASE": SOURCE_DB,
        "VECTOR_PARTIAL_UPDATE_VERIFIED": "1",
        "REQUEST_TIMEOUT_SEC": "120",
        "RETRY_WAIT_SEC": "2",
        "MAX_ATTEMPTS": "1",
    }
    for key, value in values.items():
        os.environ[key] = value
    if not os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        raise RuntimeError("Kafka bootstrap configuration is missing")
    if os.getenv("KAFKA_SECURITY_PROTOCOL", "").startswith("SASL") and not all(
        os.getenv(key) for key in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD")
    ):
        raise RuntimeError("Kafka authentication configuration is incomplete")


def kafka_config() -> dict[str, object]:
    from incremental_sync.config import Settings

    settings = Settings.from_env()
    return settings.kafka_config()


def sample_ids() -> tuple[list[str], dict[str, object]]:
    from confluent_kafka import Consumer, TopicPartition
    from incremental_sync.worker import parse_message

    config = kafka_config()
    consumer = Consumer({
        **config,
        "group.id": "metadata_perf_sampler_" + secrets.token_hex(8),
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "auto.offset.reset": "earliest",
    })
    try:
        metadata = consumer.list_topics(SAMPLE_TOPIC, timeout=15)
        topic = metadata.topics.get(SAMPLE_TOPIC)
        if topic is None or topic.error is not None or len(topic.partitions) != 1:
            raise RuntimeError("source Kafka topic is not available as a single-partition topic")
        partition = TopicPartition(SAMPLE_TOPIC, 0)
        low, high = consumer.get_watermark_offsets(partition, timeout=15)
        consumer.assign([TopicPartition(SAMPLE_TOPIC, 0, low)])
        ids: list[str] = []
        seen: set[str] = set()
        deadline = time.monotonic() + 120
        while len(ids) < SAMPLE_LIMIT and time.monotonic() < deadline:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                from confluent_kafka import KafkaError
                if message.error().code() == KafkaError._PARTITION_EOF:
                    break
                raise RuntimeError("source Kafka sampling failed")
            doc_id = parse_message(message.value())
            if doc_id not in seen:
                seen.add(doc_id)
                ids.append(doc_id)
        if len(ids) != SAMPLE_LIMIT:
            raise RuntimeError(f"source topic yielded only {len(ids)} distinct valid IDs")
        return ids, {"topic": SAMPLE_TOPIC, "partition": 0, "low": low, "high": high,
                     "sampled": len(ids), "read_only_no_commit": True}
    finally:
        consumer.close()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def index_params_from_source(client: MilvusClient) -> list[dict[str, object]]:
    result = []
    for name in client.list_indexes(SOURCE_COLLECTION):
        item = client.describe_index(SOURCE_COLLECTION, name)
        transient = {"index_name", "total_rows", "indexed_rows", "pending_index_rows", "state",
                     "index_state", "index_state_fail_reason", "field_name", "index_type", "metric_type"}
        params = {key: value for key, value in item.items() if key not in transient}
        result.append({"field_name": item["field_name"], "index_type": item["index_type"],
                       "metric_type": item.get("metric_type"), "params": params})
    if len(result) != 18 or not set(VECTOR_FIELDS) <= {item.get("field_name") for item in result}:
        raise RuntimeError("231 source collection does not have the expected 18-index contract")
    return result


def create_isolated_collection() -> tuple[MilvusClient, dict[str, object]]:
    admin = MilvusClient(uri=SOURCE_URI, timeout=120)
    source = MilvusClient(uri=SOURCE_URI, db_name=SOURCE_DB, timeout=120)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    database = "metadata_inc_perf_" + suffix.lower()
    collection = "metadata_inc_test_" + suffix.lower()
    try:
        desc = source.describe_collection(SOURCE_COLLECTION)
        if len(desc.get("fields", [])) != 41 or str(desc.get("collection_id")) != "467035880531670094":
            raise RuntimeError("231 source collection identity or schema differs from the approved baseline")
        indexes = index_params_from_source(source)
        source_schema_hash = digest(desc)
        source_indexes_hash = digest(indexes)
        if database in admin.list_databases() or source.has_collection(collection):
            raise RuntimeError("isolated target name already exists; refusing to reuse it")
        admin.create_database(db_name=database)
        target = MilvusClient(uri=SOURCE_URI, db_name=database, timeout=120)
        schema = target.create_schema(auto_id=False, enable_dynamic_field=False)
        for name, spec in FIELDS.items():
            if name == "id":
                schema.add_field(name, DataType.VARCHAR, max_length=spec.max_length, is_primary=True)
            elif spec.kind == "varchar":
                schema.add_field(name, DataType.VARCHAR, max_length=spec.max_length, nullable=True)
            elif spec.kind == "array":
                schema.add_field(name, DataType.ARRAY, element_type=DataType.VARCHAR,
                                 max_capacity=spec.max_capacity, max_length=spec.max_length, nullable=True)
            elif spec.kind == "vector":
                schema.add_field(name, DataType.FLOAT_VECTOR, dim=spec.dim)
        params = target.prepare_index_params()
        for item in indexes:
            params.add_index(field_name=item["field_name"], index_type=item.get("index_type", "AUTOINDEX"),
                             metric_type=item.get("metric_type"), params=item.get("params") or None)
        target.create_collection(collection_name=collection, schema=schema, index_params=params, timeout=300)
        target.load_collection(collection, timeout=300)
        target_desc = target.describe_collection(collection)
        target_indexes = []
        for name in target.list_indexes(collection):
            item = target.describe_index(collection, name)
            transient = {"index_name", "total_rows", "indexed_rows", "pending_index_rows", "state",
                         "index_state", "index_state_fail_reason", "field_name", "index_type", "metric_type"}
            target_indexes.append({"field_name": item["field_name"], "index_type": item["index_type"],
                                   "metric_type": item.get("metric_type"),
                                   "params": {key: value for key, value in item.items() if key not in transient}})
        if (len(target_desc.get("fields", [])) != 41 or len(target_indexes) != 18 or
                digest(sorted(target_indexes, key=lambda item: item["field_name"])) !=
                digest(sorted(indexes, key=lambda item: item["field_name"]))):
            raise RuntimeError("isolated collection schema/index creation failed verification")
        return target, {
            "uri": SOURCE_URI, "database": database, "collection": collection,
            "collection_id": str(target_desc.get("collection_id")), "fields": len(target_desc["fields"]),
            "indexes": len(target_indexes), "source_collection_id": str(desc["collection_id"]),
            "source_schema_hash": source_schema_hash, "source_indexes_hash": source_indexes_hash,
            "target_schema_hash": digest(target_desc), "target_indexes_hash": digest(target_indexes),
        }
    except Exception:
        admin.close()
        source.close()
        raise
    finally:
        admin.close()
        source.close()


def create_test_topic(name: str) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    config = kafka_config()
    admin = AdminClient({key: value for key, value in config.items() if key not in {
        "group.id", "enable.auto.commit", "enable.auto.offset.store", "auto.offset.reset", "max.poll.interval.ms"}})
    metadata = admin.list_topics(timeout=15)
    if name in metadata.topics and metadata.topics[name].error is None:
        raise RuntimeError("unique test Topic unexpectedly already exists")
    admin.create_topics([NewTopic(name, num_partitions=1, replication_factor=3)])[name].result(timeout=30)
    created = admin.list_topics(name, timeout=15).topics[name]
    if len(created.partitions) != 1 or any(len(partition.replicas) != 3 for partition in created.partitions.values()):
        raise RuntimeError("test Topic did not meet the 1-partition/3-replica requirement")


def produce_ids(topic: str, ids: list[str]) -> dict[str, int]:
    from confluent_kafka import Producer

    config = kafka_config()
    producer = Producer({key: value for key, value in config.items() if key not in {
        "group.id", "enable.auto.commit", "enable.auto.offset.store", "auto.offset.reset", "max.poll.interval.ms"}})
    delivered = 0
    last_offset = -1
    failure: list[str] = []

    def callback(error, message):
        nonlocal delivered, last_offset
        if error:
            failure.append(type(error).__name__)
        else:
            delivered += 1
            last_offset = max(last_offset, message.offset())

    for doc_id in ids:
        producer.produce(topic, value=doc_id.encode("ascii"), callback=callback)
        producer.poll(0)
    remaining = producer.flush(60)
    if remaining or failure or delivered != len(ids):
        raise RuntimeError("Kafka test-topic delivery was incomplete")
    return {"delivered": delivered, "last_offset": last_offset}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="read-only dependency and source checks")
    args = parser.parse_args()
    load_credentials()
    ids, source_sampling = sample_ids()
    if args.preflight:
        from incremental_sync.adapters import SolrReader

        client = MilvusClient(uri=SOURCE_URI, db_name=SOURCE_DB, timeout=120)
        solr = SolrReader(os.environ["SOLR_PAPER_URL"], 60)
        try:
            desc = client.describe_collection(SOURCE_COLLECTION)
            indexes = index_params_from_source(client)
            found: set[str] = set()
            for start in range(0, len(ids), 200):
                found.update(solr.fetch_many(ids[start:start + 200]))
            report = {"source_sampling": {**source_sampling, "distinct_ids": len(ids)},
                      "milvus_collection_id": str(desc.get("collection_id")),
                      "milvus_fields": len(desc.get("fields", [])), "milvus_indexes": len(indexes),
                      "solr_found": len(found), "solr_missing": len(ids) - len(found),
                      "preflight_passed": str(desc.get("collection_id")) == "467035880531670094"
                          and len(desc.get("fields", [])) == 41 and len(indexes) == 18}
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if not report["preflight_passed"]:
                raise RuntimeError("231 source collection did not match the expected baseline")
        finally:
            solr.close()
            client.close()
        return

    global OUT
    base_out = OUT
    suffix = 1
    while OUT.exists():
        suffix += 1
        OUT = base_out.with_name(base_out.name + f"_{suffix:02d}")
    OUT.mkdir(parents=True, exist_ok=False)
    state_dir = OUT / "state"
    log_dir = OUT / "logs"
    state_dir.mkdir()
    log_dir.mkdir()

    from incremental_sync.__main__ import configure_logging
    from incremental_sync.store import Ledger, atomic_json
    from incremental_sync.worker import target_identity

    target_client, target = create_isolated_collection()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)
    topic = "NSTL_METADATA_INCREMENTAL_PERF_TEST_" + run_id.upper()
    group = "nstl_metadata_perf_test_" + run_id.lower()
    create_test_topic(topic)
    producer_result = produce_ids(topic, ids)

    env = {
        "SYNC_MODE": "run", "KAFKA_INCREMENTAL_TOPIC": topic, "KAFKA_INCREMENTAL_GROUP_ID": group,
        "MILVUS_URI": SOURCE_URI, "MILVUS_DATABASE": target["database"],
        "MILVUS_COLLECTION": target["collection"], "SYNC_STATE_DIR": str(state_dir),
        "SYNC_LOG_DIR": str(log_dir), "SYNC_BATCH_SIZE": str(SAMPLE_LIMIT),
        "SYNC_BATCH_WAIT_SEC": "0.25", "VECTOR_PARTIAL_UPDATE_VERIFIED": "1",
        "MONGO_DB": "metadata_incremental_perf_test_" + run_id.lower(),
        "MONGO_FAILURE_COLLECTION": "metadata_incremental_sync_failures",
        "REQUEST_TIMEOUT_SEC": "120", "RETRY_WAIT_SEC": "2", "MAX_ATTEMPTS": "1",
    }
    os.environ.update(env)
    settings = Settings.from_env()
    configure_logging(settings)
    atomic_json(OUT / "manifest.json", {
        "run_id": run_id, "sample_ids": ids, "source_sampling": source_sampling,
        "test_topic": topic, "test_group": group, "test_target": target,
        "producer": producer_result, "message_limit": SAMPLE_LIMIT,
        "formal_topic_commits": 0,
    })
    worker = IncrementalWorker(settings)
    thread = threading.Thread(target=worker.run, name="bounded-live-kafka-worker", daemon=True)
    started = time.monotonic()
    thread.start()
    deadline = time.monotonic() + 900
    terminal = 0
    while time.monotonic() < deadline:
        if state_dir.joinpath("safety_pause.json").exists():
            raise RuntimeError("worker entered safety pause; report preserved for inspection")
        db_path = state_dir / "checkpoint.db"
        if db_path.exists():
            with sqlite3.connect(db_path, timeout=2) as conn:
                received = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                terminal = conn.execute("SELECT COUNT(*) FROM events WHERE status='done'").fetchone()[0]
        if terminal == SAMPLE_LIMIT:
            break
        if not thread.is_alive():
            raise RuntimeError("worker stopped before all test messages completed")
        time.sleep(1)
    worker.stop()
    thread.join(timeout=30)
    elapsed = time.monotonic() - started
    if thread.is_alive():
        raise RuntimeError("worker did not stop cleanly after bounded test completion")
    if terminal != SAMPLE_LIMIT or received != SAMPLE_LIMIT:
        raise RuntimeError(f"worker completed {terminal}/{SAMPLE_LIMIT} messages; report preserved")

    target_worker = MilvusTarget(SOURCE_URI, str(target["database"]), str(target["collection"]), "", 120)
    try:
        target_worker.client.flush(str(target["collection"]), timeout=300)
        stored = target_worker.client.query(str(target["collection"]), filter="", output_fields=["count(*)"],
                                            consistency_level="Strong", timeout=120)[0]["count(*)"]
        rows = target_worker.get_many(ids)
        outcomes = {}
        with sqlite3.connect(state_dir / "checkpoint.db") as conn:
            for outcome, count in conn.execute("SELECT outcome,COUNT(*) FROM events GROUP BY outcome"):
                outcomes[outcome or "pending"] = count
        mismatches = [doc_id for doc_id, row in rows.items() if set(row) != set(FIELDS)]
        if stored != SAMPLE_LIMIT or len(rows) != SAMPLE_LIMIT or mismatches:
            raise RuntimeError("Strong readback did not find all 1,000 complete 41-field records")
        summary = {
            "run_id": run_id, "started_at": datetime.fromtimestamp(time.time() - elapsed, timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(elapsed, 3),
            "rows": SAMPLE_LIMIT, "rows_per_second": round(SAMPLE_LIMIT / elapsed, 3),
            "outcomes": outcomes, "strong_readback_rows": len(rows), "milvus_count": stored,
            "schema_fields": 41, "schema_and_index_template": target,
            "topic": topic, "consumer_group": group, "formal_topic_commits": 0,
            "failure_outbox_pending": 0, "all_41_fields_present": not mismatches,
            "mismatches": mismatches, "passed": not mismatches and stored == SAMPLE_LIMIT,
        }
        atomic_json(OUT / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        target_worker.close()
        target_client.close()


if __name__ == "__main__":
    main()
