"""Explicit finite live acceptance: read Kafka without joining/committing the real group.

Run with the normal service environment plus SYNC_STATE_DIR pointing to a NEW directory.
Only the fixed manifest's 100 IDs may be written. Never creates or changes a schema.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incremental_sync.__main__ import configure_logging
from incremental_sync.adapters import SafetyPause, _equal
from incremental_sync.config import Settings
from incremental_sync.contract import FIELDS
from incremental_sync.store import Ledger, atomic_json, utc_now
from incremental_sync.worker import IncrementalWorker, parse_message, target_identity


def normalize(value):
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def index_definition(description):
    """Only schema parameters are stable; indexing progress changes after writes."""
    transient = {"total_rows", "indexed_rows", "pending_index_rows", "state", "index_state",
                 "index_state_fail_reason"}
    return {k: v for k, v in description.items() if k not in transient}


def main():
    from confluent_kafka import Consumer, ConsumerGroupTopicPartitions, TopicPartition
    from confluent_kafka.admin import AdminClient

    settings = Settings.from_env()
    if settings.milvus_uri != "http://127.0.0.1:19533" or settings.milvus_db != "default" or settings.milvus_collection != "scientific_documents_v2":
        raise RuntimeError("this acceptance tool is restricted to the approved 231 test target")
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    if any(settings.state_dir.iterdir()):
        raise RuntimeError("use a new empty acceptance state directory; never overwrite a prior run")
    configure_logging(settings)
    worker = IncrementalWorker(settings)
    worker.connect()
    ledger = None
    sampler = None
    start = time.monotonic()
    try:
        preflight = worker.preflight()
        assert preflight["collection_id"] == "467035880531670094"
        assert preflight["mongo_status"] == "connected", "production Mongo must pass this live acceptance"
        atomic_json(settings.state_dir / "preflight.json", preflight)
        config = settings.kafka_config()
        admin = AdminClient({k: v for k, v in config.items() if k not in {
            "group.id", "enable.auto.commit", "enable.auto.offset.store", "auto.offset.reset", "max.poll.interval.ms"}})

        def offsets():
            result = admin.list_consumer_group_offsets([ConsumerGroupTopicPartitions(settings.kafka_group)],
                request_timeout=15)[settings.kafka_group].result()
            return sorted([{"topic": t.topic, "partition": t.partition, "offset": t.offset}
                for t in result.topic_partitions]) if result.topic_partitions else []

        before_offsets = offsets()
        sampler = Consumer({**config, "group.id": "metadata_acceptance_no_commit_" + uuid.uuid4().hex,
            "enable.partition.eof": True})
        low, high = sampler.get_watermark_offsets(TopicPartition(settings.kafka_topic, 0), timeout=15)
        sampler.assign([TopicPartition(settings.kafka_topic, 0, low)])
        # Do not sample the active or immediately upcoming backfill shards.
        live_state = Path(__file__).resolve().parents[2] / "out/solr_backfill_v4_full_20261020/state/progress.json"
        active = json.loads(live_state.read_text(encoding="utf-8")) if live_state.exists() else {}
        prefix = int(active["current_shard"], 16) if active.get("state") == "running" else -20
        excluded = set(range(prefix, prefix + 16))
        manifest, seen = [], set()
        skipped = 0
        deadline = time.monotonic() + 90
        while len(manifest) < 100 and time.monotonic() < deadline:
            message = sampler.poll(1)
            if message is None:
                continue
            if message.error():
                from confluent_kafka import KafkaError
                if message.error().code() == KafkaError._PARTITION_EOF:
                    break
                raise RuntimeError("Kafka sampling failed")
            doc_id = parse_message(message.value())
            if doc_id in seen or int(doc_id[:2], 16) in excluded:
                skipped += 1
                continue
            seen.add(doc_id)
            manifest.append({"topic": message.topic(), "partition": message.partition(), "offset": message.offset(), "id": doc_id})
        sampler.close()
        sampler = None
        atomic_json(settings.state_dir / "sample_ids.json", {"ids": manifest, "offsets_before": before_offsets,
            "watermarks": [low, high], "no_commit": True, "skipped_duplicates_or_active_shards": skipped,
            "concurrent_backfill_shard": active.get("current_shard")})
        if len(manifest) != 100:
            raise RuntimeError(f"expected 100 distinct real IDs; found {len(manifest)}")
        target = worker.milvus
        client = target.client
        collection = settings.milvus_collection

        def evidence():
            indexes = {name: index_definition(client.describe_index(collection, name))
                       for name in sorted(client.list_indexes(collection))}
            return {"schema_hash": digest(client.describe_collection(collection)),
                "index_definitions": indexes, "indexes_hash": digest(indexes),
                "count": int(client.query(collection, filter="", output_fields=["count(*)"],
                    consistency_level="Strong", timeout=120)[0]["count(*)"])}

        before_evidence = evidence()
        snapshots = {item["id"]: normalize(target.get(item["id"])) for item in manifest}
        controls = client.query(collection, filter='id like "00%"', output_fields=["id"], limit=30,
            consistency_level="Strong", timeout=120)
        controls = {row["id"]: normalize(target.get(row["id"])) for row in controls if row["id"] not in snapshots}
        atomic_json(settings.state_dir / "before.json", {"evidence": before_evidence, "records": snapshots,
            "controls": controls, "at": utc_now()})
        ledger = Ledger(settings.state_dir, target_identity(settings))
        worker.ledger = ledger
        for item in manifest:
            ledger.enqueue(item["topic"], item["partition"], item["offset"], item["id"])
        writes = {}

        class LimitedTarget:
            def check(self): return target.check()
            def get(self, doc_id): return target.get(doc_id)
            def write(self, doc_id, before, payload, *, new):
                if doc_id not in seen:
                    raise SafetyPause("live test write escaped fixed 100-ID manifest")
                writes[doc_id] = normalize(payload)
                atomic_json(settings.state_dir / "write_intents.json", {"payloads": writes})
                target.write(doc_id, before, payload, new=new)

        worker.milvus = LimitedTarget()
        while ledger.counts().get("queued", 0) + ledger.counts().get("processing", 0):
            event = ledger.next_event()
            if event:
                worker._process_event(event)
            else:
                time.sleep(0.1)
            worker.reporter.sync(ledger)
        worker.reporter.sync(ledger, force=True)
        worker._report(force=True)
        target.client.flush(collection, timeout=120)
        mismatches = []
        for item in manifest:
            doc_id = item["id"]
            before, after = snapshots[doc_id], normalize(target.get(doc_id))
            payload = writes.get(doc_id)
            if payload:
                if not after or any(not _equal(after.get(k), v) for k, v in payload.items()):
                    mismatches.append({"id": doc_id, "reason": "payload readback differs"})
                if before and any(not _equal(after.get(k), before.get(k)) for k in FIELDS if k not in payload):
                    mismatches.append({"id": doc_id, "reason": "untouched field changed"})
            elif digest(before) != digest(after):
                mismatches.append({"id": doc_id, "reason": "record changed despite no test write"})
        for doc_id, before in controls.items():
            if digest(before) != digest(target.get(doc_id)):
                mismatches.append({"id": doc_id, "reason": "read-only control changed"})
        after_evidence, after_offsets = evidence(), offsets()
        counts = ledger.counts()
        count_expected = before_evidence["count"] + counts.get("inserted", 0)
        mongo_failures = worker.reporter.collection.count_documents({"task_identity": target_identity(settings)})
        result = {"at": utc_now(), "counts": counts, "seconds": round(time.monotonic()-start, 2),
            "before": before_evidence, "after": after_evidence, "expected_count": count_expected,
            "offsets_before": before_offsets, "offsets_after": after_offsets,
            "formal_group_offsets_unchanged": before_offsets == after_offsets,
            "schema_and_indexes_unchanged": all(before_evidence[k] == after_evidence[k] for k in ("schema_hash", "indexes_hash")),
            "mongo_failure_records": mongo_failures, "mongo_outbox_pending": ledger.outbox_pending(),
            "mismatches": mismatches, "read_only_controls": len(controls),
            "concurrent_backfill": active.get("state") == "running"}
        result["passed"] = not mismatches and result["formal_group_offsets_unchanged"] and result["schema_and_indexes_unchanged"] and after_evidence["count"] == count_expected and ledger.outbox_pending() == 0
        atomic_json(settings.state_dir / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if sampler is not None: sampler.close()
        worker.milvus = target if "target" in locals() else worker.milvus
        worker._close_connections()
        worker.reporter.close()
        if ledger is not None: ledger.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not echo exception text: drivers may embed credentials in diagnostics.
        print(json.dumps({"live_acceptance_failed": type(error).__name__}), flush=True)
        raise SystemExit(1)
