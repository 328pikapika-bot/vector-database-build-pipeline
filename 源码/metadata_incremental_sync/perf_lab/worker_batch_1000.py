"""Exercise the real incremental Worker against the isolated local Milvus only."""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from incremental_sync.adapters import MilvusTarget  # noqa: E402
from incremental_sync.config import Settings  # noqa: E402
from incremental_sync.contract import (  # noqa: E402
    FIELDS, SOLR_ARRAYS, SOLR_SCALARS, VECTOR_FIELDS,
)
from incremental_sync.store import Ledger  # noqa: E402
from incremental_sync.worker import IncrementalWorker, target_identity  # noqa: E402
from benchmark_1000 import OUT, URI, DATABASE, COLLECTION_PREFIX, dataset, reset_seed, verify_pass

logging.basicConfig(level=logging.INFO, format="%(message)s")


COLLECTION = "scientific_documents_perf_20261020_135709_868dda"
ADDED_SCALARS = {
    "volume": ("stringIMS_volume", True),
    "issue": ("stringIMS_issue", True),
    "startPage": ("stringIS_startPage", False),
    "endPage": ("stringIS_endPage", False),
    "pageCount": ("stringIS_pageCount", False),
    "createTime": ("dateIS_createTime", False),
    "publisher": ("auto_stringITMS_publisher", False),
    "labelEvaluateGrade": ("stringIMS_labelEvaluateGrade", True),
    "sysProcessingUnit": ("stringIS_sysProcessingUnit", False),
    "hasHoldinglibraryCode": ("stringIMS_hasHoldinglibraryCode", True),
    "entityType": ("stringIMS_entityType", True),
}
STRUCTURED_FIELDS = {
    "hasSource_str": "stringOM_hasSource",
    "hasFile_str": "stringOM_hasFile",
    "hasAuthor_str": "stringOM_hasAuthor",
    "hasCreateOrganization_str": "stringOM_hasCreateOrganization",
}


def _source_doc(item: dict[str, Any]) -> dict[str, Any]:
    values = item["values"]
    doc: dict[str, Any] = {"id": item["id"]}
    for name, source in SOLR_SCALARS.items():
        if name in values:
            doc[source] = values[name]
    for name, source in SOLR_ARRAYS.items():
        if name in values:
            raw = values[name] if isinstance(values[name], list) else [values[name]]
            doc[source] = ["en|" + value for value in raw]
    for name, (source, strip_language) in ADDED_SCALARS.items():
        if name in values:
            value = values[name]
            doc[source] = "en|" + value if strip_language else value
    for name, source in STRUCTURED_FIELDS.items():
        if name in values:
            doc[source] = json.loads(values[name])
    for name, source in (("hasSource", "auto_stringITMS_hasSourcetitle"),):
        if name in values and "stringOM_hasSource" not in doc:
            doc[source] = ["en|" + values[name]]
    if "hasFullTextFilename" in values or "hasFullTextFileid" in values:
        names = values.get("hasFullTextFilename", [])
        ids = values.get("hasFullTextFileid", [])
        doc["stringOM_hasFullTextFile"] = [
            {"id": ids[index] if index < len(ids) else "",
             "name": names[index] if index < len(names) else "",
             "url": f"https://local.invalid/{names[index] if index < len(names) else 'file'}.mmd"}
            for index in range(max(len(names), len(ids)))
        ]
    return doc


class PreparedSolr:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.documents = {row["id"]: _source_doc(row) for row in rows}
        self.calls = 0

    def fetch_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        self.calls += 1
        return {doc_id: self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents}


class CachedEmbedder:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.vectors = {row["id"]: row["vectors"] for row in rows}
        self.calls = 0

    def embed_many(self, documents: dict[str, dict[str, str]]) -> dict[str, dict[str, list[float]]]:
        self.calls += 1
        return {
            doc_id: {field: self.vectors[doc_id][field] for field in VECTOR_FIELDS
                     if field.removesuffix("_embed") in inputs and field in self.vectors[doc_id]}
            for doc_id, inputs in documents.items()
        }


def _settings(state: Path, size: int) -> Settings:
    return Settings(
        mode="run", kafka_brokers="local-not-used", kafka_topic="local-perf-only",
        kafka_group="local-perf-only", kafka_security_protocol="PLAINTEXT",
        kafka_sasl_mechanism="PLAIN", kafka_username="", kafka_password="",
        solr_url="http://local.invalid/solr/paper", embedding_url="http://local.invalid/embed",
        embedding_model="cached", embedding_key="", milvus_uri=URI, milvus_db=DATABASE,
        milvus_collection=COLLECTION, milvus_token="", state_dir=state,
        log_dir=state / "logs", request_timeout=90, retry_wait_sec=0.01,
        idle_window_sec=1, vector_update_verified=True, batch_size=size, batch_wait_sec=0,
    )


def _count(client: MilvusClient) -> int:
    return int(client.query(COLLECTION, filter="", output_fields=["count(*)"],
                            consistency_level="Strong", timeout=90)[0]["count(*)"])


def main() -> None:
    if URI != "http://127.0.0.1:19539" or not COLLECTION.startswith(COLLECTION_PREFIX):
        raise RuntimeError("refusing to run outside the isolated loopback benchmark collection")
    rows = dataset()
    admin = MilvusClient(uri=URI, db_name=DATABASE, timeout=90)
    target = MilvusTarget(URI, DATABASE, COLLECTION, "", 90)
    run_id = datetime.now(timezone.utc).strftime("worker_batch_%Y%m%dT%H%M%SZ")
    state = OUT / ("state_" + run_id)
    state.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(state, run_id)
    solr, embedder = PreparedSolr(rows), CachedEmbedder(rows)
    settings = _settings(state, 1000)
    worker = IncrementalWorker(settings, solr=solr, embedder=embedder, milvus=target)
    worker.ledger = ledger
    worker.solr, worker.embedder, worker.milvus = solr, embedder, target
    worker.collection_id = target.check()["collection_id"]
    report_path = OUT / (run_id + ".json")
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "IncrementalWorker.process_batch with prepared real-document source values and cached vectors",
        "milvus_uri": URI, "database": DATABASE, "collection": COLLECTION,
        "collection_id_before": worker.collection_id,
        "documents": len(rows), "batch_size": settings.batch_size,
        "schema_fields": len(FIELDS), "kafka_consumed": 0, "kafka_commits": 0,
        "production_milvus_writes": 0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if not admin.has_collection(COLLECTION) or _count(admin) != len(rows):
            raise RuntimeError("local benchmark collection identity/count check failed")
        report["count_before"] = _count(admin)
        reset_seed(admin, COLLECTION, rows)
        report["reset_seed_verified"] = True
        for offset, item in enumerate(rows, 1):
            ledger.enqueue(settings.kafka_topic, 0, offset, item["id"])

        started = time.monotonic()
        batch_sizes = []
        batch_metrics = []
        while events := ledger.next_events(settings.batch_size):
            batch_sizes.append(len(events))
            worker.process_batch(events)
            batch_metrics.append(worker.latest_batch_metrics)
        elapsed = time.monotonic() - started
        admin.flush(COLLECTION, timeout=180)
        verify_pass(admin, COLLECTION, rows)
        report.update({
            "elapsed_seconds": round(elapsed, 3),
            "rows_per_second": round(len(rows) / elapsed, 3),
            "batch_sizes": batch_sizes,
            "batch_metrics": batch_metrics,
            "source_batch_calls": solr.calls,
            "embedding_batch_calls": embedder.calls,
            "ledger_counts": ledger.counts(),
            "count_after": _count(admin),
            "collection_id_after": target.check()["collection_id"],
            "all_41_fields_verified": True,
            "passed": True,
        })
        if report["count_after"] != report["count_before"]:
            raise RuntimeError("local test record count changed")
        if report["collection_id_after"] != report["collection_id_before"]:
            raise RuntimeError("local test collection identity changed")
        if report["ledger_counts"] != {"updated": len(rows)}:
            raise RuntimeError("worker ledger did not confirm every sample update")
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ledger.close()
        target.close()
        admin.close()
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
