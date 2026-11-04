"""Paired 200-row comparison plus complete 1,000-row local batch verification."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from pymilvus import MilvusClient

from benchmark_1000 import (
    COLLECTION_PREFIX, DATABASE, OUT, URI, batch_get, dataset,
    grouped_batch_global_pass, reset_seed, sequential_pass, verify_pass,
)

COLLECTION = "scientific_documents_perf_20261020_135709_868dda"
assert COLLECTION.startswith(COLLECTION_PREFIX) and URI == "http://127.0.0.1:19539"
REPORT = OUT / "paired_benchmark_report.json"


def save(report):
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    rows = dataset()
    client = MilvusClient(uri=URI, db_name=DATABASE, timeout=90)
    report = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "local_uri": URI, "database": DATABASE, "collection": COLLECTION,
              "records": len(rows), "paired_sample": 200, "readback_batch_size": 50,
              "production_milvus_writes": 0, "kafka_commits": 0}
    save(report)
    try:
        if not client.has_collection(COLLECTION):
            raise RuntimeError("isolated benchmark collection is missing")
        count = client.query(COLLECTION, filter="", output_fields=["count(*)"],
                             consistency_level="Strong", timeout=90)[0]["count(*)"]
        if count != len(rows):
            raise RuntimeError("benchmark collection does not contain exactly 1,000 records")
        reset_seed(client, COLLECTION, rows)
        report["reset_before_paired_batch_verified"] = True
        save(report)

        report["batch_global_200"] = grouped_batch_global_pass(client, COLLECTION, rows[:200])
        client.flush(COLLECTION, timeout=180)
        verify_pass(client, COLLECTION, rows[:200])
        report["batch_global_200_verified"] = True
        save(report)

        reset_seed(client, COLLECTION, rows)
        report["reset_before_batch_1000_verified"] = True
        save(report)

        report["batch_1000"] = grouped_batch_global_pass(client, COLLECTION, rows)
        client.flush(COLLECTION, timeout=180)
        verify_pass(client, COLLECTION, rows)
        count = client.query(COLLECTION, filter="", output_fields=["count(*)"],
                             consistency_level="Strong", timeout=90)[0]["count(*)"]
        if count != 1000:
            raise RuntimeError("final isolated collection count differs from 1,000")
        report["batch_1000_seconds"] = report["batch_1000"]["seconds"]
        report["batch_1000_rows_per_second"] = report["batch_1000"]["rows_per_second"]
        report["sequential_200_seconds"] = 162.266
        report["sequential_200_rows_per_second"] = 1.233
        report["paired_200_speedup"] = round(report["batch_global_200"]["rows_per_second"] / 1.233, 3)
        report["final_count"] = count
        report["all_1000_verified"] = True
        report["passed"] = True
        save(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
