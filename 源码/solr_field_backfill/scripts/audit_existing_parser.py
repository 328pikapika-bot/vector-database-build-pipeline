"""Read-only comparison of previously filled Milvus fields with v4 Solr parsing.

The checkpoint is opened read-only. This script never submits Milvus writes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests
from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solr_field_backfill.backfill import Settings, TARGET_FIELDS  # noqa: E402
from solr_field_backfill.production_parser import (  # noqa: E402
    PARSER_VERSION,
    SOLR_QUERY_FIELDS,
    parse_all,
)


def checkpoint_ids(path: Path, sample_size: int) -> list[str]:
    db = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        count = db.execute("SELECT count(*) FROM bf_batches WHERE phase='done'").fetchone()[0]
        if count == 0:
            raise RuntimeError("no completed batches in the old checkpoint")
        batch_count = min(count, max(1, (sample_size + 99) // 100))
        offsets = sorted({i * count // batch_count for i in range(batch_count)})
        ids: list[str] = []
        for offset in offsets:
            row = db.execute(
                "SELECT ids FROM bf_batches WHERE phase='done' ORDER BY prefix,number "
                "LIMIT 1 OFFSET ?", (offset,),
            ).fetchone()
            if row is None:
                raise RuntimeError("completed batch disappeared during audit")
            ids.extend(json.loads(row[0])[:100])
        return ids[:sample_size]
    finally:
        db.close()


def audit(checkpoint: Path, sample_size: int) -> dict:
    settings = Settings()
    ids = checkpoint_ids(checkpoint, sample_size)
    milvus = MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name)
    http = requests.Session()
    http.trust_env = False
    counts = {field: Counter() for field in TARGET_FIELDS}
    mismatches: dict[str, list[str]] = defaultdict(list)
    seen = 0
    missing_solr = 0
    try:
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            current_rows = milvus.get(
                collection_name=settings.collection,
                ids=batch,
                output_fields=["id", *TARGET_FIELDS],
                consistency_level="Strong",
                timeout=settings.milvus_timeout,
            )
            current = {str(row["id"]): row for row in current_rows}
            if set(current) != set(batch):
                raise RuntimeError("sampled Milvus IDs changed; audit stopped")
            response = http.post(settings.solr_url, data={
                "q": "{!terms f=id}" + ",".join(batch),
                "fl": ",".join(SOLR_QUERY_FIELDS),
                "rows": len(batch),
                "wt": "json",
            }, timeout=settings.solr_timeout)
            response.raise_for_status()
            body = response.json()
            if body.get("responseHeader", {}).get("status", 0) or body.get("responseHeader", {}).get("partialResults"):
                raise RuntimeError("Solr returned an incomplete response")
            docs = body["response"]["docs"]
            sources = {str(doc["id"]): doc for doc in docs}
            if len(sources) != len(docs) or not set(sources).issubset(batch):
                raise RuntimeError("Solr returned duplicate or unrequested IDs")
            for row_id in batch:
                if row_id not in sources:
                    missing_solr += 1
                    continue
                expected = parse_all(sources[row_id])
                seen += 1
                for field in TARGET_FIELDS:
                    actual = current[row_id].get(field)
                    value = expected[field]
                    if actual is None and value is None:
                        category = "both_empty"
                    elif actual is None:
                        category = "needs_fill"
                    elif value is None:
                        category = "current_nonempty_source_empty"
                    elif str(actual) == value:
                        category = "equal"
                    else:
                        category = "different_nonempty"
                    counts[field][category] += 1
                    if category in {"current_nonempty_source_empty", "different_nonempty"} and len(mismatches[field]) < 10:
                        mismatches[field].append(row_id)
    finally:
        http.close()
        milvus.close()
    return {
        "read_only": True,
        "parser_version": PARSER_VERSION,
        "target": {"uri": settings.milvus_uri, "database": settings.milvus_db_name,
                   "collection": settings.collection},
        "sampled_ids": len(ids),
        "solr_found": seen,
        "solr_missing": missing_solr,
        "field_counts": {field: dict(counts[field]) for field in TARGET_FIELDS},
        "mismatch_example_ids": dict(mismatches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 10000:
        parser.error("sample-size must be between 1 and 10000")
    result = audit(args.checkpoint, args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sampled_ids": result["sampled_ids"], "solr_found": result["solr_found"],
                      "solr_missing": result["solr_missing"],
                      "different_nonempty": {name: counts.get("different_nonempty", 0)
                                             for name, counts in result["field_counts"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
