"""Isolated Milvus capability test. Creates one NEW collection; never drops anything.

This probe must pass on the target Milvus version before setting
VECTOR_PARTIAL_UPDATE_VERIFIED=1 for the incremental worker.
"""

from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime, timezone

from pymilvus import DataType, MilvusClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--database", default="default")
    parser.add_argument("--token", default="")
    args = parser.parse_args()
    name = "incremental_vector_probe_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)
    options = {"uri": args.uri, "db_name": args.database, "timeout": 60}
    if args.token:
        options["token"] = args.token
    client = MilvusClient(**options)
    assert name.startswith("incremental_vector_probe_")
    assert not client.has_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("historical", DataType.VARCHAR, max_length=64, nullable=True)
    for field in ("title_embed", "abstract_embed", "keyword_embed"):
        schema.add_field(field, DataType.FLOAT_VECTOR, dim=4)
    index = client.prepare_index_params()
    for field in ("title_embed", "abstract_embed", "keyword_embed"):
        index.add_index(field_name=field, index_type="FLAT", metric_type="COSINE")
    client.create_collection(collection_name=name, schema=schema, index_params=index)
    client.load_collection(name)
    old = {"id": "probe", "historical": "do-not-change",
           "title_embed": [1.0, 0.0, 0.0, 0.0],
           "abstract_embed": [0.0, 1.0, 0.0, 0.0],
           "keyword_embed": [0.0, 0.0, 1.0, 0.0]}
    client.insert(collection_name=name, data=[old])
    client.upsert(collection_name=name, data=[{"id": "probe", "title_embed": [0.0, 0.0, 0.0, 1.0]}],
                  partial_update=True)
    rows = client.get(collection_name=name, ids=["probe"],
                      output_fields=list(old), consistency_level="Strong")
    passed = len(rows) == 1 and rows[0]["historical"] == old["historical"] and \
        list(rows[0]["abstract_embed"]) == old["abstract_embed"] and \
        list(rows[0]["keyword_embed"]) == old["keyword_embed"] and \
        list(rows[0]["title_embed"]) == [0.0, 0.0, 0.0, 1.0]
    print(json.dumps({"passed": bool(passed), "probe_collection": name,
                      "note": "Probe collection is intentionally retained; scientific_documents_v2 was untouched."}))
    client.close()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
