"""Migrate scientific Milvus collections from the old store to the new store.

The script is intentionally resumable. It writes one checkpoint JSON per
collection and only reads from the source Milvus instance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient

ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = ROOT / "vector_sync_service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings  # noqa: E402
from app.services.milvus_chunk_service import MilvusChunkService  # noqa: E402
from app.services.milvus_image_service import MilvusImageService  # noqa: E402
from app.services.milvus_service import MilvusDynamicService  # noqa: E402


FULLTEXT = "scientific_fulltext_chunks"
IMAGES = "scientific_images"
DOCUMENTS = "scientific_documents"


def _fields(desc: dict[str, Any]) -> list[dict[str, Any]]:
    return list(desc.get("fields") or [])


def _field_name(field: dict[str, Any]) -> str:
    return str(field.get("name") or "")


def _field_type(field: dict[str, Any]) -> Any:
    return field.get("data_type", field.get("type"))


def _vector_fields(desc: dict[str, Any]) -> list[str]:
    vector_types = {
        DataType.FLOAT_VECTOR,
        DataType.FLOAT16_VECTOR,
        DataType.BFLOAT16_VECTOR,
        DataType.BINARY_VECTOR,
        DataType.SPARSE_FLOAT_VECTOR,
    }
    return [_field_name(f) for f in _fields(desc) if _field_type(f) in vector_types and _field_name(f)]


def _output_fields(desc: dict[str, Any]) -> list[str]:
    return [_field_name(f) for f in _fields(desc) if _field_name(f)]


def _primary_key(desc: dict[str, Any]) -> str:
    for field in _fields(desc):
        if field.get("is_primary"):
            return _field_name(field)
    raise RuntimeError("collection has no primary key")


def _stats(client: MilvusClient, collection: str) -> int:
    stats = client.get_collection_stats(collection_name=collection)
    return int(stats.get("row_count") or 0)


def _checkpoint_path(checkpoint_dir: Path, collection: str) -> Path:
    return checkpoint_dir / f"{collection}.json"


def _load_checkpoint(checkpoint_dir: Path, collection: str) -> dict[str, Any]:
    path = _checkpoint_path(checkpoint_dir, collection)
    if not path.exists():
        return {"migrated": 0, "last_pk": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(checkpoint_dir: Path, collection: str, payload: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, collection)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_collections(target_uri: str) -> None:
    MilvusChunkService(uri=target_uri).init_chunk_collection(FULLTEXT)
    MilvusImageService(uri=target_uri).init_image_collection(IMAGES)
    MilvusDynamicService(uri=target_uri).init_collection(DOCUMENTS)


def _probe(target_uri: str) -> dict[str, Any]:
    name = "codex_probe_scientific_milvus"
    client = MilvusClient(uri=target_uri)
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    index = client.prepare_index_params()
    # Keep the smoke collection on the same index family as production.
    # The tiny dimension uses m=2 only so that the PQ sub-vector dimension is
    # valid (the real collections use m=64 with 2560/4096-dimensional vectors).
    index.add_index(
        field_name="vec",
        index_type="IVF_PQ",
        metric_type="COSINE",
        params={"nlist": 1, "m": 2, "nbits": 8},
    )
    client.create_collection(collection_name=name, schema=schema, index_params=index)
    client.load_collection(name)
    client.insert(collection_name=name, data=[{"id": "p1", "vec": [0.1, 0.2, 0.3, 0.4]}])
    result = client.query(collection_name=name, filter='id == "p1"', output_fields=["id"], limit=1)
    client.drop_collection(name)
    return {"query": result}


def _flush_and_load(client: MilvusClient, collection: str) -> None:
    try:
        client.flush(collection_name=collection)
    except Exception:
        pass
    client.load_collection(collection)


def _insert_batch(target: MilvusClient, collection: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    target.insert(collection_name=collection, data=rows)


def migrate_collection(
    source: MilvusClient,
    target: MilvusClient,
    collection: str,
    checkpoint_dir: Path,
    batch_size: int,
    limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    desc = source.describe_collection(collection)
    pk = _primary_key(desc)
    output_fields = _output_fields(desc)
    source_count = _stats(source, collection)
    checkpoint = _load_checkpoint(checkpoint_dir, collection)
    migrated = int(checkpoint.get("migrated") or 0)
    target_count_before = _stats(target, collection) if target.has_collection(collection) else 0

    if dry_run:
        return {
            "collection": collection,
            "primary_key": pk,
            "source_count": source_count,
            "target_count_before": target_count_before,
            "output_field_count": len(output_fields),
            "vector_fields": _vector_fields(desc),
            "checkpoint": checkpoint,
        }

    if target_count_before > migrated:
        migrated = target_count_before

    iterator = source.query_iterator(
        collection_name=collection,
        batch_size=batch_size,
        output_fields=output_fields,
        expr="",
    )
    inserted_this_run = 0
    seen = 0
    last_pk = checkpoint.get("last_pk")
    started = time.time()
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            seen += len(rows)
            if seen <= migrated:
                continue
            if seen - len(rows) < migrated:
                rows = rows[migrated - (seen - len(rows)) :]
            if limit and inserted_this_run >= limit:
                break
            if limit:
                rows = rows[: max(0, limit - inserted_this_run)]
            _insert_batch(target, collection, rows)
            inserted_this_run += len(rows)
            migrated += len(rows)
            if rows:
                last_pk = rows[-1].get(pk)
            _save_checkpoint(
                checkpoint_dir,
                collection,
                {
                    "migrated": migrated,
                    "last_pk": last_pk,
                    "source_count": source_count,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            print(
                json.dumps(
                    {
                        "collection": collection,
                        "migrated": migrated,
                        "source_count": source_count,
                        "inserted_this_run": inserted_this_run,
                        "elapsed_sec": round(time.time() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if limit and inserted_this_run >= limit:
                break
    finally:
        iterator.close()

    try:
        target.flush(collection_name=collection)
    except Exception:
        pass
    return {
        "collection": collection,
        "source_count": source_count,
        "target_count": _stats(target, collection),
        "inserted_this_run": inserted_this_run,
        "checkpoint": _load_checkpoint(checkpoint_dir, collection),
    }


def validate(source: MilvusClient, target: MilvusClient, collections: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for collection in collections:
        item: dict[str, Any] = {}
        item["source_count"] = _stats(source, collection)
        item["target_count"] = _stats(target, collection)
        desc = target.describe_collection(collection)
        pk = _primary_key(desc)
        _flush_and_load(target, collection)
        item["sample"] = target.query(collection_name=collection, filter="", output_fields=[pk], limit=5)
        out[collection] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uri", default="http://127.0.0.1:19531")
    parser.add_argument("--target-uri", default="http://127.0.0.1:19533")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "recovery_logs" / "scientific_milvus_migration"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ensure", action="store_true")
    parser.add_argument("--collections", nargs="+", default=[FULLTEXT, IMAGES])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    source = MilvusClient(uri=args.source_uri)
    target = MilvusClient(uri=args.target_uri)

    if args.probe_only:
        print(json.dumps(_probe(args.target_uri), ensure_ascii=False, default=str))
        return

    if not args.skip_ensure and not args.validate_only and not args.dry_run:
        _ensure_collections(args.target_uri)

    if args.validate_only:
        print(json.dumps(validate(source, target, args.collections), ensure_ascii=False, default=str, indent=2))
        return

    results = []
    for collection in args.collections:
        results.append(
            migrate_collection(
                source=source,
                target=target,
                collection=collection,
                checkpoint_dir=Path(args.checkpoint_dir),
                batch_size=args.batch_size,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        )
    print(json.dumps(results, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
