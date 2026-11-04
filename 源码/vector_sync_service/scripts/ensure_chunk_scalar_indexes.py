"""Ensure approved scalar indexes on the Milvus fulltext chunk collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.services.milvus_chunk_service import DEFAULT_CHUNK_COLLECTION, MilvusChunkService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure fulltext chunk scalar indexes in Milvus.")
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHUNK_COLLECTION,
        help=f"Milvus chunk collection name. Defaults to {DEFAULT_CHUNK_COLLECTION!r}.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create missing indexes. Without this flag the command only reports the plan.",
    )
    args = parser.parse_args()

    service = MilvusChunkService()
    try:
        result = service.ensure_chunk_scalar_indexes(
            collection_name=args.collection,
            dry_run=not args.apply,
        )
    finally:
        service.close()

    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
