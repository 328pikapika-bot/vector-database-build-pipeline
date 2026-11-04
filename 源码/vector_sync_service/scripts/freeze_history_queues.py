#!/usr/bin/env python3
"""Export and freeze historical retry queues before restarting TXT sync.

The script is intentionally scoped to MongoDB state only. It does not touch
Milvus data, collection schemas, Kafka offsets, or auto-publisher checkpoints.
By default it runs in dry-run mode; pass --apply to mark exported rows as
HELD_HISTORY.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.models.sync_runtime import FailureStatus  # noqa: E402
from app.services.fulltext_task_manager import FullTextTaskStatus  # noqa: E402
from app.services.image_task_manager import ImageTaskStatus  # noqa: E402


HELD_REASON = "freeze_history_before_txt_vector_sync_restart"


@dataclass(frozen=True)
class FreezeTarget:
    name: str
    collection_name: str
    filter_doc: dict[str, Any]
    update_doc: dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def write_jsonl(path: Path, docs: list[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False, default=json_default))
            handle.write("\n")
            count += 1
    return count


def build_fulltext_filter(freeze_at: datetime) -> dict[str, Any]:
    return {
        "$and": [
            _created_before_or_missing(freeze_at),
            {
                "$or": [
                    {"status": FullTextTaskStatus.PENDING.value},
                    {"status": FullTextTaskStatus.LEASED.value},
                    {
                        "status": FullTextTaskStatus.FAILURE.value,
                        "retryable": True,
                        "$or": [
                            {"next_retry_at": {"$lte": freeze_at}},
                            {"next_retry_at": None},
                            {"next_retry_at": {"$exists": False}},
                        ],
                    },
                ]
            },
        ]
    }


def build_image_filter(freeze_at: datetime) -> dict[str, Any]:
    return {
        "$and": [
            _created_before_or_missing(freeze_at),
            {
                "$or": [
                    {"status": ImageTaskStatus.PENDING.value},
                    {"status": ImageTaskStatus.LEASED.value},
                    {
                        "status": ImageTaskStatus.FAILURE.value,
                        "retryable": True,
                        "$or": [
                            {"next_retry_at": {"$lte": freeze_at}},
                            {"next_retry_at": None},
                            {"next_retry_at": {"$exists": False}},
                        ],
                    },
                ]
            },
        ]
    }


def build_failure_filter(freeze_at: datetime) -> dict[str, Any]:
    return {
        **_created_before_or_missing(freeze_at),
        "status": {
            "$in": [
                FailureStatus.PENDING.value,
                FailureStatus.RETRYING.value,
            ]
        },
    }


def _created_before_or_missing(freeze_at: datetime) -> dict[str, Any]:
    return {
        "$or": [
            {"created_at": {"$lte": freeze_at}},
            {"created_at": {"$exists": False}},
        ]
    }


def build_targets(freeze_at: datetime) -> list[FreezeTarget]:
    common_task_set = {
        "status": "HELD_HISTORY",
        "claim_rank": 9,
        "held_reason": HELD_REASON,
        "held_at": freeze_at,
        "lease_owner": None,
        "lease_expires_at": None,
        "next_retry_at": None,
        "updated_at": freeze_at,
    }
    return [
        FreezeTarget(
            name="doc_fulltext_tasks",
            collection_name=settings.MONGO_FULLTEXT_TASK_COLLECTION,
            filter_doc=build_fulltext_filter(freeze_at),
            update_doc={"$set": common_task_set},
        ),
        FreezeTarget(
            name="doc_image_tasks",
            collection_name=settings.MONGO_IMAGE_TASK_COLLECTION,
            filter_doc=build_image_filter(freeze_at),
            update_doc={
                "$set": {
                    **common_task_set,
                    "claim_rank": 9,
                }
            },
        ),
        FreezeTarget(
            name="doc_sync_failures",
            collection_name=settings.MONGO_FAILURE_COLLECTION,
            filter_doc=build_failure_filter(freeze_at),
            update_doc={
                "$set": {
                    "status": FailureStatus.HELD_HISTORY.value,
                    "held_reason": HELD_REASON,
                    "held_at": freeze_at,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "next_retry_at": None,
                    "updated_at": freeze_at,
                }
            },
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export and optionally freeze historical fulltext/image/failure queues "
            "as HELD_HISTORY before restarting the TXT vector sync service."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply HELD_HISTORY updates after exporting. Default is dry-run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "recovery_logs",
        help="Directory for JSONL exports and manifest.",
    )
    parser.add_argument("--mongo-uri", default=settings.MONGO_URI)
    parser.add_argument("--mongo-db", default=settings.MONGO_DB)
    parser.add_argument("--limit", type=int, default=0, help="Optional export limit per collection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze_at = utc_now()
    stamp = freeze_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir / f"freeze_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = MongoClient(
        args.mongo_uri,
        serverSelectionTimeoutMS=settings.MONGO_SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=settings.MONGO_CONNECT_TIMEOUT_MS,
        socketTimeoutMS=settings.MONGO_SOCKET_TIMEOUT_MS,
    )
    client.admin.command("ping")
    db = client[args.mongo_db]

    manifest: dict[str, Any] = {
        "freeze_at": freeze_at.isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "mongo_db": args.mongo_db,
        "held_reason": HELD_REASON,
        "targets": {},
    }

    for target in build_targets(freeze_at):
        collection = db[target.collection_name]
        cursor = collection.find(target.filter_doc, no_cursor_timeout=True).batch_size(500)
        if args.limit > 0:
            cursor = cursor.limit(args.limit)
        export_path = out_dir / f"{target.name}.jsonl"
        exported_count = write_jsonl(export_path, cursor)
        matched_count = collection.count_documents(target.filter_doc)
        modified_count = 0
        if args.apply and args.limit <= 0:
            result = collection.update_many(target.filter_doc, target.update_doc)
            modified_count = int(getattr(result, "modified_count", 0))
        elif args.apply and args.limit > 0 and exported_count:
            ids = []
            with export_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        ids.append(json.loads(line).get("_id"))
            result = collection.update_many({"_id": {"$in": ids}}, target.update_doc)
            modified_count = int(getattr(result, "modified_count", 0))

        manifest["targets"][target.name] = {
            "collection": target.collection_name,
            "export_path": str(export_path),
            "exported": exported_count,
            "matched": int(matched_count),
            "modified": modified_count,
            "filter": target.filter_doc,
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
