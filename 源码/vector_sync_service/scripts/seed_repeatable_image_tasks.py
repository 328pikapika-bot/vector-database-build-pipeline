#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager


EXCLUDED_DOC_PREFIX = "loadtest_img_"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed repeatable image load-test tasks from successful historical tasks."
    )
    parser.add_argument("--count", type=int, default=4000, help="How many PENDING tasks to seed.")
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=200,
        help="How many successful source tasks to sample from.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. Defaults to current UTC timestamp.",
    )
    parser.add_argument(
        "--doc-prefix",
        default="loadtest_img",
        help="Synthetic doc_id prefix for seeded tasks.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional JSON report output path.",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    manager = get_mongo_state_manager()
    await manager.connect()
    try:
        collection = manager.image_tasks_collection
        run_id = str(args.run_id or utc_compact())
        seed_docs = list(
            collection.find(
                {
                    "status": "SUCCESS",
                    "relative_url": {"$exists": True, "$ne": ""},
                    "doc_id": {"$not": re.compile(f"^{re.escape(EXCLUDED_DOC_PREFIX)}")},
                },
                {
                    "_id": 1,
                    "task_id": 1,
                    "doc_id": 1,
                    "record_update_time": 1,
                    "relative_url": 1,
                    "mmd_relative_url": 1,
                    "image_id": 1,
                    "sequence": 1,
                    "doc_title": 1,
                    "keyword": 1,
                    "accessType": 1,
                    "classificationCode": 1,
                    "year": 1,
                    "hasAuthorname": 1,
                    "hasCreateOrganizationname": 1,
                    "language": 1,
                    "sysDataStatus": 1,
                    "hasSourcesysSourceId": 1,
                    "hasSourceidSet": 1,
                    "title": 1,
                    "name": 1,
                    "width": 1,
                    "height": 1,
                    "file_type": 1,
                    "size": 1,
                    "raw_img_meta": 1,
                    "img_meta": 1,
                    "context_before": 1,
                    "context_after": 1,
                    "source_item_index": 1,
                    "image_index": 1,
                    "updated_at": 1,
                },
            )
            .sort("updated_at", -1)
            .limit(max(int(args.seed_limit or 1), 1))
        )
        if not seed_docs:
            raise RuntimeError("no successful source image tasks found for seeding")

        record_update_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        docs: list[dict[str, Any]] = []
        for index in range(max(int(args.count or 0), 0)):
            source = seed_docs[index % len(seed_docs)]
            synthetic_doc_id = f"{args.doc_prefix}_{run_id}_{index + 1:06d}"
            sequence = 1
            image_id = f"load-test-image-{run_id}-{index + 1:06d}"
            task_id = sha256(
                f"{run_id}|{index}|{source.get('task_id') or source.get('_id')}".encode("utf-8")
            ).hexdigest()
            now = datetime.now(timezone.utc)

            raw_img_meta = deepcopy(source.get("raw_img_meta") or {})
            raw_img_meta["id"] = image_id
            raw_img_meta["sequence"] = str(sequence)
            raw_img_meta["load_test_run_id"] = run_id
            raw_img_meta["source_task_id"] = str(source.get("task_id") or source.get("_id") or "")

            img_meta = deepcopy(source.get("img_meta") or {})
            img_meta["task_id"] = task_id
            img_meta["load_test_run_id"] = run_id
            img_meta["load_test_seeded"] = True
            img_meta["load_test_source_task_id"] = str(
                source.get("task_id") or source.get("_id") or ""
            )
            img_meta.pop("milvus_batch_id", None)
            img_meta.pop("milvus_batch_size", None)
            img_meta.pop("success_stage", None)
            img_meta.pop("failure_stage", None)
            img_meta.pop("feature_stage_timings_ms", None)
            img_meta.pop("stage_queue_wait_ms", None)
            img_meta.pop("milvus_write_ms", None)

            docs.append(
                {
                    "_id": task_id,
                    "task_id": task_id,
                    "doc_id": synthetic_doc_id,
                    "record_update_time": record_update_time,
                    "relative_url": source.get("relative_url") or "",
                    "mmd_relative_url": source.get("mmd_relative_url") or "",
                    "image_id": image_id,
                    "sequence": str(sequence),
                    "doc_title": deepcopy(source.get("doc_title")),
                    "keyword": deepcopy(source.get("keyword")),
                    "accessType": source.get("accessType"),
                    "classificationCode": source.get("classificationCode"),
                    "year": source.get("year"),
                    "hasAuthorname": deepcopy(source.get("hasAuthorname")),
                    "hasCreateOrganizationname": deepcopy(source.get("hasCreateOrganizationname")),
                    "language": deepcopy(source.get("language")),
                    "sysDataStatus": source.get("sysDataStatus"),
                    "hasSourcesysSourceId": deepcopy(source.get("hasSourcesysSourceId")),
                    "hasSourceidSet": deepcopy(source.get("hasSourceidSet")),
                    "title": source.get("title") or "",
                    "name": source.get("name") or "",
                    "width": source.get("width"),
                    "height": source.get("height"),
                    "file_type": source.get("file_type") or "",
                    "size": source.get("size"),
                    "raw_img_meta": raw_img_meta,
                    "img_meta": img_meta,
                    "context_before": source.get("context_before"),
                    "context_after": source.get("context_after"),
                    "stage_group": "image",
                    "status": "PENDING",
                    "priority": 0,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "next_retry_at": None,
                    "retry_count": 0,
                    "attempt_count": 0,
                    "artifact_refs": [],
                    "result_summary": {},
                    "error_code": None,
                    "error_message": None,
                    "finished_at": None,
                    "retryable": False,
                    "source_item_index": int(source.get("source_item_index") or 0),
                    "image_index": int(source.get("image_index") or 0),
                    "created_at": now,
                    "updated_at": now,
                    "seeded_from_task_id": str(source.get("task_id") or source.get("_id") or ""),
                    "load_test_run_id": run_id,
                }
            )

        if docs:
            collection.insert_many(docs, ordered=False)

        report = {
            "run_id": run_id,
            "seeded_count": len(docs),
            "seed_source_count": len(seed_docs),
            "record_update_time": record_update_time,
            "task_collection": settings.MONGO_IMAGE_TASK_COLLECTION,
            "sample_source_task_ids": [
                str(item.get("task_id") or item.get("_id") or "") for item in seed_docs[:10]
            ],
            "sample_seeded_task_ids": [doc["task_id"] for doc in docs[:10]],
            "sample_seeded_doc_ids": [doc["doc_id"] for doc in docs[:10]],
        }
        if args.report_path:
            path = Path(args.report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        await close_mongo_state_manager()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> int:
    import asyncio

    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
