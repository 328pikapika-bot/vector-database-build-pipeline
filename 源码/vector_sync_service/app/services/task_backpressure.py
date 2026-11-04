"""Shared helpers for task backlog based backpressure decisions."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from app.config import settings
from app.services.mongo_state_manager import (
    PIPELINE_STATUS_PENDING,
    PIPELINE_STATUS_PROCESSING,
)


IMAGE_PRIORITY_MODES = {"image", "image_only", "image_priority", "image_core"}


def count_actionable_backlog(
    mongo,
    *,
    mode: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    """Count queues that workers can actually drain.

    The document status table may contain historical PROCESSING rows after async
    task creation. Blocking on that field can deadlock publishing/consuming, so
    backpressure is based on full-text/image task queues first.
    """

    db = mongo.db
    now = datetime.now(timezone.utc)
    detail: dict[str, Any] = {}

    def count_task_backlog(collection_name: str) -> int:
        collection = db[collection_name]
        return int(
            collection.count_documents(
                {
                    "$or": [
                        {"status": {"$in": ["PENDING", "LEASED"]}},
                        {
                            "status": "FAILURE",
                            "retryable": True,
                            "next_retry_at": {"$lte": now},
                        },
                    ]
                }
            )
        )

    def count_collection_documents(collection_name: str) -> int:
        return int(db[collection_name].count_documents({}))

    def count_held_history(collection_name: str) -> int:
        return int(db[collection_name].count_documents({"status": "HELD_HISTORY"}))

    resolved_mode = str(
        mode
        or getattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "all_tasks")
        or "all_tasks"
    ).strip().lower()

    fulltext_count = count_task_backlog(settings.MONGO_FULLTEXT_TASK_COLLECTION)
    image_count = count_task_backlog(settings.MONGO_IMAGE_TASK_COLLECTION)
    fulltext_total = count_collection_documents(settings.MONGO_FULLTEXT_TASK_COLLECTION)
    image_total = count_collection_documents(settings.MONGO_IMAGE_TASK_COLLECTION)
    image_doc_lead = int(
        mongo.status_collection.count_documents(
            {
                "core_status": "SUCCESS",
                "image_required": True,
                "image_status": {"$in": [PIPELINE_STATUS_PENDING, PIPELINE_STATUS_PROCESSING]},
            }
        )
    )
    fulltext_doc_lead = int(
        mongo.status_collection.count_documents(
            {
                "document_status": "SUCCESS",
                "fulltext_required": True,
                "fulltext_status": {"$in": [PIPELINE_STATUS_PENDING, PIPELINE_STATUS_PROCESSING]},
            }
        )
    )

    detail["backpressure_mode"] = resolved_mode
    detail["fulltext_tasks"] = fulltext_count
    detail["image_tasks"] = image_count
    detail["image_doc_lead"] = image_doc_lead
    detail["fulltext_doc_lead"] = fulltext_doc_lead
    detail["image_doc_lead_limit"] = int(
        getattr(settings, "KAFKA_BACKPRESSURE_IMAGE_DOC_LEAD_LIMIT", 0) or 0
    )
    detail["fulltext_doc_lead_limit"] = int(
        getattr(settings, "KAFKA_BACKPRESSURE_FULLTEXT_DOC_LEAD_LIMIT", 0) or 0
    )
    detail["image_pending_soft_limit"] = int(
        getattr(settings, "KAFKA_BACKPRESSURE_IMAGE_PENDING_SOFT_LIMIT", 0) or 0
    )
    detail["image_pending_resume_threshold"] = int(
        getattr(settings, "KAFKA_BACKPRESSURE_IMAGE_PENDING_RESUME_THRESHOLD", 0) or 0
    )
    detail["fulltext_held_history"] = count_held_history(
        settings.MONGO_FULLTEXT_TASK_COLLECTION
    )
    detail["image_held_history"] = count_held_history(
        settings.MONGO_IMAGE_TASK_COLLECTION
    )

    if resolved_mode in IMAGE_PRIORITY_MODES:
        return image_count, detail

    total = fulltext_count + image_count
    if total > 0:
        return total, detail

    if fulltext_total > 0 or image_total > 0:
        return 0, detail

    status_count = int(
        mongo.status_collection.count_documents(
            {"overall_status": {"$in": ["PENDING", "PROCESSING"]}}
        )
    )
    detail["status_processing_fallback"] = status_count
    return status_count, detail
