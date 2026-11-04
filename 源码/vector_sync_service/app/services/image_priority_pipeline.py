"""Kafka processor for image-first intake windows."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from pymongo import UpdateOne

from app.config import settings
from app.models.sync_runtime import (
    BatchProcessingResult,
    ConsumedRecord,
    ProcessOutcome,
    SyncOverallStatus,
    normalize_record_update_time,
    utcnow,
)
from app.observability.logger import get_logger
from app.services.image_task_manager import extract_image_tasks
from app.services.image_task_repository import get_image_task_repository
from app.services.message_utils import extract_item_value, infer_data_status
from app.services.mongo_state_manager import get_mongo_state_manager


logger = get_logger(__name__)


class ImagePriorityPipeline:
    """Only discovers image tasks and stores replay metadata for later core backfill."""

    def __init__(self) -> None:
        self.manager = get_mongo_state_manager()
        self.image_tasks = get_image_task_repository()

    async def process_batch(self, batch: list[ConsumedRecord]) -> BatchProcessingResult:
        result = BatchProcessingResult()
        for record in batch:
            if not record.is_valid:
                result.outcomes.append(
                    ProcessOutcome(
                        doc_id=None,
                        record_update_time=None,
                        final_status=SyncOverallStatus.FAILURE.value,
                        commit_safe=True,
                        message="invalid_record_for_image_priority",
                    )
                )
                continue
            outcome = await self._process_record(record)
            result.outcomes.append(outcome)
        return result

    async def _process_record(self, record: ConsumedRecord) -> ProcessOutcome:
        payload = record.parsed_payload or {}
        data = payload.get("data") or []
        if not isinstance(data, list) or not data:
            return ProcessOutcome(
                doc_id=None,
                record_update_time=None,
                final_status=SyncOverallStatus.SKIPPED_STALE.value,
                commit_safe=True,
                message="empty_data_list",
            )

        tasks = extract_image_tasks(payload)
        by_doc: dict[tuple[str, str], dict[str, Any]] = {}
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id") or item.get("sysId") or extract_item_value(item, "sysId") or "")
            record_update_time = normalize_record_update_time(
                extract_item_value(item, "sysDataUpdateTime")
            )
            if not doc_id or not record_update_time:
                continue
            by_doc[(doc_id, record_update_time)] = {
                "doc_id": doc_id,
                "record_update_time": record_update_time,
                "sub_doc_type": str(extract_item_value(item, "sysSubType") or payload.get("subDocType") or ""),
                "event_type": infer_data_status(payload),
                "article_index": index,
                "payload": {**payload, "data": [item]},
                "event_fingerprint": record.event_fingerprint,
                "topic": record.topic,
                "partition": record.partition,
                            "offset": record.offset,
                            "window_id": str(settings.IMAGE_PRIORITY_WINDOW_ID or "default"),
                        }

        tasks_by_doc: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for task in tasks:
            tasks_by_doc[(task.doc_id, task.record_update_time)].append(task)

        await self.image_tasks.upsert_tasks(tasks)
        await self._upsert_window_docs(by_doc, tasks_by_doc)
        success_docs = len(by_doc)
        return ProcessOutcome(
            doc_id=next(iter(by_doc))[0] if by_doc else None,
            record_update_time=next(iter(by_doc))[1] if by_doc else None,
            final_status=SyncOverallStatus.SUCCESS.value,
            commit_safe=True,
            message=f"image_priority_docs={success_docs};image_tasks={len(tasks)}",
        )

    async def _upsert_window_docs(
        self,
        by_doc: dict[tuple[str, str], dict[str, Any]],
        tasks_by_doc: dict[tuple[str, str], list[Any]],
    ) -> None:
        await self.manager.ensure_image_priority_window_indexes()
        now = utcnow()
        operations = []
        for key, item in by_doc.items():
            doc_id, record_update_time = key
            tasks = tasks_by_doc.get(key, [])
            operations.append(
                UpdateOne(
                    {"_id": f"{doc_id}|{record_update_time}"},
                    {
                        "$setOnInsert": {
                            "_id": f"{doc_id}|{record_update_time}",
                            "doc_id": doc_id,
                            "record_update_time": record_update_time,
                            "created_at": now,
                        },
                        "$set": {
                            "sub_doc_type": item["sub_doc_type"],
                            "event_type": item["event_type"],
                            "article_index": item["article_index"],
                            "payload": item["payload"],
                            "event_fingerprint": item["event_fingerprint"],
                            "topic": item["topic"],
                            "partition": item["partition"],
                            "offset": item["offset"],
                            "window_id": item["window_id"],
                            "image_task_count": len(tasks),
                            "image_task_ids": [task.task_id for task in tasks],
                            "image_ids": [task.image_id for task in tasks],
                            "status": "IMAGE_DISCOVERED" if tasks else "NO_IMAGES",
                            "image_ready_at": None,
                            "metadata_backfill_target": None,
                            "metadata_backfill_status": "PENDING",
                            "metadata_backfill_error": None,
                            "fulltext_backfill_target": None,
                            "fulltext_backfill_status": "PENDING",
                            "fulltext_backfill_error": None,
                            "updated_at": now,
                        },
                    },
                    upsert=True,
                )
            )
        if operations:
            await self.manager.bulk_write_image_priority_windows(operations)


_image_priority_pipeline: ImagePriorityPipeline | None = None


async def process_batch_image_priority(batch: list[ConsumedRecord]) -> BatchProcessingResult:
    global _image_priority_pipeline
    if _image_priority_pipeline is None:
        _image_priority_pipeline = ImagePriorityPipeline()
    return await _image_priority_pipeline.process_batch(batch)
