from __future__ import annotations

import pytest

from app.models.sync_runtime import ConsumedRecord
from app.services import image_priority_pipeline as pipeline_module


class _FakeImageTaskRepo:
    def __init__(self) -> None:
        self.tasks = []

    async def upsert_tasks(self, tasks):
        self.tasks.extend(tasks)
        return {"matched": 0, "upserted": len(tasks)}


class _FakeManager:
    def __init__(self) -> None:
        self.indexes_called = 0
        self.bulk_ops = []

    async def ensure_image_priority_window_indexes(self):
        self.indexes_called += 1

    async def bulk_write_image_priority_windows(self, operations):
        self.bulk_ops.extend(operations)


def _record(payload: dict) -> ConsumedRecord:
    return ConsumedRecord(
        topic="VECTOR_JOURNAL_ARTICLE",
        partition=1,
        offset=7,
        raw_bytes=b"{}",
        raw_text="{}",
        event_fingerprint="topic|1|7|hash",
        parsed_payload=payload,
    )


@pytest.mark.asyncio
async def test_image_priority_pipeline_discovers_images_and_persists_window(monkeypatch):
    fake_repo = _FakeImageTaskRepo()
    fake_manager = _FakeManager()

    monkeypatch.setattr(pipeline_module, "get_image_task_repository", lambda: fake_repo)
    monkeypatch.setattr(pipeline_module, "get_mongo_state_manager", lambda: fake_manager)
    monkeypatch.setattr(pipeline_module, "settings", type("S", (), {"IMAGE_PRIORITY_WINDOW_ID": "win-a"})())
    monkeypatch.setattr(
        pipeline_module,
        "extract_image_tasks",
        lambda payload: [
            type("T", (), {"task_id": "task-1", "doc_id": "doc-1", "record_update_time": "2026-06-18T00:00:00+00:00", "image_id": "img-1"})()
        ],
    )

    pipeline = pipeline_module.ImagePriorityPipeline()
    result = await pipeline.process_batch(
        [_record({"data": [{"id": "doc-1", "sysDataUpdateTime": "2026-06-18T00:00:00+00:00"}]})]
    )

    assert result.success == 1
    assert fake_repo.tasks
    assert fake_manager.indexes_called == 1
    assert fake_manager.bulk_ops
    stored = fake_manager.bulk_ops[0]._doc
    assert stored["$set"]["window_id"] == "win-a"
    assert stored["$set"]["status"] == "IMAGE_DISCOVERED"
    assert stored["$set"]["metadata_backfill_status"] == "PENDING"
    assert stored["$set"]["fulltext_backfill_status"] == "PENDING"
