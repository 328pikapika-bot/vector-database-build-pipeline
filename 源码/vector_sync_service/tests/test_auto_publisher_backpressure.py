from __future__ import annotations

from datetime import datetime, timezone
import json

import httpx
import pytest

from app.config import settings
from app.services.auto_publisher import AutoPublishController, _count_actionable_backlog


class _FakeCollection:
    def __init__(self, docs_or_count) -> None:
        if isinstance(docs_or_count, int):
            self.docs = [{"status": "PENDING"} for _ in range(docs_or_count)]
        else:
            self.docs = [dict(item) for item in docs_or_count]

    def count_documents(self, query: dict) -> int:
        return sum(1 for doc in self.docs if _matches(doc, query or {}))


class _FakeDb:
    def __init__(self, collections: dict[str, int | list[dict]]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> _FakeCollection:
        return _FakeCollection(self.collections.get(name, 0))


class _FakeMongo:
    def __init__(
        self,
        collections: dict[str, int | list[dict]],
        status_count: int | list[dict] = 0,
    ) -> None:
        self.db = _FakeDb(collections)
        self.status_collection = _FakeCollection(status_count)


def test_image_priority_backpressure_ignores_fulltext_backlog(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "image_priority")
    mongo = _FakeMongo(
        {
            settings.MONGO_FULLTEXT_TASK_COLLECTION: 100_000,
            settings.MONGO_IMAGE_TASK_COLLECTION: 123,
        },
        status_count=999_999,
    )

    count, detail = _count_actionable_backlog(mongo)

    assert count == 123
    assert detail["backpressure_mode"] == "image_priority"
    assert detail["fulltext_tasks"] == 100_000
    assert detail["image_tasks"] == 123


def test_all_tasks_backpressure_keeps_existing_fulltext_plus_image_behavior(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "all_tasks")
    mongo = _FakeMongo(
        {
            settings.MONGO_FULLTEXT_TASK_COLLECTION: 7,
            settings.MONGO_IMAGE_TASK_COLLECTION: 5,
        },
        status_count=999,
    )

    count, detail = _count_actionable_backlog(mongo)

    assert count == 12
    assert detail["backpressure_mode"] == "all_tasks"


def test_backpressure_excludes_held_history_and_frozen_failures(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "all_tasks")
    due = datetime(2999, 1, 1, tzinfo=timezone.utc)
    mongo = _FakeMongo(
        {
            settings.MONGO_FULLTEXT_TASK_COLLECTION: [
                {"status": "PENDING"},
                {"status": "LEASED"},
                {"status": "HELD_HISTORY"},
                {"status": "FAILURE", "retryable": True, "next_retry_at": due},
            ],
            settings.MONGO_IMAGE_TASK_COLLECTION: [
                {"status": "PENDING"},
                {"status": "HELD_HISTORY"},
                {"status": "FAILURE", "retryable": False, "next_retry_at": due},
            ],
        },
        status_count=[
            {"overall_status": "PROCESSING"},
            {"overall_status": "PENDING"},
        ],
    )

    count, detail = _count_actionable_backlog(mongo)

    assert count == 3
    assert detail["fulltext_tasks"] == 2
    assert detail["image_tasks"] == 1
    assert "status_processing_fallback" not in detail


def test_backpressure_fallback_excludes_frozen_failures_when_no_task_backlog(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "all_tasks")
    mongo = _FakeMongo(
        {
            settings.MONGO_FULLTEXT_TASK_COLLECTION: [
                {"status": "HELD_HISTORY"},
                {"status": "SUCCESS"},
            ],
            settings.MONGO_IMAGE_TASK_COLLECTION: [
                {"status": "HELD_HISTORY"},
                {"status": "SKIPPED"},
            ],
        },
        status_count=[
            {"overall_status": "PROCESSING"},
            {"overall_status": "PENDING"},
            {"overall_status": "SUCCESS"},
        ],
    )

    count, detail = _count_actionable_backlog(mongo)

    assert count == 0
    assert detail["fulltext_tasks"] == 0
    assert detail["image_tasks"] == 0
    assert detail["fulltext_held_history"] == 1
    assert detail["image_held_history"] == 1
    assert "status_processing_fallback" not in detail


def test_backpressure_uses_status_fallback_only_before_task_collections_exist(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "all_tasks")
    mongo = _FakeMongo(
        {
            settings.MONGO_FULLTEXT_TASK_COLLECTION: [],
            settings.MONGO_IMAGE_TASK_COLLECTION: [],
        },
        status_count=[
            {"overall_status": "PROCESSING"},
            {"overall_status": "PENDING"},
            {"overall_status": "SUCCESS"},
        ],
    )

    count, detail = _count_actionable_backlog(mongo)

    assert count == 2
    assert detail["status_processing_fallback"] == 2


def test_backpressure_reports_document_lead_counts(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_BACKPRESSURE_MODE", "image_priority")
    mongo = _FakeMongo(
        {
            settings.MONGO_FULLTEXT_TASK_COLLECTION: [],
            settings.MONGO_IMAGE_TASK_COLLECTION: [],
        },
        status_count=[
            {
                "core_status": "SUCCESS",
                "image_required": True,
                "image_status": "PENDING",
                "document_status": "SUCCESS",
                "fulltext_required": True,
                "fulltext_status": "PROCESSING",
            },
            {
                "core_status": "SUCCESS",
                "image_required": True,
                "image_status": "PROCESSING",
                "document_status": "SUCCESS",
                "fulltext_required": False,
                "fulltext_status": "NOT_PRESENT",
            },
        ],
    )

    _count_actionable_backlog(mongo)
    count, detail = _count_actionable_backlog(mongo)

    assert count == 0
    assert detail["image_doc_lead"] == 2
    assert detail["fulltext_doc_lead"] == 1


class _TimeoutHttpClient:
    async def post(self, *_args, **_kwargs):
        raise httpx.ReadTimeout("timed out")


class _CapturePostHttpClient:
    def __init__(self) -> None:
        self.content = b""
        self.headers = {}

    async def post(self, *_args, **kwargs):
        self.content = kwargs.get("content", b"")
        self.headers = kwargs.get("headers", {})
        return httpx.Response(
            200,
            json={"code": 0, "data": {"successCount": 1000}},
        )


class _RuntimeControl:
    async def status(self):
        return {"paused": False, "draining": False}


class _ReconnectableMongo(_FakeMongo):
    def __init__(self) -> None:
        super().__init__(
            {
                settings.MONGO_FULLTEXT_TASK_COLLECTION: [],
                settings.MONGO_IMAGE_TASK_COLLECTION: [],
            },
            status_count=[],
        )
        self.connected = False
        self.connect_count = 0

    async def connect(self):
        self.connected = True
        self.connect_count += 1


@pytest.mark.asyncio
async def test_wait_for_drain_reconnects_mongo_after_backfill_closed_global_manager(monkeypatch):
    reconnectable = _ReconnectableMongo()

    async def healthy(_self):
        return True

    monkeypatch.setattr(
        "app.services.mongo_state_manager.get_existing_mongo_state_manager",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.mongo_state_manager.get_mongo_state_manager",
        lambda: reconnectable,
    )
    monkeypatch.setattr(AutoPublishController, "_sync_service_healthy", healthy)

    controller = AutoPublishController(
        pending_threshold=0,
        check_interval_sec=0,
        runtime_control=_RuntimeControl(),
    )

    assert await controller._wait_for_drain() is True
    assert reconnectable.connected is True
    assert reconnectable.connect_count == 1


@pytest.mark.asyncio
async def test_publish_page_can_assume_read_timeout_was_accepted(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_ASSUME_TIMEOUT_SUCCESS", True)
    controller = AutoPublishController(url="http://publisher.local", token="", timeout_sec=1)
    controller._http = _TimeoutHttpClient()

    result = await controller._publish_page(15)

    assert result.status_code == 202
    assert result.page == 15
    assert result.body_excerpt == "read_timeout_assumed_accepted"


@pytest.mark.asyncio
async def test_publish_page_raises_read_timeout_by_default(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_ASSUME_TIMEOUT_SUCCESS", False)
    controller = AutoPublishController(url="http://publisher.local", token="", timeout_sec=1)
    controller._http = _TimeoutHttpClient()

    with pytest.raises(httpx.ReadTimeout):
        await controller._publish_page(15)


@pytest.mark.asyncio
async def test_publish_page_passes_requested_topic(monkeypatch):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_TOPIC", "VECTOR_JOURNAL_ARTICLE_V2")
    controller = AutoPublishController(url="http://publisher.local", token="", timeout_sec=1)
    http_client = _CapturePostHttpClient()
    controller._http = http_client

    result = await controller._publish_page(15)

    assert result.status_code == 200
    body = str(http_client.content)
    assert "pageNumer=15" in body
    assert "topic=VECTOR_JOURNAL_ARTICLE_V2" in body


@pytest.mark.asyncio
async def test_publish_round_updates_current_page_after_each_success(tmp_path):
    controller = AutoPublishController(
        url="http://publisher.local",
        token="",
        timeout_sec=1,
        checkpoint_path=tmp_path / "checkpoint.json",
        runtime_control=_RuntimeControl(),
    )

    async def fake_publish_page(page: int):
        return type(
            "Result",
            (),
            {"page": page, "status_code": 200, "elapsed_seconds": 0.01},
        )()

    controller._publish_page = fake_publish_page

    last_success = await controller._publish_round(42, 2)

    assert last_success == 43
    assert controller.status()["current_page"] == 43


@pytest.mark.asyncio
async def test_daily_publish_stops_at_record_limit_and_persists_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_DAILY_RECORD_LIMIT", 2500)
    monkeypatch.setattr(settings, "PUBLISH_BATCH_PAGE_RECORD_COUNT", 1000)

    controller = AutoPublishController(
        url="http://publisher.local",
        token="",
        timeout_sec=1,
        interval_sec=0,
        checkpoint_path=tmp_path / "checkpoint.json",
        runtime_control=_RuntimeControl(),
        sync_health_url="",
    )

    async def fake_publish_page(page: int):
        return type(
            "Result",
            (),
            {
                "page": page,
                "status_code": 200,
                "elapsed_seconds": 0.01,
                "success_count": 1000,
                "current_page_size": None,
            },
        )()

    async def fake_wait_for_drain():
        return True

    controller._publish_page = fake_publish_page
    controller._wait_for_drain = fake_wait_for_drain

    result = await controller.run_daily_once()

    assert result["stopped_reason"] == "record_limit_reached"
    assert result["successful_pages"] == 3
    assert result["published_records"] == 3000
    assert result["last_success_page"] == 3
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["last_success_page"] == 3


@pytest.mark.asyncio
async def test_daily_publish_uses_page_count_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_DAILY_RECORD_LIMIT", 2000)
    monkeypatch.setattr(settings, "PUBLISH_BATCH_PAGE_RECORD_COUNT", 1000)

    controller = AutoPublishController(
        url="http://publisher.local",
        token="",
        timeout_sec=1,
        interval_sec=0,
        checkpoint_path=tmp_path / "checkpoint.json",
        runtime_control=_RuntimeControl(),
        sync_health_url="",
    )

    async def fake_publish_page(page: int):
        return type(
            "Result",
            (),
            {
                "page": page,
                "status_code": 200,
                "elapsed_seconds": 0.01,
            },
        )()

    async def fake_wait_for_drain():
        return True

    controller._publish_page = fake_publish_page
    controller._wait_for_drain = fake_wait_for_drain

    result = await controller.run_daily_once()

    assert result["successful_pages"] == 2
    assert result["published_records"] == 2000
    assert result["fallback_record_pages"] == 2


@pytest.mark.asyncio
async def test_daily_publish_can_count_real_kafka_offset_delta(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_DAILY_RECORD_LIMIT", 1200)
    monkeypatch.setattr(settings, "PUBLISH_BATCH_PAGE_RECORD_COUNT", 1000)
    monkeypatch.setattr(settings, "PUBLISH_BATCH_COUNT_MODE", "kafka_offset")
    monkeypatch.setattr(settings, "PUBLISH_BATCH_MAX_PAGES_PER_RUN", 0)

    controller = AutoPublishController(
        url="http://publisher.local",
        token="",
        timeout_sec=1,
        interval_sec=0,
        checkpoint_path=tmp_path / "checkpoint.json",
        runtime_control=_RuntimeControl(),
        sync_health_url="",
    )
    offsets = iter([{0: 1000}, {0: 1600}, {0: 1600}, {0: 2200}])

    async def fake_publish_page(page: int):
        return type(
            "Result",
            (),
            {
                "page": page,
                "status_code": 200,
                "elapsed_seconds": 0.01,
            },
        )()

    async def fake_wait_for_drain():
        return True

    controller._publish_page = fake_publish_page
    controller._wait_for_drain = fake_wait_for_drain
    controller._read_topic_end_offsets = lambda: next(offsets)

    result = await controller.run_daily_once()

    assert result["stopped_reason"] == "record_limit_reached"
    assert result["successful_pages"] == 2
    assert result["published_records"] == 1200
    assert result["kafka_records"] == 1200
    assert result["kafka_record_pages"] == 2
    assert result["fallback_record_pages"] == 2


@pytest.mark.asyncio
async def test_daily_publish_accepts_record_limit_override(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PUBLISH_BATCH_DAILY_RECORD_LIMIT", 200000)
    monkeypatch.setattr(settings, "PUBLISH_BATCH_PAGE_RECORD_COUNT", 1000)

    controller = AutoPublishController(
        url="http://publisher.local",
        token="",
        timeout_sec=1,
        interval_sec=0,
        checkpoint_path=tmp_path / "checkpoint.json",
        runtime_control=_RuntimeControl(),
        sync_health_url="",
    )

    async def fake_publish_page(page: int):
        return type(
            "Result",
            (),
            {
                "page": page,
                "status_code": 200,
                "elapsed_seconds": 0.01,
                "success_count": 1000,
                "current_page_size": None,
            },
        )()

    async def fake_wait_for_drain():
        return True

    controller._publish_page = fake_publish_page
    controller._wait_for_drain = fake_wait_for_drain

    result = await controller.run_daily_once(record_limit=1500, run_reason="startup")

    assert result["run_reason"] == "startup"
    assert result["record_limit"] == 1500
    assert result["successful_pages"] == 2
    assert result["published_records"] == 2000
    assert result["last_success_page"] == 2


def _matches(doc: dict, query: dict) -> bool:
    if "$and" in query:
        return all(_matches(doc, item) for item in query["$and"])
    if "$or" in query:
        return any(_matches(doc, item) for item in query["$or"])
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                if actual not in set(expected["$in"]):
                    return False
            elif "$lte" in expected:
                limit = expected["$lte"]
                if actual is None or actual > limit:
                    return False
            else:
                raise AssertionError(f"unsupported fake filter operator: {expected}")
        elif actual != expected:
            return False
    return True
