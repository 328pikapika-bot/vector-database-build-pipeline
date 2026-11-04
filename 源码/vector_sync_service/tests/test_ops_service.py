from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.ops_service import app, state
from app.services import ops_scheduler


def test_ops_health_and_runtime_endpoints(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)
    monkeypatch.setattr(settings, "OPS_PUBLISH_ENABLED", True)
    monkeypatch.setattr(settings, "OPS_CLEANUP_ENABLED", True)
    monkeypatch.setattr(settings, "OPS_INSPECTION_ENABLED", True)
    monkeypatch.setattr(settings, "OPS_BACKFILL_ENABLED", True)
    monkeypatch.setattr(settings, "OPS_METADATA_BACKFILL_ENABLED", True)
    monkeypatch.setattr(settings, "OPS_FULLTEXT_BACKFILL_ENABLED", True)
    monkeypatch.setattr(settings, "OPS_METADATA_BACKFILL_INTERVAL_HOURS", 24.0)
    monkeypatch.setattr(settings, "OPS_FULLTEXT_BACKFILL_INTERVAL_HOURS", 1.0)
    monkeypatch.setattr(settings, "OPS_METADATA_BACKFILL_WINDOW_ID", "metadata-window")
    monkeypatch.setattr(settings, "OPS_FULLTEXT_BACKFILL_WINDOW_ID", "fulltext-window")
    monkeypatch.setattr(settings, "OPS_PUBLISH_SCHEDULE", "0 1 * * *")
    monkeypatch.setattr(settings, "OPS_CLEANUP_SCHEDULE", "0 3 * * *")
    monkeypatch.setattr(settings, "OPS_BACKFILL_SCHEDULE", "0 */4 * * *")
    monkeypatch.setattr(settings, "OPS_INSPECTION_SCHEDULE", "0 4 * * 1")
    monkeypatch.setattr(
        "app.ops_service._resolve_document_target_runtime",
        lambda: {
            "configured_logical_name": "docs-logical",
            "logical_name": "docs-logical",
            "resolved_name": "docs-logical__ivfpq_nlist4096_m64_nbits8_cosine",
            "index_signature": "IVFPQ|COSINE|NLIST=4096|M=64|NBITS=8",
            "current_index_signature": "IVFPQ|COSINE|NLIST=4096|M=64|NBITS=8",
            "created": False,
            "reused": True,
            "source_collection_name": "docs-logical",
        },
    )
    monkeypatch.setattr(
        "app.ops_service._resolve_fulltext_target_runtime",
        lambda: {
            "configured_logical_name": "chunks-logical",
            "logical_name": "chunks-logical",
            "resolved_name": "chunks-logical",
            "index_signature": "IVFPQ|COSINE|NLIST=4096|M=64|NBITS=8",
            "current_index_signature": "IVFPQ|COSINE|NLIST=4096|M=64|NBITS=8",
            "created": False,
            "reused": True,
            "source_collection_name": "chunks-logical",
        },
    )
    monkeypatch.setattr(
        "app.ops_service._run_publish_once",
        _async_return(
            {
                "record_limit": 200000,
                "published_records": 200000,
                "last_success_page": 200,
                "stopped_reason": "record_limit_reached",
            }
        ),
    )
    monkeypatch.setattr(
        "app.ops_service._run_cleanup_once",
        _async_return(
            {
                "mode": "daily",
                "dry_run": True,
                "mongo": {},
                "files": {"attachments_deleted": 1},
                "errors": [],
            }
        ),
    )
    monkeypatch.setattr(
        "app.ops_service._run_backfill_once",
        _async_return(
            {
                "metadata": {"processed": 2, "marked_done": 2, "marked_failed": 0},
                "fulltext": {"processed": 1, "marked_done": 1, "marked_failed": 0},
                "processed": 3,
                "marked_done": 3,
                "marked_failed": 0,
                "errors": [],
            }
        ),
    )
    monkeypatch.setattr(
        "app.ops_service._run_metadata_backfill_once",
        _async_return(
            {
                "lane": "metadata",
                "processed": 2,
                "marked_done": 2,
                "marked_failed": 0,
            }
        ),
    )
    monkeypatch.setattr(
        "app.ops_service._run_fulltext_backfill_once",
        _async_return(
            {
                "lane": "fulltext",
                "processed": 1,
                "marked_done": 1,
                "marked_failed": 0,
            }
        ),
    )
    monkeypatch.setattr(
        "app.ops_service._run_inspection_once",
        _async_return(
            {
                "checked_at": "2026-06-18T00:00:00+00:00",
                "backup": {"backup_db": "backup_latest"},
                "expired_artifacts_visible": 0,
                "notes": [],
            }
        ),
    )

    state.mongo_connected = True
    state.started_at = "2026-06-18T00:00:00+00:00"
    state.scheduler_tasks.clear()
    state.last_publish = None
    state.last_cleanup = None
    state.last_backfill = None
    state.last_inspection = None
    state.metadata_backfill.last_result = None
    state.fulltext_backfill.last_result = None
    state.publisher.resume()

    client = TestClient(app)
    try:
        health = client.get("/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["service"] == "ops"
        assert payload["runtime"]["publish_enabled"] is True
        assert payload["runtime"]["backfill_enabled"] is True

        runtime = client.get("/ops/runtime")
        assert runtime.status_code == 200
        runtime_payload = runtime.json()
        assert runtime_payload["mongo_connected"] is True
        assert runtime_payload["publisher"]["paused"] is False
        assert runtime_payload["schedules"]["backfill"] == "0 */4 * * *"
        assert runtime_payload["targets"]["document"]["resolved_name"] == "docs-logical__ivfpq_nlist4096_m64_nbits8_cosine"
        assert runtime_payload["targets"]["fulltext"]["resolved_name"] == "chunks-logical"
        assert runtime_payload["metadata_backfill"]["window_id"] == "metadata-window"
        assert runtime_payload["fulltext_backfill"]["window_id"] == "fulltext-window"

        pause = client.post("/ops/publisher/pause", json={"reason": "maintenance"})
        assert pause.status_code == 200
        assert pause.json()["pause_reason"] == "maintenance"

        resume = client.post("/ops/publisher/resume")
        assert resume.status_code == 200
        assert resume.json()["paused"] is False

        cleanup = client.post("/ops/cleanup/run", json={"mode": "daily", "dry_run": True})
        assert cleanup.status_code == 200
        assert cleanup.json()["files"]["attachments_deleted"] == 1
        assert state.last_cleanup["mode"] == "daily"

        publish = client.post("/ops/publisher/run")
        assert publish.status_code == 200
        assert publish.json()["published_records"] == 200000
        assert state.last_publish["last_success_page"] == 200

        metadata_backfill = client.post("/ops/metadata-backfill/run", json={"dry_run": True})
        assert metadata_backfill.status_code == 200
        assert metadata_backfill.json()["lane"] == "metadata"
        assert state.metadata_backfill.last_result["processed"] == 2

        fulltext_backfill = client.post("/ops/fulltext-backfill/run", json={"dry_run": True})
        assert fulltext_backfill.status_code == 200
        assert fulltext_backfill.json()["lane"] == "fulltext"
        assert state.fulltext_backfill.last_result["processed"] == 1

        backfill = client.post("/ops/backfill/run", json={"dry_run": True})
        assert backfill.status_code == 200
        assert backfill.json()["marked_done"] == 3
        assert state.last_backfill["processed"] == 3

        inspection = client.post("/ops/inspection/run")
        assert inspection.status_code == 200
        assert inspection.json()["backup"]["backup_db"] == "backup_latest"
        assert state.last_inspection["expired_artifacts_visible"] == 0
    finally:
        client.close()


def test_seconds_until_next_run_supports_steps():
    schedule = ops_scheduler.parse_cron_schedule("*/15 1 * * 1")

    assert 0 in schedule.minute
    assert 15 in schedule.minute
    assert 30 in schedule.minute
    assert 45 in schedule.minute
    assert schedule.hour == {1}
    assert schedule.weekday == {1}


def test_publisher_run_accepts_record_limit(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)
    calls = []

    async def fake_run_publish_once(*, record_limit=None, run_reason="scheduled"):
        calls.append({"record_limit": record_limit, "run_reason": run_reason})
        return {
            "record_limit": record_limit,
            "run_reason": run_reason,
            "published_records": record_limit,
            "last_success_page": 12,
        }

    monkeypatch.setattr("app.ops_service._run_publish_once", fake_run_publish_once)

    client = TestClient(app)
    try:
        publish = client.post("/ops/publisher/run", json={"record_limit": 1234})
        assert publish.status_code == 200
        assert publish.json()["record_limit"] == 1234
        assert publish.json()["run_reason"] == "manual"
        assert calls == [{"record_limit": 1234, "run_reason": "manual"}]
        assert state.last_publish["published_records"] == 1234
    finally:
        client.close()


@pytest.mark.asyncio
async def test_run_publish_once_writes_report(monkeypatch, tmp_path):
    class _Publisher:
        async def run_daily_once(self, *, record_limit=None, run_reason="scheduled"):
            return {
                "record_limit": record_limit,
                "run_reason": run_reason,
                "published_records": 500,
                "last_success_page": 9,
            }

    report_path = tmp_path / "publish_startup.json"
    monkeypatch.setattr(state, "publisher", _Publisher())
    monkeypatch.setattr("app.ops_service._runtime_report_path", lambda _prefix: report_path)

    from app.ops_service import _run_publish_once

    result = await _run_publish_once(record_limit=500, run_reason="startup")

    assert result["report_path"] == str(report_path)
    assert report_path.exists()
    assert '"run_reason": "startup"' in report_path.read_text(encoding="utf-8")
    assert '"published_records": 500' in report_path.read_text(encoding="utf-8")


def _async_return(value):
    async def _runner(*_args, **_kwargs):
        return value

    return _runner
