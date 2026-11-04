from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import dashboard


@pytest.mark.asyncio
async def test_dashboard_summary_keeps_mongo_but_uses_milvus_row_counts(monkeypatch):
    fake_milvus = {
        "connected": True,
        "uri": "http://127.0.0.1:19531",
        "database": "default",
        "collections": {
            "documents": {"row_count": 82_846},
            "fulltext_chunks": {"row_count": 1_099_756},
            "images": {"row_count": 90_236},
        },
        "total_rows": 1_272_838,
    }
    monkeypatch.setattr(dashboard, "_read_milvus_stats_sync", lambda: fake_milvus)

    summary = await dashboard.build_stats_summary(
        app_state=SimpleNamespace(
            consumer_running=True,
            kafka_connected=True,
            compensator_running=False,
            milvus_connected=False,
            embedding_available=True,
        ),
        mongo_manager=None,
        auto_publisher=None,
    )

    assert summary["mongo"]["status_counts"] == {}
    assert summary["milvus"]["collections"]["fulltext_chunks"]["row_count"] == 1_099_756
    assert summary["milvus"]["total_rows"] == 1_272_838
    assert summary["runtime"]["milvus_connected"] is True
    assert summary["config"]["attu_hint"] == dashboard.ATTU_HINT
