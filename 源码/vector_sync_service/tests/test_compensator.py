from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.config import settings
from app.services import compensator


def test_scheduled_window_accepts_current_day_time(monkeypatch):
    monkeypatch.setattr(settings, "COMPENSATOR_RUN_MODE", "scheduled")
    monkeypatch.setattr(settings, "COMPENSATOR_START_TIME", "00:00")
    monkeypatch.setattr(settings, "COMPENSATOR_RUN_HOURS", 2.0)
    monkeypatch.setattr(settings, "COMPENSATOR_TIMEZONE", "Asia/Shanghai")

    in_window, sleep_seconds = compensator._window_state(
        datetime(2026, 7, 30, 1, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    )

    assert in_window is True
    assert sleep_seconds == 0.0


def test_scheduled_window_supports_cross_midnight(monkeypatch):
    monkeypatch.setattr(settings, "COMPENSATOR_RUN_MODE", "scheduled")
    monkeypatch.setattr(settings, "COMPENSATOR_START_TIME", "23:00")
    monkeypatch.setattr(settings, "COMPENSATOR_RUN_HOURS", 2.0)
    monkeypatch.setattr(settings, "COMPENSATOR_TIMEZONE", "Asia/Shanghai")

    in_window, _ = compensator._window_state(
        datetime(2026, 7, 31, 0, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    )

    assert in_window is True


def test_scheduled_window_sleeps_until_next_start(monkeypatch):
    monkeypatch.setattr(settings, "COMPENSATOR_RUN_MODE", "scheduled")
    monkeypatch.setattr(settings, "COMPENSATOR_START_TIME", "03:00")
    monkeypatch.setattr(settings, "COMPENSATOR_RUN_HOURS", 1.0)
    monkeypatch.setattr(settings, "COMPENSATOR_TIMEZONE", "Asia/Shanghai")

    in_window, sleep_seconds = compensator._window_state(
        datetime(2026, 7, 30, 1, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )

    assert in_window is False
    assert sleep_seconds == 7200.0


@pytest.mark.parametrize("value", ["metadata,fulltext,image", " METADATA , image "])
def test_selected_pipelines_normalizes_valid_values(monkeypatch, value):
    monkeypatch.setattr(settings, "COMPENSATOR_PIPELINES", value)

    assert compensator._selected_pipelines() <= {"metadata", "fulltext", "image"}


def test_selected_pipelines_rejects_unknown_value(monkeypatch):
    monkeypatch.setattr(settings, "COMPENSATOR_PIPELINES", "metadata,unknown")

    with pytest.raises(ValueError, match="unsupported COMPENSATOR_PIPELINES"):
        compensator._selected_pipelines()


@pytest.mark.asyncio
async def test_round_claims_only_matching_target_bound_failures(monkeypatch):
    manager = FakeManager(
        [
            {
                "failure_key": "new-record",
                "pipeline": "metadata",
                "resolved_collection": "documents_v2",
                "target_signature": "signature-v2",
            }
        ]
    )
    retried: list[str] = []

    async def retry(failure):
        retried.append(failure["failure_key"])
        return type("Outcome", (), {"final_status": "SUCCESS", "commit_safe": True, "pending_failure_refs": []})()

    monkeypatch.setattr(compensator, "retry_failure_task", retry)
    monkeypatch.setattr(
        compensator,
        "_current_retry_targets",
        lambda _pipelines: {"metadata": ("documents_v2", "signature-v2")},
    )
    monkeypatch.setattr(compensator, "_reconcile_milvus_sync_status", _zero)

    processed = await compensator._run_compensator_round(manager, {"metadata"})

    assert processed == 1
    assert retried == ["new-record"]
    assert manager.claim_arguments == [
        {
            "pipelines": {"metadata"},
            "targets": {"metadata": ("documents_v2", "signature-v2")},
        },
        {
            "pipelines": {"metadata"},
            "targets": {"metadata": ("documents_v2", "signature-v2")},
        },
    ]


@pytest.mark.asyncio
async def test_scheduled_round_finishes_claimed_work_without_claiming_after_window_close(monkeypatch):
    manager = FakeManager(
        [
            {
                "failure_key": "first",
                "pipeline": "metadata",
                "resolved_collection": "documents_v2",
                "target_signature": "signature-v2",
            },
            {
                "failure_key": "second",
                "pipeline": "metadata",
                "resolved_collection": "documents_v2",
                "target_signature": "signature-v2",
            },
        ]
    )
    retried: list[str] = []
    window_states = iter([(True, 0.0), (False, 3600.0)])

    async def retry(failure):
        retried.append(failure["failure_key"])
        return type("Outcome", (), {"final_status": "SUCCESS", "commit_safe": True, "pending_failure_refs": []})()

    monkeypatch.setattr(compensator, "retry_failure_task", retry)
    monkeypatch.setattr(
        compensator,
        "_current_retry_targets",
        lambda _pipelines: {"metadata": ("documents_v2", "signature-v2")},
    )
    monkeypatch.setattr(compensator, "_reconcile_milvus_sync_status", _zero)
    monkeypatch.setattr(compensator, "_window_state", lambda: next(window_states))

    processed = await compensator._run_compensator_round(
        manager,
        {"metadata"},
        stop_claiming_when_window_closes=True,
    )

    assert processed == 1
    assert retried == ["first"]
    assert len(manager.claim_arguments) == 1


async def _zero(_manager):
    return 0


class FakeManager:
    connected = True

    def __init__(self, failures):
        self.failures = list(failures)
        self.claim_arguments = []

    async def claim_failure_for_retry_filtered(self, *, pipelines, targets):
        self.claim_arguments.append({"pipelines": set(pipelines), "targets": dict(targets)})
        return self.failures.pop(0) if self.failures else None
