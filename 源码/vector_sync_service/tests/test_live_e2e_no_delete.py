from pathlib import Path

from app.config import settings
from scripts import live_e2e_no_delete as e2e


def _base_names() -> e2e.BaseNames:
    return e2e.BaseNames(
        document_collection="scientific_documents",
        chunk_collection="scientific_fulltext_chunks",
        image_collection="scientific_images",
        mongo_status_collection="doc_sync_status",
        mongo_failure_collection="doc_sync_failures",
        mongo_attempt_collection="doc_sync_attempts",
        mongo_artifact_collection="doc_sync_artifacts",
        mongo_fulltext_task_collection="doc_fulltext_tasks",
        mongo_image_task_collection="doc_image_tasks",
    )


def test_default_run_id_is_short_for_namespaced_e2e_collections():
    run_id = e2e.default_run_id()

    assert run_id.startswith("r")
    assert len(run_id) == len("r0609175307")


def test_preflight_reports_mongo_namespace_too_long(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(e2e, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(e2e, "check_imports", lambda: {"ok": True, "errors": []})
    monkeypatch.setattr(e2e, "mongo_ping", lambda: {"ok": True})
    monkeypatch.setattr(e2e, "inspect_kafka_topic", lambda group_id: {"ok": True, "group_id": group_id})
    monkeypatch.setattr(e2e, "assert_test_collections_absent", lambda phases: {"ok": True})
    monkeypatch.setattr(settings, "MONGO_DB", "vector_sync")
    monkeypatch.setattr(settings, "IMAGE_MODEL_API_KEY", "unit-test-key")

    base = _base_names()
    phase = e2e.build_phase_names(base, "run_" + ("x" * 96), "f10", 10)

    result = e2e.run_preflight(base, [phase])

    assert result["ok"] is False
    assert any(error.startswith("mongo_namespace_too_long:") for error in result["errors"])
    assert result["mongo_namespace_lengths"]["violations"]


def test_preflight_namespace_check_accepts_short_run_id(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(e2e, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(settings, "MONGO_DB", "vector_sync")

    base = _base_names()
    phase = e2e.build_phase_names(base, "k10fix_181500", "f10", 10)

    result = e2e.validate_mongo_namespace_lengths(settings.MONGO_DB, [phase])

    assert result["ok"] is True
    assert result["max_seen"]["bytes"] <= e2e.MONGO_NAMESPACE_MAX_BYTES
