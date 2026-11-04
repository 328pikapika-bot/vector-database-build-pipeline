from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.main import app, state
from app.services import embedder as embedder_module
from app.services import fulltext_downloader as fulltext_downloader_module
from app.services import fulltext_worker as fulltext_worker_module
from app.services import image_downloader as image_downloader_module
from app.services import image_feature_pipeline as image_feature_pipeline_module
from app.services import milvus_chunk_service as milvus_chunk_service_module
from app.services import milvus_image_service as milvus_image_service_module
from app.services.runtime_control import get_background_loop_control


@pytest.fixture(autouse=True)
def _disable_admin_auth_for_legacy_endpoint_tests(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)


def test_admin_control_pause_status_drain_resume():
    client = TestClient(app)
    try:
        resume_response = client.post("/admin/control/resume")
        assert resume_response.status_code == 200

        pause_response = client.post(
            "/admin/control/pause",
            json={"reason": "unit test"},
        )
        assert pause_response.status_code == 200
        pause_payload = pause_response.json()
        assert pause_payload["control"]["paused"] is True
        assert pause_payload["control"]["pause_reason"] == "unit test"

        status_response = client.get("/admin/control/status")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["control"]["paused"] is True
        assert "consumer" in status_payload
        assert "compensator" in status_payload
        assert "mongodb" in status_payload
        assert "image" in status_payload
        assert status_payload["image"]["task_upsert_enabled"] is settings.IMAGE_TASK_UPSERT_ENABLED
        assert status_payload["image"]["worker_enabled"] is settings.IMAGE_WORKER_ENABLED
        assert status_payload["image"]["worker_running"] is False
        assert status_payload["image"]["feature_pipeline_enabled"] is settings.IMAGE_FEATURE_PIPELINE_ENABLED
        assert status_payload["image"]["milvus_write_enabled"] is settings.IMAGE_MILVUS_WRITE_ENABLED
        assert status_payload["image"]["lease_seconds"] == settings.IMAGE_WORKER_LEASE_SEC
        assert status_payload["image"]["batch_size"] == settings.IMAGE_WORKER_BATCH_SIZE
        assert status_payload["image"]["poll_interval_seconds"] == settings.IMAGE_WORKER_POLL_INTERVAL_SEC

        drain_response = client.post("/admin/control/drain")
        assert drain_response.status_code == 200
        drain_payload = drain_response.json()
        assert drain_payload["control"]["paused"] is True
        assert drain_payload["control"]["draining"] is True
        assert drain_payload["control"]["inflight_batches"] == 0

        final_resume_response = client.post("/admin/control/resume")
        assert final_resume_response.status_code == 200
        final_payload = final_resume_response.json()
        assert final_payload["control"]["paused"] is False
        assert final_payload["control"]["draining"] is False
    finally:
        client.post("/admin/control/resume")
        client.close()


def test_admin_backpressure_can_be_hot_updated_and_reset():
    client = TestClient(app)
    try:
        status_response = client.get("/admin/control/backpressure")
        assert status_response.status_code == 200
        original = status_response.json()
        assert "enabled" in original
        assert "high_watermark" in original

        update_response = client.post(
            "/admin/control/backpressure",
            json={
                "enabled": True,
                "mode": "image_priority",
                "high_watermark": 321,
                "low_watermark": 45,
                "check_interval_seconds": 1.5,
                "consumer_time_budget_ms": 4321,
                "consumer_micro_batch_size": 7,
                "image_pending_soft_limit": 654,
                "image_pending_resume_threshold": 123,
                "soft_watermark_factor": 0.66,
                "soft_throttle_sleep_ms": 222,
                "soft_micro_batch_min": 2,
            },
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["enabled"] is True
        assert updated["mode"] == "image_priority"
        assert updated["high_watermark"] == 321
        assert updated["low_watermark"] == 45
        assert updated["check_interval_seconds"] == 1.5
        assert updated["consumer_time_budget_ms"] == 4321
        assert updated["consumer_micro_batch_size"] == 7
        assert updated["soft_watermark_factor"] == 0.66
        assert updated["soft_throttle_sleep_ms"] == 222
        assert updated["soft_micro_batch_min"] == 2
        assert updated["image_pending_soft_limit"] == 654
        assert updated["image_pending_resume_threshold"] == 123

        reset_response = client.post("/admin/control/backpressure/reset")
        assert reset_response.status_code == 200
        reset_payload = reset_response.json()
        assert reset_payload["high_watermark"] == settings.KAFKA_BACKPRESSURE_HIGH_WATERMARK
        assert reset_payload["low_watermark"] == settings.KAFKA_BACKPRESSURE_LOW_WATERMARK
        assert reset_payload["mode"] == settings.KAFKA_BACKPRESSURE_MODE
        assert reset_payload["consumer_time_budget_ms"] == settings.KAFKA_CONSUMER_TIME_BUDGET_MS
        assert reset_payload["consumer_micro_batch_size"] == settings.KAFKA_CONSUMER_MICRO_BATCH_SIZE
    finally:
        client.post("/admin/control/backpressure/reset")
        client.close()


def test_admin_retention_cleanup_status_and_run(monkeypatch):
    async def fake_run(self, *, mode="daily", dry_run=False):
        return {
            "mode": mode,
            "dry_run": dry_run,
            "mongo": {},
            "files": {},
            "errors": [],
        }

    monkeypatch.setattr(
        "app.main.RetentionCleanupService.run",
        fake_run,
    )
    client = TestClient(app)
    try:
        status_response = client.get("/admin/retention/cleanup/status")
        assert status_response.status_code == 200

        run_response = client.post(
            "/admin/retention/cleanup/run",
            json={"mode": "daily", "dry_run": True},
        )
        assert run_response.status_code == 200
        payload = run_response.json()
        assert payload["mode"] == "daily"
        assert payload["dry_run"] is True
    finally:
        client.close()


def test_health_includes_runtime_status():
    client = TestClient(app)
    try:
        client.post("/admin/control/resume")

        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "ok"
        assert "components" in payload
        assert "runtime" in payload
        assert "control" in payload["runtime"]
        assert "consumer" in payload["runtime"]
        assert "backpressure" in payload["runtime"]["consumer"]
        assert "compensator" in payload["runtime"]
        assert "mongodb" in payload["runtime"]
        assert "fulltext" in payload["runtime"]
        assert "image" in payload["runtime"]
        assert "enabled" in payload["runtime"]["consumer"]["backpressure"]
        assert "active" in payload["runtime"]["consumer"]["backpressure"]
        expected_static_configured = bool(str(settings.ATTACHMENT_PREFIX or "").strip())
        assert payload["runtime"]["fulltext"]["configured"] is expected_static_configured
        assert payload["runtime"]["fulltext"]["base_url_configured"] is expected_static_configured
        assert payload["runtime"]["fulltext"]["timeout"] == 30
        assert payload["runtime"]["fulltext"]["max_retries"] == 3
        assert payload["runtime"]["fulltext"]["worker_enabled"] is settings.FULLTEXT_WORKER_ENABLED
        assert payload["runtime"]["fulltext"]["worker_running"] is False
        assert payload["runtime"]["fulltext"]["lease_seconds"] == settings.FULLTEXT_WORKER_LEASE_SEC
        assert payload["runtime"]["fulltext"]["batch_size"] == settings.FULLTEXT_WORKER_BATCH_SIZE
        assert payload["runtime"]["fulltext"]["poll_interval_seconds"] == settings.FULLTEXT_WORKER_POLL_INTERVAL_SEC
        assert "last_run_at" in payload["runtime"]["fulltext"]
        assert "last_task_id" in payload["runtime"]["fulltext"]
        assert "last_processed_count" in payload["runtime"]["fulltext"]
        assert "last_error" in payload["runtime"]["fulltext"]
        assert "max_doc_chars" in payload["runtime"]["fulltext"]
        assert "task_summary" in payload["runtime"]["fulltext"]
        assert payload["runtime"]["image"]["configured"] is True
        assert payload["runtime"]["image"]["base_url_configured"] is True
        assert payload["runtime"]["image"]["task_upsert_enabled"] is settings.IMAGE_TASK_UPSERT_ENABLED
        assert payload["runtime"]["image"]["worker_enabled"] is settings.IMAGE_WORKER_ENABLED
        assert payload["runtime"]["image"]["worker_running"] is False
        assert payload["runtime"]["image"]["feature_pipeline_enabled"] is settings.IMAGE_FEATURE_PIPELINE_ENABLED
        assert payload["runtime"]["image"]["milvus_write_enabled"] is settings.IMAGE_MILVUS_WRITE_ENABLED
        assert payload["runtime"]["image"]["lease_seconds"] == settings.IMAGE_WORKER_LEASE_SEC
        assert payload["runtime"]["image"]["batch_size"] == settings.IMAGE_WORKER_BATCH_SIZE
        assert payload["runtime"]["image"]["poll_interval_seconds"] == settings.IMAGE_WORKER_POLL_INTERVAL_SEC
        assert "last_run_at" in payload["runtime"]["image"]
        assert "last_task_id" in payload["runtime"]["image"]
        assert "last_processed_count" in payload["runtime"]["image"]
        assert "last_error" in payload["runtime"]["image"]
        assert payload["runtime"]["image"]["timeout"] == settings.IMAGE_DOWNLOAD_TIMEOUT
        assert payload["runtime"]["image"]["max_retries"] == settings.IMAGE_DOWNLOAD_MAX_RETRIES
        assert payload["runtime"]["image"]["download_concurrency"] == settings.IMAGE_DOWNLOAD_CONCURRENCY
        assert payload["runtime"]["image"]["preprocess_concurrency"] == settings.IMAGE_PREPROCESS_CONCURRENCY
        assert payload["runtime"]["image"]["max_pixels"] == settings.IMAGE_MAX_PIXELS
        assert payload["runtime"]["image"]["feature_total_timeout"] == settings.IMAGE_FEATURE_TOTAL_TIMEOUT_SEC
        assert payload["runtime"]["image"]["pixel_validation_available"] is True
        assert "task_summary" in payload["runtime"]["image"]
        assert payload["runtime"]["image"]["stage_executors"] == {}
        assert state.fulltext_worker_task is None
        assert state.image_worker_task is None
    finally:
        client.post("/admin/control/resume")
        client.close()


def test_health_reports_live_milvus_probe(monkeypatch):
    class FakeMilvusClient:
        def has_collection(self, collection_name):
            return bool(collection_name)

    class FakeMilvusService:
        client = FakeMilvusClient()

    monkeypatch.setattr("app.main.get_milvus_service", lambda: FakeMilvusService())
    monkeypatch.setattr(
        "app.main.resolve_document_collection_target",
        lambda *_args, **_kwargs: type("Target", (), {"resolved_name": "documents"})(),
    )
    state.milvus_connected = False

    client = TestClient(app)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()

        assert payload["components"]["milvus"] == "connected"
        assert payload["runtime"]["milvus"]["connected"] is True
    finally:
        state.milvus_connected = False
        client.close()


def test_image_status_does_not_create_image_runtime_clients(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("image status must not create runtime clients")

    monkeypatch.setattr("app.main.get_image_worker_runner", forbidden)
    monkeypatch.setattr(image_downloader_module.ImageDownloader, "__init__", forbidden)
    monkeypatch.setattr(milvus_image_service_module.MilvusImageService, "__init__", forbidden)
    monkeypatch.setattr(image_feature_pipeline_module, "ImageFeatureProviders", forbidden)
    state.image_worker_runner = None
    state.image_worker_task = None

    client = TestClient(app)
    try:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_image = health_response.json()["runtime"]["image"]
        _assert_image_runtime_default_off(health_image)

        status_response = client.get("/admin/control/status")
        assert status_response.status_code == 200
        status_image = status_response.json()["image"]
        _assert_image_runtime_default_off(status_image)
        assert state.image_worker_task is None
    finally:
        client.close()


def test_fulltext_status_does_not_create_fulltext_runtime_clients(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("fulltext status must not create runtime clients")

    monkeypatch.setattr("app.main.get_fulltext_worker_runner", forbidden)
    monkeypatch.setattr(
        fulltext_downloader_module.FullTextDownloader,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(fulltext_worker_module.FullTextWorker, "__init__", forbidden)
    monkeypatch.setattr(embedder_module, "get_embedding_client", forbidden)
    monkeypatch.setattr(milvus_chunk_service_module.MilvusChunkService, "__init__", forbidden)
    state.fulltext_worker_runner = None
    state.fulltext_worker_task = None
    state.mongo_connected = False

    client = TestClient(app)
    try:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_fulltext = health_response.json()["runtime"]["fulltext"]
        _assert_fulltext_runtime_default_off(health_fulltext)

        status_response = client.get("/admin/control/status")
        assert status_response.status_code == 200
        status_fulltext = status_response.json()["fulltext"]
        _assert_fulltext_runtime_default_off(status_fulltext)
        assert state.fulltext_worker_task is None
    finally:
        client.close()


def test_admin_auth_enabled_enforces_read_write_roles(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "ADMIN_READ_TOKENS", "read-token")
    monkeypatch.setattr(settings, "ADMIN_WRITE_TOKENS", "write-token")

    client = TestClient(app)
    try:
        health_response = client.get("/health")
        assert health_response.status_code == 200

        missing = client.get("/admin/control/status")
        assert missing.status_code == 401

        invalid = client.get(
            "/admin/control/status",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert invalid.status_code == 401
        assert "wrong-token" not in invalid.text

        read_status = client.get(
            "/admin/control/status",
            headers={"Authorization": "Bearer read-token"},
        )
        assert read_status.status_code == 200

        read_diagnostics = client.get(
            "/admin/diagnostics",
            headers={"Authorization": "Bearer read-token"},
        )
        assert read_diagnostics.status_code == 200

        read_pause = client.post(
            "/admin/control/pause",
            headers={"Authorization": "Bearer read-token"},
        )
        assert read_pause.status_code == 403
        assert "read-token" not in read_pause.text

        write_pause = client.post(
            "/admin/control/pause",
            json={"reason": "auth-test"},
            headers={"Authorization": "Bearer write-token"},
        )
        assert write_pause.status_code == 200
        assert write_pause.json()["control"]["paused"] is True

        write_status = client.get(
            "/admin/control/status",
            headers={"Authorization": "Bearer write-token"},
        )
        assert write_status.status_code == 200
    finally:
        client.post(
            "/admin/control/resume",
            headers={"Authorization": "Bearer write-token"},
        )
        client.close()


def test_worker_control_endpoints_and_invalid_worker_type():
    client = TestClient(app)
    try:
        for worker_type in ("fulltext", "image", "compensator"):
            client.post(f"/admin/workers/{worker_type}/resume")

            pause_response = client.post(
                f"/admin/workers/{worker_type}/pause",
                json={"reason": "maintenance"},
            )
            assert pause_response.status_code == 200
            payload = pause_response.json()
            assert payload["worker_type"] == worker_type
            assert payload["control"]["paused"] is True
            assert payload["status"]["paused"] is True

            status_response = client.get(f"/admin/workers/{worker_type}/status")
            assert status_response.status_code == 200
            assert status_response.json()["control"]["paused"] is True

            drain_response = client.post(f"/admin/workers/{worker_type}/drain")
            assert drain_response.status_code == 200
            assert drain_response.json()["control"]["draining"] is True

            resume_response = client.post(f"/admin/workers/{worker_type}/resume")
            assert resume_response.status_code == 200
            assert resume_response.json()["control"]["paused"] is False

        invalid_response = client.get("/admin/workers/not-a-worker/status")
        assert invalid_response.status_code == 404
        assert invalid_response.json()["detail"] == "unsupported worker_type"
    finally:
        for worker_type in ("fulltext", "image", "compensator"):
            client.post(f"/admin/workers/{worker_type}/resume")
        client.close()


def test_diagnostics_redacts_secrets_and_metrics_have_bounded_labels(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "ADMIN_READ_TOKENS", "diag-read-secret")
    monkeypatch.setattr(settings, "ADMIN_WRITE_TOKENS", "diag-write-secret")
    monkeypatch.setattr(settings, "MONGO_URI", "mongodb://user:pass-secret@host/db")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "api-key-secret")
    monkeypatch.setattr(settings, "KAFKA_SASL_PASSWORD", "sasl-secret")

    client = TestClient(app)
    try:
        diagnostics_response = client.get(
            "/admin/diagnostics",
            headers={"Authorization": "Bearer diag-read-secret"},
        )
        assert diagnostics_response.status_code == 200
        payload = diagnostics_response.json()
        rendered = diagnostics_response.text
        assert "diag-read-secret" not in rendered
        assert "diag-write-secret" not in rendered
        assert "pass-secret" not in rendered
        assert "api-key-secret" not in rendered
        assert "sasl-secret" not in rendered
        assert payload["redacted_settings"]["ADMIN_READ_TOKENS"] == "<redacted>"
        assert payload["redacted_settings"]["ADMIN_WRITE_TOKENS"] == "<redacted>"
        assert payload["redacted_settings"]["MONGO_URI"] == "<redacted>"
        assert payload["redacted_settings"]["EMBEDDING_API_KEY"] == "<redacted>"

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_text = metrics_response.text
        assert "nstl_worker_loop_running" in metrics_text
        assert "nstl_worker_task_backlog" in metrics_text
        assert "nstl_dependency_readiness" in metrics_text
        assert "nstl_diagnostics_generation_total" in metrics_text
        assert "doc-" not in metrics_text
        assert "diag-read-secret" not in metrics_text
        assert "http://127.0.0.1" not in metrics_text
    finally:
        client.close()


def test_document_diagnostics_auth_disabled_and_enabled_roles(monkeypatch):
    manager = FakeDocumentDiagnosticsManager(_empty_document_metadata())
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: manager)

    client = TestClient(app)
    try:
        monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)
        disabled_response = client.get("/admin/diagnostics/document/doc-missing")
        assert disabled_response.status_code == 200
        disabled_payload = disabled_response.json()
        assert disabled_payload["doc_id"] == "doc-missing"
        assert disabled_payload["status_found"] is False
        assert disabled_payload["diagnostic_status"] == "not_found"

        monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", True)
        monkeypatch.setattr(settings, "ADMIN_READ_TOKENS", "doc-read-secret")
        monkeypatch.setattr(settings, "ADMIN_WRITE_TOKENS", "doc-write-secret")

        missing = client.get("/admin/diagnostics/document/doc-missing")
        assert missing.status_code == 401

        invalid = client.get(
            "/admin/diagnostics/document/doc-missing",
            headers={"Authorization": "Bearer wrong-doc-secret"},
        )
        assert invalid.status_code == 401
        assert "wrong-doc-secret" not in invalid.text

        read_response = client.get(
            "/admin/diagnostics/document/doc-missing",
            headers={"Authorization": "Bearer doc-read-secret"},
        )
        assert read_response.status_code == 200
        assert read_response.json()["diagnostic_status"] == "not_found"
        assert "doc-read-secret" not in read_response.text

        write_response = client.get(
            "/admin/diagnostics/document/doc-missing",
            headers={"Authorization": "Bearer doc-write-secret"},
        )
        assert write_response.status_code == 200
        assert write_response.json()["diagnostic_status"] == "not_found"
        assert "doc-write-secret" not in write_response.text
    finally:
        client.close()


def test_document_recovery_plan_auth_disabled_and_enabled_roles(monkeypatch):
    manager = FakeDocumentDiagnosticsManager(_empty_document_metadata())
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: manager)

    client = TestClient(app)
    try:
        monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)
        disabled_response = client.get(
            "/admin/diagnostics/document/doc-missing/recovery-plan"
        )
        assert disabled_response.status_code == 200
        disabled_payload = disabled_response.json()
        assert disabled_payload["doc_id"] == "doc-missing"
        assert disabled_payload["plan_status"] == "no_status"
        assert _recovery_action_ids(disabled_payload) == {
            "run_data_plane_consistency_verifier",
            "escalate_manual_review",
        }

        monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", True)
        monkeypatch.setattr(settings, "ADMIN_READ_TOKENS", "recovery-read-secret")
        monkeypatch.setattr(settings, "ADMIN_WRITE_TOKENS", "recovery-write-secret")

        missing = client.get("/admin/diagnostics/document/doc-missing/recovery-plan")
        assert missing.status_code == 401

        invalid = client.get(
            "/admin/diagnostics/document/doc-missing/recovery-plan",
            headers={"Authorization": "Bearer wrong-recovery-secret"},
        )
        assert invalid.status_code == 401
        assert "wrong-recovery-secret" not in invalid.text

        read_response = client.get(
            "/admin/diagnostics/document/doc-missing/recovery-plan",
            headers={"Authorization": "Bearer recovery-read-secret"},
        )
        assert read_response.status_code == 200
        assert read_response.json()["plan_status"] == "no_status"
        assert "recovery-read-secret" not in read_response.text

        write_response = client.get(
            "/admin/diagnostics/document/doc-missing/recovery-plan",
            headers={"Authorization": "Bearer recovery-write-secret"},
        )
        assert write_response.status_code == 200
        assert write_response.json()["plan_status"] == "no_status"
        assert "recovery-write-secret" not in write_response.text
    finally:
        client.close()


def test_document_diagnostics_handles_mongo_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: None)
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)

    client = TestClient(app)
    try:
        response = client.get("/admin/diagnostics/document/doc-no-mongo")
        assert response.status_code == 200
        payload = response.json()
        assert payload["doc_id"] == "doc-no-mongo"
        assert payload["diagnostic_status"] == "mongo_unavailable"
        assert payload["mongo_connected"] is False
        assert payload["status_found"] is False
        assert payload["error_code"] == "MONGO_DIAGNOSTICS_UNAVAILABLE"

        failing_manager = FakeDocumentDiagnosticsManager(
            _empty_document_metadata(),
            error=RuntimeError("mongodb://user:password-secret@host/db"),
        )
        monkeypatch.setattr(
            "app.main.get_existing_mongo_state_manager",
            lambda: failing_manager,
        )
        failure_response = client.get("/admin/diagnostics/document/doc-mongo-error")
        assert failure_response.status_code == 200
        failure_payload = failure_response.json()
        assert failure_payload["diagnostic_status"] == "mongo_unavailable"
        assert "password-secret" not in failure_response.text
    finally:
        client.close()


def test_document_recovery_plan_handles_mongo_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: None)
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)

    client = TestClient(app)
    try:
        response = client.get("/admin/diagnostics/document/doc-no-mongo/recovery-plan")
        assert response.status_code == 200
        payload = response.json()
        assert payload["doc_id"] == "doc-no-mongo"
        assert payload["diagnostic_status"] == "mongo_unavailable"
        assert payload["plan_status"] == "mongo_unavailable"
        assert _recovery_action_ids(payload) == {
            "pause_drain_worker_before_manual_intervention",
            "escalate_manual_review",
        }

        failing_manager = FakeDocumentDiagnosticsManager(
            _empty_document_metadata(),
            error=RuntimeError("mongodb://user:password-secret@host/db"),
        )
        monkeypatch.setattr(
            "app.main.get_existing_mongo_state_manager",
            lambda: failing_manager,
        )
        failure_response = client.get(
            "/admin/diagnostics/document/doc-mongo-error/recovery-plan"
        )
        assert failure_response.status_code == 200
        assert failure_response.json()["plan_status"] == "mongo_unavailable"
        assert "password-secret" not in failure_response.text
    finally:
        client.close()


def test_document_recovery_plan_recommendations_by_status(monkeypatch):
    now = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    manager = FakeDocumentDiagnosticsByDocManager(
        {
            "doc-success": _document_metadata_with_status(
                "SUCCESS",
                updated_at=now,
            ),
            "doc-partial": _document_metadata_with_status(
                "PARTIAL_FAILURE",
                updated_at=now,
                retry_summary={
                    "has_pending_failure": True,
                    "pending_stage_count": 1,
                    "next_retry_at": now + timedelta(minutes=10),
                    "gave_up_count": 0,
                },
                failure_counts={"total": 2},
                recent_failures=[
                    _failure_metadata(
                        "doc-partial",
                        "MILVUS_WRITE_DOCUMENT",
                        "MILVUS_WRITE_ERROR",
                        now,
                    ),
                    _failure_metadata(
                        "doc-partial",
                        "IMAGE_DISCOVER",
                        "TOKEN_LEAK_CODE",
                        now,
                    ),
                ],
            ),
            "doc-gave-up": _document_metadata_with_status(
                "GAVE_UP",
                updated_at=now,
                retry_summary={
                    "has_pending_failure": False,
                    "pending_stage_count": 0,
                    "next_retry_at": None,
                    "gave_up_count": 1,
                },
                failure_counts={"total": 1},
                recent_failures=[
                    _failure_metadata(
                        "doc-gave-up",
                        "FULLTEXT_CHUNK_PIPELINE",
                        "CHUNK_PIPELINE_ERROR",
                        now,
                        status="GAVE_UP",
                    )
                ],
            ),
            "doc-deleted": _document_metadata_with_status(
                "DELETED",
                current_stage="DELETE_IMAGES",
                last_success_stage="DELETE_IMAGES",
                updated_at=now,
            ),
        }
    )
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)

    client = TestClient(app)
    try:
        success = client.get(
            "/admin/diagnostics/document/doc-success/recovery-plan"
        ).json()
        assert success["plan_status"] == "no_action_required"
        assert "no_action_required" in _recovery_action_ids(success)
        assert "prepare_supervised_replay" not in _recovery_action_ids(success)

        partial_response = client.get(
            "/admin/diagnostics/document/doc-partial/recovery-plan"
        )
        partial = partial_response.json()
        partial_ids = _recovery_action_ids(partial)
        assert partial["plan_status"] == "action_recommended"
        assert {
            "inspect_recent_failures",
            "wait_for_scheduled_retry",
            "pause_drain_worker_before_manual_intervention",
            "run_data_plane_consistency_verifier",
            "prepare_supervised_replay",
        }.issubset(partial_ids)
        assert len(partial["suggested_actions"]) <= 8
        assert all(action["mutating"] is False for action in partial["suggested_actions"])
        assert "TOKEN_LEAK_CODE" not in partial_response.text

        gave_up = client.get(
            "/admin/diagnostics/document/doc-gave-up/recovery-plan"
        ).json()
        gave_up_ids = _recovery_action_ids(gave_up)
        assert "prepare_supervised_replay" in gave_up_ids
        assert "escalate_manual_review" in gave_up_ids

        deleted = client.get(
            "/admin/diagnostics/document/doc-deleted/recovery-plan"
        ).json()
        deleted_ids = _recovery_action_ids(deleted)
        assert "run_data_plane_consistency_verifier" in deleted_ids
        assert "prepare_supervised_replay" not in deleted_ids

        metrics_text = client.get("/metrics").text
        assert 'status="recovery_plan_success"' in metrics_text
        assert "doc-partial" not in metrics_text
    finally:
        client.close()


def test_document_diagnostics_summarizes_bounds_and_redacts(monkeypatch):
    now = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    metadata = {
        "status": {
            "_id": "doc-rich",
            "doc_id": "doc-rich",
            "overall_status": "PARTIAL_FAILURE",
            "current_stage": "MILVUS_WRITE_DOCUMENT",
            "last_success_stage": "DOC_EMBED_KEYWORD",
            "latest_record_update_time": "2026-06-07T12:00:00Z",
            "latest_event_type": "UPDATE",
            "updated_at": now,
            "stage_statuses": {
                "MILVUS_WRITE_DOCUMENT": {
                    "status": "FAILURE",
                    "attempt_count": 2,
                    "started_at": now,
                    "finished_at": now + timedelta(seconds=1),
                    "updated_at": now + timedelta(seconds=1),
                    "duration_ms": 1000,
                    "artifact_ref": "artifact-stage",
                    "last_error_code": "MILVUS_WRITE_ERROR",
                    "last_error_message": "Bearer stage-secret",
                    "visual_embed": [0.1],
                }
            },
            "progress": {
                "article_count": 1,
                "document_vectors_written": 0,
                "raw_fulltext": "fulltext-secret",
                "visual_embed": [0.1],
            },
            "retry_summary": {
                "has_pending_failure": True,
                "pending_stage_count": 2,
                "next_retry_at": now + timedelta(minutes=5),
                "gave_up_count": 1,
                "token": "retry-secret",
            },
        },
        "failure_counts": {
            "total": 25,
            "by_status": {"PENDING": 25},
            "by_stage": {"MILVUS_WRITE_DOCUMENT": 25},
            "by_error_type": {"RETRYABLE": 25},
        },
        "recent_failures": [
            {
                "failure_key": f"failure-{index}",
                "doc_id": "doc-rich",
                "record_update_time": "2026-06-07T12:00:00Z",
                "sub_doc_type": "J02",
                "event_type": "UPDATE",
                "failure_stage": "MILVUS_WRITE_DOCUMENT",
                "scope_type": "document",
                "scope_id": "doc-rich",
                "status": "PENDING",
                "error_type": "RETRYABLE",
                "error_code": "MILVUS_WRITE_ERROR",
                "error_message": "password failure-secret",
                "exception_class": "RuntimeError",
                "stacktrace": "Authorization: Bearer failure-secret",
                "retry_count": index,
                "max_retries": 3,
                "next_retry_at": now,
                "last_retry_at": now,
                "updated_at": now,
                "created_at": now,
                "artifact_refs": [f"artifact-{index}", "artifact-extra"],
                "source_event_meta": {
                    "topic": "VECTOR_JOURNAL_ARTICLE",
                    "partition": 1,
                    "offset": 42,
                    "event_fingerprint": "fingerprint",
                    "Authorization": "Bearer source-secret",
                },
                "visual_embed": [0.1],
                "text_sparse_vec": {"0": 1.0},
            }
            for index in range(25)
        ],
        "artifact_counts": {
            "total": 25,
            "by_type": {"RAW_EVENT_JSON": 25},
            "by_stage": {"KAFKA_DECODE": 25},
        },
        "recent_artifacts": [
            {
                "_id": f"artifact-{index}",
                "doc_id": "doc-rich",
                "record_update_time": "2026-06-07T12:00:00Z",
                "sub_doc_type": "J02",
                "stage": "KAFKA_DECODE",
                "scope_type": "document",
                "scope_id": "doc-rich",
                "artifact_type": "RAW_EVENT_JSON",
                "content_type": "application/json",
                "sha256": f"sha-{index}",
                "size_bytes": 100 + index,
                "compression": None,
                "created_at": now,
                "expire_at": now + timedelta(days=1),
                "meta": {"Authorization": "Bearer artifact-secret"},
                "payload_json": {"token": "artifact-secret"},
                "payload_text": "raw image bytes and fulltext-secret",
                "image_record": {"visual_embed": [0.1]},
                "visual_embed": [0.1],
            }
            for index in range(25)
        ],
    }
    manager = FakeDocumentDiagnosticsManager(metadata)
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)

    client = TestClient(app)
    try:
        response = client.get("/admin/diagnostics/document/doc-rich")
        assert response.status_code == 200
        payload = response.json()
        assert payload["diagnostic_status"] == "success"
        assert payload["status_found"] is True
        assert payload["overall_status"] == "PARTIAL_FAILURE"
        assert payload["current_stage"] == "MILVUS_WRITE_DOCUMENT"
        assert payload["last_success_stage"] == "DOC_EMBED_KEYWORD"
        assert payload["progress"] == {
            "article_count": 1,
            "document_vectors_written": 0,
        }
        assert payload["retry_summary"]["has_pending_failure"] is True
        assert len(payload["recent_failures"]) == 20
        assert len(payload["recent_artifact_refs"]) == 20
        assert payload["failure_counts"]["total"] == 25
        assert payload["artifact_counts"]["total"] == 25

        stage_summary = payload["stage_summaries"]["MILVUS_WRITE_DOCUMENT"]
        assert stage_summary["last_error_code"] == "MILVUS_WRITE_ERROR"
        assert "last_error_message" not in stage_summary
        assert "visual_embed" not in stage_summary

        recent_failure = payload["recent_failures"][0]
        assert recent_failure["artifact_ref_count"] == 2
        assert "error_message" not in recent_failure
        assert "stacktrace" not in recent_failure
        assert "visual_embed" not in recent_failure
        assert "text_sparse_vec" not in recent_failure
        assert "Authorization" not in recent_failure["source_event_meta"]

        recent_artifact = payload["recent_artifact_refs"][0]
        assert recent_artifact["artifact_ref"] == "artifact-0"
        assert "payload_json" not in recent_artifact
        assert "payload_text" not in recent_artifact
        assert "payload_gzip" not in recent_artifact
        assert "meta" not in recent_artifact
        assert "image_record" not in recent_artifact
        assert "visual_embed" not in recent_artifact

        rendered = response.text
        for forbidden in (
            "stage-secret",
            "fulltext-secret",
            "retry-secret",
            "failure-secret",
            "source-secret",
            "artifact-secret",
            "raw image bytes",
            "visual_embed",
            "text_sparse_vec",
            "image_record",
            "payload_json",
            "Authorization",
        ):
            assert forbidden not in rendered

        metrics_text = client.get("/metrics").text
        assert 'status="document_success"' in metrics_text
        assert "doc-rich" not in metrics_text
    finally:
        client.close()


def test_document_diagnostics_has_no_live_client_side_effects(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("document diagnostics must not create live clients")

    manager = FakeDocumentDiagnosticsManager(_empty_document_metadata())
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: manager)
    monkeypatch.setattr("app.main.get_mongo_state_manager", forbidden)
    monkeypatch.setattr("app.main.get_milvus_service", forbidden)
    monkeypatch.setattr("app.main.get_embedding_client", forbidden)
    monkeypatch.setattr("app.main.get_fulltext_worker_runner", forbidden)
    monkeypatch.setattr("app.main.get_image_worker_runner", forbidden)
    monkeypatch.setattr(
        fulltext_downloader_module.FullTextDownloader,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(fulltext_worker_module.FullTextWorker, "__init__", forbidden)
    monkeypatch.setattr(
        image_downloader_module.ImageDownloader,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        milvus_image_service_module.MilvusImageService,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        milvus_chunk_service_module.MilvusChunkService,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(image_feature_pipeline_module, "ImageFeatureProviders", forbidden)
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)
    state.fulltext_worker_runner = None
    state.fulltext_worker_task = None
    state.image_worker_runner = None
    state.image_worker_task = None

    client = TestClient(app)
    try:
        response = client.get("/admin/diagnostics/document/doc-side-effect")
        assert response.status_code == 200
        assert response.json()["diagnostic_status"] == "not_found"
        assert manager.calls == [("doc-side-effect", 20)]

        manager.calls.clear()
        recovery_response = client.get(
            "/admin/diagnostics/document/doc-side-effect/recovery-plan"
        )
        assert recovery_response.status_code == 200
        assert recovery_response.json()["plan_status"] == "no_status"
        assert manager.calls == [("doc-side-effect", 20)]
    finally:
        client.close()


def test_document_recovery_plan_does_not_add_mutating_endpoints(monkeypatch):
    manager = FakeDocumentDiagnosticsManager(_empty_document_metadata())
    monkeypatch.setattr("app.main.get_existing_mongo_state_manager", lambda: manager)
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)

    client = TestClient(app)
    try:
        response = client.post(
            "/admin/diagnostics/document/doc-readonly/recovery-plan"
        )
        assert response.status_code == 405

        for suffix in ("replay", "skip", "give-up"):
            mutating_response = client.post(
                f"/admin/diagnostics/document/doc-readonly/{suffix}"
            )
            assert mutating_response.status_code in {404, 405}
    finally:
        client.close()


def _assert_image_runtime_default_off(image_status: dict) -> None:
    assert image_status["task_upsert_enabled"] is settings.IMAGE_TASK_UPSERT_ENABLED
    assert image_status["worker_enabled"] is settings.IMAGE_WORKER_ENABLED
    assert image_status["worker_running"] is False
    assert image_status["feature_pipeline_enabled"] is settings.IMAGE_FEATURE_PIPELINE_ENABLED
    assert image_status["milvus_write_enabled"] is settings.IMAGE_MILVUS_WRITE_ENABLED


def _assert_fulltext_runtime_default_off(fulltext_status: dict) -> None:
    assert fulltext_status["worker_enabled"] is settings.FULLTEXT_WORKER_ENABLED
    assert fulltext_status["worker_running"] is False
    assert fulltext_status["paused"] is False
    assert fulltext_status["draining"] is False


class FakeDocumentDiagnosticsManager:
    def __init__(self, metadata: dict, error: Exception | None = None) -> None:
        self.metadata = metadata
        self.error = error
        self.connected = True
        self.calls: list[tuple[str, int]] = []

    async def get_document_diagnostics_metadata(
        self,
        doc_id: str,
        *,
        recent_limit: int,
    ) -> dict:
        self.calls.append((doc_id, recent_limit))
        if self.error is not None:
            raise self.error
        return self.metadata


class FakeDocumentDiagnosticsByDocManager:
    def __init__(self, metadata_by_doc: dict[str, dict]) -> None:
        self.metadata_by_doc = metadata_by_doc
        self.connected = True
        self.calls: list[tuple[str, int]] = []

    async def get_document_diagnostics_metadata(
        self,
        doc_id: str,
        *,
        recent_limit: int,
    ) -> dict:
        self.calls.append((doc_id, recent_limit))
        return self.metadata_by_doc.get(doc_id, _empty_document_metadata())


def _empty_document_metadata() -> dict:
    return {
        "status": None,
        "failure_counts": {
            "total": 0,
            "by_status": {},
            "by_stage": {},
            "by_error_type": {},
        },
        "recent_failures": [],
        "artifact_counts": {
            "total": 0,
            "by_type": {},
            "by_stage": {},
        },
        "recent_artifacts": [],
    }


def _document_metadata_with_status(
    overall_status: str,
    *,
    current_stage: str = "MILVUS_WRITE_DOCUMENT",
    last_success_stage: str = "MILVUS_WRITE_DOCUMENT",
    updated_at: datetime,
    retry_summary: dict | None = None,
    failure_counts: dict | None = None,
    recent_failures: list[dict] | None = None,
) -> dict:
    failure_total = int((failure_counts or {}).get("total", 0))
    return {
        "status": {
            "_id": "doc-status",
            "doc_id": "doc-status",
            "overall_status": overall_status,
            "current_stage": current_stage,
            "last_success_stage": last_success_stage,
            "latest_record_update_time": "2026-06-07T12:00:00Z",
            "latest_event_type": "UPDATE",
            "updated_at": updated_at,
            "stage_statuses": {
                current_stage: {
                    "status": overall_status,
                    "attempt_count": 1,
                    "updated_at": updated_at,
                }
            },
            "progress": {"document_vectors_written": 1},
            "retry_summary": retry_summary
            or {
                "has_pending_failure": False,
                "pending_stage_count": 0,
                "next_retry_at": None,
                "gave_up_count": 0,
            },
        },
        "failure_counts": {
            "total": failure_total,
            "by_status": (failure_counts or {}).get("by_status", {}),
            "by_stage": (failure_counts or {}).get("by_stage", {}),
            "by_error_type": (failure_counts or {}).get("by_error_type", {}),
        },
        "recent_failures": recent_failures or [],
        "artifact_counts": {
            "total": 0,
            "by_type": {},
            "by_stage": {},
        },
        "recent_artifacts": [],
    }


def _failure_metadata(
    doc_id: str,
    stage: str,
    error_code: str,
    updated_at: datetime,
    *,
    status: str = "PENDING",
) -> dict:
    return {
        "failure_key": f"{doc_id}|{stage}",
        "doc_id": doc_id,
        "record_update_time": "2026-06-07T12:00:00Z",
        "sub_doc_type": "J02",
        "event_type": "UPDATE",
        "failure_stage": stage,
        "scope_type": "document",
        "scope_id": doc_id,
        "status": status,
        "error_type": "RETRYABLE",
        "error_code": error_code,
        "exception_class": "RuntimeError",
        "retry_count": 1,
        "max_retries": 3,
        "next_retry_at": updated_at + timedelta(minutes=10),
        "last_retry_at": updated_at,
        "updated_at": updated_at,
        "created_at": updated_at,
        "error_message": "raw secret should not appear",
        "stacktrace": "Authorization: Bearer stack-secret",
    }


def _recovery_action_ids(payload: dict) -> set[str]:
    return {
        action["action_id"]
        for action in payload.get("suggested_actions", [])
        if isinstance(action, dict)
    }
