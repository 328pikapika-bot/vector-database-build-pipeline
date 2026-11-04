import pytest

from app.services.data_plane_smoke import (
    GLOBAL_LIVE_SMOKE_ENV,
    LIVE_SMOKE_COMPONENT_ENVS,
    LIVE_SMOKE_DOC_ID_ENV,
    LIVE_SMOKE_SYS_ID_ENV,
    build_live_smoke_preflight,
    build_staging_smoke_plan,
    get_live_smoke_gate,
    is_safe_smoke_id,
    is_live_smoke_enabled,
    live_smoke_skip_reason,
    mask_secret_value,
    sanitize_smoke_payload,
)


def _blocker_codes(preflight):
    return {item["code"] for item in preflight["preflight_blockers"]}


def test_live_smoke_gate_skips_by_default():
    gate = get_live_smoke_gate("mongo", environ={})

    assert gate.enabled is False
    assert gate.global_enabled is False
    assert gate.component_enabled is False
    assert GLOBAL_LIVE_SMOKE_ENV in gate.skip_reason
    assert live_smoke_skip_reason("mongo", environ={}) == gate.skip_reason


@pytest.mark.parametrize("component, env_name", sorted(LIVE_SMOKE_COMPONENT_ENVS.items()))
def test_live_smoke_gate_requires_global_and_component_flag(component, env_name):
    assert is_live_smoke_enabled(component, environ={GLOBAL_LIVE_SMOKE_ENV: "1"}) is False
    assert is_live_smoke_enabled(component, environ={env_name: "1"}) is False

    gate = get_live_smoke_gate(
        component,
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "1",
            env_name: "1",
        },
    )

    assert gate.enabled is True
    assert gate.global_enabled is True
    assert gate.component_enabled is True
    assert gate.skip_reason is None


def test_live_smoke_gate_uses_exact_enabled_value():
    env_name = LIVE_SMOKE_COMPONENT_ENVS["milvus"]

    gate = get_live_smoke_gate(
        "milvus",
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "true",
            env_name: "true",
        },
    )

    assert gate.enabled is False
    assert GLOBAL_LIVE_SMOKE_ENV in gate.skip_reason


def test_live_smoke_gate_rejects_unknown_component():
    with pytest.raises(ValueError, match="unsupported live smoke component"):
        get_live_smoke_gate("unknown", environ={})


def test_mask_secret_value_never_returns_raw_value():
    assert mask_secret_value("secret-value") == "<redacted>"
    assert mask_secret_value(b"raw image bytes") == "<redacted-bytes>"


def test_sanitize_smoke_payload_masks_secrets_and_raw_bytes_without_hiding_safe_fields():
    payload = {
        "MONGO_URI": "mongodb://user:pass@host/db",
        "EMBEDDING_API_KEY": "embedding-secret",
        "Authorization": "Bearer token",
        "safe_url": "http://example.test/health",
        "size_bytes": 2048,
        "nested": {
            "password": "pw",
            "imageBytes": b"image-data",
            "items": [
                {"token": "abc"},
                {"value": "plain"},
            ],
        },
    }

    sanitized = sanitize_smoke_payload(payload)
    rendered = repr(sanitized)

    assert sanitized["MONGO_URI"] == "<redacted>"
    assert sanitized["EMBEDDING_API_KEY"] == "<redacted>"
    assert sanitized["Authorization"] == "<redacted>"
    assert sanitized["safe_url"] == "http://example.test/health"
    assert sanitized["size_bytes"] == 2048
    assert sanitized["nested"]["password"] == "<redacted>"
    assert sanitized["nested"]["imageBytes"] == "<redacted-bytes>"
    assert sanitized["nested"]["items"][0]["token"] == "<redacted>"
    assert sanitized["nested"]["items"][1]["value"] == "plain"
    assert "mongodb://user:pass@host/db" not in rendered
    assert "embedding-secret" not in rendered
    assert "Bearer token" not in rendered
    assert "image-data" not in rendered


def test_staging_smoke_plan_defaults_to_skipped_without_live_clients():
    plan = build_staging_smoke_plan(environ={})

    assert plan["live_calls_allowed"] is False
    assert sorted(plan["planned_components"]) == sorted(LIVE_SMOKE_COMPONENT_ENVS)
    assert GLOBAL_LIVE_SMOKE_ENV in plan["required_env_names"]
    assert LIVE_SMOKE_DOC_ID_ENV == plan["required_safe_id_env_names"]["doc_id"]
    assert LIVE_SMOKE_SYS_ID_ENV == plan["required_safe_id_env_names"]["sys_id"]
    assert set(plan["skip_reasons"]) == set(LIVE_SMOKE_COMPONENT_ENVS)
    assert plan["safe_ids"]["doc_id"]["provided"] is False
    assert plan["safe_ids"]["doc_id"]["safe"] is False
    assert plan["consistency"]["ready"] is False
    assert GLOBAL_LIVE_SMOKE_ENV in plan["consistency"]["skip_reason"]


def test_staging_smoke_plan_requires_component_gates_and_safe_doc_id():
    env = {
        GLOBAL_LIVE_SMOKE_ENV: "1",
        LIVE_SMOKE_COMPONENT_ENVS["mongo"]: "1",
        LIVE_SMOKE_COMPONENT_ENVS["milvus"]: "1",
        LIVE_SMOKE_DOC_ID_ENV: "SMOKE_DOC_round3_001",
    }

    plan = build_staging_smoke_plan(
        ["mongo", "milvus"],
        environ=env,
        expect_document=True,
        expect_chunks=True,
        expect_images=False,
    )

    assert plan["components"]["mongo"]["enabled"] is True
    assert plan["components"]["milvus"]["enabled"] is True
    assert plan["skip_reasons"] == {}
    assert plan["safe_ids"]["doc_id"]["safe"] is True
    assert plan["safe_ids"]["doc_id"]["value"] == "SMOKE_DOC_round3_001"
    assert plan["safe_ids"]["sys_id"]["required"] is False
    assert plan["expectations"] == {"document": True, "chunks": True, "images": False}
    assert plan["consistency"]["ready"] is True
    assert plan["consistency"]["skip_reason"] is None


def test_safe_smoke_id_validation_accepts_only_explicit_smoke_prefixes():
    assert is_safe_smoke_id("SMOKE_DOC_abc") is True
    assert is_safe_smoke_id("SMOKE_SYS_abc") is True
    assert is_safe_smoke_id("PROD_DOC_abc") is False
    assert is_safe_smoke_id("") is False
    assert is_safe_smoke_id("SMOKE_SYS_abc", allowed_prefixes=("SMOKE_DOC_",)) is False


def test_staging_smoke_plan_sanitizes_display_details_and_hides_unsafe_ids():
    plan = build_staging_smoke_plan(
        ["mongo", "milvus"],
        environ={GLOBAL_LIVE_SMOKE_ENV: "1"},
        safe_doc_id="PROD_DOC_should_not_be_echoed",
        display_details={
            "MONGO_URI": "mongodb://user:pass@host/db",
            "Authorization": "Bearer token",
            "imageBytes": b"raw-image",
            "safe_note": "staging only",
        },
    )
    rendered = repr(plan)

    assert plan["safe_ids"]["doc_id"]["provided"] is True
    assert plan["safe_ids"]["doc_id"]["safe"] is False
    assert plan["safe_ids"]["doc_id"]["value"] is None
    assert plan["display"]["details"]["MONGO_URI"] == "<redacted>"
    assert plan["display"]["details"]["Authorization"] == "<redacted>"
    assert plan["display"]["details"]["imageBytes"] == "<redacted-bytes>"
    assert plan["display"]["details"]["safe_note"] == "staging only"
    assert "mongodb://user:pass@host/db" not in rendered
    assert "Bearer token" not in rendered
    assert "raw-image" not in rendered
    assert "PROD_DOC_should_not_be_echoed" not in rendered


def test_live_smoke_preflight_defaults_to_no_live_with_blockers():
    preflight = build_live_smoke_preflight(environ={})

    assert preflight["success"] is True
    assert preflight["read_only"] is True
    assert preflight["mutation_allowed"] is False
    assert preflight["live_calls_allowed"] is False
    assert preflight["would_run"] == []
    assert preflight["has_blocker"] is True
    assert "LIVE_SMOKE_GATE_DISABLED" in _blocker_codes(preflight)
    assert "SAFE_DOC_ID_MISSING_OR_UNSAFE" in _blocker_codes(preflight)
    assert "REQUIRED_ENV_MISSING" in _blocker_codes(preflight)
    vl_required_blocker = next(
        item
        for item in preflight["components"]["vl"]["blockers"]
        if item["code"] == "REQUIRED_ENV_MISSING"
    )
    assert "IMAGE_TEXT_EMBED_URL" in vl_required_blocker["required_env_names"]
    assert "IMAGE_VL_EMBED_URL" not in vl_required_blocker["required_env_names"]
    assert "MongoStateManager" not in repr(preflight)
    assert "MilvusClient" not in repr(preflight)


def test_live_smoke_preflight_reports_missing_credentials_as_blockers_not_exceptions():
    preflight = build_live_smoke_preflight(
        ["mongo"],
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "1",
            LIVE_SMOKE_COMPONENT_ENVS["mongo"]: "1",
            LIVE_SMOKE_DOC_ID_ENV: "SMOKE_DOC_round5_001",
        },
    )

    assert preflight["components"]["mongo"]["gate"]["enabled"] is True
    assert preflight["would_run"] == []
    assert "REQUIRED_ENV_MISSING" in _blocker_codes(preflight)
    mongo_blockers = preflight["components"]["mongo"]["blockers"]
    assert mongo_blockers[0]["required_env_names"] == ["MONGO_URI", "MONGO_DB"]


def test_live_smoke_preflight_accepts_explicit_vl_provider_config():
    preflight = build_live_smoke_preflight(
        ["vl"],
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "1",
            LIVE_SMOKE_COMPONENT_ENVS["vl"]: "1",
            LIVE_SMOKE_DOC_ID_ENV: "SMOKE_DOC_round6_001",
            "IMAGE_VL_EMBED_API_KEY": "vl-secret",
            "IMAGE_TEXT_EMBED_API_KEY": "text-secret",
            "IMAGE_VL_EMBED_URL": "http://models.test/vl-embed",
            "IMAGE_VL_EMBED_MODEL": "8b",
            "IMAGE_TEXT_EMBED_URL": "http://models.test/text-embed",
            "IMAGE_TEXT_EMBED_MODEL": "medium",
            "IMAGE_FEATURE_TIMEOUT": "60",
            "IMAGE_FEATURE_MAX_RETRIES": "3",
        },
    )

    assert preflight["components"]["vl"]["blockers"] == []
    assert "VL_PROVIDER_CONFIG_MISSING" not in _blocker_codes(preflight)


def test_live_smoke_preflight_accepts_safe_ids_and_complete_mongo_env_without_echoing_secret_uri():
    raw_uri = "mongodb://user:secret@mongo.example/staging"
    preflight = build_live_smoke_preflight(
        ["mongo"],
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "1",
            LIVE_SMOKE_COMPONENT_ENVS["mongo"]: "1",
            LIVE_SMOKE_DOC_ID_ENV: "SMOKE_DOC_round5_002",
            LIVE_SMOKE_SYS_ID_ENV: "SMOKE_SYS_round5_002",
            "MONGO_URI": raw_uri,
            "MONGO_DB": "nstl_stage",
        },
    )
    rendered = repr(preflight)

    assert preflight["safe_ids"]["doc_id"]["safe"] is True
    assert preflight["safe_ids"]["sys_id"]["safe"] is True
    assert preflight["would_run"] == ["mongo"]
    assert preflight["has_blocker"] is False
    assert raw_uri not in rendered
    assert "secret" not in rendered


def test_live_smoke_preflight_marks_unsafe_ids_without_echoing_raw_values():
    raw_id = "PROD_DOC_should_not_be_echoed"
    preflight = build_live_smoke_preflight(
        ["mongo"],
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "1",
            LIVE_SMOKE_COMPONENT_ENVS["mongo"]: "1",
            "MONGO_URI": "mongodb://user:pass@mongo.example/staging",
            "MONGO_DB": "nstl_stage",
        },
        safe_doc_id=raw_id,
    )
    rendered = repr(preflight)

    assert preflight["safe_ids"]["doc_id"]["provided"] is True
    assert preflight["safe_ids"]["doc_id"]["safe"] is False
    assert preflight["safe_ids"]["doc_id"]["value"] is None
    assert "SAFE_DOC_ID_MISSING_OR_UNSAFE" in _blocker_codes(preflight)
    assert raw_id not in rendered


def test_live_smoke_preflight_flags_collection_collisions():
    preflight = build_live_smoke_preflight(
        ["mongo"],
        environ={
            GLOBAL_LIVE_SMOKE_ENV: "1",
            LIVE_SMOKE_COMPONENT_ENVS["mongo"]: "1",
            LIVE_SMOKE_DOC_ID_ENV: "SMOKE_DOC_round5_003",
            "MONGO_URI": "mongodb://user:pass@mongo.example/staging",
            "MONGO_DB": "nstl_stage",
        },
        document_collection="same",
        chunk_collection="same",
        image_collection="same",
    )

    assert "chunk_collection_points_to_document_collection" in _blocker_codes(preflight)
    assert "image_collection_points_to_document_collection" in _blocker_codes(preflight)
    assert "image_collection_points_to_chunk_collection" in _blocker_codes(preflight)


def test_live_smoke_preflight_sanitizes_display_details():
    preflight = build_live_smoke_preflight(
        ["mongo"],
        environ={},
        display_details={
            "MONGO_URI": "mongodb://user:pass@host/db",
            "Authorization": "Bearer token",
            "imageBytes": b"raw-image",
            "safe_note": "dry run",
        },
    )
    rendered = repr(preflight)

    assert preflight["display"]["details"]["MONGO_URI"] == "<redacted>"
    assert preflight["display"]["details"]["Authorization"] == "<redacted>"
    assert preflight["display"]["details"]["imageBytes"] == "<redacted-bytes>"
    assert preflight["display"]["details"]["safe_note"] == "dry run"
    assert "mongodb://user:pass@host/db" not in rendered
    assert "Bearer token" not in rendered
    assert "raw-image" not in rendered
