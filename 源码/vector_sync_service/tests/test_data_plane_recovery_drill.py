from pathlib import Path

from app.services.data_plane_recovery_drill import build_recovery_drill_recommendations


def _codes(result):
    return {item["code"] for item in result["recommendations"]}


def test_recovery_drill_no_mismatch_returns_informational_no_action():
    result = build_recovery_drill_recommendations(
        {
            "doc_id": "SMOKE_DOC_round4_001",
            "sys_id": "SMOKE_SYS_round4_001",
            "mismatch_flags": {"mongo_status_missing": False},
            "has_mismatch": False,
        }
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["mutation_allowed"] is False
    assert result["has_blocker"] is False
    assert result["identity"]["doc_id"]["value"] == "SMOKE_DOC_round4_001"
    assert result["identity"]["sys_id"]["value"] == "SMOKE_SYS_round4_001"
    assert _codes(result) == {"NO_MISMATCH_FOUND"}


def test_recovery_drill_collection_collisions_are_blockers():
    result = build_recovery_drill_recommendations(
        {
            "doc_id": "SMOKE_DOC_round4_002",
            "mismatch_flags": {
                "chunk_collection_points_to_document_collection": True,
                "image_collection_points_to_document_collection": True,
            },
        }
    )

    assert result["has_blocker"] is True
    assert "VERIFY_COLLECTION_CONFIGURATION" in _codes(result)
    assert result["mutation_allowed"] is False


def test_recovery_drill_active_missing_rows_map_to_inspection_recommendations():
    result = build_recovery_drill_recommendations(
        {
            "doc_id": "SMOKE_DOC_round4_003",
            "mismatch_flags": {
                "active_status_missing_expected_document_rows": True,
                "active_status_missing_expected_chunk_rows": True,
                "active_status_missing_expected_image_rows": True,
            },
        }
    )

    assert _codes(result) == {"INSPECT_DOCUMENT_ROWS", "INSPECT_CHUNK_ROWS", "INSPECT_IMAGE_ROWS"}
    rendered_steps = repr(result["recommendations"])
    assert "Mongo" in rendered_steps
    assert "chunk" in rendered_steps
    assert "image" in rendered_steps


def test_recovery_drill_deleted_retaining_rows_map_to_cleanup_recommendations_once_per_alias():
    result = build_recovery_drill_recommendations(
        {
            "doc_id": "SMOKE_DOC_round4_004",
            "mismatch_flags": {
                "deleted_status_has_document_rows": True,
                "deleted_status_retaining_document_rows": True,
                "deleted_status_has_chunk_rows": True,
                "deleted_status_retaining_chunk_rows": True,
            },
        }
    )

    assert result["active_flags"] == [
        "deleted_status_retaining_document_rows",
        "deleted_status_retaining_chunk_rows",
    ]
    assert _codes(result) == {"VERIFY_DELETE_CLEANUP"}


def test_recovery_drill_partial_failure_without_failure_records_escalates():
    result = build_recovery_drill_recommendations(
        {
            "doc_id": "SMOKE_DOC_round4_005",
            "mismatch_flags": {"partial_failure_has_no_failure_records": True},
        }
    )

    assert result["has_blocker"] is True
    assert _codes(result) == {"ESCALATE_PARTIAL_FAILURE_WITHOUT_RECORD"}


def test_recovery_drill_unsafe_identity_values_are_omitted():
    raw_doc_id = "PROD_DOC_should_not_be_echoed"
    raw_sys_id = "PROD_SYS_should_not_be_echoed"

    result = build_recovery_drill_recommendations(
        {
            "doc_id": raw_doc_id,
            "sys_id": raw_sys_id,
            "mismatch_flags": {"mongo_status_missing": True},
        }
    )
    rendered = repr(result)

    assert result["identity"]["doc_id"]["provided"] is True
    assert result["identity"]["doc_id"]["safe"] is False
    assert result["identity"]["doc_id"]["value"] is None
    assert result["identity"]["sys_id"]["value"] is None
    assert raw_doc_id not in rendered
    assert raw_sys_id not in rendered
    assert _codes(result) == {"INSPECT_MONGO_STATUS"}


def test_recovery_drill_source_does_not_reference_live_clients_or_providers():
    source_path = Path(__file__).parents[1] / "app" / "services" / "data_plane_recovery_drill.py"
    source = source_path.read_text(encoding="utf-8")

    forbidden = [
        "MongoStateManager",
        "MilvusClient",
        "EmbeddingClient",
        "get_embedding_client",
        "httpx",
        "requests",
        "openai",
        "allow_live_clients=True",
    ]
    for keyword in forbidden:
        assert keyword not in source
