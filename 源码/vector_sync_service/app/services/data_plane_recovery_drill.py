"""Pure recovery-drill recommendations for data-plane consistency results."""

from __future__ import annotations

from typing import Any, Mapping

from app.services.data_plane_smoke import is_safe_smoke_id


_DELETED_ALIAS_FLAGS = {
    "deleted_status_has_document_rows": "deleted_status_retaining_document_rows",
    "deleted_status_has_chunk_rows": "deleted_status_retaining_chunk_rows",
    "deleted_status_has_image_rows": "deleted_status_retaining_image_rows",
}


def build_recovery_drill_recommendations(consistency_result: Mapping[str, Any]) -> dict[str, Any]:
    """Map verifier mismatch flags to safe staged recommendations without side effects."""
    mismatch_flags = consistency_result.get("mismatch_flags") or {}
    active_flags = _active_flags(mismatch_flags)
    recommendations = [_recommendation_for_flag(flag) for flag in active_flags]
    recommendations = [item for item in recommendations if item is not None]
    if not recommendations:
        recommendations = [
            {
                "code": "NO_MISMATCH_FOUND",
                "severity": "info",
                "title": "No data-plane mismatch is indicated by the verifier result.",
                "next_steps": [
                    "Record the sanitized verifier output with the staging drill notes.",
                    "Do not run manual cleanup or replay unless a separate operator signal requires it.",
                ],
            }
        ]

    return {
        "success": True,
        "read_only": True,
        "mutation_allowed": False,
        "identity": {
            "doc_id": _safe_identity(consistency_result.get("doc_id")),
            "sys_id": _safe_identity(consistency_result.get("sys_id")),
        },
        "active_flags": active_flags,
        "recommendations": recommendations,
        "has_blocker": any(item.get("severity") == "blocker" for item in recommendations),
    }


def _active_flags(mismatch_flags: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    for flag, value in mismatch_flags.items():
        if not bool(value):
            continue
        alias = _DELETED_ALIAS_FLAGS.get(flag)
        if alias and bool(mismatch_flags.get(alias)):
            continue
        flags.append(str(flag))
    return flags


def _safe_identity(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    safe = is_safe_smoke_id(text)
    return {
        "provided": bool(text),
        "safe": safe,
        "value": text if safe else None,
    }


def _recommendation_for_flag(flag: str) -> dict[str, Any] | None:
    if flag in {
        "chunk_collection_points_to_document_collection",
        "image_collection_points_to_document_collection",
        "image_collection_points_to_chunk_collection",
    }:
        return {
            "code": "VERIFY_COLLECTION_CONFIGURATION",
            "severity": "blocker",
            "title": "Collection names collide across retrieval granularities.",
            "next_steps": [
                "Stop any planned live smoke or manual write action for this document.",
                "Verify document, chunk, and image collection names before retrying the drill.",
                "Escalate the configuration mismatch to the supervisor.",
            ],
        }

    if flag == "mongo_status_missing":
        return {
            "code": "INSPECT_MONGO_STATUS",
            "severity": "warning",
            "title": "Mongo status was checked but no status record was found.",
            "next_steps": [
                "Confirm the safe doc id and sys id used for the drill.",
                "Inspect Mongo status and artifact collections before taking Milvus action.",
                "Escalate if Milvus rows exist but the synchronization ledger is absent.",
            ],
        }

    if flag == "active_status_missing_expected_document_rows":
        return {
            "code": "INSPECT_DOCUMENT_ROWS",
            "severity": "warning",
            "title": "Active Mongo status is missing expected document collection rows.",
            "next_steps": [
                "Inspect Mongo current stage, last success stage, failures, and write-summary artifacts.",
                "Verify document collection count using the expected sysId filter.",
                "Pause and drain workers before any supervised replay or manual cleanup decision.",
            ],
        }

    if flag == "active_status_missing_expected_chunk_rows":
        return {
            "code": "INSPECT_CHUNK_ROWS",
            "severity": "warning",
            "title": "Active Mongo status is missing expected full-text chunk rows.",
            "next_steps": [
                "Inspect full-text chunk task status, failures, and artifacts.",
                "Verify chunk collection count using the expected doc_id filter.",
                "Do not manually backfill chunks without a supervised replay plan.",
            ],
        }

    if flag == "active_status_missing_expected_image_rows":
        return {
            "code": "INSPECT_IMAGE_ROWS",
            "severity": "warning",
            "title": "Active Mongo status is missing expected image rows.",
            "next_steps": [
                "Inspect image task status, image delete/write failures, and artifacts.",
                "Verify image collection count using the expected sysId filter.",
                "Do not call VL/image providers or write image vectors during this drill.",
            ],
        }

    if flag in {
        "deleted_status_retaining_document_rows",
        "deleted_status_retaining_chunk_rows",
        "deleted_status_retaining_image_rows",
        "deleted_status_has_document_rows",
        "deleted_status_has_chunk_rows",
        "deleted_status_has_image_rows",
    }:
        return {
            "code": "VERIFY_DELETE_CLEANUP",
            "severity": "warning",
            "title": "Mongo status is DELETED but retrieval rows remain.",
            "next_steps": [
                "Inspect DELETE_DOCUMENT, DELETE_CHUNKS, and DELETE_IMAGES artifacts and failures.",
                "Verify document, chunk, and image collection counts before cleanup.",
                "Prepare only a supervised replay or cleanup plan; do not mutate data from this helper.",
            ],
        }

    if flag == "partial_failure_has_no_failure_records":
        return {
            "code": "ESCALATE_PARTIAL_FAILURE_WITHOUT_RECORD",
            "severity": "blocker",
            "title": "Mongo status is PARTIAL_FAILURE but no durable failure record exists.",
            "next_steps": [
                "Inspect Mongo status, artifacts, and service logs for failure persistence gaps.",
                "Do not mark the document clean or run manual cleanup without supervisor review.",
                "Escalate because replay lacks a durable failure record.",
            ],
        }

    if flag in {
        "expected_no_document_rows_but_found",
        "expected_no_chunk_rows_but_found",
        "expected_no_image_rows_but_found",
    }:
        return {
            "code": "VERIFY_ZERO_ROW_EXPECTATION",
            "severity": "warning",
            "title": "Rows were found where the drill expected zero rows.",
            "next_steps": [
                "Confirm the expectation inputs for this staging verifier run.",
                "Inspect collection counts and DELETE artifacts before considering cleanup.",
                "Pause and drain related workers before any supervised manual action.",
            ],
        }

    return {
        "code": "REVIEW_MISMATCH_FLAG",
        "severity": "warning",
        "title": f"Review verifier mismatch flag: {flag}",
        "next_steps": [
            "Inspect the sanitized verifier result and related Mongo artifacts.",
            "Escalate if the flag is not covered by the current recovery drill runbook.",
        ],
    }
