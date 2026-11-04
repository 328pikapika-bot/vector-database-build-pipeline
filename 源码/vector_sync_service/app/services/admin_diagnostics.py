"""Admin diagnostics JSON builder."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.config import settings


SENSITIVE_KEY_PARTS = (
    "TOKEN",
    "PASSWORD",
    "API_KEY",
    "SECRET",
    "AUTHORIZATION",
)
SENSITIVE_KEYS = {
    "MONGO_URI",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
    "ADMIN_READ_TOKENS",
    "ADMIN_WRITE_TOKENS",
}
DOCUMENT_DIAGNOSTIC_RECENT_LIMIT = 20
DOCUMENT_RECOVERY_PLAN_MAX_ACTIONS = 8
STAGE_SUMMARY_FIELDS = (
    "status",
    "attempt_count",
    "started_at",
    "finished_at",
    "updated_at",
    "duration_ms",
    "artifact_ref",
    "last_error_code",
)
FAILURE_SUMMARY_FIELDS = (
    "failure_key",
    "doc_id",
    "record_update_time",
    "sub_doc_type",
    "event_type",
    "failure_stage",
    "scope_type",
    "scope_id",
    "status",
    "error_type",
    "error_code",
    "exception_class",
    "retry_count",
    "max_retries",
    "next_retry_at",
    "last_retry_at",
    "resolved_at",
    "updated_at",
    "created_at",
    "replay_mode",
)
ARTIFACT_REF_FIELDS = (
    "doc_id",
    "record_update_time",
    "sub_doc_type",
    "stage",
    "scope_type",
    "scope_id",
    "artifact_type",
    "content_type",
    "sha256",
    "size_bytes",
    "compression",
    "created_at",
    "expire_at",
)
PROGRESS_FIELDS = (
    "article_count",
    "chunk_count_expected",
    "chunk_count_written",
    "image_count_expected",
    "image_count_written",
    "document_vectors_written",
)
RETRY_SUMMARY_FIELDS = (
    "has_pending_failure",
    "pending_stage_count",
    "next_retry_at",
    "gave_up_count",
)


def build_admin_diagnostics(runtime_status: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded, redacted operational diagnostics payload."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_summary": _runtime_summary(runtime_status),
        "dependency_readiness": _dependency_readiness(runtime_status),
        "worker_summaries": _worker_summaries(runtime_status),
        "redacted_settings": _redacted_settings(),
    }


def build_document_recovery_plan(
    document_diagnostics: dict[str, Any],
    *,
    max_actions: int = DOCUMENT_RECOVERY_PLAN_MAX_ACTIONS,
) -> dict[str, Any]:
    """Build read-only operator recovery guidance from document diagnostics."""
    action_limit = max(1, min(int(max_actions), DOCUMENT_RECOVERY_PLAN_MAX_ACTIONS))
    diagnostic_status = _safe_token(
        document_diagnostics.get("diagnostic_status"),
        fallback="error",
    )
    status_found = bool(document_diagnostics.get("status_found"))
    overall_status = _safe_token(document_diagnostics.get("overall_status"))
    current_stage = _safe_token(document_diagnostics.get("current_stage"))
    last_success_stage = _safe_token(document_diagnostics.get("last_success_stage"))
    retry_summary = (
        document_diagnostics.get("retry_summary")
        if isinstance(document_diagnostics.get("retry_summary"), dict)
        else {}
    )
    failure_counts = (
        document_diagnostics.get("failure_counts")
        if isinstance(document_diagnostics.get("failure_counts"), dict)
        else {}
    )
    recent_failures = _bounded_list(document_diagnostics.get("recent_failures"), 20)
    failure_total = _safe_int(failure_counts.get("total"))
    artifact_counts = (
        document_diagnostics.get("artifact_counts")
        if isinstance(document_diagnostics.get("artifact_counts"), dict)
        else {}
    )

    actions: list[dict[str, Any]] = []
    if diagnostic_status == "mongo_unavailable":
        plan_status = "mongo_unavailable"
        actions.extend(
            [
                _recovery_action(
                    "pause_drain_worker_before_manual_intervention",
                    "Pause or drain affected workers before manual intervention",
                    "control",
                    "medium",
                    [
                        "Mongo diagnostics are unavailable",
                        "Pause/drain prevents new work while state access is restored",
                    ],
                    related_stages=[current_stage],
                ),
                _recovery_action(
                    "escalate_manual_review",
                    "Escalate Mongo availability to an operator",
                    "escalation",
                    "high",
                    [
                        "Document recovery cannot be assessed without Mongo metadata",
                        "Do not replay, skip, or give up while Mongo state is unavailable",
                    ],
                ),
            ]
        )
    elif diagnostic_status == "not_found" or not status_found:
        plan_status = "no_status"
        actions.extend(
            [
                _recovery_action(
                    "run_data_plane_consistency_verifier",
                    "Run read-only consistency verification for the document id",
                    "verification",
                    "medium",
                    [
                        "No Mongo status document was found",
                        "Verifier can confirm whether downstream rows unexpectedly exist",
                    ],
                ),
                _recovery_action(
                    "escalate_manual_review",
                    "Escalate if the document was expected to have sync state",
                    "escalation",
                    "medium",
                    [
                        "Missing status may indicate an unseen source event or stale id",
                    ],
                ),
            ]
        )
    else:
        normalized_status = overall_status.upper()
        if normalized_status == "SUCCESS" and failure_total == 0:
            plan_status = "no_action_required"
            actions.extend(
                [
                    _recovery_action(
                        "no_action_required",
                        "No recovery action is required",
                        "observation",
                        "high",
                        [
                            "Document status is SUCCESS",
                            "No failures are recorded in the diagnostics summary",
                        ],
                        related_stages=[current_stage, last_success_stage],
                    ),
                    _recovery_action(
                        "run_data_plane_consistency_verifier",
                        "Optionally run read-only consistency verification",
                        "verification",
                        "low",
                        [
                            "Use verifier only when operator assurance is needed",
                        ],
                    ),
                ]
            )
        elif normalized_status == "DELETED":
            plan_status = "action_recommended"
            actions.extend(
                [
                    _recovery_action(
                        "run_data_plane_consistency_verifier",
                        "Verify deleted document rows are absent downstream",
                        "verification",
                        "high",
                        [
                            "Document status is DELETED",
                            "Verifier can confirm document, chunk, and image row cleanup",
                        ],
                        related_stages=[current_stage, last_success_stage],
                    ),
                    _recovery_action(
                        "no_action_required",
                        "Do not replay deleted documents without supervision",
                        "observation",
                        "medium",
                        [
                            "Deleted state is terminal unless a supervisor approves recovery",
                        ],
                    ),
                ]
            )
        else:
            plan_status = "action_recommended"
            actions.extend(
                _failure_or_retry_actions(
                    document_diagnostics,
                    current_stage=current_stage,
                    last_success_stage=last_success_stage,
                    retry_summary=retry_summary,
                    recent_failures=recent_failures,
                    failure_total=failure_total,
                )
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "doc_id": _json_safe(document_diagnostics.get("doc_id")),
        "diagnostic_status": diagnostic_status,
        "status_found": status_found,
        "overall_status": overall_status,
        "current_stage": current_stage,
        "last_success_stage": last_success_stage,
        "plan_status": plan_status,
        "suggested_actions": actions[:action_limit],
        "safety_notes": [
            "Recovery plans are read-only guidance and do not mutate Mongo or Milvus.",
            "This endpoint does not enqueue replay, skip, give-up, delete, insert, or update actions.",
            "Inspect artifact references only through approved redacted diagnostics workflows.",
        ],
        "source_summary": {
            "latest_record_update_time": _json_safe(
                document_diagnostics.get("latest_record_update_time")
            ),
            "latest_event_type": _safe_token(
                document_diagnostics.get("latest_event_type")
            ),
            "status_updated_at": _json_safe(document_diagnostics.get("status_updated_at")),
            "retry_summary": _filter_dict_fields(retry_summary, RETRY_SUMMARY_FIELDS),
            "failure_count_total": failure_total,
            "artifact_count_total": _safe_int(artifact_counts.get("total")),
            "recent_failure_count": len(recent_failures),
            "recent_artifact_ref_count": len(
                _bounded_list(document_diagnostics.get("recent_artifact_refs"), 20)
            ),
        },
    }


async def build_document_diagnostics(
    doc_id: str,
    mongo_manager: Any,
    *,
    recent_limit: int = DOCUMENT_DIAGNOSTIC_RECENT_LIMIT,
) -> dict[str, Any]:
    """Build a redacted, metadata-only diagnostic payload for one document."""
    limit = _bounded_recent_limit(recent_limit)
    if mongo_manager is None or not bool(getattr(mongo_manager, "connected", False)):
        return _mongo_unavailable_document_diagnostics(doc_id, limit)

    try:
        metadata = await mongo_manager.get_document_diagnostics_metadata(
            doc_id,
            recent_limit=limit,
        )
    except Exception:
        return _mongo_unavailable_document_diagnostics(doc_id, limit)

    status_doc = metadata.get("status") or {}
    status_found = bool(status_doc)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        "diagnostic_status": "success" if status_found else "not_found",
        "mongo_connected": True,
        "status_found": status_found,
        "overall_status": _json_safe(status_doc.get("overall_status")),
        "current_stage": _json_safe(status_doc.get("current_stage")),
        "last_success_stage": _json_safe(status_doc.get("last_success_stage")),
        "latest_record_update_time": _json_safe(
            status_doc.get("latest_record_update_time")
        ),
        "latest_event_type": _json_safe(status_doc.get("latest_event_type")),
        "status_updated_at": _json_safe(status_doc.get("updated_at")),
        "stage_summaries": _summarize_stage_statuses(
            status_doc.get("stage_statuses") or {}
        ),
        "progress": _filter_dict_fields(
            status_doc.get("progress") or {},
            PROGRESS_FIELDS,
        ),
        "retry_summary": _filter_dict_fields(
            status_doc.get("retry_summary") or {},
            RETRY_SUMMARY_FIELDS,
        ),
        "failure_counts": _json_safe(metadata.get("failure_counts") or {}),
        "recent_failures": _summarize_failures(
            metadata.get("recent_failures") or [],
            limit,
        ),
        "artifact_counts": _json_safe(metadata.get("artifact_counts") or {}),
        "recent_artifact_refs": _summarize_artifacts(
            metadata.get("recent_artifacts") or [],
            limit,
        ),
        "recent_limit": limit,
    }


def _runtime_summary(runtime_status: dict[str, Any]) -> dict[str, Any]:
    consumer = runtime_status.get("consumer") or {}
    mongodb = runtime_status.get("mongodb") or {}
    milvus = runtime_status.get("milvus") or {}
    embedding_api = runtime_status.get("embedding_api") or {}
    return {
        "control": runtime_status.get("control") or {},
        "consumer": {
            "running": bool(consumer.get("running")),
            "configured_running": bool(consumer.get("configured_running")),
            "kafka_connected": bool(consumer.get("kafka_connected")),
            "topic": consumer.get("topic"),
            "group_id": consumer.get("group_id"),
        },
        "mongodb": {
            "connected": bool(mongodb.get("connected")),
            "database": mongodb.get("database"),
            "worker_id": mongodb.get("worker_id"),
        },
        "milvus": {
            "connected": bool(milvus.get("connected")),
            "collection": milvus.get("collection"),
        },
        "embedding_api": {
            "available": bool(embedding_api.get("available")),
        },
    }


def _dependency_readiness(runtime_status: dict[str, Any]) -> dict[str, bool]:
    fulltext = runtime_status.get("fulltext") or {}
    image = runtime_status.get("image") or {}
    return {
        "mongodb": bool((runtime_status.get("mongodb") or {}).get("connected")),
        "milvus": bool((runtime_status.get("milvus") or {}).get("connected")),
        "embedding_api": bool(
            (runtime_status.get("embedding_api") or {}).get("available")
        ),
        "fulltext_download_base": bool(fulltext.get("base_url_configured")),
        "image_download_base": bool(image.get("base_url_configured")),
        "image_pixel_validation": bool(image.get("pixel_validation_available")),
    }


def _worker_summaries(runtime_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "fulltext": runtime_status.get("fulltext") or {},
        "image": runtime_status.get("image") or {},
        "compensator": runtime_status.get("compensator") or {},
    }


def _redacted_settings() -> dict[str, Any]:
    try:
        values = settings.model_dump()
    except AttributeError:
        values = settings.dict()
    return {
        key: _redacted_value(key, value)
        for key, value in sorted(values.items())
        if key.isupper()
    }


def _redacted_value(key: str, value: Any) -> Any:
    if _is_sensitive_key(key):
        return "<redacted>" if value not in (None, "") else ""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return {
            str(item_key): _redacted_value(f"{key}_{item_key}", item_value)
            for item_key, item_value in value.items()
        }
    return str(value)


def _is_sensitive_key(key: str) -> bool:
    upper_key = key.upper()
    return upper_key in SENSITIVE_KEYS or any(
        part in upper_key for part in SENSITIVE_KEY_PARTS
    )


def _mongo_unavailable_document_diagnostics(
    doc_id: str,
    recent_limit: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        "diagnostic_status": "mongo_unavailable",
        "mongo_connected": False,
        "status_found": False,
        "error_code": "MONGO_DIAGNOSTICS_UNAVAILABLE",
        "error_message": "mongo state manager is not connected or unavailable",
        "overall_status": None,
        "current_stage": None,
        "last_success_stage": None,
        "latest_record_update_time": None,
        "latest_event_type": None,
        "status_updated_at": None,
        "stage_summaries": {},
        "progress": {},
        "retry_summary": {},
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
        "recent_artifact_refs": [],
        "recent_limit": recent_limit,
    }


def _bounded_recent_limit(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DOCUMENT_DIAGNOSTIC_RECENT_LIMIT
    return max(1, min(parsed, DOCUMENT_DIAGNOSTIC_RECENT_LIMIT))


def _failure_or_retry_actions(
    document_diagnostics: dict[str, Any],
    *,
    current_stage: str | None,
    last_success_stage: str | None,
    retry_summary: dict[str, Any],
    recent_failures: list[Any],
    failure_total: int,
) -> list[dict[str, Any]]:
    related_stages = _related_failure_stages(
        recent_failures,
        [current_stage, last_success_stage],
    )
    related_failure_codes = _related_failure_codes(recent_failures)
    has_pending_retry = bool(retry_summary.get("has_pending_failure")) or (
        _safe_int(retry_summary.get("pending_stage_count")) > 0
    )
    gave_up_count = _safe_int(retry_summary.get("gave_up_count"))
    next_retry_at = _json_safe(retry_summary.get("next_retry_at"))
    actions = [
        _recovery_action(
            "inspect_recent_failures",
            "Inspect recent failure metadata and artifact references",
            "inspection",
            "high",
            [
                f"Diagnostics report {failure_total} failure records",
                "Recent failures are metadata-only and bounded",
            ],
            related_stages=related_stages,
            related_failure_codes=related_failure_codes,
            safe_next_endpoint=(
                f"/admin/diagnostics/document/{document_diagnostics.get('doc_id')}"
            ),
        )
    ]
    if has_pending_retry or next_retry_at:
        reasons = ["Retry summary indicates pending or retrying failures"]
        if next_retry_at:
            reasons.append(f"Next retry is scheduled at {next_retry_at}")
        actions.append(
            _recovery_action(
                "wait_for_scheduled_retry",
                "Wait for the scheduled retry before manual replay",
                "retry",
                "medium",
                reasons,
                related_stages=related_stages,
            )
        )
    actions.extend(
        [
            _recovery_action(
                "pause_drain_worker_before_manual_intervention",
                "Pause or drain workers before manual intervention",
                "control",
                "medium",
                [
                    "Pause/drain stops new claim or scan cycles at safe boundaries",
                    "Manual recovery should not race active worker processing",
                ],
                related_stages=related_stages,
            ),
            _recovery_action(
                "run_data_plane_consistency_verifier",
                "Run read-only consistency verification",
                "verification",
                "medium",
                [
                    "Verifier can compare Mongo state with document, chunk, and image rows",
                ],
                related_stages=related_stages,
            ),
            _recovery_action(
                "prepare_supervised_replay",
                "Prepare a supervised replay plan without enqueuing it",
                "replay_preparation",
                "medium",
                [
                    "Replay is a future supervised mutation and is not executed here",
                    "Use diagnostic artifact references to prepare the request context",
                ],
                related_stages=related_stages,
                related_failure_codes=related_failure_codes,
            ),
        ]
    )
    if gave_up_count > 0 or (failure_total > 0 and not has_pending_retry):
        actions.append(
            _recovery_action(
                "escalate_manual_review",
                "Escalate unresolved failures for manual review",
                "escalation",
                "high" if gave_up_count > 0 else "medium",
                [
                    f"Retry summary reports {gave_up_count} gave-up failures",
                    "Manual review is required before skip, give-up, or replay actions",
                ],
                related_stages=related_stages,
                related_failure_codes=related_failure_codes,
            )
        )
    return actions


def _recovery_action(
    action_id: str,
    title: str,
    category: str,
    confidence: str,
    reasons: list[str],
    *,
    related_stages: list[str | None] | None = None,
    related_failure_codes: list[str | None] | None = None,
    safe_next_endpoint: str | None = None,
) -> dict[str, Any]:
    action = {
        "action_id": action_id,
        "title": title,
        "category": category,
        "mutating": False,
        "confidence": confidence,
        "reasons": [_safe_reason(reason) for reason in reasons[:5]],
    }
    stages = _clean_token_list(related_stages or [])
    if stages:
        action["related_stages"] = stages
    failure_codes = _clean_token_list(related_failure_codes or [])
    if failure_codes:
        action["related_failure_codes"] = failure_codes
    if safe_next_endpoint:
        action["safe_next_endpoint"] = str(safe_next_endpoint)
    return action


def _bounded_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[: max(0, min(int(limit), 20))]


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_token(value: Any, fallback: str | None = None) -> str | None:
    if value in (None, ""):
        return fallback
    token = str(_json_safe(value))[:80]
    upper_token = token.upper()
    if any(part in upper_token for part in SENSITIVE_KEY_PARTS):
        return fallback
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-. ")
    if any(char not in allowed for char in token):
        return fallback
    return token


def _safe_reason(value: Any) -> str:
    reason = str(_json_safe(value))[:160]
    upper_reason = reason.upper()
    if any(part in upper_reason for part in SENSITIVE_KEY_PARTS):
        return "redacted sensitive diagnostic detail"
    return reason


def _clean_token_list(values: list[str | None]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        token = _safe_token(value)
        if token and token not in cleaned:
            cleaned.append(token)
        if len(cleaned) >= 5:
            break
    return cleaned


def _related_failure_stages(
    recent_failures: list[Any],
    fallback_stages: list[str | None],
) -> list[str]:
    stages = [
        _safe_token(failure.get("failure_stage"))
        for failure in recent_failures
        if isinstance(failure, dict)
    ]
    stages.extend(fallback_stages)
    return _clean_token_list(stages)


def _related_failure_codes(recent_failures: list[Any]) -> list[str]:
    return _clean_token_list(
        [
            _safe_token(failure.get("error_code"))
            for failure in recent_failures
            if isinstance(failure, dict)
        ]
    )


def _summarize_stage_statuses(stage_statuses: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(stage_statuses, dict):
        return {}
    summaries: dict[str, Any] = {}
    for stage, details in sorted(stage_statuses.items()):
        if not isinstance(details, dict):
            continue
        summaries[str(stage)] = {
            key: _json_safe(details.get(key))
            for key in STAGE_SUMMARY_FIELDS
            if key in details
        }
    return summaries


def _filter_dict_fields(value: dict[str, Any], allowed_fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: _json_safe(value.get(key))
        for key in allowed_fields
        if key in value
    }


def _summarize_failures(
    failures: list[dict[str, Any]],
    recent_limit: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for failure in list(failures)[:recent_limit]:
        if not isinstance(failure, dict):
            continue
        summary = {
            key: _json_safe(failure.get(key))
            for key in FAILURE_SUMMARY_FIELDS
            if key in failure
        }
        artifact_refs = failure.get("artifact_refs") or []
        if isinstance(artifact_refs, list):
            summary["artifact_ref_count"] = len(artifact_refs)
            summary["artifact_refs"] = [_json_safe(ref) for ref in artifact_refs[:20]]
        source_event_meta = failure.get("source_event_meta") or {}
        if isinstance(source_event_meta, dict):
            summary["source_event_meta"] = {
                key: _json_safe(source_event_meta.get(key))
                for key in (
                    "topic",
                    "partition",
                    "offset",
                    "event_fingerprint",
                )
                if key in source_event_meta
            }
        summaries.append(summary)
    return summaries


def _summarize_artifacts(
    artifacts: list[dict[str, Any]],
    recent_limit: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for artifact in list(artifacts)[:recent_limit]:
        if not isinstance(artifact, dict):
            continue
        summary = {
            "artifact_ref": _json_safe(artifact.get("_id")),
        }
        for key in ARTIFACT_REF_FIELDS:
            if key in artifact:
                summary[key] = _json_safe(artifact.get(key))
        summaries.append(summary)
    return summaries


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
