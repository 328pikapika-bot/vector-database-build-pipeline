"""Read-only Mongo/Milvus data-plane consistency verifier."""

from __future__ import annotations

import inspect
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Optional

from app.config import settings
from app.services.milvus_chunk_service import DEFAULT_CHUNK_COLLECTION
from app.services.milvus_image_service import DEFAULT_IMAGE_COLLECTION


DEFAULT_MILVUS_QUERY_LIMIT = 16_384


async def verify_data_plane_consistency(
    doc_id: str,
    *,
    sys_id: Optional[str] = None,
    expect_document: Optional[bool] = None,
    expect_chunks: Optional[bool] = None,
    expect_images: Optional[bool] = None,
    mongo_manager: Optional[Any] = None,
    document_client: Optional[Any] = None,
    chunk_client: Optional[Any] = None,
    image_client: Optional[Any] = None,
    document_collection: str = settings.resolved_document_collection_name,
    chunk_collection: str = settings.resolved_fulltext_chunk_collection_name,
    image_collection: str = settings.IMAGE_COLLECTION_NAME,
    allow_live_clients: bool = False,
    milvus_query_limit: int = DEFAULT_MILVUS_QUERY_LIMIT,
) -> dict[str, Any]:
    """Verify one document across Mongo and Milvus without writes by default."""
    normalized_doc_id = str(doc_id or "").strip()
    if not normalized_doc_id:
        raise ValueError("doc_id is required for data-plane consistency verification")

    document_lookup_id = str(sys_id or normalized_doc_id).strip()
    owned_resources: list[Any] = []
    try:
        if allow_live_clients:
            mongo_manager = await _ensure_live_mongo_manager(mongo_manager, owned_resources)
            document_client, chunk_client, image_client = _ensure_live_milvus_clients(
                document_client,
                chunk_client,
                image_client,
                owned_resources,
            )

        mongo_summary = await _summarize_mongo(mongo_manager, normalized_doc_id)
        document_summary = _summarize_milvus_count(
            document_client,
            collection_name=document_collection,
            field_name="id",
            value=document_lookup_id,
            identity_field="id",
            query_limit=milvus_query_limit,
        )
        chunk_summary = _summarize_milvus_count(
            chunk_client,
            collection_name=chunk_collection,
            field_name="doc_id",
            value=normalized_doc_id,
            identity_field="id",
            query_limit=milvus_query_limit,
        )
        image_summary = _summarize_milvus_count(
            image_client,
            collection_name=image_collection,
            field_name="doc_id",
            value=normalized_doc_id,
            identity_field="id",
            query_limit=milvus_query_limit,
        )

        mismatch_flags = _build_mismatch_flags(
            mongo_summary=mongo_summary,
            document_summary=document_summary,
            chunk_summary=chunk_summary,
            image_summary=image_summary,
            document_collection=document_collection,
            chunk_collection=chunk_collection,
            image_collection=image_collection,
            expect_document=expect_document,
            expect_chunks=expect_chunks,
            expect_images=expect_images,
        )

        return {
            "success": True,
            "read_only": True,
            "doc_id": normalized_doc_id,
            "sys_id": document_lookup_id,
            "expectations": {
                "document": expect_document,
                "chunks": expect_chunks,
                "images": expect_images,
            },
            "collections": {
                "document": document_collection,
                "chunk": chunk_collection,
                "image": image_collection,
            },
            "mongo": mongo_summary,
            "milvus": {
                "document": document_summary,
                "chunk": chunk_summary,
                "image": image_summary,
            },
            "mismatch_flags": mismatch_flags,
            "has_mismatch": any(bool(value) for value in mismatch_flags.values()),
        }
    finally:
        await _close_owned_resources(owned_resources)


async def _ensure_live_mongo_manager(mongo_manager: Optional[Any], owned_resources: list[Any]) -> Any:
    if mongo_manager is not None:
        return mongo_manager
    from app.services.mongo_state_manager import MongoStateManager

    manager = MongoStateManager()
    await manager.connect()
    owned_resources.append(manager)
    return manager


def _ensure_live_milvus_clients(
    document_client: Optional[Any],
    chunk_client: Optional[Any],
    image_client: Optional[Any],
    owned_resources: list[Any],
) -> tuple[Any, Any, Any]:
    if document_client is not None and chunk_client is not None and image_client is not None:
        return document_client, chunk_client, image_client

    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        user=settings.MILVUS_USER or "",
        password=settings.MILVUS_PASSWORD or "",
    )
    owned_resources.append(client)
    return (
        document_client if document_client is not None else client,
        chunk_client if chunk_client is not None else client,
        image_client if image_client is not None else client,
    )


async def _summarize_mongo(mongo_manager: Optional[Any], doc_id: str) -> dict[str, Any]:
    if mongo_manager is None:
        return {
            "checked": False,
            "reason": "mongo_manager_not_provided",
            "status_found": False,
            "status": None,
            "failures": {"count": None, "status_counts": {}, "stage_counts": {}},
            "artifacts": {"count": None, "type_counts": {}, "stage_counts": {}, "recent": []},
        }

    try:
        status_doc = await _get_mongo_status(mongo_manager, doc_id)
        failures = _summarize_mongo_collection(
            getattr(mongo_manager, "failures_collection", None),
            {"doc_id": doc_id},
            group_fields=("status", "failure_stage"),
            recent_fields=("failure_key", "status", "failure_stage", "scope_type", "scope_id", "error_code"),
        )
        artifacts = _summarize_mongo_collection(
            getattr(mongo_manager, "artifacts_collection", None),
            {"doc_id": doc_id},
            group_fields=("artifact_type", "stage"),
            recent_fields=("_id", "artifact_type", "stage", "scope_type", "scope_id"),
        )
        return {
            "checked": True,
            "status_found": status_doc is not None,
            "status": _summarize_status(status_doc),
            "failures": {
                "count": failures["count"],
                "status_counts": failures["counts"].get("status", {}),
                "stage_counts": failures["counts"].get("failure_stage", {}),
                "recent": failures["recent"],
            },
            "artifacts": {
                "count": artifacts["count"],
                "type_counts": artifacts["counts"].get("artifact_type", {}),
                "stage_counts": artifacts["counts"].get("stage", {}),
                "recent": artifacts["recent"],
            },
        }
    except Exception as error:
        return {
            "checked": False,
            "reason": "mongo_read_error",
            "error_code": "MONGO_VERIFY_READ_ERROR",
            "error_message": str(error),
            "status_found": False,
            "status": None,
            "failures": {"count": None, "status_counts": {}, "stage_counts": {}, "recent": []},
            "artifacts": {"count": None, "type_counts": {}, "stage_counts": {}, "recent": []},
        }


async def _get_mongo_status(mongo_manager: Any, doc_id: str) -> Optional[dict[str, Any]]:
    if hasattr(mongo_manager, "get_status"):
        result = mongo_manager.get_status(doc_id)
        return await _maybe_await(result)

    collection = getattr(mongo_manager, "status_collection", None)
    if collection is None:
        return None
    return collection.find_one({"_id": doc_id})


def _summarize_status(status_doc: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if status_doc is None:
        return None
    return {
        "_id": _json_safe(status_doc.get("_id")),
        "doc_id": status_doc.get("doc_id") or status_doc.get("_id"),
        "overall_status": status_doc.get("overall_status"),
        "current_stage": status_doc.get("current_stage"),
        "last_success_stage": status_doc.get("last_success_stage"),
        "latest_record_update_time": status_doc.get("latest_record_update_time"),
        "latest_event_type": status_doc.get("latest_event_type"),
        "stage_statuses": _json_safe(status_doc.get("stage_statuses") or {}),
        "progress": _json_safe(status_doc.get("progress") or {}),
        "retry_summary": _json_safe(status_doc.get("retry_summary") or {}),
        "updated_at": _json_safe(status_doc.get("updated_at")),
    }


def _summarize_mongo_collection(
    collection: Optional[Any],
    query_filter: dict[str, Any],
    *,
    group_fields: tuple[str, ...],
    recent_fields: tuple[str, ...],
    recent_limit: int = 20,
) -> dict[str, Any]:
    if collection is None:
        return {"count": None, "counts": {field: {} for field in group_fields}, "recent": []}

    count = collection.count_documents(query_filter) if hasattr(collection, "count_documents") else None
    docs = _find_documents(collection, query_filter, limit=recent_limit)
    grouped: dict[str, dict[str, int]] = {}
    for field in group_fields:
        grouped[field] = dict(Counter(str(item.get(field, "")) for item in docs if item.get(field) is not None))
    recent = [
        {field: _json_safe(item.get(field)) for field in recent_fields if field in item}
        for item in docs
    ]
    return {"count": int(count) if count is not None else len(docs), "counts": grouped, "recent": recent}


def _find_documents(collection: Any, query_filter: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    if not hasattr(collection, "find"):
        return []
    try:
        cursor = collection.find(query_filter)
    except TypeError:
        cursor = collection.find(query_filter, None)
    if hasattr(cursor, "limit"):
        cursor = cursor.limit(limit)
    docs = list(cursor)
    return docs[:limit]


def _summarize_milvus_count(
    client_holder: Optional[Any],
    *,
    collection_name: str,
    field_name: str,
    value: str,
    identity_field: str,
    query_limit: int,
) -> dict[str, Any]:
    filter_expr = f'{field_name} == "{_escape_milvus_string(value)}"'
    summary: dict[str, Any] = {
        "checked": False,
        "collection_name": collection_name,
        "filter": filter_expr,
        "count": None,
        "collection_exists": None,
        "query_mode": None,
    }
    if client_holder is None:
        summary["reason"] = "milvus_client_not_provided"
        return summary

    client = getattr(client_holder, "client", client_holder)
    try:
        if hasattr(client, "has_collection"):
            exists = bool(client.has_collection(collection_name))
            summary["collection_exists"] = exists
            if not exists:
                summary.update({"checked": True, "count": 0, "query_mode": "collection_missing"})
                return summary

        try:
            rows = client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=["count(*)"],
            )
            count = _extract_milvus_count(rows)
            if count is not None:
                summary.update({"checked": True, "count": count, "query_mode": "count_query"})
                return summary
        except Exception:
            pass

        rows = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=[identity_field],
            limit=query_limit,
        )
        row_count = len(list(rows))
        summary.update(
            {
                "checked": True,
                "count": row_count,
                "query_mode": "limited_query",
                "truncated": row_count >= query_limit,
            }
        )
        return summary
    except Exception as error:
        summary.update(
            {
                "checked": False,
                "reason": "milvus_read_error",
                "error_code": "MILVUS_VERIFY_READ_ERROR",
                "error_message": str(error),
            }
        )
        return summary


def _extract_milvus_count(rows: Any) -> Optional[int]:
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        for key in ("count(*)", "COUNT(*)", "count", "row_count"):
            if key in row:
                return int(row[key])
    return None


def _build_mismatch_flags(
    *,
    mongo_summary: dict[str, Any],
    document_summary: dict[str, Any],
    chunk_summary: dict[str, Any],
    image_summary: dict[str, Any],
    document_collection: str,
    chunk_collection: str,
    image_collection: str,
    expect_document: Optional[bool],
    expect_chunks: Optional[bool],
    expect_images: Optional[bool],
) -> dict[str, bool]:
    status = mongo_summary.get("status") or {}
    overall_status = str(status.get("overall_status") or "").upper()
    active_status = bool(status) and overall_status not in {"DELETED", "SKIPPED_STALE"}
    document_count = _checked_count(document_summary)
    chunk_count = _checked_count(chunk_summary)
    image_count = _checked_count(image_summary)
    deleted_has_document_rows = overall_status == "DELETED" and (document_count or 0) > 0
    deleted_has_chunk_rows = overall_status == "DELETED" and (chunk_count or 0) > 0
    deleted_has_image_rows = overall_status == "DELETED" and (image_count or 0) > 0

    return {
        "mongo_status_missing": bool(mongo_summary.get("checked") and not mongo_summary.get("status_found")),
        "chunk_collection_points_to_document_collection": chunk_collection == document_collection,
        "image_collection_points_to_document_collection": image_collection == document_collection,
        "image_collection_points_to_chunk_collection": image_collection == chunk_collection,
        "active_status_missing_expected_document_rows": _missing_expected_rows(
            active_status=active_status,
            expected=expect_document,
            count=document_count,
        ),
        "active_status_missing_expected_chunk_rows": _missing_expected_rows(
            active_status=active_status,
            expected=expect_chunks,
            count=chunk_count,
        ),
        "active_status_missing_expected_image_rows": _missing_expected_rows(
            active_status=active_status,
            expected=expect_images,
            count=image_count,
        ),
        "expected_no_document_rows_but_found": _unexpected_rows(expected=expect_document, count=document_count),
        "expected_no_chunk_rows_but_found": _unexpected_rows(expected=expect_chunks, count=chunk_count),
        "expected_no_image_rows_but_found": _unexpected_rows(expected=expect_images, count=image_count),
        "deleted_status_has_document_rows": deleted_has_document_rows,
        "deleted_status_has_chunk_rows": deleted_has_chunk_rows,
        "deleted_status_has_image_rows": deleted_has_image_rows,
        "deleted_status_retaining_document_rows": deleted_has_document_rows,
        "deleted_status_retaining_chunk_rows": deleted_has_chunk_rows,
        "deleted_status_retaining_image_rows": deleted_has_image_rows,
        "partial_failure_has_no_failure_records": (
            overall_status == "PARTIAL_FAILURE"
            and mongo_summary.get("checked")
            and int((mongo_summary.get("failures") or {}).get("count") or 0) == 0
        ),
    }


def _checked_count(summary: dict[str, Any]) -> Optional[int]:
    if not summary.get("checked") or summary.get("count") is None:
        return None
    return int(summary["count"])


def _missing_expected_rows(*, active_status: bool, expected: Optional[bool], count: Optional[int]) -> bool:
    return active_status and expected is True and count is not None and count < 1


def _unexpected_rows(*, expected: Optional[bool], count: Optional[int]) -> bool:
    return expected is False and count is not None and count > 0


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _close_owned_resources(resources: Iterable[Any]) -> None:
    for resource in reversed(list(resources)):
        close = getattr(resource, "close", None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _escape_milvus_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
