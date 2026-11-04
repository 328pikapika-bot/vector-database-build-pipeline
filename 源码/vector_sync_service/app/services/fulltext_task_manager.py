"""Full-text task discovery helpers for Phase 2 control-plane preparation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from app.contracts.docx_contract import extract_docx_doc_meta
from app.models.sync_runtime import normalize_record_update_time, sha256_hex, utcnow
from app.utils.parser import preprocess_kafka_item


class FullTextTaskStatus(str, Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"
    HELD_HISTORY = "HELD_HISTORY"


@dataclass(slots=True)
class FullTextTask:
    task_id: str
    doc_id: str
    record_update_time: str
    relative_url: str
    title: Optional[list[str]] = None
    keyword: Optional[list[str]] = None
    accessType: Optional[str] = None
    classificationCode: Optional[str] = None
    year: Optional[str] = None
    hasAuthorname: Optional[list[str]] = None
    hasCreateOrganizationname: Optional[list[str]] = None
    language: Optional[list[str]] = None
    status: FullTextTaskStatus = FullTextTaskStatus.PENDING
    source_item_index: int = 0
    created_at: Optional[datetime] = None

    def to_mongo_document(self) -> dict[str, Any]:
        created_at = self.created_at or utcnow()
        return {
            "_id": self.task_id,
            "task_id": self.task_id,
            "doc_id": self.doc_id,
            "record_update_time": self.record_update_time,
            "relative_url": self.relative_url,
            "title": self.title,
            "keyword": self.keyword,
            "accessType": self.accessType,
            "classificationCode": self.classificationCode,
            "year": self.year,
            "hasAuthorname": self.hasAuthorname,
            "hasCreateOrganizationname": self.hasCreateOrganizationname,
            "language": self.language,
            "stage_group": "fulltext",
            "status": self.status.value,
            "priority": 0,
            "lease_owner": None,
            "lease_expires_at": None,
            "next_retry_at": None,
            "retry_count": 0,
            "source_item_index": self.source_item_index,
            "created_at": created_at,
            "updated_at": created_at,
        }


def extract_fulltext_tasks(payload: dict[str, Any]) -> list[FullTextTask]:
    """Extract pending .mmd download tasks from a Kafka payload."""
    if not isinstance(payload, dict):
        return []

    data_items = payload.get("data")
    if not isinstance(data_items, list):
        return []

    tasks: list[FullTextTask] = []
    seen_task_ids: set[str] = set()
    for item_index, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue

        doc_id = _extract_doc_id(item)
        record_update_time = _extract_record_update_time(item)
        doc_meta = _extract_docx_metadata(item)
        for relative_url in _extract_fulltext_relative_urls(item):
            task_id = build_fulltext_task_id(
                doc_id=doc_id,
                record_update_time=record_update_time,
                relative_url=relative_url,
            )
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            tasks.append(
                FullTextTask(
                    task_id=task_id,
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    relative_url=relative_url,
                    title=doc_meta.get("title"),
                    keyword=doc_meta.get("keyword"),
                    accessType=doc_meta.get("accessType"),
                    classificationCode=doc_meta.get("classificationCode"),
                    year=doc_meta.get("year"),
                    hasAuthorname=doc_meta.get("hasAuthorname"),
                    hasCreateOrganizationname=doc_meta.get("hasCreateOrganizationname"),
                    language=doc_meta.get("language"),
                    source_item_index=item_index,
                )
            )

    return tasks


def build_fulltext_task_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Mongo-oriented task documents without writing to MongoDB."""
    return [task.to_mongo_document() for task in extract_fulltext_tasks(payload)]


def build_fulltext_task_id(
    *,
    doc_id: str,
    record_update_time: str,
    relative_url: str,
) -> str:
    return sha256_hex(f"{doc_id}|{record_update_time}|{relative_url}")


def _extract_docx_metadata(item: dict[str, Any]) -> dict[str, Any]:
    try:
        flattened = preprocess_kafka_item(item)
    except Exception:
        flattened = item
    return extract_docx_doc_meta(flattened)


def build_skipped_fulltext_discover_document(
    *,
    doc_id: str,
    record_update_time: str,
    reason: str = "SKIPPED_NOT_PRESENT",
) -> dict[str, Any]:
    """Draft skipped status document for future Mongo stage-task persistence."""
    now = utcnow()
    task_id = build_fulltext_task_id(
        doc_id=doc_id,
        record_update_time=record_update_time,
        relative_url="",
    )
    return {
        "_id": task_id,
        "task_id": task_id,
        "doc_id": doc_id,
        "record_update_time": record_update_time,
        "relative_url": "",
        "stage_group": "fulltext",
        "status": FullTextTaskStatus.SKIPPED.value,
        "skip_reason": reason,
        "created_at": now,
        "updated_at": now,
    }


def _extract_doc_id(item: dict[str, Any]) -> str:
    return (
        _first_text(item.get("sysId"))
        or _first_text(item.get("id"))
        or _first_text(item.get("docId"))
        or _first_nested_field_text(item.get("dp"), "sysId")
        or _first_nested_field_text(item.get("dp"), "id")
        or ""
    )


def _extract_record_update_time(item: dict[str, Any]) -> str:
    value = (
        _first_text(item.get("sysDataUpdateTime"))
        or _first_text(item.get("recordUpdateTime"))
        or _first_text(item.get("record_update_time"))
        or _first_nested_field_text(item.get("dp"), "sysDataUpdateTime")
        or _first_nested_field_text(item.get("dp"), "recordUpdateTime")
        or ""
    )
    return normalize_record_update_time(value)


def _extract_fulltext_relative_urls(item: dict[str, Any]) -> list[str]:
    op = item.get("op")
    if not isinstance(op, dict):
        return []

    has_fulltext = op.get("hasFullTextFile")
    if not isinstance(has_fulltext, dict):
        return []

    candidates: list[str] = []
    values = has_fulltext.get("value")
    if not isinstance(values, list):
        values = [values] if values is not None else []

    for relation in values:
        candidates.extend(_extract_urls_from_relation(relation))

    filtered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned or not _looks_like_mmd(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        filtered.append(cleaned)
    return filtered


def _extract_urls_from_relation(relation: Any) -> list[str]:
    if isinstance(relation, str):
        return [relation]
    if not isinstance(relation, dict):
        return []

    candidates: list[str] = []
    for key in ("url", "fileUrl", "relativeUrl", "path", "filePath"):
        candidates.extend(_field_values(relation.get(key)))

    dp = relation.get("dp")
    if isinstance(dp, dict):
        for key in ("url", "fileUrl", "relativeUrl", "path", "filePath"):
            candidates.extend(_field_values(dp.get(key)))

    return candidates


def _field_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _field_values(item)]
    if isinstance(value, dict):
        if "value" in value:
            return _field_values(value.get("value"))
        return [_first_text(value)] if _first_text(value) else []
    return [str(value)]


def _first_nested_field_text(container: Any, key: str) -> str:
    if not isinstance(container, dict):
        return ""
    return _first_text(container.get(key))


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return _first_text(value.get("value"))
        for key in ("sysId", "id", "docId", "url", "fileUrl", "relativeUrl"):
            text = _first_text(value.get(key))
            if text:
                return text
    return ""


def _looks_like_mmd(relative_url: str) -> bool:
    path = relative_url.split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(".mmd")
