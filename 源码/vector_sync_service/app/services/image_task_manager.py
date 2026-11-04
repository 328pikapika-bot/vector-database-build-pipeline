"""Image task discovery helpers for Phase 3 control-plane preparation."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Optional

from app.config import settings
from app.contracts.docx_contract import extract_docx_doc_meta
from app.models.sync_runtime import normalize_record_update_time, sha256_hex, utcnow
from app.services.image_context_extractor import (
    compose_context_after,
    extract_mmd_image_contexts,
    find_mmd_context_for_image,
)
from app.services.image_context_text import canonicalize_context_text, first_canonical_text, join_context_parts
from app.utils.parser import preprocess_kafka_item


class ImageTaskStatus(str, Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"
    HELD_HISTORY = "HELD_HISTORY"


class ImageSourceIdRequiredError(ValueError):
    """Raised when an upstream image relation has no original image id."""


@dataclass(slots=True)
class ImageTask:
    task_id: str
    doc_id: str
    record_update_time: str
    relative_url: str
    image_id: str
    sequence: str
    mmd_relative_url: Optional[str] = None
    doc_title: Optional[list[str]] = None
    keyword: Optional[list[str]] = None
    accessType: Optional[str] = None
    classificationCode: Optional[str] = None
    year: Optional[str] = None
    hasAuthorname: Optional[list[str]] = None
    hasCreateOrganizationname: Optional[list[str]] = None
    language: Optional[list[str]] = None
    sysDataStatus: Optional[str] = None
    hasSourcesysSourceId: Optional[list[str]] = None
    hasSourceidSet: Optional[list[str]] = None
    status: ImageTaskStatus = ImageTaskStatus.PENDING
    title: str = ""
    name: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    file_type: str = ""
    size: Optional[int] = None
    raw_img_meta: dict[str, Any] = dataclass_field(default_factory=dict)
    img_meta: dict[str, Any] = dataclass_field(default_factory=dict)
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    source_item_index: int = 0
    image_index: int = 0
    created_at: Optional[datetime] = None

    def to_mongo_document(self) -> dict[str, Any]:
        created_at = self.created_at or utcnow()
        return {
            "_id": self.task_id,
            "task_id": self.task_id,
            "doc_id": self.doc_id,
            "record_update_time": self.record_update_time,
            "relative_url": self.relative_url,
            "mmd_relative_url": self.mmd_relative_url,
            "image_id": self.image_id,
            "sequence": self.sequence,
            "doc_title": self.doc_title,
            "keyword": self.keyword,
            "accessType": self.accessType,
            "classificationCode": self.classificationCode,
            "year": self.year,
            "hasAuthorname": self.hasAuthorname,
            "hasCreateOrganizationname": self.hasCreateOrganizationname,
            "language": self.language,
            "sysDataStatus": self.sysDataStatus,
            "hasSourcesysSourceId": self.hasSourcesysSourceId,
            "hasSourceidSet": self.hasSourceidSet,
            "title": self.title,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "file_type": self.file_type,
            "size": self.size,
            "raw_img_meta": self.raw_img_meta,
            "img_meta": self.img_meta,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "stage_group": "image",
            "status": self.status.value,
            "priority": 0,
            "lease_owner": None,
            "lease_expires_at": None,
            "next_retry_at": None,
            "retry_count": 0,
            "attempt_count": 0,
            "artifact_refs": [],
            "source_item_index": self.source_item_index,
            "image_index": self.image_index,
            "created_at": created_at,
            "updated_at": created_at,
        }


def extract_image_tasks(payload: dict[str, Any]) -> list[ImageTask]:
    """Extract pending image download tasks from a Kafka payload."""
    if not isinstance(payload, dict):
        return []

    data_items = payload.get("data")
    if not isinstance(data_items, list):
        return []

    tasks: list[ImageTask] = []
    seen_task_ids: set[str] = set()
    for item_index, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue

        doc_id = _extract_doc_id(item)
        record_update_time = _extract_record_update_time(item)
        doc_meta = _extract_docx_metadata(item)
        mmd_relative_url = _extract_mmd_relative_url(item)
        for image_index, metadata in enumerate(_extract_image_metadata(item)):
            relative_url = metadata["relative_url"]
            sequence = metadata["sequence"] or str(image_index + 1)
            image_id = str(metadata["image_id"] or "").strip()
            if not image_id:
                raise ImageSourceIdRequiredError(
                    "IMAGE_SOURCE_ID_REQUIRED: raw image relation id is required"
                )
            task_id = build_image_task_id(
                doc_id=doc_id,
                record_update_time=record_update_time,
                relative_url=relative_url,
                source_item_index=item_index,
                sequence=sequence,
            )
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            tasks.append(
                ImageTask(
                    task_id=task_id,
                    doc_id=doc_id,
                    record_update_time=record_update_time,
                    relative_url=relative_url,
                    mmd_relative_url=mmd_relative_url,
                    image_id=image_id,
                    sequence=sequence,
                    doc_title=doc_meta.get("title"),
                    keyword=doc_meta.get("keyword"),
                    accessType=doc_meta.get("accessType"),
                    classificationCode=doc_meta.get("classificationCode"),
                    year=doc_meta.get("year"),
                    hasAuthorname=doc_meta.get("hasAuthorname"),
                    hasCreateOrganizationname=doc_meta.get("hasCreateOrganizationname"),
                    language=doc_meta.get("language"),
                    sysDataStatus=doc_meta.get("sysDataStatus"),
                    hasSourcesysSourceId=doc_meta.get("hasSourcesysSourceId"),
                    hasSourceidSet=doc_meta.get("hasSourceidSet"),
                    title=metadata["title"],
                    name=metadata["name"],
                    width=metadata["width"],
                    height=metadata["height"],
                    file_type=metadata["file_type"],
                    size=metadata["size"],
                    raw_img_meta=metadata["raw_img_meta"],
                    img_meta=metadata["img_meta"],
                    context_before=metadata["context_before"],
                    context_after=metadata["context_after"],
                    source_item_index=item_index,
                    image_index=image_index,
                )
            )

    return tasks


def build_image_task_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Mongo-oriented image task documents without writing to MongoDB."""
    return [task.to_mongo_document() for task in extract_image_tasks(payload)]


def build_image_task_id(
    *,
    doc_id: str,
    record_update_time: str,
    relative_url: str,
    source_item_index: int,
    sequence: Optional[str] = None,
) -> str:
    sequence_or_url = (sequence or relative_url).strip()
    return sha256_hex(
        f"{doc_id}|{record_update_time}|{source_item_index}|{sequence_or_url}|{relative_url}"
    )


def build_skipped_image_discover_document(
    *,
    doc_id: str,
    record_update_time: str,
    reason: str = "SKIPPED_NOT_PRESENT",
) -> dict[str, Any]:
    """Draft skipped status document for future image stage persistence."""
    now = utcnow()
    task_id = build_image_task_id(
        doc_id=doc_id,
        record_update_time=record_update_time,
        relative_url="",
        source_item_index=0,
        sequence="skipped",
    )
    return {
        "_id": task_id,
        "task_id": task_id,
        "doc_id": doc_id,
        "record_update_time": record_update_time,
        "relative_url": "",
        "image_id": "",
        "stage_group": "image",
        "status": ImageTaskStatus.SKIPPED.value,
        "skip_reason": reason,
        "created_at": now,
        "updated_at": now,
    }


def _extract_docx_metadata(item: dict[str, Any]) -> dict[str, Any]:
    try:
        flattened = preprocess_kafka_item(item)
    except Exception:
        flattened = item
    return extract_docx_doc_meta(flattened)


def _extract_doc_id(item: dict[str, Any]) -> str:
    return (
        _first_text(item.get("sysId"))
        or _first_text(item.get("id"))
        or _first_text(item.get("docId"))
        or _first_nested_field_text(item.get("dp"), "sysId")
        or _first_nested_field_text(item.get("dp"), "id")
        or _first_nested_field_text(item.get("dp"), "docId")
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


def _extract_image_metadata(item: dict[str, Any]) -> list[dict[str, Any]]:
    op = item.get("op")
    if not isinstance(op, dict):
        return []

    has_picture = op.get("hasPictureFile")
    if has_picture is None:
        return []

    values = _relation_values(has_picture)
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, relation in enumerate(values):
        for relative_url in _extract_urls_from_relation(relation):
            cleaned_url = relative_url.strip()
            if not cleaned_url or cleaned_url in seen_urls:
                continue
            seen_urls.add(cleaned_url)
            metadata = _metadata_from_relation(relation)
            metadata["relative_url"] = cleaned_url
            metadata["sequence"] = metadata["sequence"] or str(index + 1)
            if not metadata["name"]:
                metadata["name"] = _filename_from_url(cleaned_url)
            metadata["raw_img_meta"] = _raw_img_meta_for_record(
                relation,
                metadata=metadata,
                relative_url=cleaned_url,
            )
            records.append(metadata)
    return _apply_mmd_context_metadata(records, item)


def _relation_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict) and "value" in value:
        return _relation_values(value.get("value"))
    if isinstance(value, list):
        return value
    return [value]


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


def _metadata_from_relation(relation: Any) -> dict[str, Any]:
    if not isinstance(relation, dict):
        metadata = _blank_metadata()
        if relation is not None:
            metadata["raw_img_meta"] = {"value": relation}
        return metadata

    metadata = _blank_metadata()
    metadata["image_id"] = _first_field_text(
        relation,
        ("id", "imageId", "img_id", "imgId", "image_id"),
    )
    metadata["sequence"] = _first_field_text(
        relation,
        ("sequence", "seq", "order", "index", "imageIndex"),
    )
    metadata["title"] = canonicalize_context_text(_first_field_text(
        relation,
        ("title",),
    )) or ""
    metadata["name"] = _first_field_text(
        relation,
        ("name", "fileName", "filename", "imageName"),
    )
    metadata["width"] = _first_field_int(relation, ("width", "imageWidth"))
    metadata["height"] = _first_field_int(relation, ("height", "imageHeight"))
    metadata["file_type"] = _first_field_text(
        relation,
        ("fileType", "file_type", "type", "mimeType"),
    )
    metadata["size"] = _first_field_int(
        relation,
        ("size", "fileSize", "sizeBytes", "bytes"),
    )
    metadata["raw_img_meta"] = _raw_img_meta_for_record(relation, metadata=metadata)
    metadata["img_meta"] = _img_meta_from_relation(relation)
    metadata["context_before"] = canonicalize_context_text(_first_field_text(
        relation,
        (
            "context_before",
            "contextBefore",
            "beforeContext",
            "previousParagraph",
            "prevParagraph",
            "paragraphBefore",
        ),
    )) or None
    metadata["context_after"] = canonicalize_context_text(_first_field_text(
        relation,
        (
            "context_after",
            "contextAfter",
            "afterContext",
            "nextParagraphs",
            "paragraphAfter",
        ),
    )) or None
    return metadata


def _blank_metadata() -> dict[str, Any]:
    return {
        "relative_url": "",
        "image_id": "",
        "sequence": "",
        "title": "",
        "name": "",
        "width": None,
        "height": None,
        "file_type": "",
        "size": None,
        "raw_img_meta": {},
        "img_meta": {},
        "context_before": None,
        "context_after": None,
    }


def _apply_mmd_context_metadata(
    records: list[dict[str, Any]],
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    mmd_text = _extract_mmd_text(item)
    if not records or not mmd_text:
        return records

    contexts = extract_mmd_image_contexts(
        mmd_text,
        before_window=settings.IMAGE_CONTEXT_BEFORE_PARAGRAPHS,
        after_window=settings.IMAGE_CONTEXT_AFTER_PARAGRAPHS,
    )
    if not contexts:
        return records

    updated_records: list[dict[str, Any]] = []
    for metadata in records:
        context = find_mmd_context_for_image(
            contexts,
            image_name=str(metadata.get("name") or ""),
            relative_url=str(metadata.get("relative_url") or ""),
        )
        if context is None:
            updated_records.append(metadata)
            continue

        updated = dict(metadata)
        context_update = context.to_metadata_update()
        caption = _caption_from_metadata(updated)
        updated["context_before"] = first_canonical_text(
            context_update["context_before"],
            updated.get("context_before"),
        )
        updated["context_after"] = first_canonical_text(
            join_context_parts(
                caption,
                context_update["context_after"] or "",
            ),
            updated.get("context_after"),
            caption,
        )
        updated["img_meta"] = {
            **_dict_or_empty(updated.get("img_meta")),
            **context_update["img_meta"],
        }
        if caption:
            updated["img_meta"]["caption"] = caption
        updated_records.append(updated)
    return updated_records


def _extract_mmd_text(item: dict[str, Any]) -> str:
    sources = [item]
    op = item.get("op")
    if isinstance(op, dict):
        sources.append(op)
    dp = item.get("dp")
    if isinstance(dp, dict):
        sources.append(dp)

    keys = (
        "mmd_text",
        "mmdText",
        "fulltext_mmd",
        "fullTextMmd",
        "fulltextContent",
        "fullTextContent",
        "mmd",
    )
    for source in sources:
        for key in keys:
            text = _first_text(source.get(key))
            if text:
                return text
    return ""


def _extract_mmd_relative_url(item: dict[str, Any]) -> Optional[str]:
    op = item.get("op")
    if not isinstance(op, dict):
        return None

    has_fulltext = op.get("hasFullTextFile")
    for relation in _relation_values(has_fulltext):
        for relative_url in _extract_urls_from_relation(relation):
            cleaned = relative_url.strip()
            if cleaned and _looks_like_mmd(cleaned):
                return cleaned
    return None


def _img_meta_from_relation(relation: dict[str, Any]) -> dict[str, Any]:
    img_meta: dict[str, Any] = {}
    text_fields = {
        "page": ("page", "pageNo", "pageNum", "pageNumber"),
        "bbox": ("bbox", "box", "boundingBox", "bounding_box", "rect"),
        "x": ("x", "left"),
        "y": ("y", "top"),
    }
    int_fields = {
        "width": ("width", "imageWidth"),
        "height": ("height", "imageHeight"),
    }
    for output_key, keys in text_fields.items():
        value = _first_field_text(relation, keys)
        if value:
            img_meta[output_key] = value
    for output_key, keys in int_fields.items():
        value = _first_field_int(relation, keys)
        if value is not None:
            img_meta[output_key] = value
    caption = canonicalize_context_text(_first_field_text(
        relation,
        ("title",),
    )) or _default_caption()
    img_meta["caption"] = caption
    return img_meta


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _raw_img_meta_for_record(
    relation: Any,
    *,
    metadata: dict[str, Any],
    relative_url: str = "",
) -> dict[str, Any]:
    if not isinstance(relation, dict):
        raw: dict[str, Any] = {}
        if relative_url:
            raw["url"] = [relative_url]
        for key, value in (
            ("sequence", metadata.get("sequence")),
            ("size", metadata.get("size")),
            ("width", metadata.get("width")),
            ("name", metadata.get("name")),
            ("fileType", metadata.get("file_type")),
            ("height", metadata.get("height")),
        ):
            if value not in (None, "", [], {}):
                raw[key] = value
        return raw or ({"value": relation} if relation is not None else {})

    raw: dict[str, Any] = {}
    for output_key, keys in (
        ("id", ("id",)),
        ("cls", ("cls",)),
        ("sequence", ("sequence", "seq", "order", "index", "imageIndex")),
        ("name", ("name", "fileName", "filename", "imageName")),
        ("fileType", ("fileType", "file_type", "type", "mimeType")),
        ("sysDataType", ("sysDataType", "dataType")),
    ):
        value = _first_field_text(relation, keys)
        if value:
            raw[output_key] = value

    for output_key, keys in (
        ("size", ("size", "fileSize", "sizeBytes", "bytes")),
        ("width", ("width", "imageWidth")),
        ("height", ("height", "imageHeight")),
    ):
        value = _first_field_int(relation, keys)
        if value is not None:
            raw[output_key] = value

    title_values = _first_field_values(
        relation,
        ("title", "caption", "figureTitle", "imageTitle"),
    )
    if not title_values and metadata.get("title"):
        title_values = [str(metadata["title"])]
    if title_values:
        raw["title"] = title_values

    url_values = [relative_url] if relative_url else _first_field_values(
        relation,
        ("url", "fileUrl", "relativeUrl", "path", "filePath"),
    )
    if url_values:
        raw["url"] = url_values

    metadata_fallbacks = {
        "sequence": metadata.get("sequence"),
        "size": metadata.get("size"),
        "width": metadata.get("width"),
        "name": metadata.get("name"),
        "fileType": metadata.get("file_type"),
        "height": metadata.get("height"),
    }
    for key, value in metadata_fallbacks.items():
        if key not in raw and value not in (None, "", [], {}):
            raw[key] = value

    return raw


def _first_field_text(container: dict[str, Any], keys: tuple[str, ...]) -> str:
    for source in _metadata_sources(container):
        for key in keys:
            text = _first_text(source.get(key))
            if text:
                return text
    return ""


def _first_field_values(container: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for source in _metadata_sources(container):
        for key in keys:
            values = [text for text in _field_values(source.get(key)) if text]
            if values:
                return values
    return []


def _first_field_int(container: dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    text = _first_field_text(container, keys)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _metadata_sources(container: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [container]
    dp = container.get("dp")
    if isinstance(dp, dict):
        sources.append(dp)
    return sources


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
        text = _first_text(value)
        return [text] if text else []
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
        for key in (
            "sysId",
            "id",
            "docId",
            "url",
            "fileUrl",
            "relativeUrl",
            "path",
            "filePath",
            "name",
            "title",
        ):
            text = _first_text(value.get(key))
            if text:
                return text
    return ""


def _filename_from_url(relative_url: str) -> str:
    path = relative_url.split("?", 1)[0].split("#", 1)[0].strip()
    if not path:
        return ""
    return PurePosixPath(path.replace("\\", "/")).name


def _caption_from_metadata(metadata: dict[str, Any]) -> str:
    title = canonicalize_context_text(metadata.get("title"))
    if title:
        return title
    img_meta = metadata.get("img_meta")
    if isinstance(img_meta, dict):
        caption = canonicalize_context_text(img_meta.get("caption"))
        if caption:
            return caption
    return _default_caption()


def _default_caption() -> str:
    return str(settings.IMAGE_DEFAULT_CAPTION or "No caption").strip() or "No caption"


def _looks_like_mmd(relative_url: str) -> bool:
    path = str(relative_url or "").split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(".mmd")
