"""The fixed 41-field contract and lossless Solr mapping for scientific_documents_v2."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
LANG_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_,;-]{0,31}\|")
VECTOR_FIELDS = ("title_embed", "abstract_embed", "keyword_embed")
STRUCTURED_SOURCES = {
    "hasAuthor_str": "stringOM_hasAuthor",
    "hasSource_str": "stringOM_hasSource",
    "hasFile_str": "stringOM_hasFile",
    "hasCreateOrganization_str": "stringOM_hasCreateOrganization",
}


@dataclass(frozen=True)
class Field:
    kind: str
    max_length: int = 0
    max_capacity: int = 0
    dim: int = 0
    nullable: bool = True


# Source: out/solr_enriched_100_20261016/source_schema.json (24 original fields)
# and solr_field_backfill/field_mapping.json plus nullable entityType (17 fields).
FIELDS: dict[str, Field] = {
    "id": Field("varchar", 256, nullable=False),
    "cls": Field("varchar", 256),
    "title": Field("array", 65535, 4096),
    "abstract": Field("array", 65535, 4096),
    "year": Field("varchar", 64),
    "sysSubType": Field("varchar", 256),
    "keyword": Field("array", 65535, 4096),
    "language": Field("array", 65535, 4096),
    "accessType": Field("array", 65535, 4096),
    "classificationCode": Field("array", 65535, 4096),
    "sysDataStatus": Field("varchar", 64),
    "sysDataUpdateTime": Field("varchar", 64),
    "sysSourceId": Field("varchar", 256),
    "doi": Field("varchar", 1024),
    "hasFullTextFileid": Field("array", 65535, 4096),
    "hasFullTextFilename": Field("array", 65535, 4096),
    "hasAuthorname": Field("array", 65535, 4096),
    "hasCreateOrganizationname": Field("array", 65535, 4096),
    "hasSourcesysSourceId": Field("array", 65535, 4096),
    "hasSourceidSet": Field("array", 65535, 4096),
    "sync_status": Field("varchar", 64),
    "title_embed": Field("vector", dim=2560, nullable=False),
    "abstract_embed": Field("vector", dim=2560, nullable=False),
    "keyword_embed": Field("vector", dim=2560, nullable=False),
    "hasSource": Field("varchar", 20000),
    "hasFile": Field("varchar", 500),
    "volume": Field("varchar", 50),
    "issue": Field("varchar", 50),
    "startPage": Field("varchar", 50),
    "endPage": Field("varchar", 50),
    "pageCount": Field("varchar", 50),
    "createTime": Field("varchar", 50),
    "publisher": Field("varchar", 100),
    "labelEvaluateGrade": Field("varchar", 100),
    "sysProcessingUnit": Field("varchar", 100),
    "hasHoldinglibraryCode": Field("varchar", 100),
    "hasAuthor_str": Field("varchar", 20000),
    "hasSource_str": Field("varchar", 20000),
    "hasFile_str": Field("varchar", 500),
    "hasCreateOrganization_str": Field("varchar", 20000),
    "entityType": Field("varchar", 50),
}
assert len(FIELDS) == 41

SOLR_SCALARS = {
    "cls": "type",
    "year": "stringIS_year",
    "sysSubType": "stringIS_sysSubType",
    "sysDataStatus": "stringIS_sysDataStatus",
    "sysDataUpdateTime": "dateIS_sysDataUpdateTime",
    "sysSourceId": "stringIS_sysSourceId",
    "doi": "stringIS_doi",
}
SOLR_ARRAYS = {
    "title": "auto_stringITMS_title",
    "abstract": "auto_stringITMS_abstract",
    "keyword": "auto_stringITMS_keyword",
    "language": "stringIMS_language",
    "accessType": "stringIMS_accessType",
    "classificationCode": "stringIMS_classificationCode",
    "hasAuthorname": "auto_stringITMS_hasAuthorname",
    "hasCreateOrganizationname": "auto_stringITMS_hasCreateOrganizationname",
    "hasSourcesysSourceId": "stringIMS_hasSourcesysSourceId",
    "hasSourceidSet": "stringIMS_hasSourceidSet",
}
SOLR_FIELDS = tuple(dict.fromkeys((
    "id", *SOLR_SCALARS.values(), *SOLR_ARRAYS.values(),
    "stringOM_hasFullTextFile", "stringOM_hasAuthor", "stringOM_hasCreateOrganization",
    "stringIMS_idSet", "stringIMS_volume", "stringIMS_issue", "stringIS_startPage",
    "stringIS_endPage", "stringIS_pageCount", "dateIS_createTime",
    "stringIMS_labelEvaluateGrade", "stringIS_sysProcessingUnit",
    "stringIMS_hasHoldinglibraryCode", "stringIMS_entityType",
    "stringOM_hasSource", "auto_stringITMS_hasSourcetitle",
    "stringOM_hasFile", "auto_stringITMS_publisher",
)))


class PermanentDataError(ValueError):
    """A single source row cannot be written without discarding information."""


def require_id(value: Any) -> str:
    doc_id = str(value or "").strip()
    if not ID_RE.fullmatch(doc_id):
        raise PermanentDataError("document ID must be 32 hexadecimal characters")
    return doc_id.lower()


def _values(raw: Any, *, strip_language: bool = False) -> list[str]:
    values = raw if isinstance(raw, list) else [raw]
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if strip_language:
            text = LANG_PREFIX.sub("", text)
        if text:
            result.append(text)
    return result


def _relation_parts(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items() if value is not None}
    parts: dict[str, str] = {}
    for segment in str(raw or "").split("∷"):
        key, separator, value = segment.partition("≡")
        if separator and key.strip() and value.strip():
            parts[key.strip()] = value.strip()
    return parts


def _first(raw: Any) -> Any:
    """V4 first_value: an empty first element never falls through to later ones."""
    return raw[0] if isinstance(raw, list) and raw else (None if isinstance(raw, list) else raw)


def _v4_scalar(raw: Any, *, strip_language: bool = False) -> str:
    first = _first(raw)
    if first is None or first == "":
        return ""
    value = str(first)
    return value.split("|", 1)[1] if strip_language and "|" in value else value


def _structured_item(raw: Any) -> dict[str, str]:
    """V4 structured-field parsing, including Solr's nested ≡ / = variants."""
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items() if value is not None}
    parts = str(raw).split("∷")
    parsed = {"id": parts[0]} if parts else {}
    for part in parts[1:] if len(parts) > 1 else parts:
        separator = "≡" if "≡" in part else "=" if "=" in part else None
        if separator is None:
            continue
        key, value = part.split(separator, 1)
        # The V4 input also contains e.g. fileType≡fileType=pdf. Some Solr
        # documents use fileType≡fileType≡pdf; both mean the same file type.
        if key not in {"name", "title", "fileType"}:
            nested = "=" if "=" in value else "≡" if "≡" in value else None
            if nested is not None:
                key, value = value.split(nested, 1)
        elif value.startswith(key + "=") or value.startswith(key + "≡"):
            value = value[len(key) + 1:]
        if key and value:
            parsed[key] = value
    return parsed


def _v4_structured(raw: Any, key: str) -> str:
    return ";".join(_structured_values(raw, key))


def _structured_values(raw: Any, key: str) -> list[str]:
    values = raw if isinstance(raw, list) else [raw]
    result = []
    for value in values:
        if value is None or value == "":
            continue
        parsed = _structured_item(value)
        extracted = parsed.get(key, "")
        if not extracted and set(parsed) <= {"id"}:
            extracted = str(value)
        if extracted:
            result.append(extracted)
    return result


def _v4_raw_string(raw: Any) -> str:
    if raw is None or raw == "" or raw == []:
        return ""
    return json.dumps(raw, ensure_ascii=False) if isinstance(raw, list) else str(raw)


def validate_value(name: str, value: Any) -> None:
    spec = FIELDS[name]
    if value is None:
        if not spec.nullable:
            raise PermanentDataError(f"{name} may not be null")
        return
    if spec.kind == "varchar":
        if not isinstance(value, str) or len(value.encode("utf-8")) > spec.max_length:
            raise PermanentDataError(f"{name} is not a valid VARCHAR({spec.max_length})")
    elif spec.kind == "array":
        if not isinstance(value, list) or len(value) > spec.max_capacity:
            raise PermanentDataError(f"{name} exceeds ARRAY capacity {spec.max_capacity}")
        for item in value:
            if not isinstance(item, str) or len(item.encode("utf-8")) > spec.max_length:
                raise PermanentDataError(f"{name} contains an invalid VARCHAR({spec.max_length})")
    elif spec.kind == "vector":
        if (not isinstance(value, list) or len(value) != spec.dim or
                any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)):
            raise PermanentDataError(f"{name} must be a finite {spec.dim}-dimensional vector")


def source_values(doc: dict[str, Any], expected_id: str) -> dict[str, Any]:
    """Return only fields with confirmed, nonempty Solr source values.

    Missing Solr fields are omitted so an existing value is never cleared.
    The 17 added fields follow V4's source and conversion rules where the
    Kafka ID and Solr response contain enough information.
    """
    if require_id(doc.get("id")) != require_id(expected_id):
        raise PermanentDataError("Solr returned a different document ID")
    row: dict[str, Any] = {}
    for name, source in SOLR_SCALARS.items():
        values = _values(doc.get(source))
        if values:
            row[name] = values[0]
    for name, source in SOLR_ARRAYS.items():
        values = _values(doc.get(source), strip_language=True)
        if not values and name == "hasSourceidSet":
            values = _values(doc.get("stringIMS_idSet"), strip_language=True)
        if not values and name in ("hasAuthorname", "hasCreateOrganizationname"):
            fallback = "stringOM_hasAuthor" if name == "hasAuthorname" else "stringOM_hasCreateOrganization"
            values = _structured_values(doc.get(fallback), "name")
        if values:
            row[name] = values

    raw_relations = doc.get("stringOM_hasFullTextFile") or []
    if not isinstance(raw_relations, list):
        raw_relations = [raw_relations]
    relations = [_relation_parts(value) for value in raw_relations]
    # Match the existing metadata lane's .mmd-only full-text relation behavior.
    relations = [part for part in relations if re.search(r"\.mmd(?:[?#].*)?$", part.get("url", ""), re.I)]
    names = [part["name"] for part in relations if part.get("name")]
    ids = [part.get("id") or part.get("fileId") or part.get("sourceId") for part in relations]
    if names:
        row["hasFullTextFilename"] = names
    if any(ids):
        row["hasFullTextFileid"] = [value for value in ids if value]

    # V4 scalars: first element only. Prefix removal applies only to fields
    # where V4 calls strip_lang_prefix, and never skips an empty first item.
    added_scalars = {
        "volume": ("stringIMS_volume", True),
        "issue": ("stringIMS_issue", True),
        "startPage": ("stringIS_startPage", False),
        "endPage": ("stringIS_endPage", False),
        "pageCount": ("stringIS_pageCount", False),
        "createTime": ("dateIS_createTime", False),
        "publisher": ("auto_stringITMS_publisher", False),
        "labelEvaluateGrade": ("stringIMS_labelEvaluateGrade", True),
        "sysProcessingUnit": ("stringIS_sysProcessingUnit", False),
        "hasHoldinglibraryCode": ("stringIMS_hasHoldinglibraryCode", True),
        "entityType": ("stringIMS_entityType", True),
    }
    for name, (source, strip_language) in added_scalars.items():
        value = _v4_scalar(doc.get(source), strip_language=strip_language)
        if value:
            row[name] = value

    for name, source, key in (("hasSource", "stringOM_hasSource", "title"),
                              ("hasFile", "stringOM_hasFile", "fileType")):
        value = _v4_structured(doc.get(source), key)
        if value:
            row[name] = value
    # Existing Solr documents can have a flattened source title but no
    # structured relation. Keep that confirmed source as a compatibility path.
    if "hasSource" not in row and not doc.get("stringOM_hasSource"):
        values = _values(doc.get("auto_stringITMS_hasSourcetitle"), strip_language=True)
        if values:
            row["hasSource"] = ";".join(values)
    for name, source in STRUCTURED_SOURCES.items():
        value = _v4_raw_string(doc.get(source))
        if value:
            row[name] = value

    # sync_status is an internal processing marker, not a Solr-sourced field.
    for name, value in row.items():
        validate_value(name, value)
    return row


def vector_inputs(values: dict[str, Any], *, max_chars: int = 30000) -> dict[str, str]:
    delimiters = {"title": "\n", "abstract": "\n", "keyword": "; "}
    return {name: delimiters[name].join(values[name])[:max_chars]
            for name in delimiters if name in values}


def new_record(doc_id: str, values: dict[str, Any], vectors: dict[str, list[float]]) -> dict[str, Any]:
    row = {name: None for name in FIELDS}
    row["id"] = require_id(doc_id)
    row.update(values)
    row["sync_status"] = "SUCCESS"
    for field in VECTOR_FIELDS:
        row[field] = vectors.get(field) or [1e-6, *([0.0] * 2559)]
    assert_write_payload(row, new=True)
    return row


def update_payload(doc_id: str, values: dict[str, Any], vectors: dict[str, list[float]],
                   existing: dict[str, Any]) -> dict[str, Any]:
    if require_id(existing.get("id")) != require_id(doc_id):
        raise PermanentDataError("Strong read returned a different document ID")
    proposal = {"id": require_id(doc_id), **values, **vectors}
    payload = {name: value for name, value in proposal.items()
               if name == "id" or existing.get(name) != value}
    assert_write_payload(payload, new=False)
    return payload


def assert_write_payload(row: dict[str, Any], *, new: bool) -> None:
    if "id" not in row or require_id(row["id"]) != row["id"]:
        raise PermanentDataError("invalid write ID")
    if not set(row) <= set(FIELDS):
        raise PermanentDataError("payload contains a forbidden field")
    if not new and "sync_status" in row:
        raise PermanentDataError("existing sync_status must be preserved")
    if new and set(row) != set(FIELDS):
        raise PermanentDataError("new record does not match the 41-field contract")
    for name, value in row.items():
        validate_value(name, value)
