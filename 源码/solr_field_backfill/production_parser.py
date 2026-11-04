"""Solr values parsed with the build_milvus_db_all_v4 metadata rules.

The reference importer reads local JSON, while this worker reads Solr.  The
source names below are the corresponding Solr fields; absent source values are
left as null so the worker never invents metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any


PARSER_VERSION = "build_milvus_db_all_v4-solr-v1"

SOLR_SOURCES: dict[str, str] = {
    "hasSource": "stringOM_hasSource",
    "hasFile": "stringOM_hasFile",
    "volume": "stringIMS_volume",
    "issue": "stringIMS_issue",
    "startPage": "stringIS_startPage",
    "endPage": "stringIS_endPage",
    "pageCount": "stringIS_pageCount",
    "createTime": "dateIS_createTime",
    "publisher": "auto_stringITMS_publisher",
    "labelEvaluateGrade": "stringIMS_labelEvaluateGrade",
    "sysProcessingUnit": "stringIS_sysProcessingUnit",
    "hasHoldinglibraryCode": "stringIMS_hasHoldinglibraryCode",
    "hasAuthor_str": "stringOM_hasAuthor",
    "hasSource_str": "stringOM_hasSource",
    "hasFile_str": "stringOM_hasFile",
    "hasCreateOrganization_str": "stringOM_hasCreateOrganization",
    "entityType": "stringIMS_entityType",
}

SOLR_QUERY_FIELDS = tuple(dict.fromkeys(("id", *SOLR_SOURCES.values())))
LANG_PREFIX_FIELDS = frozenset({
    "volume", "issue", "labelEvaluateGrade", "hasHoldinglibraryCode", "entityType",
})
RAW_STRUCTURED_FIELDS = frozenset({
    "hasAuthor_str", "hasSource_str", "hasFile_str", "hasCreateOrganization_str",
})
PLAIN_FILE_TYPE = re.compile(r"^[A-Za-z0-9]+$")


def first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def strip_lang_prefix(value: Any) -> str:
    text = str(value)
    return text.split("|", 1)[1] if "|" in text else text


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _parse_structured_item(value: Any) -> dict[str, str]:
    """Match the reference importer's structured ∷ and ≡ decoding."""
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}

    text = str(value)
    result: dict[str, str] = {}
    parts = text.split("∷")
    if parts:
        result["id"] = parts[0]

    for part in parts[1:] if len(parts) > 1 else parts:
        if "≡" in part:
            key, raw_value = part.split("≡", 1)
        elif "=" in part:
            key, raw_value = part.split("=", 1)
        else:
            continue

        if "=" in raw_value and key not in {"name", "title", "fileType"}:
            nested_key, nested_value = raw_value.split("=", 1)
            result[nested_key] = nested_value
        else:
            result[key] = raw_value
    return result


def _join_structured(value: Any, field_name: str) -> str | None:
    values: list[str] = []
    for item in _as_list(value):
        parsed = _parse_structured_item(item)
        parsed_value = parsed.get(field_name, "")
        if not parsed_value and set(parsed) <= {"id"}:
            # The reference importer accepts an unstructured item as-is.  A
            # file URL is not a file type, however; only a plain type token
            # may use that fallback when parsing hasFile.
            plain = str(item)
            if field_name != "fileType" or PLAIN_FILE_TYPE.fullmatch(plain):
                parsed_value = plain
        if parsed_value:
            values.append(parsed_value)
    return ";".join(values) or None


def _raw_structured_as_string(value: Any) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_field(doc: dict[str, Any], field: str) -> str | None:
    source = SOLR_SOURCES[field]
    raw = doc.get(source)
    if field == "hasSource":
        return _join_structured(raw, "title")
    if field == "hasFile":
        return _join_structured(raw, "fileType")
    if field in RAW_STRUCTURED_FIELDS:
        return _raw_structured_as_string(raw)

    value = first_value(raw)
    if value is None or value == "":
        return None
    result = strip_lang_prefix(value) if field in LANG_PREFIX_FIELDS else str(value)
    return result or None


def parse_all(doc: dict[str, Any]) -> dict[str, str | None]:
    return {field: parse_field(doc, field) for field in SOLR_SOURCES}
