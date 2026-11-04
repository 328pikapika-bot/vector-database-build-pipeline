"""DOCX acceptance field contract for vector-library synchronization."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


DOCUMENT_VECTOR_FIELDS = ("title_embed", "abstract_embed", "keyword_embed")
LEGACY_DOCUMENT_VECTOR_FIELDS = ("title_vector", "abstract_vector", "keyword_vector")
# V2 stores the Solr document ID in the lowercase `id` primary key.
LEGACY_DOCUMENT_SCHEMA_FIELDS = frozenset(("ID", "sysId", "dynamic_fields"))


@dataclass(frozen=True)
class DocumentScalarFieldSpec:
    """Milvus scalar field derived from the Article contract."""

    name: str
    kind: str
    is_array: bool = False
    max_length: int = 65535
    max_capacity: int = 4096


_DOCUMENT_EXCLUDED_SCALAR_FIELDS = frozenset(
    (
        "id",
        "ID",
        "sysId",
        "cls",
        "dynamic_fields",
        "sourceData",
        *DOCUMENT_VECTOR_FIELDS,
        *LEGACY_DOCUMENT_VECTOR_FIELDS,
    )
)

_FALLBACK_DOCUMENT_SCALAR_FIELD_SPECS = (
    DocumentScalarFieldSpec("title", "varchar", is_array=True),
    DocumentScalarFieldSpec("abstract", "varchar", is_array=True),
    DocumentScalarFieldSpec("year", "varchar", max_length=64),
    DocumentScalarFieldSpec("sysSubType", "varchar", max_length=256),
    DocumentScalarFieldSpec("keyword", "varchar", is_array=True),
    DocumentScalarFieldSpec("language", "varchar", is_array=True),
    DocumentScalarFieldSpec("accessType", "varchar", is_array=True),
    DocumentScalarFieldSpec("classificationCode", "varchar", is_array=True),
    DocumentScalarFieldSpec("sysDataStatus", "varchar", max_length=64),
    DocumentScalarFieldSpec("sysDataUpdateTime", "varchar", max_length=64),
    DocumentScalarFieldSpec("sysSourceId", "varchar", max_length=256),
    DocumentScalarFieldSpec("doi", "varchar", max_length=1024),
    DocumentScalarFieldSpec("hasFullTextFileid", "varchar", is_array=True),
    DocumentScalarFieldSpec("hasFullTextFilename", "varchar", is_array=True),
    DocumentScalarFieldSpec("hasAuthorname", "varchar", is_array=True),
    DocumentScalarFieldSpec("hasCreateOrganizationname", "varchar", is_array=True),
    DocumentScalarFieldSpec("hasSourcesysSourceId", "varchar", is_array=True),
    DocumentScalarFieldSpec("hasSourceidSet", "varchar", is_array=True),
    DocumentScalarFieldSpec("sync_status", "varchar", max_length=64),
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.joinpath("doc", "models", "base", "article.py").exists():
            return parent
    return current.parents[3]


def _annotation_to_spec(name: str, annotation: str) -> Optional[DocumentScalarFieldSpec]:
    if name in _DOCUMENT_EXCLUDED_SCALAR_FIELDS:
        return None

    normalized = annotation.replace("typing.", "").replace(" ", "")
    normalized = _unwrap_optional_annotation(normalized)
    is_array = False
    if normalized.startswith("List[") and normalized.endswith("]"):
        is_array = True
        normalized = normalized[5:-1]
    elif normalized.startswith("list[") and normalized.endswith("]"):
        is_array = True
        normalized = normalized[5:-1]

    kind_by_annotation = {
        "str": "varchar",
        "int": "int64",
        "float": "float",
    }
    kind = kind_by_annotation.get(normalized)
    if kind is None:
        return None

    max_length = 64 if name.lower().endswith(("time", "date", "year")) else 65535
    if name in {"year", "standardYear"}:
        max_length = 64
    elif name in {"sysSubType", "sysDataStatus", "sysSourceId"}:
        max_length = 256

    return DocumentScalarFieldSpec(name=name, kind=kind, is_array=is_array, max_length=max_length)


def _unwrap_optional_annotation(annotation: str) -> str:
    if annotation.startswith("Optional[") and annotation.endswith("]"):
        return annotation[len("Optional[") : -1]
    if annotation.startswith("Union[") and annotation.endswith("]"):
        inner = annotation[len("Union[") : -1]
        parts = [part for part in inner.split(",") if part not in {"None", "NoneType"}]
        if len(parts) == 1:
            return parts[0]
    return annotation


def _load_article_model_scalar_specs() -> list[DocumentScalarFieldSpec]:
    article_path = _repo_root().joinpath("doc", "models", "base", "article.py")
    if not article_path.exists():
        return []

    try:
        tree = ast.parse(article_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    specs: list[DocumentScalarFieldSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        spec = _annotation_to_spec(node.target.id, ast.unparse(node.annotation))
        if spec is not None:
            specs.append(spec)
    return specs


def _load_article_mapping_scalar_specs(existing_names: set[str]) -> list[DocumentScalarFieldSpec]:
    doc_dir = _repo_root().joinpath("doc")
    if not doc_dir.exists():
        return []

    try:
        import openpyxl
    except Exception:
        return []

    specs: list[DocumentScalarFieldSpec] = []
    for path in sorted(doc_dir.glob("*.xlsx")):
        try:
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            worksheet = workbook.worksheets[0]
            if worksheet.cell(row=3, column=1).value != "Article":
                workbook.close()
                continue

            in_data_property = False
            for row in worksheet.iter_rows(values_only=True):
                marker = str(row[0]).strip() if row and row[0] is not None else ""
                if marker == "data-property":
                    in_data_property = True
                    continue
                if marker == "object-property":
                    break
                if not in_data_property or marker in {"", "field"}:
                    continue
                if marker in existing_names or marker in _DOCUMENT_EXCLUDED_SCALAR_FIELDS:
                    continue

                type_text = str(row[2] or "string").strip().lower()
                multi_value = row[5] is True or str(row[5] or "").strip().lower() == "true"
                if type_text in {"int", "integer", "long"}:
                    kind = "int64"
                elif type_text in {"double", "float"}:
                    kind = "float"
                else:
                    kind = "varchar"
                max_length = 64 if type_text in {"date"} or marker.lower().endswith(("time", "date")) else 65535
                specs.append(
                    DocumentScalarFieldSpec(
                        name=marker,
                        kind=kind,
                        is_array=multi_value,
                        max_length=max_length,
                    )
                )
                existing_names.add(marker)
            workbook.close()
            break
        except Exception:
            continue
    return specs


def _build_document_scalar_specs() -> tuple[DocumentScalarFieldSpec, ...]:
    # Keep the document collection below Milvus' field-count limit.  The full
    # external Article/xlsx contract can exceed 200 scalar fields, so the vector
    # collection stores only the fields needed for retrieval, filtering, and
    # fulltext/image task discovery.
    specs = list(_FALLBACK_DOCUMENT_SCALAR_FIELD_SPECS)

    deduped: list[DocumentScalarFieldSpec] = []
    emitted: set[str] = set()
    for spec in specs:
        if spec.name in emitted or spec.name in _DOCUMENT_EXCLUDED_SCALAR_FIELDS:
            continue
        emitted.add(spec.name)
        deduped.append(spec)
    return tuple(deduped)


DOCUMENT_SCALAR_FIELD_SPECS = _build_document_scalar_specs()
DOCUMENT_SCALAR_FIELD_NAMES = tuple(spec.name for spec in DOCUMENT_SCALAR_FIELD_SPECS)
DOCUMENT_SCALAR_FIELD_NAME_SET = frozenset(DOCUMENT_SCALAR_FIELD_NAMES)

DOCUMENT_BASE_FIELDS = (
    "id",
    "cls",
    *DOCUMENT_SCALAR_FIELD_NAMES,
    *DOCUMENT_VECTOR_FIELDS,
)

DOCUMENT_SCALAR_INDEX_FIELDS = (
    "id",
    "sysDataStatus",
    "sysSubType",
    "sysDataUpdateTime",
    "year",
    "language",
    "accessType",
    "classificationCode",
    "sysSourceId",
    "doi",
    "hasAuthorname",
    "hasCreateOrganizationname",
    "hasSourcesysSourceId",
    "hasSourceidSet",
    "sync_status",
)

FULLTEXT_REQUIRED_FIELDS = (
    "language",
    "id",
    "chunkIndex",
    "chunkType",
    "text",
    "charStart",
    "charEnd",
    "sectionPath",
    "tokenCount",
    "text_embed",
)

FULLTEXT_DOC_FIELDS = (
    "doc_id",
    "title",
    "keyword",
    "accessType",
    "classificationCode",
    "year",
    "hasAuthorname",
    "hasCreateOrganizationname",
    "language",
)

FULLTEXT_FIELDS = (
    *FULLTEXT_DOC_FIELDS,
    "id",
    "chunkIndex",
    "chunkType",
    "text",
    "charStart",
    "charEnd",
    "sectionPath",
    "tokenCount",
    "text_embed",
)

# The V2 image collection preserves the historical JSON-oriented layout.  The
# only intentional structural change is the top-level document foreign key.
IMAGE_REQUIRED_FIELDS = (
    "id",
    "doc_id",
    "doc_meta",
    "raw_img_meta",
    "img_meta",
    "p_hash",
    "visual_embed",
    "caption_embed",
    "context_embed",
    "text",
    "text_embed",
    "visual_text_embed",
    "text_sparse_vec",
)

IMAGE_DOC_FIELDS = (
    "docId",
    "title",
    "keyword",
    "accessType",
    "classificationCode",
    "year",
    "hasAuthorname",
    "hasCreateOrganizationname",
    "language",
    "sysDataStatus",
    "hasSourcesysSourceId",
    "hasSourceidSet",
)

IMAGE_FIELDS = (
    "id",
    "doc_id",
    "doc_meta",
    "raw_img_meta",
    "img_meta",
    "context_before",
    "context_after",
    "p_hash",
    "visual_embed",
    "caption_embed",
    "context_embed",
    "text",
    "text_embed",
    "visual_text_embed",
    "text_sparse_vec",
)

IMAGE_SCALAR_INDEX_FIELDS = (
    "id",
    "doc_id",
    "img_meta.img_id",
    "doc_meta.sysDataStatus",
    "doc_meta.year",
    "doc_meta.classificationCode",
    "doc_meta.language",
    "doc_meta.accessType",
    "doc_meta.keyword",
    "doc_meta.hasAuthorname",
    "doc_meta.hasCreateOrganizationname",
    "doc_meta.hasSourcesysSourceId",
    "doc_meta.hasSourceidSet",
)

RELATION_CHILD_FIELD_ALIASES = {
    "fileName": ("name",),
    "filename": ("name",),
    "file_name": ("name",),
    "fileUrl": ("url",),
    "file_url": ("url",),
    "imageId": ("id",),
    "image_id": ("id",),
}

TEXT_VECTOR_DIM = 2560
VISUAL_VECTOR_DIM = 4096
VISUAL_TEXT_VECTOR_DIM = 4096
PHASH_BITS = 64
PHASH_BYTES = PHASH_BITS // 8


def first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, set)) and not value:
            continue
        if isinstance(value, dict) and not value:
            continue
        return value
    return None


def first_mapping_value(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            value = first_present(source.get(key))
            if value is not None:
                return value
    return None


def normalize_string_array(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items or None
    text = str(value).strip()
    return [text] if text else None


def normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            text = normalize_optional_string(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def normalize_document_id(source: Any) -> Optional[str]:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return normalize_optional_string(first_mapping_value(source, "ID", "sysId", "doc_id", "docId", "id"))
    for key in ("ID", "sysId", "doc_id", "docId", "id"):
        if hasattr(source, key):
            value = normalize_optional_string(getattr(source, key))
            if value:
                return value
    return None


def normalize_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"_raw": value}


def candidate_relation_scalar_field_names(rel_name: str, child_key: str) -> tuple[str, ...]:
    child_key = str(child_key or "").strip()
    if not rel_name or not child_key:
        return ()

    candidates: list[str] = [child_key]
    if "_" in child_key:
        candidates.append(_snake_to_camel(child_key))
    candidates.extend(RELATION_CHILD_FIELD_ALIASES.get(child_key, ()))

    field_names: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        field_name = f"{rel_name}{candidate}"
        if field_name in DOCUMENT_SCALAR_FIELD_NAME_SET and field_name not in seen:
            seen.add(field_name)
            field_names.append(field_name)
    return tuple(field_names)


def _snake_to_camel(value: str) -> str:
    parts = [part for part in value.split("_") if part]
    if not parts:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def extract_docx_doc_meta(source: Any) -> dict[str, Any]:
    """Extract document metadata fields from a model, dict, or legacy doc_meta."""

    if source is None:
        return {}
    if isinstance(source, Mapping):
        direct = dict(source)
    else:
        direct = {}
        for key in set((*FULLTEXT_DOC_FIELDS, *IMAGE_DOC_FIELDS, "sysId", "docId", "doc_id", "doc_meta")):
            if hasattr(source, key):
                direct[key] = getattr(source, key)

    nested_meta = direct.get("doc_meta")
    if isinstance(nested_meta, Mapping):
        direct = {**dict(nested_meta), **direct}

    sys_id = first_mapping_value(direct, "ID", "sysId", "doc_id", "docId", "id")
    title = first_mapping_value(direct, "title")
    keyword = first_mapping_value(direct, "keyword")
    document_id = normalize_optional_string(sys_id)
    return {
        "ID": document_id,
        "sysId": document_id,
        "docId": document_id,
        "doc_id": document_id,
        "title": normalize_string_array(title),
        "keyword": normalize_string_array(keyword),
        "accessType": normalize_optional_string(first_mapping_value(direct, "accessType")),
        "classificationCode": normalize_optional_string(first_mapping_value(direct, "classificationCode")),
        "year": normalize_optional_string(first_mapping_value(direct, "year")),
        "hasAuthorname": normalize_string_array(first_mapping_value(direct, "hasAuthorname")),
        "hasCreateOrganizationname": normalize_string_array(
            first_mapping_value(direct, "hasCreateOrganizationname")
        ),
        "language": normalize_string_array(first_mapping_value(direct, "language", "languageIdentify")),
        "sysDataStatus": normalize_optional_string(first_mapping_value(direct, "sysDataStatus")),
        "hasSourcesysSourceId": normalize_string_array(first_mapping_value(direct, "hasSourcesysSourceId")),
        "hasSourceidSet": normalize_string_array(first_mapping_value(direct, "hasSourceidSet")),
    }
