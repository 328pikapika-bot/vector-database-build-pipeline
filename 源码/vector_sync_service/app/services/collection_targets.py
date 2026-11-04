"""Resolve Milvus collection targets and index signatures for document/fulltext writes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from app.config import settings


SUPPORTED_INDEX_TYPES = {"IVFPQ"}
# Configuration uses the compact ``IVFPQ`` spelling; Milvus API metadata uses
# ``IVF_PQ``.  Keep this translation local to the target resolver so no legacy
# index-type configuration leaks into the rest of the application.
INDEX_TYPE_BY_CONFIG = {"IVFPQ": "IVF_PQ"}
TARGET_SIGNATURE_DESCRIPTION_PREFIX = "vector-sync-target-signature:"


@dataclass(frozen=True, slots=True)
class IndexSpec:
    index_type: str
    metric_type: str
    build_params: dict[str, Any]
    search_params: dict[str, Any]
    signature: str
    suffix: str


@dataclass(frozen=True, slots=True)
class CollectionTarget:
    logical_name: str
    resolved_name: str
    index_signature: str
    created: bool
    reused: bool
    source_collection_name: str
    schema_signature: str = ""
    legacy_reused: bool = False

    @property
    def target_signature(self) -> str:
        if not self.schema_signature:
            return self.index_signature
        return f"{self.schema_signature}|INDEX={self.index_signature}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "resolved_name": self.resolved_name,
            "index_signature": self.index_signature,
            "schema_signature": self.schema_signature,
            "target_signature": self.target_signature,
            "created": self.created,
            "reused": self.reused,
            "source_collection_name": self.source_collection_name,
            "legacy_reused": self.legacy_reused,
        }


def build_document_index_spec() -> IndexSpec:
    return _build_index_spec(
        index_type=settings.DOCUMENT_INDEX_TYPE,
        metric_type=settings.DOCUMENT_INDEX_METRIC_TYPE,
        ivfpq_nlist=settings.DOCUMENT_IVFPQ_NLIST,
        ivfpq_m=settings.DOCUMENT_IVFPQ_M,
        ivfpq_nbits=settings.DOCUMENT_IVFPQ_NBITS,
        ivfpq_nprobe=settings.DOCUMENT_IVFPQ_NPROBE,
    )


def build_fulltext_chunk_index_spec() -> IndexSpec:
    return _build_index_spec(
        index_type=settings.FULLTEXT_CHUNK_INDEX_TYPE,
        metric_type=settings.FULLTEXT_CHUNK_INDEX_METRIC_TYPE,
        ivfpq_nlist=settings.FULLTEXT_CHUNK_IVFPQ_NLIST,
        ivfpq_m=settings.FULLTEXT_CHUNK_IVFPQ_M,
        ivfpq_nbits=settings.FULLTEXT_CHUNK_IVFPQ_NBITS,
        ivfpq_nprobe=settings.FULLTEXT_CHUNK_IVFPQ_NPROBE,
    )


def build_image_dense_index_spec() -> IndexSpec:
    return _build_index_spec(
        index_type=settings.IMAGE_DENSE_INDEX_TYPE,
        metric_type=settings.IMAGE_DENSE_INDEX_METRIC_TYPE,
        ivfpq_nlist=settings.IMAGE_DENSE_IVFPQ_NLIST,
        ivfpq_m=settings.IMAGE_DENSE_IVFPQ_M,
        ivfpq_nbits=settings.IMAGE_DENSE_IVFPQ_NBITS,
        ivfpq_nprobe=settings.IMAGE_DENSE_IVFPQ_NPROBE,
    )


def build_image_collection_signature(*, schema_mode: Optional[str] = None) -> str:
    """Stable creation signature for the production image collection."""
    spec = build_image_dense_index_spec()
    mode = _resolve_image_schema_mode(schema_mode)
    is_history_compatible = mode == "history_compatible_v2"
    fields = "doc_meta,raw_img_meta,img_meta,p_hash,visual_embed,caption_embed,context_embed"
    if is_history_compatible:
        fields += ",text,text_embed,visual_text_embed,text_sparse_vec"
    return "|".join(
        [
            "IMAGE",
            "PK=id|DOC_ID=doc_id",
            f"MODE={mode}",
            f"FIELDS={fields}",
            "SPARSE_INDEX=PLACEHOLDER_ONLY|BM25=DISABLED"
            if is_history_compatible
            else "SPARSE_INDEX=ABSENT|BM25=DISABLED",
            "VISUAL_EMBED=ENABLED"
            if bool(getattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", False))
            else "VISUAL_EMBED=PLACEHOLDER_ONLY",
            f"VISUAL_MODEL={str(settings.IMAGE_VL_EMBED_MODEL or '').strip()}",
            f"VISUAL_MODEL_VERSION={str(settings.IMAGE_VL_EMBED_MODEL_VERSION or '').strip()}",
            f"TEXT_MODEL={str(settings.IMAGE_TEXT_EMBED_MODEL or '').strip()}",
            f"TEXT_MODEL_VERSION={str(settings.IMAGE_TEXT_EMBED_MODEL_VERSION or '').strip()}",
            f"INDEX={spec.signature}",
        ]
    )


def build_document_collection_signature(*, vector_dim: int = settings.VECTOR_DIM) -> str:
    return "|".join(
        [
            "DOCUMENT",
            "PK=id|FIELDS=title_embed,abstract_embed,keyword_embed",
            f"DIM={int(vector_dim)}",
            f"MODEL={str(settings.EMBEDDING_MODEL or '').strip()}",
            f"MODEL_VERSION={str(settings.EMBEDDING_MODEL_VERSION or '').strip()}",
        ]
    )


def build_fulltext_collection_signature(*, vector_dim: int = settings.VECTOR_DIM) -> str:
    return "|".join(
        [
            "FULLTEXT",
            "PK=id|FIELDS=text_embed",
            f"DIM={int(vector_dim)}",
            f"SCHEMA={str(settings.FULLTEXT_CHUNK_SCHEMA_VERSION or '').strip()}",
            f"MODEL={str(settings.EMBEDDING_MODEL or '').strip()}",
            f"MODEL_VERSION={str(settings.EMBEDDING_MODEL_VERSION or '').strip()}",
        ]
    )


def resolve_document_collection_target(
    client: Any,
    *,
    logical_name: Optional[str] = None,
    vector_dim: int = settings.VECTOR_DIM,
    ensure_collection: Optional[Callable[[str], None]] = None,
) -> CollectionTarget:
    return _resolve_collection_target(
        client,
        logical_name=str(logical_name or settings.resolved_document_collection_name).strip(),
        spec=build_document_index_spec(),
        vector_field_names=("title_embed", "abstract_embed", "keyword_embed"),
        required_schema_fields={"id", "title_embed", "abstract_embed", "keyword_embed"},
        required_vector_dims={
            "title_embed": int(vector_dim),
            "abstract_embed": int(vector_dim),
            "keyword_embed": int(vector_dim),
        },
        schema_signature=build_document_collection_signature(vector_dim=vector_dim),
        schema_suffix="schema_document",
        ensure_collection=ensure_collection,
    )


def resolve_fulltext_collection_target(
    client: Any,
    *,
    logical_name: Optional[str] = None,
    vector_dim: int = settings.VECTOR_DIM,
    ensure_collection: Optional[Callable[[str], None]] = None,
) -> CollectionTarget:
    return _resolve_collection_target(
        client,
        logical_name=str(logical_name or settings.resolved_fulltext_chunk_collection_name).strip(),
        spec=build_fulltext_chunk_index_spec(),
        vector_field_names=("text_embed",),
        required_schema_fields={
            "id",
            "doc_id",
            "text_embed",
        },
        schema_suffix=f"schema_{str(settings.FULLTEXT_CHUNK_SCHEMA_VERSION or 'v2').lower()}",
        required_vector_dims={"text_embed": int(vector_dim)},
        schema_signature=build_fulltext_collection_signature(vector_dim=vector_dim),
        ensure_collection=ensure_collection,
    )


def resolve_image_collection_target(
    client: Any,
    *,
    logical_name: Optional[str] = None,
    schema_mode: Optional[str] = None,
    ensure_collection: Optional[Callable[[str], None]] = None,
) -> CollectionTarget:
    mode = _resolve_image_schema_mode(schema_mode)
    is_history_compatible = mode == "history_compatible_v2"
    vector_field_names = ("visual_embed", "caption_embed", "context_embed")
    required_schema_fields = {
        "id",
        "doc_id",
        "doc_meta",
        "raw_img_meta",
        "img_meta",
        "context_before",
        "context_after",
        "p_hash",
        *vector_field_names,
    }
    required_vector_dims = {
        "visual_embed": 4096,
        "caption_embed": 2560,
        "context_embed": 2560,
    }
    if is_history_compatible:
        vector_field_names += ("text_embed", "visual_text_embed")
        required_schema_fields.update({"text", "text_embed", "visual_text_embed", "text_sparse_vec"})
        required_vector_dims.update({"text_embed": 2560, "visual_text_embed": 4096})
    return _resolve_collection_target(
        client,
        logical_name=str(logical_name or settings.IMAGE_COLLECTION_NAME).strip(),
        spec=build_image_dense_index_spec(),
        vector_field_names=vector_field_names,
        required_schema_fields=required_schema_fields,
        required_vector_dims=required_vector_dims,
        schema_signature=build_image_collection_signature(schema_mode=mode),
        schema_suffix=("schema_history_compatible_v2" if is_history_compatible else "schema_four_vector"),
        ensure_collection=ensure_collection,
    )


def _resolve_image_schema_mode(value: Optional[str]) -> str:
    mode = str(value or settings.IMAGE_VECTOR_SCHEMA_MODE or "").strip().lower()
    aliases = {"four_vector": "visual_caption_context_4_vector"}
    mode = aliases.get(mode, mode)
    if mode not in {"history_compatible_v2", "visual_caption_context_4_vector"}:
        raise ValueError(
            "image collection target resolution requires IMAGE_VECTOR_SCHEMA_MODE="
            "visual_caption_context_4_vector or history_compatible_v2"
        )
    return mode


def build_search_params_for_index_spec(spec: IndexSpec) -> dict[str, Any]:
    params = dict(spec.search_params)
    return {"metric_type": spec.metric_type, "params": params}


def build_search_params_for_collection_label(label: str) -> dict[str, Any]:
    normalized = str(label or "").strip().lower()
    if normalized == "document":
        spec = build_document_index_spec()
    elif normalized == "chunk":
        spec = build_fulltext_chunk_index_spec()
    elif normalized == "image":
        spec = build_image_dense_index_spec()
    else:
        raise ValueError(f"unsupported collection label: {label!r}")
    return build_search_params_for_index_spec(spec)


def collection_target_key(target: CollectionTarget) -> str:
    return f"{target.resolved_name}|{target.target_signature}"


def collection_target_description(target: CollectionTarget) -> str:
    """Persist a portable target marker in the Milvus schema description."""
    return collection_signature_description(target.schema_signature, target.index_signature)


def collection_signature_description(schema_signature: str, index_signature: str) -> str:
    return f"{TARGET_SIGNATURE_DESCRIPTION_PREFIX}{_target_signature(schema_signature, index_signature)}"


def collection_target_binding(
    target: CollectionTarget,
    *,
    pipeline: str,
    source_version: str = "",
    item_id: str = "",
) -> dict[str, str]:
    """Common failure/replay target binding persisted by every data pipeline."""
    return {
        "pipeline": str(pipeline or "").strip().lower(),
        "source_version": str(source_version or "").strip(),
        "item_id": str(item_id or "").strip(),
        "logical_collection": target.logical_name,
        "resolved_collection": target.resolved_name,
        "target_signature": target.target_signature,
    }


def _resolve_collection_target(
    client: Any,
    *,
    logical_name: str,
    spec: IndexSpec,
    vector_field_names: Sequence[str],
    required_schema_fields: Optional[set[str]] = None,
    forbidden_schema_fields: Optional[set[str]] = None,
    required_vector_dims: Optional[dict[str, int]] = None,
    schema_signature: str = "",
    schema_suffix: str = "",
    ensure_collection: Optional[Callable[[str], None]],
) -> CollectionTarget:
    resolved = str(logical_name or "").strip()
    if not resolved:
        raise ValueError("logical collection name is required")
    expected_target_signature = _target_signature(schema_signature, spec.signature)
    if not client.has_collection(resolved):
        if ensure_collection is not None:
            ensure_collection(resolved)
        return CollectionTarget(
            logical_name=resolved,
            resolved_name=resolved,
            index_signature=spec.signature,
            created=True,
            reused=False,
            source_collection_name=resolved,
            schema_signature=schema_signature,
        )
    index_matches = _collection_matches_index_signature(
        client, resolved, spec, vector_field_names=vector_field_names
    )
    schema_matches = _collection_matches_schema(
        client,
        resolved,
        required_fields=required_schema_fields,
        forbidden_fields=forbidden_schema_fields,
        required_vector_dims=required_vector_dims,
        expected_target_signature=expected_target_signature,
    )
    if index_matches and schema_matches:
        return CollectionTarget(
            logical_name=resolved,
            resolved_name=resolved,
            index_signature=spec.signature,
            created=False,
            reused=True,
            source_collection_name=resolved,
            schema_signature=schema_signature,
            legacy_reused=not _collection_has_recorded_target_signature(
                client,
                resolved,
                expected_target_signature,
            ),
        )

    if bool(getattr(settings, "COLLECTION_TARGET_STRICT", False)):
        raise ValueError(
            f"collection target {resolved!r} exists but its schema or index does not "
            "match the configured contract; choose an empty name for a new collection "
            "or fix the collection settings"
        )

    suffixes: list[str] = []
    if not index_matches:
        suffixes.append(spec.suffix)
    if index_matches and not schema_matches and schema_suffix:
        suffixes.append(schema_suffix)
    candidate = f"{resolved}__{'__'.join(suffixes or [spec.suffix])}"
    if client.has_collection(candidate):
        candidate_index_matches = _collection_matches_index_signature(
            client, candidate, spec, vector_field_names=vector_field_names
        )
        candidate_schema_matches = _collection_matches_schema(
            client,
            candidate,
            required_fields=required_schema_fields,
            forbidden_fields=forbidden_schema_fields,
            required_vector_dims=required_vector_dims,
            expected_target_signature=expected_target_signature,
        )
        if candidate_index_matches and candidate_schema_matches:
            return CollectionTarget(
                logical_name=resolved,
                resolved_name=candidate,
                index_signature=spec.signature,
                created=False,
                reused=True,
                source_collection_name=resolved,
                schema_signature=schema_signature,
                legacy_reused=not _collection_has_recorded_target_signature(
                    client,
                    candidate,
                    expected_target_signature,
                ),
            )
        if candidate_index_matches and not candidate_schema_matches and schema_suffix:
            candidate = f"{candidate}__{schema_suffix}"
            if client.has_collection(candidate):
                if (
                    _collection_matches_index_signature(
                        client, candidate, spec, vector_field_names=vector_field_names
                    )
                    and _collection_matches_schema(
                        client,
                        candidate,
                        required_fields=required_schema_fields,
                        forbidden_fields=forbidden_schema_fields,
                        required_vector_dims=required_vector_dims,
                        expected_target_signature=expected_target_signature,
                    )
                ):
                    return CollectionTarget(
                        logical_name=resolved,
                        resolved_name=candidate,
                        index_signature=spec.signature,
                        created=False,
                        reused=True,
                        source_collection_name=resolved,
                        schema_signature=schema_signature,
                        legacy_reused=not _collection_has_recorded_target_signature(
                            client,
                            candidate,
                            expected_target_signature,
                        ),
                    )
                raise ValueError(
                    f"collection target collision: {candidate} exists with an incompatible schema or index"
                )
        elif not candidate_index_matches:
            raise ValueError(
                f"collection target collision: {candidate} exists with an incompatible index"
            )
    if ensure_collection is not None:
        ensure_collection(candidate)
    return CollectionTarget(
        logical_name=resolved,
        resolved_name=candidate,
        index_signature=spec.signature,
        created=True,
        reused=False,
        source_collection_name=resolved,
        schema_signature=schema_signature,
    )


def _collection_matches_schema(
    client: Any,
    collection_name: str,
    *,
    required_fields: Optional[set[str]],
    forbidden_fields: Optional[set[str]] = None,
    required_vector_dims: Optional[dict[str, int]] = None,
    expected_target_signature: str = "",
) -> bool:
    if not required_fields and not forbidden_fields and not required_vector_dims:
        return True
    if not hasattr(client, "describe_collection"):
        return False
    try:
        description = client.describe_collection(collection_name)
    except Exception:
        return False
    if isinstance(description, dict):
        recorded_description = str(description.get("description") or "")
        fields = description.get("fields", [])
    else:
        recorded_description = str(getattr(description, "description", "") or "")
        fields = getattr(description, "fields", [])
    recorded_signature = _target_signature_from_description(recorded_description)
    if recorded_signature and recorded_signature != expected_target_signature:
        return False
    field_names = {
        str(field.get("name") or field.get("field_name") or "").strip()
        for field in fields
        if isinstance(field, dict)
    }
    if required_fields and not required_fields.issubset(field_names):
        return False
    if forbidden_fields and field_names.intersection(forbidden_fields):
        return False
    if not required_vector_dims:
        return True
    fields_by_name = {
        str(field.get("name") or field.get("field_name") or "").strip(): field
        for field in fields
        if isinstance(field, dict)
    }
    for field_name, expected_dim in required_vector_dims.items():
        field = fields_by_name.get(field_name)
        if field is None:
            return False
        actual_dim = field.get("dim")
        if actual_dim is None and isinstance(field.get("params"), dict):
            actual_dim = field["params"].get("dim")
        # Older Milvus descriptions can omit vector dimensions. Their field
        # set and vector index are still checked above, so do not force an
        # unnecessary collection migration solely for missing metadata.
        if actual_dim is None:
            continue
        try:
            if int(actual_dim) != int(expected_dim):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _target_signature(schema_signature: str, index_signature: str) -> str:
    if not schema_signature:
        return index_signature
    return f"{schema_signature}|INDEX={index_signature}"


def _target_signature_from_description(description: str) -> str:
    normalized = str(description or "")
    if not normalized.startswith(TARGET_SIGNATURE_DESCRIPTION_PREFIX):
        return ""
    return normalized.removeprefix(TARGET_SIGNATURE_DESCRIPTION_PREFIX).strip()


def _collection_has_recorded_target_signature(
    client: Any,
    collection_name: str,
    expected_target_signature: str,
) -> bool:
    if not expected_target_signature or not hasattr(client, "describe_collection"):
        return False
    try:
        description = client.describe_collection(collection_name)
    except Exception:
        return False
    if isinstance(description, dict):
        raw = description.get("description")
    else:
        raw = getattr(description, "description", "")
    return _target_signature_from_description(str(raw or "")) == expected_target_signature


def _collection_matches_index_signature(
    client: Any,
    collection_name: str,
    spec: IndexSpec,
    *,
    vector_field_names: Sequence[str],
) -> bool:
    details = _describe_collection_indexes(client, collection_name)
    for field_name in vector_field_names:
        if not _has_matching_vector_index(details, field_name=field_name, spec=spec):
            return False
    return True


def _has_matching_vector_index(
    index_details: Sequence[dict[str, Any]],
    *,
    field_name: str,
    spec: IndexSpec,
) -> bool:
    for detail in index_details:
        if str(detail.get("field_name") or "").strip() != field_name:
            continue
        detail_type = _normalize_milvus_index_type(detail.get("index_type"))
        if detail_type != spec.index_type:
            continue
        detail_metric = str(detail.get("metric_type") or "").strip().upper()
        if detail_metric and detail_metric != spec.metric_type:
            continue
        detail_params = detail.get("params")
        if not isinstance(detail_params, dict):
            detail_params = detail
        if all(str(detail_params.get(key)) == str(value) for key, value in spec.build_params.items()):
            return True
    return False


def _build_index_spec(
    *,
    index_type: str,
    metric_type: str,
    ivfpq_nlist: int,
    ivfpq_m: int,
    ivfpq_nbits: int,
    ivfpq_nprobe: int,
) -> IndexSpec:
    normalized_type = str(index_type or "IVFPQ").strip().upper()
    normalized_metric = str(metric_type or "COSINE").strip().upper()
    if normalized_type not in SUPPORTED_INDEX_TYPES:
        raise ValueError(f"unsupported index type: {normalized_type}")
    build_params = {
        "nlist": int(ivfpq_nlist),
        "m": int(ivfpq_m),
        "nbits": int(ivfpq_nbits),
    }
    search_params = {"nprobe": int(ivfpq_nprobe)}
    signature = (
        f"IVFPQ|{normalized_metric}|NLIST={int(ivfpq_nlist)}|M={int(ivfpq_m)}|"
        f"NBITS={int(ivfpq_nbits)}"
    )
    suffix = (
        f"ivfpq_nlist{int(ivfpq_nlist)}_m{int(ivfpq_m)}_"
        f"nbits{int(ivfpq_nbits)}_{normalized_metric.lower()}"
    )
    return IndexSpec(
        # The Stack keeps the user-facing name IVFPQ, while Milvus expects
        # IVF_PQ in the create-index request and index metadata.
        index_type=_normalize_milvus_index_type(normalized_type),
        metric_type=normalized_metric,
        build_params=build_params,
        search_params=search_params,
        signature=signature,
        suffix=suffix,
    )


def _normalize_milvus_index_type(index_type: Any) -> str:
    normalized = str(index_type or "").strip().upper()
    return INDEX_TYPE_BY_CONFIG.get(normalized, normalized)


def _describe_collection_indexes(client: Any, collection_name: str) -> list[dict[str, Any]]:
    if not hasattr(client, "list_indexes"):
        return []
    try:
        index_names = list(client.list_indexes(collection_name) or [])
    except Exception:
        return []
    details: list[dict[str, Any]] = []
    for index_name in index_names:
        if not hasattr(client, "describe_index"):
            details.append({"index_name": index_name})
            continue
        try:
            detail = client.describe_index(collection_name=collection_name, index_name=index_name)
        except TypeError:
            try:
                detail = client.describe_index(collection_name, index_name)
            except Exception:
                detail = {"index_name": index_name}
        except Exception:
            detail = {"index_name": index_name}
        if isinstance(detail, dict):
            detail.setdefault("index_name", index_name)
            details.append(detail)
        else:
            details.append({"index_name": index_name, "detail": detail})
    return details
