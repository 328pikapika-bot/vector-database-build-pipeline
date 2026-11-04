"""Read-only training vector library probe."""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from app.config import settings
from app.services.collection_targets import build_search_params_for_collection_label
from app.services.image_index_quality import image_search_quality_filter


DEFAULT_TRAINING_VECTOR_QUERY = "rheumatoid arthritis T cell receptor"
LOW_SCORE_WARNING_THRESHOLD = 0.05
PREVIEW_TEXT_CHARS = 240

DOCUMENT_SEARCHES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("document_title", "title_embed", ("id", "title", "abstract", "keyword", "year", "language")),
    ("document_abstract", "abstract_embed", ("id", "title", "abstract", "keyword", "year", "language")),
    ("document_keyword", "keyword_embed", ("id", "title", "abstract", "keyword", "year", "language")),
)
CHUNK_SEARCHES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "chunk_text",
        "text_embed",
        ("id", "doc_id", "text", "chunkIndex", "sectionPath", "language"),
    ),
)
IMAGE_TEXT_SEARCHES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("image_caption", "caption_embed", ("id", "doc_id", "img_meta")),
    ("image_context", "context_embed", ("id", "doc_id", "context_before", "context_after")),
)
IMAGE_SELF_SEARCHES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("image_visual", "visual_embed", "IP", ("id", "doc_id", "raw_img_meta", "img_meta")),
    ("image_phash", "p_hash", "HAMMING", ("id", "doc_id", "raw_img_meta", "img_meta")),
)
IMAGE_RESERVED_FIELDS: tuple[str, ...] = (
    "text",
    "text_embed",
    "visual_text_embed",
    "text_sparse_vec",
)

REQUIRED_COLLECTION_FIELDS: dict[str, set[str]] = {
    "document": {"id", "title_embed", "abstract_embed", "keyword_embed"},
    "chunk": {"id", "doc_id", "text_embed"},
    "image": {
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
        },
}


async def probe_training_vector_library(
    *,
    query: str = DEFAULT_TRAINING_VECTOR_QUERY,
    top_k: int = 3,
    milvus_client: Optional[Any] = None,
    embedder: Optional[Any] = None,
    document_collection: str = settings.resolved_document_collection_name,
    chunk_collection: str = settings.resolved_fulltext_chunk_collection_name,
    image_collection: str = settings.IMAGE_COLLECTION_NAME,
    vector_dim: int = settings.VECTOR_DIM,
) -> dict[str, Any]:
    """Probe the three retrieval collections without mutating Milvus."""
    normalized_query = str(query or "").strip() or DEFAULT_TRAINING_VECTOR_QUERY
    normalized_top_k = max(1, int(top_k or 1))
    result: dict[str, Any] = {
        "success": False,
        "read_only": True,
        "mutation_allowed": False,
        "query": normalized_query,
        "top_k": normalized_top_k,
        "milvus": {
            "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "collections": {
                "document": document_collection,
                "chunk": chunk_collection,
                "image": image_collection,
            },
        },
        "embedding": {
            "success": False,
            "model": settings.EMBEDDING_MODEL,
            "expected_dim": vector_dim,
            "dim": None,
        },
        "collections": {},
        "warnings": [],
        "errors": [],
    }

    owned_resources: list[Any] = []
    try:
        client = _ensure_milvus_client(milvus_client, owned_resources)
        embedder_client = _ensure_embedder(embedder, owned_resources)

        collection_specs = {
            "document": document_collection,
            "chunk": chunk_collection,
            "image": image_collection,
        }
        for label, collection_name in collection_specs.items():
            result["collections"][label] = _inspect_collection(
                client,
                label=label,
                collection_name=collection_name,
            )

        embedding_vector = await _embed_query(embedder_client, normalized_query, vector_dim, result)

        if embedding_vector is not None:
            _probe_document_collection(
                client,
                result["collections"]["document"],
                embedding_vector,
                top_k=normalized_top_k,
            )
            _probe_chunk_collection(
                client,
                result["collections"]["chunk"],
                embedding_vector,
                top_k=normalized_top_k,
            )
            _probe_image_text_vectors(
                client,
                result["collections"]["image"],
                embedding_vector,
                top_k=normalized_top_k,
            )

        _probe_image_self_vectors(
            client,
            result["collections"]["image"],
            top_k=normalized_top_k,
        )

        _roll_up_collection_status(result)
        result["success"] = (
            bool(result["embedding"]["success"])
            and all(details.get("ok") is True for details in result["collections"].values())
            and not result["errors"]
        )
        return result
    finally:
        await _close_owned_resources(owned_resources)


def _ensure_milvus_client(milvus_client: Optional[Any], owned_resources: list[Any]) -> Any:
    if milvus_client is not None:
        return milvus_client
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        user=settings.MILVUS_USER or "",
        password=settings.MILVUS_PASSWORD or "",
    )
    owned_resources.append(client)
    return client


def _ensure_embedder(embedder: Optional[Any], owned_resources: list[Any]) -> Any:
    if embedder is not None:
        return embedder
    from app.services.embedder import EmbeddingClient

    client = EmbeddingClient()
    owned_resources.append(client)
    return client


async def _embed_query(embedder: Any, query: str, vector_dim: int, result: dict[str, Any]) -> Optional[list[float]]:
    try:
        embedding_result = await _maybe_await(embedder.embed_single_detailed(query))
        success = bool(_get_value(embedding_result, "success"))
        vector = _get_value(embedding_result, "vector")
        dim = len(vector) if isinstance(vector, Sequence) else None
        result["embedding"].update(
            {
                "success": success,
                "dim": dim,
                "error_code": _get_value(embedding_result, "error_code"),
                "retryable": bool(_get_value(embedding_result, "retryable")),
            }
        )
        if not success or not isinstance(vector, list):
            _add_error(result, "EMBEDDING_FAILED", "Embedding query did not return a usable vector.")
            return None
        if len(vector) != vector_dim:
            _add_error(
                result,
                "EMBEDDING_DIMENSION_MISMATCH",
                f"Embedding dimension mismatch: expected={vector_dim}, actual={len(vector)}.",
            )
            return None
        return vector
    except Exception as error:
        result["embedding"].update(
            {
                "success": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _add_error(result, "EMBEDDING_EXCEPTION", f"{type(error).__name__}: {error}")
        return None


def _inspect_collection(client: Any, *, label: str, collection_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "collection_name": collection_name,
        "exists": False,
        "row_count": None,
        "primary_fields": [],
        "field_count": 0,
        "vector_fields": {},
        "missing_required_fields": [],
        "searches": [],
        "warnings": [],
        "errors": [],
        "ok": False,
    }
    try:
        exists = bool(client.has_collection(collection_name))
        summary["exists"] = exists
        if not exists:
            _add_collection_error(summary, "COLLECTION_MISSING", f"Collection does not exist: {collection_name}")
            return summary

        describe = client.describe_collection(collection_name)
        fields = _description_fields(describe)
        field_names = {_field_name(field) for field in fields if _field_name(field)}
        summary["field_count"] = len(field_names)
        summary["primary_fields"] = sorted(
            _field_name(field)
            for field in fields
            if _field_name(field) and _is_primary_field(field)
        )
        summary["vector_fields"] = {
            _field_name(field): {
                "type": _field_type_name(field),
                "dim": _field_dim(field),
            }
            for field in fields
            if _field_name(field) and "VECTOR" in _field_type_name(field)
        }

        missing = sorted(REQUIRED_COLLECTION_FIELDS[label] - field_names)
        summary["missing_required_fields"] = missing
        if label == "image":
            summary["reserved_fields"] = {
                field_name: {
                    "present": field_name in field_names,
                    "status": "reserved_not_probed",
                }
                for field_name in IMAGE_RESERVED_FIELDS
            }
        for field_name in missing:
            _add_collection_error(
                summary,
                "REQUIRED_FIELD_MISSING",
                f"{collection_name} is missing required field {field_name}.",
                field_name=field_name,
            )

        try:
            stats = client.get_collection_stats(collection_name=collection_name)
            summary["row_count"] = int(stats.get("row_count", 0))
        except Exception as error:
            _add_collection_warning(
                summary,
                "COLLECTION_STATS_UNAVAILABLE",
                f"Could not read stats for {collection_name}: {type(error).__name__}: {error}",
            )

        if summary["row_count"] == 0:
            _add_collection_error(summary, "COLLECTION_EMPTY", f"Collection is empty: {collection_name}")

        return summary
    except Exception as error:
        _add_collection_error(
            summary,
            "COLLECTION_INSPECTION_FAILED",
            f"{type(error).__name__}: {error}",
        )
        return summary


def _probe_document_collection(
    client: Any,
    collection: dict[str, Any],
    query_vector: list[float],
    *,
    top_k: int,
) -> None:
    if not _collection_ready_for_search(collection):
        return
    for name, field_name, output_fields in DOCUMENT_SEARCHES:
        _append_text_search(
            client,
            collection,
            probe_name=name,
            vector_field=field_name,
            query_vector=query_vector,
            collection_label="document",
            metric_type=settings.DOCUMENT_INDEX_METRIC_TYPE,
            output_fields=output_fields,
            top_k=top_k,
        )


def _probe_chunk_collection(
    client: Any,
    collection: dict[str, Any],
    query_vector: list[float],
    *,
    top_k: int,
) -> None:
    if not _collection_ready_for_search(collection):
        return
    for name, field_name, output_fields in CHUNK_SEARCHES:
        _append_text_search(
            client,
            collection,
            probe_name=name,
            vector_field=field_name,
            query_vector=query_vector,
            collection_label="chunk",
            metric_type=settings.FULLTEXT_CHUNK_INDEX_METRIC_TYPE,
            output_fields=output_fields,
            top_k=top_k,
        )


def _probe_image_text_vectors(
    client: Any,
    collection: dict[str, Any],
    query_vector: list[float],
    *,
    top_k: int,
) -> None:
    if not _collection_ready_for_search(collection):
        return
    image_query_vector = np.asarray(query_vector, dtype=np.float16)
    for name, field_name, output_fields in IMAGE_TEXT_SEARCHES:
        _append_text_search(
            client,
            collection,
            probe_name=name,
            vector_field=field_name,
            query_vector=image_query_vector,
            collection_label="image",
            metric_type="IP",
            output_fields=output_fields,
            top_k=top_k,
        )


def _probe_image_self_vectors(client: Any, collection: dict[str, Any], *, top_k: int) -> None:
    if not _collection_ready_for_search(collection):
        return
    for name, field_name, metric_type, output_fields in IMAGE_SELF_SEARCHES:
        _append_self_search(
            client,
            collection,
            probe_name=name,
            vector_field=field_name,
            id_field="id",
            collection_label="image",
            metric_type=metric_type,
            output_fields=output_fields,
            top_k=top_k,
        )


def _append_text_search(
    client: Any,
    collection: dict[str, Any],
    *,
    probe_name: str,
    vector_field: str,
    query_vector: Any,
    collection_label: str,
    metric_type: str,
    output_fields: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    search = _base_search_summary(probe_name, "text_query", vector_field, metric_type)
    collection["searches"].append(search)
    try:
        quality_filter = (
            image_search_quality_filter(vector_field)
            if collection_label == "image"
            else ""
        )
        search_kwargs = {
            "collection_name": collection["collection_name"],
            "data": [query_vector],
            "anns_field": vector_field,
            "limit": top_k,
            "search_params": _search_params(
                collection_label,
                metric_type,
                vector_field=vector_field,
            ),
            "output_fields": list(output_fields),
        }
        if quality_filter:
            search_kwargs["filter"] = quality_filter
        hits = client.search(
            **search_kwargs,
        )
        normalized_hits = _normalize_hits(hits, output_fields)
        search.update(
            {
                "ok": bool(normalized_hits),
                "hit_count": len(normalized_hits),
                "top_score": normalized_hits[0].get("score") if normalized_hits else None,
                "hits": normalized_hits,
            }
        )
        if not normalized_hits:
            _add_search_error(collection, search, "SEARCH_RETURNED_NO_HITS", "Search returned no hits.")
    except Exception as error:
        search.update(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _add_search_error(collection, search, "SEARCH_FAILED", f"{type(error).__name__}: {error}")
    return search


def _append_self_search(
    client: Any,
    collection: dict[str, Any],
    *,
    probe_name: str,
    vector_field: str,
    id_field: str,
    collection_label: str,
    metric_type: str,
    output_fields: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    search = _base_search_summary(probe_name, "self_vector_query", vector_field, metric_type)
    collection["searches"].append(search)
    try:
        sample_rows = client.query(
            collection_name=collection["collection_name"],
            filter="",
            output_fields=[id_field, vector_field],
            limit=1,
        )
        if not sample_rows:
            _add_search_error(collection, search, "SELF_SAMPLE_MISSING", "Could not read a sample vector.")
            return search

        sample = sample_rows[0]
        sample_id = str(sample.get(id_field) or "")
        vector_type = (collection.get("vector_fields") or {}).get(vector_field, {}).get("type")
        query_vector = _query_vector_from_sample(sample.get(vector_field), str(vector_type or ""))
        hits = client.search(
            collection_name=collection["collection_name"],
            data=[query_vector],
            anns_field=vector_field,
            limit=top_k,
            search_params=_search_params(
                collection_label,
                metric_type,
                vector_field=vector_field,
            ),
            output_fields=list(output_fields),
        )
        normalized_hits = _normalize_hits(hits, output_fields)
        top_id = _hit_identity(normalized_hits[0], id_field) if normalized_hits else None
        self_match = bool(sample_id and top_id == sample_id)
        search.update(
            {
                "ok": bool(normalized_hits) and self_match,
                "sample_id": sample_id,
                "self_match": self_match,
                "hit_count": len(normalized_hits),
                "top_score": normalized_hits[0].get("score") if normalized_hits else None,
                "hits": normalized_hits,
            }
        )
        if not normalized_hits:
            _add_search_error(collection, search, "SEARCH_RETURNED_NO_HITS", "Self-vector search returned no hits.")
        elif not self_match:
            _add_search_error(
                collection,
                search,
                "SELF_VECTOR_TOP_HIT_MISMATCH",
                f"Expected top {id_field}={sample_id}, got {top_id}.",
            )
    except Exception as error:
        search.update(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _add_search_error(collection, search, "SEARCH_FAILED", f"{type(error).__name__}: {error}")
    return search


def _query_vector_from_sample(value: Any, vector_type: str) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (bytes, bytearray)):
        return bytes(value[0])
    if "FLOAT16_VECTOR" in vector_type:
        if isinstance(value, np.ndarray):
            return value.astype(np.float16)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return np.asarray(value, dtype=np.float16)
    return value


def _base_search_summary(probe_name: str, probe_type: str, vector_field: str, metric_type: str) -> dict[str, Any]:
    return {
        "name": probe_name,
        "type": probe_type,
        "vector_field": vector_field,
        "metric_type": metric_type,
        "ok": False,
        "hit_count": 0,
        "top_score": None,
        "hits": [],
    }


def _search_params(
    collection_label: str,
    metric_type: str,
    *,
    vector_field: str = "",
) -> dict[str, Any]:
    if str(collection_label or "").strip().lower() == "image" and vector_field == "p_hash":
        return {"metric_type": metric_type, "params": {}}
    params = build_search_params_for_collection_label(collection_label)
    params["metric_type"] = metric_type
    return params


def _normalize_hits(hits: Any, output_fields: Sequence[str]) -> list[dict[str, Any]]:
    first_group = list(hits[0]) if hits else []
    normalized: list[dict[str, Any]] = []
    for rank, hit in enumerate(first_group, start=1):
        hit_dict = dict(hit) if isinstance(hit, Mapping) else _object_to_public_dict(hit)
        entity = hit_dict.get("entity")
        if entity is None:
            entity = hit_dict
        if not isinstance(entity, Mapping):
            entity = _object_to_public_dict(entity)
        score = hit_dict.get("distance", hit_dict.get("score"))
        normalized.append(
            {
                "rank": rank,
                "score": _json_safe_scalar(score),
                "entity": {
                    field: _preview_value(entity.get(field))
                    for field in output_fields
                    if field in entity
                },
            }
        )
    return normalized


def _object_to_public_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        return dict(data) if isinstance(data, Mapping) else {}
    public: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if not callable(item):
            public[name] = item
    return public


def _preview_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:PREVIEW_TEXT_CHARS]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<bytes>"
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, Mapping):
        return {
            str(key): _preview_value(item)
            for key, item in list(value.items())[:8]
        }
    if isinstance(value, list):
        return [_preview_value(item) for item in value[:4]]
    if isinstance(value, tuple):
        return tuple(_preview_value(item) for item in value[:4])
    return _json_safe_scalar(value)


def _warn_on_image_placeholder_hits(collection: dict[str, Any], search: dict[str, Any]) -> None:
    hits = search.get("hits") or []
    texts = [
        str((hit.get("entity") or {}).get("text") or "")
        for hit in hits
    ]
    if texts and all(text == "__reserved_image_text__" for text in texts):
        _add_collection_warning(
            collection,
            "IMAGE_TEXT_IS_RESERVED_PLACEHOLDER",
            f"{search['name']} returned only reserved image text placeholders.",
            probe_name=search["name"],
        )
    top_score = search.get("top_score")
    if isinstance(top_score, (int, float)) and abs(float(top_score)) < LOW_SCORE_WARNING_THRESHOLD:
        _add_collection_warning(
            collection,
            "IMAGE_TEXT_SCORE_NEAR_ZERO",
            f"{search['name']} top score is near zero: {top_score}.",
            probe_name=search["name"],
        )


def _collection_ready_for_search(collection: Mapping[str, Any]) -> bool:
    return bool(collection.get("exists")) and not collection.get("missing_required_fields")


def _roll_up_collection_status(result: dict[str, Any]) -> None:
    for label, collection in result["collections"].items():
        searches = collection.get("searches") or []
        required_search_count = {"document": 3, "chunk": 1, "image": 4}[label]
        searches_ok = len(searches) >= required_search_count and all(search.get("ok") for search in searches)
        collection["ok"] = (
            bool(collection.get("exists"))
            and not collection.get("errors")
            and collection.get("row_count") not in (0, "0")
            and searches_ok
        )
        for warning in collection.get("warnings") or []:
            result["warnings"].append({"collection": label, **warning})
        for error in collection.get("errors") or []:
            result["errors"].append({"collection": label, **error})


def _add_error(result: dict[str, Any], code: str, message: str, **extra: Any) -> None:
    result["errors"].append({"code": code, "message": message, **extra})


def _add_collection_error(collection: dict[str, Any], code: str, message: str, **extra: Any) -> None:
    collection["errors"].append({"code": code, "message": message, **extra})


def _add_collection_warning(collection: dict[str, Any], code: str, message: str, **extra: Any) -> None:
    collection["warnings"].append({"code": code, "message": message, **extra})


def _add_search_error(collection: dict[str, Any], search: dict[str, Any], code: str, message: str) -> None:
    search["error_code"] = code
    search["error_message"] = message
    _add_collection_error(
        collection,
        code,
        message,
        probe_name=search["name"],
        vector_field=search["vector_field"],
    )


def _description_fields(describe: Any) -> list[Any]:
    if isinstance(describe, Mapping):
        return list(describe.get("fields") or [])
    return list(getattr(describe, "fields", []) or [])


def _field_name(field: Any) -> Optional[str]:
    if isinstance(field, Mapping):
        return field.get("name")
    return getattr(field, "name", None)


def _field_type_name(field: Any) -> str:
    if isinstance(field, Mapping):
        field_type = field.get("type") or field.get("data_type")
    else:
        field_type = getattr(field, "type", None) or getattr(field, "data_type", None)
    return str(getattr(field_type, "name", field_type))


def _field_dim(field: Any) -> Any:
    if isinstance(field, Mapping):
        params = field.get("params")
        if isinstance(params, Mapping) and params.get("dim") is not None:
            return _json_safe_scalar(params.get("dim"))
        return _json_safe_scalar(field.get("dim"))
    return _json_safe_scalar(getattr(field, "dim", None))


def _is_primary_field(field: Any) -> bool:
    if isinstance(field, Mapping):
        return bool(field.get("is_primary") or field.get("primary_key"))
    return bool(getattr(field, "is_primary", False) or getattr(field, "primary_key", False))


def _hit_identity(hit: Mapping[str, Any], id_field: str) -> Optional[str]:
    entity = hit.get("entity") or {}
    value = entity.get(id_field) if isinstance(entity, Mapping) else None
    return str(value) if value is not None else None


def _get_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


async def _close_owned_resources(resources: Sequence[Any]) -> None:
    for resource in reversed(resources):
        close = getattr(resource, "close", None)
        if close is None:
            close = getattr(resource, "aclose", None)
        if close is None:
            continue
        await _maybe_await(close())


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _json_safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of probe output."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<bytes>"
    return _json_safe_scalar(value)
