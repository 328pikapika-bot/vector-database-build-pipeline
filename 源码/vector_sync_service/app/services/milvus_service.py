from datetime import datetime
from dataclasses import replace
import inspect
import time
from typing import Any, List, Optional

from pymilvus import DataType, MilvusClient

from app.config import settings
from app.contracts.docx_contract import (
    DOCUMENT_SCALAR_FIELD_SPECS,
    DOCUMENT_SCALAR_INDEX_FIELDS,
    DOCUMENT_VECTOR_FIELDS,
    LEGACY_DOCUMENT_SCHEMA_FIELDS,
    DocumentScalarFieldSpec,
    normalize_document_id,
)
from app.observability.logger import get_logger
from app.observability.metrics import milvus_ops_total
from app.services.collection_targets import (
    CollectionTarget,
    build_document_collection_signature,
    build_document_index_spec,
    collection_signature_description,
    resolve_document_collection_target,
)

logger = get_logger(__name__)


class MilvusDynamicService:
    def __init__(self, uri: str, user: str = "", password: str = ""):
        self.client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            timeout=settings.MILVUS_OPERATION_TIMEOUT_SEC,
        )
        self._target_cache: dict[tuple[str, int, str], CollectionTarget] = {}
        self._description_cache: dict[str, Any] = {}

    def init_collection(self, collection_name: str, vector_dim: int = 2560) -> dict[str, Any]:
        target = self.resolve_collection_target(vector_dim=vector_dim, logical_name=collection_name)
        self.client.load_collection(target.resolved_name)
        payload = target.to_dict()
        payload["collection_name"] = target.resolved_name
        return payload

    def resolve_collection_target(
        self,
        *,
        vector_dim: int = 2560,
        logical_name: Optional[str] = None,
    ) -> CollectionTarget:
        resolved_logical_name = str(logical_name or settings.resolved_document_collection_name).strip()
        index_signature = build_document_index_spec().signature
        cache_key = (resolved_logical_name, int(vector_dim), index_signature)
        target_cache = getattr(self, "_target_cache", None)
        if target_cache is None:
            target_cache = {}
            self._target_cache = target_cache
        cached_target = target_cache.get(cache_key)
        if cached_target is not None:
            # Creation is a one-time event. Subsequent calls reuse the cached
            # physical collection and must not report it as newly created.
            return replace(cached_target, created=False, reused=True)

        target = resolve_document_collection_target(
            self.client,
            logical_name=resolved_logical_name,
            vector_dim=vector_dim,
            ensure_collection=lambda name: self._ensure_document_collection_created(
                name,
                vector_dim=vector_dim,
            ),
        )
        target_cache[cache_key] = target
        return target

    def process_and_insert(
        self,
        collection_name: str,
        articles: List[Any],
        vector_dim: int = 2560,
    ):
        if not articles:
            return

        target = self.resolve_collection_target(vector_dim=vector_dim, logical_name=collection_name)
        collection_name = target.resolved_name

        if not self.client.has_collection(collection_name):
            logger.error("Collection %s does not exist!", collection_name)
            raise ValueError(f"Collection {collection_name} does not exist!")

        description_cache = getattr(self, "_description_cache", None)
        if description_cache is None:
            description_cache = {}
            self._description_cache = description_cache
        describe = description_cache.get(collection_name)
        if describe is None:
            describe = self.client.describe_collection(collection_name)
            description_cache[collection_name] = describe
        self._validate_collection_description(collection_name, describe)
        existing_fields = {_field_name(f) for f in _description_fields(describe) if _field_name(f)}
        vector_field_names = [
            _field_name(f)
            for f in _description_fields(describe)
            if _field_name(f) and _field_type(f) in {DataType.FLOAT_VECTOR, getattr(DataType, "FLOAT16_VECTOR", None)}
        ]

        field_type_map = {_field_name(f): _field_type(f) for f in _description_fields(describe) if _field_name(f)}
        field_meta = {_field_name(f): _field_meta(f) for f in _description_fields(describe) if _field_name(f)}
        insert_data = []
        for article in articles:
            data_dict = _article_to_dict(article)
            document_id = normalize_document_id(article) or normalize_document_id(data_dict)
            if not document_id:
                raise ValueError("Missing required document ID")

            data_dict["id"] = document_id
            for legacy_field in LEGACY_DOCUMENT_SCHEMA_FIELDS:
                data_dict.pop(legacy_field, None)

            data_dict = {k: v for k, v in data_dict.items() if k in existing_fields}
            for field_key, val in list(data_dict.items()):
                data_dict[field_key] = _normalize_value_for_field(
                    field_key,
                    val,
                    field_type_map.get(field_key),
                    field_meta.get(field_key) or {},
                )

            for vf_name in vector_field_names:
                vector = data_dict.get(vf_name)
                if vector is None:
                    raise ValueError(f"Missing required vector field: {vf_name}")
                if not isinstance(vector, list):
                    raise ValueError(f"Vector field {vf_name} must be a list")
                if len(vector) != vector_dim:
                    raise ValueError(
                        f"Vector field {vf_name} dimension mismatch: expected={vector_dim}, actual={len(vector)}"
                    )

            insert_data.append(data_dict)

        logger.info("Upserting %d records into %s...", len(insert_data), collection_name)
        res = _call_milvus_with_timeout(
            self.client.upsert,
            timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
            collection_name=collection_name,
            data=insert_data,
        )
        milvus_ops_total.labels(op="upsert", status="success").inc(len(insert_data))
        if settings.DOCUMENT_VISIBILITY_CHECK_ENABLED:
            sample_ids = [str(row["id"]) for row in insert_data[: settings.DOCUMENT_VISIBILITY_CHECK_LIMIT]]
            visibility = self.wait_for_visible_document_ids(
                collection_name,
                sample_ids,
                retries=settings.DOCUMENT_VISIBILITY_RETRIES,
                retry_delay_sec=settings.DOCUMENT_VISIBILITY_RETRY_DELAY_SEC,
                timeout=settings.DOCUMENT_VISIBILITY_CHECK_TIMEOUT_SEC,
                vector_dim=vector_dim,
            )
            if not visibility["success"] and settings.DOCUMENT_VISIBILITY_STRICT:
                raise RuntimeError(
                    "document write visibility check failed: "
                    f"visible={visibility['visible_count']}, expected={visibility['expected_count']}"
                )
        return res

    def count_visible_document_ids(
        self,
        collection_name: str,
        document_ids: list[str],
        *,
        timeout: Optional[float] = None,
        vector_dim: int = settings.VECTOR_DIM,
    ) -> int:
        normalized = [str(item).strip() for item in document_ids if str(item or "").strip()]
        if not normalized:
            return 0
        collection_name = self.resolve_collection_target(
            logical_name=collection_name,
            vector_dim=vector_dim,
        ).resolved_name
        escaped = ",".join(f'"{_escape_milvus_string(item)}"' for item in normalized)
        rows = _call_milvus_with_timeout(
            self.client.query,
            timeout=(settings.MILVUS_QUERY_TIMEOUT_SEC if timeout is None else timeout),
            collection_name=collection_name,
            filter=f"id in [{escaped}]",
            output_fields=["id"],
            consistency_level="Eventually",
        )
        return len({str(row.get("id")) for row in rows or [] if row.get("id")})

    def wait_for_visible_document_ids(
        self,
        collection_name: str,
        document_ids: list[str],
        *,
        retries: int,
        retry_delay_sec: float,
        timeout: Optional[float] = None,
        vector_dim: int = settings.VECTOR_DIM,
    ) -> dict[str, Any]:
        expected = len({str(item) for item in document_ids if str(item or "").strip()})
        visible = 0
        attempts = 0
        for attempts in range(1, max(int(retries), 0) + 2):
            visible = self.count_visible_document_ids(
                collection_name,
                document_ids,
                timeout=timeout,
                vector_dim=vector_dim,
            )
            if visible >= expected:
                break
            if attempts <= max(int(retries), 0):
                time.sleep(max(float(retry_delay_sec), 0.0))
        return {
            "success": visible >= expected,
            "visible_count": visible,
            "expected_count": expected,
            "attempts": attempts,
        }

    def _query_document_by_id(self, collection_name: str, sys_id: str) -> Optional[dict]:
        """Query a full document from Milvus by primary key. Returns None if not found."""
        escaped = _escape_milvus_string(str(sys_id))
        results = _call_milvus_with_timeout(
            self.client.query,
            timeout=settings.MILVUS_QUERY_TIMEOUT_SEC,
            collection_name=collection_name,
            filter=f'id == "{escaped}"',
            output_fields=["*"],
            limit=1,
        )
        return results[0] if results else None

    def update_sync_status(self, collection_name: str, sys_id: str, sync_status: str) -> bool:
        """Update the sync_status scalar field for a document in Milvus.

        Because Milvus upsert replaces the entire row, we must first query the
        existing document, modify sync_status, then upsert the full row back.
        """
        if not sys_id:
            return False
        if not self.client.has_collection(collection_name):
            logger.warning("Collection %s does not exist, cannot update sync_status", collection_name)
            return False
        if not self._collection_has_field(collection_name, "sync_status"):
            logger.debug("Collection %s has no sync_status field, skipping sync_status update", collection_name)
            return False

        try:
            existing = self._query_document_by_id(collection_name, sys_id)
            if existing is None:
                logger.debug("Document %s not found in %s, skipping sync_status update", sys_id, collection_name)
                return False
            existing["sync_status"] = sync_status
            _call_milvus_with_timeout(
                self.client.upsert,
                timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
                collection_name=collection_name,
                data=[existing],
            )
            logger.debug("Updated sync_status=%s for doc %s in %s", sync_status, sys_id, collection_name)
            return True
        except Exception:
            logger.warning("Failed to update sync_status for doc %s", sys_id, exc_info=True)
            return False

    def update_sync_status_batch(
        self, collection_name: str, updates: list[tuple[str, str]]
    ) -> dict[str, int]:
        """Batch update sync_status for multiple documents.

        Args:
            collection_name: Milvus collection name.
            updates: List of (sys_id, sync_status) tuples.

        Returns:
            dict with "updated" and "skipped" counts.
        """
        result = {"updated": 0, "skipped": 0}
        if not updates or not self.client.has_collection(collection_name):
            return result
        if not self._collection_has_field(collection_name, "sync_status"):
            return {"updated": 0, "skipped": len(updates)}

        ids = [str(uid) for uid, _ in updates]
        status_map = {str(uid): status for uid, status in updates}
        escaped_ids = ", ".join(f'"{_escape_milvus_string(uid)}"' for uid in ids)
        try:
            existing_docs = _call_milvus_with_timeout(
                self.client.query,
                timeout=settings.MILVUS_QUERY_TIMEOUT_SEC,
                collection_name=collection_name,
                filter=f"id in [{escaped_ids}]",
                output_fields=["*"],
                limit=len(ids),
            )
        except Exception:
            logger.warning("Failed to query batch documents for sync_status update", exc_info=True)
            return result

        upsert_data: list[dict] = []
        for doc in existing_docs:
            doc_id = str(doc.get("id", ""))
            if doc_id in status_map:
                doc["sync_status"] = status_map[doc_id]
                upsert_data.append(doc)
                result["updated"] += 1

        result["skipped"] = len(ids) - len(existing_docs)
        if upsert_data:
            try:
                _call_milvus_with_timeout(
                    self.client.upsert,
                    timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
                    collection_name=collection_name,
                    data=upsert_data,
                )
            except Exception:
                logger.warning("Failed to batch upsert sync_status", exc_info=True)
                return {"updated": 0, "skipped": len(ids)}
        return result

    def delete_by_sys_id(self, collection_name: str, sys_id: str) -> dict:
        collection_name = self.resolve_collection_target(logical_name=collection_name).resolved_name
        if not sys_id:
            raise ValueError("sys_id is required for delete")
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")

        escaped = _escape_milvus_string(str(sys_id))
        filter_expr = f'id == "{escaped}"'
        logger.info("Deleting records from %s with filter=%s", collection_name, filter_expr)
        result = _call_milvus_with_timeout(
            self.client.delete,
            timeout=settings.MILVUS_DELETE_TIMEOUT_SEC,
            collection_name=collection_name,
            filter=filter_expr,
        )
        milvus_ops_total.labels(op="delete", status="success").inc(1)
        return result

    def delete_by_sys_ids(self, collection_name: str, sys_ids: list[str]) -> dict:
        collection_name = self.resolve_collection_target(logical_name=collection_name).resolved_name
        normalized_ids = [str(sys_id) for sys_id in sys_ids if str(sys_id).strip()]
        if not normalized_ids:
            raise ValueError("at least one sys_id is required for batch delete")
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")

        escaped_ids = ", ".join(f'"{_escape_milvus_string(sys_id)}"' for sys_id in normalized_ids)
        filter_expr = f"id in [{escaped_ids}]"
        logger.info(
            "Deleting %d records from %s with filter=%s",
            len(normalized_ids),
            collection_name,
            filter_expr,
        )
        result = _call_milvus_with_timeout(
            self.client.delete,
            timeout=settings.MILVUS_DELETE_TIMEOUT_SEC,
            collection_name=collection_name,
            filter=filter_expr,
        )
        milvus_ops_total.labels(op="delete", status="success").inc(len(normalized_ids))
        return result

    def drop_collection_if_exists(self, collection_name: str) -> bool:
        if not self.client.has_collection(collection_name):
            return False
        self.client.drop_collection(collection_name)
        self._description_cache.pop(collection_name, None)
        self._target_cache = {
            key: target
            for key, target in getattr(self, "_target_cache", {}).items()
            if target.resolved_name != collection_name
        }
        logger.info("Collection %s dropped", collection_name)
        return True

    def get_row_count(self, collection_name: str) -> int:
        if not self.client.has_collection(collection_name):
            return 0
        _call_milvus_with_timeout(
            self.client.flush,
            timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
            collection_name=collection_name,
        )
        stats = self.client.get_collection_stats(collection_name=collection_name)
        row_count = stats.get("row_count", 0)
        return int(row_count)

    def sample_sys_ids(self, collection_name: str, limit: int = 5) -> list[str]:
        collection_name = self.resolve_collection_target(logical_name=collection_name).resolved_name
        if not self.client.has_collection(collection_name):
            return []
        results = _call_milvus_with_timeout(
            self.client.query,
            timeout=settings.MILVUS_QUERY_TIMEOUT_SEC,
            collection_name=collection_name,
            filter="",
            output_fields=["id"],
            limit=limit,
        )
        return [str(item.get("id")) for item in results if item.get("id")]

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Milvus client closed")

    def _validate_existing_collection_schema(self, collection_name: str) -> None:
        if not hasattr(self.client, "describe_collection"):
            return
        describe = self.client.describe_collection(collection_name)
        self._validate_collection_description(collection_name, describe)

    def _ensure_document_collection_created(self, collection_name: str, *, vector_dim: int) -> None:
        field_count = 2 + len(DOCUMENT_SCALAR_FIELD_SPECS) + len(DOCUMENT_VECTOR_FIELDS)
        _validate_proxy_field_capacity(field_count)

        logger.info("Creating collection %s with document scalar schema", collection_name)
        vector_spec = build_document_index_spec()
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=collection_signature_description(
                build_document_collection_signature(vector_dim=vector_dim),
                vector_spec.signature,
            ),
        )
        schema.add_field("id", DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256)
        schema.add_field("cls", DataType.VARCHAR, max_length=256, nullable=True)

        field_names = {"id", "cls"}
        for spec in DOCUMENT_SCALAR_FIELD_SPECS:
            _add_document_scalar_field(schema, spec)
            field_names.add(spec.name)

        for vf_name in DOCUMENT_VECTOR_FIELDS:
            schema.add_field(vf_name, DataType.FLOAT_VECTOR, dim=vector_dim)
            field_names.add(vf_name)

        index_params = self.client.prepare_index_params()
        for vf_name in DOCUMENT_VECTOR_FIELDS:
            index_params.add_index(
                field_name=vf_name,
                index_type=vector_spec.index_type,
                metric_type=vector_spec.metric_type,
                params=dict(vector_spec.build_params),
            )
        for field_name in DOCUMENT_SCALAR_INDEX_FIELDS:
            if field_name in field_names:
                index_params.add_index(field_name=field_name, index_type="AUTOINDEX")

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Collection %s created, indexed", collection_name)

    def _collection_has_field(self, collection_name: str, field_name: str) -> bool:
        if not hasattr(self.client, "describe_collection"):
            return False
        describe = self.client.describe_collection(collection_name)
        field_names = {_field_name(field) for field in _description_fields(describe) if _field_name(field)}
        return field_name in field_names

    @staticmethod
    def _validate_collection_description(collection_name: str, describe: Any) -> None:
        fields = _description_fields(describe)
        field_names = {_field_name(field) for field in fields if _field_name(field)}
        legacy_fields = sorted(field_names & LEGACY_DOCUMENT_SCHEMA_FIELDS)
        if legacy_fields:
            raise ValueError(
                f"Collection {collection_name} uses legacy document schema fields {legacy_fields}; "
                "use a new collection name or rebuild it with the V2 id scalar schema."
            )
        if "id" not in field_names:
            raise ValueError(f"Collection {collection_name} is missing document primary field id")
        id_field = next((field for field in fields if _field_name(field) == "id"), None)
        if id_field is not None and not _is_primary_field(id_field):
            raise ValueError(f"Collection {collection_name} field id must be the primary key")

        required_fields = {"id", "cls", *DOCUMENT_VECTOR_FIELDS, *(spec.name for spec in DOCUMENT_SCALAR_FIELD_SPECS)}
        if "sync_status" not in field_names:
            required_fields.discard("sync_status")
        missing = sorted(required_fields - field_names)
        if missing:
            preview = ", ".join(missing[:20])
            suffix = "" if len(missing) <= 20 else f", ... (+{len(missing) - 20})"
            raise ValueError(f"Collection {collection_name} is missing document schema fields: {preview}{suffix}")


def _add_document_scalar_field(schema: Any, spec: DocumentScalarFieldSpec) -> None:
    data_type = _data_type_for_kind(spec.kind)
    if spec.is_array:
        kwargs: dict[str, Any] = {
            "element_type": data_type,
            "max_capacity": spec.max_capacity,
            "nullable": True,
        }
        if data_type == DataType.VARCHAR:
            kwargs["max_length"] = spec.max_length
        schema.add_field(spec.name, DataType.ARRAY, **kwargs)
        return

    kwargs = {"nullable": True}
    if data_type == DataType.VARCHAR:
        kwargs["max_length"] = spec.max_length
    schema.add_field(spec.name, data_type, **kwargs)


def _data_type_for_kind(kind: str) -> Any:
    if kind == "int64":
        return DataType.INT64
    if kind == "float":
        return DataType.FLOAT
    return DataType.VARCHAR


def _article_to_dict(article: Any) -> dict[str, Any]:
    if hasattr(article, "model_dump"):
        return article.model_dump(exclude={"id", "sysId", "dynamic_fields"}, exclude_none=True)
    if isinstance(article, dict):
        return dict(article)
    result: dict[str, Any] = {}
    for field_name in ("id", "ID", "cls", *DOCUMENT_VECTOR_FIELDS, *(spec.name for spec in DOCUMENT_SCALAR_FIELD_SPECS)):
        if hasattr(article, field_name):
            value = getattr(article, field_name)
            if value is not None:
                result[field_name] = value
    return result


def _normalize_value_for_field(field_key: str, value: Any, data_type: Any, meta: dict[str, Any]) -> Any:
    if isinstance(value, datetime):
        value = value.strftime("%Y-%m-%dT%H:%M:%SZ")

    if data_type == DataType.ARRAY:
        values = value if isinstance(value, list) else [value]
        element_type = meta.get("element_type")
        if element_type == DataType.VARCHAR:
            return _normalize_varchar_array(field_key, values, int(meta.get("max_length") or 65535))
        if element_type == DataType.INT64:
            return [_coerce_int(item) for item in values if item is not None]
        if element_type == DataType.FLOAT:
            return [_coerce_float(item) for item in values if item is not None]
        return values

    if data_type == DataType.VARCHAR:
        if isinstance(value, list):
            value = next((item for item in value if item not in (None, "")), None)
        return str(value) if value is not None else None
    if data_type == DataType.INT64:
        return _coerce_int(value)
    if data_type == DataType.FLOAT:
        return _coerce_float(value)
    return value


def _normalize_varchar_array(field_key: str, values: list[Any], max_length: int) -> list[str]:
    normalized: list[str] = []
    for item in values:
        if item is None:
            continue
        text = item.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(item, datetime) else str(item)
        if not text:
            continue
        encoded = text.encode("utf-8")
        if len(encoded) <= max_length:
            normalized.append(text)
            continue
        chunks = _chunk_utf8_text(text, max_length)
        normalized.extend(chunks)
        logger.warning(
            "array_field_chunked",
            field_name=field_key,
            original_bytes=len(encoded),
            chunks=len(chunks),
        )
    return normalized


def _chunk_utf8_text(text: str, max_bytes: int) -> list[str]:
    """Split UTF-8 text without cutting a multibyte character."""
    encoded = text.encode("utf-8")
    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        while True:
            try:
                chunks.append(encoded[start:end].decode("utf-8"))
                start = end
                break
            except UnicodeDecodeError as error:
                end = start + error.start
                if end <= start:
                    raise ValueError("max_bytes is too small for one UTF-8 character") from error
    return chunks


def _coerce_int(value: Any) -> Any:
    if value is None or isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _coerce_float(value: Any) -> Any:
    if value is None or isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _description_fields(describe: Any) -> list[Any]:
    if isinstance(describe, dict):
        return list(describe.get("fields") or [])
    return list(getattr(describe, "fields", []) or [])


def _field_name(field: Any) -> Optional[str]:
    if isinstance(field, dict):
        return field.get("name")
    return getattr(field, "name", None)


def _field_type(field: Any) -> Any:
    if isinstance(field, dict):
        return field.get("type") or field.get("data_type")
    return getattr(field, "type", None) or getattr(field, "data_type", None)


def _field_meta(field: Any) -> dict[str, Any]:
    meta = {"type": _field_type(field)}
    if isinstance(field, dict):
        params = field.get("params") or field.get("type_params") or {}
        meta["element_type"] = field.get("element_type")
        meta["max_length"] = field.get("max_length") or params.get("max_length")
        return meta
    meta["element_type"] = getattr(field, "element_type", None)
    params = getattr(field, "params", None) or getattr(field, "type_params", None) or {}
    meta["max_length"] = getattr(field, "max_length", None) or params.get("max_length")
    return meta


def _is_primary_field(field: Any) -> bool:
    if isinstance(field, dict):
        return bool(field.get("is_primary") or field.get("primary_key"))
    return bool(getattr(field, "is_primary", False) or getattr(field, "primary_key", False))


def _validate_proxy_field_capacity(field_count: int) -> None:
    configured_limit = int(getattr(settings, "MILVUS_PROXY_MAX_FIELD_NUM", 512) or 0)
    if configured_limit and field_count > configured_limit:
        raise RuntimeError(
            "Document Milvus schema requires "
            f"{field_count} fields, but MILVUS_PROXY_MAX_FIELD_NUM={configured_limit}. "
            "Set Milvus proxy.maxFieldNum to at least 512 and keep this config in sync."
        )


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _call_milvus_with_timeout(method: Any, *, timeout: float, **kwargs: Any) -> Any:
    """Pass an RPC deadline without breaking older clients or test doubles."""
    try:
        parameters = inspect.signature(method).parameters.values()
        supports_timeout = any(
            parameter.name == "timeout"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_timeout = True
    if supports_timeout:
        kwargs["timeout"] = timeout
    return method(**kwargs)


_milvus_instance: Optional[MilvusDynamicService] = None


def get_milvus_service() -> MilvusDynamicService:
    global _milvus_instance
    if _milvus_instance is None:
        _milvus_instance = MilvusDynamicService(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            user=settings.MILVUS_USER or "",
            password=settings.MILVUS_PASSWORD or "",
        )
    return _milvus_instance


def close_milvus_service():
    global _milvus_instance
    if _milvus_instance:
        _milvus_instance.close()
        _milvus_instance = None
