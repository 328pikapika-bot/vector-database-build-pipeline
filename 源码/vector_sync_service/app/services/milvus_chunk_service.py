"""Independent Milvus service for DOCX-aligned full-text chunks."""

from __future__ import annotations

import time
import inspect
from typing import Any, Optional, Sequence

from pymilvus import DataType, MilvusClient

from app.config import settings
from app.contracts.docx_contract import FULLTEXT_FIELDS
from app.models.chunk import Chunk
from app.observability.logger import get_logger
from app.services.collection_targets import (
    CollectionTarget,
    build_fulltext_collection_signature,
    build_fulltext_chunk_index_spec,
    collection_signature_description,
    resolve_fulltext_collection_target,
)


logger = get_logger(__name__)

DEFAULT_CHUNK_COLLECTION = getattr(settings, "FULLTEXT_CHUNK_COLLECTION", "scientific_fulltext_chunks")
DEFAULT_PARSER_VERSION = "phase2_round1"
DEFAULT_CHUNKER_VERSION = "phase2_round1"
DEFAULT_SCHEMA_VERSION = settings.FULLTEXT_CHUNK_SCHEMA_VERSION
MAX_SECTION_PATH_LENGTH = 1024
CHUNK_SCALAR_INDEX_TYPE = "INVERTED"
CHUNK_SCALAR_INDEX_SPECS: tuple[dict[str, Any], ...] = (
    {
        "field_name": "doc_id",
        "index_name": "chunk_doc_id_inverted_idx",
        "index_type": CHUNK_SCALAR_INDEX_TYPE,
    },
    {
        "field_name": "accessType",
        "index_name": "chunk_access_type_inverted_idx",
        "index_type": CHUNK_SCALAR_INDEX_TYPE,
    },
    {
        "field_name": "classificationCode",
        "index_name": "chunk_classification_code_inverted_idx",
        "index_type": CHUNK_SCALAR_INDEX_TYPE,
    },
    {"field_name": "year", "index_name": "chunk_year_inverted_idx", "index_type": CHUNK_SCALAR_INDEX_TYPE},
    {
        "field_name": "hasAuthorname",
        "index_name": "chunk_author_name_inverted_idx",
        "index_type": CHUNK_SCALAR_INDEX_TYPE,
    },
    {
        "field_name": "hasCreateOrganizationname",
        "index_name": "chunk_create_org_name_inverted_idx",
        "index_type": CHUNK_SCALAR_INDEX_TYPE,
    },
    {"field_name": "language", "index_name": "chunk_language_inverted_idx", "index_type": CHUNK_SCALAR_INDEX_TYPE},
)


class MilvusChunkService:
    """Milvus writer for the independent full-text chunk collection."""

    def __init__(
        self,
        uri: str = "",
        user: str = "",
        password: str = "",
        client: Optional[Any] = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = MilvusClient(
                uri=uri or f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                user=user or settings.MILVUS_USER or "",
                password=password or settings.MILVUS_PASSWORD or "",
                timeout=settings.MILVUS_OPERATION_TIMEOUT_SEC,
            )

    def init_chunk_collection(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION,
        vector_dim: int = settings.VECTOR_DIM,
    ) -> dict[str, Any]:
        target = self.resolve_collection_target(
            collection_name=collection_name,
            vector_dim=vector_dim,
        )
        self.client.load_collection(target.resolved_name)
        payload = target.to_dict()
        payload["collection_name"] = target.resolved_name
        return payload

    def resolve_collection_target(
        self,
        *,
        collection_name: str = DEFAULT_CHUNK_COLLECTION,
        vector_dim: int = settings.VECTOR_DIM,
    ) -> CollectionTarget:
        return resolve_fulltext_collection_target(
            self.client,
            logical_name=collection_name,
            vector_dim=vector_dim,
            ensure_collection=lambda name: self._ensure_chunk_collection_created(
                name,
                vector_dim=vector_dim,
            ),
        )

    def _ensure_chunk_collection_created(
        self,
        collection_name: str,
        *,
        vector_dim: int,
    ) -> None:
        vector_spec = build_fulltext_chunk_index_spec()
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=collection_signature_description(
                build_fulltext_collection_signature(vector_dim=vector_dim),
                vector_spec.signature,
            ),
        )
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128, nullable=False)
        _add_string_array_field(schema, "title", nullable=True)
        _add_string_array_field(schema, "keyword", nullable=True)
        schema.add_field("accessType", DataType.VARCHAR, max_length=128, nullable=True)
        schema.add_field("classificationCode", DataType.VARCHAR, max_length=128, nullable=True)
        schema.add_field("year", DataType.VARCHAR, max_length=64, nullable=True)
        _add_string_array_field(schema, "hasAuthorname", nullable=True)
        _add_string_array_field(schema, "hasCreateOrganizationname", nullable=True)
        _add_string_array_field(schema, "language", nullable=False)
        schema.add_field("chunkIndex", DataType.INT64)
        schema.add_field("chunkType", DataType.VARCHAR, max_length=32)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("charStart", DataType.INT64)
        schema.add_field("charEnd", DataType.INT64)
        schema.add_field("sectionPath", DataType.VARCHAR, max_length=1024)
        schema.add_field("tokenCount", DataType.INT64)
        schema.add_field("text_embed", DataType.FLOAT_VECTOR, dim=vector_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="text_embed",
            index_type=vector_spec.index_type,
            metric_type=vector_spec.metric_type,
            params=dict(vector_spec.build_params),
        )
        self._add_chunk_scalar_indexes(index_params, CHUNK_SCALAR_INDEX_SPECS)

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("fulltext chunk collection created", collection_name=collection_name)

    def ensure_chunk_scalar_indexes(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Create the approved fulltext scalar indexes when they are missing."""
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")

        fields = _collection_field_names(self.client, collection_name)
        index_details = _describe_collection_indexes(self.client, collection_name)

        planned: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        missing_fields: list[dict[str, Any]] = []
        for spec in CHUNK_SCALAR_INDEX_SPECS:
            field_name = str(spec["field_name"])
            if fields and field_name not in fields:
                missing_fields.append(
                    {
                        "field_name": field_name,
                        "index_name": spec["index_name"],
                        "reason": "field_missing",
                    }
                )
                continue
            if _has_matching_index(index_details, spec):
                skipped.append(
                    {
                        "field_name": field_name,
                        "index_name": spec["index_name"],
                        "reason": "already_indexed",
                    }
                )
                continue
            planned.append(_public_index_spec(spec))

        created: list[dict[str, Any]] = []
        if planned and not dry_run:
            index_params = self.client.prepare_index_params()
            self._add_chunk_scalar_indexes(index_params, planned)
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            created = planned
            logger.info(
                "chunk scalar indexes created",
                collection_name=collection_name,
                indexes=[item["index_name"] for item in created],
            )

        return {
            "collection_name": collection_name,
            "dry_run": dry_run,
            "planned": planned if dry_run else [],
            "created": created,
            "skipped": skipped,
            "missing_fields": missing_fields,
        }

    def delete_chunks_by_doc_id(self, collection_name: str, doc_id: str) -> dict[str, Any]:
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        if not doc_id or not str(doc_id).strip():
            raise ValueError("doc_id is required for chunk delete")
        self._ensure_collection(collection_name)

        filter_expr = f'doc_id == "{_escape_milvus_string(str(doc_id))}"'
        return _call_milvus_with_timeout(
            self.client.delete,
            timeout=settings.MILVUS_DELETE_TIMEOUT_SEC,
            collection_name=collection_name,
            filter=filter_expr,
        )

    def insert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[Chunk],
        *,
        vector_dim: int = settings.VECTOR_DIM,
        source_record_update_time: Optional[str] = None,
        parser_version: str = DEFAULT_PARSER_VERSION,
        chunker_version: str = DEFAULT_CHUNKER_VERSION,
        embedding_model: str = settings.EMBEDDING_MODEL,
        embedding_version: str = "",
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        ingested_at: Optional[str] = None,
    ) -> dict[str, Any]:
        records = self._prepare_chunk_records(
            chunks,
            vector_dim=vector_dim,
            source_record_update_time=source_record_update_time,
            parser_version=parser_version,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            schema_version=schema_version,
            ingested_at=ingested_at,
        )
        if not records:
            return {"insert_count": 0}
        collection_name = self.resolve_collection_target(
            collection_name=collection_name,
            vector_dim=vector_dim,
        ).resolved_name
        self._ensure_collection(collection_name)
        return self._insert_records(collection_name, records)

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[Chunk],
        *,
        vector_dim: int = settings.VECTOR_DIM,
        source_record_update_time: Optional[str] = None,
        parser_version: str = DEFAULT_PARSER_VERSION,
        chunker_version: str = DEFAULT_CHUNKER_VERSION,
        embedding_model: str = settings.EMBEDDING_MODEL,
        embedding_version: str = "",
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        ingested_at: Optional[str] = None,
    ) -> dict[str, Any]:
        records = self._prepare_chunk_records(
            chunks,
            vector_dim=vector_dim,
            source_record_update_time=source_record_update_time,
            parser_version=parser_version,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            schema_version=schema_version,
            ingested_at=ingested_at,
        )
        if not records:
            return {"upsert_count": 0}
        collection_name = self.resolve_collection_target(
            collection_name=collection_name,
            vector_dim=vector_dim,
        ).resolved_name
        self._ensure_collection(collection_name)
        return self._upsert_records(collection_name, records)

    def replace_chunks_for_document(
        self,
        collection_name: str,
        doc_id: str,
        chunks: Sequence[Chunk],
        *,
        vector_dim: int = settings.VECTOR_DIM,
        source_record_update_time: Optional[str] = None,
        parser_version: str = DEFAULT_PARSER_VERSION,
        chunker_version: str = DEFAULT_CHUNKER_VERSION,
        embedding_model: str = settings.EMBEDDING_MODEL,
        embedding_version: str = "",
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        ingested_at: Optional[str] = None,
    ) -> dict[str, Any]:
        if not doc_id or not str(doc_id).strip():
            raise ValueError("doc_id is required for chunk replace")

        records = self._prepare_chunk_records(
            chunks,
            vector_dim=vector_dim,
            expected_doc_id=str(doc_id),
            source_record_update_time=source_record_update_time,
            parser_version=parser_version,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            schema_version=schema_version,
            ingested_at=ingested_at,
        )
        collection_name = self.resolve_collection_target(
            collection_name=collection_name,
            vector_dim=vector_dim,
        ).resolved_name
        if not records:
            raise ValueError(
                "empty chunk replacement is not a delete instruction; use the explicit DELETE event"
            )

        self._ensure_collection(collection_name)
        upsert_result = self._upsert_records(collection_name, records)
        active_ids = [str(record["id"]) for record in records]
        visibility = self.wait_for_visible_chunk_ids(
            collection_name,
            active_ids,
            retries=settings.FULLTEXT_CHUNK_VISIBILITY_RETRIES,
            retry_delay_sec=settings.FULLTEXT_CHUNK_VISIBILITY_RETRY_DELAY_SEC,
            timeout=settings.FULLTEXT_CHUNK_VISIBILITY_CHECK_TIMEOUT_SEC,
        )
        if not visibility["success"]:
            raise RuntimeError(
                "fulltext safe replace visibility check failed: "
                f"visible={visibility['visible_count']}, expected={visibility['expected_count']}"
            )
        escaped_doc_id = _escape_milvus_string(str(doc_id))
        escaped_ids = ",".join(
            f'"{_escape_milvus_string(chunk_id)}"' for chunk_id in active_ids
        )
        cleanup_result = _call_milvus_with_timeout(
            self.client.delete,
            timeout=settings.MILVUS_DELETE_TIMEOUT_SEC,
            collection_name=collection_name,
            filter=(
                f'doc_id == "{escaped_doc_id}" and '
                f"id not in [{escaped_ids}]"
            ),
        )
        return {
            "delete_result": cleanup_result,
            "insert_result": {
                "insert_count": int(
                    upsert_result.get("upsert_count")
                    or upsert_result.get("insert_count")
                    or len(records)
                )
            },
            "write_mode": "upsert_then_cleanup_stale",
            "visibility_check": visibility,
        }

    def close(self) -> None:
        if self.client:
            self.client.close()

    def count_visible_chunk_ids(
        self,
        collection_name: str,
        chunk_ids: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> int:
        normalized_ids = [str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id or "").strip()]
        if not normalized_ids:
            return 0
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        return self._count_visible_chunk_ids_resolved(
            collection_name,
            normalized_ids,
            timeout=timeout,
        )

    def wait_for_visible_chunk_ids(
        self,
        collection_name: str,
        chunk_ids: Sequence[str],
        *,
        retries: int,
        retry_delay_sec: float,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        normalized_ids = list(dict.fromkeys(
            str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id or "").strip()
        ))
        expected = len(normalized_ids)
        if not expected:
            return {"success": False, "visible_count": 0, "expected_count": 0, "attempts": 0}
        batch_size = max(int(settings.MILVUS_VISIBILITY_BATCH_SIZE), 1)
        visible = 0
        attempts = 0
        for attempts in range(1, max(int(retries), 0) + 2):
            visible = sum(
                self._count_visible_chunk_ids_resolved(collection_name, batch, timeout=timeout)
                for batch in _batches(normalized_ids, batch_size)
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

    def _count_visible_chunk_ids_resolved(
        self,
        collection_name: str,
        normalized_ids: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> int:
        if not normalized_ids:
            return 0
        self._ensure_collection(collection_name)
        filter_expr = "id in [" + ",".join(
            f'"{_escape_milvus_string(chunk_id)}"' for chunk_id in normalized_ids
        ) + "]"
        rows = _call_milvus_with_timeout(
            self.client.query,
            timeout=(
                settings.MILVUS_QUERY_TIMEOUT_SEC
                if timeout is None
                else timeout
            ),
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["id"],
            consistency_level="Eventually",
        )
        return len(rows or [])

    def _ensure_collection(self, collection_name: str) -> None:
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")

    def _insert_records(self, collection_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return _call_milvus_with_timeout(
            self.client.insert,
            timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
            collection_name=collection_name,
            data=records,
        )

    def _upsert_records(self, collection_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        upsert = getattr(self.client, "upsert", None)
        if upsert is None:
            insert_result = _call_milvus_with_timeout(
                self.client.insert,
                timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
                collection_name=collection_name,
                data=records,
            )
            return {"upsert_count": insert_result.get("insert_count", len(records)), "fallback": "insert"}
        return _call_milvus_with_timeout(
            upsert,
            timeout=settings.MILVUS_WRITE_TIMEOUT_SEC,
            collection_name=collection_name,
            data=records,
        )

    def _add_chunk_scalar_indexes(self, index_params: Any, specs: Sequence[dict[str, Any]]) -> None:
        for spec in specs:
            kwargs: dict[str, Any] = {
                "field_name": spec["field_name"],
                "index_name": spec["index_name"],
                "index_type": spec.get("index_type", CHUNK_SCALAR_INDEX_TYPE),
            }
            params = spec.get("params")
            if params:
                kwargs["params"] = dict(params)
            index_params.add_index(**kwargs)

    def _prepare_chunk_records(
        self,
        chunks: Sequence[Chunk],
        *,
        vector_dim: int,
        expected_doc_id: Optional[str] = None,
        source_record_update_time: Optional[str] = None,
        parser_version: str = DEFAULT_PARSER_VERSION,
        chunker_version: str = DEFAULT_CHUNKER_VERSION,
        embedding_model: str = settings.EMBEDDING_MODEL,
        embedding_version: str = "",
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        ingested_at: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for chunk in chunks:
            self._validate_writable_chunk(chunk, vector_dim=vector_dim, expected_doc_id=expected_doc_id)
            record = {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "keyword": chunk.keyword,
                "accessType": chunk.accessType,
                "classificationCode": chunk.classificationCode,
                "year": chunk.year,
                "hasAuthorname": chunk.hasAuthorname,
                "hasCreateOrganizationname": chunk.hasCreateOrganizationname,
                "language": chunk.language,
                "id": chunk.id,
                "chunkIndex": chunk.chunkIndex,
                "chunkType": chunk.chunkType,
                "text": chunk.text,
                "charStart": chunk.charStart,
                "charEnd": chunk.charEnd,
                "sectionPath": _truncate_varchar(chunk.sectionPath, MAX_SECTION_PATH_LENGTH),
                "tokenCount": chunk.tokenCount,
                "text_embed": chunk.text_embed,
            }
            records.append({key: record.get(key) for key in FULLTEXT_FIELDS})
        return records

    @staticmethod
    def _validate_writable_chunk(
        chunk: Chunk,
        *,
        vector_dim: int,
        expected_doc_id: Optional[str] = None,
    ) -> None:
        if expected_doc_id is not None and chunk.doc_id != expected_doc_id:
            raise ValueError(
                f"Chunk {chunk.child_chunk_id} doc_id mismatch: expected={expected_doc_id}, actual={chunk.doc_id}"
            )
        if not chunk.text or not chunk.text.strip():
            raise ValueError(f"Chunk {chunk.child_chunk_id} text is empty")
        if not chunk.language:
            raise ValueError(f"Chunk {chunk.child_chunk_id} is missing language")
        if not chunk.sectionPath:
            raise ValueError(f"Chunk {chunk.child_chunk_id} is missing sectionPath")
        if chunk.tokenCount is None:
            raise ValueError(f"Chunk {chunk.child_chunk_id} is missing tokenCount")
        if chunk.text_embed is None:
            raise ValueError(f"Chunk {chunk.child_chunk_id} is missing text_embed")
        if not isinstance(chunk.text_embed, list):
            raise ValueError(f"Chunk {chunk.child_chunk_id} text_embed must be a list")
        if len(chunk.text_embed) != vector_dim:
            raise ValueError(
                f"Chunk {chunk.child_chunk_id} text_embed dimension mismatch: "
                f"expected={vector_dim}, actual={len(chunk.text_embed)}"
            )


def _add_string_array_field(schema: Any, field_name: str, *, nullable: bool) -> None:
    schema.add_field(
        field_name,
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=4096,
        max_length=65535,
        nullable=nullable,
    )


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _call_milvus_with_timeout(method: Any, *, timeout: float, **kwargs: Any) -> Any:
    """Call a Milvus RPC with a deadline when the client supports it.

    Small test doubles and older pymilvus releases may not expose the
    ``timeout`` keyword.  Detect that capability rather than retrying an
    already-issued write after a ``TypeError``.
    """
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


def _truncate_varchar(value: Any, max_length: int) -> str:
    text = str(value or "")
    if len(text) <= max_length:
        return text
    return text[:max_length]


def _public_index_spec(spec: dict[str, Any]) -> dict[str, Any]:
    public = {
        "field_name": spec["field_name"],
        "index_name": spec["index_name"],
        "index_type": spec.get("index_type", CHUNK_SCALAR_INDEX_TYPE),
    }
    params = spec.get("params")
    if params:
        public["params"] = dict(params)
    return public


def _batches(values: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _collection_field_names(client: Any, collection_name: str) -> set[str]:
    if not hasattr(client, "describe_collection"):
        return set()
    return {
        name
        for name in (_field_name(field) for field in _description_fields(client.describe_collection(collection_name)))
        if name
    }


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


def _has_matching_index(index_details: Sequence[dict[str, Any]], spec: dict[str, Any]) -> bool:
    for detail in index_details:
        if detail.get("index_name") != spec["index_name"] and detail.get("field_name") != spec["field_name"]:
            continue
        if detail.get("field_name") not in {None, spec["field_name"]}:
            continue
        params = spec.get("params") or {}
        if not params:
            return True
    return False


def _description_fields(describe: Any) -> list[Any]:
    if isinstance(describe, dict):
        return list(describe.get("fields") or [])
    return list(getattr(describe, "fields", []) or [])


def _field_name(field: Any) -> Optional[str]:
    if isinstance(field, dict):
        return field.get("name")
    return getattr(field, "name", None)
