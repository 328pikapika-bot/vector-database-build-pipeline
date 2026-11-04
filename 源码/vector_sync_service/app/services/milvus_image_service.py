"""Independent Milvus service for DOCX-aligned image records."""

from __future__ import annotations

import math
import re
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from pymilvus import DataType, MilvusClient

from app.config import settings
from app.contracts.docx_contract import (
    IMAGE_DOC_FIELDS,
    IMAGE_FIELDS,
    PHASH_BITS,
    PHASH_BYTES,
    TEXT_VECTOR_DIM,
    VISUAL_TEXT_VECTOR_DIM,
    VISUAL_VECTOR_DIM,
    extract_docx_doc_meta,
    normalize_json_object,
    normalize_optional_string,
    normalize_string_array,
)
from app.models.image import ImageRecord
from app.observability.logger import get_logger
from app.services.image_features import source_image_id
from app.services.collection_targets import (
    CollectionTarget,
    build_image_collection_signature,
    build_image_dense_index_spec,
    collection_signature_description,
)


logger = get_logger(__name__)
_PRIMARY_KEY_CONFLICT_RE = re.compile(
    r"(duplicate|duplicated|primary key|already exists|duplicate key)",
    re.IGNORECASE,
)

DEFAULT_IMAGE_COLLECTION = settings.IMAGE_COLLECTION_NAME or "scientific_images"
IMAGE_SCHEMA_MODE_FOUR_VECTOR = "four_vector"
IMAGE_SCHEMA_MODE_SEVEN_VECTOR = "seven_vector"
IMAGE_SCHEMA_MODE_DOCX_5 = "docx_5_vector"
IMAGE_SCHEMA_MODE_STAGING_4 = "staging_4_vector"
IMAGE_SCHEMA_MODE_E2E_CAPTION_CONTEXT_4 = "e2e_caption_context_4"
IMAGE_SCHEMA_MODE_CAPTION_CONTEXT_2 = "caption_context_2_vector"
IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4 = "visual_caption_context_4_vector"
IMAGE_SCHEMA_MODE_HISTORY_V2 = "history_compatible_v2"
SUPPORTED_IMAGE_SCHEMA_MODES = {
    IMAGE_SCHEMA_MODE_FOUR_VECTOR,
    IMAGE_SCHEMA_MODE_SEVEN_VECTOR,
    IMAGE_SCHEMA_MODE_DOCX_5,
    IMAGE_SCHEMA_MODE_STAGING_4,
    IMAGE_SCHEMA_MODE_E2E_CAPTION_CONTEXT_4,
    IMAGE_SCHEMA_MODE_CAPTION_CONTEXT_2,
    IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    IMAGE_SCHEMA_MODE_HISTORY_V2,
}
IMAGE_SCHEMA_MODE_ALIASES = {
    # Keep the historical short name, but make it the real four-vector
    # persistence contract rather than silently adding legacy fields.
    IMAGE_SCHEMA_MODE_FOUR_VECTOR: IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    IMAGE_SCHEMA_MODE_SEVEN_VECTOR: IMAGE_SCHEMA_MODE_HISTORY_V2,
    IMAGE_SCHEMA_MODE_DOCX_5: IMAGE_SCHEMA_MODE_HISTORY_V2,
    IMAGE_SCHEMA_MODE_STAGING_4: IMAGE_SCHEMA_MODE_HISTORY_V2,
    IMAGE_SCHEMA_MODE_E2E_CAPTION_CONTEXT_4: IMAGE_SCHEMA_MODE_HISTORY_V2,
    IMAGE_SCHEMA_MODE_CAPTION_CONTEXT_2: IMAGE_SCHEMA_MODE_HISTORY_V2,
}
IMAGE_FOUR_VECTOR_FIELDS = IMAGE_FIELDS[:11]
IMAGE_SCALAR_INDEX_TYPE = "INVERTED"
IMAGE_SCALAR_INDEX_SPECS: tuple[dict[str, Any], ...] = (
    {"field_name": "id", "index_name": "image_id_inverted_idx", "index_type": IMAGE_SCALAR_INDEX_TYPE},
    {"field_name": "doc_id", "index_name": "image_doc_id_inverted_idx", "index_type": IMAGE_SCALAR_INDEX_TYPE},
    {
        "field_name": "img_meta",
        "index_name": "image_img_id_json_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'img_meta["img_id"]', "json_cast_type": "varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_sys_data_status_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["sysDataStatus"]', "json_cast_type": "varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_year_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["year"]', "json_cast_type": "varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_classification_code_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["classificationCode"]', "json_cast_type": "varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_language_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["language"]', "json_cast_type": "array_varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_access_type_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["accessType"]', "json_cast_type": "varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_keyword_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["keyword"]', "json_cast_type": "array_varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_author_name_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["hasAuthorname"]', "json_cast_type": "array_varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_create_org_name_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["hasCreateOrganizationname"]', "json_cast_type": "array_varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_source_sys_source_id_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["hasSourcesysSourceId"]', "json_cast_type": "array_varchar"},
    },
    {
        "field_name": "doc_meta",
        "index_name": "image_source_id_set_inverted_idx",
        "index_type": IMAGE_SCALAR_INDEX_TYPE,
        "params": {"json_path": 'doc_meta["hasSourceidSet"]', "json_cast_type": "array_varchar"},
    },
)


class MilvusImageService:
    """Milvus writer for the standalone image collection."""

    def __init__(
        self,
        uri: str = "",
        user: str = "",
        password: str = "",
        client: Optional[Any] = None,
        schema_mode: str = "",
    ) -> None:
        self.schema_mode = normalize_image_schema_mode(schema_mode or settings.IMAGE_VECTOR_SCHEMA_MODE)
        if client is not None:
            self.client = client
        else:
            self.client = MilvusClient(
                uri=uri or f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                user=user or settings.MILVUS_USER or "",
                password=password or settings.MILVUS_PASSWORD or "",
            )

    def init_image_collection(
        self,
        collection_name: str = DEFAULT_IMAGE_COLLECTION,
    ) -> dict[str, Any]:
        """Create or reuse the configured image collection without renaming it.

        The production image worker owns one explicitly configured collection.
        A missing collection is created once during worker startup; an existing
        collection must already satisfy the V2 contract.  In particular, this
        method never falls back to a suffixed collection name.
        """
        target = self._resolve_configured_collection_target(
            collection_name=collection_name,
            create_if_missing=True,
        )
        self.client.load_collection(target.resolved_name)
        payload = target.to_dict()
        payload["collection_name"] = target.resolved_name
        payload["collection_loaded"] = True
        return payload

    def resolve_collection_target(self, *, collection_name: str = DEFAULT_IMAGE_COLLECTION) -> CollectionTarget:
        """Resolve the configured collection for normal read/write operations.

        Runtime operations are deliberately unable to create collections.  The
        Stack startup path above is the only place that may create a missing
        image collection, which prevents a typo or index mismatch from sending
        data to a generated suffixed collection.
        """
        return self._resolve_configured_collection_target(
            collection_name=collection_name,
            create_if_missing=False,
        )

    def _resolve_configured_collection_target(
        self,
        *,
        collection_name: str,
        create_if_missing: bool,
    ) -> CollectionTarget:
        logical_name = str(collection_name or settings.IMAGE_COLLECTION_NAME).strip()
        if not logical_name:
            raise ValueError("IMAGE_COLLECTION_NAME is required")

        created = False
        if self.client.has_collection(logical_name):
            if create_if_missing:
                self._validate_configured_image_collection(logical_name)
            else:
                self._validate_existing_image_schema(logical_name)
        elif create_if_missing:
            self._ensure_image_collection_created(logical_name)
            created = True
        else:
            raise ValueError(
                f"Configured image collection {logical_name!r} does not exist. "
                "Start the image Stack once to create it before processing Kafka records."
            )

        dense_spec = build_image_dense_index_spec()
        return CollectionTarget(
            logical_name=logical_name,
            resolved_name=logical_name,
            index_signature=dense_spec.signature,
            created=created,
            reused=not created,
            source_collection_name=logical_name,
            schema_signature=build_image_collection_signature(schema_mode=self.schema_mode),
        )

    def _ensure_image_collection_created(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name):
            self._validate_existing_image_schema(collection_name)
            return

        _validate_proxy_vector_capacity(
            1 + len(self.dense_vector_index_fields) + (1 if self.includes_sparse_vector else 0)
        )

        binary_vector = _required_datatype("BINARY_VECTOR")
        float16_vector = _required_datatype("FLOAT16_VECTOR")
        sparse_float_vector = _required_datatype("SPARSE_FLOAT_VECTOR")
        json_type = _required_datatype("JSON")

        schema_kwargs: dict[str, Any] = {"auto_id": False, "enable_dynamic_field": False}
        schema_kwargs["description"] = collection_signature_description(
            build_image_collection_signature(schema_mode=self.schema_mode),
            build_image_dense_index_spec().signature,
        )
        schema = self.client.create_schema(**schema_kwargs)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_meta", json_type)
        schema.add_field("raw_img_meta", json_type, nullable=True)
        schema.add_field("img_meta", json_type)
        schema.add_field("context_before", DataType.VARCHAR, max_length=65535, nullable=True)
        schema.add_field("context_after", DataType.VARCHAR, max_length=65535, nullable=True)
        schema.add_field("p_hash", binary_vector, dim=PHASH_BITS)
        schema.add_field("visual_embed", float16_vector, dim=VISUAL_VECTOR_DIM)
        schema.add_field("caption_embed", float16_vector, dim=TEXT_VECTOR_DIM)
        schema.add_field("context_embed", float16_vector, dim=TEXT_VECTOR_DIM)
        if self.includes_text_field:
            schema.add_field("text", DataType.VARCHAR, max_length=65535, nullable=True)
            schema.add_field("text_embed", float16_vector, dim=TEXT_VECTOR_DIM)
            schema.add_field("visual_text_embed", float16_vector, dim=VISUAL_TEXT_VECTOR_DIM)
        if self.includes_sparse_vector:
            # Retained only for the seven-vector compatibility layout. Milvus
            # does not allow nullable sparse-vector fields, so that layout
            # uses a reserved sentinel and does not enable a BM25 function.
            schema.add_field("text_sparse_vec", sparse_float_vector)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="p_hash",
            index_type="BIN_FLAT",
            metric_type="HAMMING",
        )
        dense_vector_spec = build_image_dense_index_spec()
        for field_name, dim_metric in self.dense_vector_index_fields:
            index_params.add_index(
                field_name=field_name,
                index_type=dense_vector_spec.index_type,
                metric_type=dense_vector_spec.metric_type or dim_metric,
                params=dict(dense_vector_spec.build_params),
            )
        if self.includes_sparse_vector:
            index_params.add_index(
                field_name="text_sparse_vec",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        self._add_image_scalar_indexes(index_params, _resolve_image_scalar_index_specs())

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("image collection created", collection_name=collection_name)

    def ensure_image_scalar_indexes(
        self,
        collection_name: str = DEFAULT_IMAGE_COLLECTION,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Create the approved scalar/JSON-path indexes when they are missing."""
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")

        self._validate_existing_image_schema(collection_name)
        fields = _collection_field_names(self.client, collection_name)
        specs = _resolve_image_scalar_index_specs(fields)
        index_details = _describe_collection_indexes(self.client, collection_name)

        planned: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        missing_fields: list[dict[str, Any]] = []
        for spec in specs:
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
            self._add_image_scalar_indexes(index_params, planned)
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            created = planned
            logger.info(
                "image scalar indexes created",
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

    @property
    def includes_sparse_vector(self) -> bool:
        return self.schema_mode == IMAGE_SCHEMA_MODE_HISTORY_V2

    @property
    def uses_bm25_function(self) -> bool:
        return False

    @property
    def includes_caption_context_vectors(self) -> bool:
        return True

    @property
    def includes_visual_embed(self) -> bool:
        return True

    @property
    def includes_text_embed(self) -> bool:
        return self.schema_mode == IMAGE_SCHEMA_MODE_HISTORY_V2

    @property
    def includes_visual_text_embed(self) -> bool:
        return self.schema_mode == IMAGE_SCHEMA_MODE_HISTORY_V2

    @property
    def includes_text_field(self) -> bool:
        return self.schema_mode == IMAGE_SCHEMA_MODE_HISTORY_V2

    @property
    def dense_vector_index_fields(self) -> tuple[tuple[str, str], ...]:
        fields: tuple[tuple[str, str], ...] = (
            ("visual_embed", "IP"),
            ("caption_embed", "IP"),
            ("context_embed", "IP"),
        )
        if self.includes_text_embed:
            fields += (("text_embed", "IP"), ("visual_text_embed", "IP"))
        return fields

    @property
    def writable_fields(self) -> tuple[str, ...]:
        if self.schema_mode == IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4:
            return IMAGE_FOUR_VECTOR_FIELDS
        return IMAGE_FIELDS

    def delete_images_by_doc_id(self, collection_name: str, doc_id: str) -> dict[str, Any]:
        if not doc_id or not str(doc_id).strip():
            raise ValueError("doc_id is required for image delete")
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        filter_expr = f'doc_id == "{_escape_milvus_string(str(doc_id))}"'
        return self.client.delete(collection_name=collection_name, filter=filter_expr)

    def delete_image_by_img_id(self, collection_name: str, img_id: str) -> dict[str, Any]:
        if not img_id or not str(img_id).strip():
            raise ValueError("img_id is required for single image delete")
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        filter_expr = f'id == "{_escape_milvus_string(str(img_id))}"'
        return self.client.delete(collection_name=collection_name, filter=filter_expr)

    def delete_images_by_img_ids(
        self,
        collection_name: str,
        img_ids: Sequence[str],
    ) -> dict[str, Any]:
        resolved_ids = [str(item).strip() for item in img_ids if str(item or "").strip()]
        if not resolved_ids:
            raise ValueError("img_ids are required for batch image delete")
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        escaped_ids = ",".join(f'"{_escape_milvus_string(item)}"' for item in resolved_ids)
        filter_expr = f"id in [{escaped_ids}]"
        return self.client.delete(collection_name=collection_name, filter=filter_expr)

    def insert_images(
        self,
        collection_name: str,
        images: Sequence[ImageRecord | dict[str, Any]],
    ) -> dict[str, Any]:
        records = self._prepare_image_records(images)
        if not records:
            return {"insert_count": 0}
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        return self._insert_records(collection_name, records)

    def image_ids_existing(
        self,
        collection_name: str,
        img_ids: Sequence[str],
    ) -> set[str]:
        resolved_ids = [str(item).strip() for item in img_ids if str(item or "").strip()]
        if not resolved_ids:
            return set()
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        escaped_ids = ",".join(f'"{_escape_milvus_string(item)}"' for item in resolved_ids)
        rows = self.client.query(
            collection_name=collection_name,
            filter=f"id in [{escaped_ids}]",
            output_fields=["id"],
            limit=len(resolved_ids),
        )
        existing: set[str] = set()
        for row in rows or []:
            image_id = str(row.get("id") or "").strip()
            if image_id:
                existing.add(image_id)
        return existing

    def count_visible_image_ids(
        self,
        collection_name: str,
        image_ids: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> int:
        resolved_ids = [str(item).strip() for item in image_ids if str(item or "").strip()]
        if not resolved_ids:
            return 0
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        escaped_ids = ",".join(f'"{_escape_milvus_string(item)}"' for item in resolved_ids)
        rows = self.client.query(
            collection_name=collection_name,
            filter=f"id in [{escaped_ids}]",
            output_fields=["id"],
            limit=len(resolved_ids),
            consistency_level="Eventually",
            timeout=timeout,
        )
        return len({str(row.get("id")) for row in rows or [] if row.get("id")})

    def wait_for_visible_image_ids(
        self,
        collection_name: str,
        image_ids: Sequence[str],
        *,
        retries: int,
        retry_delay_sec: float,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        normalized_ids = list(dict.fromkeys(
            str(image_id).strip() for image_id in image_ids if str(image_id or "").strip()
        ))
        expected = len(normalized_ids)
        if not expected:
            return {"success": False, "visible_count": 0, "expected_count": 0, "attempts": 0}
        batch_size = max(int(settings.MILVUS_VISIBILITY_BATCH_SIZE), 1)
        visible = 0
        attempts = 0
        for attempts in range(1, max(int(retries), 0) + 2):
            visible = sum(
                self.count_visible_image_ids(collection_name, batch, timeout=timeout)
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

    def replace_images_for_document(
        self,
        collection_name: str,
        doc_id: str,
        images: Sequence[ImageRecord | dict[str, Any]],
        *,
        complete_snapshot: bool = False,
    ) -> dict[str, Any]:
        if not doc_id or not str(doc_id).strip():
            raise ValueError("doc_id is required for image replace")

        records = self._prepare_image_records(images, expected_doc_id=str(doc_id))
        if not records:
            raise ValueError(
                "empty image replacement is not a delete instruction; use the explicit DELETE event"
            )
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        upsert_result = self._upsert_records(collection_name, records)
        active_ids = [str(record["id"]) for record in records]
        visibility = self.wait_for_visible_image_ids(
            collection_name,
            active_ids,
            retries=settings.IMAGE_VISIBILITY_RETRIES,
            retry_delay_sec=settings.IMAGE_VISIBILITY_RETRY_DELAY_SEC,
            timeout=settings.IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC,
        )
        if not visibility["success"]:
            raise RuntimeError(
                "image safe replace visibility check failed: "
                f"visible={visibility['visible_count']}, expected={visibility['expected_count']}"
            )
        if complete_snapshot:
            escaped_doc_id = _escape_milvus_string(str(doc_id))
            escaped_ids = ",".join(f'"{_escape_milvus_string(image_id)}"' for image_id in active_ids)
            delete_result = self.client.delete(
                collection_name=collection_name,
                filter=(
                    f'doc_id == "{escaped_doc_id}" and '
                    f"id not in [{escaped_ids}]"
                ),
            )
        else:
            delete_result = {
                "skipped": True,
                "reason": "complete_image_snapshot_required_for_stale_cleanup",
            }
        return {
            "delete_result": delete_result,
            "insert_result": {"insert_count": _extract_insert_count(upsert_result, fallback=len(records))},
            "write_mode": "upsert_then_cleanup_stale",
            "visibility_check": visibility,
        }

    def replace_images_for_documents(
        self,
        collection_name: str,
        documents: Mapping[str, Sequence[ImageRecord | dict[str, Any]]],
        *,
        complete_snapshot: bool = False,
    ) -> dict[str, Any]:
        """Upsert multiple complete document snapshots with one vector write.

        Each document remains an independent cleanup boundary, but the expensive
        vector upsert and optional visibility read are shared by the Kafka batch.
        """
        prepared_by_doc: dict[str, list[dict[str, Any]]] = {}
        all_records: list[dict[str, Any]] = []
        for raw_doc_id, images in documents.items():
            doc_id = str(raw_doc_id or "").strip()
            if not doc_id:
                raise ValueError("doc_id is required for image batch replace")
            records = self._prepare_image_records(images, expected_doc_id=doc_id)
            if not records:
                raise ValueError(
                    "empty image replacement is not a delete instruction; use the explicit DELETE event"
                )
            prepared_by_doc[doc_id] = records
            all_records.extend(records)

        if not all_records:
            return {
                "document_count": 0,
                "image_count": 0,
                "delete_results": {},
                "insert_result": {"insert_count": 0},
                "write_mode": "upsert_then_cleanup_stale_batch",
                "visibility_check": {"skipped": True, "reason": "empty_batch"},
            }

        logger.info(
            "image_milvus_batch_write_started",
            collection_name=collection_name,
            document_count=len(prepared_by_doc),
            image_count=len(all_records),
        )
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        logger.info(
            "image_milvus_batch_write_target_resolved",
            collection_name=collection_name,
            document_count=len(prepared_by_doc),
            image_count=len(all_records),
        )
        self._ensure_collection(collection_name)
        logger.info(
            "image_milvus_batch_write_upsert_started",
            collection_name=collection_name,
            document_count=len(prepared_by_doc),
            image_count=len(all_records),
        )
        upsert_result = self._upsert_records(collection_name, all_records)
        logger.info(
            "image_milvus_batch_write_upsert_finished",
            collection_name=collection_name,
            document_count=len(prepared_by_doc),
            image_count=len(all_records),
            insert_count=_extract_insert_count(upsert_result, fallback=len(all_records)),
        )
        active_ids = [str(record["id"]) for record in all_records]
        if settings.IMAGE_VISIBILITY_CHECK_ENABLED:
            visibility = self.wait_for_visible_image_ids(
                collection_name,
                active_ids,
                retries=settings.IMAGE_VISIBILITY_RETRIES,
                retry_delay_sec=settings.IMAGE_VISIBILITY_RETRY_DELAY_SEC,
                timeout=settings.IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC,
            )
            if not visibility["success"]:
                raise RuntimeError(
                    "image batch replace visibility check failed: "
                    f"visible={visibility['visible_count']}, expected={visibility['expected_count']}"
                )
        else:
            visibility = {"skipped": True, "reason": "IMAGE_VISIBILITY_CHECK_ENABLED=false"}

        delete_results: dict[str, Any] = {}
        for doc_id, records in prepared_by_doc.items():
            if not complete_snapshot:
                delete_results[doc_id] = {
                    "skipped": True,
                    "reason": "complete_image_snapshot_required_for_stale_cleanup",
                }
                continue
            escaped_doc_id = _escape_milvus_string(doc_id)
            escaped_ids = ",".join(
                f'"{_escape_milvus_string(str(record["id"]))}"' for record in records
            )
            delete_results[doc_id] = self.client.delete(
                collection_name=collection_name,
                filter=(
                    f'doc_id == "{escaped_doc_id}" and '
                    f"id not in [{escaped_ids}]"
                ),
            )

        return {
            "document_count": len(prepared_by_doc),
            "image_count": len(all_records),
            "delete_results": delete_results,
            "insert_result": {
                "insert_count": _extract_insert_count(upsert_result, fallback=len(all_records))
            },
            "write_mode": "upsert_then_cleanup_stale_batch",
            "visibility_check": visibility,
        }

    def replace_image_by_img_id(
        self,
        collection_name: str,
        image: ImageRecord | dict[str, Any],
    ) -> dict[str, Any]:
        records = self._prepare_image_records([image])
        record = records[0]

        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        upsert_result = self._upsert_records(collection_name, records)
        visibility = self.wait_for_visible_image_ids(
            collection_name,
            [record["id"]],
            retries=settings.IMAGE_VISIBILITY_RETRIES,
            retry_delay_sec=settings.IMAGE_VISIBILITY_RETRY_DELAY_SEC,
            timeout=settings.IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC,
        )
        if not visibility["success"]:
            raise RuntimeError("image upsert visibility check failed")
        return {
            "delete_result": {"skipped": True, "reason": "single_image_primary_key_upsert"},
            "insert_result": {"insert_count": _extract_insert_count(upsert_result, fallback=1)},
            "write_mode": "primary_key_upsert",
            "visibility_check": visibility,
        }

    def replace_images_by_img_id_batch(
        self,
        collection_name: str,
        images: Sequence[ImageRecord | dict[str, Any]],
    ) -> dict[str, Any]:
        records = self._prepare_image_records(images)
        if not records:
            return {"delete_results": [], "insert_result": {"insert_count": 0}, "batch_size": 0}
        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        upsert_result = self._upsert_records(collection_name, records)
        visibility = self.wait_for_visible_image_ids(
            collection_name,
            [str(record["id"]) for record in records],
            retries=settings.IMAGE_VISIBILITY_RETRIES,
            retry_delay_sec=settings.IMAGE_VISIBILITY_RETRY_DELAY_SEC,
            timeout=settings.IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC,
        )
        if not visibility["success"]:
            raise RuntimeError("image batch upsert visibility check failed")
        return {
            "delete_results": [],
            "insert_result": {"insert_count": _extract_insert_count(upsert_result, fallback=len(records))},
            "batch_size": len(records),
            "write_mode": "primary_key_upsert",
            "visibility_check": visibility,
        }

    def upsert_images_by_img_id_batch(
        self,
        collection_name: str,
        images: Sequence[ImageRecord | dict[str, Any]],
    ) -> dict[str, Any]:
        records = self._prepare_image_records(images)
        if not records:
            return {
                "delete_results": [],
                "insert_result": {"insert_count": 0},
                "batch_size": 0,
                "existing_count": 0,
                "new_count": 0,
                "write_mode": "upsert_if_exists",
            }

        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        upsert_result = self._upsert_records(collection_name, records)
        upsert_count = _extract_insert_count(upsert_result, fallback=len(records))
        return {
            "delete_results": [],
            "insert_result": {"insert_count": upsert_count},
            "batch_size": len(records),
            "existing_count": 0,
            "new_count": len(records),
            "write_mode": "upsert_if_exists",
        }

    def insert_or_skip_images_by_img_id_batch(
        self,
        collection_name: str,
        images: Sequence[ImageRecord | dict[str, Any]],
    ) -> dict[str, Any]:
        records = self._prepare_image_records(images)
        if not records:
            return {
                "delete_results": [],
                "insert_result": {"insert_count": 0},
                "batch_size": 0,
                "existing_count": 0,
                "new_count": 0,
                "skipped_existing_count": 0,
                "write_mode": "insert_or_skip_if_exists",
            }

        collection_name = self.resolve_collection_target(collection_name=collection_name).resolved_name
        self._ensure_collection(collection_name)
        try:
            insert_result = self._insert_records(collection_name, records)
        except Exception as error:
            if not _is_primary_key_conflict(error):
                raise
            existing_ids = self.image_ids_existing(
                collection_name,
                [str(record["id"]) for record in records],
            )
            new_records = [
                record for record in records if str(record["id"]) not in existing_ids
            ]
            insert_result = {"insert_count": 0}
            if new_records:
                insert_result = self._insert_records(collection_name, new_records)
            return {
                "delete_results": [],
                "insert_result": insert_result,
                "batch_size": len(records),
                "existing_count": len(existing_ids),
                "new_count": len(new_records),
                "skipped_existing_count": len(existing_ids),
                "write_mode": "insert_or_skip_if_exists",
            }

        inserted_count = _extract_insert_count(insert_result, fallback=len(records))
        return {
            "delete_results": [],
            "insert_result": insert_result,
            "batch_size": len(records),
            "existing_count": 0,
            "new_count": inserted_count,
            "skipped_existing_count": 0,
            "write_mode": "insert_or_skip_if_exists",
        }

    def close(self) -> None:
        if self.client:
            self.client.close()

    def _ensure_collection(self, collection_name: str) -> None:
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist!")
        self._validate_existing_image_schema(collection_name)

    def _validate_existing_image_schema(self, collection_name: str) -> None:
        if not hasattr(self.client, "describe_collection"):
            return
        describe = self.client.describe_collection(collection_name)
        fields = _description_fields(describe)
        field_names = {_field_name(field) for field in fields if _field_name(field)}
        legacy_fields = sorted(
            (set(IMAGE_DOC_FIELDS) | {
                "relative_url",
                "name",
                "sequence",
                "width",
                "height",
                "fileType",
                "size",
                "caption",
            })
            & field_names
        )
        if legacy_fields:
            preview = ", ".join(legacy_fields)
            raise ValueError(
                f"Collection {collection_name} contains unsupported top-level image fields: {preview}. "
                "Use a new V2 collection name or rebuild it with the JSON metadata contract."
            )
        missing = sorted(set(self.writable_fields) - field_names)
        if missing:
            preview = ", ".join(missing[:20])
            suffix = "" if len(missing) <= 20 else f", ... (+{len(missing) - 20})"
            raise ValueError(f"Collection {collection_name} is missing image schema fields: {preview}{suffix}")

        id_field = next((field for field in fields if _field_name(field) == "id"), None)
        if id_field is not None and not _is_primary_field(id_field):
            raise ValueError(f"Collection {collection_name} field id must be the image primary key")
        if id_field is not None and _field_type(id_field) != DataType.VARCHAR:
            raise ValueError(f"Collection {collection_name} field id must be VARCHAR image id")
        doc_id_field = next((field for field in fields if _field_name(field) == "doc_id"), None)
        if doc_id_field is not None and not _field_type_is(doc_id_field, DataType.VARCHAR):
            raise ValueError(f"Collection {collection_name} field doc_id must be VARCHAR document id")
        for json_field_name in ("doc_meta", "raw_img_meta", "img_meta"):
            json_field = next((field for field in fields if _field_name(field) == json_field_name), None)
            if json_field is not None and not _field_type_is(json_field, DataType.JSON):
                raise ValueError(f"Collection {collection_name} field {json_field_name} must be JSON")

    def _validate_configured_image_collection(self, collection_name: str) -> None:
        """Validate the existing configured name before the image worker consumes Kafka."""
        self._validate_existing_image_schema(collection_name)
        describe = self.client.describe_collection(collection_name)
        fields = _description_fields(describe)
        fields_by_name = {
            name: field
            for field in fields
            if (name := _field_name(field))
        }
        expected_fields = set(self.writable_fields)
        actual_fields = set(fields_by_name)
        unexpected = sorted(actual_fields - expected_fields)
        if unexpected:
            raise ValueError(
                f"Configured image collection {collection_name} has unexpected fields: "
                f"{', '.join(unexpected)}. It will not be rebuilt automatically."
            )

        expected_types = {
            "id": DataType.VARCHAR,
            "doc_id": DataType.VARCHAR,
            "doc_meta": _required_datatype("JSON"),
            "raw_img_meta": _required_datatype("JSON"),
            "img_meta": _required_datatype("JSON"),
            "context_before": DataType.VARCHAR,
            "context_after": DataType.VARCHAR,
            "p_hash": _required_datatype("BINARY_VECTOR"),
            "visual_embed": _required_datatype("FLOAT16_VECTOR"),
            "caption_embed": _required_datatype("FLOAT16_VECTOR"),
            "context_embed": _required_datatype("FLOAT16_VECTOR"),
        }
        if self.includes_text_field:
            expected_types.update(
                {
                    "text": DataType.VARCHAR,
                    "text_embed": _required_datatype("FLOAT16_VECTOR"),
                    "visual_text_embed": _required_datatype("FLOAT16_VECTOR"),
                }
            )
        if self.includes_sparse_vector:
            expected_types["text_sparse_vec"] = _required_datatype("SPARSE_FLOAT_VECTOR")

        for field_name, expected_type in expected_types.items():
            field = fields_by_name.get(field_name)
            if field is None or not _field_type_is(field, expected_type):
                raise ValueError(
                    f"Configured image collection {collection_name} field {field_name} "
                    f"must use {getattr(expected_type, 'name', expected_type)}"
                )

        expected_dims = {
            "p_hash": PHASH_BITS,
            "visual_embed": VISUAL_VECTOR_DIM,
            "caption_embed": TEXT_VECTOR_DIM,
            "context_embed": TEXT_VECTOR_DIM,
        }
        if self.includes_text_field:
            expected_dims.update(
                {
                    "text_embed": TEXT_VECTOR_DIM,
                    "visual_text_embed": VISUAL_TEXT_VECTOR_DIM,
                }
            )
        for field_name, expected_dim in expected_dims.items():
            actual_dim = _field_dimension(fields_by_name[field_name])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Configured image collection {collection_name} field {field_name} "
                    f"must have dim={expected_dim}, got {actual_dim!r}"
                )

        expected_description = collection_signature_description(
            build_image_collection_signature(schema_mode=self.schema_mode),
            build_image_dense_index_spec().signature,
        )
        recorded_description = _collection_description(describe)
        if recorded_description.startswith("vector-sync-target-signature:") and (
            recorded_description != expected_description
        ):
            raise ValueError(
                f"Configured image collection {collection_name} has a different target signature. "
                "It will not be rebuilt automatically."
            )
        self._validate_image_index_contract(collection_name)

    def _validate_image_index_contract(self, collection_name: str) -> None:
        index_details = _describe_collection_indexes(self.client, collection_name)
        dense_spec = build_image_dense_index_spec()
        expected_specs: list[dict[str, Any]] = [
            {
                "field_name": "p_hash",
                "index_type": "BIN_FLAT",
                "metric_type": "HAMMING",
            },
            *[
                {
                    "field_name": field_name,
                    "index_type": dense_spec.index_type,
                    "metric_type": dense_spec.metric_type,
                    "params": dict(dense_spec.build_params),
                }
                for field_name, _metric in self.dense_vector_index_fields
            ],
        ]
        if self.includes_sparse_vector:
            expected_specs.append(
                {
                    "field_name": "text_sparse_vec",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                }
            )
        expected_specs.extend(_resolve_image_scalar_index_specs(set(self.writable_fields)))

        missing = [
            _index_contract_label(spec)
            for spec in expected_specs
            if not _has_matching_index_contract(index_details, spec)
        ]
        if missing:
            raise ValueError(
                f"Configured image collection {collection_name} is missing or has incompatible indexes: "
                f"{', '.join(missing)}. It will not be rebuilt automatically."
            )

    def _insert_records(self, collection_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self.client.insert(collection_name=collection_name, data=records)

    def _upsert_records(self, collection_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        upsert = getattr(self.client, "upsert", None)
        if upsert is None:
            raise RuntimeError("Milvus client does not support atomic primary-key upsert")
        return upsert(collection_name=collection_name, data=records)

    def _add_image_scalar_indexes(self, index_params: Any, specs: Sequence[dict[str, Any]]) -> None:
        for spec in specs:
            kwargs: dict[str, Any] = {
                "field_name": spec["field_name"],
                "index_name": spec["index_name"],
                "index_type": spec.get("index_type", IMAGE_SCALAR_INDEX_TYPE),
            }
            params = spec.get("params")
            if params:
                kwargs["params"] = dict(params)
            index_params.add_index(**kwargs)

    def _prepare_image_records(
        self,
        images: Sequence[ImageRecord | dict[str, Any]],
        *,
        expected_doc_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in images:
            image = _coerce_image_record(item)
            _validate_writable_image(
                image,
                expected_doc_id=expected_doc_id,
                schema_mode=self.schema_mode,
            )
            doc_meta = extract_docx_doc_meta(image)
            record: dict[str, Any] = {
                "id": image.img_id,
                "doc_id": normalize_optional_string(doc_meta.get("docId") or image.docId),
                "doc_meta": _build_doc_meta(image, doc_meta),
                "raw_img_meta": _build_raw_img_meta(image),
                "img_meta": _build_img_meta(image),
                "context_before": normalize_optional_string(image.context_before),
                "context_after": normalize_optional_string(image.context_after),
                "p_hash": _normalize_p_hash(image.p_hash, image.img_id),
                "visual_embed": _normalize_float16_vector(
                    image.visual_embed,
                    VISUAL_VECTOR_DIM,
                    image.img_id,
                    "visual_embed",
                ),
                "caption_embed": _normalize_or_reserved_float16_vector(
                    image.caption_embed,
                    TEXT_VECTOR_DIM,
                    image.img_id,
                    "caption_embed",
                ),
                "context_embed": _normalize_or_reserved_float16_vector(
                    image.context_embed,
                    TEXT_VECTOR_DIM,
                    image.img_id,
                    "context_embed",
                ),
            }
            if self.includes_text_field:
                record.update(
                    {
                        "text": normalize_optional_string(image.text) or _reserved_text_placeholder(),
                        "text_embed": _normalize_or_reserved_float16_vector(
                            image.text_embed,
                            TEXT_VECTOR_DIM,
                            image.img_id,
                            "text_embed",
                        ),
                        "visual_text_embed": _normalize_or_reserved_float16_vector(
                            image.visual_text_embed,
                            VISUAL_TEXT_VECTOR_DIM,
                            image.img_id,
                            "visual_text_embed",
                        ),
                    }
                )
            if self.includes_sparse_vector:
                record["text_sparse_vec"] = _normalize_or_reserved_sparse_vector(
                    image.text_sparse_vec,
                    image.img_id,
                )
            records.append({key: record.get(key) for key in self.writable_fields})
        return records


def validate_writable_image_record(
    image: ImageRecord,
    *,
    expected_doc_id: Optional[str] = None,
    schema_mode: str = "",
) -> None:
    """Validate image record writability without creating a Milvus client."""
    resolved_schema_mode = normalize_image_schema_mode(
        schema_mode or settings.IMAGE_VECTOR_SCHEMA_MODE
    )
    _validate_writable_image(
        image,
        expected_doc_id=expected_doc_id,
        schema_mode=resolved_schema_mode,
    )


def normalize_image_schema_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = IMAGE_SCHEMA_MODE_HISTORY_V2
    if normalized not in SUPPORTED_IMAGE_SCHEMA_MODES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_SCHEMA_MODES))
        raise ValueError(f"unsupported IMAGE_VECTOR_SCHEMA_MODE: {value!r}; allowed: {allowed}")
    return IMAGE_SCHEMA_MODE_ALIASES.get(normalized, normalized)


def _coerce_image_record(value: ImageRecord | dict[str, Any]) -> ImageRecord:
    if isinstance(value, ImageRecord):
        return value
    if isinstance(value, dict):
        return ImageRecord.model_validate(value)
    raise TypeError("image record must be ImageRecord or dict")


def _extract_insert_count(result: Any, *, fallback: int) -> int:
    if isinstance(result, dict):
        value = result.get("insert_count")
        if value is not None:
            try:
                return int(value)
            except Exception:
                return fallback
    return fallback


def _is_primary_key_conflict(error: Exception) -> bool:
    return bool(_PRIMARY_KEY_CONFLICT_RE.search(str(error or "")))


def _validate_writable_image(
    image: ImageRecord,
    *,
    expected_doc_id: Optional[str] = None,
    schema_mode: str = "",
) -> None:
    if expected_doc_id is not None and image.docId != expected_doc_id:
        raise ValueError(f"Image {image.img_id} docId mismatch: expected={expected_doc_id}, actual={image.docId}")
    if not image.docId or not str(image.docId).strip():
        raise ValueError("Image docId is required")
    if not image.img_id or not str(image.img_id).strip():
        raise ValueError("Image img_id is required")

    upstream_id = source_image_id(image.raw_img_meta)
    if not upstream_id:
        raise ValueError(f"Image {image.img_id} is missing raw_img_meta.id")
    if image.img_id != upstream_id:
        raise ValueError(f"Image {image.img_id} does not match upstream id {upstream_id}")

    if not (extract_docx_doc_meta(image).get("language") or image.language):
        raise ValueError(f"Image {image.img_id} is missing language")
    if not (extract_docx_doc_meta(image).get("sysDataStatus") or image.sysDataStatus):
        raise ValueError(f"Image {image.img_id} is missing sysDataStatus")
    # The source/provenance JSON objects are persisted as-is.  Text-generation
    # and sparse retrieval fields remain optional and receive inert placeholders
    # at write time when their providers are disabled.
    _normalize_p_hash(image.p_hash, image.img_id)
    _validate_dense_vector(image.visual_embed, VISUAL_VECTOR_DIM, image.img_id, "visual_embed")
    _validate_optional_dense_vector(image.caption_embed, TEXT_VECTOR_DIM, image.img_id, "caption_embed")
    _validate_optional_dense_vector(image.context_embed, TEXT_VECTOR_DIM, image.img_id, "context_embed")


def _build_doc_meta(image: ImageRecord, doc_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": doc_meta.get("title") or normalize_string_array(image.title),
        "keyword": doc_meta.get("keyword") or normalize_string_array(image.keyword),
        "accessType": normalize_optional_string(doc_meta.get("accessType") or image.accessType),
        "classificationCode": normalize_optional_string(
            doc_meta.get("classificationCode") or image.classificationCode
        ),
        "year": normalize_optional_string(doc_meta.get("year") or image.year),
        "hasAuthorname": doc_meta.get("hasAuthorname") or normalize_string_array(image.hasAuthorname),
        "hasCreateOrganizationname": (
            doc_meta.get("hasCreateOrganizationname")
            or normalize_string_array(image.hasCreateOrganizationname)
        ),
        "language": doc_meta.get("language") or normalize_string_array(image.language),
        "sysDataStatus": normalize_optional_string(doc_meta.get("sysDataStatus") or image.sysDataStatus),
        "hasSourcesysSourceId": (
            doc_meta.get("hasSourcesysSourceId")
            or normalize_string_array(image.hasSourcesysSourceId)
        ),
        "hasSourceidSet": doc_meta.get("hasSourceidSet") or normalize_string_array(image.hasSourceidSet),
    }


def _build_raw_img_meta(image: ImageRecord) -> dict[str, Any]:
    """Preserve the source image object in one JSON field, with safe fallbacks."""
    meta = normalize_json_object(image.raw_img_meta)
    meta.setdefault("id", image.img_id)
    for key, value in (
        ("url", image.url),
        ("name", image.name),
        ("sequence", image.sequence),
        ("width", image.width),
        ("height", image.height),
        ("fileType", image.fileType),
        ("size", image.size),
    ):
        if key not in meta and value not in (None, "", [], {}):
            meta[key] = value
    return meta


def _build_img_meta(image: ImageRecord) -> dict[str, Any]:
    meta = normalize_json_object(image.img_meta)
    meta.update(
        {
            "img_id": image.img_id,
            "schema_mode": normalize_image_schema_mode(settings.IMAGE_VECTOR_SCHEMA_MODE),
            "text_embedding_model": settings.IMAGE_TEXT_EMBED_MODEL,
            "feature_contract_version": "image-feature-v2",
        }
    )
    return {key: value for key, value in meta.items() if value not in (None, "")}


def _normalize_p_hash(value: Any, image_id: str) -> bytes:
    if isinstance(value, bytes):
        result = value
    elif isinstance(value, bytearray):
        result = bytes(value)
    elif isinstance(value, str):
        compact = value.strip().lower().removeprefix("0x")
        if len(compact) != PHASH_BYTES * 2:
            raise ValueError(f"Image {image_id} p_hash must be 16 hex characters")
        try:
            result = bytes.fromhex(compact)
        except ValueError as exc:
            raise ValueError(f"Image {image_id} p_hash must be hex encoded") from exc
    elif isinstance(value, (list, tuple)):
        if len(value) != PHASH_BYTES:
            raise ValueError(f"Image {image_id} p_hash must contain {PHASH_BYTES} bytes")
        try:
            result = bytes(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Image {image_id} p_hash byte values must be integers") from exc
    else:
        raise ValueError(f"Image {image_id} is missing p_hash")

    if len(result) != PHASH_BYTES:
        raise ValueError(f"Image {image_id} p_hash must be {PHASH_BYTES} bytes for {PHASH_BITS} bits")
    if all(byte == 0 for byte in result):
        raise ValueError(f"Image {image_id} p_hash appears to be a placeholder")
    return result


def _validate_dense_vector(value: Any, dim: int, image_id: str, field_name: str) -> None:
    if value is None:
        raise ValueError(f"Image {image_id} is missing {field_name}")
    _validate_vector_contents(value, dim, image_id, field_name)


def _validate_optional_dense_vector(value: Any, dim: int, image_id: str, field_name: str) -> None:
    if value is None:
        return
    _validate_vector_contents(value, dim, image_id, field_name)


def _validate_vector_contents(value: Any, dim: int, image_id: str, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"Image {image_id} {field_name} must be a list")
    if len(value) != dim:
        raise ValueError(f"Image {image_id} {field_name} dimension mismatch: expected={dim}, actual={len(value)}")

    floats: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Image {image_id} {field_name} must contain numbers")
        candidate = float(item)
        if not math.isfinite(candidate):
            raise ValueError(f"Image {image_id} {field_name} must contain finite numbers")
        floats.append(candidate)
    if all(item == 0.0 for item in floats):
        raise ValueError(f"Image {image_id} {field_name} appears to be a placeholder vector")


def _normalize_float16_vector(value: Any, dim: int, image_id: str, field_name: str) -> np.ndarray:
    _validate_dense_vector(value, dim, image_id, field_name)
    return np.asarray(value, dtype=np.float16)


def _normalize_optional_float16_vector(value: Any, dim: int, image_id: str, field_name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    _validate_optional_dense_vector(value, dim, image_id, field_name)
    return np.asarray(value, dtype=np.float16)


def _normalize_or_reserved_float16_vector(
    value: Any,
    dim: int,
    image_id: str,
    field_name: str,
) -> np.ndarray:
    if value is None:
        return _reserved_float16_vector(dim)
    return _normalize_float16_vector(value, dim, image_id, field_name)


def _reserved_float16_vector(dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float16)
    if dim:
        vector[0] = np.float16(1e-6)
    return vector


def _reserved_text_placeholder() -> str:
    return "__reserved_image_text__"


def _validate_sparse_vector(value: Any, image_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Image {image_id} text_sparse_vec must be a non-empty dict")
    if not value:
        raise ValueError(f"Image {image_id} text_sparse_vec must not be empty")

    has_signal = False
    for key, raw_value in value.items():
        if key is None:
            raise ValueError(f"Image {image_id} text_sparse_vec keys must not be null")
        if not isinstance(raw_value, (int, float)):
            raise ValueError(f"Image {image_id} text_sparse_vec values must be numbers")
        candidate = float(raw_value)
        if not math.isfinite(candidate):
            raise ValueError(f"Image {image_id} text_sparse_vec values must be finite")
        has_signal = has_signal or candidate != 0.0
    if not has_signal:
        raise ValueError(f"Image {image_id} text_sparse_vec appears to be a placeholder vector")


def _normalize_or_reserved_sparse_vector(value: Any, image_id: str) -> dict[int, float] | dict[Any, float]:
    if value is None:
        return {0: 1e-6}
    _validate_sparse_vector(value, image_id)
    return dict(value)


def _resolve_image_scalar_index_specs(field_names: Optional[set[str]] = None) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for raw_spec in IMAGE_SCALAR_INDEX_SPECS:
        spec = _public_index_spec(raw_spec)
        if field_names is None or spec["field_name"] in field_names:
            resolved.append(spec)
            continue

        legacy_names = tuple(raw_spec.get("legacy_field_names") or ())
        legacy_index_names = dict(raw_spec.get("legacy_index_names") or {})
        legacy_field = next((name for name in legacy_names if name in field_names), None)
        if legacy_field:
            spec["field_name"] = legacy_field
            spec["index_name"] = legacy_index_names.get(legacy_field, f"image_{legacy_field}_inverted_idx")
        resolved.append(spec)
    return resolved


def _public_index_spec(spec: dict[str, Any]) -> dict[str, Any]:
    public = {
        "field_name": spec["field_name"],
        "index_name": spec["index_name"],
        "index_type": spec.get("index_type", IMAGE_SCALAR_INDEX_TYPE),
    }
    params = spec.get("params")
    if params:
        public["params"] = dict(params)
    return public


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
        detail_params = detail.get("params") if isinstance(detail.get("params"), dict) else detail
        if all(str(detail_params.get(key)) == str(value) for key, value in params.items()):
            return True
    return False


def _has_matching_index_contract(
    index_details: Sequence[dict[str, Any]],
    spec: dict[str, Any],
) -> bool:
    """Compare the physical index fields needed by the fixed V2 image contract."""
    expected_field = str(spec["field_name"])
    expected_name = str(spec.get("index_name") or "")
    expected_type = str(spec.get("index_type") or "").upper()
    expected_metric = str(spec.get("metric_type") or "").upper()
    expected_params = dict(spec.get("params") or {})
    for detail in index_details:
        actual_field = str(detail.get("field_name") or "")
        actual_name = str(detail.get("index_name") or "")
        if actual_field != expected_field:
            continue
        if expected_name and actual_name and actual_name != expected_name:
            continue
        actual_type = str(detail.get("index_type") or "").upper()
        actual_metric = str(detail.get("metric_type") or "").upper()
        if actual_type != expected_type or actual_metric != expected_metric:
            continue
        detail_params = detail.get("params") if isinstance(detail.get("params"), dict) else detail
        if all(
            _index_param_matches(detail_params.get(key), value)
            for key, value in expected_params.items()
        ):
            return True
    return False


def _index_param_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual).strip().casefold() == str(expected).strip().casefold()
    return actual == expected


def _index_contract_label(spec: Mapping[str, Any]) -> str:
    return str(spec.get("index_name") or spec.get("field_name") or "unknown_index")


def _description_fields(describe: Any) -> list[Any]:
    if isinstance(describe, dict):
        return list(describe.get("fields") or [])
    return list(getattr(describe, "fields", []) or [])


def _collection_description(describe: Any) -> str:
    if isinstance(describe, dict):
        return str(describe.get("description") or "")
    return str(getattr(describe, "description", "") or "")


def _field_name(field: Any) -> Optional[str]:
    if isinstance(field, dict):
        return field.get("name")
    return getattr(field, "name", None)


def _field_type(field: Any) -> Any:
    if isinstance(field, dict):
        return field.get("type") or field.get("data_type")
    return getattr(field, "type", None) or getattr(field, "data_type", None)


def _field_type_is(field: Any, expected: Any) -> bool:
    actual = _field_type(field)
    if actual == expected:
        return True
    return str(getattr(actual, "name", actual)).upper() == str(getattr(expected, "name", expected)).upper()


def _field_dimension(field: Any) -> Optional[int]:
    if isinstance(field, dict):
        value = field.get("dim")
        params = field.get("params")
    else:
        value = getattr(field, "dim", None)
        params = getattr(field, "params", None)
    if value is None and isinstance(params, Mapping):
        value = params.get("dim")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_primary_field(field: Any) -> bool:
    if isinstance(field, dict):
        return bool(field.get("is_primary") or field.get("primary_key"))
    return bool(getattr(field, "is_primary", False) or getattr(field, "primary_key", False))


def _validate_proxy_vector_capacity(vector_field_count: int) -> None:
    configured_limit = int(getattr(settings, "MILVUS_PROXY_MAX_VECTOR_FIELD_NUM", 8) or 0)
    if configured_limit and vector_field_count > configured_limit:
        raise RuntimeError(
            "Image Milvus schema requires "
            f"{vector_field_count} vector fields, but MILVUS_PROXY_MAX_VECTOR_FIELD_NUM={configured_limit}. "
            "Set Milvus proxy.maxVectorFieldNum high enough for the configured image schema."
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


def _required_datatype(name: str) -> Any:
    if not hasattr(DataType, name):
        raise RuntimeError(f"pymilvus DataType.{name} is required by the image DOCX contract")
    return getattr(DataType, name)


def _batches(values: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
