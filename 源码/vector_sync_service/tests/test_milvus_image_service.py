from __future__ import annotations

import numpy as np
import pytest
from pymilvus import DataType

from app.contracts.docx_contract import IMAGE_FIELDS
from app.models.image import ImageRecord
from app.services.milvus_image_service import (
    IMAGE_FOUR_VECTOR_FIELDS,
    IMAGE_SCALAR_INDEX_SPECS,
    IMAGE_SCHEMA_MODE_HISTORY_V2,
    IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    MilvusImageService,
    TEXT_VECTOR_DIM,
    VISUAL_TEXT_VECTOR_DIM,
    VISUAL_VECTOR_DIM,
    validate_writable_image_record,
)


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict] = []
        self.functions: list[dict] = []

    def add_field(self, name, data_type, **kwargs):
        self.fields.append({"name": name, "data_type": data_type, **kwargs})

    def add_function(self, function):
        self.functions.append(function)


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self, *, collections=None, indexes=None, load_error=None) -> None:
        self.collections = set(collections or [])
        self.schemas: dict[str, FakeSchema] = {}
        self.index_params: dict[str, FakeIndexParams] = {}
        self.index_details = {
            index.get("index_name") or index["field_name"]: dict(index)
            for index in indexes or []
        }
        self.insert_calls: list[dict] = []
        self.upsert_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.create_index_calls: list[dict] = []
        self.loaded: list[str] = []
        self.load_error = load_error

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_schema(self, **kwargs):
        self.schema_kwargs = kwargs
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, *, collection_name, schema, index_params):
        self.collections.add(collection_name)
        self.schemas[collection_name] = schema
        self.index_params[collection_name] = index_params
        for index in index_params.indexes:
            self.index_details[index.get("index_name") or index["field_name"]] = dict(index)

    def create_index(self, *, collection_name, index_params):
        self.create_index_calls.append({"collection_name": collection_name, "index_params": index_params})
        for index in index_params.indexes:
            self.index_details[index.get("index_name") or index["field_name"]] = dict(index)

    def load_collection(self, collection_name):
        self.loaded.append(collection_name)
        if self.load_error is not None:
            raise self.load_error

    def describe_collection(self, collection_name):
        schema = self.schemas.get(collection_name)
        if schema is not None:
            return {"fields": schema.fields}
        return {"fields": _history_v2_schema_fields()}

    def list_indexes(self, _collection_name):
        return list(self.index_details)

    def describe_index(self, collection_name=None, index_name=None, **_kwargs):
        return self.index_details[index_name]

    def insert(self, *, collection_name, data):
        self.insert_calls.append({"collection_name": collection_name, "data": data})
        return {"insert_count": len(data)}

    def upsert(self, *, collection_name, data):
        self.upsert_calls.append({"collection_name": collection_name, "data": data})
        return {"insert_count": len(data)}

    def delete(self, *, collection_name, filter):
        self.delete_calls.append({"collection_name": collection_name, "filter": filter})
        return {"delete_count": 1}

    def query(self, **_kwargs):
        rows = self.upsert_calls[-1]["data"] if self.upsert_calls else []
        return [{"id": row["id"]} for row in rows]


def _history_v2_schema_fields():
    fields = [{"name": name, "data_type": DataType.VARCHAR} for name in IMAGE_FIELDS]
    for field in fields:
        if field["name"] == "id":
            field["is_primary"] = True
        if field["name"] in {"doc_meta", "raw_img_meta", "img_meta"}:
            field["data_type"] = DataType.JSON
    return fields


def _image(*, img_id="image-1", doc_id="doc-1", caption=True, context=True, **overrides):
    value = ImageRecord(
        img_id=img_id,
        docId=doc_id,
        title=["Document title"],
        keyword=["keyword"],
        accessType="OA",
        classificationCode="Q",
        year="2026",
        hasAuthorname=["Author"],
        hasCreateOrganizationname=["NSTL"],
        language=["en"],
        sysDataStatus="ACTIVE",
        hasSourcesysSourceId=["source"],
        hasSourceidSet=["journal"],
        url="/fulltext/doc-1/figures/image-1.jpg",
        name="image-1.jpg",
        sequence=1,
        width=640,
        height=480,
        fileType="jpg",
        size=1024,
        raw_img_meta={
            "id": img_id,
            "url": "/fulltext/doc-1/figures/image-1.jpg",
            "name": "image-1.jpg",
            "sequence": "1",
            "width": 640,
            "height": 480,
            "fileType": "jpg",
            "size": 1024,
            "title": ["Figure caption"],
        },
        img_meta={"page": 1, "bbox": "0,0,10,10", "caption": "Figure caption"},
        context_before="Before",
        context_after="After",
        p_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        visual_embed=[1.0] * VISUAL_VECTOR_DIM,
        caption_embed=[1.0] * TEXT_VECTOR_DIM if caption else None,
        context_embed=[1.0] * TEXT_VECTOR_DIM if context else None,
        text_embed=[1.0] * TEXT_VECTOR_DIM,
        visual_text_embed=[1.0] * VISUAL_TEXT_VECTOR_DIM,
    )
    return value.model_copy(update=overrides)


def test_history_v2_schema_is_exact_and_uses_ivfpq_for_dense_vectors():
    client = FakeMilvusClient()
    result = MilvusImageService(client=client, schema_mode=IMAGE_SCHEMA_MODE_HISTORY_V2).init_image_collection("images")

    assert result["collection_name"] == "images"
    assert result["created"] is True
    fields = {field["name"]: field for field in client.schemas["images"].fields}
    assert tuple(fields) == IMAGE_FIELDS
    assert fields["id"]["is_primary"] is True
    assert fields["doc_id"]["data_type"] == DataType.VARCHAR
    for name in ("doc_meta", "raw_img_meta", "img_meta"):
        assert fields[name]["data_type"] == DataType.JSON
    assert fields["p_hash"] == {"name": "p_hash", "data_type": DataType.BINARY_VECTOR, "dim": 64}
    assert fields["text_sparse_vec"]["data_type"] == DataType.SPARSE_FLOAT_VECTOR
    assert "nullable" not in fields["text_sparse_vec"]
    assert client.schemas["images"].functions == []

    indexes = client.index_params["images"].indexes
    assert indexes[0] == {"field_name": "p_hash", "index_type": "BIN_FLAT", "metric_type": "HAMMING"}
    dense = [index for index in indexes if index["field_name"] in {
        "visual_embed", "caption_embed", "context_embed", "text_embed", "visual_text_embed"
    }]
    assert [index["field_name"] for index in dense] == [
        "visual_embed", "caption_embed", "context_embed", "text_embed", "visual_text_embed"
    ]
    assert {index["index_type"] for index in dense} == {"IVF_PQ"}
    assert {index["metric_type"] for index in dense} == {"IP"}
    assert {
        "field_name": "text_sparse_vec",
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    } in indexes


def test_four_vector_schema_excludes_interpretation_mixed_and_sparse_fields():
    client = FakeMilvusClient()
    service = MilvusImageService(
        client=client,
        schema_mode=IMAGE_SCHEMA_MODE_VISUAL_CAPTION_CONTEXT_4,
    )

    service.init_image_collection("images")

    fields = {field["name"]: field for field in client.schemas["images"].fields}
    assert tuple(fields) == IMAGE_FOUR_VECTOR_FIELDS
    assert {"text", "text_embed", "visual_text_embed", "text_sparse_vec"}.isdisjoint(fields)
    assert [index["field_name"] for index in client.index_params["images"].indexes if index["field_name"] in {
        "visual_embed", "caption_embed", "context_embed"
    }] == ["visual_embed", "caption_embed", "context_embed"]
    assert "text_sparse_vec" not in {
        index["field_name"] for index in client.index_params["images"].indexes
    }

    service.insert_images("images", [_image()])
    record = client.insert_calls[0]["data"][0]
    assert tuple(record) == IMAGE_FOUR_VECTOR_FIELDS
    assert {"text", "text_embed", "visual_text_embed", "text_sparse_vec"}.isdisjoint(record)


def test_history_v2_sparse_placeholder_loads_without_bm25():
    client = FakeMilvusClient()

    result = MilvusImageService(client=client).init_image_collection("images")

    assert result["collection_loaded"] is True
    assert client.loaded == ["images"]


def test_configured_image_collection_reuses_the_exact_existing_name():
    client = FakeMilvusClient()
    service = MilvusImageService(client=client, schema_mode=IMAGE_SCHEMA_MODE_HISTORY_V2)

    created = service.init_image_collection("images")
    reused = service.init_image_collection("images")

    assert created["created"] is True
    assert reused["created"] is False
    assert reused["reused"] is True
    assert reused["resolved_name"] == "images"
    assert client.collections == {"images"}
    assert client.loaded == ["images", "images"]


def test_configured_image_collection_rejects_incompatible_name_without_suffix_creation():
    client = FakeMilvusClient(collections={"images"})
    service = MilvusImageService(client=client, schema_mode=IMAGE_SCHEMA_MODE_HISTORY_V2)

    with pytest.raises(ValueError, match="field p_hash must use"):
        service.init_image_collection("images")

    assert client.collections == {"images"}
    assert not any(name.startswith("images__") for name in client.collections)


def test_history_v2_record_keeps_json_metadata_and_moves_doc_id_top_level():
    client = FakeMilvusClient(collections={"images"})
    service = MilvusImageService(client=client, schema_mode=IMAGE_SCHEMA_MODE_HISTORY_V2)

    assert service.insert_images("images", [_image()]) == {"insert_count": 1}
    record = client.insert_calls[0]["data"][0]

    assert tuple(record) == IMAGE_FIELDS
    assert record["id"] == "image-1"
    assert record["doc_id"] == "doc-1"
    assert "docId" not in record["doc_meta"]
    assert record["doc_meta"]["title"] == ["Document title"]
    assert record["raw_img_meta"]["id"] == "image-1"
    assert record["raw_img_meta"]["url"] == "/fulltext/doc-1/figures/image-1.jpg"
    assert record["img_meta"]["img_id"] == "image-1"
    assert record["img_meta"]["page"] == 1
    for field_name, dim in (("visual_embed", 4096), ("caption_embed", 2560), ("context_embed", 2560), ("text_embed", 2560), ("visual_text_embed", 4096)):
        assert isinstance(record[field_name], np.ndarray)
        assert record[field_name].dtype == np.float16
        assert len(record[field_name]) == dim


def test_missing_caption_context_and_reserved_text_fields_use_inert_placeholders():
    client = FakeMilvusClient(collections={"images"})
    service = MilvusImageService(client=client, schema_mode=IMAGE_SCHEMA_MODE_HISTORY_V2)

    image = _image(caption=False, context=False, text=None, text_embed=None, visual_text_embed=None, text_sparse_vec=None)
    service.insert_images("images", [image])
    record = client.insert_calls[0]["data"][0]

    assert record["caption_embed"][0] > 0
    assert record["context_embed"][0] > 0
    assert record["text"] == "__reserved_image_text__"
    assert record["text_embed"][0] > 0
    assert record["visual_text_embed"][0] > 0
    assert record["text_sparse_vec"] == {0: 1e-6}


def test_image_document_cleanup_uses_top_level_doc_id():
    client = FakeMilvusClient(collections={"images"})
    service = MilvusImageService(client=client)

    result = service.replace_images_for_document("images", "doc-1", [_image()], complete_snapshot=True)

    assert result["insert_result"] == {"insert_count": 1}
    assert client.delete_calls[0]["filter"].startswith('doc_id == "doc-1"')
    assert "doc_meta" not in client.delete_calls[0]["filter"]


def test_scalar_indexes_target_json_paths_and_top_level_doc_id():
    client = FakeMilvusClient()
    MilvusImageService(client=client).init_image_collection("images")
    indexes = {index.get("index_name"): index for index in client.index_params["images"].indexes}

    assert indexes["image_doc_id_inverted_idx"]["field_name"] == "doc_id"
    assert indexes["image_img_id_json_inverted_idx"] == {
        "field_name": "img_meta",
        "index_name": "image_img_id_json_inverted_idx",
        "index_type": "INVERTED",
        "params": {"json_path": 'img_meta["img_id"]', "json_cast_type": "varchar"},
    }
    assert indexes["image_year_inverted_idx"]["params"] == {
        "json_path": 'doc_meta["year"]', "json_cast_type": "varchar"
    }
    assert {spec["index_name"] for spec in IMAGE_SCALAR_INDEX_SPECS} <= set(indexes)


def test_existing_lightweight_schema_is_rejected_without_mutation():
    fields = _history_v2_schema_fields() + [{"name": "relative_url", "data_type": DataType.VARCHAR}]
    client = FakeMilvusClient(collections={"images"})
    client.describe_collection = lambda _name: {"fields": fields}

    with pytest.raises(ValueError, match="unsupported top-level image fields"):
        MilvusImageService(client=client).ensure_image_scalar_indexes("images", dry_run=True)


def test_raw_image_id_must_match_image_primary_key():
    image = _image()
    image.raw_img_meta["id"] = "other-image"

    with pytest.raises(ValueError, match="does not match upstream id"):
        validate_writable_image_record(image, expected_doc_id="doc-1")
