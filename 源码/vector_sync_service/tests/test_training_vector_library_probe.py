import json

import numpy as np
import pytest

from app.config import settings
from app.services.training_vector_library_probe import (
    json_safe,
    probe_training_vector_library,
)


class FakeEmbedder:
    def __init__(self, vector=None, *, success=True):
        self.vector = [0.1] * settings.VECTOR_DIM if vector is None else vector
        self.success = success
        self.closed = False

    async def embed_single_detailed(self, text):
        self.text = text
        return {
            "success": self.success,
            "vector": self.vector if self.success else None,
            "error_code": None if self.success else "EMBEDDING_FAILED",
            "retryable": False,
        }

    async def close(self):
        self.closed = True


class FakeMilvusClient:
    def __init__(self, *, missing_collections=None):
        self.missing_collections = set(missing_collections or [])
        self.search_calls = []
        self.query_calls = []
        self.mutations = []
        self.closed = False
        self.samples = {
            "images": {
                "id": "IMG-1",
                "visual_embed": [np.ones(4096, dtype=np.float16).tobytes()],
                "visual_text_embed": [np.ones(4096, dtype=np.float16).tobytes()],
                "p_hash": [b"\x01\x02\x03\x04\x05\x06\x07\x08"],
            }
        }

    def has_collection(self, collection_name):
        return collection_name not in self.missing_collections

    def describe_collection(self, collection_name):
        if collection_name == "documents":
            return {
                "fields": [
                    _field("id", "VARCHAR", is_primary=True),
                    _field("title", "ARRAY"),
                    _field("abstract", "ARRAY"),
                    _field("keyword", "ARRAY"),
                    _field("year", "VARCHAR"),
                    _field("language", "ARRAY"),
                    _field("title_embed", "FLOAT_VECTOR", dim=2560),
                    _field("abstract_embed", "FLOAT_VECTOR", dim=2560),
                    _field("keyword_embed", "FLOAT_VECTOR", dim=2560),
                ]
            }
        if collection_name == "chunks":
            return {
                "fields": [
                    _field("id", "VARCHAR", is_primary=True),
                    _field("doc_id", "VARCHAR"),
                    _field("text", "VARCHAR"),
                    _field("chunkIndex", "INT64"),
                    _field("sectionPath", "VARCHAR"),
                    _field("language", "ARRAY"),
                    _field("text_embed", "FLOAT_VECTOR", dim=2560),
                ]
            }
        if collection_name == "images":
            return {
                "fields": [
                    _field("id", "VARCHAR", is_primary=True),
                    _field("doc_id", "VARCHAR"),
                    _field("doc_meta", "JSON"),
                    _field("raw_img_meta", "JSON"),
                    _field("img_meta", "JSON"),
                    _field("text", "VARCHAR"),
                    _field("p_hash", "BINARY_VECTOR", dim=64),
                    _field("visual_embed", "FLOAT16_VECTOR", dim=4096),
                    _field("caption_embed", "FLOAT16_VECTOR", dim=2560),
                    _field("context_embed", "FLOAT16_VECTOR", dim=2560),
                    _field("text_embed", "FLOAT16_VECTOR", dim=2560),
                    _field("visual_text_embed", "FLOAT16_VECTOR", dim=4096),
                    _field("text_sparse_vec", "SPARSE_FLOAT_VECTOR"),
                ]
            }
        raise AssertionError(f"unexpected collection: {collection_name}")

    def get_collection_stats(self, *, collection_name):
        return {"row_count": {"documents": 10, "chunks": 20, "images": 5}[collection_name]}

    def query(self, *, collection_name, filter, output_fields, limit=None):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "filter": filter,
                "output_fields": output_fields,
                "limit": limit,
            }
        )
        sample = self.samples.get(collection_name, {})
        return [{field: sample.get(field) for field in output_fields}]

    def search(self, *, collection_name, data, anns_field, limit, search_params, output_fields, filter=None):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "data": data,
                "anns_field": anns_field,
                "limit": limit,
                "search_params": search_params,
                "output_fields": output_fields,
                "filter": filter,
            }
        )
        if collection_name == "documents":
            return [
                [
                    {
                        "distance": 0.72,
                        "entity": {
                            "id": "DOC-1",
                            "title": ["Rheumatoid arthritis"],
                            "abstract": ["T cell receptor study"],
                            "keyword": ["T cells"],
                            "year": "2026",
                            "language": ["en"],
                        },
                    }
                ]
            ]
        if collection_name == "chunks":
            return [
                [
                    {
                        "distance": 0.68,
                        "entity": {
                            "id": "DOC-1_0",
                            "doc_id": "DOC-1",
                            "text": "T cell receptor chunk",
                            "chunkIndex": 0,
                            "sectionPath": "body",
                            "language": ["en"],
                        },
                    }
                ]
            ]
        if collection_name == "images":
            distance = 0.0 if anns_field == "p_hash" else 0.6
            if anns_field == "text_embed":
                distance = 0.0
            return [
                [
                    {
                        "distance": distance,
                        "entity": {
                            "id": "IMG-1",
                            "doc_id": "DOC-1",
                            "doc_meta": {
                                "title": ["Image document"],
                                "year": "2026",
                                "language": ["en"],
                            },
                            "raw_img_meta": {"id": "IMG-1", "url": "/images/1.jpg"},
                            "img_meta": {"img_id": "IMG-1", "caption": "Image caption"},
                            "text": "__reserved_image_text__",
                        },
                    }
                ]
            ]
        raise AssertionError(f"unexpected collection: {collection_name}")

    def insert(self, *_args, **_kwargs):
        self.mutations.append("insert")
        raise AssertionError("probe must not insert")

    def delete(self, *_args, **_kwargs):
        self.mutations.append("delete")
        raise AssertionError("probe must not delete")

    def flush(self, *_args, **_kwargs):
        self.mutations.append("flush")
        raise AssertionError("probe must not flush")

    def load_collection(self, *_args, **_kwargs):
        self.mutations.append("load_collection")
        raise AssertionError("probe must not load collections")

    def create_collection(self, *_args, **_kwargs):
        self.mutations.append("create_collection")
        raise AssertionError("probe must not create collections")

    def close(self):
        self.closed = True


def _field(name, data_type, *, is_primary=False, dim=None):
    field = {"name": name, "data_type": data_type}
    if is_primary:
        field["is_primary"] = True
    if dim is not None:
        field["dim"] = dim
    return field


@pytest.mark.asyncio
async def test_probe_runs_expected_read_only_searches_and_returns_json_safe_result():
    client = FakeMilvusClient()
    embedder = FakeEmbedder()

    result = await probe_training_vector_library(
        query="rheumatoid arthritis",
        top_k=2,
        milvus_client=client,
        embedder=embedder,
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["mutation_allowed"] is False
    assert client.mutations == []
    assert result["collections"]["document"]["ok"] is True
    assert result["collections"]["chunk"]["ok"] is True
    assert result["collections"]["image"]["ok"] is True
    assert {call["anns_field"] for call in client.search_calls if call["collection_name"] == "documents"} == {
        "title_embed",
        "abstract_embed",
        "keyword_embed",
    }
    assert {call["anns_field"] for call in client.search_calls if call["collection_name"] == "chunks"} == {
        "text_embed"
    }

    image_text_calls = [
        call
        for call in client.search_calls
        if call["collection_name"] == "images"
        and call["anns_field"] in {"caption_embed", "context_embed"}
    ]
    assert image_text_calls
    assert all(isinstance(call["data"][0], np.ndarray) for call in image_text_calls)
    assert all(call["data"][0].dtype == np.float16 for call in image_text_calls)
    assert all(call["search_params"]["metric_type"] == "IP" for call in image_text_calls)
    assert all(call["search_params"]["params"] == {"nprobe": 16} for call in image_text_calls)
    assert all(call["filter"] == 'img_meta["quality_status"] == "READY_CAPTION_CONTEXT"' for call in image_text_calls)
    assert not [
        call
        for call in client.search_calls
        if call["collection_name"] == "images" and call["anns_field"] == "text_embed"
    ]

    image_self_calls = [
        call
        for call in client.search_calls
        if call["collection_name"] == "images"
        and call["anns_field"] in {"visual_embed", "p_hash"}
    ]
    assert {call["anns_field"] for call in image_self_calls} == {
        "visual_embed",
        "p_hash",
    }
    assert next(
        call for call in image_self_calls if call["anns_field"] == "visual_embed"
    )["search_params"]["params"] == {"nprobe": 16}
    assert next(
        call for call in image_self_calls if call["anns_field"] == "p_hash"
    )["search_params"]["params"] == {}
    assert all(isinstance(call["data"][0], bytes) for call in image_self_calls)
    assert result["warnings"] == []
    assert result["collections"]["image"]["reserved_fields"]["text"]["status"] == "reserved_not_probed"
    assert result["collections"]["image"]["reserved_fields"]["text_embed"]["present"] is True
    assert result["collections"]["image"]["reserved_fields"]["visual_text_embed"]["present"] is True
    assert result["collections"]["image"]["reserved_fields"]["text_sparse_vec"]["present"] is True
    json.dumps(json_safe(result), ensure_ascii=False)


@pytest.mark.asyncio
async def test_probe_uses_collection_specific_search_params(monkeypatch):
    client = FakeMilvusClient()

    monkeypatch.setattr("app.services.collection_targets.settings.DOCUMENT_INDEX_TYPE", "IVFPQ")
    monkeypatch.setattr("app.services.collection_targets.settings.DOCUMENT_IVFPQ_NPROBE", 77)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_INDEX_TYPE", "IVFPQ")
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NPROBE", 19)

    await probe_training_vector_library(
        milvus_client=client,
        embedder=FakeEmbedder(),
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
    )

    document_calls = [call for call in client.search_calls if call["collection_name"] == "documents"]
    chunk_calls = [call for call in client.search_calls if call["collection_name"] == "chunks"]
    image_calls = [call for call in client.search_calls if call["collection_name"] == "images"]

    assert document_calls
    assert all(call["search_params"]["params"] == {"nprobe": 77} for call in document_calls)
    assert chunk_calls
    assert all(call["search_params"]["params"] == {"nprobe": 19} for call in chunk_calls)
    assert image_calls
    assert all(call["search_params"]["metric_type"] in {"IP", "HAMMING"} for call in image_calls)


@pytest.mark.asyncio
async def test_probe_fails_cleanly_when_a_collection_is_missing():
    client = FakeMilvusClient(missing_collections={"images"})
    result = await probe_training_vector_library(
        milvus_client=client,
        embedder=FakeEmbedder(),
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
    )

    assert result["success"] is False
    assert result["collections"]["image"]["exists"] is False
    assert any(error["code"] == "COLLECTION_MISSING" for error in result["errors"])
    assert not [call for call in client.search_calls if call["collection_name"] == "images"]
    assert client.mutations == []


@pytest.mark.asyncio
async def test_probe_reports_embedding_dimension_mismatch_and_keeps_image_self_probe_read_only():
    client = FakeMilvusClient()
    result = await probe_training_vector_library(
        milvus_client=client,
        embedder=FakeEmbedder(vector=[0.1, 0.2, 0.3]),
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
    )

    assert result["success"] is False
    assert any(error["code"] == "EMBEDDING_DIMENSION_MISMATCH" for error in result["errors"])
    assert not [call for call in client.search_calls if call["collection_name"] == "documents"]
    assert not [call for call in client.search_calls if call["collection_name"] == "chunks"]
    assert {call["anns_field"] for call in client.search_calls if call["collection_name"] == "images"} == {
        "visual_embed",
        "p_hash",
    }
    assert client.mutations == []
