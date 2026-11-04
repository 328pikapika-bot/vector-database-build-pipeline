import pytest
from pymilvus import DataType

from app.config import settings
from app.models.chunk import Chunk
from app.services.milvus_chunk_service import CHUNK_SCALAR_INDEX_SPECS, MilvusChunkService


_DEFAULT_EMBEDDING = object()


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, data_type, **kwargs):
        self.fields.append({"name": name, "data_type": data_type, **kwargs})


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self, *, collections=None, delete_should_fail=False, describe_fields=None, indexes=None):
        self.collections = set(collections or [])
        self.delete_should_fail = delete_should_fail
        self.schemas = {}
        self.index_params = {}
        self.index_details = {}
        for index in indexes or []:
            self._record_index(index)
        self.delete_calls = []
        self.insert_calls = []
        self.upsert_calls = []
        self.create_index_calls = []
        self.query_calls = []
        self.call_order = []
        self.loaded = []
        self.describe_fields = list(describe_fields) if describe_fields is not None else []
        self.schema_kwargs = None

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
        self.call_order.append({"op": "create_collection", "collection_name": collection_name})
        for index in index_params.indexes:
            self._record_index(index, collection_name=collection_name)

    def create_index(self, *, collection_name, index_params):
        call = {"op": "create_index", "collection_name": collection_name, "index_params": index_params}
        self.create_index_calls.append(call)
        self.call_order.append(call)
        for index in index_params.indexes:
            self._record_index(index)

    def load_collection(self, collection_name):
        self.loaded.append(collection_name)
        self.call_order.append({"op": "load_collection", "collection_name": collection_name})

    def describe_collection(self, collection_name):
        if collection_name in self.schemas:
            return {"fields": self.schemas[collection_name].fields}
        return {"fields": self.describe_fields}

    def list_indexes(self, collection_name):
        return [
            name
            for name, detail in self.index_details.items()
            if detail.get("collection_name") in {None, collection_name}
        ]

    def describe_index(self, collection_name=None, index_name=None, **_kwargs):
        return self.index_details[index_name]

    def delete(self, *, collection_name, filter):
        call = {"op": "delete", "collection_name": collection_name, "filter": filter}
        self.delete_calls.append(call)
        self.call_order.append(call)
        if self.delete_should_fail:
            raise RuntimeError("delete failed")
        return {"delete_count": 3}

    def insert(self, *, collection_name, data):
        call = {"op": "insert", "collection_name": collection_name, "data": data}
        self.insert_calls.append(call)
        self.call_order.append(call)
        return {"insert_count": len(data)}

    def upsert(self, *, collection_name, data):
        call = {"op": "upsert", "collection_name": collection_name, "data": data}
        self.upsert_calls.append(call)
        self.call_order.append(call)
        return {"upsert_count": len(data)}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return [{"id": "DOC-1_0"}, {"id": "DOC-1_1"}]

    def close(self):
        self.closed = True

    def _record_index(self, index, *, collection_name=None):
        index_name = index.get("index_name") or index["field_name"]
        self.index_details[index_name] = {
            "index_name": index_name,
            "collection_name": collection_name,
            **index,
        }


def _chunk(chunk_id="DOC-1_0", doc_id="DOC-1", embedding=_DEFAULT_EMBEDDING):
    text = "chunk text"
    return Chunk(
        chunkId=chunk_id,
        docId=doc_id,
        title=[f"Document {doc_id}"],
        keyword=["chunk"],
        accessType="OA",
        classificationCode="Q",
        year="2026",
        hasAuthorname=["Alice"],
        hasCreateOrganizationname=["NSTL"],
        language=["zh"],
        chunkIndex=int(chunk_id.rsplit("_", 1)[-1]),
        chunkType="paragraph",
        text=text,
        charStart=0,
        charEnd=len(text),
        sectionPath="Intro",
        tokenCount=2,
        embedding=[1.0] * 2560 if embedding is _DEFAULT_EMBEDDING else embedding,
    )


def test_init_chunk_collection_creates_v2_schema_and_ivfpq_cosine_index():
    fake = FakeMilvusClient()
    service = MilvusChunkService(client=fake)

    result = service.init_chunk_collection("chunks", vector_dim=2560)

    assert result["collection_name"] == "chunks"
    assert result["logical_name"] == "chunks"
    assert result["resolved_name"] == "chunks"
    assert result["created"] is True
    assert result["reused"] is False
    assert fake.schema_kwargs["auto_id"] is False
    assert fake.schema_kwargs["enable_dynamic_field"] is False
    assert fake.schema_kwargs["description"].startswith("vector-sync-target-signature:FULLTEXT")

    fields = {field["name"]: field for field in fake.schemas["chunks"].fields}
    assert fields["id"]["data_type"] == DataType.VARCHAR
    assert fields["doc_id"]["nullable"] is False
    assert fields["id"]["is_primary"] is True
    assert fields["doc_id"]["data_type"] == DataType.VARCHAR
    assert fields["title"]["data_type"] == DataType.ARRAY
    assert fields["keyword"]["data_type"] == DataType.ARRAY
    assert fields["language"]["data_type"] == DataType.ARRAY
    assert fields["language"]["nullable"] is False
    assert fields["chunkIndex"]["data_type"] == DataType.INT64
    assert fields["text"]["data_type"] == DataType.VARCHAR
    assert "nullable" not in fields["sectionPath"]
    assert "nullable" not in fields["tokenCount"]
    assert fields["text_embed"]["data_type"] == DataType.FLOAT_VECTOR
    assert fields["text_embed"]["dim"] == 2560
    for legacy_field in (
        "child_chunk_id",
        "chunkId",
        "docId",
        "embedding",
        "doc_meta",
    ):
        assert legacy_field not in fields

    indexes = fake.index_params["chunks"].indexes
    index = indexes[0]
    assert index["field_name"] == "text_embed"
    assert index["index_type"] == "IVF_PQ"
    assert index["metric_type"] == settings.FULLTEXT_CHUNK_INDEX_METRIC_TYPE
    assert index["index_type"] == "IVF_PQ"
    assert index["params"] == {
        "nlist": settings.FULLTEXT_CHUNK_IVFPQ_NLIST,
        "m": settings.FULLTEXT_CHUNK_IVFPQ_M,
        "nbits": settings.FULLTEXT_CHUNK_IVFPQ_NBITS,
    }
    expected_scalar_index_names = {spec["index_name"] for spec in CHUNK_SCALAR_INDEX_SPECS}
    assert {item["index_name"] for item in indexes[1:]} == expected_scalar_index_names
    assert "chunk_doc_id_inverted_idx" in expected_scalar_index_names
    assert "chunk_create_org_name_inverted_idx" in expected_scalar_index_names
    assert {
        "chunk_title_inverted_idx",
        "chunk_keyword_inverted_idx",
        "chunk_type_inverted_idx",
        "chunk_section_path_inverted_idx",
    }.isdisjoint(expected_scalar_index_names)
    assert fake.loaded == ["chunks"]


def test_chunk_query_nprobe_change_reuses_existing_collection(monkeypatch):
    fake = FakeMilvusClient()
    service = MilvusChunkService(client=fake)
    service.init_chunk_collection("chunks", vector_dim=2560)

    monkeypatch.setattr(
        "app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NPROBE",
        64,
    )

    target = service.resolve_collection_target(collection_name="chunks", vector_dim=2560)

    assert target.resolved_name == "chunks"
    assert target.created is False
    assert target.reused is True


def test_ensure_chunk_scalar_indexes_creates_missing_and_skips_existing():
    existing_year_index = {
        "field_name": "year",
        "index_name": "chunk_year_inverted_idx",
        "index_type": "INVERTED",
    }
    fake = FakeMilvusClient(
        collections={"chunks"},
        indexes=[_matching_chunk_vector_index(), existing_year_index],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)

    result = service.ensure_chunk_scalar_indexes("chunks")

    assert result["dry_run"] is False
    assert result["planned"] == []
    assert result["missing_fields"] == []
    assert result["skipped"] == [
        {
            "field_name": "year",
            "index_name": "chunk_year_inverted_idx",
            "reason": "already_indexed",
        }
    ]
    created_by_name = {index["index_name"]: index for index in result["created"]}
    assert set(created_by_name) == {
        spec["index_name"] for spec in CHUNK_SCALAR_INDEX_SPECS if spec["index_name"] != "chunk_year_inverted_idx"
    }
    assert len(fake.create_index_calls) == 1
    created_params = fake.create_index_calls[0]["index_params"].indexes
    assert {index["index_name"] for index in created_params} == set(created_by_name)


def test_ensure_chunk_scalar_indexes_dry_run_does_not_create_indexes():
    fake = FakeMilvusClient(
        collections={"chunks"},
        indexes=[_matching_chunk_vector_index()],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)

    result = service.ensure_chunk_scalar_indexes("chunks", dry_run=True)

    assert result["created"] == []
    assert result["skipped"] == []
    assert result["missing_fields"] == []
    assert {index["index_name"] for index in result["planned"]} == {
        spec["index_name"] for spec in CHUNK_SCALAR_INDEX_SPECS
    }
    assert fake.create_index_calls == []


def test_replace_chunks_for_document_deletes_then_inserts_and_escapes_doc_id():
    fake = FakeMilvusClient(
        collections={"chunks"},
        indexes=[_matching_chunk_vector_index()],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)
    doc_id = 'DOC-"quoted"\\slash'
    chunks = [_chunk("DOC-1_0", doc_id=doc_id), _chunk("DOC-1_1", doc_id=doc_id)]

    result = service.replace_chunks_for_document(
        "chunks",
        doc_id,
        chunks,
        source_record_update_time="2026-06-07T00:00:00Z",
        embedding_version="embed-v1",
        ingested_at="2026-06-07T03:00:00Z",
    )

    assert result["delete_result"] == {"delete_count": 3}
    assert result["insert_result"] == {"insert_count": 2}
    assert result["write_mode"] == "upsert_then_cleanup_stale"
    assert result["visibility_check"]["success"] is True
    assert [call["op"] for call in fake.call_order] == ["upsert", "delete"]

    escaped = doc_id.replace("\\", "\\\\").replace('"', '\\"')
    assert fake.delete_calls[0]["filter"] == (
        f'doc_id == "{escaped}" and '
        'id not in ["DOC-1_0","DOC-1_1"]'
    )

    inserted = fake.upsert_calls[0]["data"]
    assert inserted[0]["id"] == "DOC-1_0"
    assert inserted[0]["doc_id"] == doc_id
    assert inserted[0]["title"] == [f"Document {doc_id}"]
    assert inserted[0]["keyword"] == ["chunk"]
    assert inserted[0]["language"] == ["zh"]
    assert inserted[0]["text_embed"] == [1.0] * 2560
    for legacy_field in (
        "child_chunk_id",
        "chunkId",
        "docId",
        "embedding",
        "doc_meta",
    ):
        assert legacy_field not in inserted[0]


def test_upsert_chunks_does_not_delete_existing_chunks():
    fake = FakeMilvusClient(
        collections={"chunks"},
        indexes=[_matching_chunk_vector_index()],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)
    chunks = [_chunk("DOC-1_0"), _chunk("DOC-1_1")]

    result = service.upsert_chunks(
        "chunks",
        chunks,
        source_record_update_time="2026-06-07T00:00:00Z",
        embedding_version="embed-v1",
        ingested_at="2026-06-07T03:00:00Z",
    )

    assert result == {"upsert_count": 2}
    assert [call["op"] for call in fake.call_order] == ["upsert"]
    assert fake.delete_calls == []
    upserted = fake.upsert_calls[0]["data"]
    assert upserted[0]["id"] == "DOC-1_0"
    assert upserted[0]["doc_id"] == "DOC-1"


def test_count_visible_chunk_ids_queries_sample_ids_with_timeout_and_escaping():
    fake = FakeMilvusClient(
        collections={"chunks"},
        indexes=[_matching_chunk_vector_index()],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)

    result = service.count_visible_chunk_ids(
        "chunks",
        ["DOC-1_0", 'DOC-"quoted"_1'],
        timeout=5,
    )

    assert result == 2
    assert fake.query_calls == [
        {
            "collection_name": "chunks",
                "filter": 'id in ["DOC-1_0","DOC-\\"quoted\\"_1"]',
                "output_fields": ["id"],
            "consistency_level": "Eventually",
            "timeout": 5,
        }
    ]


def test_prepare_chunk_records_truncates_section_path_to_milvus_limit():
    fake = FakeMilvusClient(collections={"chunks"}, indexes=[_matching_chunk_vector_index()])
    service = MilvusChunkService(client=fake)
    chunk = _chunk("DOC-1_0")
    chunk.sectionPath = "A" * 1500

    result = service.upsert_chunks("chunks", [chunk])

    assert result == {"upsert_count": 1}
    upserted = fake.upsert_calls[0]["data"][0]
    assert len(upserted["sectionPath"]) == 1024
    assert upserted["sectionPath"] == "A" * 1024


def test_replace_chunks_cleanup_failure_keeps_new_upserted_chunks():
    fake = FakeMilvusClient(
        collections={"chunks"},
        delete_should_fail=True,
        indexes=[_matching_chunk_vector_index()],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)

    with pytest.raises(RuntimeError, match="delete failed"):
        service.replace_chunks_for_document("chunks", "DOC-1", [_chunk()])

    assert [call["op"] for call in fake.call_order] == ["upsert", "delete"]
    assert len(fake.upsert_calls) == 1
    assert fake.insert_calls == []


def test_replace_chunks_empty_list_never_deletes_existing_document_chunks():
    fake = FakeMilvusClient(
        collections={"chunks"},
        indexes=[_matching_chunk_vector_index()],
        describe_fields=_chunk_schema_fields(),
    )
    service = MilvusChunkService(client=fake)

    with pytest.raises(ValueError, match="not a delete instruction"):
        service.replace_chunks_for_document("chunks", "DOC-1", [])

    assert fake.call_order == []


@pytest.mark.parametrize(
    "bad_chunk, message",
    [
        (_chunk(embedding=None), "missing text_embed"),
        (_chunk(embedding=[1.0] * 10), "dimension mismatch"),
        (_chunk(doc_id="OTHER"), "doc_id mismatch"),
    ],
)
def test_replace_chunks_for_document_invalid_chunks_do_not_delete_or_insert(bad_chunk, message):
    fake = FakeMilvusClient(collections={"chunks"}, indexes=[_matching_chunk_vector_index()])
    service = MilvusChunkService(client=fake)

    with pytest.raises(ValueError, match=message):
        service.replace_chunks_for_document("chunks", "DOC-1", [bad_chunk])

    assert fake.call_order == []


def test_existing_chunk_collection_with_mismatched_index_rolls_to_suffixed_target(monkeypatch):
    existing_index = {
        "field_name": "text_embed",
        "index_name": "text_embed",
        "index_type": "IVF_PQ",
        "metric_type": "COSINE",
        "params": {"nlist": 64, "m": 64, "nbits": 8},
    }
    fake = FakeMilvusClient(collections={"chunks"}, indexes=[existing_index], describe_fields=_chunk_schema_fields())
    service = MilvusChunkService(client=fake)

    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_INDEX_TYPE", "IVFPQ")
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NLIST", 512)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_M", 32)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NBITS", 8)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NPROBE", 16)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_INDEX_METRIC_TYPE", "COSINE")

    target = service.resolve_collection_target(collection_name="chunks", vector_dim=2560)

    assert target.logical_name == "chunks"
    assert target.resolved_name == "chunks__ivfpq_nlist512_m32_nbits8_cosine"
    assert target.created is True
    assert target.reused is False
    assert fake.has_collection("chunks__ivfpq_nlist512_m32_nbits8_cosine")


def test_existing_index_target_with_legacy_schema_adds_schema_suffix(monkeypatch):
    candidate = "chunks__ivfpq_nlist512_m32_nbits8_cosine"
    ivfpq_index = {
        "field_name": "text_embed",
        "index_name": "text_embed",
        "index_type": "IVF_PQ",
        "metric_type": "COSINE",
        "params": {"nlist": 512, "m": 32, "nbits": 8},
        "collection_name": candidate,
    }
    fake = FakeMilvusClient(
        collections={"chunks", candidate},
        indexes=[ivfpq_index],
        describe_fields=[
            {"name": "child_chunk_id", "data_type": DataType.VARCHAR},
            {"name": "doc_id", "data_type": DataType.VARCHAR},
            {"name": "text_embed", "data_type": DataType.FLOAT_VECTOR},
        ],
    )
    service = MilvusChunkService(client=fake)

    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_INDEX_TYPE", "IVFPQ")
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NLIST", 512)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_M", 32)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NBITS", 8)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_INDEX_METRIC_TYPE", "COSINE")

    target = service.resolve_collection_target(collection_name="chunks", vector_dim=2560)

    assert target.resolved_name == f"{candidate}__schema_v2"
    assert target.created is True


def test_ivfpq_config_uses_milvus_ivf_pq_index_name(monkeypatch):
    fake = FakeMilvusClient()
    service = MilvusChunkService(client=fake)

    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_INDEX_TYPE", "IVFPQ")
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NLIST", 512)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_M", 64)
    monkeypatch.setattr("app.services.collection_targets.settings.FULLTEXT_CHUNK_IVFPQ_NBITS", 8)

    service.init_chunk_collection("chunks", vector_dim=2560)

    index = fake.index_params["chunks"].indexes[0]
    assert index["index_type"] == "IVF_PQ"
    assert index["params"] == {"nlist": 512, "m": 64, "nbits": 8}


def _chunk_schema_fields():
    return [
        {"name": "id", "data_type": DataType.VARCHAR, "is_primary": True},
        {"name": "doc_id", "data_type": DataType.VARCHAR},
        {"name": "title", "data_type": DataType.ARRAY},
        {"name": "keyword", "data_type": DataType.ARRAY},
        {"name": "accessType", "data_type": DataType.VARCHAR},
        {"name": "classificationCode", "data_type": DataType.VARCHAR},
        {"name": "year", "data_type": DataType.VARCHAR},
        {"name": "hasAuthorname", "data_type": DataType.ARRAY},
        {"name": "hasCreateOrganizationname", "data_type": DataType.ARRAY},
        {"name": "language", "data_type": DataType.ARRAY},
        {"name": "chunkIndex", "data_type": DataType.INT64},
        {"name": "chunkType", "data_type": DataType.VARCHAR},
        {"name": "text", "data_type": DataType.VARCHAR},
        {"name": "charStart", "data_type": DataType.INT64},
        {"name": "charEnd", "data_type": DataType.INT64},
        {"name": "sectionPath", "data_type": DataType.VARCHAR},
        {"name": "tokenCount", "data_type": DataType.INT64},
        {"name": "text_embed", "data_type": DataType.FLOAT_VECTOR},
    ]


def _matching_chunk_vector_index():
    return {
        "field_name": "text_embed",
        "index_name": "text_embed",
        "index_type": "IVF_PQ",
        "metric_type": settings.FULLTEXT_CHUNK_INDEX_METRIC_TYPE,
        "params": {
            "nlist": settings.FULLTEXT_CHUNK_IVFPQ_NLIST,
            "m": settings.FULLTEXT_CHUNK_IVFPQ_M,
            "nbits": settings.FULLTEXT_CHUNK_IVFPQ_NBITS,
        },
    }
