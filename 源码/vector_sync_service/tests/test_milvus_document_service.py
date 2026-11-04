import pytest
from pymilvus import DataType

from app.contracts.docx_contract import (
    DOCUMENT_SCALAR_FIELD_SPECS,
    DOCUMENT_SCALAR_INDEX_FIELDS,
    DOCUMENT_VECTOR_FIELDS,
)
from app.models.base.article import Article
from app.services.milvus_service import (
    MilvusDynamicService,
    _field_meta,
    _normalize_value_for_field,
)


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, data_type, **kwargs):
        self.fields.append({"name": name, "data_type": data_type, **kwargs})


def test_array_varchar_limit_is_read_from_milvus_field_params():
    meta = _field_meta(
        {
            "name": "abstract",
            "type": DataType.ARRAY,
            "element_type": DataType.VARCHAR,
            "params": {"max_length": 5, "max_capacity": 4096},
        }
    )

    normalized = _normalize_value_for_field(
        "abstract",
        ["abcdefghijk"],
        DataType.ARRAY,
        meta,
    )

    assert meta["max_length"] == 5
    assert normalized == ["abcde", "fghij", "k"]

    multibyte_normalized = _normalize_value_for_field(
        "abstract",
        ["界" * 4],
        DataType.ARRAY,
        {"element_type": DataType.VARCHAR, "max_length": 10},
    )
    assert multibyte_normalized == ["界" * 3, "界"]
    assert all(len(item.encode("utf-8")) <= 10 for item in multibyte_normalized)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self):
        self.collections = set()
        self.schemas = {}
        self.index_params = {}
        self.index_details = {}
        self.loaded = []
        self.insert_calls = []
        self.upsert_calls = []
        self.delete_calls = []
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
        for index in index_params.indexes:
            self._record_index(index)

    def load_collection(self, collection_name):
        self.loaded.append(collection_name)

    def describe_collection(self, collection_name):
        return {"fields": self.schemas[collection_name].fields}

    def list_indexes(self, collection_name):
        return [
            name
            for name, detail in self.index_details.items()
            if detail.get("collection_name") == collection_name
        ]

    def describe_index(self, collection_name=None, index_name=None, **_kwargs):
        return self.index_details[index_name]

    def insert(self, *, collection_name, data):
        self.insert_calls.append({"collection_name": collection_name, "data": data})
        return {"insert_count": len(data)}

    def upsert(self, *, collection_name, data):
        call = {"collection_name": collection_name, "data": data}
        self.upsert_calls.append(call)
        self.insert_calls.append(call)
        return {"insert_count": len(data)}

    def query(self, **_kwargs):
        if not self.upsert_calls:
            return []
        return [{"id": row["id"]} for row in self.upsert_calls[-1]["data"]]

    def delete(self, *, collection_name, filter):
        self.delete_calls.append({"collection_name": collection_name, "filter": filter})
        return {"delete_count": 1}

    def _record_index(self, index):
        index_name = index.get("index_name") or index["field_name"]
        self.index_details[index_name] = {
            "index_name": index_name,
            "collection_name": next(
                (name for name, params in self.index_params.items() if index in params.indexes),
                None,
            ),
            **index,
        }


def _service(fake: FakeMilvusClient) -> MilvusDynamicService:
    service = MilvusDynamicService.__new__(MilvusDynamicService)
    service.client = fake
    return service


def test_init_collection_creates_document_id_scalar_schema_and_indexes():
    fake = FakeMilvusClient()
    service = _service(fake)

    result = service.init_collection("documents", vector_dim=3)

    assert result["collection_name"] == "documents"
    assert result["logical_name"] == "documents"
    assert result["resolved_name"] == "documents"
    assert result["created"] is True
    assert result["reused"] is False

    assert fake.schema_kwargs["auto_id"] is False
    assert fake.schema_kwargs["enable_dynamic_field"] is False
    assert fake.schema_kwargs["description"].startswith(
        "vector-sync-target-signature:DOCUMENT"
    )
    fields = {field["name"]: field for field in fake.schemas["documents"].fields}
    assert fields["id"]["data_type"] == DataType.VARCHAR
    assert fields["id"]["is_primary"] is True
    assert fields["id"]["auto_id"] is False
    assert "ID" not in fields
    assert "sysId" not in fields
    assert "dynamic_fields" not in fields
    assert fields["hasFullTextFileid"]["data_type"] == DataType.ARRAY
    assert fields["hasAuthorname"]["data_type"] == DataType.ARRAY
    for vector_field in DOCUMENT_VECTOR_FIELDS:
        assert fields[vector_field]["data_type"] == DataType.FLOAT_VECTOR
        assert fields[vector_field]["dim"] == 3

    indexed_fields = {index["field_name"] for index in fake.index_params["documents"].indexes}
    assert set(DOCUMENT_VECTOR_FIELDS).issubset(indexed_fields)
    assert set(DOCUMENT_SCALAR_INDEX_FIELDS).issubset(indexed_fields)
    vector_indexes = [
        index
        for index in fake.index_params["documents"].indexes
        if index["field_name"] in DOCUMENT_VECTOR_FIELDS
    ]
    assert {index["index_type"] for index in vector_indexes} == {"IVF_PQ"}
    assert {tuple(sorted(index["params"].items())) for index in vector_indexes} == {
        (("m", 64), ("nbits", 8), ("nlist", 4096))
    }
    assert fake.loaded == ["documents"]


def test_document_collection_schema_stays_under_milvus_field_limit():
    field_count = 2 + len(DOCUMENT_SCALAR_FIELD_SPECS) + len(DOCUMENT_VECTOR_FIELDS)

    assert field_count <= 200


def test_existing_legacy_document_schema_fails_fast():
    fake = FakeMilvusClient()
    fake.collections.add("legacy")
    fake.schemas["legacy"] = FakeSchema()
    fake.schemas["legacy"].fields = [
        {"name": "id", "data_type": DataType.INT64, "is_primary": True},
        {"name": "sysId", "data_type": DataType.VARCHAR},
        {"name": "dynamic_fields", "data_type": DataType.JSON},
    ]
    service = _service(fake)

    result = service.init_collection("legacy", vector_dim=3)

    assert result["logical_name"] == "legacy"
    assert result["resolved_name"] == "legacy__ivfpq_nlist4096_m64_nbits8_cosine"
    assert result["created"] is True
    assert result["reused"] is False
    assert fake.has_collection("legacy__ivfpq_nlist4096_m64_nbits8_cosine")


def test_process_and_insert_writes_id_and_top_level_scalars_without_dynamic_fields():
    fake = FakeMilvusClient()
    service = _service(fake)
    service.init_collection("documents", vector_dim=3)
    article = Article(
        sysId="DOC-1",
        cls="Article",
        title=["Title"],
        abstract=["Abstract"],
        keyword=["kw"],
        year="2026",
        title_embed=[1.0, 0.0, 0.0],
        abstract_embed=[0.0, 1.0, 0.0],
        keyword_embed=[0.0, 0.0, 1.0],
        hasFullTextFileid=["fulltext-1"],
        hasFullTextFilename=["paper.mmd"],
    )
    article.dynamic_fields["legacy"] = {"ignored": True}

    result = service.process_and_insert("documents", [article], vector_dim=3)

    assert result == {"insert_count": 1}
    inserted = fake.insert_calls[0]["data"][0]
    assert inserted["id"] == "DOC-1"
    assert inserted["hasFullTextFileid"] == ["fulltext-1"]
    assert inserted["hasFullTextFilename"] == ["paper.mmd"]
    assert "ID" not in inserted
    assert "sysId" not in inserted
    assert "dynamic_fields" not in inserted
    assert len(fake.upsert_calls) == 1
    assert fake.delete_calls == []

    service.delete_by_sys_id("documents", 'DOC-"quoted"\\slash')
    assert fake.delete_calls[0]["filter"] == 'id == "DOC-\\"quoted\\"\\\\slash"'


def test_delete_by_sys_ids_builds_in_filter():
    fake = FakeMilvusClient()
    service = _service(fake)
    service.init_collection("documents", vector_dim=3)

    service.delete_by_sys_ids("documents", ['DOC-"quoted"\\slash', "DOC-2"])

    assert (
        fake.delete_calls[0]["filter"]
        == 'id in ["DOC-\\"quoted\\"\\\\slash", "DOC-2"]'
    )


def test_document_collection_rejects_non_ivfpq_index_type(monkeypatch):
    fake = FakeMilvusClient()
    service = _service(fake)

    service.init_collection("documents", vector_dim=3)

    monkeypatch.setattr("app.services.collection_targets.settings.DOCUMENT_INDEX_TYPE", "UNSUPPORTED")
    with pytest.raises(ValueError, match="unsupported index type"):
        service.resolve_collection_target(vector_dim=3, logical_name="documents")


def test_document_query_nprobe_change_reuses_existing_collection(monkeypatch):
    fake = FakeMilvusClient()
    service = _service(fake)
    service.init_collection("documents", vector_dim=3)

    monkeypatch.setattr(
        "app.services.collection_targets.settings.DOCUMENT_IVFPQ_NPROBE",
        256,
    )

    target = service.resolve_collection_target(vector_dim=3, logical_name="documents")

    assert target.resolved_name == "documents"
    assert target.created is False
    assert target.reused is True
