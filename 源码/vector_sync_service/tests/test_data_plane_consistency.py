import re

import pytest

from app.config import settings
from app.services.data_plane_consistency import verify_data_plane_consistency
from app.services.milvus_chunk_service import DEFAULT_CHUNK_COLLECTION
from app.services.milvus_image_service import DEFAULT_IMAGE_COLLECTION


class FakeCursor(list):
    def limit(self, value):
        return FakeCursor(self[:value])


class FakeCollection:
    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = []
        self.mutations = []

    def find_one(self, query, *args, **kwargs):
        self.calls.append(("find_one", query))
        for item in self.documents:
            if _matches_query(item, query):
                return item
        return None

    def count_documents(self, query):
        self.calls.append(("count_documents", query))
        return sum(1 for item in self.documents if _matches_query(item, query))

    def find(self, query, *args, **kwargs):
        self.calls.append(("find", query))
        return FakeCursor([item for item in self.documents if _matches_query(item, query)])

    def insert_one(self, *_args, **_kwargs):
        self.mutations.append("insert_one")
        raise AssertionError("verifier must not write Mongo")

    def update_one(self, *_args, **_kwargs):
        self.mutations.append("update_one")
        raise AssertionError("verifier must not write Mongo")

    def delete_one(self, *_args, **_kwargs):
        self.mutations.append("delete_one")
        raise AssertionError("verifier must not write Mongo")


class FakeMongoManager:
    def __init__(self, *, status=None, failures=None, artifacts=None):
        self.status_collection = FakeCollection([status] if status else [])
        self.failures_collection = FakeCollection(failures or [])
        self.artifacts_collection = FakeCollection(artifacts or [])

    async def get_status(self, doc_id):
        return self.status_collection.find_one({"_id": doc_id})


class FakeMilvusClient:
    def __init__(self, data_by_collection, *, count_query_fails=False):
        self.data_by_collection = {key: list(value) for key, value in data_by_collection.items()}
        self.count_query_fails = count_query_fails
        self.query_calls = []
        self.has_collection_calls = []
        self.mutations = []

    def has_collection(self, collection_name):
        self.has_collection_calls.append(collection_name)
        return collection_name in self.data_by_collection

    def query(self, *, collection_name, filter, output_fields, limit=None):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "filter": filter,
                "output_fields": output_fields,
                "limit": limit,
            }
        )
        if output_fields == ["count(*)"] and self.count_query_fails:
            raise RuntimeError("count query unsupported")
        rows = [item for item in self.data_by_collection.get(collection_name, []) if _matches_milvus_filter(item, filter)]
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        limited = rows[:limit] if limit is not None else rows
        return [{field: item.get(field) for field in output_fields} for item in limited]

    def delete(self, *_args, **_kwargs):
        self.mutations.append("delete")
        raise AssertionError("verifier must not delete Milvus data")

    def insert(self, *_args, **_kwargs):
        self.mutations.append("insert")
        raise AssertionError("verifier must not insert Milvus data")

    def flush(self, *_args, **_kwargs):
        self.mutations.append("flush")
        raise AssertionError("verifier must not flush Milvus data")

    def load_collection(self, *_args, **_kwargs):
        self.mutations.append("load_collection")
        raise AssertionError("verifier must not load Milvus collections")

    def create_collection(self, *_args, **_kwargs):
        self.mutations.append("create_collection")
        raise AssertionError("verifier must not create Milvus collections")


def _matches_query(document, query):
    return all(document.get(key) == value for key, value in query.items())


def _matches_milvus_filter(document, filter_expr):
    match = re.fullmatch(r'(\w+)(?:\["([^"]+)"\])? == "(.*)"', filter_expr)
    assert match, filter_expr
    field, nested_key, raw_value = match.groups()
    value = raw_value.replace('\\"', '"').replace("\\\\", "\\")
    actual = document.get(field)
    if nested_key is not None:
        actual = actual.get(nested_key) if isinstance(actual, dict) else None
    return actual == value


@pytest.mark.asyncio
async def test_verifier_defaults_to_unchecked_without_creating_live_clients():
    result = await verify_data_plane_consistency("DOC-1")

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["mongo"]["checked"] is False
    assert result["milvus"]["document"]["checked"] is False
    assert result["milvus"]["chunk"]["checked"] is False
    assert result["milvus"]["image"]["checked"] is False
    assert result["collections"] == {
        "document": settings.MILVUS_COLLECTION,
        "chunk": DEFAULT_CHUNK_COLLECTION,
        "image": DEFAULT_IMAGE_COLLECTION,
    }
    assert result["mismatch_flags"]["chunk_collection_points_to_document_collection"] is False
    assert result["mismatch_flags"]["image_collection_points_to_document_collection"] is False


@pytest.mark.asyncio
async def test_verifier_active_document_with_expected_rows_has_no_mismatch():
    mongo = FakeMongoManager(status={"_id": "DOC-A", "doc_id": "DOC-A", "overall_status": "SUCCESS"})
    milvus = FakeMilvusClient(
        {
            "documents": [{"id": "DOC-A"}],
            "chunks": [{"doc_id": "DOC-A", "id": "DOC-A_chunk_0"}],
            "images": [{"doc_id": "DOC-A", "id": "DOC-A_img_0"}],
        }
    )

    result = await verify_data_plane_consistency(
        "DOC-A",
        mongo_manager=mongo,
        document_client=milvus,
        chunk_client=milvus,
        image_client=milvus,
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
        expect_document=True,
        expect_chunks=True,
        expect_images=True,
    )

    assert result["expectations"] == {"document": True, "chunks": True, "images": True}
    assert result["mismatch_flags"]["active_status_missing_expected_document_rows"] is False
    assert result["mismatch_flags"]["active_status_missing_expected_chunk_rows"] is False
    assert result["mismatch_flags"]["active_status_missing_expected_image_rows"] is False
    assert result["has_mismatch"] is False
    assert milvus.mutations == []


@pytest.mark.asyncio
async def test_verifier_flags_active_status_missing_expected_rows():
    mongo = FakeMongoManager(status={"_id": "DOC-MISSING", "doc_id": "DOC-MISSING", "overall_status": "SUCCESS"})
    milvus = FakeMilvusClient({"documents": [], "chunks": [], "images": []})

    result = await verify_data_plane_consistency(
        "DOC-MISSING",
        mongo_manager=mongo,
        document_client=milvus,
        chunk_client=milvus,
        image_client=milvus,
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
        expect_document=True,
        expect_chunks=True,
        expect_images=True,
    )

    assert result["mismatch_flags"]["active_status_missing_expected_document_rows"] is True
    assert result["mismatch_flags"]["active_status_missing_expected_chunk_rows"] is True
    assert result["mismatch_flags"]["active_status_missing_expected_image_rows"] is True
    assert result["has_mismatch"] is True
    assert milvus.mutations == []


@pytest.mark.asyncio
async def test_verifier_flags_expected_no_rows_when_rows_are_found():
    mongo = FakeMongoManager(status={"_id": "DOC-NO-ROWS", "doc_id": "DOC-NO-ROWS", "overall_status": "SUCCESS"})
    milvus = FakeMilvusClient(
        {
            "documents": [{"id": "DOC-NO-ROWS"}],
            "chunks": [{"doc_id": "DOC-NO-ROWS", "id": "DOC-NO-ROWS_chunk_0"}],
            "images": [{"doc_id": "DOC-NO-ROWS", "id": "DOC-NO-ROWS_img_0"}],
        }
    )

    result = await verify_data_plane_consistency(
        "DOC-NO-ROWS",
        mongo_manager=mongo,
        document_client=milvus,
        chunk_client=milvus,
        image_client=milvus,
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
        expect_document=False,
        expect_chunks=False,
        expect_images=False,
    )

    assert result["mismatch_flags"]["expected_no_document_rows_but_found"] is True
    assert result["mismatch_flags"]["expected_no_chunk_rows_but_found"] is True
    assert result["mismatch_flags"]["expected_no_image_rows_but_found"] is True
    assert result["has_mismatch"] is True


@pytest.mark.asyncio
async def test_verifier_summarizes_fake_mongo_and_milvus_counts_read_only():
    doc_id = 'DOC-"quoted"\\slash'
    mongo = FakeMongoManager(
        status={
            "_id": doc_id,
            "doc_id": doc_id,
            "overall_status": "SUCCESS",
            "current_stage": "MILVUS_WRITE_DOCUMENT",
            "last_success_stage": "MILVUS_WRITE_DOCUMENT",
            "progress": {"chunk_count_written": 2, "image_count_written": 1},
        },
        failures=[
            {
                "doc_id": doc_id,
                "failure_key": "failure-1",
                "status": "PENDING",
                "failure_stage": "DELETE_IMAGES",
                "scope_type": "IMAGE",
                "scope_id": doc_id,
                "error_code": "IMAGE_DELETE_WRITER_ERROR",
            }
        ],
        artifacts=[
            {"doc_id": doc_id, "_id": "artifact-1", "artifact_type": "MILVUS_WRITE_SUMMARY", "stage": "DELETE_IMAGES"},
            {"doc_id": doc_id, "_id": "artifact-2", "artifact_type": "CHUNK_PLAN_JSON", "stage": "FULLTEXT_BUILD_CHUNKS"},
        ],
    )
    milvus = FakeMilvusClient(
        {
            "documents": [{"id": doc_id}, {"id": "OTHER"}],
            "chunks": [{"doc_id": doc_id, "id": "c1"}, {"doc_id": doc_id, "id": "c2"}],
            "images": [{"doc_id": doc_id, "id": "i1"}],
        }
    )

    result = await verify_data_plane_consistency(
        doc_id,
        mongo_manager=mongo,
        document_client=milvus,
        chunk_client=milvus,
        image_client=milvus,
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
    )

    assert result["mongo"]["status_found"] is True
    assert result["mongo"]["status"]["overall_status"] == "SUCCESS"
    assert result["mongo"]["failures"]["count"] == 1
    assert result["mongo"]["failures"]["status_counts"] == {"PENDING": 1}
    assert result["mongo"]["artifacts"]["count"] == 2
    assert result["mongo"]["artifacts"]["type_counts"]["MILVUS_WRITE_SUMMARY"] == 1
    assert result["milvus"]["document"]["count"] == 1
    assert result["milvus"]["chunk"]["count"] == 2
    assert result["milvus"]["image"]["count"] == 1
    assert result["mismatch_flags"]["mongo_status_missing"] is False
    assert milvus.mutations == []
    assert mongo.status_collection.mutations == []
    assert mongo.failures_collection.mutations == []
    assert mongo.artifacts_collection.mutations == []
    assert milvus.query_calls[0]["filter"] == 'id == "DOC-\\"quoted\\"\\\\slash"'
    assert milvus.query_calls[1]["filter"] == 'doc_id == "DOC-\\"quoted\\"\\\\slash"'
    assert milvus.query_calls[2]["filter"] == 'doc_id == "DOC-\\"quoted\\"\\\\slash"'


@pytest.mark.asyncio
async def test_verifier_flags_deleted_status_with_remaining_rows():
    mongo = FakeMongoManager(status={"_id": "DOC-1", "doc_id": "DOC-1", "overall_status": "DELETED"})
    milvus = FakeMilvusClient(
        {
            "documents": [{"id": "DOC-1"}],
            "chunks": [{"doc_id": "DOC-1", "id": "DOC-1_0"}],
            "images": [{"doc_id": "DOC-1", "id": "DOC-1_img_0"}],
        }
    )

    result = await verify_data_plane_consistency(
        "DOC-1",
        mongo_manager=mongo,
        document_client=milvus,
        chunk_client=milvus,
        image_client=milvus,
        document_collection="documents",
        chunk_collection="chunks",
        image_collection="images",
    )

    assert result["mismatch_flags"]["deleted_status_has_document_rows"] is True
    assert result["mismatch_flags"]["deleted_status_has_chunk_rows"] is True
    assert result["mismatch_flags"]["deleted_status_has_image_rows"] is True
    assert result["mismatch_flags"]["deleted_status_retaining_document_rows"] is True
    assert result["mismatch_flags"]["deleted_status_retaining_chunk_rows"] is True
    assert result["mismatch_flags"]["deleted_status_retaining_image_rows"] is True
    assert result["has_mismatch"] is True


@pytest.mark.asyncio
async def test_verifier_does_not_flag_partial_failure_when_failure_records_exist():
    mongo = FakeMongoManager(
        status={"_id": "DOC-PF", "doc_id": "DOC-PF", "overall_status": "PARTIAL_FAILURE"},
        failures=[{"doc_id": "DOC-PF", "failure_key": "failure-1", "status": "PENDING", "failure_stage": "DELETE_IMAGES"}],
    )

    result = await verify_data_plane_consistency("DOC-PF", mongo_manager=mongo)

    assert result["mismatch_flags"]["partial_failure_has_no_failure_records"] is False


@pytest.mark.asyncio
async def test_verifier_flags_partial_failure_without_failure_records():
    mongo = FakeMongoManager(status={"_id": "DOC-2", "doc_id": "DOC-2", "overall_status": "PARTIAL_FAILURE"})

    result = await verify_data_plane_consistency("DOC-2", mongo_manager=mongo)

    assert result["mismatch_flags"]["partial_failure_has_no_failure_records"] is True


@pytest.mark.asyncio
async def test_verifier_flags_collection_name_collisions():
    result = await verify_data_plane_consistency(
        "DOC-3",
        document_collection="same",
        chunk_collection="same",
        image_collection="same",
    )

    assert result["mismatch_flags"]["chunk_collection_points_to_document_collection"] is True
    assert result["mismatch_flags"]["image_collection_points_to_document_collection"] is True
    assert result["mismatch_flags"]["image_collection_points_to_chunk_collection"] is True


@pytest.mark.asyncio
async def test_verifier_uses_limited_query_fallback_when_count_query_is_unavailable():
    milvus = FakeMilvusClient(
        {"chunks": [{"doc_id": "DOC-4", "id": "DOC-4_0"}, {"doc_id": "DOC-4", "id": "DOC-4_1"}]},
        count_query_fails=True,
    )

    result = await verify_data_plane_consistency(
        "DOC-4",
        chunk_client=milvus,
        chunk_collection="chunks",
    )

    assert result["milvus"]["chunk"]["count"] == 2
    assert result["milvus"]["chunk"]["query_mode"] == "limited_query"
    assert milvus.mutations == []
