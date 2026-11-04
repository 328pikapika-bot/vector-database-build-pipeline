from pathlib import Path

from app.services import image_delete_pipeline as pipeline
from app.services.image_delete_pipeline import (
    IMAGE_DELETE_INVALID_DOC_ID,
    IMAGE_DELETE_VALIDATION_ERROR,
    IMAGE_DELETE_WRITER_ERROR,
    delete_images_for_document,
)
from app.services.milvus_image_service import DEFAULT_IMAGE_COLLECTION


class FakeImageWriter:
    def __init__(self, *, delete_result=None, delete_error=None, close_error=None):
        self.delete_result = delete_result or {"delete_count": 3}
        self.delete_error = delete_error
        self.close_error = close_error
        self.delete_calls = []
        self.closed = False

    def delete_images_by_doc_id(self, collection_name, doc_id):
        self.delete_calls.append({"collection_name": collection_name, "doc_id": doc_id})
        if self.delete_error is not None:
            raise self.delete_error
        return self.delete_result

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def test_delete_images_for_document_success_uses_default_collection_with_injected_writer():
    writer = FakeImageWriter(delete_result={"delete_count": 5})

    result = delete_images_for_document(" DOC-1 ", writer=writer)

    assert result == {
        "success": True,
        "doc_id": "DOC-1",
        "collection_name": DEFAULT_IMAGE_COLLECTION,
        "delete_result": {"delete_count": 5},
        "error_code": None,
        "error_message": None,
        "retryable": False,
    }
    assert writer.delete_calls == [{"collection_name": DEFAULT_IMAGE_COLLECTION, "doc_id": "DOC-1"}]
    assert writer.closed is False


def test_delete_images_for_document_passes_custom_collection_name():
    writer = FakeImageWriter()

    result = delete_images_for_document("DOC-2", writer=writer, collection_name="custom_images")

    assert result["success"] is True
    assert result["collection_name"] == "custom_images"
    assert writer.delete_calls == [{"collection_name": "custom_images", "doc_id": "DOC-2"}]


def test_delete_images_for_document_empty_doc_id_does_not_call_writer():
    writer = FakeImageWriter()

    result = delete_images_for_document("   ", writer=writer)

    assert result == {
        "success": False,
        "doc_id": "",
        "collection_name": DEFAULT_IMAGE_COLLECTION,
        "delete_result": None,
        "error_code": IMAGE_DELETE_INVALID_DOC_ID,
        "error_message": "doc_id is required for image delete",
        "retryable": False,
    }
    assert writer.delete_calls == []
    assert writer.closed is False


def test_delete_images_for_document_writer_value_error_is_non_retryable():
    writer = FakeImageWriter(delete_error=ValueError("Collection missing"))

    result = delete_images_for_document("DOC-3", writer=writer)

    assert result["success"] is False
    assert result["doc_id"] == "DOC-3"
    assert result["delete_result"] is None
    assert result["error_code"] == IMAGE_DELETE_VALIDATION_ERROR
    assert result["error_message"] == "Collection missing"
    assert result["retryable"] is False
    assert writer.delete_calls == [{"collection_name": DEFAULT_IMAGE_COLLECTION, "doc_id": "DOC-3"}]


def test_delete_images_for_document_writer_runtime_error_is_retryable():
    writer = FakeImageWriter(delete_error=RuntimeError("milvus unavailable"))

    result = delete_images_for_document("DOC-4", writer=writer)

    assert result["success"] is False
    assert result["doc_id"] == "DOC-4"
    assert result["delete_result"] is None
    assert result["error_code"] == IMAGE_DELETE_WRITER_ERROR
    assert result["error_message"] == "milvus unavailable"
    assert result["retryable"] is True
    assert writer.delete_calls == [{"collection_name": DEFAULT_IMAGE_COLLECTION, "doc_id": "DOC-4"}]


def test_delete_images_for_document_created_writer_is_closed(monkeypatch):
    writer = FakeImageWriter()
    monkeypatch.setattr(pipeline, "MilvusImageService", lambda: writer)

    result = pipeline.delete_images_for_document("DOC-5")

    assert result["success"] is True
    assert writer.delete_calls == [{"collection_name": DEFAULT_IMAGE_COLLECTION, "doc_id": "DOC-5"}]
    assert writer.closed is True


def test_delete_images_for_document_created_writer_close_error_is_best_effort(monkeypatch):
    writer = FakeImageWriter(close_error=RuntimeError("close failed"))
    monkeypatch.setattr(pipeline, "MilvusImageService", lambda: writer)

    result = pipeline.delete_images_for_document("DOC-6")

    assert result["success"] is True
    assert result["delete_result"] == {"delete_count": 3}
    assert writer.closed is True


def test_delete_images_for_document_writer_factory_error_is_retryable(monkeypatch):
    def broken_factory():
        raise RuntimeError("cannot create writer")

    monkeypatch.setattr(pipeline, "MilvusImageService", broken_factory)

    result = pipeline.delete_images_for_document("DOC-7")

    assert result["success"] is False
    assert result["doc_id"] == "DOC-7"
    assert result["delete_result"] is None
    assert result["error_code"] == IMAGE_DELETE_WRITER_ERROR
    assert result["error_message"] == "cannot create writer"
    assert result["retryable"] is True


def test_image_delete_pipeline_source_does_not_reference_document_collection_or_live_clients():
    source = Path(__file__).parents[1].joinpath("app/services/image_delete_pipeline.py").read_text(encoding="utf-8")

    forbidden = [
        "scientific_documents",
        "MILVUS_COLLECTION",
        "get_embedding_client",
        "EmbeddingClient",
        "httpx",
        "requests",
        "openai",
    ]
    for text in forbidden:
        assert text not in source
