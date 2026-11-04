import pytest

from app.models.sync_runtime import EmbeddingItemResult, ErrorType
from app.services import fulltext_chunk_pipeline as pipeline_module
from app.services.fulltext_chunk_pipeline import (
    delete_chunks_for_document,
    process_fulltext_chunk_plan,
)


class FakeEmbedder:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    async def embed_texts_detailed(self, texts):
        self.calls.append(texts)
        if self.error is not None:
            raise self.error
        return self.results


class FakeChunkWriter:
    def __init__(self, *, replace_error=None, delete_error=None, visible_count=None, visibility_error=None):
        self.replace_error = replace_error
        self.delete_error = delete_error
        self.visible_count = visible_count
        self.visibility_error = visibility_error
        self.replace_calls = []
        self.delete_calls = []
        self.insert_calls = []
        self.upsert_calls = []
        self.visibility_calls = []

    def replace_chunks_for_document(self, collection_name, doc_id, chunks, **kwargs):
        call = {
            "collection_name": collection_name,
            "doc_id": doc_id,
            "chunks": list(chunks),
            "kwargs": kwargs,
        }
        self.replace_calls.append(call)
        if self.replace_error is not None:
            raise self.replace_error
        return {
            "delete_result": {"delete_count": 2},
            "insert_result": {"insert_count": len(chunks)},
        }

    def delete_chunks_by_doc_id(self, collection_name, doc_id):
        call = {"collection_name": collection_name, "doc_id": doc_id}
        self.delete_calls.append(call)
        if self.delete_error is not None:
            raise self.delete_error
        return {"delete_count": 4}

    def insert_chunks(self, collection_name, chunks, **kwargs):
        call = {
            "collection_name": collection_name,
            "chunks": list(chunks),
            "kwargs": kwargs,
        }
        self.insert_calls.append(call)
        return {"insert_count": len(chunks)}

    def upsert_chunks(self, collection_name, chunks, **kwargs):
        call = {
            "collection_name": collection_name,
            "chunks": list(chunks),
            "kwargs": kwargs,
        }
        self.upsert_calls.append(call)
        return {"upsert_count": len(chunks)}

    def count_visible_chunk_ids(self, collection_name, chunk_ids, **kwargs):
        call = {
            "collection_name": collection_name,
            "chunk_ids": list(chunk_ids),
            "kwargs": kwargs,
        }
        self.visibility_calls.append(call)
        if self.visibility_error is not None:
            raise self.visibility_error
        if self.visible_count is not None:
            return self.visible_count
        return len(chunk_ids)


@pytest.fixture(autouse=True)
def _default_replace_write_mode(monkeypatch):
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_CHUNK_WRITE_MODE", "replace")
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_CHUNK_VISIBILITY_STRICT", False)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_CHUNK_VISIBILITY_RETRIES", 1)
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_CHUNK_VISIBILITY_RETRY_DELAY_SEC", 0)


def _chunk_dict(chunk_id="DOC-1_0", doc_id="DOC-1", text="chunk text"):
    return {
        "chunkId": chunk_id,
        "docId": doc_id,
        "chunkIndex": int(chunk_id.rsplit("_", 1)[-1]),
        "chunkType": "paragraph",
        "text": text,
        "charStart": 0,
        "charEnd": len(text),
        "sectionPath": "Intro",
        "tokenCount": 2,
        "embedding": None,
    }


def _chunk_plan():
    return {
        "doc_id": "DOC-1",
        "record_update_time": "2026-06-07T00:00:00+00:00",
        "metadata": {
            "parserVersion": "parser-v2",
            "chunker_version": "chunker-v2",
            "embeddingModel": "medium-v2",
            "embeddingVersion": "embed-v1",
            "schemaVersion": "schema-v2",
            "ingestedAt": "2026-06-07T04:00:00+00:00",
        },
        "chunks": [
            _chunk_dict("DOC-1_0", text="first chunk"),
            _chunk_dict("DOC-1_1", text="second chunk"),
        ],
    }


@pytest.mark.asyncio
async def test_process_chunk_plan_embeds_then_replaces_chunks_and_passes_metadata():
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    assert result.doc_id == "DOC-1"
    assert result.chunk_count == 2
    assert result.embedded_count == 2
    assert result.failed_count == 0
    assert result.failure_items == []
    assert result.write_summary == {
        "delete_result": {"delete_count": 2},
        "insert_result": {"insert_count": 2},
    }
    assert embedder.calls == [["first chunk", "second chunk"]]

    replace_call = writer.replace_calls[0]
    assert replace_call["collection_name"] == "chunks"
    assert replace_call["doc_id"] == "DOC-1"
    assert [chunk.chunkId for chunk in replace_call["chunks"]] == ["DOC-1_0", "DOC-1_1"]
    assert [len(chunk.embedding) for chunk in replace_call["chunks"]] == [2560, 2560]
    assert [chunk.language for chunk in replace_call["chunks"]] == [["unknown"], ["unknown"]]
    assert [chunk.sectionPath for chunk in replace_call["chunks"]] == ["Intro", "Intro"]
    assert replace_call["kwargs"] == {
        "vector_dim": 2560,
        "source_record_update_time": "2026-06-07T00:00:00+00:00",
        "parser_version": "parser-v2",
        "chunker_version": "chunker-v2",
        "embedding_model": "medium-v2",
        "embedding_version": "embed-v1",
        "schema_version": "schema-v2",
        "ingested_at": "2026-06-07T04:00:00+00:00",
    }
    assert result.to_dict()["failure_items"] == []


@pytest.mark.asyncio
async def test_process_chunk_plan_defaults_missing_section_path():
    embedder = FakeEmbedder([EmbeddingItemResult(success=True, vector=[1.0] * 2560)])
    writer = FakeChunkWriter()
    chunk = _chunk_dict("DOC-1_0", text="body text")
    chunk["sectionPath"] = None

    result = await process_fulltext_chunk_plan(
        {"doc_id": "DOC-1", "chunks": [chunk]},
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    written_chunk = writer.replace_calls[0]["chunks"][0]
    assert written_chunk.sectionPath == "body"


@pytest.mark.asyncio
async def test_process_chunk_plan_injects_doc_metadata_before_milvus_write():
    embedder = FakeEmbedder([EmbeddingItemResult(success=True, vector=[1.0] * 2560)])
    writer = FakeChunkWriter()
    plan = {
        "doc_id": "DOC-1",
        "doc_meta": {
            "title": ["Paper title"],
            "keyword": ["vector"],
            "accessType": "open",
            "classificationCode": "TP391",
            "year": "2026",
            "hasAuthorname": ["Alice"],
            "hasCreateOrganizationname": ["NSTL"],
            "language": ["en"],
        },
        "chunks": [_chunk_dict("DOC-1_0", text="body text")],
    }

    result = await process_fulltext_chunk_plan(
        plan,
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    written_chunk = writer.replace_calls[0]["chunks"][0]
    assert written_chunk.title == ["Paper title"]
    assert written_chunk.keyword == ["vector"]
    assert written_chunk.accessType == "open"
    assert written_chunk.classificationCode == "TP391"
    assert written_chunk.year == "2026"
    assert written_chunk.hasAuthorname == ["Alice"]
    assert written_chunk.hasCreateOrganizationname == ["NSTL"]
    assert written_chunk.language == ["en"]


@pytest.mark.asyncio
async def test_process_chunk_plan_no_delete_inserts_without_replace_or_delete():
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks_e2e",
        no_delete=True,
    )

    assert result.success is True
    assert writer.replace_calls == []
    assert writer.delete_calls == []
    assert len(writer.insert_calls) == 1
    assert result.write_summary["delete_result"] == {
        "skipped": True,
        "status": "DELETE_NOOP_RECORDED",
        "reason": "LIVE_E2E_NO_DELETE",
        "operation": "replace_chunks_predelete",
        "collection_name": "chunks_e2e",
        "doc_id": "DOC-1",
    }
    assert result.write_summary["insert_result"] == {"insert_count": 2}


@pytest.mark.asyncio
async def test_process_chunk_plan_write_success_but_invisible_rows_is_non_strict_warning():
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter(visible_count=0)

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    assert result.failure_items == []
    assert result.write_summary["delete_result"] == {"delete_count": 2}
    assert result.write_summary["insert_result"] == {"insert_count": 2}
    assert result.write_summary["visibility_check"]["success"] is False
    assert result.write_summary["visibility_check"]["strict"] is False
    assert result.write_summary["visibility_check"]["failure"]["error_code"] == "MILVUS_CHUNK_VISIBILITY_ERROR"
    assert writer.visibility_calls[0]["chunk_ids"] == ["DOC-1_0", "DOC-1_1"]


@pytest.mark.asyncio
async def test_process_chunk_plan_strict_visibility_failure_is_retryable_failure(monkeypatch):
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_CHUNK_VISIBILITY_STRICT", True)
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter(visible_count=0)

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is False
    assert result.write_summary["visibility_check"]["strict"] is True
    assert result.write_summary["visibility_check"]["failure"]["error_code"] == "MILVUS_CHUNK_VISIBILITY_ERROR"
    assert result.failure_items[0].error_code == "MILVUS_CHUNK_VISIBILITY_ERROR"
    assert result.failure_items[0].retryable is True


@pytest.mark.asyncio
async def test_process_chunk_plan_upsert_mode_maps_to_safe_replace(monkeypatch):
    monkeypatch.setattr(pipeline_module.settings, "FULLTEXT_CHUNK_WRITE_MODE", "upsert")
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    assert len(writer.replace_calls) == 1
    assert writer.delete_calls == []
    assert writer.insert_calls == []
    assert writer.upsert_calls == []
    assert result.write_summary["delete_result"] == {"delete_count": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_key", ["content", "payload"])
async def test_process_chunk_plan_accepts_artifact_wrappers(wrapper_key):
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        {wrapper_key: _chunk_plan()},
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    assert result.doc_id == "DOC-1"
    assert len(writer.replace_calls) == 1


@pytest.mark.asyncio
async def test_process_chunk_plan_falls_back_to_first_chunk_doc_id():
    embedder = FakeEmbedder([EmbeddingItemResult(success=True, vector=[1.0] * 2560)])
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        {"chunks": [_chunk_dict("DOC-1_0", doc_id="DOC-1", text="first chunk")]},
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    assert result.doc_id == "DOC-1"
    assert writer.replace_calls[0]["doc_id"] == "DOC-1"


@pytest.mark.asyncio
async def test_process_chunk_plan_embedding_failure_does_not_write_milvus():
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(
                success=False,
                vector=None,
                error_type=ErrorType.RETRYABLE,
                error_code="EMBEDDING_RATE_LIMIT",
                error_message="rate limited",
                retryable=True,
            ),
        ]
    )
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is False
    assert result.chunk_count == 2
    assert result.embedded_count == 1
    assert result.failed_count == 1
    assert result.write_summary is None
    assert [(failure.chunkId, failure.error_code, failure.retryable) for failure in result.failure_items] == [
        ("DOC-1_1", "EMBEDDING_RATE_LIMIT", True)
    ]
    assert writer.replace_calls == []


@pytest.mark.asyncio
async def test_process_chunk_plan_empty_text_and_wrong_dimension_do_not_write_milvus():
    plan = {
        "docId": "DOC-1",
        "chunks": [
            _chunk_dict("DOC-1_0", text="   "),
            _chunk_dict("DOC-1_1", text="valid text"),
        ],
    }
    embedder = FakeEmbedder([EmbeddingItemResult(success=True, vector=[0.0] * 10)])
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        plan,
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is False
    assert embedder.calls == [["valid text"]]
    assert [failure.error_code for failure in result.failure_items] == [
        "CHUNK_EMPTY_TEXT",
        "CHUNK_EMBEDDING_DIMENSION_MISMATCH",
    ]
    assert writer.replace_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_payload",
    [
        {"doc_id": "DOC-1"},
        {"doc_id": "DOC-1", "chunks": [{"chunkId": "DOC-1_0"}]},
    ],
)
async def test_process_chunk_plan_invalid_payload_returns_failure_without_embedding_or_write(bad_payload):
    embedder = FakeEmbedder()
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        bad_payload,
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is False
    assert result.doc_id == ""
    assert result.failed_count == 1
    assert result.failure_items[0].error_code == "CHUNK_PLAN_INVALID"
    assert result.write_summary is None
    assert embedder.calls == []
    assert writer.replace_calls == []


@pytest.mark.asyncio
async def test_process_chunk_plan_milvus_replace_failure_returns_explicit_failure():
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )
    writer = FakeChunkWriter(replace_error=RuntimeError("milvus down"))

    result = await process_fulltext_chunk_plan(
        _chunk_plan(),
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is False
    assert result.embedded_count == 2
    assert result.failed_count == 1
    assert result.write_summary is None
    assert result.failure_items[0].chunkId is None
    assert result.failure_items[0].error_code == "MILVUS_CHUNK_WRITE_ERROR"
    assert result.failure_items[0].error_message == "milvus down"
    assert result.failure_items[0].retryable is True


@pytest.mark.asyncio
async def test_process_empty_chunk_plan_deletes_stale_chunks_and_succeeds():
    embedder = FakeEmbedder()
    writer = FakeChunkWriter()

    result = await process_fulltext_chunk_plan(
        {"doc_id": "DOC-1", "chunks": []},
        embedder=embedder,
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result.success is True
    assert result.chunk_count == 0
    assert result.embedded_count == 0
    assert result.write_summary == {
        "delete_result": {"delete_count": 4},
        "insert_result": {"insert_count": 0},
        "empty_snapshot": True,
    }
    assert embedder.calls == []
    assert writer.replace_calls == []
    assert writer.delete_calls == [{"collection_name": "chunks", "doc_id": "DOC-1"}]


def test_delete_chunks_for_document_calls_writer_by_doc_id():
    writer = FakeChunkWriter()

    result = delete_chunks_for_document(
        " DOC-1 ",
        chunk_writer=writer,
        collection_name="chunks",
    )

    assert result == {
        "doc_id": "DOC-1",
        "collection_name": "chunks",
        "delete_result": {"delete_count": 4},
    }
    assert writer.delete_calls == [{"collection_name": "chunks", "doc_id": "DOC-1"}]


def test_delete_chunks_for_document_failure_is_explicit():
    writer = FakeChunkWriter(delete_error=RuntimeError("delete failed"))

    with pytest.raises(RuntimeError, match="delete failed"):
        delete_chunks_for_document("DOC-1", chunk_writer=writer, collection_name="chunks")


def test_delete_chunks_for_document_requires_doc_id_before_writer_call():
    writer = FakeChunkWriter()

    with pytest.raises(ValueError, match="doc_id is required"):
        delete_chunks_for_document("   ", chunk_writer=writer, collection_name="chunks")

    assert writer.delete_calls == []
