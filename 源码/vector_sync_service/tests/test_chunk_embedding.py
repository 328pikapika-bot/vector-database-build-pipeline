import pytest

from app.models.chunk import Chunk
from app.models.sync_runtime import EmbeddingItemResult, ErrorType
from app.services.chunk_embedding import embed_chunks


class FakeEmbedder:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    async def embed_texts_detailed(self, texts):
        self.calls.append(texts)
        if self.error:
            raise self.error
        return self.results


def _chunk(chunk_id="DOC-1_0", text="chunk text", embedding=None):
    return Chunk(
        chunkId=chunk_id,
        docId="DOC-1",
        chunkIndex=int(chunk_id.rsplit("_", 1)[-1]),
        chunkType="paragraph",
        text=text,
        charStart=0,
        charEnd=len(text),
        sectionPath="Intro",
        tokenCount=2,
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_embed_chunks_success_returns_chunks_with_2560_dimension_vectors():
    chunks = [_chunk("DOC-1_0", "first"), _chunk("DOC-1_1", "second")]
    embedder = FakeEmbedder(
        [
            EmbeddingItemResult(success=True, vector=[1.0] * 2560),
            EmbeddingItemResult(success=True, vector=[2.0] * 2560),
        ]
    )

    result = await embed_chunks(chunks, embedder)

    assert embedder.calls == [["first", "second"]]
    assert result.failures == []
    assert [chunk.chunkId for chunk in result.chunks] == ["DOC-1_0", "DOC-1_1"]
    assert [len(chunk.embedding) for chunk in result.chunks] == [2560, 2560]
    assert chunks[0].embedding is None


@pytest.mark.asyncio
async def test_embed_chunks_records_single_embedding_failure_without_writable_chunk():
    chunks = [_chunk("DOC-1_0", "first"), _chunk("DOC-1_1", "second")]
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

    result = await embed_chunks(chunks, embedder)

    assert [chunk.chunkId for chunk in result.chunks] == ["DOC-1_0"]
    assert len(result.failures) == 1
    assert result.failures[0].chunkId == "DOC-1_1"
    assert result.failures[0].error_code == "EMBEDDING_RATE_LIMIT"
    assert result.failures[0].error_message == "rate limited"
    assert result.failures[0].retryable is True


@pytest.mark.asyncio
async def test_embed_chunks_rejects_empty_text_and_wrong_dimension_without_zero_fallback():
    chunks = [_chunk("DOC-1_0", "   "), _chunk("DOC-1_1", "valid text")]
    embedder = FakeEmbedder([EmbeddingItemResult(success=True, vector=[0.0] * 10)])

    result = await embed_chunks(chunks, embedder)

    assert embedder.calls == [["valid text"]]
    assert result.chunks == []
    assert [(failure.chunkId, failure.error_code) for failure in result.failures] == [
        ("DOC-1_0", "CHUNK_EMPTY_TEXT"),
        ("DOC-1_1", "CHUNK_EMBEDDING_DIMENSION_MISMATCH"),
    ]


@pytest.mark.asyncio
async def test_embed_chunks_records_result_count_mismatch_as_failures():
    chunks = [_chunk("DOC-1_0", "first"), _chunk("DOC-1_1", "second")]
    embedder = FakeEmbedder([EmbeddingItemResult(success=True, vector=[1.0] * 2560)])

    result = await embed_chunks(chunks, embedder)

    assert result.chunks == []
    assert [failure.error_code for failure in result.failures] == [
        "CHUNK_EMBEDDING_RESULT_COUNT_MISMATCH",
        "CHUNK_EMBEDDING_RESULT_COUNT_MISMATCH",
    ]
