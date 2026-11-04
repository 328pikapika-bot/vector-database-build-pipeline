from __future__ import annotations

import inspect

import pytest

import app.services.fulltext_worker as fulltext_worker_module
from app.services.fulltext_downloader import (
    FullTextConfigurationError,
    FullTextDocumentTooLarge,
    FullTextDownloadResult,
    FullTextNonRetryableDownloadError,
    FullTextRetryableDownloadError,
)
from app.services.fulltext_worker import FullTextWorker


@pytest.mark.asyncio
async def test_worker_builds_chunk_plan_and_marks_success_without_mongo_artifacts():
    repository = FakeRepository()
    artifacts = FakeArtifactManager()
    embedder = object()
    chunk_writer = FakeChunkWriter()
    chunk_pipeline = FakeChunkPipeline(success=True, chunk_count=2)
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(text="# Intro\n\nBody text."),
        state_manager=artifacts,
        embedder=embedder,
        chunk_writer=chunk_writer,
        chunk_pipeline=chunk_pipeline,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.chunk_count == 2
    assert repository.successes == [
        {
            "task_id": "task-1",
            "artifact_refs": [],
            "chunk_count": result.chunk_count,
        }
    ]
    assert artifacts.saved == []
    assert chunk_pipeline.calls[0]["embedder"] is embedder
    assert chunk_pipeline.calls[0]["chunk_writer"] is chunk_writer
    assert chunk_pipeline.calls[0]["chunk_plan"]["task_id"] == "task-1"
    assert artifacts.refresh_calls == []


@pytest.mark.asyncio
async def test_worker_preserves_task_doc_metadata_in_chunk_plan():
    repository = FakeRepository()
    artifacts = FakeArtifactManager()
    chunk_pipeline = FakeChunkPipeline(success=True, chunk_count=1)
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(text="# Intro\n\nBody text."),
        state_manager=artifacts,
        chunk_pipeline=chunk_pipeline,
    )

    task = {
        **_task(),
        "title": ["Paper title"],
        "keyword": ["vector"],
        "accessType": "open",
        "classificationCode": "TP391",
        "year": "2026",
        "hasAuthorname": ["Alice"],
        "hasCreateOrganizationname": ["NSTL"],
        "language": ["en"],
    }

    result = await worker.process_task(task)

    assert result.success is True
    chunk_plan = chunk_pipeline.calls[0]["chunk_plan"]
    assert chunk_plan["doc_meta"] == {
        "title": ["Paper title"],
        "keyword": ["vector"],
        "accessType": "open",
        "classificationCode": "TP391",
        "year": "2026",
        "hasAuthorname": ["Alice"],
        "hasCreateOrganizationname": ["NSTL"],
        "language": ["en"],
    }
    assert artifacts.saved == []


@pytest.mark.asyncio
async def test_worker_marks_retryable_download_failure():
    repository = FakeRepository()
    artifacts = FakeArtifactManager()
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(error=FullTextRetryableDownloadError("network")),
        state_manager=artifacts,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.retryable is True
    assert result.error_code == "FULLTEXT_DOWNLOAD_RETRYABLE"
    assert repository.failures[-1]["retryable"] is True
    failure = artifacts.persisted_failures[-1]
    assert failure["pipeline"] == "fulltext"
    assert failure["source_version"] == "2026-06-07T00:00:00+00:00"
    assert failure["item_id"] == "task-1"
    assert failure["logical_collection"] == failure["resolved_collection"]
    assert failure["target_signature"].startswith("FULLTEXT|")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (FullTextConfigurationError("missing base"), "FULLTEXT_CONFIG_ERROR"),
        (FullTextNonRetryableDownloadError("404"), "FULLTEXT_DOWNLOAD_NON_RETRYABLE"),
        (FullTextDocumentTooLarge("too large"), "FULLTEXT_DOCUMENT_TOO_LARGE"),
    ],
)
async def test_worker_marks_download_non_retryable_failures(error, expected_code):
    repository = FakeRepository()
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(error=error),
        state_manager=FakeArtifactManager(),
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.retryable is False
    assert result.error_code == expected_code
    assert repository.failures[-1]["retryable"] is False


@pytest.mark.asyncio
async def test_worker_marks_parse_failure_non_retryable_without_success_artifact():
    repository = FakeRepository()
    artifacts = FakeArtifactManager()

    def fail_parse(_: str):
        raise ValueError("bad mmd")

    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(text="# Intro"),
        state_manager=artifacts,
        parse_func=fail_parse,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.retryable is False
    assert result.error_code == "FULLTEXT_PARSE_ERROR"
    assert artifacts.saved == []
    assert repository.failures[-1]["retryable"] is False


@pytest.mark.asyncio
async def test_worker_does_not_require_mongo_for_success_path():
    repository = FakeRepository()
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(text="# Intro"),
        state_manager=FakeArtifactManager(error=RuntimeError("mongo down")),
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert repository.failures == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_items", "expected_retryable"),
    [
        (
            [
                {
                    "chunkId": "doc-1_0",
                    "error_code": "EMBEDDING_RATE_LIMIT",
                    "error_message": "rate limited",
                    "retryable": True,
                }
            ],
            True,
        ),
        (
            [
                {
                    "chunkId": "doc-1_0",
                    "error_code": "CHUNK_EMPTY_TEXT",
                    "error_message": "empty",
                    "retryable": False,
                }
            ],
            False,
        ),
    ],
)
async def test_worker_marks_chunk_pipeline_failure_and_saves_summary(
    failure_items,
    expected_retryable,
):
    repository = FakeRepository()
    artifacts = FakeArtifactManager()
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(text="# Intro\n\nBody text."),
        state_manager=artifacts,
        embedder=object(),
        chunk_writer=FakeChunkWriter(),
        chunk_pipeline=FakeChunkPipeline(
            success=False,
            failure_items=failure_items,
            chunk_count=1,
        ),
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "FULLTEXT_CHUNK_PIPELINE_ERROR"
    assert result.retryable is expected_retryable
    assert repository.failures[-1]["error_code"] == "FULLTEXT_CHUNK_PIPELINE_ERROR"
    assert repository.failures[-1]["retryable"] is expected_retryable
    assert "failure_items" in repository.failures[-1]["error_message"]
    assert failure_items[0]["error_code"] in repository.failures[-1]["error_message"]
    assert len(artifacts.saved) == 1
    assert artifacts.saved[0]["artifact_type"].value == "MILVUS_WRITE_SUMMARY"
    assert artifacts.saved[0]["success_path"] is False
    assert artifacts.saved[0]["content"]["failure_items"] == failure_items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_retryable"),
    [
        (ValueError("bad chunk plan"), False),
        (RuntimeError("pipeline down"), True),
    ],
)
async def test_worker_classifies_chunk_pipeline_exceptions(error, expected_retryable):
    repository = FakeRepository()
    worker = FullTextWorker(
        repository=repository,
        downloader=FakeDownloader(text="# Intro\n\nBody text."),
        state_manager=FakeArtifactManager(),
        embedder=object(),
        chunk_writer=FakeChunkWriter(),
        chunk_pipeline=FakeChunkPipeline(error=error),
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "FULLTEXT_CHUNK_PIPELINE_ERROR"
    assert result.retryable is expected_retryable
    assert repository.failures[-1]["retryable"] is expected_retryable


def test_worker_module_does_not_import_document_level_milvus():
    source = inspect.getsource(fulltext_worker_module)

    assert "EmbeddingClient" not in source
    assert "get_milvus_service" not in source
    assert "scientific_documents" not in source
    assert "batch_processor_existing" not in source


def _task() -> dict:
    return {
        "_id": "task-1",
        "task_id": "task-1",
        "doc_id": "doc-1",
        "record_update_time": "2026-06-07T00:00:00+00:00",
        "relative_url": "/fulltext/paper.mmd",
    }


class FakeDownloader:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    async def download_text(self, relative_url: str) -> FullTextDownloadResult:
        if self.error is not None:
            raise self.error
        return FullTextDownloadResult(
            relative_url=relative_url,
            url=f"http://files.test{relative_url}",
            text=self.text,
            char_count=len(self.text),
        )


class FakeRepository:
    def __init__(self) -> None:
        self.successes: list[dict] = []
        self.failures: list[dict] = []

    async def mark_task_success(
        self,
        task_id: str,
        *,
        artifact_refs: list[str],
        chunk_count: int,
    ) -> None:
        self.successes.append(
            {
                "task_id": task_id,
                "artifact_refs": artifact_refs,
                "chunk_count": chunk_count,
            }
        )

    async def mark_task_failure(
        self,
        task_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> None:
        self.failures.append(
            {
                "task_id": task_id,
                "error_code": error_code,
                "error_message": error_message,
                "retryable": retryable,
            }
        )


class FakeArtifactManager:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.saved: list[dict] = []
        self.refresh_calls: list[dict] = []
        self.persisted_failures: list[dict] = []

    async def save_artifact(self, **kwargs) -> str:
        if self.error is not None:
            raise self.error
        self.saved.append(kwargs)
        return f"artifact-{len(self.saved)}"

    async def refresh_async_pipeline_status(self, **kwargs) -> None:
        self.refresh_calls.append(kwargs)

    async def persist_failure(self, **kwargs) -> None:
        self.persisted_failures.append(kwargs)


class FakeChunkWriter:
    pass


class FakeChunkPipeline:
    def __init__(
        self,
        *,
        success: bool = True,
        failure_items: list[dict] | None = None,
        chunk_count: int = 1,
        error: Exception | None = None,
    ) -> None:
        self.success = success
        self.failure_items = failure_items or []
        self.chunk_count = chunk_count
        self.error = error
        self.calls: list[dict] = []

    async def __call__(self, chunk_plan, *, embedder, chunk_writer):
        self.calls.append(
            {
                "chunk_plan": chunk_plan,
                "embedder": embedder,
                "chunk_writer": chunk_writer,
            }
        )
        if self.error is not None:
            raise self.error
        return FakeChunkPipelineResult(
            success=self.success,
            chunk_count=self.chunk_count,
            failure_items=self.failure_items,
        )


class FakeChunkPipelineResult:
    def __init__(
        self,
        *,
        success: bool,
        chunk_count: int,
        failure_items: list[dict],
    ) -> None:
        self.success = success
        self.chunk_count = chunk_count
        self.failure_items = failure_items

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "doc_id": "doc-1",
            "chunk_count": self.chunk_count,
            "embedded_count": self.chunk_count if self.success else 0,
            "failed_count": len(self.failure_items),
            "failure_items": self.failure_items,
            "write_summary": (
                {
                    "delete_result": {"delete_count": 1},
                    "insert_result": {"insert_count": self.chunk_count},
                }
                if self.success
                else None
            ),
            "metadata": {},
        }
