from __future__ import annotations

import hashlib
import inspect
import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import settings
from app.models.image import ImageRecord
from app.models.sync_runtime import ArtifactType, ScopeType, SyncStage
from app.services.fulltext_downloader import FullTextDownloadResult
from app.services.image_feature_pipeline import ImageFeatureProviders
from app.services.image_downloader import (
    ImageConfigurationError,
    ImageDownloadResult,
    ImageNonRetryableDownloadError,
    ImageRetryableDownloadError,
)
from app.services.image_worker import ImageWorker
from app.services import image_worker as image_worker_module
from app.services.milvus_image_service import (
    TEXT_VECTOR_DIM,
    VISUAL_TEXT_VECTOR_DIM,
    VISUAL_VECTOR_DIM,
)


@pytest.fixture(autouse=True)
def runtime_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "upload"))
    monkeypatch.setattr(settings, "IMAGE_FEATURE_PIPELINE_ENABLED", False)
    monkeypatch.setattr(settings, "IMAGE_MILVUS_WRITE_ENABLED", False)
    monkeypatch.setattr(settings, "IMAGE_SUCCESS_ARTIFACT_MODE", "full")
    monkeypatch.setattr(settings, "IMAGE_SUCCESS_RESULT_MODE", "full")
    monkeypatch.setattr(settings, "IMAGE_RUNTIME_ATTACHMENT_MODE", "full")
    monkeypatch.setattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "required")
    monkeypatch.setattr(settings, "IMAGE_TEXT_INDEX_POLICY", "legacy")
    monkeypatch.setattr(settings, "IMAGE_ALLOW_DOCUMENT_TITLE_CONTEXT_FALLBACK", True)


@pytest.mark.asyncio
async def test_process_task_downloads_without_writing_success_manifest_to_mongo():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    downloader = FakeDownloader(
        ImageDownloadResult(
            relative_url="/images/fig1.png",
            url="http://files.test/images/fig1.png",
            content=content,
            content_type="image/png",
            size_bytes=len(content),
        )
    )
    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=downloader,
    )

    result = await worker.process_task(_task())
    await asyncio.sleep(0)

    assert result.success is True
    assert result.artifact_refs == []
    assert downloader.calls == ["/images/fig1.png"]
    assert state_manager.refresh_calls == []
    result_summary = repository.success_calls[0]["result_summary"]
    assert result_summary["image_id"] == "doc-1_img_1"
    assert result_summary["relative_url"] == "/images/fig1.png"
    assert result_summary["bytes_sha256"] == hashlib.sha256(content).hexdigest()
    assert result_summary["feature_pipeline_enabled"] is False
    assert result_summary["success_stage"] == "download"
    assert result_summary["mmd_relative_url"] is None
    assert result_summary["milvus_write_mode"] == "upsert"
    assert repository.failure_calls == []
    assert state_manager.artifacts == []


@pytest.mark.asyncio
async def test_process_task_with_injected_pipeline_skips_success_feature_summary_artifact():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    seen = {}

    async def fake_pipeline(image_bytes, draft_record, *, providers, context_text):
        seen["image_bytes"] = image_bytes
        seen["draft_id"] = draft_record.img_id
        seen["providers"] = providers
        seen["context_text"] = context_text
        return SimpleNamespace(
            success=True,
            image_id=draft_record.img_id,
            doc_id=draft_record.docId,
            feature_summary=SimpleNamespace(
                p_hash_present=True,
                visual_dim=4096,
                text_dim=2560,
                visual_text_dim=4096,
                sparse_terms=3,
                bm25_text_chars=27,
            ),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=fake_pipeline,
        image_write_enabled=False,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.artifact_refs == []
    assert seen["image_bytes"] == content
    assert seen["draft_id"] == "doc-1_img_1"
    assert seen["providers"] is None
    assert seen["context_text"] == "Figure 1 caption"
    assert repository.failure_calls == []
    assert repository.success_calls[0]["artifact_refs"] == []
    assert repository.success_calls[0]["result_summary"]["feature_pipeline_enabled"] is True
    assert repository.success_calls[0]["result_summary"]["feature_success"] is True
    assert repository.success_calls[0]["result_summary"]["feature_summary"]["visual_dim"] == 4096

    assert state_manager.artifacts == []


@pytest.mark.asyncio
async def test_process_task_downloads_mmd_and_uses_it_for_image_context():
    content = _png_bytes(width=320, height=240)
    mmd_text = "\n\n".join(
        [
            "Previous paragraph.",
            "![Figure](figures/fig1.png)",
            "Next paragraph one.",
            "Next paragraph two.",
            "Next paragraph three.",
        ]
    )
    seen = {}

    async def fake_pipeline(image_bytes, draft_record, *, providers, context_text):
        seen["context_before"] = draft_record.context_before
        seen["context_after"] = draft_record.context_after
        seen["img_meta"] = dict(draft_record.img_meta)
        seen["raw_img_meta"] = dict(draft_record.raw_img_meta)
        seen["context_text"] = context_text
        return SimpleNamespace(
            success=True,
            image_id=draft_record.img_id,
            doc_id=draft_record.docId,
            feature_summary=SimpleNamespace(
                p_hash_present=True,
                caption_dim=TEXT_VECTOR_DIM,
                context_dim=TEXT_VECTOR_DIM,
            ),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    worker = ImageWorker(
        repository=FakeRepository(),
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        fulltext_downloader=FakeFullTextDownloader(
            FullTextDownloadResult(
                relative_url="/fulltext/doc-1/doc-1.mmd",
                url="https://static.nstl.gov.cn/fulltext/doc-1/doc-1.mmd",
                text=mmd_text,
                char_count=len(mmd_text),
            )
        ),
        feature_pipeline=fake_pipeline,
    )

    result = await worker.process_task(
        {**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
    )

    assert result.success is True
    assert worker.fulltext_downloader.calls == ["/fulltext/doc-1/doc-1.mmd"]
    assert seen["context_before"] == "Previous paragraph."
    assert seen["context_after"] == "Figure 1\n\nNext paragraph one.\n\nNext paragraph two."
    assert seen["img_meta"]["caption"] == "Figure 1"
    assert seen["img_meta"]["mmd_context_located"] is True
    assert seen["raw_img_meta"] == _task()["raw_img_meta"]
    assert "mmd_image_context" not in seen["raw_img_meta"]
    assert "Previous paragraph." in seen["context_text"]
    assert "Next paragraph one." in seen["context_text"]
    assert "Next paragraph two." in seen["context_text"]


@pytest.mark.asyncio
async def test_required_mmd_failure_starts_visual_but_does_not_write_record():
    calls = []

    async def fake_pipeline(*_args, **_kwargs):
        calls.append("visual")
        return SimpleNamespace(
            success=True,
            image_id="doc-1_img_1",
            doc_id="doc-1",
            feature_summary=SimpleNamespace(p_hash_present=True, visual_dim=VISUAL_VECTOR_DIM),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        fulltext_downloader=FakeFullTextDownloader(
            error=ImageRetryableDownloadError("temporary MMD failure")
        ),
        feature_pipeline=fake_pipeline,
        feature_pipeline_enabled=True,
        image_write_enabled=False,
    )

    result = await worker.process_task(
        {**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
    )

    assert result.success is False
    assert result.error_code == "IMAGE_MMD_CONTEXT_REQUIRED"
    assert calls == []
    assert repository.success_calls == []
    assert repository.failure_calls[0]["error_code"] == "IMAGE_MMD_CONTEXT_REQUIRED"


@pytest.mark.asyncio
async def test_process_task_background_mmd_mode_keeps_image_hot_path_light():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    seen = {}

    async def fake_pipeline(image_bytes, draft_record, *, providers, context_text):
        seen["context_before"] = draft_record.context_before
        seen["context_after"] = draft_record.context_after
        seen["img_meta"] = dict(draft_record.img_meta)
        seen["context_text"] = context_text
        return SimpleNamespace(
            success=True,
            image_id=draft_record.img_id,
            doc_id=draft_record.docId,
            feature_summary=SimpleNamespace(
                p_hash_present=True,
                caption_dim=TEXT_VECTOR_DIM,
                context_dim=TEXT_VECTOR_DIM,
            ),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background")
    try:
        worker = ImageWorker(
            repository=repository,
            state_manager=state_manager,
            downloader=FakeDownloader(_download_result(content)),
            fulltext_downloader=FakeFullTextDownloader(
                error=AssertionError("mmd should not block hot path")
            ),
            feature_pipeline=fake_pipeline,
            image_write_enabled=False,
        )

        result = await worker.process_task(
            {**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
        )
    finally:
        monkeypatch.undo()

    assert result.success is True
    assert seen["context_after"]
    assert seen["context_text"]
    assert seen["img_meta"]["mmd_backfill_status"] == "pending"
    assert seen["img_meta"]["mmd_context_source"] == "hot_path_fallback"
    assert worker.fulltext_downloader.calls == []
    assert repository.success_calls[0]["result_summary"]["success_stage"] == "feature"


@pytest.mark.asyncio
async def test_verified_background_task_still_runs_visual_embedding(monkeypatch):
    calls = []

    async def fake_pipeline(*_args, **_kwargs):
        calls.append("embedding")
        return SimpleNamespace(
            success=True,
            image_id="doc-1_img_1",
            doc_id="doc-1",
            feature_summary=SimpleNamespace(p_hash_present=True, visual_dim=VISUAL_VECTOR_DIM),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    monkeypatch.setattr(settings, "IMAGE_TEXT_INDEX_POLICY", "verified_only")
    monkeypatch.setattr(settings, "IMAGE_ALLOW_DOCUMENT_TITLE_CONTEXT_FALLBACK", False)
    monkeypatch.setattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background")
    worker = ImageWorker(
        repository=FakeRepository(),
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        fulltext_downloader=FakeFullTextDownloader(
            error=AssertionError("background mode must not block the image download")
        ),
        feature_pipeline=fake_pipeline,
        image_write_enabled=False,
    )

    result = await worker.process_task(
        {**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
    )

    assert result.success is True
    assert calls == ["embedding"]
    summary = worker.repository.success_calls[0]["result_summary"]
    assert summary["success_stage"] == "feature"
    assert summary["mmd_context_status"]["quality_status"] == "NOT_INDEXABLE_MMD_PENDING"


@pytest.mark.asyncio
async def test_process_task_background_mmd_mode_fills_missing_before_from_doc_title_only():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    seen = {}

    async def fake_pipeline(image_bytes, draft_record, *, providers, context_text):
        seen["context_before"] = draft_record.context_before
        seen["context_after"] = draft_record.context_after
        seen["context_text"] = context_text
        return SimpleNamespace(
            success=True,
            image_id=draft_record.img_id,
            doc_id=draft_record.docId,
            feature_summary=SimpleNamespace(
                p_hash_present=True,
                caption_dim=TEXT_VECTOR_DIM,
                context_dim=TEXT_VECTOR_DIM,
            ),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background")
    try:
        worker = ImageWorker(
            repository=repository,
            state_manager=state_manager,
            downloader=FakeDownloader(_download_result(content)),
            fulltext_downloader=FakeFullTextDownloader(
                error=AssertionError("mmd should not block hot path")
            ),
            feature_pipeline=fake_pipeline,
            image_write_enabled=False,
        )

        task = _task()
        task["context_before"] = None
        task["context_after"] = "Figure 1 caption"
        result = await worker.process_task(
            {**task, "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
        )
    finally:
        monkeypatch.undo()

    assert result.success is True
    assert seen["context_before"] == "Document 1"
    assert seen["context_after"] == "Figure 1 caption"
    assert seen["context_text"] == "Document 1\nFigure 1 caption"


@pytest.mark.asyncio
async def test_process_task_background_mmd_mode_does_not_use_url_name_or_image_id_as_context():
    content = _png_bytes(width=320, height=240)
    seen = {}

    async def fake_pipeline(image_bytes, draft_record, *, providers, context_text):
        seen["context_before"] = draft_record.context_before
        seen["context_after"] = draft_record.context_after
        seen["context_text"] = context_text
        return SimpleNamespace(
            success=True,
            image_id=draft_record.img_id,
            doc_id=draft_record.docId,
            feature_summary=SimpleNamespace(
                p_hash_present=True,
                caption_dim=TEXT_VECTOR_DIM,
                context_dim=TEXT_VECTOR_DIM,
            ),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "IMAGE_MMD_HOT_PATH_MODE", "background")
    try:
        worker = ImageWorker(
            repository=FakeRepository(),
            state_manager=FakeStateManager(),
            downloader=FakeDownloader(_download_result(content)),
            fulltext_downloader=FakeFullTextDownloader(
                error=AssertionError("mmd should not block hot path")
            ),
            feature_pipeline=fake_pipeline,
            image_write_enabled=False,
        )

        task = _task()
        task["doc_title"] = None
        task["context_before"] = None
        task["context_after"] = ""
        task["title"] = "Figure 1"
        result = await worker.process_task(
            {**task, "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
        )
    finally:
        monkeypatch.undo()

    assert result.success is True
    assert seen["context_before"] is None
    assert seen["context_after"] == "Figure 1"
    assert seen["context_text"] == "Figure 1"


@pytest.mark.asyncio
async def test_process_task_canonicalizes_escaped_html_before_embedding_input():
    content = _png_bytes(width=320, height=240)
    seen = {}

    async def fake_pipeline(image_bytes, draft_record, *, providers, context_text):
        seen["context_before"] = draft_record.context_before
        seen["context_after"] = draft_record.context_after
        seen["caption"] = draft_record.img_meta.get("caption")
        seen["context_text"] = context_text
        return SimpleNamespace(
            success=True,
            image_id=draft_record.img_id,
            doc_id=draft_record.docId,
            feature_summary=SimpleNamespace(
                p_hash_present=True,
                caption_dim=TEXT_VECTOR_DIM,
                context_dim=TEXT_VECTOR_DIM,
            ),
            failure_items=[],
            image_record=_writable_image_record(),
        )

    mmd_text = "\n\n".join(
        [
            "Previous &amp;lt;i&amp;gt;text&amp;lt;/i&amp;gt;.",
            "![Figure](figures/fig1.png)",
            "After &amp;lt;i&amp;gt;text&amp;lt;/i&amp;gt; one.",
            "After two.",
        ]
    )
    worker = ImageWorker(
        repository=FakeRepository(),
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        fulltext_downloader=FakeFullTextDownloader(
            FullTextDownloadResult(
                relative_url="/fulltext/doc-1/doc-1.mmd",
                url="https://static.nstl.gov.cn/fulltext/doc-1/doc-1.mmd",
                text=mmd_text,
                char_count=len(mmd_text),
            )
        ),
        feature_pipeline=fake_pipeline,
    )

    task = _task()
    task["title"] = "Taxonomic Studies on Some Members of the Genus &amp;lt;i&amp;gt;Abutilon&amp;lt;/i&amp;gt; Mill. (Malvaceae)"
    result = await worker.process_task(
        {**task, "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
    )

    assert result.success is True
    assert seen["context_before"] == "Previous text."
    assert seen["caption"] == "Taxonomic Studies on Some Members of the Genus Abutilon Mill. (Malvaceae)"
    assert seen["context_after"] == (
        "Taxonomic Studies on Some Members of the Genus Abutilon Mill. (Malvaceae)\n\n"
        "After text one.\n\nAfter two."
    )
    assert "&amp;lt;" not in seen["context_text"]
    assert "<i>" not in seen["context_text"]
    assert "Abutilon" in seen["context_text"]


@pytest.mark.asyncio
async def test_process_task_skips_mmd_download_when_located_context_exists(monkeypatch):
    content = _png_bytes(width=320, height=240)
    fulltext = FakeFullTextDownloader(error=AssertionError("mmd should be skipped"))
    worker = ImageWorker(
        repository=FakeRepository(),
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        fulltext_downloader=fulltext,
    )
    task = {
        **_task(),
        "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd",
        "context_before": "Located before",
        "context_after": "Located after",
        "img_meta": {"mmd_context_located": True},
    }

    result = await worker.process_task(task)

    assert result.success is True
    assert fulltext.calls == []
    status = worker.repository.success_calls[0]["result_summary"]["mmd_context_status"]
    assert status["mmd_download_skipped"] is True
    assert status["mmd_context_source"] == "task"


@pytest.mark.asyncio
async def test_process_task_reuses_mmd_context_cache_for_same_document():
    content = _png_bytes(width=320, height=240)
    mmd_text = "\n\n".join(
        [
            "Before one.",
            "![Figure 1](figures/fig1.png)",
            "After one.",
            "Before two.",
            "![Figure 2](figures/fig2.png)",
            "After two.",
        ]
    )
    fulltext = FakeFullTextDownloader(
        FullTextDownloadResult(
            relative_url="/fulltext/doc-1/doc-1.mmd",
            url="https://static.nstl.gov.cn/fulltext/doc-1/doc-1.mmd",
            text=mmd_text,
            char_count=len(mmd_text),
        )
    )
    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        fulltext_downloader=fulltext,
    )

    result1 = await worker.process_task(
        {**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
    )
    result2 = await worker.process_task(
        {
            **_task(),
            "task_id": "task-2",
            "image_id": "doc-1_img_2",
            "raw_img_meta": {**_task()["raw_img_meta"], "id": "doc-1_img_2"},
            "sequence": "2",
            "name": "fig2.png",
            "relative_url": "/images/fig2.png",
            "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd",
        }
    )

    assert result1.success is True
    assert result2.success is True
    assert fulltext.calls == ["/fulltext/doc-1/doc-1.mmd"]
    assert repository.success_calls[1]["result_summary"]["mmd_context_status"][
        "mmd_context_available"
    ] is True


@pytest.mark.asyncio
async def test_process_task_shares_inflight_mmd_download_for_same_document():
    content = _png_bytes(width=320, height=240)
    mmd_text = "\n\n".join(
        [
            "Before one.",
            "![Figure 1](figures/fig1.png)",
            "After one.",
            "Before two.",
            "![Figure 2](figures/fig2.png)",
            "After two.",
        ]
    )
    gate = asyncio.Event()
    started = 0

    class SlowFullTextDownloader(FakeFullTextDownloader):
        async def download_text(self, relative_url: str) -> FullTextDownloadResult:
            nonlocal started
            started += 1
            if started == 1:
                gate.set()
                await asyncio.sleep(0)
            else:
                await gate.wait()
            return await super().download_text(relative_url)

    fulltext = SlowFullTextDownloader(
        FullTextDownloadResult(
            relative_url="/fulltext/doc-1/doc-1.mmd",
            url="https://static.nstl.gov.cn/fulltext/doc-1/doc-1.mmd",
            text=mmd_text,
            char_count=len(mmd_text),
        )
    )
    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        fulltext_downloader=fulltext,
    )
    task1 = {**_task(), "task_id": "task-1", "image_id": "doc-1_img_1"}
    task2 = {
        **_task(),
        "task_id": "task-2",
        "image_id": "doc-1_img_2",
        "raw_img_meta": {**_task()["raw_img_meta"], "id": "doc-1_img_2"},
        "sequence": "2",
        "name": "fig2.png",
        "relative_url": "/images/fig2.png",
        "img_meta": {"page": 2},
    }

    result1, result2 = await asyncio.gather(
        worker.process_task({**task1, "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}),
        worker.process_task({**task2, "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}),
    )

    assert result1.success is True
    assert result2.success is True
    assert fulltext.calls == ["/fulltext/doc-1/doc-1.mmd"]
    assert started == 1
    assert len(repository.success_calls) == 2


@pytest.mark.asyncio
async def test_process_task_with_injected_providers_uses_feature_pipeline(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_VECTOR_SCHEMA_MODE", "four_vector")
    content = _png_bytes(width=320, height=240)
    worker = ImageWorker(
        repository=FakeRepository(),
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        feature_providers=_feature_providers(),
        image_write_enabled=False,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    result_summary = worker.repository.success_calls[0]["result_summary"]
    feature_summary = result_summary["feature_summary"]
    assert result_summary["feature_success"] is True
    assert feature_summary["p_hash_present"] is True
    assert feature_summary["visual_dim"] == VISUAL_VECTOR_DIM
    assert feature_summary["caption_dim"] == TEXT_VECTOR_DIM
    assert feature_summary["context_dim"] == TEXT_VECTOR_DIM
    assert feature_summary["text_dim"] is None
    assert feature_summary["visual_text_dim"] is None
    assert feature_summary["sparse_terms"] is None
    assert worker.state_manager.artifacts == []


@pytest.mark.asyncio
async def test_feature_enabled_without_real_provider_config_is_non_retryable(monkeypatch):
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    monkeypatch.setattr(settings, "IMAGE_MODEL_API_KEY", "")
    monkeypatch.setattr(settings, "IMAGE_VL_EMBED_API_KEY", "")
    monkeypatch.setattr(settings, "IMAGE_TEXT_EMBED_API_KEY", "")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", False)

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline_enabled=True,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "IMAGE_FEATURE_PIPELINE_ERROR"
    assert result.retryable is False
    assert repository.success_calls == []
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": "IMAGE_FEATURE_PIPELINE_ERROR",
            "error_message": "image_provider_config:IMAGE_PROVIDER_CONFIG_MISSING:missing image provider configuration: IMAGE_TEXT_EMBED_API_KEY|IMAGE_MODEL_API_KEY|EMBEDDING_API_KEY",
            "retryable": False,
        }
    ]
    summary = state_manager.artifacts[1]["content"]
    assert summary["feature_success"] is False
    assert summary["failure_items"] == [
        {
            "feature_name": "image_provider_config",
            "error_code": "IMAGE_PROVIDER_CONFIG_MISSING",
            "error_message": "missing image provider configuration: IMAGE_TEXT_EMBED_API_KEY|IMAGE_MODEL_API_KEY|EMBEDDING_API_KEY",
            "retryable": False,
        }
    ]


@pytest.mark.asyncio
async def test_feature_success_with_record_without_writer_does_not_write_milvus():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=_feature_pipeline_with_record(_writable_image_record()),
        image_write_enabled=False,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.artifact_refs == []
    assert repository.failure_calls == []
    result_summary = repository.success_calls[0]["result_summary"]
    assert "milvus_write_enabled" not in result_summary
    assert "milvus_write_success" not in result_summary
    assert "milvus_write_summary" not in result_summary
    assert state_manager.artifacts == []


@pytest.mark.asyncio
async def test_feature_success_with_injected_writer_saves_milvus_write_summary():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    image_record = _writable_image_record()
    writer = FakeImageWriter(
        result={
            "delete_result": {"delete_count": 1},
            "insert_result": {"insert_count": 1},
        }
    )

    async def fake_pipeline(*_, **__):
        return SimpleNamespace(
            success=True,
            image_id=image_record.img_id,
            doc_id=image_record.docId,
            feature_summary=SimpleNamespace(p_hash_present=True),
            failure_items=[],
            image_record=image_record,
        )

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=fake_pipeline,
        image_writer=writer,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.artifact_refs == []
    assert writer.calls == []
    assert writer.insert_calls == []
    assert writer.upsert_calls == [
        {
            "collection_name": settings.IMAGE_COLLECTION_NAME,
            "images": [image_record],
        }
    ]
    assert writer.insert_or_skip_calls == []
    assert repository.failure_calls == []
    assert repository.success_calls[0]["artifact_refs"] == []
    result_summary = repository.success_calls[0]["result_summary"]
    assert result_summary["milvus_write_enabled"] is True
    assert result_summary["milvus_write_success"] is True
    assert result_summary["milvus_write_mode"] == "upsert"
    assert result_summary["milvus_write_summary"]["write_success"] is True
    assert worker.stage_executor_status() == {}

    assert state_manager.artifacts == []
    summary = result_summary["milvus_write_summary"]
    assert summary["task_id"] == "task-1"
    assert summary["image_id"] == "doc-1_img_1"
    assert summary["doc_id"] == "doc-1"
    assert summary["collection_name"] == settings.IMAGE_COLLECTION_NAME
    assert summary["write_success"] is True
    assert summary["artifact_refs"] == []
    assert summary["failure_details"] is None
    assert "content" not in summary
    diagnostics = result_summary["image_diagnostics"]
    assert diagnostics["feature_stage_timings_ms"]["download"] >= 0
    assert diagnostics["feature_stage_timings_ms"]["feature_total"] >= 0
    assert diagnostics["milvus_write_ms"] >= 0
    assert "imageBytes" not in summary
    assert "image_record" not in summary
    assert "visual_embed" not in summary
    assert "text_embed" not in summary
    assert "visual_text_embed" not in summary
    assert "text_sparse_vec" not in summary
    assert "p_hash" not in summary


@pytest.mark.asyncio
async def test_no_delete_mode_inserts_image_without_replace(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_E2E_NO_DELETE", True)
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    image_record = _writable_image_record()
    writer = FakeImageWriter()

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=_feature_pipeline_with_record(image_record),
        image_writer=writer,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert writer.calls == []
    assert writer.insert_calls == [
        {
            "collection_name": settings.IMAGE_COLLECTION_NAME,
            "images": [image_record],
        }
    ]
    write_summary = repository.success_calls[0]["result_summary"]["milvus_write_summary"]
    assert write_summary["delete_result"] == {
        "skipped": True,
        "status": "DELETE_NOOP_RECORDED",
        "reason": "LIVE_E2E_NO_DELETE",
        "operation": "replace_image_predelete",
        "collection_name": settings.IMAGE_COLLECTION_NAME,
        "img_id": image_record.img_id,
    }
    assert write_summary["insert_result"] == {"insert_count": 1}


@pytest.mark.asyncio
async def test_replace_write_mode_maps_to_safe_primary_key_upsert(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MILVUS_WRITE_MODE", "replace")
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    image_record = _writable_image_record()
    writer = FakeImageWriter()

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=_feature_pipeline_with_record(image_record),
        image_writer=writer,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert writer.calls == []
    assert writer.upsert_calls == [
        {
            "collection_name": settings.IMAGE_COLLECTION_NAME,
            "images": [image_record],
        }
    ]


@pytest.mark.asyncio
async def test_minimal_success_result_mode_writes_small_summary(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_SUCCESS_RESULT_MODE", "minimal")
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
    )

    result = await worker.process_task(_task())

    assert result.success is True
    result_summary = repository.success_calls[0]["result_summary"]
    assert result_summary["task_id"] == "task-1"
    assert result_summary["image_id"] == "doc-1_img_1"
    assert result_summary["feature_pipeline_enabled"] is False
    assert result_summary["milvus_write_mode"] == "upsert"
    assert "download_url" not in result_summary
    assert "image_diagnostics" not in result_summary
    assert "artifact_refs" not in result_summary


@pytest.mark.asyncio
async def test_status_only_success_result_mode_writes_smallest_summary(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_SUCCESS_RESULT_MODE", "status_only")
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
    )

    result = await worker.process_task(_task())

    assert result.success is True
    result_summary = repository.success_calls[0]["result_summary"]
    assert result_summary == {
        "task_id": "task-1",
        "image_id": "doc-1_img_1",
        "success_stage": "download",
        "milvus_write_mode": "upsert",
    }
    assert repository.success_minimal_calls == [
        {
            "task_id": "task-1",
            "image_id": "doc-1_img_1",
            "success_stage": "download",
            "milvus_write_mode": "upsert",
            "artifact_refs": [],
        }
    ]


@pytest.mark.asyncio
async def test_writer_value_error_marks_non_retryable_write_failure():
    repository = FakeRepository()
    state_manager = FakeStateManager()
    writer = FakeImageWriter(error=ValueError("missing p_hash"))

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        feature_pipeline=_feature_pipeline_with_record(_writable_image_record()),
        image_writer=writer,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "IMAGE_MILVUS_WRITE_ERROR"
    assert result.retryable is False
    assert result.artifact_refs == ["artifact-1", "artifact-2", "artifact-3"]
    assert repository.success_calls == []
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": "IMAGE_MILVUS_WRITE_ERROR",
            "error_message": "missing p_hash",
            "retryable": False,
        }
    ]
    summary = state_manager.artifacts[2]["content"]
    assert summary["write_success"] is False
    assert summary["failure_details"] == {
        "error_code": "IMAGE_MILVUS_WRITE_ERROR",
        "error_message": "missing p_hash",
        "retryable": False,
        "exception_type": "ValueError",
    }


@pytest.mark.asyncio
async def test_writer_runtime_error_marks_retryable_write_failure():
    repository = FakeRepository()
    writer = FakeImageWriter(error=RuntimeError("milvus unavailable"))

    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        feature_pipeline=_feature_pipeline_with_record(_writable_image_record()),
        image_writer=writer,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "IMAGE_MILVUS_WRITE_ERROR"
    assert result.retryable is False
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": "IMAGE_MILVUS_WRITE_ERROR",
            "error_message": "milvus unavailable",
            "retryable": False,
        }
    ]


@pytest.mark.asyncio
async def test_success_path_does_not_depend_on_write_summary_artifact():
    repository = FakeRepository()
    state_manager = FakeStateManager(
        error=RuntimeError("mongo down"),
        error_on_artifact_type=ArtifactType.MILVUS_WRITE_SUMMARY,
    )
    writer = FakeImageWriter(
        result={
            "delete_result": {"delete_count": 1},
            "insert_result": {"insert_count": 1},
        }
    )

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        feature_pipeline=_feature_pipeline_with_record(_writable_image_record()),
        image_writer=writer,
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.error_code is None
    assert result.artifact_refs == []
    assert writer.calls or writer.insert_calls or writer.upsert_calls or writer.insert_or_skip_calls
    assert repository.failure_calls == []
    assert state_manager.artifacts == []


@pytest.mark.asyncio
async def test_feature_pipeline_failure_marks_retryable_task_failure():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()

    async def fake_pipeline(*_, **__):
        return SimpleNamespace(
            success=False,
            feature_summary=SimpleNamespace(visual_dim=None),
            failure_items=[
                {
                    "feature_name": "visual",
                    "error_code": "VISUAL_TEMPORARY",
                    "error_message": "temporary visual failure",
                    "retryable": True,
                }
            ],
        )

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=fake_pipeline,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "IMAGE_FEATURE_PIPELINE_ERROR"
    assert result.retryable is False
    assert result.artifact_refs == ["artifact-1", "artifact-2"]
    assert repository.success_calls == []
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": "IMAGE_FEATURE_PIPELINE_ERROR",
            "error_message": "visual:VISUAL_TEMPORARY:temporary visual failure",
            "retryable": False,
        }
    ]
    summary = state_manager.artifacts[1]["content"]
    assert summary["feature_success"] is False
    assert summary["failure_items"][0]["retryable"] is True
    assert "content" not in summary
    assert "imageBytes" not in summary


@pytest.mark.asyncio
async def test_feature_pipeline_exception_records_summary_and_retryable_failure():
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()

    async def fake_pipeline(*_, **__):
        raise RuntimeError("pipeline exploded")

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=fake_pipeline,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "IMAGE_FEATURE_PIPELINE_ERROR"
    assert result.retryable is False
    assert repository.failure_calls[0]["error_code"] == "IMAGE_FEATURE_PIPELINE_ERROR"
    assert repository.failure_calls[0]["retryable"] is False
    summary = state_manager.artifacts[1]["content"]
    assert summary["feature_success"] is False
    assert summary["failure_items"] == [
        {
            "feature_name": "image_feature_pipeline",
            "error_code": "IMAGE_FEATURE_PIPELINE_ERROR",
            "error_message": "pipeline exploded",
            "retryable": True,
        }
    ]


@pytest.mark.asyncio
async def test_feature_pipeline_total_timeout_records_retryable_failure(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_FEATURE_TOTAL_TIMEOUT_SEC", 0.01)
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    release = asyncio.Event()

    async def slow_pipeline(*_, **__):
        await release.wait()

    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=slow_pipeline,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == "IMAGE_FEATURE_PIPELINE_ERROR"
    assert result.retryable is False
    assert repository.failure_calls[0]["error_code"] == "IMAGE_FEATURE_PIPELINE_ERROR"
    assert "IMAGE_FEATURE_PIPELINE_TIMEOUT" in repository.failure_calls[0]["error_message"]
    summary = state_manager.artifacts[1]["content"]
    assert summary["feature_success"] is False
    assert summary["failure_items"][0]["error_code"] == "IMAGE_FEATURE_PIPELINE_TIMEOUT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "error_code", "retryable"),
    [
        (ImageConfigurationError("missing base"), "IMAGE_CONFIG_ERROR", False),
        (
            ImageRetryableDownloadError("network down"),
            "IMAGE_DOWNLOAD_RETRYABLE",
            False,
        ),
        (
            ImageNonRetryableDownloadError("not found"),
            "IMAGE_DOWNLOAD_NON_RETRYABLE",
            False,
        ),
    ],
)
async def test_download_failures_mark_task_failure(error, error_code, retryable):
    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(error=error),
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == error_code
    assert result.retryable is retryable
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": error_code,
            "error_message": str(error),
            "retryable": retryable,
        }
    ]


@pytest.mark.asyncio
async def test_invalid_task_marks_non_retryable_failure():
    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(error=AssertionError("should not download")),
    )

    result = await worker.process_task({"task_id": "task-1"})

    assert result.success is False
    assert result.error_code == "IMAGE_TASK_INVALID"
    assert result.retryable is False
    assert repository.failure_calls[0]["error_code"] == "IMAGE_TASK_INVALID"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_case", "error_code"),
    [
        ("empty", "IMAGE_BYTES_EMPTY"),
        ("invalid", "IMAGE_DIMENSION_DECODE_ERROR"),
    ],
)
async def test_dimension_validation_failures_mark_non_retryable_failure(
    content_case,
    error_code,
):
    content = {
        "empty": b"",
        "invalid": b"not-an-image",
    }[content_case]
    repository = FakeRepository()
    state_manager = FakeStateManager()
    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(
            ImageDownloadResult(
                relative_url="/images/fig1.png",
                url="http://files.test/images/fig1.png",
                content=content,
                content_type="image/png",
                size_bytes=len(content),
            )
        ),
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert result.error_code == error_code
    assert result.retryable is False
    assert state_manager.artifacts == []
    assert repository.failure_calls == [
        {
            "task_id": "task-1",
            "error_code": error_code,
            "error_message": result.error_message,
            "retryable": False,
        }
    ]


@pytest.mark.asyncio
async def test_oversize_image_is_recorded_and_allowed_for_model_downsample():
    content = _png_bytes(width=2000, height=1001)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert repository.failure_calls == []
    summary = repository.success_calls[0]["result_summary"]
    assert summary["width"] == 2000
    assert summary["height"] == 1001
    assert summary["pixel_count"] == 2_002_000
    assert summary["oversize_for_model"] is True
    assert state_manager.artifacts == []
    assert summary["image_diagnostics"]["original_pixel_count"] == 2_002_000
    assert summary["image_diagnostics"]["oversize_for_model"] is True


@pytest.mark.asyncio
async def test_manifest_validation_failure_marks_non_retryable_failure():
    repository = FakeRepository()
    state_manager = FakeStateManager()
    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(
            ImageDownloadResult(
                relative_url="/images/fig1.png",
                url="http://files.test/images/fig1.png",
                content=_png_bytes(width=320, height=240),
                content_type="image/png",
                size_bytes=len(_png_bytes(width=320, height=240)),
            )
        ),
    )

    result = await worker.process_task({**_task(), "image_id": "doc-1_img_2"})

    assert result.success is False
    assert result.error_code == "IMAGE_MANIFEST_VALIDATION_FAILED"
    assert result.retryable is False
    assert "image_id:IMAGE_MANIFEST_INVALID_IMAGE_ID" in result.error_message
    assert state_manager.artifacts == []
    assert repository.failure_calls[0]["error_code"] == (
        "IMAGE_MANIFEST_VALIDATION_FAILED"
    )
    assert repository.failure_calls[0]["retryable"] is False
    failure = state_manager.persisted_failures[-1]
    assert failure["pipeline"] == "image"
    assert failure["source_version"] == "2026-06-07T00:00:00+00:00"
    assert failure["item_id"] == "doc-1_img_2"
    assert failure["logical_collection"] == failure["resolved_collection"]
    assert failure["target_signature"].startswith("IMAGE|")


@pytest.mark.asyncio
async def test_success_path_does_not_depend_on_manifest_artifact():
    repository = FakeRepository()
    state_manager = FakeStateManager(error=RuntimeError("mongo down"))
    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(
            ImageDownloadResult(
                relative_url="/images/fig1.png",
                url="http://files.test/images/fig1.png",
                content=_png_bytes(width=320, height=240),
                content_type="image/png",
                size_bytes=len(_png_bytes(width=320, height=240)),
            )
        ),
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.error_code is None
    assert repository.failure_calls == []


@pytest.mark.asyncio
async def test_success_summary_mode_skips_success_artifacts_and_runtime_attachments(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "IMAGE_SUCCESS_ARTIFACT_MODE", "summary")
    monkeypatch.setattr(settings, "IMAGE_RUNTIME_ATTACHMENT_MODE", "failure_only")
    content = _png_bytes(width=320, height=240)
    repository = FakeRepository()
    state_manager = FakeStateManager()
    worker = ImageWorker(
        repository=repository,
        state_manager=state_manager,
        downloader=FakeDownloader(_download_result(content)),
    )

    result = await worker.process_task(_task())

    assert result.success is True
    assert result.artifact_refs == []
    assert state_manager.artifacts == []
    assert repository.success_calls[0]["artifact_refs"] == []
    assert repository.success_calls[0]["result_summary"]["success_stage"] == "download"
    assert not (tmp_path / "upload" / "attachments").exists()


@pytest.mark.asyncio
async def test_failure_only_runtime_writes_failed_assets_on_feature_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "IMAGE_RUNTIME_ATTACHMENT_MODE", "failure_only")
    monkeypatch.setattr(settings, "IMAGE_SUCCESS_ARTIFACT_MODE", "summary")
    content = _png_bytes(width=320, height=240)

    async def fake_pipeline(*_, **__):
        return SimpleNamespace(
            success=False,
            feature_summary=SimpleNamespace(visual_dim=None),
            failure_items=[
                {
                    "feature_name": "visual",
                    "error_code": "VISUAL_FAILED",
                    "error_message": "visual failed",
                    "retryable": False,
                }
            ],
        )

    worker = ImageWorker(
        repository=FakeRepository(),
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(content)),
        feature_pipeline=fake_pipeline,
    )

    result = await worker.process_task(_task())

    assert result.success is False
    assert not (tmp_path / "upload" / "attachments").exists()
    failed_manifest = (
        tmp_path / "upload" / "failed" / "doc-1" / "task-1" / "manifest.json"
    )
    assert failed_manifest.exists()
    assert (tmp_path / "upload" / "failed" / "doc-1" / "task-1" / "fig1.png").exists()


def test_image_worker_source_guard_excludes_external_clients_and_deletes():
    source = inspect.getsource(image_worker_module)

    forbidden = [
        "httpx",
        "requests",
        "openai",
        "DELETE_IMAGES",
    ]
    for text in forbidden:
        assert text not in source


@pytest.mark.asyncio
async def test_default_pipeline_starts_visual_stage_before_mmd_finishes():
    visual_started = asyncio.Event()
    release_mmd = asyncio.Event()

    async def visual_provider(_context):
        visual_started.set()
        return [1.0] * VISUAL_VECTOR_DIM

    class DelayedFullTextDownloader(FakeFullTextDownloader):
        async def download_text(self, relative_url: str) -> FullTextDownloadResult:
            await release_mmd.wait()
            return FullTextDownloadResult(
                relative_url=relative_url,
                url="https://static.nstl.gov.cn/fulltext/doc-1/doc-1.mmd",
                text="Before one.\n\n![Figure 1](figures/fig1.png)\n\nAfter one.",
                char_count=60,
            )

    providers = _feature_providers()
    providers.visual_embed_provider = visual_provider
    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        fulltext_downloader=DelayedFullTextDownloader(),
        feature_providers=providers,
        feature_pipeline_enabled=True,
        image_write_enabled=False,
    )
    task = asyncio.create_task(
        worker.process_task({**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"})
    )
    await asyncio.wait_for(visual_started.wait(), timeout=1)
    assert not task.done()
    release_mmd.set()
    result = await task
    assert result.success is True
    assert repository.success_calls[0]["result_summary"]["mmd_context_status"][
        "mmd_context_located"
    ] is True


@pytest.mark.asyncio
async def test_default_pipeline_does_not_write_when_required_mmd_cannot_match():
    repository = FakeRepository()
    worker = ImageWorker(
        repository=repository,
        state_manager=FakeStateManager(),
        downloader=FakeDownloader(_download_result(_png_bytes(width=320, height=240))),
        fulltext_downloader=FakeFullTextDownloader(
            FullTextDownloadResult(
                relative_url="/fulltext/doc-1/doc-1.mmd",
                url="https://static.nstl.gov.cn/fulltext/doc-1/doc-1.mmd",
                text="![Other image](figures/other.png)",
                char_count=34,
            )
        ),
        feature_providers=_feature_providers(),
        feature_pipeline_enabled=True,
        image_write_enabled=False,
    )
    result = await worker.process_task(
        {**_task(), "mmd_relative_url": "/fulltext/doc-1/doc-1.mmd"}
    )
    assert result.success is False
    assert result.error_code == "IMAGE_MMD_CONTEXT_REQUIRED"
    assert repository.success_calls == []


def _task() -> dict[str, Any]:
    return {
        "task_id": "task-1",
        "doc_id": "doc-1",
        "record_update_time": "2026-06-07T00:00:00+00:00",
        "relative_url": "/images/fig1.png",
        "image_id": "doc-1_img_1",
        "sequence": "1",
        "doc_title": ["Document 1"],
        "keyword": ["science"],
        "accessType": "OA",
        "classificationCode": "Q",
        "year": "2026",
        "hasAuthorname": ["Alice"],
        "hasCreateOrganizationname": ["NSTL"],
        "language": ["zh"],
        "sysDataStatus": "ACTIVE",
        "hasSourcesysSourceId": ["SRC-1"],
        "hasSourceidSet": ["JOURNAL-1"],
        "title": "Figure 1",
        "name": "fig1.png",
        "width": 800,
        "height": 600,
        "file_type": "png",
        "size": 12345,
        "raw_img_meta": {
            "id": "doc-1_img_1",
            "cls": "PictureFile",
            "sequence": "1",
            "size": 12345,
            "width": 800,
            "name": "fig1.png",
            "title": ["Figure 1"],
            "url": ["/images/fig1.png"],
            "fileType": "png",
            "height": 600,
            "sysDataType": "file",
        },
        "img_meta": {"page": 1},
        "context_before": None,
        "context_after": "Figure 1 caption",
        "source_item_index": 0,
        "image_index": 0,
    }


def _png_bytes(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def _download_result(content: bytes) -> ImageDownloadResult:
    return ImageDownloadResult(
        relative_url="/images/fig1.png",
        url="http://files.test/images/fig1.png",
        content=content,
        content_type="image/png",
        size_bytes=len(content),
    )


def _feature_providers() -> ImageFeatureProviders:
    return ImageFeatureProviders(
        p_hash_provider=lambda _: b"\x01\x02\x03\x04\x05\x06\x07\x08",
        visual_embed_provider=lambda _: [1.0] * VISUAL_VECTOR_DIM,
        caption_embed_provider=lambda _: [1.0] * TEXT_VECTOR_DIM,
        context_embed_provider=lambda _: [1.0] * TEXT_VECTOR_DIM,
        text_embed_provider=lambda _: [1.0] * TEXT_VECTOR_DIM,
        visual_text_embed_provider=lambda _: [1.0] * VISUAL_TEXT_VECTOR_DIM,
        text_sparse_vec_provider=lambda _: {1: 0.5},
        bm25_text_provider=lambda _: "Figure 1 caption",
    )


def _writable_image_record() -> ImageRecord:
    return ImageRecord(
        img_id="doc-1_img_1",
        docId="doc-1",
        url="http://files.test/images/fig1.png",
        title="Figure 1",
        keyword=["science"],
        language=["zh"],
        sysDataStatus="ACTIVE",
        name="fig1.png",
        width=320,
        height=240,
        sequence=1,
        fileType="png",
        size=1024,
        p_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        visual_embed=[1.0] * VISUAL_VECTOR_DIM,
        caption_embed=[1.0] * TEXT_VECTOR_DIM,
        context_embed=[1.0] * TEXT_VECTOR_DIM,
        text_embed=[1.0] * TEXT_VECTOR_DIM,
        visual_text_embed=[1.0] * VISUAL_TEXT_VECTOR_DIM,
        text="Figure 1 caption",
        text_sparse_vec={1: 0.5},
        doc_meta={"doc_id": "doc-1", "language": ["zh"], "sysDataStatus": "ACTIVE"},
        img_meta={"task_id": "task-1"},
    )


def _feature_pipeline_with_record(image_record: ImageRecord):
    async def fake_pipeline(*_, **__):
        return SimpleNamespace(
            success=True,
            image_id=image_record.img_id,
            doc_id=image_record.docId,
            feature_summary=SimpleNamespace(p_hash_present=True),
            failure_items=[],
            image_record=image_record,
        )

    return fake_pipeline


class FakeDownloader:
    def __init__(
        self,
        result: ImageDownloadResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    async def download_bytes(self, relative_url: str) -> ImageDownloadResult:
        self.calls.append(relative_url)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeFullTextDownloader:
    def __init__(
        self,
        result: FullTextDownloadResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    async def download_text(self, relative_url: str) -> FullTextDownloadResult:
        self.calls.append(relative_url)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeRepository:
    def __init__(self) -> None:
        self.success_calls: list[dict[str, Any]] = []
        self.success_minimal_calls: list[dict[str, Any]] = []
        self.failure_calls: list[dict[str, Any]] = []

    async def mark_task_success(
        self,
        task_id: str,
        *,
        artifact_refs: list[str],
        result_summary: dict[str, Any],
    ) -> None:
        self.success_calls.append(
            {
                "task_id": task_id,
                "artifact_refs": artifact_refs,
                "result_summary": result_summary,
            }
        )

    async def mark_task_success_minimal(
        self,
        task_id: str,
        *,
        image_id: str,
        success_stage: str,
        milvus_write_mode: str,
        artifact_refs: list[str] | None = None,
    ) -> None:
        payload = {
            "task_id": task_id,
            "image_id": image_id,
            "success_stage": success_stage,
            "milvus_write_mode": milvus_write_mode,
            "artifact_refs": list(artifact_refs or []),
        }
        self.success_minimal_calls.append(payload)
        self.success_calls.append(
            {
                "task_id": task_id,
                "artifact_refs": list(artifact_refs or []),
                "result_summary": {
                    "task_id": task_id,
                    "image_id": image_id,
                    "success_stage": success_stage,
                    "milvus_write_mode": milvus_write_mode,
                },
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
        self.failure_calls.append(
            {
                "task_id": task_id,
                "error_code": error_code,
                "error_message": error_message,
                "retryable": retryable,
            }
        )


class FakeStateManager:
    def __init__(
        self,
        error: Exception | None = None,
        *,
        error_on_artifact_type: ArtifactType | None = None,
    ) -> None:
        self.error = error
        self.error_on_artifact_type = error_on_artifact_type
        self.artifacts: list[dict[str, Any]] = []
        self.refresh_calls: list[dict[str, Any]] = []
        self.persisted_failures: list[dict[str, Any]] = []

    async def save_artifact(self, **kwargs) -> str:
        if self.error is not None and (
            self.error_on_artifact_type is None
            or kwargs.get("artifact_type") == self.error_on_artifact_type
        ):
            raise self.error
        self.artifacts.append(kwargs)
        return f"artifact-{len(self.artifacts)}"

    async def refresh_async_pipeline_status(self, **kwargs) -> None:
        self.refresh_calls.append(kwargs)

    async def persist_failure(self, **kwargs) -> None:
        self.persisted_failures.append(kwargs)


class FakeImageWriter:
    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or {
            "delete_result": {"delete_count": 0},
            "insert_result": {"insert_count": 1},
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.insert_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.insert_or_skip_calls: list[dict[str, Any]] = []

    def replace_image_by_img_id(
        self,
        collection_name: str,
        image_record: ImageRecord,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "collection_name": collection_name,
                "image_record": image_record,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result

    def insert_images(
        self,
        collection_name: str,
        images: list[ImageRecord],
    ) -> dict[str, Any]:
        self.insert_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
            }
        )
        if self.error is not None:
            raise self.error
        return {"insert_count": len(images)}

    def upsert_images_by_img_id_batch(
        self,
        collection_name: str,
        images: list[ImageRecord],
    ) -> dict[str, Any]:
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
            }
        )
        if self.error is not None:
            raise self.error
        return {
            "delete_results": [{"delete_count": 0}],
            "insert_result": {"insert_count": len(images)},
            "existing_count": 0,
            "new_count": len(images),
            "write_mode": "upsert_if_exists",
        }

    def insert_or_skip_images_by_img_id_batch(
        self,
        collection_name: str,
        images: list[ImageRecord],
    ) -> dict[str, Any]:
        self.insert_or_skip_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
            }
        )
        if self.error is not None:
            raise self.error
        return {
            "delete_results": [],
            "insert_result": {"insert_count": len(images)},
            "existing_count": 0,
            "new_count": len(images),
            "skipped_existing_count": 0,
            "write_mode": "insert_or_skip_if_exists",
        }
