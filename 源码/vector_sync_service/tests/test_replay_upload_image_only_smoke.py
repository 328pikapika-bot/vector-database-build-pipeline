from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from app.services.image_feature_pipeline import ImageFeatureProviders
from app.services.milvus_image_service import (
    TEXT_VECTOR_DIM,
    VISUAL_TEXT_VECTOR_DIM,
    VISUAL_VECTOR_DIM,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "replay_upload_image_only_smoke.py"
spec = importlib.util.spec_from_file_location("replay_upload_image_only_smoke", SCRIPT_PATH)
smoke = importlib.util.module_from_spec(spec)
sys.modules["replay_upload_image_only_smoke"] = smoke
assert spec.loader is not None
spec.loader.exec_module(smoke)


def test_scan_selects_unique_samples_and_prefers_active_manifest(tmp_path):
    upload_dir = tmp_path / "upload"
    image_a = b"same-image-a"
    image_b = b"unique-image-b"

    active_a = _write_upload_manifest(
        upload_dir,
        state="attachments",
        doc_id="doc-a",
        task_id="task-a",
        image_name="fig-a.jpg",
        image_bytes=image_a,
    )
    _write_upload_manifest(
        upload_dir,
        state="failed",
        doc_id="doc-a",
        task_id="task-a-old",
        image_name="fig-a.jpg",
        image_bytes=image_a,
        failure_code="IMAGE_FEATURE_PIPELINE_ERROR",
    )
    fallback_b = _write_upload_manifest(
        upload_dir,
        state="failed",
        doc_id="doc-b",
        task_id="task-b",
        image_name="fig-b.jpg",
        image_bytes=image_b,
        local_path_points_elsewhere=True,
        failure_code="IMAGE_PIXEL_LIMIT_EXCEEDED",
    )

    samples = smoke.scan_upload_samples(upload_dir)
    selected = smoke.select_samples(samples, limit=10, unique_only=True)

    assert len(samples) == 3
    assert len(selected) == 2
    assert selected[0].manifest_path == active_a
    assert selected[0].upload_state == "attachments"
    assert selected[1].manifest_path == fallback_b
    assert selected[1].image_path == fallback_b.parent / "fig-b.jpg"


def test_build_pipeline_input_reconstructs_context_and_stable_smoke_id(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.settings, "IMAGE_CONTEXT_BEFORE_PARAGRAPHS", 1)
    monkeypatch.setattr(smoke.settings, "IMAGE_CONTEXT_AFTER_PARAGRAPHS", 2)
    manifest_path = _write_upload_manifest(
        tmp_path / "upload",
        state="attachments",
        doc_id="doc-1",
        task_id="task-1",
        image_name="fig-1.png",
        image_bytes=_png_bytes(width=320, height=240),
        mmd_text="\n\n".join(
            [
                "Before paragraph.",
                "![Figure](figures/fig-1.png)",
                "Caption paragraph.",
                "After paragraph.",
            ]
        ),
    )
    sample = smoke.scan_upload_samples(tmp_path / "upload")[0]

    pipeline_input = smoke.build_pipeline_input(sample, sequence=3)
    draft = pipeline_input.draft_record

    assert sample.manifest_path == manifest_path
    assert draft.img_id == "doc-1_img_3"
    assert draft.docId == "doc-1"
    assert draft.sequence == 3
    assert draft.language == ["unknown"]
    assert draft.sysDataStatus == "ACTIVE"
    assert draft.context_before == "Before paragraph."
    assert "Caption paragraph." in draft.context_after
    assert "After paragraph." in draft.context_after
    assert "Before paragraph." in pipeline_input.context_text
    assert draft.img_meta["source_task_id"] == "task-1"
    assert draft.img_meta["mmd_context_located"] is True
    assert draft.img_meta["oversize_for_model"] is False
    assert draft.visual_embed is None
    assert draft.text_embed is None
    assert draft.visual_text_embed is None


def test_build_pipeline_input_marks_oversize_without_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.settings, "IMAGE_MAX_PIXELS", 2_000_000)
    _write_upload_manifest(
        tmp_path / "upload",
        state="failed",
        doc_id="doc-big",
        task_id="task-big",
        image_name="big.png",
        image_bytes=_png_bytes(width=3000, height=1000),
        failure_code="IMAGE_PIXEL_LIMIT_EXCEEDED",
    )
    sample = smoke.scan_upload_samples(tmp_path / "upload")[0]

    pipeline_input = smoke.build_pipeline_input(sample, sequence=1)

    assert pipeline_input.diagnostics["oversize_for_model"] is True
    assert pipeline_input.diagnostics["original_pixel_count"] == 3_000_000
    assert pipeline_input.draft_record.width == 3000
    assert pipeline_input.draft_record.height == 1000
    assert pipeline_input.draft_record.img_meta["oversize_for_model"] is True


@pytest.mark.asyncio
async def test_process_sample_feature_success_report_has_no_raw_vectors(tmp_path):
    _write_upload_manifest(
        tmp_path / "upload",
        state="attachments",
        doc_id="doc-ok",
        task_id="task-ok",
        image_name="ok.png",
        image_bytes=_png_bytes(width=16, height=16),
    )
    sample = smoke.scan_upload_samples(tmp_path / "upload")[0]

    result = await smoke.process_sample(
        sample,
        sequence=1,
        total=1,
        providers=_fake_providers(),
        schema_mode="four_vector",
        feature_timeout_sec=5,
        write_milvus=False,
        collection_name="images",
        image_service=None,
        quiet=True,
    )

    assert result["success"] is True
    assert result["feature_success"] is True
    assert result["feature_summary"]["visual_dim"] == VISUAL_VECTOR_DIM
    assert result["feature_summary"]["text_dim"] == TEXT_VECTOR_DIM
    assert result["feature_summary"]["visual_text_dim"] == VISUAL_TEXT_VECTOR_DIM
    assert {item["feature"] for item in result["feature_timings"]} == {
        "p_hash",
        "visual_embed",
        "text_embed",
        "visual_text_embed",
        "bm25_text",
    }
    serialized = json.dumps(result, ensure_ascii=False)
    assert '"visual_embed": [' not in serialized
    assert '"text_embed": [' not in serialized
    assert '"visual_text_embed": [' not in serialized


def test_smoke_script_does_not_touch_fulltext_chunk_or_mongo_task_paths():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    forbidden = [
        "MongoStateManager",
        "get_image_task_repository",
        "MilvusDynamicService",
        "MilvusChunkService",
        "FullTextWorker",
        "KafkaConsumer",
    ]
    for keyword in forbidden:
        assert keyword not in source


def test_cli_defaults_context_cleaning_off_for_smoke_speed():
    args = smoke.parse_args([])

    assert args.enable_context_cleaning is False
    assert args.concurrency >= 1


def test_cli_accepts_concurrency_override():
    args = smoke.parse_args(["--concurrency", "8"])

    assert args.concurrency == 8


def _fake_providers() -> ImageFeatureProviders:
    return ImageFeatureProviders(
        p_hash_provider=lambda _context: b"\x01\x02\x03\x04\x05\x06\x07\x08",
        visual_embed_provider=lambda _context: [1.0] * VISUAL_VECTOR_DIM,
        text_embed_provider=lambda _context: [1.0] * TEXT_VECTOR_DIM,
        visual_text_embed_provider=lambda _context: [1.0] * VISUAL_TEXT_VECTOR_DIM,
        bm25_text_provider=lambda _context: "image interpretation text",
    )


def _write_upload_manifest(
    upload_dir: Path,
    *,
    state: str,
    doc_id: str,
    task_id: str,
    image_name: str,
    image_bytes: bytes,
    mmd_text: str | None = None,
    failure_code: str = "",
    local_path_points_elsewhere: bool = False,
) -> Path:
    sample_dir = upload_dir / state / doc_id / task_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = sample_dir / image_name
    mmd_path = sample_dir / f"{doc_id}.mmd"
    image_path.write_bytes(image_bytes)
    mmd_path.write_text(
        mmd_text
        or "\n\n".join(["Before.", f"![Figure](figures/{image_name})", "After."]),
        encoding="utf-8",
    )
    image_local_path = (
        str(sample_dir / "missing_source_dir" / image_name)
        if local_path_points_elsewhere
        else str(image_path)
    )
    manifest = {
        "doc_id": doc_id,
        "batch_id": task_id,
        "task_id": task_id,
        "image_relative_url": f"/fulltext/{doc_id}/figures/{image_name}",
        "mmd_relative_url": f"/fulltext/{doc_id}/{doc_id}.mmd",
        "downloads": {
            "image": {
                "relative_url": f"/fulltext/{doc_id}/figures/{image_name}",
                "local_path": image_local_path,
                "content_type": "image/png" if image_name.endswith(".png") else "image/jpeg",
                "size_bytes": len(image_bytes),
                "sha256": smoke.sha256(image_bytes).hexdigest(),
            },
            "mmd": {
                "relative_url": f"/fulltext/{doc_id}/{doc_id}.mmd",
                "local_path": str(mmd_path),
                "char_count": len(mmd_path.read_text(encoding="utf-8")),
                "sha256": smoke.sha256(mmd_path.read_bytes()).hexdigest(),
            },
        },
        "context": {
            "mmd_context_available": True,
            "mmd_context_located": True,
            "caption_chars": 5,
        },
        "failure": None,
    }
    if failure_code:
        manifest["failure"] = {
            "error_code": failure_code,
            "error_message": f"{failure_code}: sample failure",
        }
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _png_bytes(*, width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + int(width).to_bytes(4, "big")
        + int(height).to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )
