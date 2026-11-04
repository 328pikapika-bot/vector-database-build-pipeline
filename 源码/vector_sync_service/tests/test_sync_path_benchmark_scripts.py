from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"


def _load_module(name: str, filename: str):
    script_path = SCRIPTS_DIR / filename
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


shared = _load_module("benchmark_shared_mod", "benchmark_shared.py")
full_write = _load_module("run_image_full_write_benchmark_mod", "run_image_full_write_benchmark.py")
isolated_load = _load_module("run_isolated_kafka_load_test_mod", "run_isolated_kafka_load_test.py")


def test_scan_upload_samples_prefers_non_failed_sample(tmp_path):
    upload_dir = tmp_path / "upload"
    _write_manifest(
        upload_dir,
        state="failed",
        doc_id="doc-1",
        task_id="task-old",
        image_name="fig-1.jpg",
        image_bytes=b"old",
    )
    preferred = _write_manifest(
        upload_dir,
        state="attachments",
        doc_id="doc-1",
        task_id="task-new",
        image_name="fig-1.jpg",
        image_bytes=b"newer",
    )

    samples = shared.scan_upload_samples(upload_dir)

    assert len(samples) == 1
    assert samples[0].manifest_path == preferred
    assert samples[0].upload_state == "attachments"


def test_build_pipeline_input_reconstructs_context(tmp_path):
    upload_dir = tmp_path / "upload"
    manifest_path = _write_manifest(
        upload_dir,
        state="attachments",
        doc_id="doc-2",
        task_id="task-2",
        image_name="fig-2.jpg",
        image_bytes=_tiny_jpeg_bytes(),
        mmd_text="\n\n".join(
            [
                "Before paragraph.",
                "![Figure](figures/fig-2.jpg)",
                "Caption paragraph.",
                "After paragraph.",
            ]
        ),
    )
    sample = shared.scan_upload_samples(upload_dir)[0]

    pipeline_input = shared.build_pipeline_input(sample, sequence=3)

    assert sample.manifest_path == manifest_path
    assert pipeline_input.draft_record.img_id == "source-task-2"
    assert pipeline_input.draft_record.context_before == "Before paragraph."
    assert "Caption paragraph." in str(pipeline_input.draft_record.context_after)
    assert pipeline_input.context_text


def test_compare_stage_bottlenecks_returns_lowest_stage():
    report = shared.compare_stage_bottlenecks(
        [
            {"stage": "download", "best_case": {"throughput_per_second": 100}},
            {"stage": "image_model", "best_case": {"throughput_per_second": 12.5}},
            {"stage": "text_embedding", "best_case": {"throughput_per_second": 50}},
        ]
    )

    assert report["bottleneck_stage"] == "image_model"
    assert report["ordered"][0]["stage"] == "image_model"


def test_full_write_parse_args_defaults_to_clean_collection_mode():
    args = full_write.parse_args([])

    assert args.mode == "clean-collection"
    assert args.reuse_existing_collection is False
    assert args.prepare_collection_only is False
    assert args.sequence_offset == 0


def test_full_write_parse_args_supports_prepare_collection_only():
    args = full_write.parse_args(["--prepare-collection-only", "--reuse-existing-collection"])

    assert args.prepare_collection_only is True
    assert args.reuse_existing_collection is True


def test_full_write_report_exposes_feature_stage_latency_breakdown():
    case = shared.BenchmarkCaseResult(
        case_name="full_c16",
        stage="image_full_write",
        concurrency=16,
        extras={
            "stage_latency_ms": {
                "feature_pipeline_total": {"avg": 5600.0},
                "milvus_write": {"avg": 400.0},
                "p_hash": {"avg": 1.2},
                "data_url": {"avg": 48.5},
                "visual_embed": {"avg": 650.0},
                "caption_embed": {"avg": 2100.0},
                "context_embed": {"avg": 2300.0},
            }
        },
    )

    summary = case.to_summary()

    assert summary["extras"]["stage_latency_ms"]["feature_pipeline_total"]["avg"] == 5600.0
    assert summary["extras"]["stage_latency_ms"]["data_url"]["avg"] == 48.5
    assert summary["extras"]["stage_latency_ms"]["visual_embed"]["avg"] == 650.0
    assert summary["extras"]["stage_latency_ms"]["caption_embed"]["avg"] == 2100.0
    assert summary["extras"]["stage_latency_ms"]["context_embed"]["avg"] == 2300.0


def test_mutate_payload_rewrites_only_test_image_primary_keys():
    original_url = "/fulltext/original/figures/figure-1.jpg"
    original_mmd_url = "/fulltext/original/original.mmd"
    payload = {
        "id": "source-doc",
        "data": [
            {
                "id": "source-doc",
                "op": {
                    "hasPictureFile": {
                        "value": [
                            {"id": "source-image", "dp": {"url": {"value": [original_url]}}}
                        ]
                    },
                    "hasFullTextFile": {
                        "value": [{"id": "source-mmd", "dp": {"url": {"value": [original_mmd_url]}}}]
                    },
                },
            }
        ],
    }

    result = isolated_load.mutate_payload(
        base_payload=payload,
        doc_id="benchmark-doc",
        update_ms=123,
        publish_ms=456,
        rewrite_image_ids=True,
    )

    item = result["data"][0]
    assert result["id"] == "benchmark-doc"
    assert item["id"] == "benchmark-doc"
    assert item["op"]["hasPictureFile"]["value"][0]["id"] == "benchmark-doc__image_0000"
    assert item["op"]["hasPictureFile"]["value"][0]["dp"]["url"]["value"] == [original_url]
    assert item["op"]["hasFullTextFile"]["value"][0]["id"] == "source-mmd"
    assert item["op"]["hasFullTextFile"]["value"][0]["dp"]["url"]["value"] == [original_mmd_url]


def _write_manifest(
    upload_dir: Path,
    *,
    state: str,
    doc_id: str,
    task_id: str,
    image_name: str,
    image_bytes: bytes,
    mmd_text: str | None = None,
) -> Path:
    sample_dir = upload_dir / state / doc_id / task_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = sample_dir / image_name
    image_path.write_bytes(image_bytes)
    mmd_path = sample_dir / f"{doc_id}.mmd"
    mmd_path.write_text(
        mmd_text or "\n\n".join(["Before", f"![Figure](figures/{image_name})", "After"]),
        encoding="utf-8",
    )
    manifest = {
        "doc_id": doc_id,
        "task_id": task_id,
        "image_id": f"source-{task_id}",
        "raw_img_meta": {"id": f"source-{task_id}"},
        "image_relative_url": f"/fulltext/{doc_id}/figures/{image_name}",
        "mmd_relative_url": f"/fulltext/{doc_id}/{doc_id}.mmd",
        "downloads": {
            "image": {
                "relative_url": f"/fulltext/{doc_id}/figures/{image_name}",
                "local_path": str(image_path),
                "content_type": "image/jpeg",
                "size_bytes": len(image_bytes),
            },
            "mmd": {
                "relative_url": f"/fulltext/{doc_id}/{doc_id}.mmd",
                "local_path": str(mmd_path),
            },
        },
        "context": {},
    }
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _tiny_jpeg_bytes() -> bytes:
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000"
        "ffc00011080001000103012200021101031101"
        "ffda000c03010002110311003f00"
        "00"
        "ffd9"
    )
