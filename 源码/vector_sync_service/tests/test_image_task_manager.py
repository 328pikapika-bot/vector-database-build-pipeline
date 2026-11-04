from __future__ import annotations

import pytest

from app.services.image_task_manager import (
    ImageTaskStatus,
    build_image_task_id,
    build_skipped_image_discover_document,
    extract_image_tasks,
)


def test_no_has_picture_file_returns_no_tasks_and_skipped_draft_is_available():
    payload = {
        "data": [
            {
                "sysId": "doc-1",
                "sysDataUpdateTime": "2026-06-07T00:00:00+00:00",
                "op": {},
            }
        ]
    }

    assert extract_image_tasks(payload) == []

    skipped = build_skipped_image_discover_document(
        doc_id="doc-1",
        record_update_time="2026-06-07T00:00:00+00:00",
    )
    assert skipped["status"] == ImageTaskStatus.SKIPPED.value
    assert skipped["skip_reason"] == "SKIPPED_NOT_PRESENT"
    assert skipped["stage_group"] == "image"


def test_extracts_nested_dp_image_task_with_metadata():
    payload = _payload_with_relation(
        {
            "id": "image-1",
            "cls": "PictureFile",
            "dp": {
                "url": {"value": ["/images/fig1.png"]},
                "sequence": {"value": ["2"]},
                "title": {"value": ["Figure 1"]},
                "fileName": {"value": ["fig1.png"]},
                "width": {"value": ["800"]},
                "height": {"value": ["600"]},
                "fileType": {"value": ["png"]},
                "size": {"value": ["12345"]},
                "sysDataType": {"value": ["file"]},
            }
        }
    )

    tasks = extract_image_tasks(payload)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.doc_id == "doc-1"
    assert task.record_update_time == "2026-06-07T00:00:00+00:00"
    assert task.relative_url == "/images/fig1.png"
    assert task.mmd_relative_url is None
    assert task.sequence == "2"
    assert task.image_id == "image-1"
    assert task.title == "Figure 1"
    assert task.name == "fig1.png"
    assert task.width == 800
    assert task.height == 600
    assert task.file_type == "png"
    assert task.size == 12345
    assert task.raw_img_meta == {
        "id": "image-1",
        "cls": "PictureFile",
        "sequence": "2",
        "name": "fig1.png",
        "fileType": "png",
        "sysDataType": "file",
        "size": 12345,
        "width": 800,
        "height": 600,
        "title": ["Figure 1"],
        "url": ["/images/fig1.png"],
    }
    assert task.status == ImageTaskStatus.PENDING
    assert task.task_id == build_image_task_id(
        doc_id="doc-1",
        record_update_time="2026-06-07T00:00:00+00:00",
        relative_url="/images/fig1.png",
        source_item_index=0,
        sequence="2",
    )

    document = task.to_mongo_document()
    assert document["_id"] == task.task_id
    assert document["stage_group"] == "image"
    assert document["status"] == ImageTaskStatus.PENDING.value
    assert document["mmd_relative_url"] is None


def test_extracts_flat_file_url_and_task_id_is_stable():
    payload = _payload_with_relation(
        {
            "fileUrl": "/images/fig2.jpg",
            "sequence": 3,
            "title": "Figure 2",
        }
    )

    first = extract_image_tasks(payload)
    second = extract_image_tasks(payload)

    assert len(first) == 1
    assert first[0].relative_url == "/images/fig2.jpg"
    assert first[0].sequence == "3"
    assert first[0].title == "Figure 2"
    assert first[0].name == "fig2.jpg"
    assert first[0].task_id == second[0].task_id


def test_image_relation_without_original_id_is_rejected():
    payload = _payload_with_relation({"fileUrl": "/images/no-id.jpg", "sequence": 1})
    del payload["data"][0]["op"]["hasPictureFile"]["value"][0]["id"]

    from app.services.image_task_manager import ImageSourceIdRequiredError

    with pytest.raises(ImageSourceIdRequiredError, match="IMAGE_SOURCE_ID_REQUIRED"):
        extract_image_tasks(payload)


def test_multiple_data_items_include_source_item_index_in_task_id():
    payload = {
        "data": [
            _data_item("doc-1", "/images/same.png"),
            _data_item("doc-1", "/images/same.png"),
        ]
    }

    tasks = extract_image_tasks(payload)

    assert len(tasks) == 2
    assert tasks[0].source_item_index == 0
    assert tasks[1].source_item_index == 1
    assert tasks[0].task_id != tasks[1].task_id


def test_multiple_images_use_each_original_image_id_as_collection_primary_key():
    payload = _payload_with_relation(
        {"id": "raw-image-a", "fileUrl": "/images/a.png", "sequence": 1},
        {"id": "raw-image-b", "fileUrl": "/images/b.png", "sequence": 2},
    )

    tasks = extract_image_tasks(payload)

    assert [task.image_id for task in tasks] == ["raw-image-a", "raw-image-b"]
    assert [task.raw_img_meta["id"] for task in tasks] == ["raw-image-a", "raw-image-b"]


def test_extract_image_tasks_prefers_mmd_context_and_filename_metadata(monkeypatch):
    filename = "123-456-whxyp_0.1000_0.2000_0.3000_0.4000_5.jpg"
    payload = _payload_with_relation(
        {
            "fileUrl": f"/download/figures/{filename}",
            "sequence": 4,
            "title": "Relation caption",
            "context_before": "relation before",
            "context_after": "relation after",
        }
    )
    payload["data"][0]["mmd_text"] = "\n\n".join(
        [
            "MMD before one.",
            "MMD before two.",
            f"![Figure](figures/{filename})",
            "MMD after one.",
            "MMD after two.",
        ]
    )
    monkeypatch.setattr("app.services.image_task_manager.settings.IMAGE_CONTEXT_BEFORE_PARAGRAPHS", 1)
    monkeypatch.setattr("app.services.image_task_manager.settings.IMAGE_CONTEXT_AFTER_PARAGRAPHS", 2)

    tasks = extract_image_tasks(payload)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.context_before == "MMD before two."
    assert task.context_after == "Relation caption\n\nMMD after one.\n\nMMD after two."
    assert task.img_meta["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert task.img_meta["page"] == 5
    assert task.img_meta["caption"] == "Relation caption"
    assert task.img_meta["source_filename"] == filename
    assert task.raw_img_meta["sequence"] == "4"
    assert task.raw_img_meta["name"] == filename
    assert task.raw_img_meta["title"] == ["Relation caption"]
    assert task.raw_img_meta["url"] == [f"/download/figures/{filename}"]
    assert "hasPictureFile" not in task.raw_img_meta
    assert "mmd_image_context" not in task.raw_img_meta


def test_extract_image_tasks_keeps_relation_before_when_mmd_before_is_empty():
    payload = _payload_with_relation(
        {
            "fileUrl": "/download/figures/fig-a.jpg",
            "sequence": 1,
            "title": "Relation caption",
            "context_before": "relation before",
            "context_after": "relation after",
        }
    )
    payload["data"][0]["mmd_text"] = "\n\n".join(
        [
            "![Figure](figures/fig-a.jpg)",
            "MMD after one.",
            "MMD after two.",
        ]
    )

    tasks = extract_image_tasks(payload)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.context_before == "relation before"
    assert task.context_after == "Relation caption\n\nMMD after one.\n\nMMD after two."


def test_extract_image_tasks_canonicalizes_escaped_html_in_caption_and_context():
    payload = _payload_with_relation(
        {
            "fileUrl": "/download/figures/fig-a.jpg",
            "sequence": 1,
            "title": "Taxonomic Studies on Some Members of the Genus &amp;lt;i&amp;gt;Abutilon&amp;lt;/i&amp;gt; Mill. (Malvaceae)",
            "context_before": "Taxonomic Studies on Some Members of the Genus &amp;lt;i&amp;gt;Abutilon&amp;lt;/i&amp;gt; Mill. (Malvaceae)",
        }
    )
    payload["data"][0]["mmd_text"] = "\n\n".join(
        [
            "Previous &amp;lt;i&amp;gt;text&amp;lt;/i&amp;gt;.",
            "![Figure](figures/fig-a.jpg)",
            "After &amp;lt;i&amp;gt;text&amp;lt;/i&amp;gt; one.",
            "After two.",
        ]
    )

    tasks = extract_image_tasks(payload)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.context_before == "Previous text."
    assert task.context_after == (
        "Taxonomic Studies on Some Members of the Genus Abutilon Mill. (Malvaceae)\n\n"
        "After text one.\n\nAfter two."
    )
    assert task.img_meta["caption"] == "Taxonomic Studies on Some Members of the Genus Abutilon Mill. (Malvaceae)"


def test_extract_image_tasks_carries_mmd_relative_url_from_has_fulltext_file():
    payload = _payload_with_relation(
        {
            "fileUrl": "/fulltext/doc-1/figures/fig1.jpg",
            "sequence": 1,
            "title": "Figure caption",
        }
    )
    payload["data"][0]["op"]["hasFullTextFile"] = {
        "field": "hasFullTextFile",
        "value": [
            {
                "dp": {
                    "url": {
                        "field": "url",
                        "value": ["/fulltext/doc-1/doc-1.mmd"],
                    }
                }
            }
        ],
    }

    tasks = extract_image_tasks(payload)

    assert len(tasks) == 1
    assert tasks[0].mmd_relative_url == "/fulltext/doc-1/doc-1.mmd"
    assert tasks[0].img_meta["caption"] == "Figure caption"


def _payload_with_relation(*relations: dict) -> dict:
    normalized_relations = []
    for index, relation in enumerate(relations, start=1):
        normalized = dict(relation)
        normalized.setdefault("id", f"source-image-{index}")
        normalized_relations.append(normalized)
    return {
        "data": [
            {
                "sysId": "doc-1",
                "sysDataUpdateTime": "2026-06-07T00:00:00+00:00",
                "op": {
                    "hasPictureFile": {
                        "field": "hasPictureFile",
                        "value": normalized_relations,
                    }
                },
            }
        ]
    }


def _data_item(doc_id: str, url: str) -> dict:
    return {
        "sysId": doc_id,
        "sysDataUpdateTime": "2026-06-07T00:00:00+00:00",
        "op": {
            "hasPictureFile": {
                "value": [
                    {
                        "id": f"source-{doc_id}-{url}",
                        "fileUrl": url,
                    }
                ]
            }
        },
    }
