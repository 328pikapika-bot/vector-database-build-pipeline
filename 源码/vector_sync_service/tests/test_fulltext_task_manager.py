from __future__ import annotations

from app.services.fulltext_task_manager import (
    FullTextTaskStatus,
    build_fulltext_task_id,
    build_skipped_fulltext_discover_document,
    extract_fulltext_tasks,
)


def test_no_has_fulltext_file_returns_no_tasks_and_skipped_draft_is_available():
    payload = {
        "data": [
            {
                "sysId": "doc-1",
                "sysDataUpdateTime": "2026-06-07T00:00:00+00:00",
                "op": {},
            }
        ]
    }

    assert extract_fulltext_tasks(payload) == []

    skipped = build_skipped_fulltext_discover_document(
        doc_id="doc-1",
        record_update_time="2026-06-07T00:00:00+00:00",
    )
    assert skipped["status"] == FullTextTaskStatus.SKIPPED.value
    assert skipped["skip_reason"] == "SKIPPED_NOT_PRESENT"


def test_extracts_nested_dp_url_mmd_task():
    payload = _payload_with_relation(
        {
            "dp": {
                "url": {
                    "field": "url",
                    "value": ["/fulltext/paper.mmd"],
                }
            }
        }
    )

    tasks = extract_fulltext_tasks(payload)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.doc_id == "doc-1"
    assert task.record_update_time == "2026-06-07T00:00:00+00:00"
    assert task.relative_url == "/fulltext/paper.mmd"
    assert task.status == FullTextTaskStatus.PENDING
    assert task.task_id == build_fulltext_task_id(
        doc_id="doc-1",
        record_update_time="2026-06-07T00:00:00+00:00",
        relative_url="/fulltext/paper.mmd",
    )

    document = task.to_mongo_document()
    assert document["_id"] == task.task_id
    assert document["stage_group"] == "fulltext"
    assert document["status"] == FullTextTaskStatus.PENDING.value


def test_extracts_existing_file_url_shape_and_task_id_is_stable():
    payload = _payload_with_relation(
        {
            "dp": {
                "fileName": {
                    "field": "fileName",
                    "value": ["paper.mmd"],
                },
                "fileUrl": {
                    "field": "fileUrl",
                    "value": ["/fulltext/paper.mmd"],
                },
            }
        }
    )

    first = extract_fulltext_tasks(payload)
    second = extract_fulltext_tasks(payload)

    assert len(first) == 1
    assert first[0].relative_url == "/fulltext/paper.mmd"
    assert first[0].task_id == second[0].task_id


def test_extracts_doc_metadata_from_raw_nested_dp_payload():
    payload = {
        "data": [
            {
                "cls": "Article",
                "dp": {
                    "sysId": _field("doc-raw"),
                    "sysDataUpdateTime": _field("2026-06-07T00:00:00+00:00"),
                    "title": _field(
                        "Degradation of liquid crystal devices",
                        multi_value=True,
                    ),
                    "keyword": _field(["liquid crystal", "electrochemistry"], multi_value=True),
                    "accessType": _field(["archives"], multi_value=True),
                    "classificationCode": _field("O441"),
                    "year": _field(1976),
                    "hasAuthorname": _field(["B.Gosse"], multi_value=True),
                    "hasCreateOrganizationname": _field(
                        ["Centre National de la Recherche Scientifique"],
                        multi_value=True,
                    ),
                    "language": _field(["en"], multi_value=True),
                },
                "op": {
                    "hasFullTextFile": {
                        "field": "hasFullTextFile",
                        "value": [
                            {
                                "dp": {
                                    "fileUrl": {
                                        "field": "fileUrl",
                                        "value": ["/fulltext/raw-paper.mmd"],
                                    }
                                }
                            }
                        ],
                    }
                },
            }
        ]
    }

    tasks = extract_fulltext_tasks(payload)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.doc_id == "doc-raw"
    assert task.relative_url == "/fulltext/raw-paper.mmd"
    assert task.title == ["Degradation of liquid crystal devices"]
    assert task.keyword == ["liquid crystal", "electrochemistry"]
    assert task.accessType == "archives"
    assert task.classificationCode == "O441"
    assert task.year == "1976"
    assert task.hasAuthorname == ["B.Gosse"]
    assert task.hasCreateOrganizationname == ["Centre National de la Recherche Scientifique"]
    assert task.language == ["en"]


def test_ignores_non_mmd_urls_and_accepts_case_insensitive_mmd():
    payload = _payload_with_relation(
        {"dp": {"fileUrl": {"value": ["/fulltext/paper.pdf"]}}},
        {"fileUrl": "/fulltext/PAPER.MMD?download=1"},
    )

    tasks = extract_fulltext_tasks(payload)

    assert len(tasks) == 1
    assert tasks[0].relative_url == "/fulltext/PAPER.MMD?download=1"


def _payload_with_relation(*relations: dict) -> dict:
    return {
        "data": [
            {
                "sysId": "doc-1",
                "sysDataUpdateTime": "2026-06-07T00:00:00+00:00",
                "op": {
                    "hasFullTextFile": {
                        "field": "hasFullTextFile",
                        "value": list(relations),
                    }
                },
            }
        ]
    }


def _field(value, *, multi_value: bool = False, type_: str = "string") -> dict:
    values = value if isinstance(value, list) else [value]
    return {
        "field": "field",
        "multiValue": multi_value,
        "type": type_,
        "value": values,
    }
