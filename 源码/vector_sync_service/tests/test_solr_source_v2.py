from __future__ import annotations

import httpx
import pytest

from app.models.sync_runtime import ConsumedRecord
from app.services.fulltext_task_manager import build_fulltext_task_documents
from app.services.image_task_manager import build_image_task_documents
from app.services.independent_kafka_worker import IndependentKafkaWorker
from app.services.kafka_consumer import parse_message
from app.services.solr_source import SolrResolvedDocument, SolrSourceResolver
from app.services.static_resource_url import build_static_resource_url


DOC_ID = "a" * 32


class _KafkaMessage:
    def error(self):
        return None

    def value(self):
        return DOC_ID.encode("ascii")

    def topic(self):
        return "VECTOR_JOURNAL_ARTICLE_FULLTEXT"

    def partition(self):
        return 3

    def offset(self):
        return 7


def test_bare_kafka_doc_id_is_accepted_as_a_solr_source_event(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.kafka_consumer.settings.KAFKA_DUMP_FILE", str(tmp_path / "events.jsonl"))

    record = parse_message(_KafkaMessage())

    assert record is not None
    assert record.is_valid is True
    assert record.parsed_payload == {
        "data": [{"id": DOC_ID, "sysId": DOC_ID}],
        "operation": "UPDATE",
        "source_mode": "solr_doc_id",
    }


@pytest.mark.asyncio
async def test_solr_resolver_builds_fulltext_and_image_tasks_with_relative_paths():
    paper = {
        "id": DOC_ID,
        "type": "article",
        "auto_stringITMS_title": ["en,zh|A title"],
        "auto_stringITMS_abstract": ["en|Abstract"],
        "auto_stringITMS_keyword": ["en|keyword"],
        "stringIS_year": "2026",
        "stringIMS_language": ["en"],
        "stringIMS_accessType": ["open"],
        "stringIS_sysDataStatus": "enable",
        "dateIS_sysDataUpdateTime": "2026-08-01T00:00:00Z",
        "stringOM_hasFullTextFile": [
            "sysDataType≡file∷url≡/fulltext/doc/paper.mmd∷name≡paper.mmd"
        ],
    }
    image = {
        "id": "b" * 32,
        "stringIS_sourceId": DOC_ID,
        "stringIMS_url": "/fulltext/doc/figures/figure-1.jpg",
        "auto_stringITS_name": "figure-1.jpg",
        "stringIS_sequence": "1",
        "intIS_width": 320,
        "intIS_height": 240,
        "stringIS_fileType": "jpg",
        "intIS_size": 1024,
        "auto_stringITMS_title": ["en|Figure caption"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/select")
        assert request.url.params["fl"]
        if "paper6.6" in request.url.path:
            return httpx.Response(200, json={"response": {"docs": [paper]}})
        return httpx.Response(200, json={"response": {"docs": [image]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = SolrSourceResolver(
        paper_url="http://solr/paper6.6",
        image_url="http://solr/standard6.6",
        client=client,
    )
    try:
        resolved = await resolver.resolve_document(DOC_ID, include_images=True)
    finally:
        await client.aclose()

    assert resolved.found is True
    item = resolved.payload["data"][0]
    assert item["dp"]["title"]["value"] == ["A title"]
    assert item["dp"]["hasFullTextFilename"]["value"] == ["paper.mmd"]
    # The paper exists even though source rows did not carry every optional
    # metadata field, so this remains a source-data gap rather than a failure.
    assert "classificationCode" not in item["dp"]

    fulltext_tasks = build_fulltext_task_documents(resolved.payload)
    image_tasks = build_image_task_documents(resolved.payload)
    assert fulltext_tasks[0]["doc_id"] == DOC_ID
    assert fulltext_tasks[0]["relative_url"] == "/fulltext/doc/paper.mmd"
    assert image_tasks[0]["image_id"] == "b" * 32
    assert image_tasks[0]["doc_id"] == DOC_ID
    assert image_tasks[0]["relative_url"] == "/fulltext/doc/figures/figure-1.jpg"
    assert build_static_resource_url(
        image_tasks[0]["relative_url"],
        base_url="http://static.internal:12001/",
    ) == "http://static.internal:12001/fulltext/doc/figures/figure-1.jpg"


@pytest.mark.asyncio
async def test_solr_resolver_keeps_document_found_when_optional_fields_are_empty():
    paper = {"id": DOC_ID, "stringIMS_language": ["en"], "stringIS_sysDataStatus": "ACTIVE"}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": {"docs": [paper]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = SolrSourceResolver(
        paper_url="http://solr/paper6.6",
        image_url="http://solr/standard6.6",
        client=client,
    )
    try:
        resolved = await resolver.resolve_document(DOC_ID, include_images=False)
    finally:
        await client.aclose()

    assert resolved.found is True
    assert resolved.payload["data"][0]["id"] == DOC_ID
    assert "title" not in resolved.payload["data"][0]["dp"]


@pytest.mark.asyncio
async def test_solr_resolver_retries_a_transient_source_connection_error():
    calls = 0
    paper = {"id": DOC_ID}

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary source connection reset")
        return httpx.Response(200, json={"response": {"docs": [paper]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = SolrSourceResolver(
        paper_url="http://solr/paper6.6",
        image_url="http://solr/standard6.6",
        request_max_retries=2,
        request_retry_backoff_ms=0,
        client=client,
    )
    try:
        resolved = await resolver.resolve_document(DOC_ID, include_images=False)
    finally:
        await client.aclose()

    assert resolved.found is True
    assert calls == 2


@pytest.mark.asyncio
async def test_solr_resolver_does_not_retry_non_transient_client_errors():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "invalid query"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = SolrSourceResolver(
        paper_url="http://solr/paper6.6",
        image_url="http://solr/standard6.6",
        request_max_retries=2,
        request_retry_backoff_ms=0,
        client=client,
    )
    try:
        with pytest.raises(Exception, match="Solr query failed"):
            await resolver.resolve_document(DOC_ID, include_images=False)
    finally:
        await client.aclose()

    assert calls == 1


class _Resolver:
    async def resolve_document(self, doc_id: str, *, include_images: bool):
        return SolrResolvedDocument(doc_id=doc_id, payload={"data": [{"id": doc_id}]}, found=False)

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_missing_solr_document_becomes_a_lane_delete():
    record = ConsumedRecord(
        topic="topic",
        partition=0,
        offset=1,
        raw_bytes=DOC_ID.encode("ascii"),
        raw_text=DOC_ID,
        event_fingerprint="topic|0|1|hash",
        parsed_payload={
            "data": [{"id": DOC_ID, "sysId": DOC_ID}],
            "operation": "UPDATE",
            "source_mode": "solr_doc_id",
        },
    )
    worker = IndependentKafkaWorker("fulltext", source_resolver=_Resolver())

    hydrated = await worker._hydrate_record_from_solr(record, include_images=False)

    assert hydrated.parsed_payload is not None
    assert hydrated.parsed_payload["operation"] == "DELETE"
    assert hydrated.parsed_payload["data"][0]["id"] == DOC_ID
