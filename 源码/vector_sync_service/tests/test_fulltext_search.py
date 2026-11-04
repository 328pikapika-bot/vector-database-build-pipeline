import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.config import settings
from app.services.fulltext_search import FulltextSearchRequest, FulltextSearchService, non_body_text, search_fulltext


class Reader:
    def __init__(self): self.search_kwargs = {}
    def describe_collection(self, name, **kwargs):
        if name == settings.resolved_document_collection_name:
            return {"fields": [{"name": "id", "is_primary": True}, {"name": "doi"}]}
        return {"description": f"FULLTEXT|MODEL={settings.EMBEDDING_MODEL}|DIM={settings.VECTOR_DIM}",
            "fields": [{"name": key, "params": {"dim": settings.VECTOR_DIM}} for key in
                ["text_embed", "id", "doc_id", "text", "title", "hasAuthorname", "year", "accessType", "language", "charStart", "charEnd", "sectionPath"]]}
    def list_indexes(self, *args, **kwargs): return ["text_embed"]
    def describe_index(self, *args, **kwargs): return {"metric_type": "COSINE"}
    def query(self, **kwargs): return [{"id": "doc-1", "doi": "10.1000/a"}]
    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [[{"id": "chunk-1", "distance": .8, "entity": {"id": "chunk-1", "doc_id": "doc-1",
            "text": "Real paragraph", "title": ["Paper"], "hasAuthorname": ["Li"], "year": "2025",
            "sectionPath": "Results", "charStart": 0, "charEnd": 100}}]]


def test_reader_is_bounded_filters_authorization_and_invalid_offsets(monkeypatch):
    monkeypatch.setattr(settings, "FULLTEXT_SEARCH_ACCESS_TYPES", "archives")
    reader = Reader()
    service = FulltextSearchService(client=reader)
    result = service._search(FulltextSearchRequest(query="battery interfaces", doc_ids=['x" or true'], year_range=[2020, 2026], languages=["en"]), [0.] * settings.VECTOR_DIM)
    assert reader.search_kwargs["limit"] <= 90
    assert 'accessType in ["archives"]' in reader.search_kwargs["filter"]
    assert 'x\\" or true' in reader.search_kwargs["filter"]
    chunk = result["data"]["chunks"][0]
    assert chunk["charStart"] is None and not chunk["offsets_consistent"]
    assert chunk["doi"] == "10.1000/a"
    assert chunk["source_hash"] == hashlib.sha256(chunk["text"].encode()).hexdigest()


def test_missing_scope_and_model_mismatch_fail_closed(monkeypatch):
    monkeypatch.setattr(settings, "FULLTEXT_SEARCH_ACCESS_TYPES", "")
    with pytest.raises(ValueError, match="授权来源"):
        FulltextSearchService(client=Reader())._search(FulltextSearchRequest(query="battery"), [0.] * settings.VECTOR_DIM)
    class Incompatible(Reader):
        def describe_collection(self, *args, **kwargs):
            schema = super().describe_collection(*args, **kwargs)
            schema["description"] = "MODEL=other|"
            return schema
    with pytest.raises(ValueError, match="模型"):
        FulltextSearchService(client=Incompatible())._search(FulltextSearchRequest(query="battery"), [0.] * settings.VECTOR_DIM)


def test_query_vector_dimension_checked_before_database_use():
    class Embedder:
        async def embed_single_detailed(self, query): return SimpleNamespace(success=True, vector=[1.])
    with pytest.raises(ValueError, match="维度"):
        asyncio.run(FulltextSearchService(client=Reader(), embedder=Embedder()).search(FulltextSearchRequest(query="battery")))


def test_authentication_cannot_be_disabled_by_request(monkeypatch):
    monkeypatch.setattr(settings, "FULLTEXT_SEARCH_TOKEN", "secret")
    with pytest.raises(HTTPException) as error:
        asyncio.run(search_fulltext(FulltextSearchRequest(query="battery"), authorization="Bearer other"))
    assert error.value.status_code == 401


def test_reference_lists_and_frontmatter_are_not_body_text():
    assert non_body_text("Citation:", "Citation: Paper title", 3)
    assert non_body_text("3. Author A. Paper. 2022;1:2.", "3. Author A. Paper. 2022;1:2.\n4. Author B. Other. 2021;2:3.", 35)
    assert non_body_text("Article Paper title", "Article Paper title\nA. Author, B. Author", 2)
    assert not non_body_text("Results", "Pressure improved cycling stability in the reported cells.", 8)
