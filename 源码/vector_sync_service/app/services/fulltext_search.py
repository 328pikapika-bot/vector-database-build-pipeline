"""Bounded read-only semantic search; never initialize or mutate collections."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import logging
import threading
from collections import Counter
from secrets import compare_digest
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from app.config import settings
from app.services.embedder import get_embedding_client


logger = logging.getLogger(__name__)
_query_slots = threading.BoundedSemaphore(6)


class FulltextSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    doc_ids: list[str] = Field(default_factory=list, max_length=30)
    top_k: int = Field(default=15, ge=1, le=30)
    top_k_per_doc: int = Field(default=3, ge=1, le=3)
    year_range: list[int] = Field(default_factory=list, max_length=2)
    languages: list[str] = Field(default_factory=list, max_length=5)
    intent_id: str = Field(default="", max_length=128)


def strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def non_body_text(section: str, text: str, chunk_index: Any) -> bool:
    if re.search(r"\b(?:references?|bibliography|contents|abstract|citation)\b|参考文献|目录|摘要", section, re.I):
        return True
    if re.match(r"^\s*\d{1,3}\.\s+\S.+\b(?:19|20)\d{2}\b", section):
        return True
    if len(re.findall(r"(?:^|\n)\s*\d{1,3}\.\s+\S", text)) >= 2:
        return True
    return isinstance(chunk_index, int) and chunk_index <= 2 and re.match(r"^\s*(?:article\s+)?" + re.escape(section.strip()), text, re.I) is not None


class FulltextSearchService:
    def __init__(self, client: Any = None, embedder: Any = None) -> None:
        self.client = client
        self.embedder = embedder

    async def search(self, request: FulltextSearchRequest) -> dict[str, Any]:
        # This is the same embedding client/model used by chunk ingestion.
        embedder = self.embedder or get_embedding_client()
        result = await asyncio.wait_for(embedder.embed_single_detailed(request.query), timeout=12)
        if not result.success or not result.vector:
            raise ValueError("全文查询向量生成失败")
        if len(result.vector) != settings.VECTOR_DIM or not all(math.isfinite(v) for v in result.vector):
            raise ValueError("全文查询向量维度或数值不兼容")
        return await asyncio.to_thread(self._search, request, result.vector)

    def _search(self, request: FulltextSearchRequest, vector: list[float]) -> dict[str, Any]:
        # Hold the slot until the blocking database operation actually exits,
        # including when its awaiting HTTP request has timed out.
        if not _query_slots.acquire(blocking=False):
            raise ValueError("全文查询繁忙")
        try:
            return self._search_with_slot(request, vector)
        finally:
            _query_slots.release()

    def _search_with_slot(self, request: FulltextSearchRequest, vector: list[float]) -> dict[str, Any]:
        client = self.client or MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            user=settings.MILVUS_USER or "", password=settings.MILVUS_PASSWORD or "", timeout=10,
        )
        try:
            collection = settings.resolved_fulltext_chunk_collection_name
            schema = client.describe_collection(collection, timeout=10)
            fields = {f["name"]: f for f in schema.get("fields", [])}
            if int(fields.get("text_embed", {}).get("params", {}).get("dim", 0)) != len(vector):
                raise ValueError("全文集合向量维度不兼容")
            signature = str(schema.get("description") or "")
            model = re.search(r"(?:^|\|)MODEL=([^|]+)", signature)
            if not model or model.group(1) != settings.EMBEDDING_MODEL:
                raise ValueError("全文集合缺少匹配的向量模型签名")
            version = re.search(r"(?:^|\|)MODEL_VERSION=([^|]*)", signature)
            if settings.EMBEDDING_MODEL_VERSION and (not version or version.group(1) != settings.EMBEDDING_MODEL_VERSION):
                raise ValueError("全文集合向量模型版本不兼容")
            # accessType is a source category, NOT proof that a caller may read
            # it. The trusted service configuration defines the allowed set.
            allowed = [v.strip() for v in settings.FULLTEXT_SEARCH_ACCESS_TYPES.split(",") if v.strip()]
            if not allowed:
                raise ValueError("尚未配置全文查询授权来源范围")
            clauses = ["accessType in " + json.dumps(allowed)]
            if "sectionPath" in fields:
                clauses.append('sectionPath not in ["REFERENCES", "References", "参考文献", "Bibliography", "BIBLIOGRAPHY", "Contents", "CONTENTS", "Abstract", "ABSTRACT", "摘要"]')
            if "chunkIndex" in fields:
                clauses.append("chunkIndex > 1")
            if request.doc_ids:
                clauses.append("doc_id in " + json.dumps(request.doc_ids))
            if request.year_range:
                if len(request.year_range) != 2 or request.year_range[0] > request.year_range[1]:
                    raise ValueError("年份范围无效")
                clauses.extend([f"year >= {json.dumps(str(request.year_range[0]))}", f"year <= {json.dumps(str(request.year_range[1]))}"])
            if request.languages:
                clauses.append("ARRAY_CONTAINS_ANY(language, " + json.dumps(request.languages) + ")")
            index = next((client.describe_index(collection, name, timeout=10) for name in client.list_indexes(collection, timeout=10) if name == "text_embed"), {})
            metric = index.get("metric_type", "COSINE")
            if metric != "COSINE":
                raise ValueError("全文查询当前仅支持余弦相似度索引")
            output = ["id", "doc_id", "title", "hasAuthorname", "year", "language", "accessType", "text", "chunkIndex", "chunkType", "sectionPath", "charStart", "charEnd"]
            hits = client.search(
                collection_name=collection, data=[vector], anns_field="text_embed",
                filter=" and ".join(clauses), limit=min(90, request.top_k * 6),
                search_params={"metric_type": metric, "params": {"nprobe": 64}},
                output_fields=[name for name in output if name in fields], timeout=12,
                consistency_level="Eventually",
            )
            chunks: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            seen: set[str] = set()
            for hit in hits[0] if hits else []:
                row = hit.get("entity", hit)
                doc_id = str(row.get("doc_id") or "")
                text = str(row.get("text") or "")
                section = str(row.get("sectionPath") or "")
                if non_body_text(section, text, row.get("chunkIndex")):
                    continue
                key = doc_id + ":" + hashlib.sha256(text.encode()).hexdigest()
                if not doc_id or not text.strip() or key in seen or counts[doc_id] >= request.top_k_per_doc:
                    continue
                start, end = row.get("charStart"), row.get("charEnd")
                offsets_valid = isinstance(start, int) and isinstance(end, int) and start >= 0 and end - start == len(text)
                chunks.append({
                    "doc_id": doc_id, "chunk_id": str(row.get("id") or hit.get("id") or ""),
                    "text": text, "title": "；".join(strings(row.get("title"))),
                    "authors": strings(row.get("hasAuthorname")), "year": str(row.get("year") or ""),
                    "languages": strings(row.get("language")), "source": "NSTL 全文库",
                    "access_status": "authorized", "access_type": row.get("accessType"),
                    "chunkIndex": row.get("chunkIndex"), "chunkType": row.get("chunkType"),
                    "sectionPath": section, "charStart": start if offsets_valid else None,
                    "charEnd": end if offsets_valid else None, "offsets_consistent": offsets_valid,
                    "locator_verified": True, "score": float(hit.get("distance", 0)),
                    "source_hash": hashlib.sha256(text.encode()).hexdigest(), "intent_id": request.intent_id,
                })
                counts[doc_id] += 1
                seen.add(key)
                if len(chunks) >= request.top_k:
                    break
            self._enrich_metadata(client, chunks)
            return {"success": True, "data": {"chunks": chunks}, "intent_id": request.intent_id}
        finally:
            if self.client is None:
                client.close()

    @staticmethod
    def _enrich_metadata(client: Any, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        try:
            collection = settings.resolved_document_collection_name
            schema = client.describe_collection(collection, timeout=5)
            fields = {f["name"] for f in schema.get("fields", [])}
            pk = next(f["name"] for f in schema["fields"] if f.get("is_primary"))
            rows = client.query(collection_name=collection,
                filter=pk + " in " + json.dumps(list({c["doc_id"] for c in chunks})),
                output_fields=list({pk, *({"doi", "url", "title", "year", "hasAuthorname"} & fields)}),
                limit=30, timeout=5)
            metadata = {str(row[pk]): row for row in rows}
            for chunk in chunks:
                row = metadata.get(chunk["doc_id"], {})
                chunk["doi"] = next(iter(strings(row.get("doi"))), "")
                chunk["source_url"] = next(iter(strings(row.get("url"))), "")
                if not chunk["title"]:
                    chunk["title"] = "；".join(strings(row.get("title")))
        except Exception:
            # Chunks retain their own title/authors/year; a metadata outage
            # never invents citations or turns an ID into a publication title.
            return


router = APIRouter()


@router.post("/v1/fulltext/search")
async def search_fulltext(request: FulltextSearchRequest, authorization: str = Header(default="")):
    token = settings.FULLTEXT_SEARCH_TOKEN
    if not token or not compare_digest(authorization, "Bearer " + token):
        raise HTTPException(status_code=401, detail="全文只读查询需要服务凭证")
    try:
        return await asyncio.wait_for(FulltextSearchService().search(request), timeout=30)
    except Exception:
        # Do not expose database addresses or embedding credentials in errors.
        logger.warning("全文只读查询未完成：intent_id=%s", request.intent_id)
        raise HTTPException(status_code=503, detail="全文查询暂不可用，请保留摘要依据") from None
