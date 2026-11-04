"""Manual semantic-search utility.

This file is intentionally pytest-collectable but defines no tests. Run it
directly only in an authorized live environment.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import httpx
from pymilvus import MilvusClient


MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19531
MILVUS_COLLECTION = "scientific_documents_v29"
VECTOR_DIM = 2560

EMBEDDING_API_URL = "http://127.0.0.1:8080/v1/embeddings"
EMBEDDING_MODEL = "medium"

METRIC_TYPE = "COSINE"
SEARCH_LIMIT = 5

OUTPUT_FIELDS = [
    "ID",
    "title",
    "abstract",
    "keyword",
    "year",
    "sysSubType",
    "language",
    "doi",
    "url",
    "startPage",
    "pageRange",
]

VECTOR_FIELDS = {
    "title": "title_embed",
    "abstract": "abstract_embed",
    "keyword": "keyword_embed",
}


async def embed_text(text: str) -> list[float]:
    payload = {"model": EMBEDDING_MODEL, "input": text}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(EMBEDDING_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    embedding_items = data.get("data", [])
    if not embedding_items:
        raise RuntimeError(f"Embedding API returned no data: {data}")
    vector = embedding_items[0].get("embedding", [])
    if not isinstance(vector, list):
        raise RuntimeError(f"Embedding API returned invalid vector: {data}")
    return vector


def connect_milvus() -> MilvusClient:
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    if not client.has_collection(MILVUS_COLLECTION):
        raise RuntimeError(f"Collection does not exist: {MILVUS_COLLECTION}")
    client.load_collection(MILVUS_COLLECTION)
    return client


def search_in_field(
    client: MilvusClient,
    vector: list[float],
    vector_field: str,
    limit: int,
) -> list[dict[str, Any]]:
    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[vector],
        anns_field=vector_field,
        limit=limit,
        search_params={"metric_type": METRIC_TYPE, "params": {"nprobe": 16}},
        output_fields=OUTPUT_FIELDS,
    )
    return list(results[0]) if results else []


def format_result(hit: dict[str, Any], rank: int, search_field: str) -> str:
    entity = hit.get("entity", hit)
    score = hit.get("distance", hit.get("score", 0))
    title = "; ".join(entity.get("title") or []) or "(no title)"
    abstract = "; ".join(entity.get("abstract") or []) or "(no abstract)"
    keyword = ", ".join(entity.get("keyword") or []) or "-"
    return "\n".join(
        [
            f"#{rank} score={score:.6f} field={search_field}",
            f"  title: {title[:120]}",
            f"  abstract: {abstract[:200]}",
            f"  keyword: {keyword}",
            f"  ID={entity.get('ID')} year={entity.get('year')} type={entity.get('sysSubType')}",
        ]
    )


async def run_semantic_search(
    query: str,
    *,
    top_k: int = SEARCH_LIMIT,
    search_title: bool = True,
    search_abstract: bool = True,
    search_keyword: bool = True,
) -> None:
    vector = await embed_text(query)
    if len(vector) != VECTOR_DIM:
        print(f"warning: expected vector dim {VECTOR_DIM}, got {len(vector)}")

    client = connect_milvus()
    try:
        active_fields: list[tuple[str, str]] = []
        if search_title:
            active_fields.append(("title", VECTOR_FIELDS["title"]))
        if search_abstract:
            active_fields.append(("abstract", VECTOR_FIELDS["abstract"]))
        if search_keyword:
            active_fields.append(("keyword", VECTOR_FIELDS["keyword"]))

        for label, field_name in active_fields:
            print(f"\n--- search {label} ({field_name}) ---")
            for rank, hit in enumerate(search_in_field(client, vector, field_name, top_k), start=1):
                print(format_result(hit, rank, label))
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Milvus semantic search")
    parser.add_argument("query", help="query text")
    parser.add_argument("--top-k", type=int, default=SEARCH_LIMIT)
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--no-abstract", action="store_true")
    parser.add_argument("--no-keyword", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run_semantic_search(
            args.query,
            top_k=args.top_k,
            search_title=not args.no_title,
            search_abstract=not args.no_abstract,
            search_keyword=not args.no_keyword,
        )
    )


if __name__ == "__main__":
    main()
