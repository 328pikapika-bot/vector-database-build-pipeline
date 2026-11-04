#!/usr/bin/env python3
"""Measure isolated IVFPQ Milvus ingestion for document, chunk, and image schemas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import DataType, Function, FunctionType, MilvusClient


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.models.base.article import Article
from app.models.chunk import Chunk
from app.services.milvus_chunk_service import MilvusChunkService
from app.services.milvus_image_service import IMAGE_SCALAR_INDEX_SPECS
from app.services.milvus_service import MilvusDynamicService


VECTOR_DIM = 2560
VISUAL_DIM = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--document-rows", type=int, default=1024)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--image-rows", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--keep-collections", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out") / "ivfpq_collection_benchmark.json")
    return parser.parse_args()


def _client() -> MilvusClient:
    return MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        user=settings.MILVUS_USER or "",
        password=settings.MILVUS_PASSWORD or "",
        timeout=120,
    )


def _random_vector(rng: np.random.Generator, dim: int, *, float16: bool = False) -> list[float] | np.ndarray:
    values = rng.random(dim, dtype=np.float32)
    if float16:
        return values.astype(np.float16)
    return values.tolist()


def _wait_for_indexes(client: MilvusClient, collection_name: str) -> float:
    started = time.perf_counter()
    connection = client._get_connection()
    for index_name in client.list_indexes(collection_name=collection_name):
        connection.wait_for_creating_index(
            collection_name,
            index_name,
            timeout=600,
            context=client._generate_call_context(),
        )
    return time.perf_counter() - started


def _rate(rows: int, seconds: float) -> float:
    return rows / seconds if seconds > 0 else 0.0


def _prepare_document_collection(name: str, nlist: int) -> MilvusDynamicService:
    settings.DOCUMENT_INDEX_TYPE = "IVFPQ"
    settings.DOCUMENT_IVFPQ_NLIST = nlist
    settings.DOCUMENT_IVFPQ_M = 64
    settings.DOCUMENT_IVFPQ_NBITS = 8
    service = MilvusDynamicService(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    service.init_collection(name, VECTOR_DIM)
    return service


def _document_rows(count: int, rng: np.random.Generator, prefix: str) -> list[Article]:
    rows: list[Article] = []
    for index in range(count):
        doc_id = f"{prefix}-doc-{index:08d}"
        rows.append(
            Article(
                sysId=doc_id,
                cls="Article",
                title=["IVFPQ document benchmark"],
                abstract=["Measured Milvus document ingestion payload."],
                keyword=["benchmark"],
                year="2026",
                language=["zh"],
                accessType=["OA"],
                classificationCode=["TEST"],
                title_embed=_random_vector(rng, VECTOR_DIM),
                abstract_embed=_random_vector(rng, VECTOR_DIM),
                keyword_embed=_random_vector(rng, VECTOR_DIM),
            )
        )
    return rows


def _prepare_chunk_collection(name: str, nlist: int) -> MilvusChunkService:
    settings.FULLTEXT_CHUNK_INDEX_TYPE = "IVFPQ"
    settings.FULLTEXT_CHUNK_IVFPQ_NLIST = nlist
    settings.FULLTEXT_CHUNK_IVFPQ_M = 64
    settings.FULLTEXT_CHUNK_IVFPQ_NBITS = 8
    service = MilvusChunkService()
    service.init_chunk_collection(name, VECTOR_DIM)
    return service


def _chunk_rows(count: int, rng: np.random.Generator, prefix: str) -> list[Chunk]:
    rows: list[Chunk] = []
    for index in range(count):
        doc_id = f"{prefix}-doc-{index // 64:08d}"
        rows.append(
            Chunk(
                chunkId=f"{prefix}-chunk-{index:08d}",
                docId=doc_id,
                title=["IVFPQ chunk benchmark"],
                keyword=["benchmark"],
                accessType="OA",
                classificationCode="TEST",
                year="2026",
                hasAuthorname=["benchmark"],
                hasCreateOrganizationname=["NSTL"],
                language=["zh"],
                chunkIndex=index % 64,
                chunkType="paragraph",
                text="Measured Milvus fulltext chunk ingestion payload.",
                charStart=0,
                charEnd=48,
                sectionPath="benchmark",
                tokenCount=8,
                embedding=_random_vector(rng, VECTOR_DIM),
            )
        )
    return rows


def _prepare_image_collection(client: MilvusClient, name: str, nlist: int) -> None:
    if client.has_collection(name):
        raise RuntimeError(f"refusing to reuse existing collection: {name}")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("doc_meta", DataType.JSON)
    schema.add_field("raw_img_meta", DataType.JSON, nullable=True)
    schema.add_field("img_meta", DataType.JSON)
    schema.add_field("context_before", DataType.VARCHAR, max_length=65535, nullable=True)
    schema.add_field("context_after", DataType.VARCHAR, max_length=65535, nullable=True)
    schema.add_field("p_hash", DataType.BINARY_VECTOR, dim=64)
    schema.add_field("visual_embed", DataType.FLOAT16_VECTOR, dim=VISUAL_DIM)
    schema.add_field("caption_embed", DataType.FLOAT16_VECTOR, dim=VECTOR_DIM)
    schema.add_field("context_embed", DataType.FLOAT16_VECTOR, dim=VECTOR_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, nullable=True, enable_analyzer=True, analyzer_params={"type": "chinese"})
    schema.add_field("text_embed", DataType.FLOAT16_VECTOR, dim=VECTOR_DIM)
    schema.add_field("visual_text_embed", DataType.FLOAT16_VECTOR, dim=VISUAL_DIM)
    schema.add_field("text_sparse_vec", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="text_to_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["text_sparse_vec"],
        )
    )
    params = client.prepare_index_params()
    params.add_index(field_name="p_hash", index_type="BIN_FLAT", metric_type="HAMMING")
    for field_name in ("visual_embed", "caption_embed", "context_embed", "text_embed", "visual_text_embed"):
        params.add_index(
            field_name=field_name,
            index_type="IVF_PQ",
            metric_type="IP",
            params={"nlist": nlist, "m": 64, "nbits": 8},
        )
    params.add_index(field_name="text_sparse_vec", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for spec in IMAGE_SCALAR_INDEX_SPECS:
        params.add_index(
            field_name=spec["field_name"],
            index_name=spec["index_name"],
            index_type=spec["index_type"],
            params=dict(spec.get("params") or {}),
        )
    client.create_collection(collection_name=name, schema=schema, index_params=params)
    client.load_collection(name)


def _image_rows(count: int, rng: np.random.Generator, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        doc_id = f"{prefix}-doc-{index // 5:08d}"
        rows.append(
            {
                "id": f"{prefix}-image-{index:08d}",
                "doc_meta": {"docId": doc_id, "year": "2026", "language": ["zh"], "accessType": ["OA"]},
                "raw_img_meta": {"source": "benchmark"},
                "img_meta": {"img_id": f"{prefix}-image-{index:08d}"},
                "context_before": "benchmark before context",
                "context_after": "benchmark after context",
                "p_hash": bytes.fromhex("0123456789abcdef"),
                "visual_embed": _random_vector(rng, VISUAL_DIM, float16=True),
                "caption_embed": _random_vector(rng, VECTOR_DIM, float16=True),
                "context_embed": _random_vector(rng, VECTOR_DIM, float16=True),
                "text": "IVFPQ image benchmark text",
                "text_embed": _random_vector(rng, VECTOR_DIM, float16=True),
                "visual_text_embed": _random_vector(rng, VISUAL_DIM, float16=True),
            }
        )
    return rows


def _insert_batches(client: MilvusClient, collection_name: str, rows: list[Any], batch_size: int, *, chunk_service: MilvusChunkService | None = None, document_service: MilvusDynamicService | None = None) -> dict[str, float]:
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        if document_service is not None:
            document_service.process_and_insert(collection_name, batch, VECTOR_DIM)
        elif chunk_service is not None:
            chunk_service.insert_chunks(collection_name, batch, vector_dim=VECTOR_DIM)
        else:
            client.insert(collection_name=collection_name, data=batch)
    insert_seconds = time.perf_counter() - started
    flush_started = time.perf_counter()
    client.flush(collection_name=collection_name)
    flush_seconds = time.perf_counter() - flush_started
    return {"insert_seconds": insert_seconds, "flush_seconds": flush_seconds}


def _run_role(role: str, count: int, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    name = f"benchmark_ivfpq_{role}_{args.run_id}".lower()
    client = _client()
    if client.has_collection(name):
        raise RuntimeError(f"refusing to reuse existing collection: {name}")
    created_at = time.perf_counter()
    service: MilvusDynamicService | MilvusChunkService | None = None
    try:
        if role == "document":
            service = _prepare_document_collection(name, args.nlist)
            rows = _document_rows(count, rng, args.run_id)
        elif role == "chunk":
            service = _prepare_chunk_collection(name, args.nlist)
            rows = _chunk_rows(count, rng, args.run_id)
        else:
            _prepare_image_collection(client, name, args.nlist)
            rows = _image_rows(count, rng, args.run_id)
        create_seconds = time.perf_counter() - created_at
        insert_result = _insert_batches(
            client,
            name,
            rows,
            args.batch_size,
            chunk_service=service if isinstance(service, MilvusChunkService) else None,
            document_service=service if isinstance(service, MilvusDynamicService) else None,
        )
        index_seconds = _wait_for_indexes(client, name)
        rows_written = int(client.get_collection_stats(collection_name=name).get("row_count") or 0)
        return {
            "role": role,
            "collection_name": name,
            "rows_requested": count,
            "rows_written": rows_written,
            "batch_size": args.batch_size,
            "nlist": args.nlist,
            "m": 64,
            "nbits": 8,
            "create_seconds": create_seconds,
            **insert_result,
            "index_wait_seconds": index_seconds,
            "write_rows_per_second": _rate(rows_written, insert_result["insert_seconds"]),
            "write_rows_per_hour": _rate(rows_written, insert_result["insert_seconds"]) * 3600,
        }
    finally:
        if service is not None:
            service.close()
        if client.has_collection(name) and not args.keep_collections:
            client.drop_collection(name)
        client.close()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(20260722)
    report = {
        "started_at": time.time(),
        "milvus_uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        "index_type_config": "IVFPQ",
        "index_type_sent_to_milvus": "IVF_PQ",
        "vector_dim": VECTOR_DIM,
        "keep_collections": args.keep_collections,
        "results": [
            _run_role("document", args.document_rows, args, rng),
            _run_role("chunk", args.chunk_rows, args, rng),
            _run_role("image", args.image_rows, args, rng),
        ],
    }
    report["finished_at"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
