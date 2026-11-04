"""Copy a bounded 2026 journal-paper sample into a separate Milvus collection.

The source collection is only read. The target keeps every source field and adds
one freshly embedded title/keywords/abstract vector. Progress is resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections


SOURCE = "scientific_documents_v2"
SOURCE_ID = "467035880531670094"
TARGET = "scientific_documents_v2_merged_100k_20261024"
URI = "http://127.0.0.1:19533"
FILTER = 'year == "2026" and sysSubType in ["J02", "Z01"] and cls == "Article"'
MODEL_URL = "http://127.0.0.1:16703/v1/embeddings"
MODEL = "qwen3-embed"
LIMIT = 100_000
DIM = 2560
INSTRUCTION = (
    "Generate a dense vector for the scientific paper based on its title, keywords, and abstract.\n"
    "The vector should primarily encode its research domain and sub-domain.\n"
    "Analyze the text for broad field, specific sub-field, core concepts, goals, techniques, experiments, tools, and results."
)
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "out" / "merged_2026_100k"
DB = STATE / "checkpoint.db"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def connect() -> MilvusClient:
    return MilvusClient(uri=URI, db_name="default", timeout=120)


def validate_source(client: MilvusClient) -> list[str]:
    description = client.describe_collection(SOURCE)
    fields = description["fields"]
    assert str(description["collection_id"]) == SOURCE_ID, "source collection identity changed"
    assert len(fields) == 41, "source field count changed"
    assert {field["name"] for field in fields if field["type"] == DataType.FLOAT_VECTOR} == {
        "title_embed", "abstract_embed", "keyword_embed"
    }, "source vector fields changed"
    return [field["name"] for field in fields]


def create_or_validate_target(client: MilvusClient, source_fields: list[str]) -> None:
    if not client.has_collection(TARGET):
        connections.connect(alias="merged_sample", uri=URI, db_name="default")
        old_schema = Collection(SOURCE, using="merged_sample").schema
        new_schema = CollectionSchema(
            fields=[*old_schema.fields, FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)],
            description="2026 J02/Z01 Article sample; original 41 fields plus title-keywords-abstract embedding",
            enable_dynamic_field=old_schema.enable_dynamic_field,
        )
        Collection(TARGET, schema=new_schema, using="merged_sample")
        print(json.dumps({"event": "collection_created", "name": TARGET, "at": now()}), flush=True)
    target = client.describe_collection(TARGET)
    target_fields = {field["name"]: field for field in target["fields"]}
    assert set(target_fields) == set(source_fields) | {"text_embedding"}, "target schema mismatch"
    assert target_fields["text_embedding"]["params"]["dim"] == DIM, "target vector dimension mismatch"


def ensure_indexes(client: MilvusClient) -> None:
    existing = set(client.list_indexes(TARGET))
    for index_name in client.list_indexes(SOURCE):
        if index_name in existing:
            continue
        info = client.describe_index(SOURCE, index_name)
        params: dict[str, Any] = {"index_type": info["index_type"], "index_name": index_name}
        if "metric_type" in info:
            params["metric_type"] = info["metric_type"]
        if info["index_type"] == "IVF_PQ":
            params["params"] = {key: int(info[key]) for key in ("nlist", "m", "nbits")}
        index = client.prepare_index_params()
        index.add_index(field_name=info["field_name"], **params)
        client.create_index(TARGET, index)
        existing.add(index_name)
        print(json.dumps({"event": "index_created", "name": index_name, "at": now()}), flush=True)
    if "text_embedding" not in existing:
        index = client.prepare_index_params()
        index.add_index(
            field_name="text_embedding", index_name="text_embedding", index_type="HNSW",
            metric_type="L2", params={"M": 16, "efConstruction": 256},
        )
        client.create_index(TARGET, index)
        print(json.dumps({"event": "index_created", "name": "text_embedding", "at": now()}), flush=True)
    assert len(client.list_indexes(TARGET)) == len(client.list_indexes(SOURCE)) + 1


def open_state() -> sqlite3.Connection:
    STATE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS sample (seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'selected', attempts INTEGER NOT NULL DEFAULT 0, error TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT OR IGNORE INTO meta VALUES ('source_id', ?)", (SOURCE_ID,))
    db.execute("INSERT OR IGNORE INTO meta VALUES ('filter', ?)", (FILTER,))
    db.execute("INSERT OR IGNORE INTO meta VALUES ('target', ?)", (TARGET,))
    db.commit()
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta["source_id"] == SOURCE_ID and meta["filter"] == FILTER and meta["target"] == TARGET
    return db


def select_ids(client: MilvusClient, db: sqlite3.Connection) -> None:
    selected = db.execute("SELECT COUNT(*) FROM sample").fetchone()[0]
    if selected >= LIMIT:
        return
    seen = {row[0] for row in db.execute("SELECT id FROM sample")}
    iterator = client.query_iterator(
        collection_name=SOURCE, filter=FILTER, output_fields=["id"],
        batch_size=500, consistency_level="Strong", timeout=120,
    )
    try:
        while selected < LIMIT:
            rows = iterator.next()
            if not rows:
                raise RuntimeError(f"only {selected} eligible source IDs found")
            additions = []
            for row in rows:
                identifier = row["id"]
                if identifier not in seen:
                    selected += 1
                    seen.add(identifier)
                    additions.append((selected, identifier))
                    if selected == LIMIT:
                        break
            db.executemany("INSERT INTO sample(seq,id) VALUES (?,?)", additions)
            db.commit()
            if selected % 5000 == 0 or selected == LIMIT:
                print(json.dumps({"event": "selected", "count": selected, "at": now()}), flush=True)
    finally:
        iterator.close()


def model_key() -> str:
    text = Path(r"C:\Users\NSTL\.codex\skills\model-config\references\model-environments.md").read_text(encoding="utf-8")
    block = text.split("### 测试环境", 1)[1].split("### 生产环境", 1)[0]
    key = next(line.split("=", 1)[1].strip().strip('"').strip("'")
               for line in block.splitlines() if line.startswith("LLM_API_KEY="))
    assert key and key != "[REDACTED]"
    return key


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = next(iter(value), "")
    text = str(value)
    return text.split("|", 1)[1] if "|" in text else text


def embed(session: requests.Session, key: str, rows: list[dict[str, Any]]) -> list[list[float]]:
    texts = []
    for row in rows:
        title = first_text(row.get("title"))
        abstract = first_text(row.get("abstract"))
        keywords = row.get("keyword") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = ",".join(first_text(item) for item in keywords if item)
        texts.append(f"{INSTRUCTION}\n\nTitle: {title}\n\nKeywords: {keywords}\n\nAbstract: {abstract}")
    def request_chunk(chunk: list[str]) -> list[list[float]]:
        for attempt in range(4):
            response = None
            try:
                response = session.post(
                    MODEL_URL, headers={"Authorization": "Bearer " + key},
                    json={"model": MODEL, "input": chunk}, timeout=120,
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                vectors = [item["embedding"] for item in data]
                assert len(vectors) == len(chunk) and all(len(vector) == DIM for vector in vectors)
                return vectors
            except requests.HTTPError as exc:
                body = response.text[:1000] if response is not None else str(exc)
                context_overflow = (
                    response is not None and response.status_code == 400
                    and any(marker in body.lower() for marker in (
                        "maximum context length", "max context length", "context length",
                        "requested tokens", "input for embedding generation",
                    ))
                )
                if context_overflow:
                    if len(chunk) > 1:
                        midpoint = len(chunk) // 2
                        return request_chunk(chunk[:midpoint]) + request_chunk(chunk[midpoint:])
                    text = chunk[0]
                    if len(text) > 2:
                        midpoint = len(text) // 2
                        boundary = text.rfind(" ", max(1, midpoint - 2048), min(len(text), midpoint + 2048))
                        if boundary > 0:
                            midpoint = boundary
                        left = text[:midpoint].strip()
                        right = (INSTRUCTION + "\n\nContinuation of the same paper: "
                                 + text[midpoint:].strip())
                        left_vector = request_chunk([left])[0]
                        right_vector = request_chunk([right])[0]
                        # Pool both halves so the stored vector represents the complete
                        # input while respecting the model's per-request context limit.
                        weight_total = max(len(left) + len(right), 1)
                        pooled = [
                            (float(a) * len(left) + float(b) * len(right)) / weight_total
                            for a, b in zip(left_vector, right_vector)
                        ]
                        norm = math.sqrt(sum(value * value for value in pooled))
                        if not math.isfinite(norm) or norm == 0:
                            raise RuntimeError("embedding failed: pooled vector is invalid")
                        return [[value / norm for value in pooled]]
                    raise RuntimeError(
                        "embedding failed: input exceeds model context limit for a single record; "
                        f"status=400; detail={body[:300]}"
                    ) from None
                if attempt == 3:
                    raise RuntimeError(
                        f"embedding failed: {type(exc).__name__}; "
                        f"status={getattr(response, 'status_code', 'none')}"
                    ) from None
                time.sleep(2 ** attempt)
            except (requests.RequestException, KeyError, ValueError, AssertionError) as exc:
                if attempt == 3:
                    raise RuntimeError(
                        f"embedding failed: {type(exc).__name__}; "
                        f"status={getattr(response, 'status_code', 'none')}"
                    ) from None
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    return request_chunk(texts)


def normalize(row: dict[str, Any], fields: list[str],
              allowed_years: tuple[str, ...] = ("2026",)) -> dict[str, Any]:
    result = {}
    for field in fields:
        value = row.get(field)
        if field in {"title_embed", "abstract_embed", "keyword_embed"}:
            assert value is not None and len(value) == DIM, f"invalid source vector: {field}"
            result[field] = list(value)
        elif value is not None and not isinstance(value, (str, int, float, bool, dict, list)):
            result[field] = list(value)
        else:
            result[field] = value
    assert result["year"] in allowed_years and result["sysSubType"] in {"J02", "Z01"}
    assert result["cls"] == "Article"
    return result


def process(client: MilvusClient, db: sqlite3.Connection, fields: list[str],
            batch_size: int, stop_after: int, workers: int) -> None:
    key = model_key()
    sessions = threading.local()
    done = db.execute("SELECT COUNT(*) FROM sample WHERE status='done'").fetchone()[0]
    if done >= stop_after:
        return
    pending = [row[0] for row in db.execute(
        "SELECT id FROM sample WHERE status!='done' ORDER BY seq LIMIT ?", (stop_after - done,)
    )]
    assert len(pending) == stop_after - done, "sample exhausted before target count"
    batches = iter([pending[i:i + batch_size] for i in range(0, len(pending), batch_size)])

    def write_batch(ids: list[str]) -> list[str]:
        if not hasattr(sessions, "http"):
            sessions.http = requests.Session()
        expr = "id in " + json.dumps(ids, ensure_ascii=False)
        fetched = client.query(SOURCE, filter=expr, output_fields=fields, consistency_level="Strong", timeout=120)
        mapped = {row["id"]: normalize(row, fields) for row in fetched}
        assert set(mapped) == set(ids), "selected source IDs missing or duplicated"
        ordered = [mapped[identifier] for identifier in ids]
        vectors = embed(sessions.http, key, ordered)
        for row, vector in zip(ordered, vectors):
            row["text_embedding"] = vector
        client.upsert(TARGET, data=ordered, timeout=120)
        return ids

    first_error = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def enqueue() -> bool:
            try:
                ids = next(batches)
            except StopIteration:
                return False
            active[pool.submit(write_batch, ids)] = ids
            return True

        for _ in range(workers * 2):
            if not enqueue():
                break
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                ids = active.pop(future)
                try:
                    future.result()
                    db.executemany("UPDATE sample SET status='done', attempts=attempts+1, error=NULL WHERE id=?", [(identifier,) for identifier in ids])
                    db.commit()
                    done += len(ids)
                    if done % 1000 < len(ids) or done == stop_after:
                        print(json.dumps({"event": "written", "done": done, "target": LIMIT, "at": now()}), flush=True)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {str(exc)[:160]}"
                    db.executemany("UPDATE sample SET status='failed', attempts=attempts+1, error=? WHERE id=?", [(message, identifier) for identifier in ids])
                    db.commit()
                    first_error = exc
                if first_error is None:
                    enqueue()
    if first_error is not None:
        raise first_error


def verify(client: MilvusClient, db: sqlite3.Connection, expected: int) -> None:
    client.flush(TARGET)
    client.load_collection(TARGET, timeout=600)
    done = db.execute("SELECT COUNT(*) FROM sample WHERE status='done'").fetchone()[0]
    count = client.query(TARGET, filter="", output_fields=["count(*)"], consistency_level="Strong", timeout=120)[0]["count(*)"]
    assert count == done == expected, f"target count={count}, checkpoint done={done}, expected={expected}"
    bad = client.query(TARGET, filter=f'year != "2026" or sysSubType not in ["J02", "Z01"] or cls != "Article"', output_fields=["count(*)"], consistency_level="Strong", timeout=120)[0]["count(*)"]
    assert bad == 0, f"unexpected rows: {bad}"
    samples = client.query(TARGET, filter="", output_fields=["id", "text_embedding", "title", "abstract", "keyword"], limit=3, consistency_level="Strong", timeout=120)
    assert len(samples) == 3 and all(len(row["text_embedding"]) == DIM for row in samples)
    report = {"collection": TARGET, "source_collection": SOURCE, "source_collection_id": SOURCE_ID,
              "filter": FILTER, "selected": LIMIT, "written": done, "strong_count": count,
              "invalid_filter_count": bad, "field_count": len(client.describe_collection(TARGET)["fields"]),
              "index_count": len(client.list_indexes(TARGET)), "verified_at": now()}
    (STATE / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "verified", **report}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", type=int, default=LIMIT, help="smoke test or full target, at most 100000")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.stop_after <= LIMIT and 1 <= args.batch_size <= 64 and 1 <= args.workers <= 4
    client = connect()
    fields = validate_source(client)
    db = open_state()
    create_or_validate_target(client, fields)
    ensure_indexes(client)
    select_ids(client, db)
    process(client, db, fields, args.batch_size, args.stop_after, args.workers)
    verify(client, db, args.stop_after)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"event": "stopped", "error_type": type(error).__name__,
                          "message": str(error)[:200], "at": now()}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise
