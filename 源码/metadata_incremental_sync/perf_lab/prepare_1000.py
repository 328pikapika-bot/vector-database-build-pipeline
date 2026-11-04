"""Read 1,000 real Solr records and batch their embeddings without touching Milvus/Kafka."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from incremental_sync.contract import (  # noqa: E402
    ID_RE, SOLR_FIELDS, VECTOR_FIELDS, PermanentDataError,
    source_values, validate_value, vector_inputs,
)

OUT = ROOT / "out/metadata_incremental_perf_20261020"
SOURCES = OUT / "sources.jsonl"
DATASET = OUT / "dataset.jsonl"
SOLR_URL = "http://solr-bak1.example.invalid:8984/solr/paper6.6/select"
EMBEDDING_URL = "http://127.0.0.1:16703/v1/embeddings"
MODEL = "qwen3-embed"
TARGET = 1000


def json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def key_from_local_env() -> str:
    path = ROOT / "vector_sync_service/.env"
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("EMBEDDING_API_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("local embedding key is unavailable")


def collect_sources(session: requests.Session) -> list[dict]:
    if SOURCES.exists():
        rows = json_lines(SOURCES)
        if len(rows) != TARGET or len({item["id"] for item in rows}) != TARGET:
            raise RuntimeError("existing source manifest is incomplete; preserve and inspect it")
        return rows
    rows, seen, skipped = [], set(), 0
    started = time.monotonic()
    for offset in range(0, 5000, 500):
        response = session.get(SOLR_URL, params={"q": "*:*", "sort": "id asc", "start": offset,
            "rows": 500, "fl": ",".join(SOLR_FIELDS), "wt": "json"}, timeout=90)
        response.raise_for_status()
        body = response.json()
        if body.get("responseHeader", {}).get("partialResults"):
            raise RuntimeError("Solr returned partial results")
        for doc in body["response"]["docs"]:
            doc_id = str(doc.get("id", ""))
            if not ID_RE.fullmatch(doc_id) or doc_id in seen:
                skipped += 1
                continue
            try:
                values = source_values(doc, doc_id)
            except PermanentDataError:
                skipped += 1
                continue
            seen.add(doc_id)
            rows.append({"id": doc_id, "values": values})
            if len(rows) == TARGET:
                break
        if len(rows) == TARGET:
            break
    if len(rows) != TARGET:
        raise RuntimeError(f"Solr yielded only {len(rows)} valid unique records")
    with SOURCES.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"source_count": len(rows), "skipped": skipped,
        "source_seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
    return rows


def embed_batch(session: requests.Session, rows: list[dict]) -> list[dict]:
    inputs, mapping = [], []
    for item in rows:
        for field, text in vector_inputs(item["values"]).items():
            mapping.append((item["id"], field + "_embed"))
            inputs.append(text)
    result: dict[str, dict] = {item["id"]: {} for item in rows}
    if inputs:
        response = session.post(EMBEDDING_URL, json={"model": MODEL, "input": inputs}, timeout=120)
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != len(inputs):
            raise RuntimeError("embedding count differs from input count")
        observed = set()
        for position, item in enumerate(data):
            index = int(item.get("index", position))
            if index < 0 or index >= len(mapping) or index in observed:
                raise RuntimeError("embedding response indexes are invalid")
            observed.add(index)
            doc_id, field = mapping[index]
            vector = item["embedding"]
            validate_value(field, vector)
            result[doc_id][field] = vector
    return [{**item, "vectors": result[item["id"]]} for item in rows]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    sources = collect_sources(session)
    existing = json_lines(DATASET)
    source_ids = [item["id"] for item in sources]
    if [item["id"] for item in existing] != source_ids[:len(existing)]:
        raise RuntimeError("existing dataset does not match fixed Solr manifest")
    session.headers["Authorization"] = "Bearer " + key_from_local_env()
    started = time.monotonic()
    api_calls, text_count = 0, 0
    with DATASET.open("a", encoding="utf-8") as stream:
        for start in range(len(existing), TARGET, 8):
            group = sources[start:start + 8]
            if not group:
                break
            for attempt in range(3):
                try:
                    generated = embed_batch(session, group)
                    break
                except (requests.RequestException, ValueError, KeyError):
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            for item in generated:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                text_count += len(item["vectors"])
            stream.flush()
            os.fsync(stream.fileno())
            api_calls += 1
            if (start + len(group)) % 100 == 0:
                print(json.dumps({"prepared": start + len(group), "target": TARGET}), flush=True)
            time.sleep(0.75)
    session.close()
    dataset = json_lines(DATASET)
    if len(dataset) != TARGET or len({item["id"] for item in dataset}) != TARGET:
        raise RuntimeError("prepared dataset is not exactly 1,000 distinct records")
    digest = hashlib.sha256("\n".join(source_ids).encode()).hexdigest()
    report = {"records": TARGET, "id_manifest_sha256": digest,
        "vector_fields": list(VECTOR_FIELDS), "embedding_inputs_this_run": text_count,
        "embedding_requests_this_run": api_calls,
        "embedding_elapsed_seconds_this_run": round(time.monotonic() - started, 2),
        "source": "Solr paper6.6", "milvus_writes": 0, "kafka_commits": 0}
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
