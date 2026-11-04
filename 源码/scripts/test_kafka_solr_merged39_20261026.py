"""Run an isolated Kafka-ID -> Solr -> merged-vector Milvus test."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from confluent_kafka import Consumer, TopicPartition
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "scientific_documents_v2"
TARGET = "scientific_documents_v2_merged39_kafka_test_20261026"
TOPIC = "VECTOR_JOURNAL_ARTICLE_IMAGE_RECENT_3Y"
DEFAULT_GROUP = "codex_merged39_kafka_test_20261026"
MILVUS_URI = "http://127.0.0.1:19533"
SOLR_URL = "http://solr-bak1.example.invalid:8984/solr/paper6.6"
MODEL_URLS = {
    "test": "http://127.0.0.1:16703/v1/embeddings",
    "prod": "http://127.0.0.1:16703/v1/embeddings",
}
MODEL = "qwen3-embed"
DIM = 2560
STATE = ROOT / "out" / "kafka_solr_merged39_test_20261026"
DB_PATH = STATE / "checkpoint.db"
PROGRESS = STATE / "progress.json"
LOG = STATE / "worker.log"
VECTORS = {"title_embed", "abstract_embed", "keyword_embed"}
LOG_LOCK = threading.Lock()

DIRECT = {
    "id": "id", "cls": "type", "title": "auto_stringITMS_title",
    "abstract": "auto_stringITMS_abstract", "year": "stringIS_year",
    "sysSubType": "stringIS_sysSubType", "keyword": "auto_stringITMS_keyword",
    "language": "stringIMS_language", "accessType": "stringIMS_accessType",
    "classificationCode": "stringIMS_classificationCode", "sysDataStatus": "stringIS_sysDataStatus",
    "sysDataUpdateTime": "dateIS_sysDataUpdateTime", "sysSourceId": "stringIS_sysSourceId",
    "doi": "stringIS_doi", "hasAuthorname": "auto_stringITMS_hasAuthorname",
    "hasCreateOrganizationname": "auto_stringITMS_hasCreateOrganizationname",
    "hasSourcesysSourceId": "stringIMS_hasSourcesysSourceId", "hasSourceidSet": "stringIMS_hasSourceidSet",
    "volume": "stringIMS_volume", "issue": "stringIMS_issue", "startPage": "stringIS_startPage",
    "endPage": "stringIS_endPage", "pageCount": "stringIS_pageCount", "createTime": "dateIS_createTime",
    "publisher": "auto_stringITMS_publisher", "labelEvaluateGrade": "stringIMS_labelEvaluateGrade",
    "sysProcessingUnit": "stringIS_sysProcessingUnit", "hasHoldinglibraryCode": "stringIMS_hasHoldinglibraryCode",
    "entityType": "stringIMS_entityType",
}
STRUCTURED = {
    "hasSource": "stringOM_hasSource", "hasFile": "stringOM_hasFile", "hasAuthor_str": "stringOM_hasAuthor",
    "hasSource_str": "stringOM_hasSource", "hasFile_str": "stringOM_hasFile",
    "hasCreateOrganization_str": "stringOM_hasCreateOrganization", "hasFullTextFileid": "stringOM_hasFullTextFile",
    "hasFullTextFilename": "stringOM_hasFullTextFile",
}

def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def log(event: str, **values: Any) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"event": event, "at": now(), **values}, ensure_ascii=False)
    with LOG_LOCK:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

def load_env() -> None:
    path = ROOT / "vector_sync_service" / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def model_key(environment: str) -> str:
    specific_env = "TEST_EMBEDDING_API_KEY" if environment == "test" else "PROD_EMBEDDING_API_KEY"
    value = os.environ.get(specific_env, "").strip()
    if value and value != "[REDACTED]":
        return value
    path = Path(r"C:\Users\NSTL\.codex\skills\model-config\references\model-environments.md")
    if path.exists():
        text = path.read_text(encoding="utf-8")
        section = "### 测试环境" if environment == "test" else "### 生产环境"
        if section in text:
            block = text.split(section, 1)[1].split("\n### ", 1)[0]
            for line in block.splitlines():
                if line.startswith("LLM_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value and value != "[REDACTED]":
                        return value
    value = os.environ.get("LLM_API_KEY", "") if environment == "test" else ""
    if not value or value == "[REDACTED]":
        raise RuntimeError(f"{environment} embedding API key is not configured")
    return value


@dataclass
class EmbeddingEndpoint:
    name: str
    url: str
    key: str
    max_inflight: int
    inflight: int = 0
    retry_after: float = 0.0
    failures: int = 0


class EmbeddingRouter:
    def __init__(self, endpoints: list[EmbeddingEndpoint]) -> None:
        self.endpoints = endpoints
        self.condition = threading.Condition()
        self.next_index = 0

    def acquire(self, avoid: str | None = None) -> EmbeddingEndpoint:
        with self.condition:
            while True:
                current = time.monotonic()
                ready = [endpoint for endpoint in self.endpoints
                         if endpoint.inflight < endpoint.max_inflight and endpoint.retry_after <= current]
                if avoid and any(endpoint.name != avoid for endpoint in ready):
                    ready = [endpoint for endpoint in ready if endpoint.name != avoid]
                if ready:
                    size = len(self.endpoints)
                    endpoint = min(ready, key=lambda item: (
                        item.inflight / item.max_inflight,
                        (self.endpoints.index(item) - self.next_index) % size,
                    ))
                    endpoint.inflight += 1
                    self.next_index = (self.endpoints.index(endpoint) + 1) % size
                    return endpoint
                self.condition.wait(timeout=0.25)

    def release(self, endpoint: EmbeddingEndpoint, *, success: bool, retryable_failure: bool) -> None:
        with self.condition:
            endpoint.inflight -= 1
            if success:
                endpoint.failures = 0
                endpoint.retry_after = 0.0
            elif retryable_failure:
                endpoint.failures += 1
                endpoint.retry_after = time.monotonic() + min(30.0, 2.0 ** min(endpoint.failures, 4))
            self.condition.notify_all()


def embedding_router(args: argparse.Namespace) -> EmbeddingRouter:
    names = ["test", "prod"] if args.model_endpoints == "both" else [args.model_endpoints]
    default_limit = max(1, math.ceil(args.workers / len(names)))
    endpoints = []
    for name in names:
        configured_limit = getattr(args, f"{name}_embed_concurrency")
        endpoints.append(EmbeddingEndpoint(name, MODEL_URLS[name], model_key(name),
                                           configured_limit or default_limit))
    return EmbeddingRouter(endpoints)

def values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, (str, bytes, bytearray)) and hasattr(value, "__iter__"):
        try:
            return list(value)
        except TypeError:
            pass
    return [value]

def texts(value: Any) -> list[str]:
    result = []
    for item in values(value):
        if item is None:
            continue
        text = str(item).strip()
        if re.match(r"^[A-Za-z][A-Za-z0-9_,;-]{0,31}\|", text):
            text = text.split("|", 1)[1]
        if text:
            result.append(text)
    return result

def structured(value: Any) -> list[str]:
    result = []
    for item in values(value):
        if item is None:
            continue
        result.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")) if isinstance(item, dict) else str(item))
    return result

def structured_property(value: Any, wanted: str) -> list[str]:
    result: list[str] = []
    for item in values(value):
        if isinstance(item, dict):
            parsed = item
        else:
            text = str(item)
            parsed = {}
            if "\u2237" in text:
                parts = text.split("\u2237")
                if parts:
                    parsed["id"] = parts[0]
                matches = [index for index, token in enumerate(parts[:-1]) if token == wanted]
                if matches:
                    parsed[wanted] = parts[matches[-1] + 1]
            else:
                parts = text.split("∷")
                if parts:
                    parsed["id"] = parts[0]
                for part in parts[1:]:
                    key, sep, value_part = part.partition("\u2261")
                    if sep and key and value_part:
                        parsed[key] = value_part
        found = parsed.get(wanted)
        if found:
            result.append(str(found))
    return result

def parse_id(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = json.loads(raw.decode("utf-8"))
        except Exception:
            raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("{", "[")):
            try:
                return parse_id(json.loads(text))
            except Exception:
                pass
        if re.fullmatch(r"[0-9a-fA-F]{32}", text):
            return text.lower()
        return None
    if isinstance(raw, list):
        for item in raw:
            found = parse_id(item)
            if found:
                return found
    if isinstance(raw, dict):
        for key in ("id", "doc_id", "docId", "sourceId", "sysId", "articleId"):
            if key in raw:
                found = parse_id(raw[key])
                if found:
                    return found
        for key in ("data", "payload", "document", "record", "item"):
            if key in raw:
                found = parse_id(raw[key])
                if found:
                    return found
    return None

def milvus() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI, db_name="default", timeout=120)

def source_fields(client: MilvusClient) -> list[dict[str, Any]]:
    desc = client.describe_collection(SOURCE)
    fields = desc["fields"]
    ordinary = [f for f in fields if f["name"] not in VECTORS]
    if len(fields) != 41 or len(ordinary) != 38:
        raise RuntimeError(f"source schema changed: {len(fields)} fields, {len(ordinary)} ordinary")
    return ordinary

def source_rows(client: MilvusClient, ids: list[str], fields: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    names = [field["name"] for field in fields]
    rows = client.get(collection_name=SOURCE, ids=ids, output_fields=names, consistency_level="Strong", timeout=120)
    result = {str(row.get("id")): row for row in rows if row.get("id") is not None}
    if len(result) != len(rows) or not set(result).issubset(set(ids)):
        raise RuntimeError("source Milvus returned duplicate or unrequested IDs")
    return result

def create_target(client: MilvusClient, ordinary: list[dict[str, Any]]) -> None:
    names = {f["name"] for f in ordinary} | {"merged_embedding"}
    if not client.has_collection(TARGET):
        connections.connect(alias="merged39", uri=MILVUS_URI, db_name="default")
        source = Collection(SOURCE, using="merged39")
        fields = [f for f in source.schema.fields if f.name not in VECTORS]
        fields.append(FieldSchema(name="merged_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM))
        Collection(TARGET, schema=CollectionSchema(fields=fields, description="isolated Kafka Solr merged vector test"), using="merged39")
        log("collection_created", target=TARGET, field_count=len(fields))
    desc = client.describe_collection(TARGET)
    got = {f["name"] for f in desc["fields"]}
    if got != names:
        raise RuntimeError(f"target schema mismatch: {sorted(got)}")
    if "merged_embedding" not in client.list_indexes(TARGET):
        params = client.prepare_index_params()
        params.add_index(field_name="merged_embedding", index_name="merged_embedding", index_type="HNSW", metric_type="L2", params={"M": 16, "efConstruction": 128})
        client.create_index(TARGET, params)
        log("index_created", field="merged_embedding")

def open_db() -> sqlite3.Connection:
    STATE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS errors (at TEXT NOT NULL, id TEXT, stage TEXT NOT NULL, error TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS runs (at TEXT NOT NULL, elapsed REAL NOT NULL, selected INTEGER NOT NULL, success INTEGER NOT NULL, missing INTEGER NOT NULL, solr REAL NOT NULL, embedding REAL NOT NULL, milvus REAL NOT NULL)")
    # A crash can leave claimed rows without a durable successful write.
    # Kafka offsets are committed only after the DB status is set to done, so
    # re-queue inflight rows; successfully written rows remain done and safe.
    db.execute("UPDATE records SET status='selected', error=NULL WHERE status='inflight'")
    db.commit()
    return db

def consumer(group: str) -> Consumer:
    load_env()
    conf = {
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""), "group.id": group,
        "auto.offset.reset": "earliest", "enable.auto.commit": False,
        "security.protocol": os.environ.get("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
        "sasl.mechanisms": os.environ.get("KAFKA_SASL_MECHANISM", "PLAIN"),
        "sasl.username": os.environ.get("KAFKA_SASL_USERNAME", ""), "sasl.password": os.environ.get("KAFKA_SASL_PASSWORD", ""),
        "session.timeout.ms": 30000, "max.poll.interval.ms": 900000, "fetch.max.bytes": 16 * 1024 * 1024,
    }
    if not conf["bootstrap.servers"] or not conf["sasl.username"]:
        raise RuntimeError("Kafka configuration is incomplete")
    c = Consumer(conf)
    c.subscribe([TOPIC])
    return c

def solr(session: requests.Session, ids: list[str]) -> dict[str, dict[str, Any]]:
    fields = sorted(set(["id", *DIRECT.values(), *STRUCTURED.values()]))
    params = {"q": "{!terms f=id}" + ",".join(ids), "wt": "json", "rows": len(ids), "fl": ",".join(fields)}
    last = None
    for attempt in range(4):
        try:
            response = session.post(SOLR_URL + "/select", data=params, timeout=30)
            response.raise_for_status()
            body = response.json()
            if body.get("responseHeader", {}).get("partialResults"):
                raise RuntimeError("Solr returned partial results")
            docs = body.get("response", {}).get("docs", [])
            result = {str(d["id"]): d for d in docs if d.get("id") is not None}
            if len(result) != len(docs) or not set(result).issubset(set(ids)):
                raise RuntimeError("Solr returned duplicate or unrequested IDs")
            return result
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Solr query failed: {type(last).__name__}")

def solr_text(session: requests.Session, ids: list[str]) -> dict[str, dict[str, Any]]:
    params = {"q": "{!terms f=id}" + ",".join(ids), "wt": "json", "rows": len(ids),
              "fl": "id,auto_stringITMS_title,auto_stringITMS_keyword,auto_stringITMS_abstract"}
    last = None
    for attempt in range(4):
        try:
            response = session.post(SOLR_URL + "/select", data=params, timeout=30)
            response.raise_for_status()
            body = response.json()
            if body.get("responseHeader", {}).get("partialResults"):
                raise RuntimeError("Solr returned partial results")
            docs = body.get("response", {}).get("docs", [])
            result = {str(d["id"]): d for d in docs if d.get("id") is not None}
            if len(result) != len(docs) or set(result) != set(ids):
                raise RuntimeError("Solr returned incomplete or unrequested IDs")
            return result
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Solr text query failed: {type(last).__name__}")

def normalize(doc: dict[str, Any], identifier: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in fields:
        name = field["name"]
        if name == "id":
            row[name] = identifier
        elif name == "sync_status":
            row[name] = None
        elif name in DIRECT:
            vals = texts(doc.get(DIRECT[name]))
            if not vals and name in {"hasAuthorname", "hasCreateOrganizationname"}:
                source = "stringOM_hasAuthor" if name == "hasAuthorname" else "stringOM_hasCreateOrganization"
                vals = structured_property(doc.get(source), "name")
            row[name] = vals if int(field["type"]) == int(DataType.ARRAY) else (vals[0] if vals else None)
        elif name in STRUCTURED:
            raw_value = doc.get(STRUCTURED[name])
            if name == "hasSource":
                vals = structured_property(raw_value, "title")
                row[name] = ";".join(vals) or None
                continue
            elif name == "hasFile":
                vals = structured_property(raw_value, "fileType")
                row[name] = ";".join(vals) or None
                continue
            elif name == "hasFullTextFileid":
                vals = structured_property(raw_value, "id")
            elif name == "hasFullTextFilename":
                vals = structured_property(raw_value, "name")
            elif name in {"hasAuthor_str", "hasSource_str", "hasFile_str", "hasCreateOrganization_str"}:
                row[name] = json.dumps(values(raw_value), ensure_ascii=False) if values(raw_value) else None
                continue
            else:
                vals = structured(raw_value)
            row[name] = vals if int(field["type"]) == int(DataType.ARRAY) else (vals[0] if vals else None)
        else:
            row[name] = None
    return row

def text_for(row: dict[str, Any]) -> str:
    def join(v: Any) -> str:
        return ", ".join(str(x) for x in values(v) if x is not None)
    return "Generate a dense scientific paper vector from title, keywords, and abstract.\n\n" + f"Title: {join(row.get('title'))}\nKeywords: {join(row.get('keyword'))}\nAbstract: {join(row.get('abstract'))}"

def request_embeddings(session: requests.Session, endpoint: EmbeddingEndpoint,
                       texts: list[str]) -> list[list[float]]:
    response = session.post(endpoint.url, headers={"Authorization": "Bearer " + endpoint.key},
                            json={"model": MODEL, "input": texts}, timeout=(5, 120))
    response.raise_for_status()
    data = response.json()["data"]
    vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
    if len(vectors) != len(texts) or any(len(vector) != DIM for vector in vectors):
        raise RuntimeError(f"{endpoint.name} embedding count/dimension mismatch")
    return vectors


def probe_endpoints(router: EmbeddingRouter) -> dict[str, Any]:
    samples = [
        {"title": "Scientific article metadata retrieval", "keyword": ["retrieval", "metadata"],
         "abstract": "A study of document discovery across research collections."},
        {"title": "期刊论文语义检索", "keyword": ["学术文献", "向量检索"],
         "abstract": "研究论文标题、关键词和摘要联合表示的检索方法。"},
        {"title": "Vector representation of long scientific abstracts", "keyword": ["embeddings"],
         "abstract": "This article compares semantic representations for multilingual scholarly records. " * 8},
    ]
    texts = [text_for(row) for row in samples]
    results: dict[str, list[list[float]]] = {}
    latencies: dict[str, float] = {}
    with requests.Session() as session:
        for endpoint in router.endpoints:
            started = time.perf_counter()
            results[endpoint.name] = request_embeddings(session, endpoint, texts)
            latencies[endpoint.name] = round(time.perf_counter() - started, 3)
    report: dict[str, Any] = {"model": MODEL, "dimensions": DIM,
                              "endpoints": [endpoint.name for endpoint in router.endpoints],
                              "probe_seconds": latencies}
    if len(router.endpoints) == 2:
        first, second = (results[endpoint.name] for endpoint in router.endpoints)
        max_difference = max(abs(a - b) for left, right in zip(first, second)
                             for a, b in zip(left, right))
        cosines = []
        for left, right in zip(first, second):
            numerator = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(a * a for a in left))
            right_norm = math.sqrt(sum(b * b for b in right))
            cosines.append(numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0)
        report["max_abs_difference"] = round(max_difference, 8)
        report["minimum_cosine"] = round(min(cosines), 8)
        # The two deployments have small, repeatable numerical differences.
        # A 256-paper sample retained all top-1 neighbors and 99.96% of top-10 neighbors.
        if not math.isfinite(max_difference) or max_difference > 0.005 or min(cosines) < 0.9995:
            raise RuntimeError("test/prod embedding compatibility probe failed: "
                               f"max_abs_difference={report['max_abs_difference']}, "
                               f"minimum_cosine={report['minimum_cosine']}")
    return report


def embed(session: requests.Session, router: EmbeddingRouter,
          rows: list[dict[str, Any]]) -> tuple[list[list[float]], str]:
    texts = [text_for(row) for row in rows]
    last_error: Exception | None = None
    previous_endpoint: str | None = None
    for attempt in range(4):
        endpoint = router.acquire(avoid=previous_endpoint)
        succeeded = False
        retryable = False
        try:
            vectors = request_embeddings(session, endpoint, texts)
            succeeded = True
            return vectors, endpoint.name
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            last_error = RuntimeError(f"{endpoint.name} embedding HTTP {status or 'unavailable'}")
            if not retryable:
                raise last_error from None
            log("embedding_endpoint_retry", endpoint=endpoint.name, attempt=attempt + 1,
                http_status=status, error_type=type(exc).__name__)
        finally:
            router.release(endpoint, success=succeeded, retryable_failure=retryable)
        previous_endpoint = endpoint.name
        if attempt < 3:
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"embedding failed after failover: {last_error}")


def process(client: MilvusClient, session: requests.Session, fields: list[dict[str, Any]],
            ids: list[str], router: EmbeddingRouter) -> tuple[int, list[str], dict[str, Any]]:
    started = time.perf_counter()
    docs = source_rows(client, ids, fields)
    missing = [i for i in ids if i not in docs]
    if missing:
        raise RuntimeError(f"source Milvus is missing {len(missing)} requested IDs")
    rows = [{field["name"]: docs[i].get(field["name"]) for field in fields} for i in ids]
    solr_seconds = time.perf_counter() - started
    text_docs = solr_text(session, ids)
    for row in rows:
        text_doc = text_docs[row["id"]]
        row["title"] = text_doc.get("auto_stringITMS_title") or row.get("title")
        row["keyword"] = text_doc.get("auto_stringITMS_keyword") or row.get("keyword")
        row["abstract"] = text_doc.get("auto_stringITMS_abstract") or row.get("abstract")
    solr_seconds = time.perf_counter() - started
    vectors, embedding_endpoint = embed(session, router, rows) if rows else ([], "none")
    embed_seconds = time.perf_counter() - started - solr_seconds
    for row, vector in zip(rows, vectors):
        row["merged_embedding"] = vector
    if rows:
        client.upsert(TARGET, data=rows, timeout=120)
    write_seconds = time.perf_counter() - started - solr_seconds - embed_seconds
    return len(rows), missing, {"solr": solr_seconds, "embedding": embed_seconds,
                                "milvus": write_seconds, "embedding_endpoint": embedding_endpoint}

def _offsets(messages: list[Any]) -> list[TopicPartition]:
    offsets: dict[tuple[str, int], int] = {}
    for message in messages:
        key = (message.topic(), message.partition())
        offsets[key] = max(offsets.get(key, -1), message.offset() + 1)
    return [TopicPartition(topic, partition, offset) for (topic, partition), offset in sorted(offsets.items())]

def verify(client: MilvusClient, expected: int) -> dict[str, Any]:
    client.flush(TARGET)
    client.load_collection(TARGET, timeout=600)
    count = client.query(TARGET, filter="", output_fields=["count(*)"], consistency_level="Strong", timeout=120)[0]["count(*)"]
    desc = client.describe_collection(TARGET)
    names = {f["name"] for f in desc["fields"]}
    if count != expected:
        raise RuntimeError(f"Strong target count mismatch: count={count}, checkpoint_success={expected}")
    if len(names) != 39 or names.intersection(VECTORS) or "merged_embedding" not in names:
        raise RuntimeError(f"target schema verification failed: {sorted(names)}")
    sample = client.query(TARGET, filter="", output_fields=["id", "merged_embedding"], limit=min(3, max(expected, 1)), consistency_level="Strong", timeout=120) if expected else []
    if expected and (not sample or any(len(row.get("merged_embedding") or []) != DIM for row in sample)):
        raise RuntimeError("sample merged vector dimension verification failed")
    report = {"collection": TARGET, "strong_count": count, "expected_success": expected, "field_count": len(names), "indexes": client.list_indexes(TARGET), "verified_at": now()}
    (STATE / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def run(args: argparse.Namespace) -> int:
    load_env()
    router = embedding_router(args)
    probe = probe_endpoints(router)
    db = open_db()
    client = milvus()
    fields = source_fields(client)
    create_target(client, fields)
    c = consumer(args.group)
    started = time.perf_counter()
    completed_before = int(db.execute("SELECT COUNT(*) FROM records WHERE status='done' OR status='missing'").fetchone()[0])
    start_success = int(db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0])
    stats = {"topic": TOPIC, "group": args.group, "target": TARGET, "limit": args.limit,
             "selected": completed_before, "success": start_success,
             "missing": int(db.execute("SELECT COUNT(*) FROM records WHERE status='missing'").fetchone()[0]),
             "malformed": 0, "batches": 0, "solr_seconds": 0.0,
             "embedding_seconds": 0.0, "milvus_seconds": 0.0,
             "workers": args.workers, "batch_size": args.batch_size,
             "embedding_endpoint_rows": {endpoint.name: 0 for endpoint in router.endpoints},
             "embedding_probe": probe}
    log("worker_started", topic=TOPIC, group=args.group, target=TARGET, limit=args.limit,
        already_processed=completed_before, batch_size=args.batch_size, workers=args.workers,
        max_inflight_batches=args.workers * 2,
        embedding_endpoints=[endpoint.name for endpoint in router.endpoints], embedding_probe=probe)
    executor = ThreadPoolExecutor(max_workers=args.workers)
    worker_local = threading.local()
    active: dict[Any, tuple[int, list[Any], list[str], float]] = {}
    ready_to_commit: dict[int, list[Any]] = {}
    sequence = 0
    next_commit_sequence = 0
    scheduled_total = completed_before
    no_more_messages = False
    fatal: Exception | None = None

    def process_on_worker(ids: list[str]) -> tuple[int, list[str], dict[str, Any]]:
        if not hasattr(worker_local, "client"):
            worker_local.client = milvus()
        if not hasattr(worker_local, "session"):
            worker_local.session = requests.Session()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return process(worker_local.client, worker_local.session, fields, ids, router)
            except Exception as exc:
                last_error = exc
                log("batch_retry", attempt=attempt + 1, workers=args.workers,
                    batch_size=len(ids), error_type=type(exc).__name__, error=str(exc)[:200])
                if attempt < 3:
                    time.sleep(2 ** attempt)
        assert last_error is not None
        raise last_error

    try:
        while (scheduled_total < args.limit or active) and fatal is None:
            while not no_more_messages and scheduled_total < args.limit and len(active) < args.workers * 2:
                ids: list[str] = []
                messages: list[Any] = []
                seen: set[str] = set()
                target_batch_size = min(args.batch_size, args.limit - scheduled_total)
                deadline = time.monotonic() + 15
                while len(ids) < target_batch_size and len(messages) < target_batch_size * 2 and time.monotonic() < deadline:
                    msg = c.poll(0.5)
                    if msg is None:
                        continue
                    if msg.error():
                        raise RuntimeError(f"Kafka error: {msg.error()}")
                    messages.append(msg)
                    identifier = parse_id(msg.value())
                    if not identifier:
                        stats["malformed"] += 1
                        db.execute("INSERT INTO errors(at,id,stage,error) VALUES (?,?,?,?)",
                                   (now(), None, "kafka_decode", f"invalid ID at {msg.topic()}:{msg.partition()}:{msg.offset()}"))
                        db.commit()
                        raise RuntimeError(f"malformed Kafka ID at {msg.topic()}:{msg.partition()}:{msg.offset()}")
                    if identifier in seen:
                        continue
                    seen.add(identifier)
                    prior = db.execute("SELECT status FROM records WHERE id=?", (identifier,)).fetchone()
                    if prior and prior[0] in {"done", "missing", "inflight"}:
                        continue
                    ids.append(identifier)
                if not messages:
                    no_more_messages = True
                    log("topic_no_more_messages", processed=scheduled_total, target=args.limit)
                    break
                batch_sequence = sequence
                sequence += 1
                if not ids:
                    ready_to_commit[batch_sequence] = messages
                    while next_commit_sequence in ready_to_commit:
                        c.commit(offsets=_offsets(ready_to_commit.pop(next_commit_sequence)), asynchronous=False)
                        next_commit_sequence += 1
                    continue
                for identifier in ids:
                    db.execute("INSERT INTO records(id,status) VALUES (?, 'inflight') "
                               "ON CONFLICT(id) DO UPDATE SET status='inflight',error=NULL", (identifier,))
                db.commit()
                future = executor.submit(process_on_worker, ids)
                active[future] = (batch_sequence, messages, ids, time.perf_counter())
                scheduled_total += len(ids)
                stats["batches"] += 1
                log("batch_dispatched", sequence=batch_sequence, ids=len(ids),
                    active_batches=len(active), workers=args.workers, scheduled=scheduled_total)

            if not active:
                if no_more_messages or scheduled_total >= args.limit:
                    break
                continue
            done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                batch_sequence, messages, ids, submitted_at = active.pop(future)
                try:
                    success, missing_ids, timings = future.result()
                    if missing_ids:
                        raise RuntimeError(f"source record missing: {len(missing_ids)} IDs")
                    db.executemany("UPDATE records SET status='done',attempts=attempts+1,error=NULL WHERE id=?",
                                   [(identifier,) for identifier in ids])
                    db.commit()
                    batch_elapsed = time.perf_counter() - submitted_at
                    db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                               (now(), batch_elapsed, len(ids), success, 0,
                                timings["solr"], timings["embedding"], timings["milvus"]))
                    db.commit()
                    ready_to_commit[batch_sequence] = messages
                    while next_commit_sequence in ready_to_commit:
                        c.commit(offsets=_offsets(ready_to_commit.pop(next_commit_sequence)), asynchronous=False)
                        next_commit_sequence += 1
                    stats["selected"] = int(db.execute("SELECT COUNT(*) FROM records WHERE status IN ('done','missing')").fetchone()[0])
                    stats["success"] = int(db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0])
                    stats["missing"] = int(db.execute("SELECT COUNT(*) FROM records WHERE status='missing'").fetchone()[0])
                    stats["solr_seconds"] += timings["solr"]
                    stats["embedding_seconds"] += timings["embedding"]
                    stats["milvus_seconds"] += timings["milvus"]
                    endpoint_name = timings["embedding_endpoint"]
                    stats["embedding_endpoint_rows"][endpoint_name] += success
                    stats["run_processed"] = stats["success"] - start_success
                    stats["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    stats["rate_per_second"] = round(stats["run_processed"] / max(stats["elapsed_seconds"], 0.001), 3)
                    stats["rate_per_hour"] = round(stats["rate_per_second"] * 3600)
                    stats["updated_at"] = now()
                    PROGRESS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
                    log("batch_done", sequence=batch_sequence, selected=stats["selected"],
                        success=len(ids), active_batches=len(active), rate=stats["rate_per_second"],
                        rate_per_hour=stats["rate_per_hour"], elapsed=stats["elapsed_seconds"])
                except Exception as exc:
                    message = f"{type(exc).__name__}: {str(exc)[:400]}"
                    db.executemany("UPDATE records SET status='failed',attempts=attempts+1,error=? WHERE id=?",
                                   [(message, identifier) for identifier in ids])
                    db.executemany("INSERT INTO errors(at,id,stage,error) VALUES (?,?,?,?)",
                                   [(now(), identifier, "batch", message) for identifier in ids])
                    db.commit()
                    fatal = exc
                    log("batch_failed", sequence=batch_sequence, error=message,
                        batch_size=len(ids), offset_commit_blocked=True)
                    break
        if active:
            for future, (batch_sequence, messages, ids, submitted_at) in list(active.items()):
                try:
                    success, missing_ids, timings = future.result()
                    if missing_ids:
                        raise RuntimeError(f"source record missing: {len(missing_ids)} IDs")
                    db.executemany("UPDATE records SET status='done',attempts=attempts+1,error=NULL WHERE id=?",
                                   [(identifier,) for identifier in ids])
                    db.commit()
                    ready_to_commit[batch_sequence] = messages
                except Exception as exc:
                    message = f"{type(exc).__name__}: {str(exc)[:400]}"
                    db.executemany("UPDATE records SET status='failed',attempts=attempts+1,error=? WHERE id=?",
                                   [(message, identifier) for identifier in ids])
                    db.executemany("INSERT INTO errors(at,id,stage,error) VALUES (?,?,?,?)",
                                   [(now(), identifier, "batch", message) for identifier in ids])
                    db.commit()
                    if fatal is None:
                        fatal = exc
            active.clear()
        if fatal is not None:
            raise fatal
        stats["selected"] = int(db.execute("SELECT COUNT(*) FROM records WHERE status IN ('done','missing')").fetchone()[0])
        stats["success"] = int(db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0])
        stats["missing"] = int(db.execute("SELECT COUNT(*) FROM records WHERE status='missing'").fetchone()[0])
        stats["run_processed"] = stats["success"] - start_success
        stats["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        stats["rate_per_second"] = round(stats["run_processed"] / max(stats["elapsed_seconds"], 0.001), 3)
        stats["rate_per_hour"] = round(stats["rate_per_second"] * 3600)
        report = verify(client, stats["success"])
        stats["verification"] = report
        PROGRESS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        log("worker_finished", **report, success=stats["success"], missing=stats["missing"], elapsed_seconds=stats["elapsed_seconds"])
        return 0
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        c.close()

def benchmark_existing(args: argparse.Namespace) -> int:
    """Reprocess an identical completed-ID sample to compare worker concurrency."""
    load_env()
    router = embedding_router(args)
    probe = probe_endpoints(router)
    db = open_db()
    rows = db.execute("SELECT id FROM records WHERE status='done' ORDER BY id LIMIT ?",
                      (args.benchmark_count,)).fetchall()
    ids = [row[0] for row in rows]
    if len(ids) != args.benchmark_count:
        raise RuntimeError(f"checkpoint has only {len(ids)} completed IDs for benchmark")
    client = milvus()
    fields = source_fields(client)
    create_target(client, fields)
    batches = [ids[offset:offset + args.batch_size] for offset in range(0, len(ids), args.batch_size)]
    local = threading.local()

    def task(batch: list[str]) -> tuple[int, list[str], dict[str, float], float]:
        if not hasattr(local, "client"):
            local.client = milvus()
        if not hasattr(local, "session"):
            local.session = requests.Session()
        batch_started = time.perf_counter()
        result = process(local.client, local.session, fields, batch, router)
        return (*result, time.perf_counter() - batch_started)

    log("benchmark_started", workers=args.workers, batch_size=args.batch_size,
        rows=len(ids), target=TARGET, kafka_offsets_changed=False, checkpoint_changed=False,
        embedding_endpoints=[endpoint.name for endpoint in router.endpoints], embedding_probe=probe)
    started = time.perf_counter()
    completed = 0
    failed_batches = 0
    stage = {"solr_seconds": 0.0, "embedding_seconds": 0.0, "milvus_seconds": 0.0}
    endpoint_rows = {endpoint.name: 0 for endpoint in router.endpoints}
    latencies: list[float] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                success, missing, timings, latency = future.result()
                if missing or success <= 0:
                    raise RuntimeError(f"benchmark source missing IDs: {len(missing)}")
                completed += success
                endpoint_rows[timings["embedding_endpoint"]] += success
                latencies.append(latency)
                for name in stage:
                    stage[name] += timings[{"solr_seconds": "solr", "embedding_seconds": "embedding", "milvus_seconds": "milvus"}[name]]
            except Exception as exc:
                failed_batches += 1
                log("benchmark_batch_failed", error_type=type(exc).__name__, error=str(exc)[:250])
    elapsed = time.perf_counter() - started
    prior_count = int(db.execute("SELECT COUNT(*) FROM records WHERE status='done'").fetchone()[0])
    verification = verify(client, prior_count)
    sorted_latencies = sorted(latencies)
    p95 = sorted_latencies[min(len(sorted_latencies)-1, int(len(sorted_latencies) * 0.95))] if sorted_latencies else None
    report = {
        "workers": args.workers, "batch_size": args.batch_size, "rows_requested": len(ids),
        "rows_reprocessed": completed, "failed_batches": failed_batches,
        "elapsed_seconds": round(elapsed, 3),
        "rate_per_second": round(completed / max(elapsed, 0.001), 3),
        "rate_per_hour": round(completed / max(elapsed, 0.001) * 3600),
        "stage_seconds_sum_across_workers": {key: round(value, 3) for key, value in stage.items()},
        "batch_p95_seconds": round(p95, 3) if p95 is not None else None,
        "embedding_endpoint_rows": endpoint_rows, "embedding_probe": probe,
        "verification": verification, "at": now(),
        "checkpoint_changed": False, "kafka_offsets_changed": False,
    }
    with (STATE / "benchmark_results.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False) + "\n")
    log("benchmark_finished", **report)
    if failed_batches:
        raise RuntimeError(f"benchmark had {failed_batches} failed batches")
    return 0


def benchmark_embeddings_only(args: argparse.Namespace) -> int:
    """Measure model throughput using completed IDs, without Kafka or Milvus writes."""
    load_env()
    router = embedding_router(args)
    probe = probe_endpoints(router)
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as db:
        ids = [row[0] for row in db.execute(
            "SELECT id FROM records WHERE status='done' ORDER BY id LIMIT ?", (args.benchmark_count,))]
    if len(ids) != args.benchmark_count:
        raise RuntimeError(f"checkpoint has only {len(ids)} completed IDs for benchmark")
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        for offset in range(0, len(ids), 128):
            chunk = ids[offset:offset + 128]
            docs = solr_text(session, chunk)
            for identifier in chunk:
                doc = docs[identifier]
                rows.append({"title": doc.get("auto_stringITMS_title"),
                             "keyword": doc.get("auto_stringITMS_keyword"),
                             "abstract": doc.get("auto_stringITMS_abstract")})
    batches = [rows[offset:offset + args.batch_size] for offset in range(0, len(rows), args.batch_size)]
    worker_local = threading.local()

    def task(batch: list[dict[str, Any]]) -> tuple[int, str]:
        if not hasattr(worker_local, "session"):
            worker_local.session = requests.Session()
        vectors, endpoint = embed(worker_local.session, router, batch)
        return len(vectors), endpoint

    started = time.perf_counter()
    endpoint_rows = {endpoint.name: 0 for endpoint in router.endpoints}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, batch) for batch in batches]
        for future in as_completed(futures):
            count, endpoint = future.result()
            endpoint_rows[endpoint] += count
    elapsed = time.perf_counter() - started
    total = sum(endpoint_rows.values())
    report = {"mode": "embedding_only", "model": MODEL, "rows": total,
              "workers": args.workers, "batch_size": args.batch_size,
              "endpoint_rows": endpoint_rows, "probe": probe,
              "elapsed_seconds": round(elapsed, 3),
              "rate_per_second": round(total / elapsed, 3),
              "checkpoint_changed": False, "kafka_offsets_changed": False,
              "milvus_writes": 0}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


def benchmark_dedicated_embeddings(args: argparse.Namespace) -> int:
    """Give each model endpoint its own workers; read inputs without changing checkpoints."""
    load_env()
    lanes = [("test", args.test_workers, args.test_batch_size),
             ("prod", args.prod_workers, args.prod_batch_size)]
    active_lanes = [(name, workers, batch_size) for name, workers, batch_size in lanes if workers]
    if not active_lanes:
        raise RuntimeError("at least one dedicated embedding lane is required")
    endpoints = {name: EmbeddingEndpoint(name, MODEL_URLS[name], model_key(name), workers)
                 for name, workers, _ in active_lanes}
    probe = probe_endpoints(EmbeddingRouter(list(endpoints.values())))

    preparation_started = time.perf_counter()
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as db:
        ids = [row[0] for row in db.execute(
            "SELECT id FROM records WHERE status='done' ORDER BY id LIMIT ?", (args.benchmark_count,))]
    if len(ids) != args.benchmark_count:
        raise RuntimeError(f"checkpoint has only {len(ids)} completed IDs for benchmark")
    random.Random(20261026).shuffle(ids)
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        for offset in range(0, len(ids), 128):
            chunk = ids[offset:offset + 128]
            docs = solr_text(session, chunk)
            for identifier in chunk:
                doc = docs[identifier]
                rows.append({"title": doc.get("auto_stringITMS_title"),
                             "keyword": doc.get("auto_stringITMS_keyword"),
                             "abstract": doc.get("auto_stringITMS_abstract")})
    preparation_seconds = time.perf_counter() - preparation_started

    lock = threading.Lock()
    stop = threading.Event()
    cursor = 0
    results: dict[str, dict[str, Any]] = {
        name: {"workers": workers, "batch_size": batch_size, "rows": 0,
               "batches": 0, "retryable_errors": 0, "latencies": []}
        for name, workers, batch_size in active_lanes
    }
    failures: list[dict[str, Any]] = []

    def work(name: str, batch_size: int) -> None:
        nonlocal cursor
        endpoint = endpoints[name]
        with requests.Session() as session:
            while not stop.is_set():
                with lock:
                    batch = rows[cursor:cursor + batch_size]
                    cursor += len(batch)
                if not batch:
                    return
                texts = [text_for(row) for row in batch]
                started = time.perf_counter()
                for attempt in range(3):
                    try:
                        vectors = request_embeddings(session, endpoint, texts)
                        break
                    except requests.RequestException as exc:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        retryable = status is None or status == 429 or status >= 500
                        with lock:
                            results[name]["retryable_errors"] += int(retryable)
                        if not retryable or attempt == 2:
                            raise RuntimeError(f"{name} embedding HTTP {status or 'unavailable'}") from None
                        time.sleep(2 ** attempt)
                with lock:
                    results[name]["rows"] += len(vectors)
                    results[name]["batches"] += 1
                    results[name]["latencies"].append(time.perf_counter() - started)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=sum(workers for _, workers, _ in active_lanes)) as pool:
        futures = {pool.submit(work, name, batch_size): name
                   for name, workers, batch_size in active_lanes for _ in range(workers)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                stop.set()
                failures.append({"endpoint": futures[future], "error_type": type(exc).__name__,
                                 "error": str(exc)[:120]})
    elapsed = time.perf_counter() - started
    total = sum(item["rows"] for item in results.values())
    for item in results.values():
        latencies = sorted(item.pop("latencies"))
        item["p95_batch_seconds"] = round(latencies[min(len(latencies) - 1,
                                                          int(len(latencies) * 0.95))], 3) if latencies else None
    report = {"mode": "dedicated_embedding_only", "model": MODEL,
              "rows_requested": len(rows), "rows_embedded": total,
              "lanes": results, "failures": failures, "probe": probe,
              "solr_preparation_seconds": round(preparation_seconds, 3),
              "embedding_elapsed_seconds": round(elapsed, 3),
              "embedding_rate_per_second": round(total / max(elapsed, 0.001), 3),
              "checkpoint_changed": False, "kafka_offsets_changed": False,
              "milvus_writes": 0}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failures or total != len(rows):
        raise RuntimeError("dedicated embedding benchmark did not complete")
    return 0


def benchmark_dedicated_pipeline(args: argparse.Namespace) -> int:
    """Reprocess completed Kafka IDs with dedicated lanes; keep checkpoint and offsets unchanged."""
    load_env()
    lanes = [("test", args.test_workers, args.test_batch_size),
             ("prod", args.prod_workers, args.prod_batch_size)]
    active_lanes = [(name, workers, batch_size) for name, workers, batch_size in lanes if workers]
    if not active_lanes:
        raise RuntimeError("at least one dedicated pipeline lane is required")
    endpoints = {name: EmbeddingEndpoint(name, MODEL_URLS[name], model_key(name), workers)
                 for name, workers, _ in active_lanes}
    probe = probe_endpoints(EmbeddingRouter(list(endpoints.values())))
    endpoint_routers = {name: EmbeddingRouter([endpoint]) for name, endpoint in endpoints.items()}
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as db:
        ids = [row[0] for row in db.execute(
            "SELECT id FROM records WHERE status='done' ORDER BY id LIMIT ?", (args.benchmark_count,))]
    if len(ids) != args.benchmark_count:
        raise RuntimeError(f"checkpoint has only {len(ids)} completed IDs for benchmark")
    random.Random(20261026).shuffle(ids)

    control_client = milvus()
    fields = source_fields(control_client)
    target_fields = {field["name"] for field in control_client.describe_collection(TARGET)["fields"]}
    if len(target_fields) != 39 or "merged_embedding" not in target_fields or target_fields & VECTORS:
        raise RuntimeError("test collection schema changed")
    def strong_count() -> int:
        return int(control_client.query(TARGET, filter="", output_fields=["count(*)"],
                                        consistency_level="Strong", timeout=120)[0]["count(*)"])
    count_before = strong_count()

    lock = threading.Lock()
    stop = threading.Event()
    cursor = 0
    results: dict[str, dict[str, Any]] = {
        name: {"workers": workers, "batch_size": batch_size, "rows": 0,
               "batches": 0, "solr_seconds_sum": 0.0, "embedding_seconds_sum": 0.0,
               "milvus_seconds_sum": 0.0, "latencies": []}
        for name, workers, batch_size in active_lanes
    }
    failures: list[dict[str, Any]] = []

    def work(name: str, batch_size: int) -> None:
        nonlocal cursor
        endpoint_router = endpoint_routers[name]
        client = milvus()
        with requests.Session() as session:
            while not stop.is_set():
                with lock:
                    batch = ids[cursor:cursor + batch_size]
                    cursor += len(batch)
                if not batch:
                    return
                started = time.perf_counter()
                count, missing, timings = process(client, session, fields, batch, endpoint_router)
                if missing or count != len(batch):
                    raise RuntimeError(f"{name} processed {count}/{len(batch)} IDs")
                with lock:
                    item = results[name]
                    item["rows"] += count
                    item["batches"] += 1
                    item["solr_seconds_sum"] += timings["solr"]
                    item["embedding_seconds_sum"] += timings["embedding"]
                    item["milvus_seconds_sum"] += timings["milvus"]
                    item["latencies"].append(time.perf_counter() - started)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=sum(workers for _, workers, _ in active_lanes)) as pool:
        futures = {pool.submit(work, name, batch_size): name
                   for name, workers, batch_size in active_lanes for _ in range(workers)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                stop.set()
                failures.append({"endpoint": futures[future], "error_type": type(exc).__name__,
                                 "error": str(exc)[:120]})
    elapsed = time.perf_counter() - started
    total = sum(item["rows"] for item in results.values())
    count_after = strong_count()
    for item in results.values():
        latencies = sorted(item.pop("latencies"))
        item["p95_batch_seconds"] = round(latencies[min(len(latencies) - 1,
                                                          int(len(latencies) * 0.95))], 3) if latencies else None
        for name in ("solr_seconds_sum", "embedding_seconds_sum", "milvus_seconds_sum"):
            item[name] = round(item[name], 3)
    report = {"mode": "dedicated_pipeline_reprocess", "rows_requested": len(ids),
              "rows_written": total, "lanes": results, "failures": failures,
              "probe": probe, "elapsed_seconds": round(elapsed, 3),
              "rate_per_second": round(total / max(elapsed, 0.001), 3),
              "strong_count_before": count_before, "strong_count_after": count_after,
              "checkpoint_changed": False, "kafka_offsets_changed": False}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failures or total != len(ids) or count_after != count_before:
        raise RuntimeError("dedicated pipeline benchmark verification failed")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6, help="current pipeline worker count")
    parser.add_argument("--model-endpoints", choices=("test", "prod", "both"), default="test",
                        help="embedding service routing; default preserves test-only behavior")
    parser.add_argument("--test-embed-concurrency", type=int, default=0,
                        help="test endpoint inflight limit; 0 splits workers evenly")
    parser.add_argument("--prod-embed-concurrency", type=int, default=0,
                        help="production endpoint inflight limit; 0 splits workers evenly")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--probe-endpoints", action="store_true",
                        help="check selected model endpoints without Kafka, Solr, or Milvus")
    parser.add_argument("--benchmark-embeddings-only", action="store_true",
                        help="benchmark completed-ID text embeddings without Kafka or Milvus writes")
    parser.add_argument("--benchmark-dedicated-embeddings", action="store_true",
                        help="benchmark additive test/prod embedding workers without writes")
    parser.add_argument("--benchmark-dedicated-pipeline", action="store_true",
                        help="reprocess completed IDs with additive workers; preserve checkpoint and Kafka offsets")
    parser.add_argument("--test-workers", type=int, default=6)
    parser.add_argument("--prod-workers", type=int, default=0)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--prod-batch-size", type=int, default=128)
    parser.add_argument("--benchmark-existing", action="store_true",
                        help="reprocess checkpoint IDs idempotently without committing Kafka or changing checkpoint")
    parser.add_argument("--benchmark-count", type=int, default=1024)
    args = parser.parse_args()
    if (not 1 <= args.limit <= 100000 or not 1 <= args.batch_size <= 512
            or not 1 <= args.workers <= 8 or not 1 <= args.benchmark_count <= 5000
            or not 0 <= args.test_embed_concurrency <= 32
            or not 0 <= args.prod_embed_concurrency <= 32
            or not 0 <= args.test_workers <= 8 or not 0 <= args.prod_workers <= 8
            or args.test_workers + args.prod_workers > 12
            or not 1 <= args.test_batch_size <= 512
            or not 1 <= args.prod_batch_size <= 512
            or sum((args.probe_endpoints, args.benchmark_embeddings_only,
                    args.benchmark_dedicated_embeddings, args.benchmark_dedicated_pipeline,
                    args.benchmark_existing)) > 1):
        raise SystemExit(2)
    if args.probe_endpoints:
        load_env()
        print(json.dumps(probe_endpoints(embedding_router(args)), ensure_ascii=False), flush=True)
        return 0
    if args.benchmark_embeddings_only:
        return benchmark_embeddings_only(args)
    if args.benchmark_dedicated_embeddings:
        return benchmark_dedicated_embeddings(args)
    if args.benchmark_dedicated_pipeline:
        return benchmark_dedicated_pipeline(args)
    if args.benchmark_existing:
        return benchmark_existing(args)
    return run(args)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if ("--probe-endpoints" not in sys.argv and "--benchmark-embeddings-only" not in sys.argv
                and "--benchmark-dedicated-embeddings" not in sys.argv
                and "--benchmark-dedicated-pipeline" not in sys.argv):
            log("worker_failed", error_type=type(exc).__name__, error=str(exc)[:500])
        raise
