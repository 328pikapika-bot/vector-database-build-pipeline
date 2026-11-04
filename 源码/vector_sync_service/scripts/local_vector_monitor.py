"""Small read-only monitor for the three vector-sync lanes.

It reads Kafka metadata, watermarks and committed offsets only.  It does not
subscribe, poll, commit or write to Kafka/Milvus.  A JSONL snapshot once per
minute is used to calculate daily Kafka consumption and an ETA in days.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic, time
from typing import Any
from urllib.parse import urlparse


SERVICE_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_FILE = Path(__file__).with_name("local_vector_monitor.html")
DEFAULT_MILVUS_URI = "http://127.0.0.1:19533"
DEFAULT_MILVUS_DATABASE = "default"
DEFAULT_SNAPSHOT_FILE = "/app/data/snapshots.jsonl"
DEFAULT_RETENTION_DAYS = 7
DEFAULT_RATE_WINDOW_DAYS = 1
MIN_ROUGH_SAMPLE_SECONDS = 30 * 60
MONGO_QUERY_TIMEOUT_MS = 30_000

MONGO_COLLECTIONS = {
    "failure_records": ("文献级失败", "doc_sync_failures"),
    "fulltext_tasks": ("全文任务", "doc_fulltext_tasks"),
    "image_tasks": ("图片任务", "doc_image_tasks"),
}
MONGO_COLLECTION_ENV = {
    "failure_records": "VECTOR_MONITOR_MONGO_FAILURE_COLLECTION",
    "fulltext_tasks": "VECTOR_MONITOR_MONGO_FULLTEXT_TASK_COLLECTION",
    "image_tasks": "VECTOR_MONITOR_MONGO_IMAGE_TASK_COLLECTION",
}
FAILURE_STATUSES = ("PENDING", "RETRYING", "DONE", "SKIPPED_STALE", "GAVE_UP", "HELD_HISTORY")
TASK_STATUSES = ("PENDING", "LEASED", "SUCCESS", "FAILURE", "SKIPPED", "HELD_HISTORY")

COLLECTIONS = {
    "documents": ("文献", "scientific_documents_v2"),
    "fulltext_chunks": ("全文切片", "scientific_fulltext_chunks_v2"),
    "images": ("图片", "scientific_images_v2"),
}
LANES = {
    "documents": ("VECTOR_JOURNAL_ARTICLE_METADATA_BASE", "vector_sync_metadata_v2--target-c7d6acbfbe71"),
    "fulltext_chunks": ("VECTOR_JOURNAL_ARTICLE_FULLTEXT", "vector_sync_fulltext_v2--target-75335bbcddee"),
    "images": ("VECTOR_JOURNAL_ARTICLE_IMAGE", "vector_sync_image_v2--target-e72d233f5c35"),
}
ATTU_HINT = {
    "txt_attu": "http://127.0.0.1:18000",
    "txt_external_milvus": DEFAULT_MILVUS_URI,
    "txt_internal_milvus": "scientific-milvus-standalone:19530",
    "database": DEFAULT_MILVUS_DATABASE,
}
WORKER_TOPIC_ENV = {
    "documents": "KAFKA_METADATA_TOPIC",
    "fulltext_chunks": "KAFKA_FULLTEXT_TOPIC",
    "images": "KAFKA_IMAGE_TOPIC",
}
WORKER_GROUP_ENV = {
    "documents": "KAFKA_METADATA_GROUP_ID",
    "fulltext_chunks": "KAFKA_FULLTEXT_GROUP_ID",
    "images": "KAFKA_IMAGE_GROUP_ID",
}

_PROJECT_ENV: dict[str, str] | None = None
_SUMMARY_LOCK = threading.Lock()
_LAST_SUMMARY: dict[str, Any] | None = None
_LAST_SUMMARY_AT = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _project_env() -> dict[str, str]:
    """Use the local worker .env for convenient local viewing only."""
    global _PROJECT_ENV
    if _PROJECT_ENV is not None:
        return _PROJECT_ENV
    values: dict[str, str] = {}
    try:
        for line in (SERVICE_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    except OSError:
        pass
    _PROJECT_ENV = values
    return values


def _value(monitor_key: str, worker_key: str = "", default: str = "") -> str:
    return str(
        os.getenv(monitor_key)
        or (os.getenv(worker_key) if worker_key else "")
        or (_project_env().get(worker_key) if worker_key else "")
        or default
    ).strip()


def get_config() -> dict[str, Any]:
    retention_days = max(1, min(30, _safe_int(os.getenv("VECTOR_MONITOR_SNAPSHOT_RETENTION_DAYS", "7"))))
    rate_window_days = max(
        1,
        min(retention_days, _safe_int(os.getenv("VECTOR_MONITOR_RATE_WINDOW_DAYS", str(DEFAULT_RATE_WINDOW_DAYS)))),
    )
    refresh_seconds = max(30, min(300, _safe_int(os.getenv("VECTOR_MONITOR_REFRESH_SECONDS", "60"))))
    lanes = {}
    for key, (topic, group_id) in LANES.items():
        prefix = key.upper()
        lanes[key] = {
            "topic": _value(f"VECTOR_MONITOR_{prefix}_TOPIC", WORKER_TOPIC_ENV[key], topic),
            "group_id": _value(f"VECTOR_MONITOR_{prefix}_GROUP_ID", WORKER_GROUP_ENV[key], group_id),
        }
    return {
        "milvus_uri": _value("VECTOR_MONITOR_MILVUS_URI", "MILVUS_URI", DEFAULT_MILVUS_URI),
        "milvus_database": os.getenv("VECTOR_MONITOR_MILVUS_DATABASE", DEFAULT_MILVUS_DATABASE),
        "milvus_token": os.getenv("VECTOR_MONITOR_MILVUS_TOKEN", ""),
        "mongo_uri": _value("VECTOR_MONITOR_MONGO_URI", "MONGO_URI"),
        "mongo_database": _value(
            "VECTOR_MONITOR_MONGO_DB",
            "MONGO_DB",
            "nstl_vector_sync_scientific_prod",
        ),
        "mongo_collections": {
            key: os.getenv(MONGO_COLLECTION_ENV[key], name)
            for key, (_, name) in MONGO_COLLECTIONS.items()
        },
        "collections": {
            key: {
                "label": label,
                "name": os.getenv(f"VECTOR_MONITOR_{key.upper()}_COLLECTION", name),
            }
            for key, (label, name) in COLLECTIONS.items()
        },
        "kafka": {
            "bootstrap_servers": _value("VECTOR_MONITOR_KAFKA_BOOTSTRAP_SERVERS", "KAFKA_BOOTSTRAP_SERVERS"),
            "security_protocol": _value("VECTOR_MONITOR_KAFKA_SECURITY_PROTOCOL", "KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
            "sasl_mechanism": _value("VECTOR_MONITOR_KAFKA_SASL_MECHANISM", "KAFKA_SASL_MECHANISM", "PLAIN"),
            "sasl_username": _value("VECTOR_MONITOR_KAFKA_SASL_USERNAME", "KAFKA_SASL_USERNAME"),
            "sasl_password": _value("VECTOR_MONITOR_KAFKA_SASL_PASSWORD", "KAFKA_SASL_PASSWORD"),
            "lanes": lanes,
        },
        "refresh_seconds": refresh_seconds,
        "snapshot_file": os.getenv("VECTOR_MONITOR_SNAPSHOT_FILE", DEFAULT_SNAPSHOT_FILE),
        "retention_days": retention_days,
        "rate_window_days": rate_window_days,
        "token": os.getenv("VECTOR_MONITOR_TOKEN", "").strip(),
    }


def _count_documents(collection: Any, query: dict[str, Any]) -> int:
    try:
        return int(collection.count_documents(query, maxTimeMS=MONGO_QUERY_TIMEOUT_MS))
    except TypeError:
        return int(collection.count_documents(query))


def _aggregate(collection: Any, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        return list(
            collection.aggregate(
                pipeline,
                allowDiskUse=True,
                maxTimeMS=MONGO_QUERY_TIMEOUT_MS,
            )
        )
    except TypeError:
        return list(collection.aggregate(pipeline))


def _status_counts(collection: Any, statuses: tuple[str, ...]) -> dict[str, int]:
    return {status: _count_documents(collection, {"status": status}) for status in statuses}


def _top_breakdown(
    collection: Any,
    *,
    field: str,
    match: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = _aggregate(
        collection,
        [
            {"$match": match},
            {
                "$group": {
                    "_id": f"${field}",
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": max(1, min(limit, 20))},
        ],
    )
    return [
        {"value": str(row.get("_id") or "(空)"), "count": _safe_int(row.get("count"))}
        for row in rows
    ]


def _mongo_collection_summary(
    collection: Any,
    *,
    name: str,
    statuses: tuple[str, ...],
    failure_records: bool,
) -> dict[str, Any]:
    counts = _status_counts(collection, statuses)
    # Status enums can gain new values; total must remain the collection's
    # real size instead of silently dropping an unknown status.
    total = _count_documents(collection, {})
    if failure_records:
        pending_count = counts.get("PENDING", 0) + counts.get("RETRYING", 0)
        handled_count = counts.get("DONE", 0)
        processing_count = counts.get("RETRYING", 0)
        error_match = {}
        error_field = "error_code"
        stage = _top_breakdown(collection, field="failure_stage", match={})
    else:
        pending_count = counts.get("PENDING", 0)
        handled_count = counts.get("SUCCESS", 0)
        processing_count = counts.get("LEASED", 0)
        error_match = {"status": "FAILURE"}
        error_field = "error_code"
        stage = []
    retryable_match = (
        {"error_type": "RETRYABLE"}
        if failure_records
        else {"status": "FAILURE", "retryable": True}
    )
    return {
        "name": name,
        "total": total,
        "statuses": counts,
        "pending_count": pending_count,
        "processing_count": processing_count,
        "handled_count": handled_count,
        "failure_count": counts.get("FAILURE", total if failure_records else 0),
        "retryable_failures": _count_documents(collection, retryable_match),
        "top_stages": stage,
        "top_error_codes": _top_breakdown(collection, field=error_field, match=error_match),
    }


def read_mongo(config: dict[str, Any]) -> dict[str, Any]:
    """Read failure/task counters only; this function deliberately has no writes."""
    result: dict[str, Any] = {
        "connected": False,
        "ready": False,
        "database": config["mongo_database"],
        "collections": {},
    }
    if not config["mongo_uri"]:
        result["error"] = "Mongo URI is not configured"
        return result
    client: Any = None
    try:
        from pymongo import MongoClient

        client = MongoClient(
            config["mongo_uri"],
            serverSelectionTimeoutMS=5_000,
            connectTimeoutMS=5_000,
            socketTimeoutMS=MONGO_QUERY_TIMEOUT_MS,
        )
        client.admin.command("ping")
        db = client[config["mongo_database"]]
        for key, (label, default_name) in MONGO_COLLECTIONS.items():
            name = config["mongo_collections"].get(key, default_name)
            try:
                result["collections"][key] = {
                    "label": label,
                    **_mongo_collection_summary(
                        db[name],
                        name=name,
                        statuses=FAILURE_STATUSES if key == "failure_records" else TASK_STATUSES,
                        failure_records=key == "failure_records",
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - one collection must not hide the others.
                result["collections"][key] = {"label": label, "name": name, "error": str(exc)}
        result["connected"] = True
        result["ready"] = all("error" not in item for item in result["collections"].values())
    except Exception as exc:  # noqa: BLE001 - dependency failure is dashboard data.
        result["error"] = str(exc)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _summarize_schema(schema: Any) -> dict[str, Any]:
    """Keep schema details compact while preserving the old API shape."""
    normalized = _json_safe(schema)
    if isinstance(normalized, dict) and isinstance(normalized.get("schema"), dict):
        normalized = normalized["schema"]
    fields = normalized.get("fields", []) if isinstance(normalized, dict) else []
    if not isinstance(fields, list):
        fields = []
    primary_fields: list[str] = []
    vector_fields: list[dict[str, Any]] = []
    scalar_fields: list[str] = []
    field_names: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "")
        if name:
            field_names.append(name)
        if field.get("is_primary"):
            primary_fields.append(name)
        dtype = str(field.get("type") or field.get("dtype") or field.get("data_type") or "")
        params = field.get("params") if isinstance(field.get("params"), dict) else {}
        element_type = str(field.get("element_type") or "")
        type_text = f"{dtype} {element_type}".upper()
        is_vector = "VECTOR" in type_text or params.get("dim") is not None or field.get("dim") is not None
        if is_vector:
            vector_fields.append(
                {
                    "name": name,
                    "type": dtype or element_type,
                    "dim": params.get("dim") or field.get("dim"),
                }
            )
        elif name:
            scalar_fields.append(name)
    return {
        "field_count": len(fields),
        "field_names": field_names,
        "primary_fields": primary_fields,
        "vector_fields": vector_fields,
        "scalar_field_sample": scalar_fields[:20],
    }


def read_milvus(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "connected": False,
        "uri": config["milvus_uri"],
        "database": config["milvus_database"],
        "attu_hint": {**ATTU_HINT, "database": config["milvus_database"]},
        "collections": {},
        "total_rows": 0,
    }
    client: Any = None
    try:
        from pymilvus import MilvusClient

        options: dict[str, Any] = {
            "uri": config["milvus_uri"],
            "db_name": config["milvus_database"],
            "timeout": 20,
        }
        if config["milvus_token"]:
            options["token"] = config["milvus_token"]
        client = MilvusClient(**options)
        for key, collection in config["collections"].items():
            item = {
                "key": key,
                "label": collection["label"],
                "name": collection["name"],
                "database": config["milvus_database"],
                "row_count": 0,
                "exists": False,
                "schema": {},
            }
            try:
                item["exists"] = bool(client.has_collection(collection["name"]))
                if item["exists"]:
                    stats = client.get_collection_stats(collection_name=collection["name"])
                    item["row_count"] = _safe_int(stats.get("row_count"))
                    item["schema"] = _summarize_schema(client.describe_collection(collection["name"]))
                    result["total_rows"] += item["row_count"]
            except Exception as exc:  # noqa: BLE001 - collection failures are dashboard data.
                item["error"] = str(exc)
            result["collections"][key] = item
        result["connected"] = True
    except Exception as exc:  # noqa: BLE001 - dependency failures are dashboard data.
        result["error"] = str(exc)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing must not change a successful result.
                pass
    return result


def read_kafka(config: dict[str, Any]) -> dict[str, Any]:
    """Read topic metadata, watermarks and committed offsets without consuming."""
    kafka = config["kafka"]
    result: dict[str, Any] = {"connected": False, "lanes": {}, "total_lag": 0, "unknown_lanes": 0}
    if not kafka["bootstrap_servers"]:
        result["error"] = "Kafka bootstrap servers are not configured"
        return result
    try:
        from confluent_kafka import Consumer, TopicPartition

        options = {
            "bootstrap.servers": kafka["bootstrap_servers"],
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "socket.timeout.ms": 10_000,
            "session.timeout.ms": 10_000,
        }
        for option, config_key in (
            ("security.protocol", "security_protocol"),
            ("sasl.mechanism", "sasl_mechanism"),
            ("sasl.username", "sasl_username"),
            ("sasl.password", "sasl_password"),
        ):
            if kafka[config_key]:
                options[option] = kafka[config_key]

        for key, lane_config in kafka["lanes"].items():
            topic_name = lane_config["topic"]
            item: dict[str, Any] = {
                "topic": topic_name,
                "group_id": lane_config["group_id"],
                "partition_count": 0,
                "tail": 0,
                "committed": None,
                "lag": None,
                "committed_known": False,
            }
            client = Consumer({**options, "group.id": lane_config["group_id"]})
            try:
                metadata = client.list_topics(topic=topic_name, timeout=10)
                topic = metadata.topics.get(topic_name)
                if topic is None or topic.error is not None:
                    raise RuntimeError(str(topic.error if topic is not None else "topic not found"))
                partitions = [TopicPartition(topic_name, number) for number in sorted(topic.partitions)]
                committed = client.committed(partitions, timeout=10) or []
                committed_by_partition = {
                    int(partition.partition): int(partition.offset)
                    for partition in committed
                    if getattr(partition, "offset", -1) >= 0
                }
                lows_highs = [client.get_watermark_offsets(partition, timeout=10, cached=False) for partition in partitions]
                item["partition_count"] = len(partitions)
                item["tail"] = sum(int(high) for _, high in lows_highs)
                item["committed_known"] = len(committed_by_partition) == len(partitions)
                if item["committed_known"]:
                    item["committed"] = sum(committed_by_partition[partition.partition] for partition in partitions)
                    item["lag"] = max(item["tail"] - item["committed"], 0)
                    result["total_lag"] += item["lag"]
                else:
                    result["unknown_lanes"] += 1
            except Exception as exc:  # noqa: BLE001 - lane failure must not hide the other two lanes.
                item["error"] = str(exc)
            finally:
                client.close()
            result["lanes"][key] = item
        result["connected"] = bool(result["lanes"]) and all("error" not in lane for lane in result["lanes"].values())
    except Exception as exc:  # noqa: BLE001 - dependency failure is dashboard data.
        result["error"] = str(exc)
    return result


def _snapshot_path(config: dict[str, Any]) -> Path:
    return Path(config["snapshot_file"]).expanduser()


def _read_snapshots(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        lines = _snapshot_path(config).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    snapshots = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("ts"):
                snapshots.append(value)
        except (TypeError, ValueError):
            pass
    return snapshots[-20_000:]


def _append_snapshot(config: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Keep one JSONL snapshot per refresh and trim expired history atomically."""
    path = _snapshot_path(config)
    cutoff = time() - config["retention_days"] * 86400
    kept = []
    for old in _read_snapshots(config):
        try:
            if datetime.fromisoformat(str(old["ts"]).replace("Z", "+00:00")).timestamp() >= cutoff:
                kept.append(old)
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            for item in [*kept, snapshot]:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            temporary_path = handle.name
        os.replace(temporary_path, path)
    except OSError as exc:
        print(f"snapshot write failed: {exc}")


def _snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": summary["ts"],
        "rows": {key: item["row_count"] for key, item in summary["milvus"]["collections"].items()},
        "offsets": {
            key: {"tail": lane.get("tail"), "committed": lane.get("committed"), "lag": lane.get("lag")}
            for key, lane in summary["kafka"]["lanes"].items()
        },
    }


def _daily_rate(key: str, snapshots: list[dict[str, Any]], horizon_seconds: int) -> dict[str, Any]:
    values = []
    for sample in snapshots:
        try:
            timestamp = datetime.fromisoformat(str(sample["ts"]).replace("Z", "+00:00")).timestamp()
            committed = sample.get("offsets", {}).get(key, {}).get("committed")
            if committed is not None:
                values.append((timestamp, int(committed)))
        except (TypeError, ValueError, OverflowError):
            pass
    if not values:
        return {"status": "no_data", "messages_per_day": None, "available_seconds": 0}
    values.sort()
    current_time, current_offset = values[-1]
    candidates = [sample for sample in values[:-1] if sample[0] <= current_time - horizon_seconds]
    if not candidates:
        elapsed = current_time - values[0][0]
        if len(values) < 2 or elapsed < MIN_ROUGH_SAMPLE_SECONDS:
            return {
                "status": "warming_up",
                "messages_per_day": None,
                "available_seconds": int(max(elapsed, 0)),
            }
        start_time, start_offset = values[0]
        delta = current_offset - start_offset
        if delta < 0:
            return {"status": "counter_reset", "messages_per_day": None, "elapsed_seconds": int(elapsed)}
        return {
            "status": "rough",
            "messages_per_day": round(delta * 86400 / elapsed, 1),
            "elapsed_seconds": int(elapsed),
            "delta_messages": delta,
        }
    start_time, start_offset = candidates[-1]
    elapsed = current_time - start_time
    delta = current_offset - start_offset
    if elapsed <= 0:
        return {"status": "no_data", "messages_per_day": None, "available_seconds": 0}
    if delta < 0:
        return {"status": "counter_reset", "messages_per_day": None, "elapsed_seconds": int(elapsed)}
    return {
        "status": "ready",
        "messages_per_day": round(delta * 86400 / elapsed, 1),
        "elapsed_seconds": int(elapsed),
        "delta_messages": delta,
    }


def _estimate_days(lane: dict[str, Any], rate: dict[str, Any]) -> dict[str, Any]:
    if lane.get("lag") is None:
        return {"status": "offset_unknown", "days": None}
    if lane["lag"] == 0:
        return {"status": "complete", "days": 0.0}
    per_day = rate.get("messages_per_day") or 0
    if rate.get("status") not in {"ready", "rough"} or per_day <= 0:
        return {"status": "warming_up", "days": None}
    return {"status": rate["status"], "days": round(lane["lag"] / per_day, 1)}


def _add_progress(summary: dict[str, Any], config: dict[str, Any]) -> None:
    snapshots = _read_snapshots(config)
    horizon = config["rate_window_days"] * 86400
    summary["history"] = {
        "sample_count": len(snapshots),
        "retention_days": config["retention_days"],
        "rate_window_days": config["rate_window_days"],
    }
    summary["lanes"] = {}
    for key, collection in config["collections"].items():
        milvus = summary["milvus"]["collections"].get(key, {})
        kafka = summary["kafka"]["lanes"].get(key, {})
        daily_rate = _daily_rate(key, snapshots, horizon)
        summary["lanes"][key] = {
            "label": collection["label"],
            "collection": collection["name"],
            "row_count": milvus.get("row_count", 0),
            "collection_exists": milvus.get("exists", False),
            "collection_error": milvus.get("error"),
            "topic": kafka.get("topic"),
            "lag": kafka.get("lag"),
            "committed_known": kafka.get("committed_known", False),
            "kafka_error": kafka.get("error"),
            "daily_rate": daily_rate,
            "eta": _estimate_days(kafka, daily_rate),
        }


def build_summary(force: bool = False) -> dict[str, Any]:
    global _LAST_SUMMARY, _LAST_SUMMARY_AT
    config = get_config()
    with _SUMMARY_LOCK:
        if not force and _LAST_SUMMARY and monotonic() - _LAST_SUMMARY_AT < config["refresh_seconds"]:
            return _LAST_SUMMARY
        summary: dict[str, Any] = {
            "ts": utc_now(),
            "config": {
                "refresh_seconds": config["refresh_seconds"],
                "rate_window_days": config["rate_window_days"],
                "collections": {key: item["name"] for key, item in config["collections"].items()},
                "kafka_lanes": config["kafka"]["lanes"],
            },
            "milvus": read_milvus(config),
            "kafka": read_kafka(config),
            "mongo": read_mongo(config),
        }
        if summary["milvus"]["connected"] and summary["kafka"]["connected"]:
            _append_snapshot(config, _snapshot(summary))
        _add_progress(summary, config)
        _LAST_SUMMARY = summary
        _LAST_SUMMARY_AT = monotonic()
        return summary


def build_health() -> tuple[int, dict[str, Any]]:
    summary = build_summary()
    age = max(time() - datetime.fromisoformat(summary["ts"]).timestamp(), 0)
    dependencies = {
        "milvus": bool(summary["milvus"]["connected"]),
        "kafka": bool(summary["kafka"]["connected"]),
        "mongo": bool(summary.get("mongo", {}).get("ready")),
    }
    healthy = all(dependencies.values()) and age <= summary["config"]["refresh_seconds"] * 3
    return (200 if healthy else 503), {
        "status": "ok" if healthy else "degraded",
        "ts": summary["ts"],
        "age_seconds": round(age, 1),
        "dependencies": dependencies,
        "snapshot_samples": summary["history"]["sample_count"],
    }


try:
    INDEX_HTML = DASHBOARD_FILE.read_text(encoding="utf-8")
except OSError:
    INDEX_HTML = "<h1>Dashboard file is missing</h1>"


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "VectorSyncMonitor/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/health":
            status, payload = build_health()
            self._send(json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8", status)
        elif path == "/api/summary":
            config = get_config()
            if config["token"] and self.headers.get("Authorization") != f"Bearer {config['token']}":
                self._send('{"detail":"Bearer token required"}', "application/json; charset=utf-8", 401)
            else:
                self._send(json.dumps(build_summary(), ensure_ascii=False), "application/json; charset=utf-8")
        else:
            self.send_error(404, "Not found")

    def _send(self, body: str, content_type: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="三路向量同步只读监控")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(json.dumps(build_summary(force=True), ensure_ascii=False, indent=2))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"本地监控页已启动: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止本地监控页...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
