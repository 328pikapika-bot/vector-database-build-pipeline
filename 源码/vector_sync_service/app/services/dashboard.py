"""Embedded read-only dashboard for TXT vector sync status."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import settings


TXT_MILVUS_DATABASE = "default"
ATTU_HINT = {
    "txt_attu": "http://127.0.0.1:18002",
    "txt_external_milvus": "http://127.0.0.1:19533",
    "txt_internal_milvus": "scientific-milvus-standalone:19530",
    "database": TXT_MILVUS_DATABASE,
    "note": (
        "Scientific ATTU must connect to default/scientific_documents, "
        "scientific_fulltext_chunks, and scientific_images on the scientific "
        "Milvus endpoint. Do not compare with the legacy TXT test Milvus."
    ),
}
COLLECTIONS = {
    "documents": {
        "name_attr": "MILVUS_COLLECTION",
        "default": "scientific_documents",
        "label": "Documents",
        "description": "Document-level TXT vectors",
    },
    "fulltext_chunks": {
        "name_attr": "FULLTEXT_CHUNK_COLLECTION",
        "default": "scientific_fulltext_chunks",
        "label": "Fulltext Chunks",
        "description": "Physical rows in the fulltext chunk collection",
    },
    "images": {
        "name_attr": "IMAGE_COLLECTION_NAME",
        "default": "scientific_images",
        "label": "Images",
        "description": "Image vector rows",
    },
}


def render_dashboard_html() -> str:
    return _HTML_PAGE


async def build_stats_summary(
    *,
    app_state,
    mongo_manager,
    auto_publisher: Optional[object] = None,
) -> dict[str, Any]:
    """Build dashboard JSON.

    Main collection counts are physical Milvus row counts. MongoDB numbers are
    retained only as queue/status diagnostics.
    """

    mongo_stats = _empty_mongo_stats()
    if mongo_manager is not None and mongo_manager.connected:
        try:
            mongo_stats = await asyncio.to_thread(_read_mongo_stats_sync, mongo_manager)
        except Exception as exc:  # noqa: BLE001 - monitoring must surface dependency errors.
            mongo_stats["error"] = str(exc)

    milvus_stats = await asyncio.to_thread(_read_milvus_stats_sync)

    runtime = {
        "consumer_running": app_state.consumer_running if app_state else False,
        "kafka_connected": app_state.kafka_connected if app_state else False,
        "compensator_running": app_state.compensator_running if app_state else False,
        "milvus_connected": bool(milvus_stats.get("connected")),
        "embedding_available": app_state.embedding_available if app_state else False,
    }

    publisher = auto_publisher.status() if auto_publisher is not None else {}

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "publisher": publisher,
        "mongo": mongo_stats,
        "milvus": milvus_stats,
        "config": {
            "milvus_uri": _milvus_uri(),
            "milvus_database": TXT_MILVUS_DATABASE,
            "collections": {
                key: _collection_name(info)
                for key, info in COLLECTIONS.items()
            },
            "attu_hint": ATTU_HINT,
        },
    }


def _empty_mongo_stats() -> dict[str, Any]:
    return {
        "connected": False,
        "database": settings.MONGO_DB,
        "status_counts": {},
        "fulltext_tasks": {},
        "image_tasks": {},
        "failure_counts": {},
        "total_failures": 0,
        "actionable_failures": 0,
        "held_history_failures": 0,
        "recent_failures": [],
    }


def _read_mongo_stats_sync(mongo_manager) -> dict[str, Any]:
    db = mongo_manager.db
    stats = _empty_mongo_stats()
    stats["connected"] = True

    stats["status_counts"] = _group_counts(
        db[settings.MONGO_STATUS_COLLECTION],
        "overall_status",
    )
    stats["fulltext_tasks"] = _group_counts(
        db[settings.MONGO_FULLTEXT_TASK_COLLECTION],
        "status",
    )
    stats["image_tasks"] = _group_counts(
        db[settings.MONGO_IMAGE_TASK_COLLECTION],
        "status",
    )
    stats["failure_counts"] = _group_counts(
        db[settings.MONGO_FAILURE_COLLECTION],
        "status",
    )
    stats["total_failures"] = int(
        db[settings.MONGO_FAILURE_COLLECTION].count_documents({})
    )
    stats["actionable_failures"] = int(
        db[settings.MONGO_FAILURE_COLLECTION].count_documents(
            {"status": {"$in": ["PENDING", "RETRYING"]}}
        )
    )
    stats["held_history_failures"] = int(
        db[settings.MONGO_FAILURE_COLLECTION].count_documents(
            {"status": "HELD_HISTORY"}
        )
    )

    failure_cursor = (
        db[settings.MONGO_FAILURE_COLLECTION]
        .find(
            {},
            {
                "doc_id": 1,
                "status": 1,
                "msg": 1,
                "error_type": 1,
                "error_code": 1,
                "error_message": 1,
                "updated_at": 1,
                "_id": 0,
            },
        )
        .sort("updated_at", -1)
        .limit(20)
    )
    stats["recent_failures"] = [_json_safe(item) for item in failure_cursor]
    return stats


def _group_counts(collection, field: str) -> dict[str, int]:
    rows = collection.aggregate(
        [
            {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
    )
    return {
        str(row.get("_id") or "UNKNOWN"): _safe_int(row.get("count"))
        for row in rows
    }


def _read_milvus_stats_sync() -> dict[str, Any]:
    result: dict[str, Any] = {
        "connected": False,
        "uri": _milvus_uri(),
        "database": TXT_MILVUS_DATABASE,
        "attu_hint": ATTU_HINT,
        "collections": {},
        "total_rows": 0,
    }
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=result["uri"], db_name=TXT_MILVUS_DATABASE, timeout=20)
        try:
            for key, info in COLLECTIONS.items():
                name = _collection_name(info)
                item = {
                    "key": key,
                    "label": info["label"],
                    "description": info["description"],
                    "name": name,
                    "database": TXT_MILVUS_DATABASE,
                    "exists": False,
                    "row_count": 0,
                    "schema": {},
                }
                try:
                    exists = bool(client.has_collection(name))
                    item["exists"] = exists
                    if exists:
                        stats = client.get_collection_stats(collection_name=name)
                        item["row_count"] = _safe_int(stats.get("row_count"))
                        item["schema"] = _summarize_schema(client.describe_collection(name))
                        result["total_rows"] += int(item["row_count"])
                except Exception as coll_exc:  # noqa: BLE001
                    item["error"] = str(coll_exc)
                result["collections"][key] = item
            result["connected"] = True
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def _milvus_uri() -> str:
    return f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"


def _collection_name(info: dict[str, Any]) -> str:
    return str(getattr(settings, str(info["name_attr"]), info["default"]) or info["default"])


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _summarize_schema(schema: Any) -> dict[str, Any]:
    schema = _json_safe(schema)
    if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
        schema = schema["schema"]
    fields = schema.get("fields") if isinstance(schema, dict) else []
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
        is_vector = (
            "VECTOR" in type_text
            or params.get("dim") is not None
            or field.get("dim") is not None
        )
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


_HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NSTL TXT Vector Sync</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f5f7fb;
    --panel: #ffffff;
    --panel-soft: #f9fafb;
    --text: #172033;
    --muted: #637083;
    --border: #dbe2ea;
    --blue: #1f6feb;
    --green: #16794c;
    --amber: #9a6700;
    --red: #cf222e;
    --ink: #253043;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
  }
  .shell { max-width: 1280px; margin: 0 auto; padding: 22px; }
  header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 18px;
  }
  h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 0; }
  .sub { margin-top: 6px; color: var(--muted); font-size: 13px; }
  .status-line { color: var(--muted); font-size: 13px; text-align: right; }
  .grid { display: grid; gap: 12px; }
  .grid.cards { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .grid.two { grid-template-columns: 1.05fr 0.95fr; margin-top: 12px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  }
  .card-title { color: var(--muted); font-size: 13px; font-weight: 600; margin-bottom: 8px; }
  .num { font-size: 30px; line-height: 1.15; font-weight: 760; color: var(--ink); }
  .small { color: var(--muted); font-size: 12px; margin-top: 6px; min-height: 16px; }
  .ok { color: var(--green); } .warn { color: var(--amber); } .bad { color: var(--red); } .info { color: var(--blue); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { border-bottom: 1px solid var(--border); padding: 9px 8px; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 700; background: var(--panel-soft); }
  tbody tr:last-child td { border-bottom: 0; }
  .kv { display: grid; grid-template-columns: 150px 1fr; gap: 8px 12px; font-size: 13px; }
  .kv div:nth-child(odd) { color: var(--muted); }
  .badge {
    display: inline-flex;
    align-items: center;
    min-height: 22px;
    padding: 2px 8px;
    border-radius: 999px;
    background: #eef2f7;
    color: var(--ink);
    font-size: 12px;
    font-weight: 700;
    margin: 0 6px 6px 0;
  }
  .badge.ok { background: #dafbe1; color: var(--green); }
  .badge.warn { background: #fff8c5; color: var(--amber); }
  .badge.bad { background: #ffebe9; color: var(--red); }
  .badge.info { background: #ddf4ff; color: var(--blue); }
  .error-box {
    background: #ffebe9;
    color: var(--red);
    border: 1px solid #ffccd2;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    margin-bottom: 12px;
    display: none;
    white-space: pre-wrap;
  }
  .mono {
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 12px;
    word-break: break-all;
  }
  .truncate {
    max-width: 460px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  @media (max-width: 920px) {
    .grid.cards, .grid.two { grid-template-columns: 1fr; }
    header { align-items: flex-start; flex-direction: column; }
    .status-line { text-align: left; }
  }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div>
      <h1>NSTL TXT Vector Sync</h1>
      <div class="sub">Main cards show TXT Milvus physical row counts. MongoDB numbers are queue/status diagnostics only.</div>
    </div>
    <div class="status-line">
      <div>Updated: <span id="updated">--</span></div>
      <div>Status: <span id="conn">--</span></div>
    </div>
  </header>

  <div id="error" class="error-box"></div>

  <section class="grid cards">
    <div class="card">
      <div class="card-title">Documents Rows</div>
      <div class="num" id="docRows">--</div>
      <div class="small mono" id="docDelta">scientific_documents</div>
    </div>
    <div class="card">
      <div class="card-title">Fulltext Chunk Rows</div>
      <div class="num" id="chunkRows">--</div>
      <div class="small mono" id="chunkDelta">scientific_fulltext_chunks</div>
    </div>
    <div class="card">
      <div class="card-title">Image Rows</div>
      <div class="num" id="imageRows">--</div>
      <div class="small mono" id="imageDelta">scientific_images</div>
    </div>
    <div class="card">
      <div class="card-title">Milvus Total Rows</div>
      <div class="num" id="totalRows">--</div>
      <div class="small">Sum of the three TXT collections from Milvus row_count.</div>
    </div>
  </section>

  <section class="grid two">
    <div class="card">
      <div class="card-title">Milvus TXT Connection</div>
      <div id="milvusConfig" class="kv"></div>
    </div>
    <div class="card">
      <div class="card-title">ATTU Check</div>
      <div id="attuHint" class="kv"></div>
    </div>
  </section>

  <section class="grid two">
    <div class="card">
      <div class="card-title">Milvus Collection Probe</div>
      <table>
        <thead>
          <tr><th>Collection</th><th>Purpose</th><th>Rows</th><th>Schema Key Fields</th><th>Status</th></tr>
        </thead>
        <tbody id="milvusTable"></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Runtime</div>
      <div id="runtimeInfo"></div>
    </div>
  </section>

  <section class="grid two">
    <div class="card">
      <div class="card-title">Mongo Document Status</div>
      <div id="mongoStatus"></div>
    </div>
    <div class="card">
      <div class="card-title">Failure Compensation</div>
      <div id="failureTasks"></div>
      <div class="small" id="failureSummary"></div>
    </div>
  </section>

  <section class="grid two">
    <div class="card">
      <div class="card-title">Fulltext Queue</div>
      <div id="fulltextTasks"></div>
    </div>
    <div class="card">
      <div class="card-title">Image Queue</div>
      <div id="imageTasks"></div>
    </div>
  </section>

  <section class="card" style="margin-top:12px">
    <div class="card-title">Recent Failure Records</div>
    <table>
      <thead>
        <tr><th>doc_id</th><th>status</th><th>error_type</th><th>error_code</th><th>message</th></tr>
      </thead>
      <tbody id="failTbody"></tbody>
    </table>
  </section>
</div>

<script>
let previous = null;
let currentData = null;

const order = [
  "SUCCESS", "CORE_SUCCESS", "PROCESSING", "PENDING", "LEASED",
  "RETRYING", "FAILURE", "HELD_HISTORY", "SKIPPED", "DONE",
  "GAVE_UP", "PARTIAL_FAILURE", "DELETED", "UNKNOWN"
];

function fmt(n) {
  return Number(n || 0).toLocaleString("zh-CN");
}

function statusClass(status) {
  if (status === "SUCCESS" || status === "CORE_SUCCESS" || status === "DONE") return "ok";
  if (status === "FAILURE" || status === "GAVE_UP" || status === "PARTIAL_FAILURE") return "bad";
  if (status === "HELD_HISTORY") return "info";
  if (status === "PENDING" || status === "PROCESSING" || status === "LEASED" || status === "RETRYING") return "warn";
  return "info";
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function rowCount(data, key) {
  return data?.milvus?.collections?.[key]?.row_count || 0;
}

function renderDelta(elementId, key, current) {
  const el = document.getElementById(elementId);
  const collection = currentData?.milvus?.collections?.[key]?.name || key;
  if (!previous) {
    el.textContent = collection;
    return;
  }
  const prev = rowCount(previous, key);
  const diff = current - prev;
  const currentTs = Date.parse(currentData.ts);
  const prevTs = Date.parse(previous.ts);
  const seconds = Math.max((currentTs - prevTs) / 1000, 1);
  const perMin = diff / seconds * 60;
  el.innerHTML = `${escapeHtml(collection)} ` + (diff >= 0
    ? `<span class="ok">+${fmt(diff)}</span>, approx ${fmt(Math.round(perMin))}/min`
    : `<span class="bad">${fmt(diff)}</span>`);
}

function renderKv(id, rows) {
  document.getElementById(id).innerHTML = rows.map(([key, value]) =>
    `<div>${escapeHtml(key)}</div><div class="mono">${escapeHtml(value || "--")}</div>`
  ).join("");
}

async function refresh() {
  const errorBox = document.getElementById("error");
  try {
    const resp = await fetch("/api/stats/summary", { cache: "no-store" });
    currentData = await resp.json();
    if (!resp.ok) throw new Error(JSON.stringify(currentData));

    document.getElementById("updated").textContent = new Date(currentData.ts).toLocaleString("zh-CN");
    const mongoOk = !!currentData.mongo?.connected;
    const milvusOk = !!currentData.milvus?.connected;
    document.getElementById("conn").innerHTML =
      `<span class="${mongoOk ? "ok" : "bad"}">Mongo ${mongoOk ? "OK" : "ERR"}</span> / ` +
      `<span class="${milvusOk ? "ok" : "bad"}">Milvus ${milvusOk ? "OK" : "ERR"}</span>`;

    errorBox.style.display = "none";
    errorBox.textContent = "";
    if (currentData.mongo?.error || currentData.milvus?.error) {
      errorBox.style.display = "block";
      errorBox.textContent = [
        currentData.mongo?.error ? `Mongo error: ${currentData.mongo.error}` : "",
        currentData.milvus?.error ? `Milvus error: ${currentData.milvus.error}` : ""
      ].filter(Boolean).join("\\n");
    }

    const docs = rowCount(currentData, "documents");
    const chunks = rowCount(currentData, "fulltext_chunks");
    const images = rowCount(currentData, "images");
    document.getElementById("docRows").textContent = fmt(docs);
    document.getElementById("chunkRows").textContent = fmt(chunks);
    document.getElementById("imageRows").textContent = fmt(images);
    document.getElementById("totalRows").textContent = fmt(currentData.milvus?.total_rows || docs + chunks + images);
    renderDelta("docDelta", "documents", docs);
    renderDelta("chunkDelta", "fulltext_chunks", chunks);
    renderDelta("imageDelta", "images", images);

    renderMilvusConfig(currentData);
    renderMilvusTable(currentData.milvus?.collections || {});
    renderRuntime(currentData);
    renderCounts("mongoStatus", currentData.mongo?.status_counts || {});
    renderCounts("fulltextTasks", currentData.mongo?.fulltext_tasks || {});
    renderCounts("imageTasks", currentData.mongo?.image_tasks || {});
    renderCounts("failureTasks", currentData.mongo?.failure_counts || {});
    document.getElementById("failureSummary").textContent =
      `Actionable: ${fmt(currentData.mongo?.actionable_failures)} / Held history: ${fmt(currentData.mongo?.held_history_failures)} / Total: ${fmt(currentData.mongo?.total_failures)}`;
    renderFailures(currentData.mongo?.recent_failures || []);

    previous = currentData;
  } catch (err) {
    errorBox.style.display = "block";
    errorBox.textContent = `Refresh failed: ${err.message || err}`;
  }
}

function renderMilvusConfig(data) {
  const milvus = data.milvus || {};
  const hint = milvus.attu_hint || {};
  renderKv("milvusConfig", [
    ["URI", milvus.uri],
    ["Database", milvus.database],
    ["Documents", data.config?.collections?.documents],
    ["Fulltext chunks", data.config?.collections?.fulltext_chunks],
    ["Images", data.config?.collections?.images],
  ]);
  renderKv("attuHint", [
    ["TXT ATTU", hint.txt_attu],
    ["TXT external", hint.txt_external_milvus],
    ["TXT internal", hint.txt_internal_milvus],
    ["Database", hint.database],
    ["Hint", hint.note],
  ]);
}

function renderMilvusTable(collections) {
  const tbody = document.getElementById("milvusTable");
  const rows = ["documents", "fulltext_chunks", "images"].map((key) => {
    const item = collections[key] || {};
    const schema = item.schema || {};
    const vectors = (schema.vector_fields || []).map((field) =>
      `${field.name}:${field.type}${field.dim ? "(" + field.dim + ")" : ""}`
    ).join(", ");
    const primary = (schema.primary_fields || []).join(", ") || "--";
    const state = item.exists
      ? `<span class="badge ok">exists</span>`
      : `<span class="badge bad">missing</span>`;
    return `<tr>
      <td><div>${escapeHtml(item.label || key)}</div><div class="small mono">${escapeHtml(item.database || "")}/${escapeHtml(item.name || "")}</div></td>
      <td>${escapeHtml(item.description || "")}</td>
      <td><strong>${fmt(item.row_count)}</strong></td>
      <td>
        <div>fields: ${fmt(schema.field_count || 0)}</div>
        <div class="small mono">primary: ${escapeHtml(primary)}</div>
        <div class="small mono">vector: ${escapeHtml(vectors || "--")}</div>
      </td>
      <td>${state}${item.error ? `<div class="small bad">${escapeHtml(item.error)}</div>` : ""}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join("");
}

function renderRuntime(data) {
  const rt = data.runtime || {};
  const pub = data.publisher || {};
  renderKv("runtimeInfo", [
    ["Kafka consumer", rt.consumer_running ? "running" : "stopped"],
    ["Kafka", rt.kafka_connected ? "connected" : "disconnected"],
    ["Compensator", rt.compensator_running ? "running" : "stopped"],
    ["Embedding", rt.embedding_available ? "available" : "unavailable"],
    ["Publisher", pub.running ? "running" : (pub.finished ? "finished" : "idle")],
    ["Publisher page", pub.current_page == null ? "--" : String(pub.current_page)],
  ]);
}

function renderCounts(id, counts) {
  const keys = Array.from(new Set([...order, ...Object.keys(counts)])).filter((key) => counts[key]);
  const html = keys.length
    ? keys.map((key) => `<span class="badge ${statusClass(key)}">${key}: ${fmt(counts[key])}</span>`).join("")
    : `<span class="small">No data</span>`;
  document.getElementById(id).innerHTML = html;
}

function renderFailures(failures) {
  const tbody = document.getElementById("failTbody");
  if (!failures.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="ok">No recent failures</td></tr>`;
    return;
  }
  tbody.innerHTML = failures.map((item) => `<tr>
    <td class="mono">${escapeHtml(item.doc_id || "--")}</td>
    <td><span class="badge ${statusClass(item.status)}">${escapeHtml(item.status || "--")}</span></td>
    <td>${escapeHtml(item.error_type || "--")}</td>
    <td>${escapeHtml(item.error_code || "--")}</td>
    <td><div class="truncate" title="${escapeHtml(item.error_message || item.msg || "")}">${escapeHtml(item.error_message || item.msg || "--")}</div></td>
  </tr>`).join("");
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""
