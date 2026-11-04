"""Kafka 消息辅助函数。"""
from __future__ import annotations

from typing import Any, Tuple

from app.models.sync_runtime import normalize_record_update_time


def _dp_first_value(raw: Any) -> Any:
    if isinstance(raw, dict) and "value" in raw:
        values = raw.get("value") or []
        return values[0] if values else None
    return raw


def extract_item_value(item: dict[str, Any], field_name: str) -> Any:
    dp = item.get("dp") if isinstance(item.get("dp"), dict) else {}
    if field_name in dp:
        return _dp_first_value(dp[field_name])
    return item.get(field_name)


def extract_message_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """提取日志与幂等所需的关键元信息。"""
    data = payload.get("data") or []
    first = data[0] if data and isinstance(data[0], dict) else {}

    sys_id = first.get("id") or first.get("sysId") or extract_item_value(first, "sysId")
    sys_update_time = extract_item_value(first, "sysDataUpdateTime") or first.get("sysDataUpdateTime")
    sys_sub_type = extract_item_value(first, "sysSubType") or first.get("sysSubType") or payload.get("subDocType")

    return {
        "subDocType": payload.get("subDocType"),
        "dataType": payload.get("dataType"),
        "articleCount": len(data),
        "firstDocId": sys_id,
        "sysId": sys_id,
        "sysDataUpdateTime": normalize_record_update_time(sys_update_time),
        "sysSubType": sys_sub_type,
    }


def build_unique_key(payload: dict[str, Any]) -> Tuple[str, str, str]:
    meta = extract_message_meta(payload)
    return (
        str(meta.get("sysId") or ""),
        str(meta.get("sysDataUpdateTime") or ""),
        str(meta.get("sysSubType") or ""),
    )


def infer_data_status(payload: dict[str, Any]) -> str:
    for key in ("dataStatus", "opType", "operation", "type"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            upper = value.strip().upper()
            if upper in {"ADD", "UPDATE", "DELETE"}:
                return upper
    return "ADD"
