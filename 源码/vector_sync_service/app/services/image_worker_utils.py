"""Small shared helpers for image worker persistence."""

from __future__ import annotations

from typing import Any


def sanitize_for_mongo(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_mongo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_mongo(item) for item in obj]
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return [sanitize_for_mongo(item) for item in obj]
        except Exception:
            return str(obj)
    if hasattr(obj, "primary_keys"):
        primary_keys = getattr(obj, "primary_keys", None)
        try:
            primary_keys = list(primary_keys) if primary_keys is not None else None
        except Exception:
            primary_keys = str(primary_keys)
        return {
            "insert_count": getattr(obj, "insert_count", None),
            "delete_count": getattr(obj, "delete_count", None),
            "primary_keys": primary_keys,
            "timestamp": getattr(obj, "timestamp", None),
        }
    try:
        return str(obj)
    except Exception:
        return repr(obj)
