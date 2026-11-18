"""Shared field-only backfill integrity primitives; no SSH dependency."""
import hashlib
import json
import zlib
import numpy as np
from datetime import datetime, timedelta, timezone
TZ = timezone(timedelta(hours=8))

class IntegrityError(RuntimeError):
    pass


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(event, **kwargs):
    print(json.dumps({"at": now(), "event": event, **kwargs}, ensure_ascii=False), flush=True)


def encode(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), 1)


def decode(value):
    return json.loads(zlib.decompress(value))


def original_hash(row, fields):
    # All original scalar fields plus exact FLOAT_VECTOR float32 bytes.
    scalars = {k: row[k] for k in fields if k != "text_embed"}
    h = hashlib.sha256(json.dumps(scalars, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode())
    h.update(np.asarray(row["text_embed"], dtype="<f4").tobytes())
    return h.hexdigest()


def valid_source(value):
    return (isinstance(value, list) and 0 < len(value) <= 4096
            and all(isinstance(v, str) and v.strip() and len(v.encode()) <= 65535 for v in value))

