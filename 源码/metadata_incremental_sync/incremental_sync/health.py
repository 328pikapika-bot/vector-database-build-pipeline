"""Container health: process heartbeat must be recent; state is explicit."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    path = Path(os.getenv("SYNC_STATE_DIR", "/app/state")) / "progress.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(data["at"])).total_seconds()
        if age > 45 or data.get("status") in {"safety_pause", "storage_safety_pause"}:
            return 1
        return 0
    except (OSError, KeyError, ValueError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
