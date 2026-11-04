#!/usr/bin/env python3
"""Run Mongo operational backup once."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.services.mongo_ops_backup import run_backup
from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager


async def _main() -> int:
    manager = get_mongo_state_manager()
    await manager.connect()
    result = await run_backup(manager)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    await close_mongo_state_manager()
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
