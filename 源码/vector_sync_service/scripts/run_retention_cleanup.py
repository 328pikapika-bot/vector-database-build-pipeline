#!/usr/bin/env python3
"""Run Mongo/local retention cleanup once."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.services.retention_cleanup import RetentionCleanupService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retention cleanup once.")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(
        RetentionCleanupService().run(
            mode=args.mode,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
