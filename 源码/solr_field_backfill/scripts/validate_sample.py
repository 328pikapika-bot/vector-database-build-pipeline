"""Read-only Solr validation for 100/1000 IDs from the target Milvus collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from the repository root without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from solr_field_backfill.backfill import BackfillWorker, Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(100, 1000), default=100)
    args = parser.parse_args()
    settings = Settings(mode="check", sample_size=0)
    worker = BackfillWorker(settings)
    try:
        ids = worker._query_ids("00")[: args.count]
        rows, missing = worker._solr_batch(ids)
        report = {"requested": len(ids), "solr_rows": len(rows), "missing": len(missing),
                  "missing_ids": missing[:20], "read_only": True}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
