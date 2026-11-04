"""Snapshot retained Zijian Kafka IDs for metadata v3 without committing offsets.

This scans Kafka into the local SQLite manifest only. The metadata-v3 worker
materializes the IDs and writes matching Milvus records after all three journal
years are complete.
"""

from __future__ import annotations

import threading

from backfill_metadata_v3 import select_topic


if __name__ == "__main__":
    select_topic(threading.Event(), materialize=False)
