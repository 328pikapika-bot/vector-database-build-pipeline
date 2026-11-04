"""Idempotent delivery from the durable SQLite outbox to a separate Mongo namespace."""
from __future__ import annotations

import logging
import time
from typing import Any

from .config import Settings
from .store import Ledger

LOG = logging.getLogger(__name__)


class FailureReporter:
    def __init__(self, settings: Settings, *, collection: Any = None) -> None:
        if settings.mongo_failure_collection != "metadata_incremental_sync_failures":
            raise ValueError("incremental failures require their own metadata_incremental_sync_failures namespace")
        self.settings = settings
        self.client = None
        self.collection = collection
        self.last_sync = 0.0
        self.status = "not_checked"

    def _connect(self) -> None:
        if self.collection is None:
            from pymongo import MongoClient
            if not self.settings.mongo_uri:
                raise ValueError("MONGO_URI is required")
            self.client = MongoClient(self.settings.mongo_uri, serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000, socketTimeoutMS=3000, retryWrites=False)
            self.collection = self.client[self.settings.mongo_db][self.settings.mongo_failure_collection]

    def check(self) -> str:
        try:
            self._connect()
            self.client.admin.command("ping")
            self.collection.find_one({}, {"_id": 1})
            self.status = "connected"
        except Exception as error:
            self.status = "unavailable:" + type(error).__name__
        return self.status

    def sync(self, ledger: Ledger, *, force: bool = False) -> int:
        if not force and time.monotonic() - self.last_sync < 15:
            return 0
        self.last_sync = time.monotonic()
        documents = ledger.unsent()
        if not documents:
            return 0
        try:
            self._connect()
            from pymongo import UpdateOne
            # Preserve a human's resolved status on redelivery after a crash.
            writes = [UpdateOne({"_id": d["_id"]}, {"$setOnInsert": {
                k: v for k, v in d.items() if k != "_id"}}, upsert=True) for d in documents]
            self.collection.bulk_write(writes, ordered=False)
        except Exception as error:
            self.status = "unavailable:" + type(error).__name__
            LOG.warning("mongo_sync_deferred error_type=%s local_outbox_retained=true", type(error).__name__)
            return 0
        # SQLite errors must propagate and pause writes; never acknowledge unsaved state.
        ledger.mark_sent([d["_id"] for d in documents])
        self.status = "connected"
        LOG.info("mongo_failures_synced count=%s", len(documents))
        return len(documents)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
