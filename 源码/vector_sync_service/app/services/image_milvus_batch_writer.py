"""Async micro-batch writer for image Milvus records."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import count
import time
from typing import Any, Optional

from app.config import settings
from app.observability.metrics import (
    image_milvus_batch_flush_total,
    image_stage_latency_seconds,
)
from app.services.image_worker_utils import sanitize_for_mongo
from app.services.milvus_image_service import MilvusImageService


@dataclass(slots=True)
class _PendingImageWrite:
    collection_name: str
    image_record: Any
    future: asyncio.Future[dict[str, Any]]
    write_mode: str


class MilvusImageBatchWriter:
    """Collect image writes briefly and flush them with one batch insert."""

    def __init__(
        self,
        *,
        service: Optional[MilvusImageService] = None,
        batch_size: Optional[int] = None,
        flush_interval_ms: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> None:
        self.service = service or MilvusImageService()
        self.batch_size = max(int(batch_size or settings.IMAGE_MILVUS_WRITE_BATCH_SIZE), 1)
        self.flush_interval_seconds = max(
            int(flush_interval_ms or settings.IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS),
            1,
        ) / 1000
        self.concurrency = max(
            int(concurrency or settings.IMAGE_MILVUS_WRITE_CONCURRENCY or 1),
            1,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix="image-milvus",
        )
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._lock = asyncio.Lock()
        self._pending: list[_PendingImageWrite] = []
        self._timer_task: Optional[asyncio.Task[None]] = None
        self._batch_ids = count(1)
        self._closed = False

    def _write_mode(self) -> str:
        mode = str(settings.IMAGE_MILVUS_WRITE_MODE or "replace").strip().lower()
        if settings.LIVE_E2E_NO_DELETE:
            return "insert_only"
        if mode in {"insert", "insert_only"}:
            return "insert_only"
        if mode in {
            "insert_or_skip_if_exists",
            "insert_or_skip",
            "skip_if_exists",
        }:
            return "insert_or_skip_if_exists"
        if mode in {"upsert", "upsert_if_exists", "replace", "replace_if_exists"}:
            return "upsert"
        raise ValueError(
            "IMAGE_MILVUS_WRITE_MODE must be one of insert_only, insert_or_skip_if_exists, or upsert"
        )

    async def replace_image_by_img_id(
        self,
        collection_name: str,
        image_record: Any,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("MilvusImageBatchWriter is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        batch: list[_PendingImageWrite] = []
        async with self._lock:
            self._pending.append(
                _PendingImageWrite(
                    collection_name=collection_name,
                    image_record=image_record,
                    future=future,
                    write_mode=self._write_mode(),
                )
            )
            if len(self._pending) >= self.batch_size:
                batch = self._take_pending_locked()
            elif self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._flush_after_delay())
        if batch:
            asyncio.create_task(self._flush_batch(batch))
        return await future

    async def _enqueue_single_write(
        self,
        collection_name: str,
        image_record: Any,
        *,
        write_mode: str,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("MilvusImageBatchWriter is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        batch: list[_PendingImageWrite] = []
        async with self._lock:
            self._pending.append(
                _PendingImageWrite(
                    collection_name=collection_name,
                    image_record=image_record,
                    future=future,
                    write_mode=write_mode,
                )
            )
            if len(self._pending) >= self.batch_size:
                batch = self._take_pending_locked()
            elif self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._flush_after_delay())
        if batch:
            asyncio.create_task(self._flush_batch(batch))
        return await future

    async def insert_images(
        self,
        collection_name: str,
        images: list[Any],
    ) -> dict[str, Any]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            started = time.perf_counter()
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    self.service.insert_images,
                    collection_name,
                    images,
                )
            finally:
                image_stage_latency_seconds.labels(stage="milvus_batch_flush").observe(
                    time.perf_counter() - started
                )
            return sanitize_for_mongo(result) or {}

    async def insert_or_skip_images_by_img_id_batch(
        self,
        collection_name: str,
        images: list[Any],
    ) -> dict[str, Any]:
        if len(images) == 1:
            return await self._enqueue_single_write(
                collection_name,
                images[0],
                write_mode="insert_or_skip_if_exists",
            )
        return await self._run_batch_write(
            collection_name,
            images,
            self.service.insert_or_skip_images_by_img_id_batch,
        )

    async def upsert_images_by_img_id_batch(
        self,
        collection_name: str,
        images: list[Any],
    ) -> dict[str, Any]:
        if len(images) == 1:
            return await self._enqueue_single_write(
                collection_name,
                images[0],
                write_mode="upsert",
            )
        return await self._run_batch_write(
            collection_name,
            images,
            self.service.upsert_images_by_img_id_batch,
        )

    async def flush(self) -> None:
        async with self._lock:
            batch = self._take_pending_locked()
        if batch:
            await self._flush_batch(batch)

    async def close(self) -> None:
        self._closed = True
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        await asyncio.to_thread(self.service.close)
        self._executor.shutdown(wait=True, cancel_futures=False)

    async def _flush_after_delay(self) -> None:
        await asyncio.sleep(self.flush_interval_seconds)
        async with self._lock:
            batch = self._take_pending_locked()
        if batch:
            await self._flush_batch(batch)

    def _take_pending_locked(self) -> list[_PendingImageWrite]:
        batch = self._pending
        self._pending = []
        return batch

    async def _flush_batch(self, batch: list[_PendingImageWrite]) -> None:
        if not batch:
            return
        batch_id = next(self._batch_ids)
        started = time.perf_counter()
        try:
            result = await self._replace_batch(batch)
        except Exception as error:
            image_milvus_batch_flush_total.labels(status="failure").inc()
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)
            return
        finally:
            image_stage_latency_seconds.labels(stage="milvus_batch_flush").observe(
                time.perf_counter() - started
            )

        image_milvus_batch_flush_total.labels(status="success").inc()
        delete_results = list(result.get("delete_results") or [])
        insert_result = result.get("insert_result")
        for index, item in enumerate(batch):
            if item.future.done():
                continue
            item.future.set_result(
                {
                    "delete_result": sanitize_for_mongo(
                        delete_results[index] if index < len(delete_results) else None
                    ),
                    "insert_result": sanitize_for_mongo(insert_result),
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                    "write_mode": item.write_mode,
                }
            )

    async def _replace_batch(
        self,
        batch: list[_PendingImageWrite],
    ) -> dict[str, Any]:
        by_collection: dict[tuple[str, str], list[_PendingImageWrite]] = {}
        for item in batch:
            by_collection.setdefault((item.collection_name, item.write_mode), []).append(item)
        combined_delete_results: list[Any] = []
        combined_insert_count = 0
        combined_existing_count = 0
        combined_new_count = 0
        write_mode = self._write_mode()
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            for (collection_name, write_mode), items in by_collection.items():
                if write_mode == "insert_only":
                    insert_result = await loop.run_in_executor(
                        self._executor,
                        self.service.insert_images,
                        collection_name,
                        [item.image_record for item in items],
                    )
                    result = {
                        "delete_results": [
                            {
                                "skipped": True,
                                "status": "DELETE_NOOP_RECORDED",
                                "reason": (
                                    "LIVE_E2E_NO_DELETE"
                                    if settings.LIVE_E2E_NO_DELETE
                                    else "IMAGE_MILVUS_WRITE_MODE_INSERT_ONLY"
                                ),
                                "operation": "replace_image_predelete_batch",
                                "collection_name": collection_name,
                            }
                        ],
                        "insert_result": insert_result,
                        "batch_size": len(items),
                        "existing_count": 0,
                        "new_count": len(items),
                        "write_mode": write_mode,
                    }
                elif write_mode == "insert_or_skip_if_exists":
                    result = await loop.run_in_executor(
                        self._executor,
                        self.service.insert_or_skip_images_by_img_id_batch,
                        collection_name,
                        [item.image_record for item in items],
                    )
                elif write_mode == "upsert":
                    result = await loop.run_in_executor(
                        self._executor,
                        self.service.upsert_images_by_img_id_batch,
                        collection_name,
                        [item.image_record for item in items],
                    )
                else:
                    result = await loop.run_in_executor(
                        self._executor,
                        self.service.replace_images_by_img_id_batch,
                        collection_name,
                        [item.image_record for item in items],
                    )
                combined_delete_results.extend(result.get("delete_results") or [])
                insert_result = result.get("insert_result") or {}
                combined_insert_count += int(insert_result.get("insert_count") or 0)
                combined_existing_count += int(result.get("existing_count") or 0)
                combined_new_count += int(result.get("new_count") or 0)
                self._verify_visibility(
                    collection_name,
                    [item.image_record for item in items],
                )
        return {
            "delete_results": combined_delete_results,
            "insert_result": {"insert_count": combined_insert_count},
            "existing_count": combined_existing_count,
            "new_count": combined_new_count,
            "write_mode": write_mode,
        }

    def _verify_visibility(self, collection_name: str, images: list[Any]) -> None:
        if not settings.IMAGE_VISIBILITY_CHECK_ENABLED:
            return
        wait_visible = getattr(self.service, "wait_for_visible_image_ids", None)
        count_visible = getattr(self.service, "count_visible_image_ids", None)
        if wait_visible is None and count_visible is None:
            return
        image_ids = [
            _image_id(item) for item in images
        ]
        image_ids = [item for item in image_ids if item]
        if not image_ids:
            return
        expected = len(set(image_ids))
        if wait_visible is not None:
            status = wait_visible(
                collection_name,
                image_ids,
                retries=settings.IMAGE_VISIBILITY_RETRIES,
                retry_delay_sec=settings.IMAGE_VISIBILITY_RETRY_DELAY_SEC,
                timeout=settings.IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC,
            )
            if bool(status.get("success")):
                return
            visible = int(status.get("visible_count") or 0)
        else:
            visible = 0
            for attempt in range(max(int(settings.IMAGE_VISIBILITY_RETRIES), 0) + 1):
                visible = int(
                    count_visible(
                        collection_name,
                        image_ids,
                        timeout=settings.IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC,
                    )
                )
                if visible >= expected:
                    return
                if attempt < max(int(settings.IMAGE_VISIBILITY_RETRIES), 0):
                    time.sleep(max(float(settings.IMAGE_VISIBILITY_RETRY_DELAY_SEC), 0.0))
        if settings.IMAGE_VISIBILITY_STRICT:
            raise RuntimeError(
                "image write visibility check failed: "
                f"visible={visible}, expected={expected}, collection={collection_name}"
            )

    async def _run_batch_write(
        self,
        collection_name: str,
        images: list[Any],
        func: Any,
    ) -> dict[str, Any]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            started = time.perf_counter()
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    func,
                    collection_name,
                    images,
                )
            finally:
                image_stage_latency_seconds.labels(stage="milvus_batch_flush").observe(
                    time.perf_counter() - started
                )
            return sanitize_for_mongo(result) or {}


def _image_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("img_id") or value.get("imageId") or value.get("id") or "").strip()
    return str(
        getattr(value, "img_id", None)
        or getattr(value, "imageId", None)
        or getattr(value, "id", None)
        or ""
    ).strip()
