from __future__ import annotations

import asyncio

import pytest

from app.services.image_milvus_batch_writer import MilvusImageBatchWriter


@pytest.mark.asyncio
async def test_milvus_batch_writer_combines_concurrent_replaces():
    service = FakeBatchService()
    writer = MilvusImageBatchWriter(
        service=service,
        batch_size=3,
        flush_interval_ms=1000,
        concurrency=1,
    )

    results = await asyncio.gather(
        writer.replace_image_by_img_id("images", {"img_id": "img-1"}),
        writer.replace_image_by_img_id("images", {"img_id": "img-2"}),
        writer.replace_image_by_img_id("images", {"img_id": "img-3"}),
    )
    await writer.close()

    assert service.batch_calls == [
        {
            "collection_name": "images",
            "images": [{"img_id": "img-1"}, {"img_id": "img-2"}, {"img_id": "img-3"}],
            "mode": "upsert_if_exists",
        }
    ]
    assert [result["batch_size"] for result in results] == [3, 3, 3]
    assert [result["batch_id"] for result in results] == [1, 1, 1]
    assert [result["insert_result"] for result in results] == [
        {"insert_count": 3},
        {"insert_count": 3},
        {"insert_count": 3},
    ]
    assert [result["delete_result"] for result in results] == [None, None, None]


@pytest.mark.asyncio
async def test_milvus_batch_writer_fails_whole_batch_on_flush_error():
    writer = MilvusImageBatchWriter(
        service=FakeBatchService(error=RuntimeError("milvus down")),
        batch_size=2,
        flush_interval_ms=1000,
        concurrency=1,
    )

    with pytest.raises(RuntimeError, match="milvus down"):
        await asyncio.gather(
            writer.replace_image_by_img_id("images", {"img_id": "img-1"}),
            writer.replace_image_by_img_id("images", {"img_id": "img-2"}),
        )
    await writer.close()


@pytest.mark.asyncio
async def test_milvus_batch_writer_exposes_insert_or_skip_and_upsert_batch_calls():
    service = FakeBatchService()
    writer = MilvusImageBatchWriter(
        service=service,
        batch_size=2,
        flush_interval_ms=1000,
        concurrency=1,
    )

    insert_or_skip_result = await writer.insert_or_skip_images_by_img_id_batch(
        "images",
        [{"img_id": "img-1"}],
    )
    upsert_result = await writer.upsert_images_by_img_id_batch(
        "images",
        [{"img_id": "img-2"}],
    )
    await writer.close()

    assert service.batch_calls == [
        {
            "collection_name": "images",
            "images": [{"img_id": "img-1"}],
            "mode": "insert_or_skip_if_exists",
        },
        {
            "collection_name": "images",
            "images": [{"img_id": "img-2"}],
            "mode": "upsert_if_exists",
        },
    ]
    assert insert_or_skip_result["write_mode"] == "insert_or_skip_if_exists"
    assert upsert_result["write_mode"] == "upsert"


@pytest.mark.asyncio
async def test_single_insert_or_skip_calls_are_aggregated_into_shared_flush_batch():
    service = FakeBatchService()
    writer = MilvusImageBatchWriter(
        service=service,
        batch_size=3,
        flush_interval_ms=1000,
        concurrency=1,
    )

    results = await asyncio.gather(
        writer.insert_or_skip_images_by_img_id_batch("images", [{"img_id": "img-1"}]),
        writer.insert_or_skip_images_by_img_id_batch("images", [{"img_id": "img-2"}]),
        writer.insert_or_skip_images_by_img_id_batch("images", [{"img_id": "img-3"}]),
    )
    await writer.close()

    assert service.batch_calls == [
        {
            "collection_name": "images",
            "images": [{"img_id": "img-1"}, {"img_id": "img-2"}, {"img_id": "img-3"}],
            "mode": "insert_or_skip_if_exists",
        }
    ]
    assert [result["batch_size"] for result in results] == [3, 3, 3]
    assert [result["batch_id"] for result in results] == [1, 1, 1]


class FakeBatchService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.batch_calls: list[dict] = []
        self.closed = False

    def replace_images_by_img_id_batch(self, collection_name: str, images: list[dict]):
        self.batch_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
                "mode": "replace",
            }
        )
        if self.error is not None:
            raise self.error
        return {
            "delete_results": [{"delete_count": 0} for _ in images],
            "insert_result": {"insert_count": len(images)},
        }

    def insert_or_skip_images_by_img_id_batch(self, collection_name: str, images: list[dict]):
        self.batch_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
                "mode": "insert_or_skip_if_exists",
            }
        )
        if self.error is not None:
            raise self.error
        return {
            "delete_results": [],
            "insert_result": {"insert_count": len(images)},
            "existing_count": 0,
            "new_count": len(images),
            "write_mode": "insert_or_skip_if_exists",
        }

    def upsert_images_by_img_id_batch(self, collection_name: str, images: list[dict]):
        self.batch_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
                "mode": "upsert_if_exists",
            }
        )
        if self.error is not None:
            raise self.error
        return {
            "delete_results": [],
            "insert_result": {"insert_count": len(images)},
            "existing_count": 0,
            "new_count": len(images),
            "write_mode": "upsert_if_exists",
        }

    def insert_images(self, collection_name: str, images: list[dict]):
        self.batch_calls.append(
            {
                "collection_name": collection_name,
                "images": list(images),
                "mode": "insert_only",
            }
        )
        if self.error is not None:
            raise self.error
        return {"insert_count": len(images)}

    def close(self) -> None:
        self.closed = True
