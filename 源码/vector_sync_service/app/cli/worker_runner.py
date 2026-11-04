"""Shared process lifecycle for the three independent Kafka workers."""
from __future__ import annotations

import asyncio

from app.config import settings
from app.observability.logger import get_logger, setup_logging
from app.services.embedder import close_embedding_client
from app.services.independent_kafka_worker import IndependentKafkaWorker, kafka_group_id_for_role
from app.services.kafka_consumer import create_consumer, run_consumer
from app.services.milvus_image_service import MilvusImageService
from app.services.milvus_service import close_milvus_service
from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager

logger = get_logger(__name__)


def initialize_independent_worker_target(role: str) -> None:
    """Perform the one permitted image collection create-or-reuse operation.

    This runs before the Kafka consumer is constructed.  It makes the Stack
    collection setting authoritative and gives an operator a clear startup
    failure instead of allowing a running writer to choose a suffixed target.
    """
    if role != "image":
        return

    service = MilvusImageService()
    try:
        target = service.init_image_collection(settings.IMAGE_COLLECTION_NAME)
    finally:
        service.close()
    logger.info(
        "image_collection_startup_ready",
        collection_name=target["resolved_name"],
        created=target["created"],
        reused=target["reused"],
        target_signature=target["target_signature"],
    )


async def run_independent_worker(role: str) -> None:
    """Run one role with its own Kafka group and direct data-plane processor."""
    configured_role = str(settings.SYNC_WORKER_ROLE or "").strip().lower()
    if configured_role and configured_role != role:
        raise ValueError(
            f"SYNC_WORKER_ROLE={configured_role!r} does not match CLI role={role!r}"
        )

    manager = get_mongo_state_manager()
    await manager.connect()
    processor: IndependentKafkaWorker | None = None
    consumer = None
    try:
        initialize_independent_worker_target(role)
        processor = IndependentKafkaWorker(role, state_manager=manager)
        consumer = create_consumer(group_id=kafka_group_id_for_role(role))
        logger.info(
            "independent_kafka_worker_started",
            role=role,
            topic=settings.KAFKA_TOPIC,
            group_id=kafka_group_id_for_role(role),
        )
        await run_consumer(consumer, processor.process_batch)
    finally:
        try:
            if consumer is not None:
                await asyncio.to_thread(consumer.close)
        finally:
            try:
                if processor is not None:
                    await processor.close()
            finally:
                await close_embedding_client()
                close_milvus_service()
                await close_mongo_state_manager()
                logger.info("independent_kafka_worker_stopped", role=role)


def main(role: str) -> None:
    setup_logging()
    asyncio.run(run_independent_worker(role))
