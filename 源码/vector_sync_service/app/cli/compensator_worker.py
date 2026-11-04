"""Start the scheduled/continuous/once failure compensator process."""
from __future__ import annotations

import asyncio

from app.observability.logger import setup_logging
from app.services.compensator import run_compensator
from app.services.embedder import close_embedding_client
from app.services.milvus_service import close_milvus_service
from app.services.mongo_state_manager import close_mongo_state_manager, get_mongo_state_manager


async def _run() -> None:
    manager = get_mongo_state_manager()
    await manager.connect()
    try:
        await run_compensator()
    finally:
        await close_embedding_client()
        close_milvus_service()
        await close_mongo_state_manager()


def main() -> None:
    setup_logging()
    asyncio.run(_run())


if __name__ == "__main__":
    main()
