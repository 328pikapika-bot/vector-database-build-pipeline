"""Reusable runtime resources for the image worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings
from app.services.fulltext_downloader import FullTextDownloader
from app.services.image_downloader import ImageDownloader
from app.services.image_milvus_batch_writer import MilvusImageBatchWriter
from app.services.image_offline_providers import (
    OfflineImageModelClient,
    build_offline_image_feature_providers,
)


@dataclass(slots=True)
class ImageWorkerResourcePool:
    # Attachment and model traffic deliberately use separate clients. A high
    # volume of image GETs must not consume the sockets needed by embedding
    # POSTs, otherwise increasing download concurrency starves the model.
    client: httpx.AsyncClient
    model_client_http: httpx.AsyncClient
    downloader: ImageDownloader
    fulltext_downloader: FullTextDownloader
    model_client: OfflineImageModelClient
    writer: Optional[MilvusImageBatchWriter] = None
    feature_providers: Optional[object] = None

    def get_feature_providers(self) -> object:
        if self.feature_providers is None:
            self.feature_providers = build_offline_image_feature_providers(
                client=self.model_client
            )
        return self.feature_providers

    async def close(self) -> None:
        if self.writer is not None:
            await self.writer.close()
        await self.client.aclose()
        await self.model_client_http.aclose()


def create_image_worker_resource_pool(
    *,
    include_writer: bool,
    download_timeout: Optional[int] = None,
    download_concurrency: Optional[int] = None,
) -> ImageWorkerResourcePool:
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            max(
                int(settings.IMAGE_DOWNLOAD_TIMEOUT or 0),
                int(settings.FULLTEXT_DOWNLOAD_TIMEOUT or 0),
                int(settings.IMAGE_FEATURE_TIMEOUT or 0),
                1,
            )
        ),
        limits=httpx.Limits(
            max_connections=max(int(settings.IMAGE_HTTP_POOL_MAX_CONNECTIONS or 1), 1),
            max_keepalive_connections=max(
                int(settings.IMAGE_HTTP_POOL_MAX_KEEPALIVE or 0),
                0,
            ),
        ),
    )
    model_client_http = httpx.AsyncClient(
        timeout=httpx.Timeout(max(int(settings.IMAGE_FEATURE_TIMEOUT or 0), 1)),
        limits=httpx.Limits(
            max_connections=max(int(settings.IMAGE_MODEL_HTTP_POOL_MAX_CONNECTIONS or 1), 1),
            max_keepalive_connections=max(
                int(settings.IMAGE_MODEL_HTTP_POOL_MAX_KEEPALIVE or 0), 0
            ),
        ),
    )
    return ImageWorkerResourcePool(
        client=client,
        model_client_http=model_client_http,
        downloader=ImageDownloader(
            client=client,
            timeout=download_timeout,
            concurrency=download_concurrency,
        ),
        fulltext_downloader=FullTextDownloader(
            client=client,
            timeout=settings.IMAGE_MMD_DOWNLOAD_TIMEOUT,
            max_retries=settings.IMAGE_MMD_DOWNLOAD_MAX_RETRIES,
        ),
        model_client=OfflineImageModelClient.from_settings(http_client=model_client_http),
        writer=MilvusImageBatchWriter() if include_writer else None,
    )
