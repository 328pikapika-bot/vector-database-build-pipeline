"""Query-only entry point. Starts no ingestion, Kafka, or cleanup workers."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.embedder import close_embedding_client
from app.services.fulltext_search import router


@asynccontextmanager
async def lifespan(app):
    yield
    await close_embedding_client()


app = FastAPI(title="NSTL Fulltext Read-only Search", lifespan=lifespan)
app.include_router(router)
