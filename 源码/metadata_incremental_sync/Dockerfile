# Build/push through the configured NSTL Docker daemon after acceptance:
# docker -H tcp://127.0.0.1:2375 build --progress=plain -t docker-hosted.example.invalid/nstl/vector-sync/nstl-metadata-incremental-sync:v0.3.0 .
# docker -H tcp://127.0.0.1:2375 push docker-hosted.example.invalid/nstl/vector-sync/nstl-metadata-incremental-sync:v0.3.0
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai \
    TZDIR=/usr/local/lib/python3.11/site-packages/tzdata/zoneinfo SYNC_MODE=check
LABEL org.opencontainers.image.title="nstl-metadata-incremental-sync" \
      org.opencontainers.image.version="0.3.0"
WORKDIR /app
COPY requirements.txt /app/requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r /app/requirements.txt
COPY incremental_sync /app/incremental_sync
# Internal status endpoint only; the Stack does not publish a host port.
EXPOSE 8099
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -m incremental_sync.health
CMD ["python", "-m", "incremental_sync"]
