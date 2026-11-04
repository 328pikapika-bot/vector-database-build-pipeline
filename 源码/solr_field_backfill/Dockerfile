# Build: docker -H tcp://127.0.0.1:2375 build -t docker-hosted.example.invalid/nstl/vector-sync/solr-field-backfill:v1.2.0 solr_field_backfill
# Push: docker -H tcp://127.0.0.1:2375 push docker-hosted.example.invalid/nstl/vector-sync/solr-field-backfill:v1.2.0
# Batch worker has no HTTP listener; no EXPOSE or published ports are needed.
FROM python:3.11-slim-bookworm

ENV TZ=Asia/Shanghai \
    BACKFILL_MODE=check \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backfill.py resilient.py production_parser.py __init__.py field_mapping.json ./
RUN mkdir -p /app/state /app/logs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
VOLUME ["/app/state", "/app/logs"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -m backfill --healthcheck
CMD ["python", "-m", "backfill"]
