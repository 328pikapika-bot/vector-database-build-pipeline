# Vector Database Build Pipeline

English | [简体中文](README.md)

Ingests literature metadata, full text, and images into a traceable vector-index pipeline. It also handles incremental updates, retries, backpressure, and consistency checks.

## Capabilities

- Consumes Kafka literature events and updates document vectors using stable identifiers and versions.
- Downloads, cleans, chunks, and embeds full text asynchronously; the image path produces image and text features.
- Includes metadata synchronization, Solr field backfill, legacy imports, monitoring, and diagnostic tools.

## Code and setup

`源码/vector_sync_service/` is the main service; `源码/metadata_incremental_sync/` handles incremental metadata; `源码/solr_field_backfill/` handles backfill; `源码/milvus_import_old/` holds historical import tools; `源码/kafbat-ui/` is a Kafka UI example. The main entry is `源码/vector_sync_service/app/main.py`, with configuration in `app/config.py`.

Choose the subproject you need and install its own dependencies. The broader system uses Python, Kafka, Milvus, MongoDB, Solr, object storage, and text/vision embedding services. The subprojects do not need to run together.

## Configuration

Kafka reads `KAFKA_SASL_USERNAME` and `KAFKA_SASL_PASSWORD`; Milvus reads `MILVUS_USER` and `MILVUS_PASSWORD`. Provide model keys through the relevant environment variables. Use the `.env.example` files under `源码/` as templates; example addresses are local placeholders.

## Verification and limits

The legacy importer is kept for compatibility and is not the main service's default path. Real migrations, consumer offsets, vector writes, and load behavior were not exercised during the 2026-11-04 public-code review. Run migration and stress tools only against an isolated target that you configure.

See the [publication notes](仓库发布说明.md) for scope. Archived documents and conversations remain local.
