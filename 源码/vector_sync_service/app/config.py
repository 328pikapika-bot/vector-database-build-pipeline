"""
NSTL 向量库同步服务 - 配置中心
基于 pydantic-settings，所有配置从环境变量 /.env 读取
"""
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone
from uuid import uuid4


_AUTO_COLLECTION_RUN_ID = (
    f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid4().hex[:8]}"
)


class Settings(BaseSettings):
    """全局配置，所有值从环境变量或 .env 文件读取"""

    # ========== Kafka ==========
    KAFKA_BOOTSTRAP_SERVERS: str = "127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092,127.0.0.1:9092"
    KAFKA_TOPIC: str = "VECTOR_JOURNAL_ARTICLE"
    KAFKA_GROUP_ID: str = "python_consumer_group_scientific_20260617"
    # Independent workers consume the same topic with different groups.  Kafka
    # keeps one offset per group, so metadata/fulltext/image never consume one
    # another's messages.
    KAFKA_METADATA_GROUP_ID: str = "vector_sync_metadata_v1"
    KAFKA_FULLTEXT_GROUP_ID: str = "vector_sync_fulltext_v1"
    KAFKA_IMAGE_GROUP_ID: str = "vector_sync_image_v1"
    KAFKA_MAX_MESSAGE_BYTES: int = 10 * 1024 * 1024
    KAFKA_MAX_BATCH_BYTES: int = 64 * 1024 * 1024
    # New collection/index targets must read the topic from the beginning.
    # The base group id is suffixed with a deterministic target version.
    KAFKA_TARGET_VERSIONED_GROUP_ENABLED: bool = True
    # Set by the dedicated Stack as an explicit deployment guard.
    SYNC_WORKER_ROLE: str = ""
    KAFKA_SECURITY_PROTOCOL: str = "SASL_PLAINTEXT"
    KAFKA_SASL_MECHANISM: str = "PLAIN"
    KAFKA_SASL_USERNAME: str = ""
    KAFKA_SASL_PASSWORD: str = ""
    KAFKA_DLQ_ENABLED: bool = False
    KAFKA_DLQ_TOPIC: str = ""
    KAFKA_DUMP_FILE: str = "kafka_messages.jsonl"
    KAFKA_ERROR_DUMP_FILE: str = "kafka_messages_error.jsonl"
    KAFKA_PROCESSED_DUMP_FILE: str = "kafka_messages_processed.jsonl"
    KAFKA_CONSUMER_ENABLED: bool = True
    KAFKA_CONSUME_LIMIT: int = 0
    KAFKA_MAX_POLL_INTERVAL_MS: int = 300000
    KAFKA_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_HEARTBEAT_INTERVAL_MS: int = 10000
    KAFKA_CONSUMER_TIME_BUDGET_MS: int = 8000
    KAFKA_CONSUMER_MICRO_BATCH_SIZE: int = 8
    KAFKA_START_PAUSED: bool = False
    KAFKA_BACKPRESSURE_ENABLED: bool = True
    KAFKA_BACKPRESSURE_MODE: str = "image_priority"
    KAFKA_BACKPRESSURE_HIGH_WATERMARK: int = 10000
    KAFKA_BACKPRESSURE_LOW_WATERMARK: int = 3000
    KAFKA_BACKPRESSURE_CHECK_INTERVAL_SEC: int = 5
    KAFKA_BACKPRESSURE_IMAGE_PENDING_SOFT_LIMIT: int = 8000
    KAFKA_BACKPRESSURE_IMAGE_PENDING_RESUME_THRESHOLD: int = 2500
    KAFKA_BACKPRESSURE_SOFT_WATERMARK_FACTOR: float = 0.55
    KAFKA_BACKPRESSURE_SOFT_THROTTLE_SLEEP_MS: int = 200
    KAFKA_BACKPRESSURE_SOFT_MICRO_BATCH_MIN: int = 1
    KAFKA_BACKPRESSURE_DOC_LEAD_RESUME_FACTOR: float = 0.5
    KAFKA_BACKPRESSURE_IMAGE_DOC_LEAD_LIMIT: int = 5000
    KAFKA_BACKPRESSURE_FULLTEXT_DOC_LEAD_LIMIT: int = 5000

    # ========== Source data resolution ==========
    # Dedicated workers receive only a document ID from Kafka.  They resolve
    # the authoritative document and image metadata from Solr before work.
    SOLR_PAPER_URL: str = "http://solr-bak1.example.invalid:8984/solr/paper6.6"
    SOLR_IMAGE_URL: str = "http://solr-bak1.example.invalid:8984/solr/standard6.6"
    SOLR_REQUEST_TIMEOUT_SEC: float = 20.0
    # A brief source-side connection reset should not turn an otherwise valid
    # Kafka record into a deferred failure.  Only transport and 408/429/5xx
    # responses are retried by the resolver.
    SOLR_REQUEST_MAX_RETRIES: int = 2
    SOLR_REQUEST_RETRY_BACKOFF_MS: int = 150
    SOLR_IMAGE_ROWS: int = 1000
    SOLR_RESOLVE_CONCURRENCY: int = 32
    PIPELINE_VERSIONED_DEDUP_ENABLED: bool = True

    # ========== Milvus ==========
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: int = 19533
    MILVUS_COLLECTION: str = "scientific_documents_v2"
    DOCUMENT_COLLECTION_NAME: str = ""
    IMAGE_COLLECTION_NAME: str = "scientific_images_v2"
    # Dedicated Stacks enable this so an empty collection variable gets a
    # process-unique target. Legacy deployments keep the historical defaults.
    COLLECTION_AUTO_CREATE_ON_EMPTY: bool = False
    # When true, an explicitly named existing collection must match its schema
    # and index; never silently redirect writes to a suffixed collection.
    COLLECTION_TARGET_STRICT: bool = False
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    VECTOR_DIM: int = 2560
    FULLTEXT_SEARCH_TOKEN: str = ""
    FULLTEXT_SEARCH_ACCESS_TYPES: str = ""
    MILVUS_PROXY_MAX_FIELD_NUM: int = 512
    MILVUS_PROXY_MAX_VECTOR_FIELD_NUM: int = 8
    # Bound every data-plane RPC.  A stuck proxy call must become a retryable
    # task failure instead of holding the Kafka ordered-commit head forever.
    MILVUS_OPERATION_TIMEOUT_SEC: float = 120.0
    MILVUS_WRITE_TIMEOUT_SEC: float = 120.0
    MILVUS_QUERY_TIMEOUT_SEC: float = 20.0
    MILVUS_DELETE_TIMEOUT_SEC: float = 60.0

    # ========== MongoDB ==========
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "nstl_vector_sync_scientific_20260617"
    MONGO_USERNAME: Optional[str] = None
    MONGO_PASSWORD: Optional[str] = None
    MONGO_STATUS_COLLECTION: str = "doc_sync_status"
    MONGO_FAILURE_COLLECTION: str = "doc_sync_failures"
    MONGO_ATTEMPT_COLLECTION: str = "doc_sync_attempts"
    MONGO_ARTIFACT_COLLECTION: str = "doc_sync_artifacts"
    # Compact source-version watermarks for direct Kafka idempotency. These are
    # operational state records, not vector business data.
    MONGO_PIPELINE_VERSION_COLLECTION: str = "doc_sync_pipeline_versions"
    # When false, successful metadata documents do not read or write audit data in Mongo.
    # Mongo remains the durable store for failures, retry payloads, and error diagnostics.
    MONGO_SUCCESS_AUDIT_ENABLED: bool = False
    MONGO_FULLTEXT_TASK_COLLECTION: str = "doc_fulltext_tasks"
    MONGO_IMAGE_TASK_COLLECTION: str = "doc_image_tasks"
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 3000
    MONGO_CONNECT_TIMEOUT_MS: int = 3000
    MONGO_SOCKET_TIMEOUT_MS: int = 10000
    MONGO_ARTIFACT_COMPRESS_THRESHOLD_BYTES: int = 65536
    MONGO_ARTIFACT_SUCCESS_RETENTION_DAYS: int = 7
    MONGO_ARTIFACT_FAILURE_RETENTION_DAYS: int = 180
    MONGO_FAILURE_TTL_DAYS: int = 180
    MONGO_FAILURE_RETENTION_ENABLED: bool = False
    MONGO_ATTEMPT_TTL_DAYS: int = 180
    MONGO_FULLTEXT_TASK_TTL_DAYS: int = 180
    MONGO_IMAGE_TASK_TTL_DAYS: int = 180
    MONGO_SUCCESS_TASK_RETENTION_DAYS: int = 7
    RETENTION_CLEANUP_ENABLED: bool = False
    RETENTION_CLEANUP_DAILY_INTERVAL_HOURS: int = 24
    RETENTION_CLEANUP_WEEKLY_DAY: int = 0
    RETENTION_CLEANUP_REPORT_RETENTION_DAYS: int = 7
    IMAGE_LOCAL_SUCCESS_RETENTION_HOURS: int = 24
    LOCAL_FAILED_RETENTION_DAYS: int = 60
    MONGO_BACKUP_DB: str = "nstl_vector_sync_scientific_backup_latest"
    MONGO_BACKUP_SCHEDULE: str = "0 2 * * *"

    # ========== Embedding API ==========
    EMBEDDING_API_URL: str = "http://127.0.0.1:16703/v1/embeddings"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "qwen3-embed"
    EMBEDDING_MODEL_VERSION: str = ""
    EMBEDDING_BATCH_SIZE: int = 32
    # Merge concurrent short fulltext requests into one embedding request.
    EMBEDDING_CROSS_DOCUMENT_BATCH_ENABLED: bool = True
    EMBEDDING_CROSS_DOCUMENT_BATCH_WAIT_MS: int = 25
    # Optional per-worker budget in input texts per second. Production Stack
    # files enable this separately for image, fulltext, and metadata lanes.
    EMBEDDING_RATE_LIMIT_ENABLED: bool = False
    EMBEDDING_LANE: str = ""
    EMBEDDING_TEXT_RATE: float = 0.0
    EMBEDDING_TEXT_BURST: int = 32

    # ========== Processing Parameters ==========
    BATCH_MAX_SIZE: int = 200
    BATCH_MAX_WAIT_MS: int = 3000
    SYNC_PIPELINE_MODE: str = "standard"
    IMAGE_PRIORITY_WINDOW_COLLECTION: str = "image_priority_windows"
    IMAGE_PRIORITY_WINDOW_ID: str = ""
    DOCUMENT_PIPELINE_CONCURRENCY: int = 8
    # metadata_bulk batches parsing, embedding, and Milvus writes across documents.
    DOCUMENT_BULK_PIPELINE_ENABLED: bool = False
    DOCUMENT_BULK_MILVUS_BATCH_SIZE: int = 512
    # A dedicated metadata_bulk consumer may keep a small number of batches
    # in flight so embedding for the next batch overlaps Milvus I/O for the
    # previous one. Keep this disabled for the combined fan-out consumer.
    DOCUMENT_BULK_PIPELINE_INFLIGHT_BATCHES: int = 1
    # Keep a pathological source abstract below the embedding endpoint context
    # window. The original metadata remains intact in Milvus; this only bounds
    # the text used to derive its dense vector.
    METADATA_EMBEDDING_MAX_CHARS: int = 30_000
    # Disabled by default. Benchmark runs may enable this to expose the
    # metadata bulk pipeline's phase timings without affecting production.
    METADATA_BULK_TIMING_ENABLED: bool = False
    # Full-text chunks target roughly 2k characters while keeping a bounded
    # upper limit so adjacent paragraphs can be aggregated without crossing
    # section boundaries.
    CHUNK_TARGET_CHARS: int = 2000
    CHUNK_MAX_CHARS: int = 2500
    CHUNK_OVERLAP_CHARS: int = 200
    CHUNK_MIN_CHARS: int = 300
    CHUNK_TAIL_MERGE_CHARS: int = 600
    CHUNK_MAX_PER_DOC: int = 0

    # ========== Paragraph Splitting Optimization ==========
    CHUNK_SEMANTIC_SPLIT_ENABLED: bool = True
    CHUNK_SEMANTIC_SEARCH_WINDOW: int = 200
    CHUNK_PARAGRAPH_MERGE_ENABLED: bool = True
    CHUNK_SECTION_TITLE_ATTACH_ENABLED: bool = True
    CHUNK_CONTEXT_OVERLAP_ENABLED: bool = True
    CHUNK_CONTEXT_OVERLAP_CHARS: int = 50

    EMBEDDING_MAX_CONCURRENCY: int = 80
    EMBEDDING_TIMEOUT: int = 30
    EMBEDDING_MAX_RETRIES: int = 3
    MILVUS_BATCH_SIZE: int = 1000
    MAX_RETRY: int = 5
    COMPENSATOR_ENABLED: bool = False
    COMPENSATOR_INTERVAL_SEC: int = 300
    COMPENSATOR_BATCH_SIZE: int = 100
    COMPENSATOR_CONCURRENCY: int = 8
    COMPENSATOR_LEASE_SEC: int = 900
    COMPENSATOR_RUN_MODE: str = "scheduled"
    COMPENSATOR_START_TIME: str = "00:00"
    COMPENSATOR_RUN_HOURS: float = 2.0
    COMPENSATOR_TIMEZONE: str = "Asia/Shanghai"
    COMPENSATOR_PIPELINES: str = "metadata,fulltext,image"
    RETRY_BASE_DELAY_SEC: float = 1.0
    RETRY_MAX_DELAY_SEC: float = 60.0

    # ========== Admin Control ==========
    ADMIN_AUTH_ENABLED: bool = True
    ADMIN_READ_TOKENS: str = ""
    ADMIN_WRITE_TOKENS: str = ""

    # ========== Auto Publish (Kafka 数据投递) ==========
    AUTO_PUBLISH_ENABLED: bool = False
    PUBLISH_BATCH_URL: str = "http://127.0.0.1:17123/api/service/interfaces/vector/job/publishBatch"
    PUBLISH_BATCH_TOKEN: str = ""
    PUBLISH_BATCH_SUB_DOC_TYPE: str = "A79"
    PUBLISH_BATCH_TOPIC: str = ""
    PUBLISH_BATCH_START_PAGE: int = 1
    PUBLISH_BATCH_END_PAGE: int = 0
    PUBLISH_BATCH_PAGES_PER_ROUND: int = 50
    PUBLISH_BATCH_PENDING_THRESHOLD: int = 500
    PUBLISH_BATCH_CHECK_INTERVAL_SEC: int = 30
    PUBLISH_BATCH_TIMEOUT_SEC: int = 900
    PUBLISH_BATCH_ASSUME_TIMEOUT_SUCCESS: bool = False
    PUBLISH_BATCH_INTERVAL_SEC: float = 2.0
    PUBLISH_BATCH_BACKPRESSURE_MODE: str = "all_tasks"
    PUBLISH_BATCH_DAILY_RECORD_LIMIT: int = 200_000
    PUBLISH_BATCH_PAGE_RECORD_COUNT: int = 1_000
    PUBLISH_BATCH_COUNT_MODE: str = "response_or_fallback"
    PUBLISH_BATCH_KAFKA_OFFSET_TIMEOUT_SEC: int = 10
    PUBLISH_BATCH_MAX_PAGES_PER_RUN: int = 0

    # ========== Full-text Download ==========
    FULLTEXT_CHUNK_COLLECTION: str = "scientific_fulltext_chunks_v2"
    FULLTEXT_CHUNK_COLLECTION_NAME: str = ""
    FULLTEXT_CHUNK_WRITE_MODE: str = "safe_replace"
    FULLTEXT_CHUNK_VISIBILITY_CHECK_ENABLED: bool = True
    FULLTEXT_CHUNK_VISIBILITY_STRICT: bool = True
    FULLTEXT_CHUNK_VISIBILITY_RETRIES: int = 3
    FULLTEXT_CHUNK_VISIBILITY_RETRY_DELAY_SEC: float = 1.0
    FULLTEXT_CHUNK_VISIBILITY_CHECK_LIMIT: int = 3
    FULLTEXT_CHUNK_VISIBILITY_CHECK_TIMEOUT_SEC: int = 10
    MILVUS_VISIBILITY_BATCH_SIZE: int = 1000
    FULLTEXT_CHUNK_SCHEMA_VERSION: str = "v2"
    FULLTEXT_DOWNLOAD_TIMEOUT: int = 30
    FULLTEXT_DOWNLOAD_MAX_RETRIES: int = 3
    FULLTEXT_TASK_UPSERT_ENABLED: bool = False
    FULLTEXT_DELETE_CHUNKS_ENABLED: bool = False
    FULLTEXT_WORKER_LEASE_SEC: int = 120
    FULLTEXT_WORKER_BATCH_SIZE: int = 50
    FULLTEXT_WORKER_CONCURRENCY: int = 16
    FULLTEXT_WORKER_ENABLED: bool = False
    FULLTEXT_WORKER_CLAIM_CREATED_AFTER: str = ""
    FULLTEXT_WORKER_POLL_INTERVAL_SEC: int = 3
    FULLTEXT_WORKER_TASK_TIMEOUT_SEC: int = 180
    FULLTEXT_WORKER_MIN_TASK_TIMEOUT_SEC: int = 1800
    # A dedicated fulltext Kafka consumer may overlap document-level download,
    # embedding and Milvus work while retaining FIFO offset commits.
    FULLTEXT_PIPELINE_INFLIGHT_BATCHES: int = 1
    MAX_DOC_CHARS: Optional[int] = None
    # A non-zero value rejects one downloaded MMD before parsing/chunking when
    # its estimated token count exceeds the configured hard limit.
    FULLTEXT_MAX_TOKENS: int = 0

    # ========== Milvus IVF_PQ Index ==========
    DOCUMENT_INDEX_TYPE: str = "IVFPQ"
    DOCUMENT_INDEX_METRIC_TYPE: str = "COSINE"
    DOCUMENT_IVFPQ_NLIST: int = 4096
    DOCUMENT_IVFPQ_M: int = 64
    DOCUMENT_IVFPQ_NBITS: int = 8
    DOCUMENT_IVFPQ_NPROBE: int = 16
    DOCUMENT_VISIBILITY_CHECK_ENABLED: bool = True
    DOCUMENT_VISIBILITY_STRICT: bool = True
    DOCUMENT_VISIBILITY_RETRIES: int = 3
    DOCUMENT_VISIBILITY_RETRY_DELAY_SEC: float = 1.0
    DOCUMENT_VISIBILITY_CHECK_LIMIT: int = 3
    DOCUMENT_VISIBILITY_CHECK_TIMEOUT_SEC: int = 10
    FULLTEXT_CHUNK_INDEX_TYPE: str = "IVFPQ"
    FULLTEXT_CHUNK_INDEX_METRIC_TYPE: str = "COSINE"
    FULLTEXT_CHUNK_IVFPQ_NLIST: int = 4096
    FULLTEXT_CHUNK_IVFPQ_M: int = 64
    FULLTEXT_CHUNK_IVFPQ_NBITS: int = 8
    FULLTEXT_CHUNK_IVFPQ_NPROBE: int = 16
    # Production image dense vectors use IVFPQ by default. Search-time nprobe
    # can be tuned without changing the collection target.
    IMAGE_DENSE_INDEX_TYPE: str = "IVFPQ"
    IMAGE_DENSE_INDEX_METRIC_TYPE: str = "IP"
    IMAGE_DENSE_IVFPQ_NLIST: int = 4096
    IMAGE_DENSE_IVFPQ_M: int = 64
    IMAGE_DENSE_IVFPQ_NBITS: int = 8
    IMAGE_DENSE_IVFPQ_NPROBE: int = 16

    # ========== Data Filter & Type Override ==========
    DP_IGNORE_FIELDS: str = "sysFieldInfo,cls"
    DYNAMIC_JSON_FIELDS: str = "hasJournalLink,hasSeries,hasPermissions,hasSupplementary,hasIssnHistory,hasReview,hasFullTextFile,hasFile,hasStandardOrganization,hasPictureFile,hasTable,hasJournalHistory,hasAuthor,hasReferencePaper,hasContactPerson,hasEditor,hasPublished,hasOtherPeople,hasFoundation,hasDataSources,hasStandardFoundation,hasHolding,hasSponsor,hasProceedings,hasSource,hasDataBese,hasCreateOrganization"

    # ========== Image Processing ==========
    # Internal high-throughput origin for image files and MMD resources.
    ATTACHMENT_PREFIX: str = "http://127.0.0.1:12001/"
    UPLOAD_DIR: str = "upload"
    IMAGE_TASK_UPSERT_ENABLED: bool = False
    IMAGE_WORKER_ENABLED: bool = False
    IMAGE_WORKER_LEASE_SEC: int = 180
    IMAGE_WORKER_BATCH_SIZE: int = 256
    IMAGE_WORKER_CONCURRENCY: int = 48
    IMAGE_WORKER_POLL_INTERVAL_SEC: int = 1
    IMAGE_WORKER_TASK_TIMEOUT_SEC: int = 150
    IMAGE_TASK_MAX_ATTEMPTS: int = 3
    IMAGE_DOWNLOAD_TIMEOUT: int = 45
    IMAGE_DOWNLOAD_MAX_RETRIES: int = 0
    IMAGE_DOWNLOAD_CONCURRENCY: int = 48
    IMAGE_DOWNLOAD_LANE_ENABLED: bool = True
    IMAGE_FAST_DOWNLOAD_TIMEOUT_SEC: int = 15
    IMAGE_SLOW_DOWNLOAD_TIMEOUT_SEC: int = 120
    IMAGE_SLOW_DOWNLOAD_CONCURRENCY: int = 8
    IMAGE_SLOW_RETRY_DELAY_SEC: int = 300
    IMAGE_MMD_DOWNLOAD_TIMEOUT: int = 15
    IMAGE_MMD_DOWNLOAD_MAX_RETRIES: int = 0
    # New image tasks require a real MMD match before they are written.  The
    # worker still starts visual embedding as soon as image bytes arrive.
    IMAGE_MMD_HOT_PATH_MODE: str = "required"
    IMAGE_TEXT_INDEX_POLICY: str = "verified_only"
    IMAGE_ALLOW_DOCUMENT_TITLE_CONTEXT_FALLBACK: bool = False
    IMAGE_MMD_CONTEXT_REQUIRED_FOR_TEXT_INDEX: bool = True
    IMAGE_QUALITY_AUDIT_ENABLED: bool = True
    IMAGE_QUALITY_PIPELINE_VERSION: str = "image-quality-v1"
    IMAGE_MAX_PIXELS: int = 1_200_000
    IMAGE_FEATURE_PIPELINE_ENABLED: bool = False
    IMAGE_VL_CHAT_URL: str = "http://127.0.0.1:50019/v1/chat/completions"
    IMAGE_VL_CHAT_MODEL: str = "qwen3.5-122b"
    IMAGE_VL_EMBED_URL: str = "http://127.0.0.1:16703/v1/embeddings"
    IMAGE_VL_EMBED_MODEL: str = "qwen3-vl-embed"
    IMAGE_VL_EMBED_MODEL_VERSION: str = ""
    IMAGE_TEXT_EMBED_URL: str = "http://127.0.0.1:16703/v1/embeddings"
    IMAGE_TEXT_EMBED_MODEL: str = "qwen3-embed"
    IMAGE_TEXT_EMBED_MODEL_VERSION: str = ""
    IMAGE_MODEL_API_KEY: str = ""
    IMAGE_VL_CHAT_API_KEY: str = ""
    IMAGE_VL_EMBED_API_KEY: str = ""
    IMAGE_TEXT_EMBED_API_KEY: str = ""
    IMAGE_VL_CHAT_MAX_TOKENS: int = 5000
    IMAGE_VL_TEMPERATURE: float = 0.2
    IMAGE_FEATURE_TIMEOUT: int = 45
    IMAGE_FEATURE_TOTAL_TIMEOUT_SEC: int = 120
    IMAGE_FEATURE_MAX_RETRIES: int = 0
    IMAGE_FEATURE_STAGE_QUEUE_WAIT_REPORT_LIMIT: int = 200
    IMAGE_VL_RESPONSE_PREVIEW_CHARS: int = 800
    IMAGE_VL_CONTEXT_MAX_CHARS: int = 500
    IMAGE_CONTEXT_BEFORE_PARAGRAPHS: int = 1
    IMAGE_CONTEXT_AFTER_PARAGRAPHS: int = 2
    IMAGE_CONTEXT_CLEAN_ENABLED: bool = False
    IMAGE_DEFAULT_CAPTION: str = "No caption"
    # Pillow decoding, pHash, resizing and JPEG encoding are CPU-bound. Keep
    # them off the asyncio event loop and bound their shared worker pressure.
    # Keep CPU-bound image decode/resize/pHash ahead of the model queue. The
    # effective model limit remains the separate shared gateway semaphore.
    IMAGE_PREPROCESS_CONCURRENCY: int = 32
    IMAGE_MILVUS_WRITE_ENABLED: bool = True
    IMAGE_DELETE_IMAGES_ENABLED: bool = False
    # Preserve the historical image field layout. V2 changes the primary-key
    # semantics (id=image id, doc_id=document foreign key), not the schema.
    IMAGE_VECTOR_SCHEMA_MODE: str = "history_compatible_v2"
    # visual_embed is the primary image representation and is generated by
    # the configured qwen3-vl-embed model. The history-compatible fields
    # text_embed, visual_text_embed and text_sparse_vec remain placeholders.
    IMAGE_VISUAL_EMBED_ENABLED: bool = True
    IMAGE_VISIBILITY_CHECK_ENABLED: bool = True
    IMAGE_VISIBILITY_STRICT: bool = True
    IMAGE_VISIBILITY_RETRIES: int = 3
    IMAGE_VISIBILITY_RETRY_DELAY_SEC: float = 1.0
    IMAGE_VISIBILITY_CHECK_LIMIT: int = 3
    IMAGE_VISIBILITY_CHECK_TIMEOUT_SEC: int = 10
    IMAGE_VL_CHAT_CONCURRENCY: int = 32
    # The visual model benchmark is highest and most stable at 32 in-flight
    # requests. Higher values increase provider queueing without increasing
    # throughput.
    IMAGE_VL_EMBED_CONCURRENCY: int = 32
    IMAGE_VL_EMBED_BATCH_SIZE: int = 16
    IMAGE_TEXT_EMBED_CONCURRENCY: int = 20
    # Dedicated text-input budget for the image lane. This overrides the
    # generic worker embedding budget without changing metadata/fulltext.
    IMAGE_TEXT_EMBED_RATE: float = 100.0
    IMAGE_TEXT_EMBED_BATCH_SIZE: int = 32
    # A small coalescing window is essential here: caption/context providers
    # arrive from separate image tasks and otherwise become mostly single-item
    # HTTP requests, starving the visual lane with avoidable gateway latency.
    IMAGE_TEXT_EMBED_BATCH_DELAY_MS: int = 50
    IMAGE_TEXT_EMBED_MAX_IN_FLIGHT_BATCHES: int = 4
    IMAGE_MILVUS_WRITE_CONCURRENCY: int = 8
    IMAGE_MILVUS_WRITE_BATCH_SIZE: int = 64
    IMAGE_MILVUS_WRITE_FLUSH_INTERVAL_MS: int = 500
    IMAGE_MILVUS_WRITE_MODE: str = "upsert"
    # The direct Kafka image worker keeps this many Kafka micro-batches in
    # flight. Commits remain FIFO, so a later successful batch never skips an
    # earlier failed one.
    IMAGE_DIRECT_PIPELINE_INFLIGHT_BATCHES: int = 2
    # Direct Kafka image consumers can skip stale-image cleanup during an
    # initial replay into an empty collection, then enable it for increments.
    IMAGE_DIRECT_STALE_CLEANUP_ENABLED: bool = True
    IMAGE_HTTP_POOL_MAX_CONNECTIONS: int = 512
    IMAGE_HTTP_POOL_MAX_KEEPALIVE: int = 64
    IMAGE_MODEL_HTTP_POOL_MAX_CONNECTIONS: int = 64
    IMAGE_MODEL_HTTP_POOL_MAX_KEEPALIVE: int = 32
    # One total budget protects the shared embedding endpoint when visual and
    # optional text requests run in the same worker. Visual requests receive
    # priority inside this budget.
    IMAGE_MODEL_TOTAL_CONCURRENCY: int = 32
    IMAGE_MMD_CONTEXT_CACHE_SIZE: int = 2048
    IMAGE_MMD_CONTEXT_CACHE_TTL_SEC: int = 1800
    IMAGE_RUNTIME_ATTACHMENT_MODE: str = "full"
    IMAGE_SUCCESS_ARTIFACT_MODE: str = "summary"
    IMAGE_SUCCESS_RESULT_MODE: str = "status_only"
    IMAGE_STAGE_TIMING_REPORT_MODE: str = "summary"
    IMAGE_STAGE_TIMING_REPORT_LIMIT: int = 200
    IMAGE_MODEL_INPUT_FORMAT: str = "jpeg"
    IMAGE_MODEL_INPUT_MAX_PIXELS: int = 1_200_000
    IMAGE_MODEL_INPUT_MAX_SIDE: int = 1280
    IMAGE_MODEL_INPUT_JPEG_QUALITY: int = 82
    IMAGE_MODEL_INPUT_MIN_JPEG_QUALITY: int = 70
    IMAGE_MODEL_INPUT_MAX_BYTES: int = 512_000
    INIT_VECTOR_COLLECTIONS_ON_STARTUP: bool = False
    ENABLE_IMAGE_PROCESSING: bool = True
    LIVE_E2E_NO_DELETE: bool = False

    # ========== Ops Service ==========
    OPS_SERVICE_PORT: int = 8710
    OPS_PUBLISH_ENABLED: bool = True
    OPS_CLEANUP_ENABLED: bool = True
    OPS_INSPECTION_ENABLED: bool = True
    OPS_BACKFILL_ENABLED: bool = False
    OPS_METADATA_BACKFILL_ENABLED: bool = False
    OPS_METADATA_BACKFILL_INTERVAL_HOURS: float = 24.0
    OPS_METADATA_BACKFILL_STARTUP_RUN_ENABLED: bool = False
    OPS_METADATA_BACKFILL_WINDOW_ID: str = ""
    OPS_METADATA_BACKFILL_BATCH_SIZE: int = 16
    OPS_METADATA_BACKFILL_LIMIT: int = 0
    OPS_FULLTEXT_BACKFILL_ENABLED: bool = False
    OPS_FULLTEXT_BACKFILL_INTERVAL_HOURS: float = 1.0
    OPS_FULLTEXT_BACKFILL_STARTUP_RUN_ENABLED: bool = False
    OPS_FULLTEXT_BACKFILL_WINDOW_ID: str = ""
    OPS_FULLTEXT_BACKFILL_BATCH_SIZE: int = 16
    OPS_FULLTEXT_BACKFILL_LIMIT: int = 0
    OPS_FULLTEXT_BACKFILL_DRAIN_TIMEOUT_SECONDS: int = 3600
    OPS_STARTUP_PUBLISH_ENABLED: bool = False
    OPS_STARTUP_PUBLISH_RECORD_LIMIT: int = 0
    OPS_SYNC_HEALTH_URL: str = "http://127.0.0.1:8700/health"
    OPS_PUBLISH_SCHEDULE: str = "0 0 * * *"
    OPS_CLEANUP_SCHEDULE: str = "0 0 * * *"
    OPS_INSPECTION_SCHEDULE: str = "0 4 * * 1"
    OPS_BACKFILL_SCHEDULE: str = "0 */4 * * *"
    OPS_BACKFILL_WINDOW_ID: str = ""
    OPS_BACKFILL_BATCH_SIZE: int = 16
    OPS_BACKFILL_LIMIT: int = 0
    OPS_BACKFILL_FULLTEXT_DRAIN_TIMEOUT_SECONDS: int = 3600

    @model_validator(mode="after")
    def _materialize_auto_collection_names(self) -> "Settings":
        """Turn empty target variables into stable names for this worker run."""
        if not self.COLLECTION_AUTO_CREATE_ON_EMPTY:
            return self
        suffix = _AUTO_COLLECTION_RUN_ID
        if not str(self.DOCUMENT_COLLECTION_NAME or "").strip():
            generated = f"scientific_documents_auto_{suffix}"
            self.DOCUMENT_COLLECTION_NAME = generated
            # Older code paths still read MILVUS_COLLECTION directly.
            self.MILVUS_COLLECTION = generated
        if not str(self.FULLTEXT_CHUNK_COLLECTION_NAME or "").strip():
            generated = f"scientific_fulltext_chunks_auto_{suffix}"
            self.FULLTEXT_CHUNK_COLLECTION_NAME = generated
            self.FULLTEXT_CHUNK_COLLECTION = generated
        if not str(self.IMAGE_COLLECTION_NAME or "").strip():
            self.IMAGE_COLLECTION_NAME = f"scientific_images_auto_{suffix}"
        return self

    @property
    def dp_ignore_set(self) -> set:
        return {f.strip() for f in self.DP_IGNORE_FIELDS.split(",") if f.strip()}

    @property
    def dynamic_json_set(self) -> set:
        return {f.strip() for f in self.DYNAMIC_JSON_FIELDS.split(",") if f.strip()}

    @property
    def resolved_document_collection_name(self) -> str:
        return str(self.DOCUMENT_COLLECTION_NAME or self.MILVUS_COLLECTION).strip()

    @property
    def resolved_fulltext_chunk_collection_name(self) -> str:
        return str(self.FULLTEXT_CHUNK_COLLECTION_NAME or self.FULLTEXT_CHUNK_COLLECTION).strip()

    _env_file_path = Path(__file__).resolve().parent.parent / ".env"

    model_config = {
        "env_file": str(_env_file_path),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# 全局单例
settings = Settings()
