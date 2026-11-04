"""
NSTL 向量库同步服务 - Prometheus 指标
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# ========== Counters ==========

kafka_messages_total = Counter(
    "nstl_kafka_messages_total",
    "Total Kafka messages processed",
    ["status"],  # success, skip, fail
)

milvus_ops_total = Counter(
    "nstl_milvus_ops_total",
    "Total Milvus operations",
    ["op", "status"],  # op: insert/delete, status: success/fail
)

embedding_requests_total = Counter(
    "nstl_embedding_requests_total",
    "Total embedding API requests",
    ["status"],  # success, 4xx, 5xx, timeout
)

idempotency_skip_total = Counter(
    "nstl_idempotency_skip_total",
    "Total events skipped by idempotency check",
)

batch_flush_total = Counter(
    "nstl_batch_flush_total",
    "Total batch flushes",
)

compensator_rounds_total = Counter(
    "nstl_compensator_rounds_total",
    "Total compensator scan rounds",
)

gave_up_total = Counter(
    "nstl_gave_up_total",
    "Total events given up after max retries",
)

verify_mismatch_total = Counter(
    "nstl_verify_mismatch_total",
    "Total write verification mismatches",
)

worker_task_claims_total = Counter(
    "nstl_worker_task_claims_total",
    "Total async worker task claim attempts",
    ["worker_type", "result"],
)

worker_task_processed_total = Counter(
    "nstl_worker_task_processed_total",
    "Total async worker task process results",
    ["worker_type", "status", "error_code"],
)

diagnostics_generation_total = Counter(
    "nstl_diagnostics_generation_total",
    "Total Admin diagnostics generation attempts",
    ["status"],
)

# ========== Histograms ==========

embedding_latency_seconds = Histogram(
    "nstl_embedding_latency_seconds",
    "Embedding API request latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

milvus_op_latency_seconds = Histogram(
    "nstl_milvus_op_latency_seconds",
    "Milvus operation latency in seconds",
    ["op"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

batch_process_latency_seconds = Histogram(
    "nstl_batch_process_latency_seconds",
    "Batch processing latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

image_stage_latency_seconds = Histogram(
    "nstl_image_stage_latency_seconds",
    "Image pipeline stage latency in seconds",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

image_mmd_cache_total = Counter(
    "nstl_image_mmd_cache_total",
    "Image MMD context cache lookups",
    ["result"],  # hit, miss, skipped
)

image_milvus_batch_flush_total = Counter(
    "nstl_image_milvus_batch_flush_total",
    "Image Milvus batch flush results",
    ["status"],
)

image_pipeline_stage_runs_total = Counter(
    "nstl_image_pipeline_stage_runs_total",
    "Image staged pipeline execution count",
    ["stage", "status"],
)

image_pipeline_stage_queue_wait_seconds = Histogram(
    "nstl_image_pipeline_stage_queue_wait_seconds",
    "Time an image item spends waiting in a stage queue",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ========== Gauges ==========

kafka_consumer_lag = Gauge(
    "nstl_kafka_consumer_lag",
    "Kafka consumer lag",
)

failure_pending_count = Gauge(
    "nstl_failure_pending_count",
    "Number of pending failure records in MongoDB",
)

runtime_paused = Gauge(
    "nstl_runtime_paused",
    "Runtime control paused flag, 1 when paused",
)

runtime_draining = Gauge(
    "nstl_runtime_draining",
    "Runtime control draining flag, 1 when draining",
)

runtime_inflight_batches = Gauge(
    "nstl_runtime_inflight_batches",
    "Number of batches currently being processed",
)

worker_loop_running = Gauge(
    "nstl_worker_loop_running",
    "Async worker loop running flag, 1 when running",
    ["worker_type"],
)

worker_control_paused = Gauge(
    "nstl_worker_control_paused",
    "Async worker control paused flag, 1 when paused",
    ["worker_type"],
)

worker_control_draining = Gauge(
    "nstl_worker_control_draining",
    "Async worker control draining flag, 1 when draining",
    ["worker_type"],
)

worker_control_inflight_runs = Gauge(
    "nstl_worker_control_inflight_runs",
    "Async worker run_once or scan rounds currently in progress",
    ["worker_type"],
)

worker_task_backlog = Gauge(
    "nstl_worker_task_backlog",
    "Async worker task backlog by bounded task status",
    ["worker_type", "status"],
)

image_pipeline_stage_inflight = Gauge(
    "nstl_image_pipeline_stage_inflight",
    "Current in-flight items in image staged pipeline",
    ["stage"],
)

image_pipeline_stage_backlog = Gauge(
    "nstl_image_pipeline_stage_backlog",
    "Current queued backlog in image staged pipeline",
    ["stage"],
)

dependency_readiness = Gauge(
    "nstl_dependency_readiness",
    "Dependency readiness flag, 1 when ready",
    ["dependency"],
)

document_completeness = Gauge(
    "nstl_document_completeness",
    "Number of documents in Milvus by sync_status",
    ["sync_status"],
)

status_sync_latency_seconds = Histogram(
    "nstl_status_sync_latency_seconds",
    "Latency of MongoDB-to-Milvus sync_status mirror operations",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


def get_metrics_response():
    """生成 Prometheus 格式的指标响应"""
    return generate_latest()
