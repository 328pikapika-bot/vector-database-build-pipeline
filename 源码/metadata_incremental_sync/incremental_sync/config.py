"""Environment-only configuration; no connection credentials in source control."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TOPIC = "VECTOR_JOURNAL_ARTICLE_METADATA_INCREMENTAL_IDS"
DEFAULT_GROUP = "nstl_metadata_incremental_sync"


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


@dataclass(frozen=True)
class Settings:
    mode: str
    kafka_brokers: str
    kafka_topic: str
    kafka_group: str
    kafka_security_protocol: str
    kafka_sasl_mechanism: str
    kafka_username: str
    kafka_password: str
    solr_url: str
    embedding_url: str
    embedding_model: str
    embedding_key: str
    milvus_uri: str
    milvus_db: str
    milvus_collection: str
    milvus_token: str
    state_dir: Path
    log_dir: Path
    request_timeout: float
    retry_wait_sec: float
    idle_window_sec: int
    vector_update_verified: bool
    mongo_uri: str = ""
    mongo_db: str = "nstl_vector_sync_scientific_prod"
    mongo_failure_collection: str = "metadata_incremental_sync_failures"
    max_attempts: int = 3
    batch_size: int = 50
    batch_wait_sec: float = 0.25

    @classmethod
    def from_env(cls) -> "Settings":
        if os.getenv("MILVUS_TARGETS_JSON", "").strip():
            raise ValueError("multi-target mode is not implemented; cross-Milvus atomic writes are unavailable")
        mode = os.getenv("SYNC_MODE", "check").strip().lower()
        if mode not in {"check", "run"}:
            raise ValueError("SYNC_MODE must be check or run")
        topic = os.getenv("KAFKA_INCREMENTAL_TOPIC", DEFAULT_TOPIC).strip()
        group = os.getenv("KAFKA_INCREMENTAL_GROUP_ID", DEFAULT_GROUP).strip()
        if not topic or not group:
            raise ValueError("Kafka topic and group must not be empty")
        return cls(
            mode=mode,
            kafka_brokers=_required("KAFKA_BOOTSTRAP_SERVERS"),
            kafka_topic=topic,
            kafka_group=group,
            kafka_security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
            kafka_sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
            kafka_username=os.getenv("KAFKA_SASL_USERNAME", ""),
            kafka_password=os.getenv("KAFKA_SASL_PASSWORD", ""),
            solr_url=_required("SOLR_PAPER_URL").rstrip("/"),
            embedding_url=_required("EMBEDDING_API_URL"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3-embed"),
            embedding_key=os.getenv("EMBEDDING_API_KEY", ""),
            milvus_uri=_required("MILVUS_URI"),
            milvus_db=os.getenv("MILVUS_DATABASE", "default"),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "scientific_documents_v2"),
            milvus_token=os.getenv("MILVUS_TOKEN", ""),
            state_dir=Path(os.getenv("SYNC_STATE_DIR", "/app/state")),
            log_dir=Path(os.getenv("SYNC_LOG_DIR", "/app/logs")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT_SEC", "60")),
            retry_wait_sec=float(os.getenv("RETRY_WAIT_SEC", "30")),
            idle_window_sec=int(os.getenv("IDLE_WINDOW_SEC", "300")),
            vector_update_verified=os.getenv("VECTOR_PARTIAL_UPDATE_VERIFIED", "0") == "1",
            mongo_uri=os.getenv("MONGO_URI", ""),
            mongo_db=os.getenv("MONGO_DB", "nstl_vector_sync_scientific_prod"),
            mongo_failure_collection=os.getenv("MONGO_FAILURE_COLLECTION", "metadata_incremental_sync_failures"),
            max_attempts=max(1, int(os.getenv("MAX_ATTEMPTS", "3"))),
            batch_size=max(1, min(1000, int(os.getenv("SYNC_BATCH_SIZE", "200")))),
            batch_wait_sec=max(0.0, min(2.0, float(os.getenv("SYNC_BATCH_WAIT_SEC", "0.25")))),
        )

    def kafka_config(self) -> dict[str, object]:
        result: dict[str, object] = {
            "bootstrap.servers": self.kafka_brokers,
            "group.id": self.kafka_group,
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 900000,
            "security.protocol": self.kafka_security_protocol,
        }
        if self.kafka_security_protocol.startswith("SASL"):
            if not self.kafka_username or not self.kafka_password:
                raise ValueError("Kafka SASL credentials are required")
            result.update({
                "sasl.mechanism": self.kafka_sasl_mechanism,
                "sasl.username": self.kafka_username,
                "sasl.password": self.kafka_password,
            })
        return result
