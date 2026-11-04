"""运行期同步状态、失败任务和批处理结果模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Optional


class SyncOverallStatus(str, Enum):
    PROCESSING = "PROCESSING"
    CORE_SUCCESS = "CORE_SUCCESS"
    SUCCESS = "SUCCESS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILURE = "FAILURE"
    SKIPPED_STALE = "SKIPPED_STALE"
    DELETED = "DELETED"


class StageExecutionStatus(str, Enum):
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED_EMPTY = "SKIPPED_EMPTY"
    SKIPPED_NOT_PRESENT = "SKIPPED_NOT_PRESENT"
    SKIPPED_STALE = "SKIPPED_STALE"


class FailureStatus(str, Enum):
    PENDING = "PENDING"
    RETRYING = "RETRYING"
    DONE = "DONE"
    SKIPPED_STALE = "SKIPPED_STALE"
    GAVE_UP = "GAVE_UP"
    HELD_HISTORY = "HELD_HISTORY"


class ErrorType(str, Enum):
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"


class ScopeType(str, Enum):
    EVENT = "event"
    DOCUMENT = "document"
    ARTICLE = "article"
    FULLTEXT_FILE = "fulltext_file"
    CHUNK = "chunk"
    IMAGE = "image"
    MILVUS_BATCH = "milvus_batch"


class ReplayMode(str, Enum):
    FROM_RAW_EVENT = "FROM_RAW_EVENT"
    FROM_PARSED_DOCUMENT = "FROM_PARSED_DOCUMENT"
    FROM_MMD_TEXT = "FROM_MMD_TEXT"
    FROM_CHUNK_PLAN = "FROM_CHUNK_PLAN"
    FROM_IMAGE_MANIFEST = "FROM_IMAGE_MANIFEST"
    FROM_DELETE_CONTEXT = "FROM_DELETE_CONTEXT"


class ArtifactType(str, Enum):
    RAW_EVENT_JSON = "RAW_EVENT_JSON"
    RAW_EVENT_BYTES_META = "RAW_EVENT_BYTES_META"
    PARSED_DOCUMENT_JSON = "PARSED_DOCUMENT_JSON"
    PARSED_ARTICLE_JSON = "PARSED_ARTICLE_JSON"
    FULLTEXT_FILE_META = "FULLTEXT_FILE_META"
    MMD_RAW_TEXT = "MMD_RAW_TEXT"
    FULLTEXT_PLAIN_TEXT = "FULLTEXT_PLAIN_TEXT"
    CHUNK_PLAN_JSON = "CHUNK_PLAN_JSON"
    IMAGE_MANIFEST_JSON = "IMAGE_MANIFEST_JSON"
    IMAGE_FEATURE_SUMMARY_JSON = "IMAGE_FEATURE_SUMMARY_JSON"
    EMBED_INPUT_JSON = "EMBED_INPUT_JSON"
    MILVUS_WRITE_SUMMARY = "MILVUS_WRITE_SUMMARY"
    ERROR_CONTEXT_JSON = "ERROR_CONTEXT_JSON"


class SyncStage(str, Enum):
    KAFKA_DECODE = "KAFKA_DECODE"
    KAFKA_VALIDATE = "KAFKA_VALIDATE"
    SOURCE_RESOLVE = "SOURCE_RESOLVE"
    ARTICLE_NORMALIZE = "ARTICLE_NORMALIZE"
    IDEMPOTENCY_CHECK = "IDEMPOTENCY_CHECK"
    DOC_EMBED_TITLE = "DOC_EMBED_TITLE"
    DOC_EMBED_ABSTRACT = "DOC_EMBED_ABSTRACT"
    DOC_EMBED_KEYWORD = "DOC_EMBED_KEYWORD"
    MILVUS_WRITE_DOCUMENT = "MILVUS_WRITE_DOCUMENT"
    FULLTEXT_DISCOVER = "FULLTEXT_DISCOVER"
    FULLTEXT_DOWNLOAD = "FULLTEXT_DOWNLOAD"
    FULLTEXT_PARSE_MMD = "FULLTEXT_PARSE_MMD"
    FULLTEXT_BUILD_CHUNKS = "FULLTEXT_BUILD_CHUNKS"
    FULLTEXT_EMBED_CHUNKS = "FULLTEXT_EMBED_CHUNKS"
    MILVUS_WRITE_CHUNKS = "MILVUS_WRITE_CHUNKS"
    IMAGE_DISCOVER = "IMAGE_DISCOVER"
    IMAGE_DOWNLOAD = "IMAGE_DOWNLOAD"
    IMAGE_EMBED_VISUAL = "IMAGE_EMBED_VISUAL"
    IMAGE_EMBED_CAPTION = "IMAGE_EMBED_CAPTION"
    MILVUS_WRITE_IMAGES = "MILVUS_WRITE_IMAGES"
    DELETE_DOCUMENT = "DELETE_DOCUMENT"
    DELETE_CHUNKS = "DELETE_CHUNKS"
    DELETE_IMAGES = "DELETE_IMAGES"
    VERIFY_WRITE = "VERIFY_WRITE"
    MONGO_STATUS_WRITE = "MONGO_STATUS_WRITE"
    COMPENSATOR_RETRY = "COMPENSATOR_RETRY"


@dataclass(slots=True)
class SourceEventMeta:
    topic: str
    partition: int
    offset: int
    event_fingerprint: str


@dataclass(slots=True)
class ConsumedRecord:
    topic: str
    partition: int
    offset: int
    raw_bytes: bytes
    raw_text: str
    event_fingerprint: str
    parsed_payload: Optional[dict[str, Any]] = None
    decode_error: Optional[str] = None
    validation_error: Optional[str] = None

    @property
    def source_event_meta(self) -> SourceEventMeta:
        return SourceEventMeta(
            topic=self.topic,
            partition=self.partition,
            offset=self.offset,
            event_fingerprint=self.event_fingerprint,
        )

    @property
    def is_valid(self) -> bool:
        return (
            self.parsed_payload is not None
            and self.decode_error is None
            and self.validation_error is None
        )


@dataclass(slots=True)
class DocumentContext:
    doc_id: str
    record_update_time: str
    sub_doc_type: str
    event_type: str
    payload: dict[str, Any]
    article_raw: dict[str, Any]
    article_index: int
    source_event_meta: SourceEventMeta
    event_fingerprint: str

    @property
    def article_scope_id(self) -> str:
        article_id = self.article_raw.get("id") or self.article_raw.get("sysId") or self.doc_id
        return f"{article_id}:{self.article_index}"


@dataclass(slots=True)
class EmbeddingItemResult:
    success: bool
    vector: Optional[list[float]]
    error_type: Optional[ErrorType] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    skipped_empty: bool = False
    queue_wait_ms: float = 0.0


@dataclass(slots=True)
class FailureRef:
    failure_key: str
    failure_stage: SyncStage
    scope_type: ScopeType
    scope_id: str


@dataclass(slots=True)
class ProcessOutcome:
    doc_id: Optional[str]
    record_update_time: Optional[str]
    final_status: str
    commit_safe: bool
    sub_doc_type: Optional[str] = None
    event_type: Optional[str] = None
    pending_failure_refs: list[str] = field(default_factory=list)
    failure_persisted: bool = False
    message: str = ""

    @property
    def is_success(self) -> bool:
        return self.final_status in {
            SyncOverallStatus.CORE_SUCCESS.value,
            SyncOverallStatus.SUCCESS.value,
            SyncOverallStatus.DELETED.value,
        }

    @property
    def is_skip(self) -> bool:
        return self.final_status == SyncOverallStatus.SKIPPED_STALE.value

    @property
    def is_failure(self) -> bool:
        return not self.is_success and not self.is_skip


@dataclass(slots=True)
class BatchProcessingResult:
    outcomes: list[ProcessOutcome] = field(default_factory=list)
    skip_duplicates: int = 0

    @property
    def success(self) -> int:
        return sum(1 for item in self.outcomes if item.is_success)

    @property
    def skip(self) -> int:
        return sum(1 for item in self.outcomes if item.is_skip) + self.skip_duplicates

    @property
    def fail(self) -> int:
        return sum(1 for item in self.outcomes if item.is_failure)

    @property
    def total(self) -> int:
        return self.success + self.skip + self.fail

    @property
    def commit_safe(self) -> bool:
        return all(item.commit_safe for item in self.outcomes)


def utcnow() -> datetime:
    """统一使用 UTC aware 时间。"""
    return datetime.now(timezone.utc)


def normalize_record_update_time(value: Any) -> str:
    """将更新时间统一标准化为 ISO 字符串。"""
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    if not text:
        return ""
    parsed = parse_record_update_time(text)
    if parsed is None:
        return text
    return parsed.astimezone(timezone.utc).isoformat()


def parse_record_update_time(value: Any) -> Optional[datetime]:
    """尽量兼容毫秒时间戳、秒时间戳和 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e12:
            number = number / 1000.0
        return datetime.fromtimestamp(number, tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return parse_record_update_time(int(text))
    try:
        candidate = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def sha256_hex(raw: bytes | str) -> str:
    """生成十六进制 sha256。"""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return sha256(raw).hexdigest()
