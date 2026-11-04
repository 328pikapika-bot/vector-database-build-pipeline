"""Default-off live smoke gate and redaction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.config import settings


GLOBAL_LIVE_SMOKE_ENV = "RUN_LIVE_SMOKE"
LIVE_SMOKE_COMPONENT_ENVS: dict[str, str] = {
    "mongo": "RUN_LIVE_MONGO",
    "milvus": "RUN_LIVE_MILVUS",
    "kafka": "RUN_LIVE_KAFKA",
    "embedding": "RUN_LIVE_EMBEDDING",
    "vl": "RUN_LIVE_VL",
}
LIVE_SMOKE_DOC_ID_ENV = "LIVE_SMOKE_DOC_ID"
LIVE_SMOKE_SYS_ID_ENV = "LIVE_SMOKE_SYS_ID"
DEFAULT_SAFE_DOC_ID_PREFIX = "SMOKE_DOC_"
DEFAULT_SAFE_SYS_ID_PREFIX = "SMOKE_SYS_"
DEFAULT_PREFLIGHT_CHUNK_COLLECTION = "scientific_fulltext_chunks"
DEFAULT_PREFLIGHT_IMAGE_COLLECTION = "scientific_images"

_ENABLED_VALUE = "1"
_REDACTED = "<redacted>"
_REDACTED_BYTES = "<redacted-bytes>"
_SENSITIVE_KEY_PARTS = (
    "PASSWORD",
    "TOKEN",
    "API_KEY",
    "AUTHORIZATION",
    "SECRET",
    "MONGO_URI",
    "CONNECTION_STRING",
)
_RAW_BYTES_KEY_PARTS = (
    "IMAGEBYTES",
    "IMAGE_BYTES",
    "RAW_BYTES",
    "PAYLOAD_GZIP",
)
_PLACEHOLDER_VALUES = {"", "your-api-key-here", "change-me", "changeme", "placeholder"}
_VL_PROVIDER_REQUIRED_ENV_NAMES = (
    "IMAGE_VL_EMBED_URL",
    "IMAGE_VL_EMBED_MODEL",
    "IMAGE_TEXT_EMBED_URL",
    "IMAGE_TEXT_EMBED_MODEL",
    "IMAGE_FEATURE_TIMEOUT",
    "IMAGE_FEATURE_MAX_RETRIES",
)


@dataclass(frozen=True)
class LiveSmokeGate:
    component: str
    global_env: str
    component_env: str
    global_enabled: bool
    component_enabled: bool
    enabled: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_live_smoke_gate(component: str, environ: Mapping[str, str] | None = None) -> LiveSmokeGate:
    """Return the gate state for a live smoke component without side effects."""
    normalized = _normalize_component(component)
    env = environ if environ is not None else __import__("os").environ
    component_env = LIVE_SMOKE_COMPONENT_ENVS[normalized]
    global_enabled = str(env.get(GLOBAL_LIVE_SMOKE_ENV, "")).strip() == _ENABLED_VALUE
    component_enabled = str(env.get(component_env, "")).strip() == _ENABLED_VALUE
    enabled = global_enabled and component_enabled

    skip_reason: str | None = None
    if not global_enabled:
        skip_reason = f"{GLOBAL_LIVE_SMOKE_ENV}=1 is required to run live {normalized} smoke"
    elif not component_enabled:
        skip_reason = f"{component_env}=1 is required to run live {normalized} smoke"

    return LiveSmokeGate(
        component=normalized,
        global_env=GLOBAL_LIVE_SMOKE_ENV,
        component_env=component_env,
        global_enabled=global_enabled,
        component_enabled=component_enabled,
        enabled=enabled,
        skip_reason=skip_reason,
    )


def live_smoke_skip_reason(component: str, environ: Mapping[str, str] | None = None) -> str | None:
    return get_live_smoke_gate(component, environ=environ).skip_reason


def is_live_smoke_enabled(component: str, environ: Mapping[str, str] | None = None) -> bool:
    return get_live_smoke_gate(component, environ=environ).enabled


def mask_secret_value(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _REDACTED_BYTES
    return _REDACTED


def sanitize_smoke_payload(payload: Any) -> Any:
    """Recursively redact secrets and raw bytes before logging or returning smoke details."""
    if isinstance(payload, Mapping):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_text = str(key)
            if _is_raw_bytes_key(key_text):
                sanitized[key_text] = _REDACTED_BYTES
            elif _is_sensitive_key(key_text):
                sanitized[key_text] = mask_secret_value(value)
            else:
                sanitized[key_text] = sanitize_smoke_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_smoke_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(sanitize_smoke_payload(item) for item in payload)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return _REDACTED_BYTES
    return payload


def is_safe_smoke_id(value: Any, *, allowed_prefixes: Sequence[str] | None = None) -> bool:
    """Return whether a provided staging smoke id is explicitly marked safe."""
    text = str(value or "").strip()
    if not text:
        return False
    prefixes = tuple(allowed_prefixes or (DEFAULT_SAFE_DOC_ID_PREFIX, DEFAULT_SAFE_SYS_ID_PREFIX))
    return any(text.startswith(prefix) for prefix in prefixes)


def build_staging_smoke_plan(
    components: Iterable[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    safe_doc_id: str | None = None,
    safe_sys_id: str | None = None,
    expect_document: bool | None = None,
    expect_chunks: bool | None = None,
    expect_images: bool | None = None,
    display_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a staging smoke plan without constructing live dependency clients."""
    env = environ if environ is not None else __import__("os").environ
    normalized_components = [
        _normalize_component(component)
        for component in (components if components is not None else LIVE_SMOKE_COMPONENT_ENVS.keys())
    ]
    gates = {
        component: get_live_smoke_gate(component, environ=env).to_dict()
        for component in normalized_components
    }
    doc_id_source = "argument" if safe_doc_id is not None else "environment"
    sys_id_source = "argument" if safe_sys_id is not None else "environment"
    doc_id_value = safe_doc_id if safe_doc_id is not None else env.get(LIVE_SMOKE_DOC_ID_ENV)
    sys_id_value = safe_sys_id if safe_sys_id is not None else env.get(LIVE_SMOKE_SYS_ID_ENV)

    safe_ids = {
        "doc_id": _safe_id_status(
            doc_id_value,
            source=doc_id_source,
            env_name=LIVE_SMOKE_DOC_ID_ENV,
            allowed_prefixes=(DEFAULT_SAFE_DOC_ID_PREFIX,),
            required=True,
        ),
        "sys_id": _safe_id_status(
            sys_id_value,
            source=sys_id_source,
            env_name=LIVE_SMOKE_SYS_ID_ENV,
            allowed_prefixes=(DEFAULT_SAFE_SYS_ID_PREFIX, DEFAULT_SAFE_DOC_ID_PREFIX),
            required=False,
        ),
    }

    consistency_skip_reason = _build_consistency_skip_reason(gates=gates, safe_ids=safe_ids)
    display = {
        "components": normalized_components,
        "gates": gates,
        "safe_ids": safe_ids,
        "expectations": {
            "document": expect_document,
            "chunks": expect_chunks,
            "images": expect_images,
        },
        "details": display_details or {},
    }

    return {
        "live_calls_allowed": False,
        "planned_components": normalized_components,
        "components": gates,
        "skip_reasons": {
            component: gate["skip_reason"]
            for component, gate in gates.items()
            if gate.get("skip_reason")
        },
        "required_env_names": [
            GLOBAL_LIVE_SMOKE_ENV,
            *[LIVE_SMOKE_COMPONENT_ENVS[component] for component in normalized_components],
        ],
        "required_safe_id_env_names": {
            "doc_id": LIVE_SMOKE_DOC_ID_ENV,
            "sys_id": LIVE_SMOKE_SYS_ID_ENV,
        },
        "safe_ids": safe_ids,
        "expectations": {
            "document": expect_document,
            "chunks": expect_chunks,
            "images": expect_images,
        },
        "consistency": {
            "required_components": ["mongo", "milvus"],
            "ready": consistency_skip_reason is None,
            "skip_reason": consistency_skip_reason,
        },
        "display": sanitize_smoke_payload(display),
    }


def build_live_smoke_preflight(
    components: Iterable[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    safe_doc_id: str | None = None,
    safe_sys_id: str | None = None,
    expect_document: bool | None = None,
    expect_chunks: bool | None = None,
    expect_images: bool | None = None,
    document_collection: str | None = None,
    chunk_collection: str | None = None,
    image_collection: str | None = None,
    display_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-live preflight view for staging smoke readiness."""
    env = environ if environ is not None else __import__("os").environ
    plan = build_staging_smoke_plan(
        components,
        environ=env,
        safe_doc_id=safe_doc_id,
        safe_sys_id=safe_sys_id,
        expect_document=expect_document,
        expect_chunks=expect_chunks,
        expect_images=expect_images,
        display_details=display_details,
    )
    resolved_collections = {
        "document": str(document_collection or env.get("MILVUS_COLLECTION") or settings.MILVUS_COLLECTION).strip(),
        "chunk": str(chunk_collection or DEFAULT_PREFLIGHT_CHUNK_COLLECTION).strip(),
        "image": str(
            image_collection
            or env.get("IMAGE_COLLECTION_NAME")
            or settings.IMAGE_COLLECTION_NAME
            or DEFAULT_PREFLIGHT_IMAGE_COLLECTION
        ).strip(),
    }

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    components_result: dict[str, dict[str, Any]] = {}
    for component in plan["planned_components"]:
        gate = plan["components"][component]
        component_blockers = _gate_blockers(component, gate)
        component_warnings: list[dict[str, Any]] = []
        extra_blockers, extra_warnings, required_env_names = _component_preflight(component, env)
        component_blockers.extend(extra_blockers)
        component_warnings.extend(extra_warnings)
        blockers.extend(component_blockers)
        warnings.extend(component_warnings)
        components_result[component] = {
            "gate": gate,
            "required_env_names": required_env_names,
            "blockers": component_blockers,
            "warnings": component_warnings,
            "would_run": bool(gate.get("enabled")) and not component_blockers,
        }

    safe_id_blockers = _safe_id_blockers(plan["safe_ids"])
    collection_blockers = _collection_policy_blockers(resolved_collections)
    blockers.extend(safe_id_blockers)
    blockers.extend(collection_blockers)

    display = {
        "plan": plan["display"],
        "collections": resolved_collections,
        "details": display_details or {},
    }
    return {
        "success": True,
        "read_only": True,
        "mutation_allowed": False,
        "live_calls_allowed": False,
        "planned_components": plan["planned_components"],
        "components": components_result,
        "would_run": [
            component
            for component, details in components_result.items()
            if details["would_run"]
        ],
        "skip_reasons": plan["skip_reasons"],
        "preflight_blockers": blockers,
        "preflight_warnings": warnings,
        "has_blocker": bool(blockers),
        "safe_ids": plan["safe_ids"],
        "expectations": plan["expectations"],
        "collection_policy": {
            "collections": resolved_collections,
            "blockers": collection_blockers,
        },
        "display": sanitize_smoke_payload(display),
    }


def _normalize_component(component: str) -> str:
    normalized = str(component or "").strip().lower()
    if normalized not in LIVE_SMOKE_COMPONENT_ENVS:
        allowed = ", ".join(sorted(LIVE_SMOKE_COMPONENT_ENVS))
        raise ValueError(f"unsupported live smoke component: {component!r}; allowed: {allowed}")
    return normalized


def _safe_id_status(
    value: Any,
    *,
    source: str,
    env_name: str,
    allowed_prefixes: Sequence[str],
    required: bool,
) -> dict[str, Any]:
    provided = str(value or "").strip() != ""
    safe = is_safe_smoke_id(value, allowed_prefixes=allowed_prefixes)
    return {
        "provided": provided,
        "safe": safe,
        "required": required,
        "source": source if provided else "missing",
        "env_name": env_name,
        "accepted_prefixes": list(allowed_prefixes),
        "value": str(value).strip() if safe else None,
    }


def _gate_blockers(component: str, gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    if gate.get("enabled"):
        return []
    if not gate.get("global_enabled"):
        return [
            _blocker(
                "LIVE_SMOKE_GATE_DISABLED",
                component=component,
                message=f"{GLOBAL_LIVE_SMOKE_ENV}=1 is required before live {component} smoke can run.",
                required_env_names=[GLOBAL_LIVE_SMOKE_ENV],
            )
        ]
    return [
        _blocker(
            "COMPONENT_GATE_DISABLED",
            component=component,
            message=f"{gate.get('component_env')}=1 is required before live {component} smoke can run.",
            required_env_names=[str(gate.get("component_env"))],
        )
    ]


def _component_preflight(
    component: str,
    env: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if component == "mongo":
        required = ["MONGO_URI", "MONGO_DB"]
        return _missing_env_blockers(component, required, env), [], required
    if component == "milvus":
        required = ["MILVUS_HOST", "MILVUS_PORT", "MILVUS_COLLECTION"]
        blockers = _missing_env_blockers(component, required, env)
        warnings = []
        user_present = _has_env_value(env, "MILVUS_USER")
        password_present = _has_env_value(env, "MILVUS_PASSWORD")
        if user_present != password_present:
            blockers.append(
                _blocker(
                    "MILVUS_AUTH_PAIR_INCOMPLETE",
                    component=component,
                    message="MILVUS_USER and MILVUS_PASSWORD must be provided together when Milvus auth is used.",
                    required_env_names=["MILVUS_USER", "MILVUS_PASSWORD"],
                )
            )
        elif not user_present and not password_present:
            warnings.append(
                _warning(
                    "MILVUS_AUTH_NOT_PROVIDED",
                    component=component,
                    message="Milvus auth env values are absent; confirm staging allows unauthenticated smoke access.",
                    required_env_names=["MILVUS_USER", "MILVUS_PASSWORD"],
                )
            )
        return blockers, warnings, [*required, "MILVUS_USER", "MILVUS_PASSWORD"]
    if component == "kafka":
        required = ["KAFKA_BOOTSTRAP_SERVERS", "KAFKA_TOPIC", "KAFKA_GROUP_ID", "KAFKA_SECURITY_PROTOCOL"]
        blockers = _missing_env_blockers(component, required, env)
        warnings = []
        protocol = str(env.get("KAFKA_SECURITY_PROTOCOL", "")).strip().upper()
        if protocol in {"SASL_SSL", "SASL_PLAINTEXT"}:
            sasl_required = ["KAFKA_SASL_MECHANISM", "KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"]
            blockers.extend(_missing_env_blockers(component, sasl_required, env, code="KAFKA_SASL_CREDENTIALS_MISSING"))
            required = [*required, *sasl_required]
        elif protocol == "PLAINTEXT":
            warnings.append(
                _warning(
                    "KAFKA_PLAINTEXT_REQUIRES_POLICY_CONFIRMATION",
                    component=component,
                    message="Kafka security protocol is PLAINTEXT; confirm this is approved for staging smoke.",
                    required_env_names=["KAFKA_SECURITY_PROTOCOL"],
                )
            )
        return blockers, warnings, required
    if component == "embedding":
        required = [
            "EMBEDDING_API_URL",
            "EMBEDDING_MODEL",
            "EMBEDDING_API_KEY",
            "EMBEDDING_TIMEOUT",
            "EMBEDDING_MAX_RETRIES",
            "VECTOR_DIM",
        ]
        blockers = _missing_env_blockers(component, required, env)
        if _has_env_value(env, "EMBEDDING_API_KEY") and _is_placeholder_secret(env.get("EMBEDDING_API_KEY")):
            blockers.append(
                _blocker(
                    "EMBEDDING_API_KEY_MISSING",
                    component=component,
                    message="EMBEDDING_API_KEY must be explicitly set to a non-placeholder value.",
                    required_env_names=["EMBEDDING_API_KEY"],
                )
            )
        blockers.extend(_positive_int_blockers(component, env, ["EMBEDDING_TIMEOUT", "EMBEDDING_MAX_RETRIES"]))
        vector_dim = _parse_int_env(env, "VECTOR_DIM")
        if vector_dim is not None and vector_dim != settings.VECTOR_DIM:
            blockers.append(
                _blocker(
                    "EMBEDDING_VECTOR_DIMENSION_MISMATCH",
                    component=component,
                    message=f"VECTOR_DIM must match the expected text embedding dimension {settings.VECTOR_DIM}.",
                    required_env_names=["VECTOR_DIM"],
                )
            )
        return blockers, [], required
    if component == "vl":
        visual_embed_enabled = str(
            env.get("IMAGE_VISUAL_EMBED_ENABLED", settings.IMAGE_VISUAL_EMBED_ENABLED)
        ).strip().lower() in {"1", "true", "yes", "on"}
        required = [
            "IMAGE_TEXT_EMBED_URL",
            "IMAGE_TEXT_EMBED_MODEL",
            "IMAGE_FEATURE_TIMEOUT",
            "IMAGE_FEATURE_MAX_RETRIES",
        ]
        if visual_embed_enabled:
            required.extend(["IMAGE_VL_EMBED_URL", "IMAGE_VL_EMBED_MODEL"])
        blockers = _missing_env_blockers(component, required, env)
        has_vl_embed_key = any(
            _has_env_value(env, name)
            for name in ("IMAGE_VL_EMBED_API_KEY", "IMAGE_MODEL_API_KEY")
        )
        has_text_embed_key = any(
            _has_env_value(env, name)
            for name in ("IMAGE_TEXT_EMBED_API_KEY", "IMAGE_MODEL_API_KEY", "EMBEDDING_API_KEY")
        )
        if visual_embed_enabled and not has_vl_embed_key:
            blockers.append(
                _blocker(
                    "IMAGE_VL_EMBED_API_KEY_MISSING",
                    component=component,
                    message="IMAGE_VL_EMBED_API_KEY or IMAGE_MODEL_API_KEY must be explicitly set for image embedding.",
                    required_env_names=["IMAGE_VL_EMBED_API_KEY", "IMAGE_MODEL_API_KEY"],
                )
            )
        if not has_text_embed_key:
            blockers.append(
                _blocker(
                    "IMAGE_TEXT_EMBED_API_KEY_MISSING",
                    component=component,
                    message="IMAGE_TEXT_EMBED_API_KEY, IMAGE_MODEL_API_KEY, or EMBEDDING_API_KEY must be explicitly set for image text embedding.",
                    required_env_names=["IMAGE_TEXT_EMBED_API_KEY", "IMAGE_MODEL_API_KEY", "EMBEDDING_API_KEY"],
                )
            )
        if _has_env_value(env, "IMAGE_MODEL_API_KEY") and _is_placeholder_secret(env.get("IMAGE_MODEL_API_KEY")):
            blockers.append(
                _blocker(
                    "IMAGE_MODEL_API_KEY_MISSING",
                    component=component,
                    message="IMAGE_MODEL_API_KEY must be explicitly set to a non-placeholder value.",
                    required_env_names=["IMAGE_MODEL_API_KEY"],
                )
            )
        if visual_embed_enabled and _has_env_value(env, "IMAGE_VL_EMBED_API_KEY") and _is_placeholder_secret(env.get("IMAGE_VL_EMBED_API_KEY")):
            blockers.append(
                _blocker(
                    "IMAGE_VL_EMBED_API_KEY_MISSING",
                    component=component,
                    message="IMAGE_VL_EMBED_API_KEY must be explicitly set to a non-placeholder value.",
                    required_env_names=["IMAGE_VL_EMBED_API_KEY"],
                )
            )
        if _has_env_value(env, "IMAGE_TEXT_EMBED_API_KEY") and _is_placeholder_secret(env.get("IMAGE_TEXT_EMBED_API_KEY")):
            blockers.append(
                _blocker(
                    "IMAGE_TEXT_EMBED_API_KEY_MISSING",
                    component=component,
                    message="IMAGE_TEXT_EMBED_API_KEY must be explicitly set to a non-placeholder value.",
                    required_env_names=["IMAGE_TEXT_EMBED_API_KEY"],
                )
            )
        blockers.extend(
            _positive_int_blockers(
                component,
                env,
                ["IMAGE_FEATURE_TIMEOUT", "IMAGE_FEATURE_MAX_RETRIES", "IMAGE_VL_CHAT_MAX_TOKENS"],
            )
        )
        return blockers, [], required
    return [], [], []


def _safe_id_blockers(safe_ids: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    blockers = []
    doc_id = safe_ids.get("doc_id") or {}
    sys_id = safe_ids.get("sys_id") or {}
    if not doc_id.get("safe"):
        blockers.append(
            _blocker(
                "SAFE_DOC_ID_MISSING_OR_UNSAFE",
                component="consistency",
                message=f"{LIVE_SMOKE_DOC_ID_ENV} must be provided and start with {DEFAULT_SAFE_DOC_ID_PREFIX}.",
                required_env_names=[LIVE_SMOKE_DOC_ID_ENV],
            )
        )
    if sys_id.get("provided") and not sys_id.get("safe"):
        blockers.append(
            _blocker(
                "SAFE_SYS_ID_UNSAFE",
                component="consistency",
                message=f"{LIVE_SMOKE_SYS_ID_ENV}, when provided, must start with {DEFAULT_SAFE_SYS_ID_PREFIX} or {DEFAULT_SAFE_DOC_ID_PREFIX}.",
                required_env_names=[LIVE_SMOKE_SYS_ID_ENV],
            )
        )
    return blockers


def _collection_policy_blockers(collections: Mapping[str, str]) -> list[dict[str, Any]]:
    blockers = []
    for name, value in collections.items():
        if not value:
            blockers.append(
                _blocker(
                    "COLLECTION_NAME_MISSING",
                    component=name,
                    message=f"{name} collection name must be non-empty for staging smoke planning.",
                    required_env_names=[],
                )
            )
    document = collections.get("document")
    chunk = collections.get("chunk")
    image = collections.get("image")
    if document and chunk and document == chunk:
        blockers.append(_collection_collision_blocker("chunk_collection_points_to_document_collection"))
    if document and image and document == image:
        blockers.append(_collection_collision_blocker("image_collection_points_to_document_collection"))
    if chunk and image and chunk == image:
        blockers.append(_collection_collision_blocker("image_collection_points_to_chunk_collection"))
    return blockers


def _missing_env_blockers(
    component: str,
    env_names: Sequence[str],
    env: Mapping[str, str],
    *,
    code: str = "REQUIRED_ENV_MISSING",
) -> list[dict[str, Any]]:
    missing = [name for name in env_names if not _has_env_value(env, name)]
    if not missing:
        return []
    return [
        _blocker(
            code,
            component=component,
            message=f"Required live smoke env values are missing for {component}.",
            required_env_names=missing,
        )
    ]


def _positive_int_blockers(component: str, env: Mapping[str, str], env_names: Sequence[str]) -> list[dict[str, Any]]:
    blockers = []
    for name in env_names:
        value = _parse_int_env(env, name)
        if value is not None and value <= 0:
            blockers.append(
                _blocker(
                    "REQUIRED_ENV_INVALID",
                    component=component,
                    message=f"{name} must be a positive integer.",
                    required_env_names=[name],
                )
            )
    return blockers


def _has_env_value(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip() != ""


def _parse_int_env(env: Mapping[str, str], name: str) -> int | None:
    value = str(env.get(name, "")).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return 0


def _is_placeholder_secret(value: Any) -> bool:
    return str(value or "").strip().lower() in _PLACEHOLDER_VALUES


def _collection_collision_blocker(code: str) -> dict[str, Any]:
    return _blocker(
        code,
        component="collection_policy",
        message="Document, chunk, and image collection names must remain separate.",
        required_env_names=["MILVUS_COLLECTION", "IMAGE_COLLECTION_NAME"],
    )


def _blocker(
    code: str,
    *,
    component: str,
    message: str,
    required_env_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "code": code,
        "component": component,
        "severity": "blocker",
        "message": message,
        "required_env_names": list(required_env_names),
    }


def _warning(
    code: str,
    *,
    component: str,
    message: str,
    required_env_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "code": code,
        "component": component,
        "severity": "warning",
        "message": message,
        "required_env_names": list(required_env_names),
    }


def _build_consistency_skip_reason(*, gates: dict[str, dict[str, Any]], safe_ids: dict[str, dict[str, Any]]) -> str | None:
    for component in ("mongo", "milvus"):
        gate = gates.get(component)
        if gate is None:
            return f"{component} component must be included for live consistency smoke"
        if gate.get("skip_reason"):
            return str(gate["skip_reason"])
    if not safe_ids["doc_id"]["safe"]:
        return f"{LIVE_SMOKE_DOC_ID_ENV} must be a safe id starting with {DEFAULT_SAFE_DOC_ID_PREFIX}"
    return None


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").upper()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _is_raw_bytes_key(key: str) -> bool:
    normalized = key.replace("-", "_").upper()
    return any(part in normalized for part in _RAW_BYTES_KEY_PARTS)
