"""Image metadata and perceptual hash helpers.

This module intentionally does not invent feature values. If optional image
libraries are unavailable or bytes cannot be decoded, pHash computation returns
a structured failure instead of a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha256
import importlib.util
from io import BytesIO
import re
from typing import Any, Optional
from urllib.parse import urlparse

from app.contracts.docx_contract import extract_docx_doc_meta
from app.config import settings
from app.models.image import ImageRecord
from app.services.image_context_text import canonicalize_context_text


IMAGE_PHASH_DEPENDENCY_MISSING = "IMAGE_PHASH_DEPENDENCY_MISSING"
IMAGE_PHASH_DECODE_ERROR = "IMAGE_PHASH_DECODE_ERROR"
IMAGE_BYTES_EMPTY = "IMAGE_BYTES_EMPTY"
IMAGE_DIMENSION_DECODE_ERROR = "IMAGE_DIMENSION_DECODE_ERROR"
IMAGE_PIXEL_LIMIT_EXCEEDED = "IMAGE_PIXEL_LIMIT_EXCEEDED"
IMAGE_MANIFEST_FIELD_REQUIRED = "IMAGE_MANIFEST_FIELD_REQUIRED"
IMAGE_MANIFEST_INVALID_SEQUENCE = "IMAGE_MANIFEST_INVALID_SEQUENCE"
IMAGE_MANIFEST_INVALID_IMAGE_ID = "IMAGE_MANIFEST_INVALID_IMAGE_ID"
IMAGE_MANIFEST_INVALID_URL = "IMAGE_MANIFEST_INVALID_URL"
IMAGE_MANIFEST_INVALID_CONTENT_TYPE = "IMAGE_MANIFEST_INVALID_CONTENT_TYPE"
IMAGE_MANIFEST_INVALID_SIZE = "IMAGE_MANIFEST_INVALID_SIZE"
IMAGE_MANIFEST_INVALID_SHA256 = "IMAGE_MANIFEST_INVALID_SHA256"
IMAGE_MANIFEST_SHA256_MISMATCH = "IMAGE_MANIFEST_SHA256_MISMATCH"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass(slots=True)
class ImagePhashResult:
    success: bool
    p_hash: Optional[bytes] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False


@dataclass(slots=True)
class ImageFeatureReadiness:
    pil_available: bool
    imagehash_available: bool
    phash_available: bool
    missing_dependencies: list[str] = dataclass_field(default_factory=list)


@dataclass(slots=True)
class ImageDimensionValidationResult:
    success: bool
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_count: Optional[int] = None
    max_pixels: int = settings.IMAGE_MAX_PIXELS
    image_format: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False


@dataclass(slots=True)
class ImageManifestValidationFailure:
    field_name: str
    error_code: str
    error_message: str
    retryable: bool = False


@dataclass(slots=True)
class ValidatedImageManifest:
    doc_id: str
    image_id: str
    relative_url: str
    content_type: str
    size_bytes: int
    bytes_sha256: str
    sequence: int
    title: str = ""
    name: str = ""
    file_type: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    raw_manifest: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(slots=True)
class ImageManifestValidationResult:
    success: bool
    manifest: Optional[ValidatedImageManifest] = None
    failures: list[ImageManifestValidationFailure] = dataclass_field(default_factory=list)


def get_image_feature_readiness() -> ImageFeatureReadiness:
    """Report optional local dependencies for image feature extraction."""
    pil_available = importlib.util.find_spec("PIL") is not None
    imagehash_available = importlib.util.find_spec("imagehash") is not None
    missing = []
    if not pil_available:
        missing.append("PIL")
    if not imagehash_available:
        missing.append("imagehash")
    return ImageFeatureReadiness(
        pil_available=pil_available,
        imagehash_available=imagehash_available,
        phash_available=pil_available and imagehash_available,
        missing_dependencies=missing,
    )


def build_image_record(doc_id: str, sequence: int, **metadata: Any) -> ImageRecord:
    """Build an ImageRecord whose primary key is the upstream image id."""
    payload = dict(metadata)
    payload.setdefault("docId", doc_id)
    payload.setdefault("sequence", sequence)
    if "img_id" not in payload and "imageId" in payload:
        payload["img_id"] = payload.pop("imageId")
    if not str(payload.get("img_id") or "").strip():
        raise ValueError("upstream image id is required")
    raw_img_meta = _dict_or_empty(payload.get("raw_img_meta"))
    raw_img_meta.setdefault("id", str(payload["img_id"]).strip())
    payload["raw_img_meta"] = raw_img_meta
    return ImageRecord.model_validate(payload)


def validate_image_bytes_dimensions(
    image_bytes: bytes,
    max_pixels: Optional[int] = None,
) -> ImageDimensionValidationResult:
    """Validate PNG/JPEG dimensions without image libraries."""
    resolved_max_pixels = settings.IMAGE_MAX_PIXELS if max_pixels is None else int(max_pixels)
    if not image_bytes:
        return ImageDimensionValidationResult(
            success=False,
            max_pixels=resolved_max_pixels,
            error_code=IMAGE_BYTES_EMPTY,
            error_message="image bytes are empty",
            retryable=False,
        )

    try:
        image_format, width, height = _parse_image_dimensions(image_bytes)
    except ValueError as exc:
        return ImageDimensionValidationResult(
            success=False,
            max_pixels=resolved_max_pixels,
            error_code=IMAGE_DIMENSION_DECODE_ERROR,
            error_message=str(exc),
            retryable=False,
        )

    pixel_count = width * height
    if pixel_count > resolved_max_pixels:
        return ImageDimensionValidationResult(
            success=False,
            width=width,
            height=height,
            pixel_count=pixel_count,
            max_pixels=resolved_max_pixels,
            image_format=image_format,
            error_code=IMAGE_PIXEL_LIMIT_EXCEEDED,
            error_message=f"image pixel count {pixel_count} exceeds max {resolved_max_pixels}",
            retryable=False,
        )

    return ImageDimensionValidationResult(
        success=True,
        width=width,
        height=height,
        pixel_count=pixel_count,
        max_pixels=resolved_max_pixels,
        image_format=image_format,
        retryable=False,
    )


def compute_phash(image_bytes: bytes) -> ImagePhashResult:
    """Compute a real 64-bit pHash when optional libraries are available."""
    if not image_bytes:
        return ImagePhashResult(
            success=False,
            error_code=IMAGE_PHASH_DECODE_ERROR,
            error_message="image bytes are empty",
            retryable=False,
        )

    try:
        from PIL import Image
        import imagehash
    except ImportError as exc:
        return ImagePhashResult(
            success=False,
            error_code=IMAGE_PHASH_DEPENDENCY_MISSING,
            error_message=f"required image hash dependency is missing: {getattr(exc, 'name', str(exc))}",
            retryable=False,
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            hash_value = imagehash.phash(image, hash_size=8)
        hex_value = str(hash_value)
        return ImagePhashResult(success=True, p_hash=bytes.fromhex(hex_value))
    except Exception as exc:  # pragma: no cover - exact decoder failures depend on installed libs
        return ImagePhashResult(
            success=False,
            error_code=IMAGE_PHASH_DECODE_ERROR,
            error_message=str(exc),
            retryable=False,
        )


def validate_image_manifest(
    manifest: dict[str, Any],
    image_bytes: Optional[bytes] = None,
) -> ImageManifestValidationResult:
    """Validate and normalize a worker-produced image manifest."""
    if not isinstance(manifest, dict):
        return ImageManifestValidationResult(
            success=False,
            failures=[
                ImageManifestValidationFailure(
                    field_name="manifest",
                    error_code=IMAGE_MANIFEST_FIELD_REQUIRED,
                    error_message="manifest must be a dict",
                )
            ],
        )

    failures: list[ImageManifestValidationFailure] = []
    doc_id = _required_text(manifest, ("doc_id", "docId"), "doc_id", failures)
    image_id = _required_text(manifest, ("image_id", "imageId", "img_id"), "image_id", failures)
    relative_url = _required_text(manifest, ("relative_url", "relativeUrl"), "relative_url", failures)
    content_type = _required_text(manifest, ("content_type", "contentType"), "content_type", failures)
    sequence = _required_int(manifest, ("sequence",), "sequence", failures)
    size_bytes = _required_int(manifest, ("size_bytes", "sizeBytes", "size"), "size_bytes", failures)
    bytes_sha256 = _required_text(
        manifest,
        ("bytes_sha256", "bytesSha256", "sha256"),
        "bytes_sha256",
        failures,
    )

    if relative_url and not _is_relative_url(relative_url):
        failures.append(
            ImageManifestValidationFailure(
                field_name="relative_url",
                error_code=IMAGE_MANIFEST_INVALID_URL,
                error_message="relative_url must be relative, not absolute",
            )
        )
    if content_type and not content_type.lower().split(";", 1)[0].strip().startswith("image/"):
        failures.append(
            ImageManifestValidationFailure(
                field_name="content_type",
                error_code=IMAGE_MANIFEST_INVALID_CONTENT_TYPE,
                error_message="content_type must start with image/",
            )
        )
    if size_bytes is not None and size_bytes <= 0:
        failures.append(
            ImageManifestValidationFailure(
                field_name="size_bytes",
                error_code=IMAGE_MANIFEST_INVALID_SIZE,
                error_message="size_bytes must be positive",
            )
        )
    if bytes_sha256 and not _SHA256_PATTERN.fullmatch(bytes_sha256):
        failures.append(
            ImageManifestValidationFailure(
                field_name="bytes_sha256",
                error_code=IMAGE_MANIFEST_INVALID_SHA256,
                error_message="bytes_sha256 must be 64 hex characters",
            )
        )
    if image_bytes is not None and bytes_sha256 and _SHA256_PATTERN.fullmatch(bytes_sha256):
        actual = sha256(image_bytes).hexdigest()
        if actual.lower() != bytes_sha256.lower():
            failures.append(
                ImageManifestValidationFailure(
                    field_name="bytes_sha256",
                    error_code=IMAGE_MANIFEST_SHA256_MISMATCH,
                    error_message="bytes_sha256 does not match image bytes",
                )
            )
    original_image_id = _source_image_id(manifest)
    if not original_image_id:
        failures.append(
            ImageManifestValidationFailure(
                field_name="raw_img_meta.id",
                error_code=IMAGE_MANIFEST_FIELD_REQUIRED,
                error_message="raw_img_meta.id is required as the image collection primary key",
            )
        )
    elif image_id and image_id != original_image_id:
        failures.append(
            ImageManifestValidationFailure(
                field_name="image_id",
                error_code=IMAGE_MANIFEST_INVALID_IMAGE_ID,
                error_message=f"image_id must equal raw_img_meta.id {original_image_id}",
            )
        )

    width = _optional_positive_int(manifest, ("width", "imageWidth"))
    height = _optional_positive_int(manifest, ("height", "imageHeight"))
    if failures:
        return ImageManifestValidationResult(success=False, failures=failures)

    normalized_content_type = content_type.lower().split(";", 1)[0].strip()
    return ImageManifestValidationResult(
        success=True,
        manifest=ValidatedImageManifest(
            doc_id=doc_id,
            image_id=image_id,
            relative_url=relative_url,
            content_type=normalized_content_type,
            size_bytes=size_bytes,
            bytes_sha256=bytes_sha256.lower(),
            sequence=sequence,
            title=_optional_text(manifest, ("title", "caption", "imageTitle")),
            name=_optional_text(manifest, ("name", "fileName", "filename", "imageName")),
            file_type=_optional_text(manifest, ("file_type", "fileType")) or _file_type_from_content_type(normalized_content_type),
            width=width,
            height=height,
            raw_manifest=dict(manifest),
        ),
    )


def build_image_record_draft_from_manifest(
    manifest: dict[str, Any] | ValidatedImageManifest | ImageManifestValidationResult,
    *,
    dimension_result: Optional[ImageDimensionValidationResult] = None,
    feature_metadata: Optional[dict[str, Any]] = None,
) -> ImageRecord:
    """Build a metadata-only ImageRecord draft without fake feature values."""
    validated = _coerce_validated_manifest(manifest)
    metadata = feature_metadata or {}
    doc_meta = _build_doc_meta(validated.doc_id, metadata, validated.raw_manifest)
    docx_meta = extract_docx_doc_meta(doc_meta)

    width = validated.width or 0
    height = validated.height or 0
    if dimension_result is not None and dimension_result.width and dimension_result.height:
        width = dimension_result.width or width
        height = dimension_result.height or height

    img_meta = {
        "relative_url": validated.relative_url,
        "content_type": validated.content_type,
        "bytes_sha256": validated.bytes_sha256,
        "size_bytes": validated.size_bytes,
    }
    if dimension_result is not None:
        img_meta.update(
            {
                "dimension_validation_success": dimension_result.success,
                "image_format": dimension_result.image_format,
                "pixel_count": dimension_result.pixel_count,
                "max_pixels": dimension_result.max_pixels,
                "dimension_error_code": dimension_result.error_code,
            }
        )
    img_meta.update(_dict_or_empty(metadata.get("img_meta")))
    image_caption = canonicalize_context_text(
        metadata.get("image_title")
        or metadata.get("caption")
        or validated.title
    ) or _default_caption()
    img_meta.setdefault("caption", image_caption)

    raw_img_meta = _dict_or_empty(metadata.get("raw_img_meta"))
    if not raw_img_meta:
        raw_img_meta = _dict_or_empty(validated.raw_manifest.get("raw_img_meta"))
    if not raw_img_meta:
        raw_img_meta = _dict_or_empty(validated.raw_manifest.get("source_metadata"))

    context_before = canonicalize_context_text(metadata.get("context_before"))
    context_after = canonicalize_context_text(metadata.get("context_after")) or image_caption

    return ImageRecord(
        img_id=validated.image_id,
        docId=docx_meta.get("sysId") or validated.doc_id,
        url=validated.relative_url,
        title=docx_meta.get("title"),
        keyword=docx_meta.get("keyword"),
        accessType=docx_meta.get("accessType"),
        classificationCode=docx_meta.get("classificationCode"),
        year=docx_meta.get("year"),
        hasAuthorname=docx_meta.get("hasAuthorname"),
        hasCreateOrganizationname=docx_meta.get("hasCreateOrganizationname"),
        language=docx_meta.get("language"),
        sysDataStatus=docx_meta.get("sysDataStatus"),
        hasSourcesysSourceId=docx_meta.get("hasSourcesysSourceId"),
        hasSourceidSet=docx_meta.get("hasSourceidSet"),
        name=str(metadata.get("name") or validated.name or ""),
        width=width,
        height=height,
        sequence=validated.sequence,
        fileType=validated.file_type,
        size=validated.size_bytes,
        raw_img_meta=raw_img_meta,
        img_meta=img_meta,
        context_before=context_before,
        context_after=context_after,
        text=_optional_metadata_text(metadata.get("text")),
        doc_meta=doc_meta,
    )


def _parse_image_dimensions(image_bytes: bytes) -> tuple[str, int, int]:
    if image_bytes.startswith(_PNG_SIGNATURE):
        return _parse_png_dimensions(image_bytes)
    if image_bytes.startswith(b"\xff\xd8"):
        return _parse_jpeg_dimensions(image_bytes)
    raise ValueError("unsupported or malformed image bytes")


def _default_caption() -> str:
    return str(settings.IMAGE_DEFAULT_CAPTION or "No caption").strip() or "No caption"


def _parse_png_dimensions(image_bytes: bytes) -> tuple[str, int, int]:
    if len(image_bytes) < 24 or image_bytes[12:16] != b"IHDR":
        raise ValueError("malformed PNG IHDR")
    width = int.from_bytes(image_bytes[16:20], "big")
    height = int.from_bytes(image_bytes[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    return "png", width, height


def _parse_jpeg_dimensions(image_bytes: bytes) -> tuple[str, int, int]:
    index = 2
    length = len(image_bytes)
    while index < length:
        while index < length and image_bytes[index] != 0xFF:
            index += 1
        while index < length and image_bytes[index] == 0xFF:
            index += 1
        if index >= length:
            break

        marker = image_bytes[index]
        index += 1
        if marker == 0xD9:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > length:
            raise ValueError("truncated JPEG segment length")

        segment_length = int.from_bytes(image_bytes[index:index + 2], "big")
        if segment_length < 2:
            raise ValueError("invalid JPEG segment length")
        segment_end = index + segment_length
        if segment_end > length:
            raise ValueError("truncated JPEG segment")

        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 7:
                raise ValueError("invalid JPEG SOF segment")
            height = int.from_bytes(image_bytes[index + 3:index + 5], "big")
            width = int.from_bytes(image_bytes[index + 5:index + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("JPEG dimensions must be positive")
            return "jpeg", width, height
        index = segment_end
    raise ValueError("JPEG SOF dimensions not found")


def _coerce_validated_manifest(
    manifest: dict[str, Any] | ValidatedImageManifest | ImageManifestValidationResult,
) -> ValidatedImageManifest:
    if isinstance(manifest, ValidatedImageManifest):
        return manifest
    if isinstance(manifest, ImageManifestValidationResult):
        if manifest.success and manifest.manifest is not None:
            return manifest.manifest
        raise ValueError(_format_manifest_failures(manifest.failures))
    result = validate_image_manifest(manifest)
    if result.success and result.manifest is not None:
        return result.manifest
    raise ValueError(_format_manifest_failures(result.failures))


def _format_manifest_failures(failures: list[ImageManifestValidationFailure]) -> str:
    if not failures:
        return "image manifest validation failed"
    return "; ".join(f"{failure.field_name}:{failure.error_code}" for failure in failures)


def _required_text(
    manifest: dict[str, Any],
    keys: tuple[str, ...],
    field_name: str,
    failures: list[ImageManifestValidationFailure],
) -> str:
    value = _optional_text(manifest, keys)
    if not value:
        failures.append(
            ImageManifestValidationFailure(
                field_name=field_name,
                error_code=IMAGE_MANIFEST_FIELD_REQUIRED,
                error_message=f"{field_name} is required",
            )
        )
    return value


def _required_int(
    manifest: dict[str, Any],
    keys: tuple[str, ...],
    field_name: str,
    failures: list[ImageManifestValidationFailure],
) -> Optional[int]:
    value = _first_value(manifest, keys)
    if value is None or str(value).strip() == "":
        failures.append(
            ImageManifestValidationFailure(
                field_name=field_name,
                error_code=IMAGE_MANIFEST_FIELD_REQUIRED,
                error_message=f"{field_name} is required",
            )
        )
        return None
    try:
        integer = int(str(value).strip())
    except (TypeError, ValueError):
        failures.append(
            ImageManifestValidationFailure(
                field_name=field_name,
                error_code=IMAGE_MANIFEST_INVALID_SEQUENCE if field_name == "sequence" else IMAGE_MANIFEST_INVALID_SIZE,
                error_message=f"{field_name} must be an integer",
            )
        )
        return None
    if field_name == "sequence" and integer < 0:
        failures.append(
            ImageManifestValidationFailure(
                field_name=field_name,
                error_code=IMAGE_MANIFEST_INVALID_SEQUENCE,
                error_message="sequence must be non-negative",
            )
        )
    return integer


def _optional_positive_int(manifest: dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    value = _first_value(manifest, keys)
    if value is None or str(value).strip() == "":
        return None
    try:
        integer = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def _optional_text(manifest: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_value(manifest, keys)
    if value is None:
        return ""
    return str(value).strip()


def _first_value(manifest: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in manifest:
            return manifest.get(key)
    return None


def _is_relative_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return not parsed.scheme and not parsed.netloc


def _file_type_from_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if "/" not in media_type:
        return ""
    return media_type.rsplit("/", 1)[-1]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def source_image_id(value: Any) -> str:
    """Return the stable upstream image id when the source supplied one."""
    if not isinstance(value, dict):
        return ""
    for key in ("id", "imageId", "img_id", "imgId", "image_id"):
        candidate = value.get(key)
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return ""


def _source_image_id(manifest: dict[str, Any]) -> str:
    for key in ("raw_img_meta", "source_metadata"):
        candidate = source_image_id(manifest.get(key))
        if candidate:
            return candidate
    return ""


def _build_doc_meta(
    doc_id: str,
    metadata: dict[str, Any],
    raw_manifest: dict[str, Any],
) -> dict[str, Any]:
    doc_meta = {
        **_dict_or_empty(raw_manifest.get("doc_metadata")),
        **_dict_or_empty(raw_manifest.get("doc_meta")),
        **_dict_or_empty(metadata.get("doc_meta")),
    }
    for source in (raw_manifest, metadata):
        for key, output_key in (
            ("doc_title", "title"),
            ("keyword", "keyword"),
            ("accessType", "accessType"),
            ("classificationCode", "classificationCode"),
            ("year", "year"),
            ("hasAuthorname", "hasAuthorname"),
            ("hasCreateOrganizationname", "hasCreateOrganizationname"),
            ("language", "language"),
            ("sysDataStatus", "sysDataStatus"),
            ("hasSourcesysSourceId", "hasSourcesysSourceId"),
            ("hasSourceidSet", "hasSourceidSet"),
        ):
            value = source.get(key)
            if output_key not in doc_meta and value not in (None, "", [], {}):
                doc_meta[output_key] = value
    doc_meta.setdefault("sysId", doc_id)
    doc_meta.setdefault("doc_id", doc_id)
    return doc_meta


def _optional_metadata_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
