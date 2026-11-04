import builtins
from hashlib import sha256
import importlib.util

import pytest

from app.config import settings
from app.services.image_features import (
    IMAGE_BYTES_EMPTY,
    IMAGE_DIMENSION_DECODE_ERROR,
    IMAGE_MANIFEST_INVALID_CONTENT_TYPE,
    IMAGE_MANIFEST_INVALID_IMAGE_ID,
    IMAGE_MANIFEST_INVALID_SEQUENCE,
    IMAGE_MANIFEST_INVALID_SHA256,
    IMAGE_MANIFEST_INVALID_SIZE,
    IMAGE_MANIFEST_INVALID_URL,
    IMAGE_MANIFEST_SHA256_MISMATCH,
    IMAGE_PHASH_DECODE_ERROR,
    IMAGE_PHASH_DEPENDENCY_MISSING,
    IMAGE_PIXEL_LIMIT_EXCEEDED,
    build_image_record_draft_from_manifest,
    build_image_record,
    compute_phash,
    get_image_feature_readiness,
    validate_image_bytes_dimensions,
    validate_image_manifest,
)
from app.services.image_context_text import canonicalize_context_text, join_context_parts


def test_build_image_record_uses_stable_id_and_preserves_metadata():
    record = build_image_record(
        "DOC-9",
        2,
        img_id="source-image-9-2",
        raw_img_meta={"id": "source-image-9-2"},
        url="figures/2.jpg",
        title="Two phase curve",
        name="2.jpg",
        width=100,
        height=80,
        fileType="jpg",
        size=1234,
        doc_meta={"sysId": "DOC-9"},
        img_meta={"page": 7},
        imageBytes=b"raw",
    )

    assert record.img_id == "source-image-9-2"
    assert record.imageId == "source-image-9-2"
    assert record.docId == "DOC-9"
    assert record.sequence == 2
    assert record.doc_meta == {"sysId": "DOC-9"}
    assert record.img_meta == {"page": 7}
    assert "imageBytes" not in record.model_dump()


def test_build_image_record_requires_original_image_id():
    with pytest.raises(ValueError, match="upstream image id is required"):
        build_image_record("DOC-9", 2, raw_img_meta={})


def test_image_feature_readiness_reports_missing_phash_dependencies(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"PIL", "imagehash"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    readiness = get_image_feature_readiness()

    assert readiness.pil_available is False
    assert readiness.imagehash_available is False
    assert readiness.phash_available is False
    assert readiness.missing_dependencies == ["PIL", "imagehash"]


def test_compute_phash_reports_dependency_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "imagehash" or name == "PIL" or name.startswith("PIL."):
            raise ImportError("missing test dependency", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = compute_phash(b"not-a-real-image")

    assert result.success is False
    assert result.p_hash is None
    assert result.error_code == IMAGE_PHASH_DEPENDENCY_MISSING
    assert "missing" in result.error_message


def test_compute_phash_never_returns_fake_hash_for_invalid_bytes():
    result = compute_phash(b"not-a-real-image")

    assert result.success is False
    assert result.p_hash is None
    assert result.error_code in {IMAGE_PHASH_DEPENDENCY_MISSING, IMAGE_PHASH_DECODE_ERROR}


def test_validate_png_dimensions_success(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MAX_PIXELS", 2_000_000)
    result = validate_image_bytes_dimensions(_png_bytes(width=320, height=240))

    assert result.success is True
    assert result.image_format == "png"
    assert result.width == 320
    assert result.height == 240
    assert result.pixel_count == 76800
    assert result.max_pixels == 2_000_000


def test_validate_jpeg_dimensions_success():
    result = validate_image_bytes_dimensions(_jpeg_bytes(width=640, height=480))

    assert result.success is True
    assert result.image_format == "jpeg"
    assert result.width == 640
    assert result.height == 480
    assert result.pixel_count == 307200


def test_validate_dimensions_rejects_images_above_pixel_limit(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MAX_PIXELS", 2_000_000)
    result = validate_image_bytes_dimensions(_png_bytes(width=2000, height=1001))

    assert result.success is False
    assert result.error_code == IMAGE_PIXEL_LIMIT_EXCEEDED
    assert result.width == 2000
    assert result.height == 1001
    assert result.pixel_count == 2_002_000
    assert result.max_pixels == 2_000_000


def test_validate_dimensions_reports_empty_and_invalid_bytes_without_fake_dimensions():
    empty = validate_image_bytes_dimensions(b"")
    invalid = validate_image_bytes_dimensions(b"not-an-image")

    assert empty.success is False
    assert empty.error_code == IMAGE_BYTES_EMPTY
    assert empty.width is None
    assert empty.height is None
    assert invalid.success is False
    assert invalid.error_code == IMAGE_DIMENSION_DECODE_ERROR
    assert invalid.width is None
    assert invalid.height is None


def test_validate_image_manifest_success_normalizes_worker_fields():
    image_bytes = b"downloaded-image"
    manifest = _manifest(image_bytes=image_bytes)

    result = validate_image_manifest(manifest, image_bytes=image_bytes)

    assert result.success is True
    assert result.failures == []
    assert result.manifest.doc_id == "DOC-10"
    assert result.manifest.image_id == "DOC-10_img_2"
    assert result.manifest.relative_url == "/images/2.png"
    assert result.manifest.content_type == "image/png"
    assert result.manifest.size_bytes == len(image_bytes)
    assert result.manifest.bytes_sha256 == sha256(image_bytes).hexdigest()
    assert result.manifest.sequence == 2
    assert result.manifest.file_type == "png"


def test_validate_image_manifest_rejects_required_invalid_fields():
    image_bytes = b"downloaded-image"
    manifest = _manifest(
        image_bytes=image_bytes,
        imageId="DOC-10_img_999",
        relativeUrl="https://files.test/images/2.png",
        contentType="text/plain",
        sequence="bad",
        sizeBytes=0,
        bytesSha256="not-hex",
    )

    result = validate_image_manifest(manifest, image_bytes=image_bytes)
    codes = {failure.error_code for failure in result.failures}

    assert result.success is False
    assert IMAGE_MANIFEST_INVALID_IMAGE_ID in codes
    assert IMAGE_MANIFEST_INVALID_URL in codes
    assert IMAGE_MANIFEST_INVALID_CONTENT_TYPE in codes
    assert IMAGE_MANIFEST_INVALID_SEQUENCE in codes
    assert IMAGE_MANIFEST_INVALID_SIZE in codes
    assert IMAGE_MANIFEST_INVALID_SHA256 in codes


def test_validate_image_manifest_rejects_stable_id_and_sha_mismatch():
    image_bytes = b"downloaded-image"
    manifest = _manifest(
        image_bytes=image_bytes,
        imageId="DOC-10_img_3",
        bytesSha256=sha256(b"other-bytes").hexdigest(),
    )

    result = validate_image_manifest(manifest, image_bytes=image_bytes)
    codes = {failure.error_code for failure in result.failures}

    assert result.success is False
    assert IMAGE_MANIFEST_INVALID_IMAGE_ID in codes
    assert IMAGE_MANIFEST_SHA256_MISMATCH in codes


def test_build_image_record_draft_from_manifest_preserves_metadata_without_fake_features():
    image_bytes = b"downloaded-image"
    manifest_result = validate_image_manifest(_manifest(image_bytes=image_bytes), image_bytes=image_bytes)
    dimensions = validate_image_bytes_dimensions(_png_bytes(width=320, height=240))

    draft = build_image_record_draft_from_manifest(
        manifest_result,
        dimension_result=dimensions,
        feature_metadata={
            "doc_meta": {
                "sysId": "DOC-10",
                "title": "Metadata title",
                "language": ["zh"],
                "sysDataStatus": "ACTIVE",
            },
            "img_meta": {"source": "worker_manifest"},
        },
    )

    assert draft.img_id == "DOC-10_img_2"
    assert draft.docId == "DOC-10"
    assert draft.url == "/images/2.png"
    assert draft.title == ["Metadata title"]
    assert draft.language == ["zh"]
    assert draft.sysDataStatus == "ACTIVE"
    assert draft.context_after == "Figure 2"
    assert draft.width == 320
    assert draft.height == 240
    assert draft.sequence == 2
    assert draft.fileType == "png"
    assert draft.size == len(image_bytes)
    assert draft.doc_meta == {
        "sysId": "DOC-10",
        "title": "Metadata title",
        "language": ["zh"],
        "sysDataStatus": "ACTIVE",
        "doc_id": "DOC-10",
    }
    assert draft.img_meta["bytes_sha256"] == sha256(image_bytes).hexdigest()
    assert draft.img_meta["pixel_count"] == 76800
    assert draft.img_meta["source"] == "worker_manifest"
    assert draft.p_hash is None
    assert draft.visual_embed is None
    assert draft.text_embed is None
    assert draft.visual_text_embed is None
    assert draft.text_sparse_vec is None


def test_build_image_record_draft_from_manifest_defaults_missing_caption(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_DEFAULT_CAPTION", "No caption")
    image_bytes = b"downloaded-image"
    manifest_result = validate_image_manifest(
        _manifest(image_bytes=image_bytes, title=""),
        image_bytes=image_bytes,
    )

    draft = build_image_record_draft_from_manifest(manifest_result)

    assert draft.img_meta["caption"] == "No caption"
    assert draft.context_after == "No caption"


def test_canonicalize_context_text_decodes_double_escaped_html_and_strips_inline_tags():
    raw = "Taxonomic Studies on Some Members of the Genus &amp;lt;i&amp;gt;Abutilon&amp;lt;/i&amp;gt; Mill. (Malvaceae)"

    assert canonicalize_context_text(raw) == (
        "Taxonomic Studies on Some Members of the Genus Abutilon Mill. (Malvaceae)"
    )


def test_canonicalize_context_text_normalizes_whitespace_and_newlines():
    raw = "  Figure 1 \r\n\r\n  line 2 \t  \r\n\r\n\r\n line 3 "

    assert canonicalize_context_text(raw) == "Figure 1\n\nline 2\n\nline 3"


def test_join_context_parts_skips_empty_and_garbage_values():
    joined = join_context_parts("valid before", "&amp;lt;i&amp;gt;", "valid after")

    assert joined == "valid before\n\nvalid after"


def test_build_image_record_draft_from_manifest_canonicalizes_context_text():
    image_bytes = b"downloaded-image"
    manifest_result = validate_image_manifest(_manifest(image_bytes=image_bytes), image_bytes=image_bytes)

    draft = build_image_record_draft_from_manifest(
        manifest_result,
        feature_metadata={
            "context_before": "Taxonomic Studies &amp;lt;i&amp;gt;on&amp;lt;/i&amp;gt;",
            "context_after": "Figure &amp;lt;i&amp;gt;caption&amp;lt;/i&amp;gt;",
            "image_title": "Figure &amp;lt;i&amp;gt;caption&amp;lt;/i&amp;gt;",
        },
    )

    assert draft.context_before == "Taxonomic Studies on"
    assert draft.context_after == "Figure caption"
    assert draft.img_meta["caption"] == "Figure caption"


def _manifest(image_bytes=b"downloaded-image", **overrides):
    manifest = {
        "docId": "DOC-10",
        "imageId": "DOC-10_img_2",
        "relativeUrl": "/images/2.png",
        "contentType": "image/png; charset=binary",
        "sizeBytes": len(image_bytes),
        "bytesSha256": sha256(image_bytes).hexdigest(),
        "sequence": "2",
        "title": "Figure 2",
        "name": "2.png",
        "raw_img_meta": {"id": "DOC-10_img_2"},
    }
    manifest.update(overrides)
    return manifest


def _png_bytes(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def _jpeg_bytes(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + (4).to_bytes(2, "big") + b"JF"
    sof0 = (
        b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"
