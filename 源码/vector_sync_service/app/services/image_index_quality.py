"""Decide whether an image has trustworthy text for semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from app.services.image_context_text import canonicalize_context_text


READY_CAPTION_CONTEXT = "READY_CAPTION_CONTEXT"
NOT_INDEXABLE_MMD_PENDING = "NOT_INDEXABLE_MMD_PENDING"
NOT_INDEXABLE_MMD_NOT_FOUND = "NOT_INDEXABLE_MMD_NOT_FOUND"
NOT_INDEXABLE_CONTEXT_MISSING = "NOT_INDEXABLE_CONTEXT_MISSING"
NOT_INDEXABLE_CAPTION_MISSING = "NOT_INDEXABLE_CAPTION_MISSING"
NOT_INDEXABLE_NO_TEXT = "NOT_INDEXABLE_NO_TEXT"
SOURCE_CONFLICT = "SOURCE_CONFLICT"
LEGACY_UNVERIFIED = "UNVERIFIED_LEGACY"


@dataclass(frozen=True, slots=True)
class ImageIndexQuality:
    status: str
    reason_code: str
    caption_text: str
    context_text: str
    mmd_matched: bool
    indexable: bool

    @property
    def caption_input_hash(self) -> str | None:
        return _text_hash(self.caption_text)

    @property
    def context_input_hash(self) -> str | None:
        return _text_hash(self.context_text)

    def as_img_meta(self) -> dict[str, Any]:
        return {
            "quality_status": self.status,
            "quality_reason": self.reason_code,
            "caption_input_hash": self.caption_input_hash,
            "context_input_hash": self.context_input_hash,
            "caption_indexable": self.indexable,
            "context_indexable": self.indexable,
        }


def evaluate_image_indexability(
    task: dict[str, Any],
    context_status: dict[str, Any] | None,
    *,
    policy: str = "verified_only",
    require_mmd_context: bool = True,
) -> ImageIndexQuality:
    """Return a conservative text-index decision for a completed image task.

    The legacy policy preserves the existing pipeline behavior for controlled
    compatibility tests. Production defaults to verified_only: a record is
    text-indexable only when the MMD image reference was matched and both a
    real caption and adjacent MMD text are present.
    """
    if str(policy or "verified_only").strip().lower() in {"legacy", "permissive"}:
        return ImageIndexQuality(
            status=LEGACY_UNVERIFIED,
            reason_code="IMAGE_TEXT_INDEX_POLICY_LEGACY",
            caption_text=_caption_text(task),
            context_text=_joined_context(task),
            mmd_matched=bool((context_status or {}).get("mmd_context_located")),
            indexable=True,
        )

    img_meta = _as_dict(task.get("img_meta"))
    if str(img_meta.get("quality_status") or "").strip() == SOURCE_CONFLICT or bool(
        img_meta.get("source_conflict")
    ):
        return _quality(SOURCE_CONFLICT, "IMAGE_SOURCE_CONFLICT", task, False, False)

    status = context_status or {}
    img_meta = _as_dict(task.get("img_meta"))
    mmd_evidence = (
        status.get("mmd_image_path")
        or img_meta.get("mmd_image_path")
        or img_meta.get("mmd_paragraph_index") is not None
    )
    mmd_matched = status.get("mmd_context_located") is True and bool(mmd_evidence)
    if require_mmd_context and not mmd_matched:
        if str(status.get("mmd_backfill_status") or "").strip().lower() == "pending":
            return _quality(
                NOT_INDEXABLE_MMD_PENDING,
                "IMAGE_MMD_CONTEXT_PENDING",
                task,
                False,
                False,
            )
        return _quality(
            NOT_INDEXABLE_MMD_NOT_FOUND,
            "IMAGE_MMD_CONTEXT_NOT_FOUND",
            task,
            False,
            False,
        )

    caption = _caption_text(task)
    adjacent_context = _adjacent_context_text(task, caption)
    if caption and adjacent_context:
        return ImageIndexQuality(
            status=READY_CAPTION_CONTEXT,
            reason_code="IMAGE_MMD_CONTEXT_VERIFIED",
            caption_text=caption,
            context_text="\n\n".join((caption, adjacent_context)),
            mmd_matched=True,
            indexable=True,
        )
    if caption:
        return _quality(
            NOT_INDEXABLE_CONTEXT_MISSING,
            "IMAGE_ADJACENT_CONTEXT_MISSING",
            task,
            True,
            False,
        )
    if adjacent_context:
        return _quality(
            NOT_INDEXABLE_CAPTION_MISSING,
            "IMAGE_CAPTION_MISSING",
            task,
            True,
            False,
        )
    return _quality(NOT_INDEXABLE_NO_TEXT, "IMAGE_TEXT_MISSING", task, True, False)


def apply_image_index_quality(record: Any, quality: ImageIndexQuality, *, pipeline_version: str) -> None:
    """Persist quality evidence on the writer-ready record before embedding."""
    img_meta = _as_dict(getattr(record, "img_meta", None))
    img_meta.update(quality.as_img_meta())
    img_meta["pipeline_version"] = pipeline_version
    if quality.status != LEGACY_UNVERIFIED and quality.caption_text:
        img_meta["caption"] = quality.caption_text
    elif quality.status != LEGACY_UNVERIFIED:
        img_meta.pop("caption", None)
    record.img_meta = img_meta


def image_search_quality_filter(vector_field: str) -> str:
    """Milvus filter for vectors that passed the verified-text gate."""
    if vector_field not in {"caption_embed", "context_embed"}:
        return ""
    return f'img_meta["quality_status"] == "{READY_CAPTION_CONTEXT}"'


def _quality(
    status: str,
    reason_code: str,
    task: dict[str, Any],
    mmd_matched: bool,
    indexable: bool,
) -> ImageIndexQuality:
    caption = _caption_text(task)
    return ImageIndexQuality(
        status=status,
        reason_code=reason_code,
        caption_text=caption,
        context_text=_joined_context(task),
        mmd_matched=mmd_matched,
        indexable=indexable,
    )


def _caption_text(task: dict[str, Any]) -> str:
    candidates = [
        task.get("title"),
        _as_dict(task.get("img_meta")).get("caption"),
    ]
    raw_img_meta = _as_dict(task.get("raw_img_meta"))
    candidates.extend(raw_img_meta.get(key) for key in ("caption", "figureTitle", "imageTitle"))
    for value in candidates:
        text = _clean_text(value)
        if text and text.casefold() != "no caption":
            return text
    return ""


def _adjacent_context_text(task: dict[str, Any], caption: str) -> str:
    before = _clean_text(task.get("context_before"))
    after = _clean_text(task.get("context_after"))
    if after.casefold() == "no caption":
        after = ""
    if caption and after:
        folded_caption = caption.casefold()
        if after.casefold() == folded_caption:
            after = ""
        elif after.casefold().startswith(folded_caption):
            after = after[len(caption):].lstrip("\n\r ")
    return "\n\n".join(part for part in (before, after) if part)


def _joined_context(task: dict[str, Any]) -> str:
    return "\n\n".join(
        part
        for part in (_clean_text(task.get("context_before")), _clean_text(task.get("context_after")))
        if part
    )


def _text_hash(text: str) -> str | None:
    normalized = _clean_text(text)
    return sha256(normalized.encode("utf-8")).hexdigest() if normalized else None


def _clean_text(value: Any) -> str:
    return str(canonicalize_context_text(value) or "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
