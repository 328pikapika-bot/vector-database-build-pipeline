"""Offline-script compatible image context extraction from MMD text."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Optional


IMAGE_REF_RE = re.compile(r"!\[[^\]]*]\((?:\./)?figures/([^) \t\r\n]+)\)")
ASSET_REF_RE = re.compile(r"!\[[^\]]*]\((?:\./)?(?:figures|tables)/[^)]+\)")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
FILENAME_META_RE = re.compile(
    r"^\d+-\d+-whxyp_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_(\d+)\.jpe?g$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MmdImageContext:
    """Context and metadata located for one image reference in an MMD file."""

    image_path: str
    filename: str
    paragraph_index: int
    context_before: str
    context_after: str
    context_for_model: str
    img_meta: dict[str, Any] = dataclass_field(default_factory=dict)
    raw_img_meta: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_metadata_update(self) -> dict[str, Any]:
        return {
            "context_before": self.context_before or None,
            "context_after": self.context_after or None,
            "img_meta": dict(self.img_meta),
            "raw_img_meta": dict(self.raw_img_meta),
        }


def extract_mmd_image_contexts(
    mmd_text: str,
    *,
    window: Optional[int] = None,
    before_window: int = 1,
    after_window: int = 2,
) -> list[MmdImageContext]:
    """Locate image references and surrounding paragraphs using the offline script rules."""

    if not isinstance(mmd_text, str) or not mmd_text.strip():
        return []

    if window is not None:
        before_window = int(window or 0)
        after_window = int(window or 0)
    before_paragraph_window = max(int(before_window or 0), 0)
    after_paragraph_window = max(int(after_window or 0), 0)
    paragraphs = split_paragraphs(mmd_text)
    contexts: list[MmdImageContext] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        if not is_image_reference_paragraph(paragraph):
            continue
        refs = IMAGE_REF_RE.findall(paragraph)
        before = _collect_before(paragraphs, paragraph_index, before_paragraph_window)
        after = _collect_after(paragraphs, paragraph_index, after_paragraph_window)
        before_text = "\n\n".join(before).strip()
        after_text = "\n\n".join(after).strip()
        context_for_model = "\n\n".join(
            item for item in (before_text, after_text) if item.strip()
        )
        for image_path in refs:
            filename = _filename(image_path)
            img_meta = _filename_img_meta(filename)
            img_meta.update(
                {
                    "source_filename": filename,
                    "mmd_image_path": image_path,
                    "mmd_paragraph_index": paragraph_index,
                }
            )
            contexts.append(
                MmdImageContext(
                    image_path=image_path,
                    filename=filename,
                    paragraph_index=paragraph_index,
                    context_before=before_text,
                    context_after=after_text,
                    context_for_model=context_for_model,
                    img_meta=img_meta,
                    raw_img_meta={
                        "mmd_image_path": image_path,
                        "mmd_paragraph": paragraph,
                        "mmd_paragraph_index": paragraph_index,
                    },
                )
            )
    return contexts


def find_mmd_context_for_image(
    contexts: Iterable[MmdImageContext],
    *,
    image_name: str = "",
    relative_url: str = "",
) -> Optional[MmdImageContext]:
    """Return the best MMD context match for an image name or relative URL."""

    candidates = _candidate_names(image_name, relative_url)
    if not candidates:
        return None
    for context in contexts:
        keys = _candidate_names(context.filename, context.image_path)
        if candidates & keys:
            return context
    return None


def compose_context_after(caption: str = "", after_text: str = "") -> Optional[str]:
    """Build context_after as caption followed by the next MMD paragraphs."""

    caption_text = str(caption or "").strip()
    after = str(after_text or "").strip()
    if caption_text and after.lower().startswith(caption_text.lower()):
        return after
    parts = [part for part in (caption_text, after) if part]
    if not parts:
        return None
    return "\n\n".join(parts)


def split_paragraphs(text: str) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return [item.strip() for item in PARAGRAPH_SPLIT_RE.split(normalized) if item.strip()]


def is_asset_reference_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in str(paragraph or "").splitlines() if line.strip()]
    return bool(lines) and all(ASSET_REF_RE.fullmatch(line) for line in lines)


def is_image_reference_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in str(paragraph or "").splitlines() if line.strip()]
    return bool(lines) and all(IMAGE_REF_RE.fullmatch(line) for line in lines)


def _collect_before(paragraphs: list[str], paragraph_index: int, window: int) -> list[str]:
    collected: list[str] = []
    cursor = paragraph_index - 1
    while cursor >= 0 and len(collected) < window:
        if is_asset_reference_paragraph(paragraphs[cursor]):
            break
        collected.append(paragraphs[cursor])
        cursor -= 1
    collected.reverse()
    return collected


def _collect_after(paragraphs: list[str], paragraph_index: int, window: int) -> list[str]:
    collected: list[str] = []
    cursor = paragraph_index + 1
    while cursor < len(paragraphs) and len(collected) < window:
        if is_asset_reference_paragraph(paragraphs[cursor]):
            break
        collected.append(paragraphs[cursor])
        cursor += 1
    return collected


def _filename(path: str) -> str:
    return PurePosixPath(str(path or "").replace("\\", "/")).name


def _candidate_names(*values: str) -> set[str]:
    names: set[str] = set()
    for value in values:
        text = str(value or "").strip().replace("\\", "/")
        if not text:
            continue
        names.add(text.lower())
        names.add(_filename(text).lower())
    return {name for name in names if name}


def _filename_img_meta(filename: str) -> dict[str, Any]:
    match = FILENAME_META_RE.match(str(filename or ""))
    if not match:
        return {}
    x, y, width, height, page = match.groups()
    return {
        "bbox": [float(x), float(y), float(width), float(height)],
        "page": int(page),
    }
