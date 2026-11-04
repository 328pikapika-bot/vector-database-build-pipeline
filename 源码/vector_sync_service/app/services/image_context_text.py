"""Deterministic canonicalization for image caption/context text."""

from __future__ import annotations

from html import unescape
import re
from typing import Any, Optional


_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_WHITESPACE_RE = re.compile(r"[^\S\r\n]+")
_NEWLINE_PAD_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ENTITY_FRAGMENT_RE = re.compile(r"&(?:amp;)*(?:lt|gt|nbsp|quot|apos);", re.IGNORECASE)


def canonicalize_context_text(value: Any) -> Optional[str]:
    text = _coerce_text(value)
    if not text:
        return None
    text = _recursive_unescape(text)
    text = _TAG_RE.sub("", text)
    text = _normalize_text(text)
    if not text or _looks_like_garbage(text):
        return None
    return text


def first_canonical_text(*values: Any) -> Optional[str]:
    for value in values:
        text = canonicalize_context_text(value)
        if text:
            return text
    return None


def join_context_parts(*values: Any, separator: str = "\n\n") -> Optional[str]:
    parts: list[str] = []
    for value in values:
        text = canonicalize_context_text(value)
        if text:
            parts.append(text)
    if not parts:
        return None
    return separator.join(parts)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value if str(item or "").strip())
    return str(value)


def _recursive_unescape(value: str) -> str:
    previous = value
    for _ in range(5):
        current = unescape(previous)
        if current == previous:
            return current
        previous = current
    return previous


def _normalize_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _NEWLINE_PAD_RE.sub("\n", text)
    paragraphs = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    normalized: list[str] = []
    for paragraph in paragraphs:
        lines = [
            _WHITESPACE_RE.sub(" ", line).strip()
            for line in paragraph.split("\n")
            if line.strip()
        ]
        if lines:
            normalized.append("\n".join(lines))
    text = "\n\n".join(normalized)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _looks_like_garbage(value: str) -> bool:
    if not value.strip():
        return True
    if _ENTITY_FRAGMENT_RE.search(value):
        return True
    visible = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return not bool(visible)
