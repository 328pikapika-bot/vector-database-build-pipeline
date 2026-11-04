"""Static resource URL helpers for NSTL .mmd and image files."""

from __future__ import annotations

from typing import Optional
from urllib.parse import unquote, urlsplit

from app.config import settings


class StaticResourceConfigurationError(RuntimeError):
    """Raised when the static resource prefix is not configured."""


def build_static_resource_url(
    relative_url: str,
    *,
    base_url: Optional[str] = None,
) -> str:
    """Join ATTACHMENT_PREFIX and a relative static resource path."""

    resolved_base_url = settings.ATTACHMENT_PREFIX if base_url is None else base_url
    resolved_base_url = str(resolved_base_url or "").strip()
    if not resolved_base_url:
        raise StaticResourceConfigurationError("ATTACHMENT_PREFIX is required")

    relative = str(relative_url or "").strip()
    if not relative:
        raise ValueError("relative_url is required")

    parsed = urlsplit(relative)
    if parsed.scheme or parsed.netloc:
        raise ValueError("relative_url must not be an absolute URL")
    decoded_path = unquote(parsed.path)
    if "\\" in relative or "\\" in decoded_path:
        raise ValueError("relative_url must use forward slashes")
    if any(ord(character) < 32 for character in decoded_path):
        raise ValueError("relative_url contains control characters")
    path_segments = [segment for segment in decoded_path.split("/") if segment]
    if any(segment in {".", ".."} for segment in path_segments):
        raise ValueError("relative_url contains an unsafe path segment")

    return f"{resolved_base_url.rstrip('/')}/{relative.lstrip('/')}"


def is_static_resource_configured(*, base_url: Optional[str] = None) -> bool:
    resolved_base_url = settings.ATTACHMENT_PREFIX if base_url is None else base_url
    return bool(str(resolved_base_url or "").strip())
