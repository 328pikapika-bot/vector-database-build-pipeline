"""Admin authentication and role checks."""
from __future__ import annotations

from secrets import compare_digest
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_admin_read(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict[str, str]:
    return _authorize("read", authorization)


async def require_admin_write(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict[str, str]:
    return _authorize("write", authorization)


def _authorize(required_role: str, authorization: Optional[str]) -> dict[str, str]:
    if not settings.ADMIN_AUTH_ENABLED:
        return {"role": "disabled"}

    token = _extract_bearer_token(authorization)
    if token is None:
        raise _unauthorized("admin authentication required")

    read_tokens = _split_tokens(settings.ADMIN_READ_TOKENS)
    write_tokens = _split_tokens(settings.ADMIN_WRITE_TOKENS)
    has_read = _token_matches(token, read_tokens)
    has_write = _token_matches(token, write_tokens)

    if required_role == "read" and (has_read or has_write):
        return {"role": "write" if has_write else "read"}
    if required_role == "write" and has_write:
        return {"role": "write"}
    if required_role == "write" and has_read:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin write role required",
        )

    raise _unauthorized("invalid admin token")


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def _split_tokens(raw_value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw_value.split(",") if token.strip())


def _token_matches(candidate: str, tokens: tuple[str, ...]) -> bool:
    return any(compare_digest(candidate, token) for token in tokens)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )
