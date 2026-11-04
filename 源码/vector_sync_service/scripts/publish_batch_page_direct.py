#!/usr/bin/env python3
"""Directly call the upstream publishBatch endpoint for one or more pages."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:17123/api/service/interfaces/vector/job/publishBatch"
DEFAULT_SUB_DOC_TYPE = "A79"
DEFAULT_PAGE_PARAM = "pageNumer"
DEFAULT_TOPIC = "VECTOR_JOURNAL_ARTICLE_V2"
DEFAULT_TIMEOUT_SECONDS = 900


@dataclass
class PublishOutcome:
    page: int
    ok: bool
    status_code: int | None
    elapsed_seconds: float
    request_form: dict[str, str]
    response_json: Any | None
    response_text_excerpt: str
    error: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct publishBatch caller.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", required=True)
    parser.add_argument("--sub-doc-type", default=DEFAULT_SUB_DOC_TYPE)
    parser.add_argument("--page-param", default=DEFAULT_PAGE_PARAM)
    parser.add_argument("--start-page", type=int, required=True)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def decode_excerpt(body: bytes, limit: int = 1000) -> str:
    if not body:
        return ""
    return body.decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")[:limit]


def parse_json(body: bytes) -> Any | None:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def publish_page(
    *,
    url: str,
    token: str,
    sub_doc_type: str,
    page_param: str,
    page: int,
    topic: str,
    timeout_seconds: int,
) -> PublishOutcome:
    form = {
        "subDocType": sub_doc_type,
        page_param: str(page),
        "topic": topic,
    }
    request = Request(
        url,
        data=urlencode(form).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "vector-sync-direct-publish/1.0",
        },
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            elapsed = time.monotonic() - started
            return PublishOutcome(
                page=page,
                ok=True,
                status_code=int(response.status),
                elapsed_seconds=round(elapsed, 3),
                request_form=form,
                response_json=parse_json(body),
                response_text_excerpt=decode_excerpt(body),
                error=None,
            )
    except HTTPError as exc:
        body = exc.read()
        elapsed = time.monotonic() - started
        return PublishOutcome(
            page=page,
            ok=False,
            status_code=int(exc.code),
            elapsed_seconds=round(elapsed, 3),
            request_form=form,
            response_json=parse_json(body),
            response_text_excerpt=decode_excerpt(body),
            error=f"HTTP {exc.code}",
        )
    except (URLError, TimeoutError, socket.timeout) as exc:
        elapsed = time.monotonic() - started
        return PublishOutcome(
            page=page,
            ok=False,
            status_code=None,
            elapsed_seconds=round(elapsed, 3),
            request_form=form,
            response_json=None,
            response_text_excerpt="",
            error=str(exc),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_page < 1:
        raise SystemExit("--start-page must be >= 1")
    if args.pages < 1:
        raise SystemExit("--pages must be >= 1")

    outcomes = []
    for page in range(args.start_page, args.start_page + args.pages):
        outcomes.append(
            publish_page(
                url=args.url,
                token=args.token,
                sub_doc_type=args.sub_doc_type,
                page_param=args.page_param,
                page=page,
                topic=args.topic,
                timeout_seconds=args.timeout_seconds,
            )
        )

    payload = {
        "url": args.url,
        "topic": args.topic,
        "start_page": args.start_page,
        "pages": args.pages,
        "timeout_seconds": args.timeout_seconds,
        "results": [asdict(item) for item in outcomes],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item.ok for item in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
