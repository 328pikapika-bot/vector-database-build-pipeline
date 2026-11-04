"""CLI for the read-only training vector library probe."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.services.training_vector_library_probe import (
    DEFAULT_TRAINING_VECTOR_QUERY,
    json_safe,
    probe_training_vector_library,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe the read-only NSTL training vector collections.")
    parser.add_argument(
        "--query",
        default=DEFAULT_TRAINING_VECTOR_QUERY,
        help="Text query used for embedding-backed document, chunk, and image text-vector searches.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K search result count for each probe.")
    parser.add_argument(
        "--document-collection",
        default=settings.MILVUS_COLLECTION,
        help=f"Document collection name. Defaults to {settings.MILVUS_COLLECTION!r}.",
    )
    parser.add_argument(
        "--chunk-collection",
        default=settings.FULLTEXT_CHUNK_COLLECTION,
        help=f"Full-text chunk collection name. Defaults to {settings.FULLTEXT_CHUNK_COLLECTION!r}.",
    )
    parser.add_argument(
        "--image-collection",
        default=settings.IMAGE_COLLECTION_NAME,
        help=f"Image collection name. Defaults to {settings.IMAGE_COLLECTION_NAME!r}.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of pretty JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(
            probe_training_vector_library(
                query=args.query,
                top_k=args.top_k,
                document_collection=args.document_collection,
                chunk_collection=args.chunk_collection,
                image_collection=args.image_collection,
            )
        )
    except Exception as error:
        result = {
            "success": False,
            "read_only": True,
            "mutation_allowed": False,
            "errors": [
                {
                    "code": "PROBE_UNHANDLED_EXCEPTION",
                    "message": f"{type(error).__name__}: {error}",
                }
            ],
        }

    print(
        json.dumps(
            json_safe(result),
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0 if result.get("success") else 2


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
