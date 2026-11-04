"""Data-plane helper for future DELETE_IMAGES cleanup.

This module is intentionally not wired into the document mainline. It only
wraps the independent image collection writer with stable failure semantics.
"""

from __future__ import annotations

from typing import Any, Optional

from app.observability.logger import get_logger
from app.services.milvus_image_service import DEFAULT_IMAGE_COLLECTION, MilvusImageService


logger = get_logger(__name__)

IMAGE_DELETE_INVALID_DOC_ID = "IMAGE_DELETE_INVALID_DOC_ID"
IMAGE_DELETE_VALIDATION_ERROR = "IMAGE_DELETE_VALIDATION_ERROR"
IMAGE_DELETE_WRITER_ERROR = "IMAGE_DELETE_WRITER_ERROR"


def delete_images_for_document(
    doc_id: str,
    *,
    writer: Optional[Any] = None,
    collection_name: Optional[str] = None,
) -> dict[str, Any]:
    """Delete all image records for a document from the image collection."""
    resolved_collection = collection_name or DEFAULT_IMAGE_COLLECTION
    normalized_doc_id = str(doc_id).strip() if doc_id is not None else ""
    if not normalized_doc_id:
        return _result(
            success=False,
            doc_id=normalized_doc_id,
            collection_name=resolved_collection,
            delete_result=None,
            error_code=IMAGE_DELETE_INVALID_DOC_ID,
            error_message="doc_id is required for image delete",
            retryable=False,
        )

    owns_writer = writer is None
    image_writer = writer
    try:
        if image_writer is None:
            image_writer = MilvusImageService()
        delete_result = image_writer.delete_images_by_doc_id(resolved_collection, normalized_doc_id)
        return _result(
            success=True,
            doc_id=normalized_doc_id,
            collection_name=resolved_collection,
            delete_result=delete_result,
            error_code=None,
            error_message=None,
            retryable=False,
        )
    except ValueError as exc:
        return _result(
            success=False,
            doc_id=normalized_doc_id,
            collection_name=resolved_collection,
            delete_result=None,
            error_code=IMAGE_DELETE_VALIDATION_ERROR,
            error_message=str(exc),
            retryable=False,
        )
    except Exception as exc:
        return _result(
            success=False,
            doc_id=normalized_doc_id,
            collection_name=resolved_collection,
            delete_result=None,
            error_code=IMAGE_DELETE_WRITER_ERROR,
            error_message=str(exc),
            retryable=True,
        )
    finally:
        if owns_writer and image_writer is not None:
            _close_writer_best_effort(image_writer)


def _result(
    *,
    success: bool,
    doc_id: str,
    collection_name: str,
    delete_result: Optional[dict[str, Any]],
    error_code: Optional[str],
    error_message: Optional[str],
    retryable: bool,
) -> dict[str, Any]:
    return {
        "success": success,
        "doc_id": doc_id,
        "collection_name": collection_name,
        "delete_result": delete_result,
        "error_code": error_code,
        "error_message": error_message,
        "retryable": retryable,
    }


def _close_writer_best_effort(writer: Any) -> None:
    close = getattr(writer, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:
        logger.warning("image delete writer close failed", error=str(exc))
