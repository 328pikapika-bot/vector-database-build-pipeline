"""Resolve document and image work inputs from the current Solr indexes."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import httpx

from app.config import settings
from app.observability.logger import get_logger


logger = get_logger(__name__)
_LANG_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_,;-]{0,31}\|")
_MMD_SUFFIX_RE = re.compile(r"\.mmd(?:[?#].*)?$", re.IGNORECASE)

PAPER_SELECT_FIELDS = (
    "id",
    "type",
    "auto_stringITMS_title",
    "auto_stringITMS_abstract",
    "auto_stringITMS_keyword",
    "stringIS_year",
    "stringIMS_language",
    "stringIMS_accessType",
    "stringIMS_classificationCode",
    "auto_stringITMS_hasAuthorname",
    "auto_stringITMS_hasCreateOrganizationname",
    "stringOM_hasFullTextFile",
    # These optional historical metadata fields are retained when present.
    "stringIS_sysSubType",
    "stringIS_sysDataStatus",
    "dateIS_sysDataUpdateTime",
    "stringIS_sysSourceId",
    "stringIS_doi",
    "stringIMS_hasSourcesysSourceId",
    "stringIMS_hasSourceidSet",
    "stringOM_hasAuthor",
    "stringOM_hasCreateOrganization",
)

IMAGE_SELECT_FIELDS = (
    "id",
    "stringIS_sourceId",
    "stringIMS_url",
    "auto_stringITS_name",
    "stringIS_sequence",
    "intIS_width",
    "intIS_height",
    "stringIS_fileType",
    "intIS_size",
    "auto_stringITMS_title",
)


class SolrSourceError(RuntimeError):
    """The current source data could not be queried or normalized."""


@dataclass(frozen=True, slots=True)
class SolrResolvedDocument:
    """Normalized source representation consumed by all three V2 lanes."""

    doc_id: str
    payload: dict[str, Any]
    found: bool


class SolrSourceResolver:
    """Read authoritative document/image metadata from the two Solr cores."""

    def __init__(
        self,
        *,
        paper_url: str = "",
        image_url: str = "",
        timeout_sec: Optional[float] = None,
        request_max_retries: Optional[int] = None,
        request_retry_backoff_ms: Optional[int] = None,
        image_rows: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.paper_url = str(paper_url or settings.SOLR_PAPER_URL).rstrip("/")
        self.image_url = str(image_url or settings.SOLR_IMAGE_URL).rstrip("/")
        self.timeout_sec = float(timeout_sec or settings.SOLR_REQUEST_TIMEOUT_SEC)
        self.request_max_retries = max(
            int(
                settings.SOLR_REQUEST_MAX_RETRIES
                if request_max_retries is None
                else request_max_retries
            ),
            0,
        )
        self.request_retry_backoff_seconds = max(
            int(
                settings.SOLR_REQUEST_RETRY_BACKOFF_MS
                if request_retry_backoff_ms is None
                else request_retry_backoff_ms
            ),
            0,
        ) / 1000.0
        self.image_rows = max(int(image_rows or settings.SOLR_IMAGE_ROWS), 1)
        self._client = client
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def resolve_document(self, doc_id: str, *, include_images: bool) -> SolrResolvedDocument:
        normalized_doc_id = str(doc_id or "").strip()
        if not normalized_doc_id:
            raise SolrSourceError("doc_id is required")

        paper = await self._select_one(
            self.paper_url,
            f'id:"{_escape_solr_phrase(normalized_doc_id)}"',
            fields=PAPER_SELECT_FIELDS,
        )
        if paper is None:
            return SolrResolvedDocument(doc_id=normalized_doc_id, payload={}, found=False)

        images: list[dict[str, Any]] = []
        if include_images:
            images = await self._select_many(
                self.image_url,
                f"stringIS_sourceId:{_escape_solr_term(normalized_doc_id)}",
                rows=self.image_rows,
                fields=IMAGE_SELECT_FIELDS,
            )
        return SolrResolvedDocument(
            doc_id=normalized_doc_id,
            payload=_build_normalized_payload(paper, images),
            found=True,
        )

    async def _select_one(
        self,
        base_url: str,
        query: str,
        *,
        fields: Iterable[str],
    ) -> Optional[dict[str, Any]]:
        documents = await self._select_many(base_url, query, rows=1, fields=fields)
        return documents[0] if documents else None

    async def _select_many(
        self,
        base_url: str,
        query: str,
        *,
        rows: int,
        fields: Iterable[str],
    ) -> list[dict[str, Any]]:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec))
            self._client = client
        params = {
            "q": query,
            "wt": "json",
            "rows": rows,
            "fl": ",".join(dict.fromkeys(str(field) for field in fields if str(field))),
        }
        for attempt in range(self.request_max_retries + 1):
            try:
                response = await client.get(f"{base_url}/select", params=params)
                response.raise_for_status()
                body = response.json()
                break
            except (httpx.HTTPError, ValueError) as error:
                if not _is_retryable_solr_error(error) or attempt >= self.request_max_retries:
                    raise SolrSourceError(f"Solr query failed for {base_url}: {error}") from error
                delay = self.request_retry_backoff_seconds * (2 ** attempt)
                if delay:
                    await asyncio.sleep(delay)

        response_body = body.get("response") if isinstance(body, dict) else None
        docs = response_body.get("docs") if isinstance(response_body, dict) else None
        if not isinstance(docs, list):
            raise SolrSourceError(f"Solr response has no document list for {base_url}")
        return [dict(item) for item in docs if isinstance(item, dict)]


def _is_retryable_solr_error(error: Exception) -> bool:
    """Retry only transient source-read errors, never malformed source data."""
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == 408 or status_code == 429 or status_code >= 500
    return isinstance(error, httpx.RequestError)


def is_solr_doc_id_payload(payload: dict[str, Any]) -> bool:
    """Whether parse_message normalized a bare Kafka document ID."""
    return str(payload.get("source_mode") or "").strip() == "solr_doc_id"


def _build_normalized_payload(paper: dict[str, Any], images: list[dict[str, Any]]) -> dict[str, Any]:
    doc_id = _first_text(paper.get("id"))
    if not doc_id:
        raise SolrSourceError("paper Solr document has no id")

    scalar_map: dict[str, tuple[str, bool]] = {
        "title": ("auto_stringITMS_title", True),
        "abstract": ("auto_stringITMS_abstract", True),
        "keyword": ("auto_stringITMS_keyword", True),
        "year": ("stringIS_year", False),
        "sysSubType": ("stringIS_sysSubType", False),
        "language": ("stringIMS_language", True),
        "accessType": ("stringIMS_accessType", True),
        "classificationCode": ("stringIMS_classificationCode", True),
        "sysDataStatus": ("stringIS_sysDataStatus", False),
        "sysDataUpdateTime": ("dateIS_sysDataUpdateTime", False),
        "sysSourceId": ("stringIS_sysSourceId", False),
        "doi": ("stringIS_doi", False),
        "hasAuthorname": ("auto_stringITMS_hasAuthorname", True),
        "hasCreateOrganizationname": ("auto_stringITMS_hasCreateOrganizationname", True),
        "hasSourcesysSourceId": ("stringIMS_hasSourcesysSourceId", True),
        "hasSourceidSet": ("stringIMS_hasSourceidSet", True),
    }
    dp: dict[str, dict[str, Any]] = {}
    for target, (source, is_multi) in scalar_map.items():
        values = _solr_text_values(paper.get(source))
        if not values:
            # The current corpus uses these object fields in some source rows.
            fallback = {
                "hasAuthorname": "stringOM_hasAuthor",
                "hasCreateOrganizationname": "stringOM_hasCreateOrganization",
            }.get(target)
            if fallback:
                values = _solr_text_values(paper.get(fallback))
        if not values:
            continue
        dp[target] = {
            "value": values if is_multi else [values[0]],
            "multiValue": is_multi,
            "type": "string",
        }

    fulltext_relations = _fulltext_relations(paper)
    fulltext_names = [relation.get("name", "") for relation in fulltext_relations if relation.get("name")]
    fulltext_ids = [
        relation.get("id") or relation.get("fileId") or relation.get("sourceId") or ""
        for relation in fulltext_relations
    ]
    if fulltext_names:
        dp["hasFullTextFilename"] = {
            "value": fulltext_names,
            "multiValue": True,
            "type": "string",
        }
    if any(fulltext_ids):
        dp["hasFullTextFileid"] = {
            "value": [value for value in fulltext_ids if value],
            "multiValue": True,
            "type": "string",
        }
    image_relations = [_image_relation(image) for image in images]
    item: dict[str, Any] = {
        "id": doc_id,
        "sysId": doc_id,
        "cls": _first_text(paper.get("type")),
        "dp": dp,
        "op": {
            "hasFullTextFile": {"value": fulltext_relations},
            "hasPictureFile": {"value": image_relations},
        },
    }
    update_time = _first_text(paper.get("dateIS_sysDataUpdateTime"))
    if update_time:
        item["sysDataUpdateTime"] = update_time
    sub_doc_type = _first_text(paper.get("stringIS_sysSubType")) or _first_text(paper.get("type")) or "article"
    return {
        "success": True,
        "subDocType": sub_doc_type,
        "dataType": _first_text(paper.get("type")) or sub_doc_type,
        "data": [item],
        "operation": "UPDATE",
        "source_mode": "solr",
    }


def _fulltext_relations(paper: dict[str, Any]) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in _solr_values(paper.get("stringOM_hasFullTextFile")):
        parsed = _parse_fulltext_relation(raw)
        relative_url = parsed.get("url", "")
        if not relative_url or relative_url in seen or not _MMD_SUFFIX_RE.search(relative_url):
            continue
        seen.add(relative_url)
        relations.append(parsed)
    return relations


def _parse_fulltext_relation(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): _first_text(item) for key, item in value.items() if _first_text(item)}
    text = str(value or "").strip()
    result: dict[str, str] = {}
    for part in text.split("∷"):
        key, separator, raw_value = part.partition("≡")
        if separator and key.strip() and raw_value.strip():
            result[key.strip()] = raw_value.strip()
    if "url" not in result and _MMD_SUFFIX_RE.search(text):
        result["url"] = text
    return result


def _image_relation(image: dict[str, Any]) -> dict[str, Any]:
    relative_url = _first_text(image.get("stringIMS_url"))
    file_type = _first_text(image.get("stringIS_fileType"))
    return {
        "id": _first_text(image.get("id")),
        "url": relative_url,
        "name": (_solr_text_values(image.get("auto_stringITS_name")) or [_filename_from_url(relative_url)])[0],
        "sequence": _first_text(image.get("stringIS_sequence")),
        "width": _first_int(image.get("intIS_width")),
        "height": _first_int(image.get("intIS_height")),
        "fileType": file_type,
        "size": _first_int(image.get("intIS_size")),
        "title": (_solr_text_values(image.get("auto_stringITMS_title")) or [""])[0],
    }


def _solr_values(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _solr_text_values(value: Any) -> list[str]:
    result: list[str] = []
    for item in _solr_values(value):
        text = _first_text(item)
        if text:
            result.append(_LANG_PREFIX_RE.sub("", text))
    return result


def _first_text(value: Any) -> str:
    for item in _solr_values(value):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            return text
    return ""


def _first_int(value: Any) -> Optional[int]:
    text = _first_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _filename_from_url(value: str) -> str:
    return str(value or "").rstrip("/").rsplit("/", 1)[-1]


def _escape_solr_phrase(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _escape_solr_term(value: str) -> str:
    return re.sub(r"([+\-!(){}\[\]^\"~*?:\\/])", r"\\\1", str(value))
