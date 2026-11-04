"""Narrow external-service adapters. This module has no delete or drop path."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import requests

from .contract import FIELDS, SOLR_FIELDS, VECTOR_FIELDS, PermanentDataError, assert_write_payload, require_id, validate_value


LOG = logging.getLogger(__name__)


class TemporaryDependencyError(RuntimeError):
    pass


class SafetyPause(RuntimeError):
    pass


def _request_error(response: requests.Response, service: str) -> None:
    if response.status_code in (408, 429) or response.status_code >= 500:
        raise TemporaryDependencyError(f"{service} HTTP {response.status_code}")
    if response.status_code in (401, 403, 404):
        raise SafetyPause(f"{service} endpoint or permission is invalid (HTTP {response.status_code})")
    if response.status_code >= 400:
        raise PermanentDataError(f"{service} HTTP {response.status_code}")


class SolrReader:
    def __init__(self, url: str, timeout: float) -> None:
        self.url = url.rstrip("/") + "/select"
        self.timeout = timeout
        self.http = requests.Session()
        # These endpoints are internal services; ambient desktop proxy variables
        # otherwise route Solr requests to an unrelated proxy (observed HTTP 503).
        self.http.trust_env = False

    def fetch(self, doc_id: str) -> dict[str, Any] | None:
        doc_id = require_id(doc_id)
        try:
            response = self.http.get(self.url, params={
                "q": f'id:"{doc_id}"', "rows": 2, "fl": ",".join(SOLR_FIELDS), "wt": "json",
            }, timeout=self.timeout)
            _request_error(response, "Solr")
            body = response.json()
            header = body.get("responseHeader", {})
            if header.get("partialResults") or header.get("status", 0) != 0:
                raise TemporaryDependencyError("Solr returned partial or nonzero-status response")
            docs = body["response"]["docs"]
            if not isinstance(docs, list) or len(docs) > 1:
                raise SafetyPause("Solr returned duplicate or malformed ID results")
            if docs and require_id(docs[0].get("id")) != doc_id:
                raise SafetyPause("Solr returned an unrequested document ID")
            return docs[0] if docs else None
        except (requests.Timeout, requests.ConnectionError) as error:
            raise TemporaryDependencyError(f"Solr request unavailable: {type(error).__name__}") from error
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, (SafetyPause, PermanentDataError)):
                raise
            raise TemporaryDependencyError("Solr response is malformed") from error

    def fetch_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(require_id(doc_id) for doc_id in doc_ids))
        if not ids:
            return {}
        query = "id:(" + " OR ".join(ids) + ")"
        try:
            response = self.http.get(self.url, params={
                "q": query, "rows": len(ids) + 1, "fl": ",".join(SOLR_FIELDS), "wt": "json",
            }, timeout=self.timeout)
            _request_error(response, "Solr")
            body = response.json()
            header = body.get("responseHeader", {})
            if header.get("partialResults") or header.get("status", 0) != 0:
                raise TemporaryDependencyError("Solr returned partial or nonzero-status response")
            docs = body["response"]["docs"]
            if not isinstance(docs, list) or len(docs) > len(ids):
                raise SafetyPause("Solr returned duplicate or malformed batch results")
            requested = set(ids)
            result: dict[str, dict[str, Any]] = {}
            for doc in docs:
                doc_id = require_id(doc.get("id"))
                if doc_id not in requested or doc_id in result:
                    raise SafetyPause("Solr batch returned an unexpected or duplicate ID")
                result[doc_id] = doc
            return result
        except (requests.Timeout, requests.ConnectionError) as error:
            raise TemporaryDependencyError(f"Solr batch unavailable: {type(error).__name__}") from error
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, (SafetyPause, PermanentDataError, TemporaryDependencyError)):
                raise
            raise TemporaryDependencyError("Solr batch response is malformed") from error

    def close(self) -> None:
        self.http.close()


class EmbeddingReader:
    def __init__(self, url: str, model: str, api_key: str, timeout: float) -> None:
        self.url, self.model, self.timeout = url, model, timeout
        self.http = requests.Session()
        self.http.trust_env = False
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, inputs: dict[str, str]) -> dict[str, list[float]]:
        names = list(inputs)
        if not names:
            return {}
        try:
            response = self.http.post(self.url, json={
                "model": self.model, "input": [inputs[name] for name in names],
            }, timeout=self.timeout)
            _request_error(response, "Embedding")
            data = response.json()["data"]
            if len(data) != len(names):
                raise TemporaryDependencyError("Embedding count differs from requested text count")
            indexed: dict[int, list[float]] = {}
            for index, item in enumerate(data):
                position = int(item.get("index", index))
                vector = item["embedding"]
                if position in indexed:
                    raise TemporaryDependencyError("Embedding response contains duplicate index")
                indexed[position] = vector
            result = {field + "_embed": indexed[index] for index, field in enumerate(names)}
            for name, vector in result.items():
                validate_value(name, vector)
                # Do not send NaN, Infinity, or an all-zero vector to an indexed field.
                if not any(math.fabs(float(value)) > 0 for value in vector):
                    raise PermanentDataError(f"{name} is all zero")
            return result
        except (requests.Timeout, requests.ConnectionError) as error:
            raise TemporaryDependencyError(f"Embedding request unavailable: {type(error).__name__}") from error
        except (KeyError, IndexError, TypeError, ValueError) as error:
            if isinstance(error, (TemporaryDependencyError, PermanentDataError)):
                raise
            raise TemporaryDependencyError("Embedding response is malformed") from error

    def embed_many(self, documents: dict[str, dict[str, str]]) -> dict[str, dict[str, list[float]]]:
        mapping: list[tuple[str, str]] = []
        inputs: list[str] = []
        result = {doc_id: {} for doc_id in documents}
        for doc_id, fields in documents.items():
            for field, text in fields.items():
                mapping.append((doc_id, field + "_embed"))
                inputs.append(text)
        if not inputs:
            return result
        try:
            response = self.http.post(self.url, json={"model": self.model, "input": inputs}, timeout=self.timeout)
            _request_error(response, "Embedding")
            data = response.json()["data"]
            if len(data) != len(mapping):
                raise TemporaryDependencyError("Embedding count differs from batch input count")
            seen: set[int] = set()
            for default_index, item in enumerate(data):
                index = int(item.get("index", default_index))
                if index < 0 or index >= len(mapping) or index in seen:
                    raise TemporaryDependencyError("Embedding batch contains invalid indexes")
                seen.add(index)
                doc_id, field = mapping[index]
                vector = item["embedding"]
                validate_value(field, vector)
                if not any(math.fabs(float(value)) > 0 for value in vector):
                    raise PermanentDataError(f"{field} is all zero")
                result[doc_id][field] = vector
            return result
        except (requests.Timeout, requests.ConnectionError) as error:
            raise TemporaryDependencyError(f"Embedding batch unavailable: {type(error).__name__}") from error
        except (KeyError, IndexError, TypeError, ValueError) as error:
            if isinstance(error, (TemporaryDependencyError, PermanentDataError)):
                raise
            raise TemporaryDependencyError("Embedding batch response is malformed") from error

    def close(self) -> None:
        self.http.close()


def _kind(field: dict[str, Any]) -> str:
    value = field.get("type", field.get("data_type", field.get("dataType")))
    if str(value).upper() in {"21", "VARCHAR", "DATATYPE.VARCHAR"}:
        return "varchar"
    if str(value).upper() in {"22", "ARRAY", "DATATYPE.ARRAY"}:
        return "array"
    if str(value).upper() in {"101", "FLOAT_VECTOR", "DATATYPE.FLOAT_VECTOR"}:
        return "vector"
    return str(value)


def _param(field: dict[str, Any], key: str) -> int:
    params = field.get("params") or field.get("type_params") or {}
    if isinstance(params, list):
        params = {item.get("key"): item.get("value") for item in params}
    value = params.get(key, field.get(key, 0))
    return int(value or 0)


def check_schema(description: dict[str, Any]) -> None:
    actual = {field["name"]: field for field in description.get("fields", [])}
    if set(actual) != set(FIELDS):
        raise SafetyPause(f"Milvus schema differs from 41-field contract: missing={sorted(set(FIELDS)-set(actual))}, extra={sorted(set(actual)-set(FIELDS))}")
    if description.get("enable_dynamic_field") is True:
        raise SafetyPause("dynamic fields are enabled unexpectedly")
    for name, expected in FIELDS.items():
        field = actual[name]
        if _kind(field) != expected.kind:
            raise SafetyPause(f"Milvus field {name} has unexpected type")
        if expected.kind in ("varchar", "array") and _param(field, "max_length") != expected.max_length:
            raise SafetyPause(f"Milvus field {name} has unexpected max_length")
        if expected.kind == "array" and _param(field, "max_capacity") != expected.max_capacity:
            raise SafetyPause(f"Milvus field {name} has unexpected max_capacity")
        if expected.kind == "vector" and _param(field, "dim") != expected.dim:
            raise SafetyPause(f"Milvus field {name} has unexpected dimension")
        if name == "id" and not (field.get("is_primary") or field.get("is_primary_key")):
            raise SafetyPause("id is not the Milvus primary key")
        if name != "id" and expected.kind != "vector" and field.get("nullable") is False:
            raise SafetyPause(f"Milvus field {name} is not nullable")


def _equal(left: Any, right: Any) -> bool:
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(x, y) for x, y in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)
    return left == right


class MilvusTarget:
    """Single-target adapter with Strong read-before-write and read-after-write."""

    def __init__(self, uri: str, db: str, collection: str, token: str, timeout: float) -> None:
        from pymilvus import MilvusClient
        self.collection, self.timeout = collection, timeout
        options = {"uri": uri, "db_name": db, "timeout": timeout}
        if token:
            options["token"] = token
        self.client = MilvusClient(**options)

    def check(self) -> dict[str, Any]:
        if not self.client.has_collection(self.collection):
            raise SafetyPause("target Milvus collection is absent")
        description = self.client.describe_collection(self.collection)
        check_schema(description)
        if not description.get("collection_id"):
            raise SafetyPause("Milvus description has no collection identity")
        indexes = self.client.list_indexes(self.collection)
        indexed_fields = set()
        for index in indexes:
            data = self.client.describe_index(self.collection, index)
            indexed_fields.add(data.get("field_name"))
        if not set(VECTOR_FIELDS) <= indexed_fields:
            raise SafetyPause("one or more original vector indexes are absent")
        return {"collection_id": str(description["collection_id"]),
                "fields": len(description["fields"]), "indexes": len(indexes)}

    def get(self, doc_id: str) -> dict[str, Any] | None:
        try:
            rows = self.client.get(self.collection, ids=[doc_id],
                                   output_fields=list(FIELDS), consistency_level="Strong", timeout=self.timeout)
        except Exception as error:
            raise TemporaryDependencyError("Milvus Strong read unavailable") from error
        if len(rows) > 1:
            raise SafetyPause("Strong read returned duplicate primary keys")
        if rows and require_id(rows[0].get("id")) != doc_id:
            raise SafetyPause("Strong read returned a different primary key")
        if not rows:
            return None
        row = dict(rows[0])
        # PyMilvus can return ARRAY values as protobuf RepeatedScalarContainer,
        # which is not JSON serializable and has surprising list comparison behavior.
        for name, spec in FIELDS.items():
            if spec.kind == "array" and row.get(name) is not None:
                row[name] = list(row[name])
        return row

    def get_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(require_id(doc_id) for doc_id in doc_ids))
        if not ids:
            return {}
        try:
            rows = self.client.get(self.collection, ids=ids, output_fields=list(FIELDS),
                                   consistency_level="Strong", timeout=self.timeout)
        except Exception as error:
            raise TemporaryDependencyError("Milvus Strong batch read unavailable") from error
        result: dict[str, dict[str, Any]] = {}
        requested = set(ids)
        for raw in rows:
            row = dict(raw)
            doc_id = require_id(row.get("id"))
            if doc_id not in requested or doc_id in result:
                raise SafetyPause("Milvus Strong batch read returned an unexpected or duplicate ID")
            for name, spec in FIELDS.items():
                if spec.kind == "array" and row.get(name) is not None:
                    row[name] = list(row[name])
            result[doc_id] = row
        return result

    def write_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        max_write_rows = 200
        new_records = [item for item in records if item["new"]]
        updates: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for item in records:
            assert_write_payload(item["payload"], new=item["new"])
            if item["new"] != (item["before"] is None):
                raise SafetyPause("batch write branch does not match Strong existence checks")
            if not item["new"]:
                updates.setdefault(tuple(sorted(item["payload"])), []).append(item)
        try:
            for start in range(0, len(new_records), max_write_rows):
                self.client.insert(collection_name=self.collection,
                    data=[item["payload"] for item in new_records[start:start + max_write_rows]],
                    timeout=self.timeout)
            for group in updates.values():
                for start in range(0, len(group), max_write_rows):
                    self.client.upsert(collection_name=self.collection,
                        data=[item["payload"] for item in group[start:start + max_write_rows]],
                        partial_update=True, timeout=self.timeout)
        except Exception:
            # Batch RPC results can be ambiguous. Strong-read each row and use the
            # existing idempotent single-row verifier before deciding what to retry.
            for item in records:
                self._recover_batch_item(item)
            return

        ids = [item["id"] for item in records]
        for start in range(0, len(ids), 50):
            subset = records[start:start + 50]
            after = self.get_many([item["id"] for item in subset])
            for item in subset:
                row = after.get(item["id"])
                if row is not None and self._row_matches(item, row):
                    continue
                self._recover_batch_item(item, current=row)

    def _row_matches(self, item: dict[str, Any], row: dict[str, Any]) -> bool:
        for name, value in item["payload"].items():
            if not _equal(row.get(name), value):
                return False
        before = item["before"]
        if before is not None:
            for name in FIELDS:
                if name not in item["payload"] and not _equal(before.get(name), row.get(name)):
                    raise SafetyPause(f"unchanged historical field {name} differs after batch update")
        return True

    def _recover_batch_item(self, item: dict[str, Any], current: dict[str, Any] | None = None) -> None:
        row = current if current is not None else self.get(item["id"])
        if row is not None and self._row_matches(item, row):
            return
        if item["new"] and row is not None:
            raise SafetyPause("new-record batch insert has an unexpected existing row")
        if not item["new"] and row is None:
            raise SafetyPause("existing record disappeared during batch-write recovery; refusing to recreate it")
        self.write(item["id"], item["before"], item["payload"], new=item["new"])

    def _verify(self, doc_id: str, before: dict[str, Any] | None,
                payload: dict[str, Any]) -> bool:
        after = self.get(doc_id)
        if not after:
            return False
        for name, value in payload.items():
            if not _equal(after.get(name), value):
                return False
        if before is not None:
            for name in FIELDS:
                if name not in payload and not _equal(before.get(name), after.get(name)):
                    raise SafetyPause(f"unchanged historical field {name} differs after update")
        return True

    def write(self, doc_id: str, before: dict[str, Any] | None,
              payload: dict[str, Any], *, new: bool) -> None:
        assert_write_payload(payload, new=new)
        if (new and before is not None) or (not new and before is None):
            raise SafetyPause("insert/update branch does not match the Strong existence check")
        try:
            if new:
                self.client.insert(collection_name=self.collection, data=[payload], timeout=self.timeout)
            else:
                self.client.upsert(collection_name=self.collection, data=[payload],
                                   partial_update=True, timeout=self.timeout)
        except Exception as error:
            # The server might have applied the write despite a client timeout.
            for _ in range(3):
                time.sleep(2)
                try:
                    if self._verify(doc_id, before, payload):
                        return
                except SafetyPause:
                    raise
                except Exception:
                    continue
            if new:
                raise SafetyPause("new-record insert result is uncertain; refusing a duplicate insert") from error
            raise TemporaryDependencyError("Milvus update result is uncertain; will Strong-read before retry") from error
        if not self._verify(doc_id, before, payload):
            raise SafetyPause("Milvus write returned success but Strong readback differs")

    def close(self) -> None:
        self.client.close()
