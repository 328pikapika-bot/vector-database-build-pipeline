"""Service-native image providers adapted from the offline ingestion script."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import random
import re
import time
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import httpx
import numpy as np

from app.config import settings
from app.observability.metrics import image_stage_latency_seconds
from app.services.image_feature_pipeline import (
    ImageFeaturePipelineContext,
    ImageFeatureProviders,
)
from app.services.image_features import compute_phash
from app.services.embedding_rate_limiter import build_text_rate_limiter


IMAGE_PROVIDER_CONFIG_MISSING = "IMAGE_PROVIDER_CONFIG_MISSING"
IMAGE_PROVIDER_DEPENDENCY_MISSING = "IMAGE_PROVIDER_DEPENDENCY_MISSING"
IMAGE_PROVIDER_HTTP_ERROR = "IMAGE_PROVIDER_HTTP_ERROR"
IMAGE_PROVIDER_NETWORK_ERROR = "IMAGE_PROVIDER_NETWORK_ERROR"
IMAGE_PROVIDER_RESPONSE_INVALID = "IMAGE_PROVIDER_RESPONSE_INVALID"
IMAGE_ANALYSIS_TEXT_MISSING = "IMAGE_ANALYSIS_TEXT_MISSING"
IMAGE_CAPTION_TEXT_MISSING = "IMAGE_CAPTION_TEXT_MISSING"
IMAGE_CONTEXT_TEXT_MISSING = "IMAGE_CONTEXT_TEXT_MISSING"
IMAGE_VL_EMBED_INPUT_TOO_LARGE = "IMAGE_VL_EMBED_INPUT_TOO_LARGE"

DRAFT_SYSTEM_PROMPT = (
    "You are a scientific image analysis expert for semantic retrieval. "
    "Analyze the image with the provided paper context and return only one JSON object "
    "containing domain_relevance, basic_info, surface_extraction, deep_semantics, "
    "scientific_entities, and search_keywords. Do not use markdown, code fences, or "
    "explanatory prose. Be factual and avoid unsupported claims."
)
CONTEXT_CLEAN_PROMPT = (
    "Clean noisy scientific paper context before image analysis. Remove meaningless OCR "
    "garbage, but preserve formulas, chemical expressions, quantities, units, labels, "
    "and other useful scientific statements even when they appear as isolated fragments. "
    "Return only one JSON object with cleaned_context, removed_items, and removal_ratio. "
    "Do not use markdown, code fences, or explanatory prose."
)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
_DATA_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\r\n]+")


class ImageProviderConfigurationError(RuntimeError):
    """Raised when real image providers are requested without full configuration."""

    error_code = IMAGE_PROVIDER_CONFIG_MISSING
    retryable = False

    def __init__(self, missing_env_names: list[str]) -> None:
        self.missing_env_names = missing_env_names
        super().__init__(
            "missing image provider configuration: " + ", ".join(missing_env_names)
        )


class ImageProviderError(RuntimeError):
    """Provider error with retry metadata consumed by the feature pipeline."""

    def __init__(self, error_code: str, message: str, *, retryable: bool) -> None:
        self.error_code = error_code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ImageInterpretation:
    raw_json: dict[str, Any]
    deep_text: str
    keyword_text: str
    context_cleaning_fallback: bool = False
    context_cleaning_error: str = ""
    context_cleaning_preview: str = ""
    context_cleaning_raw_length: int = 0
    draft_chat_ms: int = 0
    vl_context_chars: int = 0
    vl_context_truncated: bool = False

    @property
    def combined_text(self) -> str:
        return "\n".join(
            part for part in (self.deep_text.strip(), self.keyword_text.strip()) if part
        )


@dataclass(frozen=True, slots=True)
class ContextCleaningResult:
    cleaned_context: str
    fallback: bool = False
    error_message: str = ""
    raw_preview: str = ""
    raw_length: int = 0


class OfflineImageModelClient:
    """Async client for the VL and text model APIs used by the offline script."""

    def __init__(
        self,
        *,
        api_key: str = "",
        vl_chat_api_key: str = "",
        vl_embed_api_key: str = "",
        text_embed_api_key: str = "",
        vl_chat_url: str = "",
        vl_chat_model: str = "",
        vl_embed_url: str = "",
        vl_embed_model: str = "",
        text_embed_url: str = "",
        text_embed_model: str = "",
        temperature: float = 0.2,
        max_tokens: int = 5000,
        timeout: int = 60,
        max_retries: int = 3,
        max_pixels: int = 2_000_000,
        context_max_chars: int = 6000,
        http_client: Optional[httpx.AsyncClient] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        shared_api_key = api_key or settings.IMAGE_MODEL_API_KEY
        self.api_key = shared_api_key
        self.vl_chat_api_key = (
            vl_chat_api_key or settings.IMAGE_VL_CHAT_API_KEY or shared_api_key
        )
        self.vl_embed_api_key = (
            vl_embed_api_key or settings.IMAGE_VL_EMBED_API_KEY or shared_api_key
        )
        self.text_embed_api_key = (
            text_embed_api_key
            or settings.IMAGE_TEXT_EMBED_API_KEY
            or shared_api_key
            or settings.EMBEDDING_API_KEY
        )
        self.vl_chat_url = vl_chat_url or settings.IMAGE_VL_CHAT_URL
        self.vl_chat_model = vl_chat_model or settings.IMAGE_VL_CHAT_MODEL
        self.vl_embed_url = vl_embed_url or settings.IMAGE_VL_EMBED_URL
        self.vl_embed_model = vl_embed_model or settings.IMAGE_VL_EMBED_MODEL
        self.text_embed_url = text_embed_url or settings.IMAGE_TEXT_EMBED_URL
        self.text_embed_model = text_embed_model or settings.IMAGE_TEXT_EMBED_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(max_retries, 0)
        self.max_pixels = max(max_pixels, 1)
        self.context_max_chars = max(int(context_max_chars or 0), 0)
        self._http_client = http_client
        self._sleep = sleep
        self._image_preprocess_semaphore = asyncio.Semaphore(
            _positive_setting("IMAGE_PREPROCESS_CONCURRENCY", 8)
        )
        self._text_rate_limiter = build_text_rate_limiter(
            enabled=bool(getattr(settings, "EMBEDDING_RATE_LIMIT_ENABLED", False)),
            rate_per_second=float(
                getattr(
                    settings,
                    "IMAGE_TEXT_EMBED_RATE",
                    getattr(settings, "EMBEDDING_TEXT_RATE", 0.0),
                )
                or 0.0
            ),
            burst=int(getattr(settings, "EMBEDDING_TEXT_BURST", 32) or 32),
        )

    @classmethod
    def from_settings(
        cls,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> "OfflineImageModelClient":
        return cls(
            api_key=settings.IMAGE_MODEL_API_KEY,
            vl_chat_api_key=settings.IMAGE_VL_CHAT_API_KEY,
            vl_embed_api_key=settings.IMAGE_VL_EMBED_API_KEY,
            text_embed_api_key=settings.IMAGE_TEXT_EMBED_API_KEY,
            vl_chat_url=settings.IMAGE_VL_CHAT_URL,
            vl_chat_model=settings.IMAGE_VL_CHAT_MODEL,
            vl_embed_url=settings.IMAGE_VL_EMBED_URL,
            vl_embed_model=settings.IMAGE_VL_EMBED_MODEL,
            text_embed_url=settings.IMAGE_TEXT_EMBED_URL,
            text_embed_model=settings.IMAGE_TEXT_EMBED_MODEL,
            temperature=settings.IMAGE_VL_TEMPERATURE,
            max_tokens=settings.IMAGE_VL_CHAT_MAX_TOKENS,
            timeout=settings.IMAGE_FEATURE_TIMEOUT,
            max_retries=settings.IMAGE_FEATURE_MAX_RETRIES,
            max_pixels=settings.IMAGE_MAX_PIXELS,
            context_max_chars=settings.IMAGE_VL_CONTEXT_MAX_CHARS,
            http_client=http_client,
            sleep=sleep,
        )

    def missing_configuration(
        self,
        *,
        require_visual_embed: bool = True,
        require_vl_chat: bool = False,
    ) -> list[str]:
        required: dict[str, str] = {}
        if require_visual_embed:
            required.update(
                {
                    "IMAGE_VL_EMBED_API_KEY|IMAGE_MODEL_API_KEY": self.vl_embed_api_key,
                    "IMAGE_VL_EMBED_URL": self.vl_embed_url,
                    "IMAGE_VL_EMBED_MODEL": self.vl_embed_model,
                }
            )
        required.update(
            {
                "IMAGE_TEXT_EMBED_API_KEY|IMAGE_MODEL_API_KEY|EMBEDDING_API_KEY": self.text_embed_api_key,
                "IMAGE_TEXT_EMBED_URL": self.text_embed_url,
                "IMAGE_TEXT_EMBED_MODEL": self.text_embed_model,
            }
        )
        if require_vl_chat:
            required.update(
                {
                    "IMAGE_VL_CHAT_API_KEY|IMAGE_MODEL_API_KEY": self.vl_chat_api_key,
                    "IMAGE_VL_CHAT_URL": self.vl_chat_url,
                    "IMAGE_VL_CHAT_MODEL": self.vl_chat_model,
                }
            )
        return [name for name, value in required.items() if not str(value or "").strip()]

    async def embed_image(self, image_bytes: bytes, *, data_url: Optional[str] = None) -> list[float]:
        data_url = data_url or await self._prepare_data_url(image_bytes)
        payload = {
            "model": self.vl_embed_model,
            "input": _vl_embedding_input(data_url),
            "encoding_format": "float",
        }
        started = time.perf_counter()
        data = await self._post_json(
            self.vl_embed_url,
            payload,
            api_key=self.vl_embed_api_key,
        )
        image_stage_latency_seconds.labels(stage="vl_embed").observe(time.perf_counter() - started)
        return _normalized_vector(_extract_embedding(data))

    async def embed_image_data_urls(self, data_urls: Sequence[str]) -> list[list[float]]:
        resolved_data_urls = [str(data_url or "") for data_url in data_urls]
        if not resolved_data_urls:
            return []

        async def embed_one(data_url: str) -> list[float]:
            payload = {
                "model": self.vl_embed_model,
                "input": _vl_embedding_input(data_url),
                "encoding_format": "float",
            }
            data = await self._post_json(
                self.vl_embed_url,
                payload,
                api_key=self.vl_embed_api_key,
            )
            return _normalized_vector(_extract_embedding(data))

        started = time.perf_counter()
        try:
            return list(await asyncio.gather(*(embed_one(data_url) for data_url in resolved_data_urls)))
        finally:
            image_stage_latency_seconds.labels(stage="vl_embed").observe(time.perf_counter() - started)

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        resolved_texts = [str(text or "") for text in texts]
        if not resolved_texts:
            return []
        if self._text_rate_limiter is not None:
            await self._text_rate_limiter.acquire(len(resolved_texts))
        payload = {
            "model": self.text_embed_model,
            "input": resolved_texts[0] if len(resolved_texts) == 1 else resolved_texts,
            "encoding_format": "float",
        }
        started = time.perf_counter()
        data = await self._post_json(
            self.text_embed_url,
            payload,
            api_key=self.text_embed_api_key,
        )
        image_stage_latency_seconds.labels(stage="text_embed").observe(time.perf_counter() - started)
        return [_normalized_vector(item) for item in _extract_embeddings(data, len(resolved_texts))]

    async def embed_image_text(
        self,
        image_bytes: bytes,
        text: str,
        *,
        data_url: Optional[str] = None,
    ) -> list[float]:
        data_url = data_url or await self._prepare_data_url(image_bytes)
        payload = {
            "model": self.vl_embed_model,
            "input": _vl_embedding_input(data_url, text=text),
            "encoding_format": "float",
        }
        started = time.perf_counter()
        data = await self._post_json(
            self.vl_embed_url,
            payload,
            api_key=self.vl_embed_api_key,
        )
        image_stage_latency_seconds.labels(stage="vl_embed").observe(time.perf_counter() - started)
        return _normalized_vector(_extract_embedding(data))

    async def interpret_image(
        self,
        image_bytes: bytes,
        context_text: str,
        *,
        data_url: Optional[str] = None,
    ) -> ImageInterpretation:
        data_url = data_url or await self._prepare_data_url(image_bytes)
        context_cleaning = await self._clean_context_with_diagnostics(context_text)
        cleaned_context, context_truncated = _trim_context_text(
            context_cleaning.cleaned_context,
            self.context_max_chars,
        )
        draft_content = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {
                "type": "text",
                "text": (
                    f"Context:\n{cleaned_context}"
                    if cleaned_context
                    else "Analyze this image without additional context."
                ),
            },
        ]
        draft_started = time.perf_counter()
        draft_raw = await self._chat_json_text(DRAFT_SYSTEM_PROMPT, draft_content)
        draft_chat_ms = _elapsed_ms(draft_started)
        image_stage_latency_seconds.labels(stage="vl_chat").observe(draft_chat_ms / 1000)
        corrected = _loads_json_object(draft_raw, "draft analysis")
        deep_text = str(corrected.get("deep_semantics") or "").strip()
        keywords = corrected.get("search_keywords") or []
        if not isinstance(keywords, list):
            keywords = [keywords]
        keyword_text = " ".join(str(item).strip() for item in keywords if str(item).strip())
        return ImageInterpretation(
            raw_json=corrected,
            deep_text=deep_text,
            keyword_text=keyword_text,
            context_cleaning_fallback=context_cleaning.fallback,
            context_cleaning_error=context_cleaning.error_message,
            context_cleaning_preview=context_cleaning.raw_preview,
            context_cleaning_raw_length=context_cleaning.raw_length,
            draft_chat_ms=draft_chat_ms,
            vl_context_chars=len(cleaned_context),
            vl_context_truncated=context_truncated,
        )

    async def _prepare_data_url(self, image_bytes: bytes) -> str:
        return str(
            await _run_in_thread_with_semaphore(
                self._image_preprocess_semaphore,
                image_bytes_to_data_url,
                image_bytes,
                max_pixels=self.max_pixels,
            )
        )

    async def clean_context(self, context_text: str) -> str:
        return (await self._clean_context_with_diagnostics(context_text)).cleaned_context

    async def _clean_context_with_diagnostics(self, context_text: str) -> ContextCleaningResult:
        raw_context = str(context_text or "").strip()
        if not raw_context:
            return ContextCleaningResult(cleaned_context="")
        if not settings.IMAGE_CONTEXT_CLEAN_ENABLED:
            return ContextCleaningResult(cleaned_context=raw_context)
        user_content = [
            {
                "type": "text",
                "text": "Raw context:\n" + raw_context,
            }
        ]
        try:
            raw = await self._chat_json_text(CONTEXT_CLEAN_PROMPT, user_content)
        except ImageProviderError as exc:
            return ContextCleaningResult(
                cleaned_context=raw_context,
                fallback=True,
                error_message=str(exc),
            )

        try:
            result = _loads_json_object(raw, "context cleaning")
        except ImageProviderError as exc:
            return ContextCleaningResult(
                cleaned_context=raw_context,
                fallback=True,
                error_message=str(exc),
                raw_preview=_safe_text_preview(raw),
                raw_length=_text_length(raw),
            )
        cleaned = str(result.get("cleaned_context") or "").strip()
        removed_items = result.get("removed_items") or []
        if not cleaned and not removed_items:
            return ContextCleaningResult(
                cleaned_context=raw_context,
                fallback=True,
                error_message="context cleaning returned empty result without removal evidence",
                raw_preview=_safe_text_preview(raw),
                raw_length=_text_length(raw),
            )
        return ContextCleaningResult(cleaned_context=cleaned or raw_context)

    async def _chat_json_text(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.vl_chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = await self._post_json(
            self.vl_chat_url,
            payload,
            api_key=self.vl_chat_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        try:
            return _chat_content_to_text(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ImageProviderError(
                IMAGE_PROVIDER_RESPONSE_INVALID,
                f"unexpected chat API response: {_safe_response_preview(data)}",
                retryable=True,
            ) from exc

    async def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        api_key: str = "",
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error: Optional[Exception] = None
        resolved_retries = self.max_retries if max_retries is None else max(int(max_retries), 0)
        for attempt in range(resolved_retries + 1):
            try:
                response = await self._request_once(url, payload, headers, timeout=timeout)
                if response.status_code != 200:
                    response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                retryable = _is_retryable_status(error.response.status_code)
                last_error = error
                if not retryable:
                    raise ImageProviderError(
                        f"{IMAGE_PROVIDER_HTTP_ERROR}_{error.response.status_code}",
                        _safe_http_error_message(error),
                        retryable=False,
                    ) from error
            except _RETRYABLE_NETWORK_ERRORS as error:
                last_error = error
            except ValueError as error:
                raise ImageProviderError(
                    IMAGE_PROVIDER_RESPONSE_INVALID,
                    str(error),
                    retryable=False,
                ) from error

            if attempt >= resolved_retries:
                break
            await self._sleep(_backoff_delay(attempt))

        raise ImageProviderError(
            IMAGE_PROVIDER_NETWORK_ERROR,
            str(last_error),
            retryable=True,
        ) from last_error

    async def _request_once(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        *,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        resolved_timeout = httpx.Timeout(timeout or self.timeout)
        if self._http_client is not None:
            return await self._http_client.post(
                url,
                json=dict(payload),
                headers=dict(headers),
                timeout=resolved_timeout,
            )
        async with httpx.AsyncClient(timeout=resolved_timeout) as client:
            return await client.post(url, json=dict(payload), headers=dict(headers))


def _vl_embedding_input(data_url: str, *, text: str = "") -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "image_url", "image_url": {"url": data_url}})
    return [{"role": "user", "content": content}]


class _VisualPriorityModelGate:
    """Bound total model traffic and let visual requests pass first.

    The visual and text models may share one OpenAI-compatible gateway. Two
    independent semaphores therefore allow their combined in-flight requests
    to exceed the gateway's useful capacity. Text requests also must not hold
    the last slots while a visual request is waiting.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(int(limit or 1), 1)
        self._active = 0
        self._waiting_visual = 0
        self._condition = asyncio.Condition()

    async def run(self, awaitable: Awaitable[Any], *, priority: str) -> Any:
        visual = priority == "visual"
        async with self._condition:
            if visual:
                self._waiting_visual += 1
            try:
                while self._active >= self._limit or (
                    not visual and self._waiting_visual > 0
                ):
                    await self._condition.wait()
                if visual:
                    self._waiting_visual -= 1
                self._active += 1
            except BaseException:
                if visual:
                    self._waiting_visual = max(self._waiting_visual - 1, 0)
                self._condition.notify_all()
                close = getattr(awaitable, "close", None)
                if close is not None:
                    close()
                raise

        try:
            return await awaitable
        finally:
            async with self._condition:
                self._active = max(self._active - 1, 0)
                self._condition.notify_all()


def build_offline_image_feature_providers(
    *,
    client: Optional[OfflineImageModelClient] = None,
) -> ImageFeatureProviders:
    """Build providers that follow the root offline image vectorization script."""

    model_client = client or OfflineImageModelClient.from_settings()
    # The physical field is retained for history compatibility. When disabled,
    # the feature pipeline writes the sentinel and does not need VL settings.
    requires_visual_embed = bool(getattr(settings, "IMAGE_VISUAL_EMBED_ENABLED", False))
    missing = model_client.missing_configuration(
        require_visual_embed=requires_visual_embed,
    )
    if missing:
        raise ImageProviderConfigurationError(missing)

    analysis_cache: dict[tuple[str, str], ImageInterpretation | Exception] = {}
    analysis_tasks: dict[tuple[str, str], asyncio.Task[ImageInterpretation]] = {}
    image_preprocess_semaphore = asyncio.Semaphore(
        _positive_setting("IMAGE_PREPROCESS_CONCURRENCY", 8)
    )
    vl_chat_semaphore = asyncio.Semaphore(_positive_setting("IMAGE_VL_CHAT_CONCURRENCY", 8))
    vl_embed_semaphore = asyncio.Semaphore(_positive_setting("IMAGE_VL_EMBED_CONCURRENCY", 12))
    text_embed_semaphore = asyncio.Semaphore(_positive_setting("IMAGE_TEXT_EMBED_CONCURRENCY", 16))
    configured_total = _positive_setting("IMAGE_MODEL_TOTAL_CONCURRENCY", 32)
    visual_limit = _positive_setting("IMAGE_VL_EMBED_CONCURRENCY", 12)
    text_limit = _positive_setting("IMAGE_TEXT_EMBED_CONCURRENCY", 16)
    # Keep independent endpoints independent. The shared budget is only
    # restrictive when both providers actually target the same gateway.
    same_embed_endpoint = _same_endpoint(
        model_client.vl_embed_url,
        model_client.text_embed_url,
    )
    model_gate = _VisualPriorityModelGate(
        configured_total
        if same_embed_endpoint
        else max(configured_total, visual_limit + text_limit)
    )
    text_batcher = _TextEmbeddingBatcher(
        model_client,
        text_embed_semaphore,
        model_gate=model_gate,
    )

    def _record_stage_ms(
        context: ImageFeaturePipelineContext,
        stage_name: str,
        elapsed_ms: float,
    ) -> None:
        timings = context.draft_record.img_meta.setdefault("feature_stage_timings_ms", {})
        if not isinstance(timings, dict):
            timings = {}
            context.draft_record.img_meta["feature_stage_timings_ms"] = timings
        timings[stage_name] = round(float(elapsed_ms), 3)

    async def analysis(context: ImageFeaturePipelineContext) -> ImageInterpretation:
        key = (sha256(context.image_bytes).hexdigest(), context.context_text)
        if key in analysis_cache:
            image_analysis = analysis_cache[key]
        else:
            task = analysis_tasks.get(key)
            if task is None:
                data_url = await _context_data_url(
                    context,
                    model_client.max_pixels,
                    semaphore=image_preprocess_semaphore,
                )
                task = asyncio.create_task(
                    _run_with_semaphore(
                        vl_chat_semaphore,
                        model_client.interpret_image(
                            context.image_bytes,
                            context.context_text,
                            data_url=data_url,
                        ),
                    )
                )
                analysis_tasks[key] = task
            try:
                image_analysis = await task
            except Exception as error:
                analysis_cache[key] = error
                _apply_analysis_error_diagnostics(context, error, cached=False)
                raise
            else:
                analysis_cache[key] = image_analysis
            finally:
                analysis_tasks.pop(key, None)
        if isinstance(image_analysis, Exception):
            _apply_analysis_error_diagnostics(context, image_analysis, cached=True)
            raise image_analysis
        _apply_context_cleaning_diagnostics(context, image_analysis)
        return image_analysis

    async def p_hash_provider(context: ImageFeaturePipelineContext) -> Any:
        started = time.perf_counter()
        try:
            return await _run_in_thread_with_semaphore(
                image_preprocess_semaphore,
                compute_phash,
                context.image_bytes,
            )
        finally:
            _record_stage_ms(context, "p_hash", _elapsed_ms(started))

    async def visual_embed_provider(context: ImageFeaturePipelineContext) -> list[float]:
        started = time.perf_counter()
        try:
            data_url = await _context_data_url(
                context,
                model_client.max_pixels,
                semaphore=image_preprocess_semaphore,
            )
            return await _run_with_semaphore(
                vl_embed_semaphore,
                model_gate.run(
                    model_client.embed_image(
                        context.image_bytes,
                        data_url=data_url,
                    ),
                    priority="visual",
                ),
            )
        finally:
            _record_stage_ms(context, "visual_embed", _elapsed_ms(started))

    async def caption_embed_provider(
        context: ImageFeaturePipelineContext,
    ) -> Optional[list[float]]:
        caption = _caption_text(context)
        if not caption:
            _record_stage_ms(context, "caption_embed", 0.0)
            return None
        started = time.perf_counter()
        try:
            return await text_batcher.embed_text(
                caption,
                context=context,
                feature_name="caption_embed",
            )
        finally:
            _record_stage_ms(context, "caption_embed", _elapsed_ms(started))

    async def context_embed_provider(
        context: ImageFeaturePipelineContext,
    ) -> Optional[list[float]]:
        context_text = _record_context_text(context)
        if not context_text:
            _record_stage_ms(context, "context_embed", 0.0)
            return None
        started = time.perf_counter()
        try:
            return await text_batcher.embed_text(
                context_text,
                context=context,
                feature_name="context_embed",
            )
        finally:
            _record_stage_ms(context, "context_embed", _elapsed_ms(started))

    async def text_embed_provider(context: ImageFeaturePipelineContext) -> list[float]:
        image_analysis = await analysis(context)
        if not image_analysis.deep_text:
            raise ImageProviderError(
                IMAGE_ANALYSIS_TEXT_MISSING,
                "VL interpretation did not return deep_semantics",
                retryable=True,
            )
        started = time.perf_counter()
        try:
            return await text_batcher.embed_text(
                image_analysis.deep_text,
                context=context,
                feature_name="text_embed",
            )
        finally:
            context.draft_record.img_meta["embed_text_ms"] = _elapsed_ms(started)

    async def visual_text_embed_provider(context: ImageFeaturePipelineContext) -> list[float]:
        image_analysis = await analysis(context)
        if not image_analysis.deep_text:
            raise ImageProviderError(
                IMAGE_ANALYSIS_TEXT_MISSING,
                "VL interpretation did not return deep_semantics",
                retryable=True,
            )
        data_url = await _context_data_url(
            context,
            model_client.max_pixels,
            semaphore=image_preprocess_semaphore,
        )
        return await _run_with_semaphore(
            vl_embed_semaphore,
            model_gate.run(
                model_client.embed_image_text(
                    context.image_bytes,
                    image_analysis.deep_text,
                    data_url=data_url,
                ),
                priority="visual",
            ),
        )

    async def bm25_text_provider(context: ImageFeaturePipelineContext) -> str:
        image_analysis = await analysis(context)
        return image_analysis.combined_text

    return ImageFeatureProviders(
        p_hash_provider=p_hash_provider,
        visual_embed_provider=visual_embed_provider if requires_visual_embed else None,
        caption_embed_provider=caption_embed_provider,
        context_embed_provider=context_embed_provider,
        # These image text fields are reserved placeholders in the Milvus schema.
        # The feature pipeline fills them without calling model providers.
        text_embed_provider=None,
        visual_text_embed_provider=None,
        text_sparse_vec_provider=None,
        bm25_text_provider=None,
    )


def _caption_text(context: ImageFeaturePipelineContext) -> str:
    caption = context.draft_record.img_meta.get("caption")
    value = str(caption or "").strip()
    return "" if value.casefold() == "no caption" else value


def _record_context_text(context: ImageFeaturePipelineContext) -> str:
    return "\n\n".join(
        part
        for part in (
            str(context.draft_record.context_before or "").strip(),
            str(context.draft_record.context_after or "").strip(),
        )
        if part
    )


def _apply_context_cleaning_diagnostics(
    context: ImageFeaturePipelineContext,
    image_analysis: ImageInterpretation,
) -> None:
    context.draft_record.img_meta["context_cleaning_fallback"] = (
        image_analysis.context_cleaning_fallback
    )
    if image_analysis.context_cleaning_error:
        context.draft_record.img_meta["context_cleaning_error"] = (
            image_analysis.context_cleaning_error
        )
    if image_analysis.context_cleaning_preview:
        context.draft_record.img_meta["context_cleaning_preview"] = (
            image_analysis.context_cleaning_preview
        )
    if image_analysis.context_cleaning_raw_length:
        context.draft_record.img_meta["context_cleaning_raw_length"] = (
            image_analysis.context_cleaning_raw_length
        )
    context.draft_record.img_meta["draft_chat_ms"] = image_analysis.draft_chat_ms
    context.draft_record.img_meta["vl_context_chars"] = image_analysis.vl_context_chars
    context.draft_record.img_meta["vl_context_truncated"] = (
        image_analysis.vl_context_truncated
    )


def _apply_analysis_error_diagnostics(
    context: ImageFeaturePipelineContext,
    error: Exception,
    *,
    cached: bool,
) -> None:
    context.draft_record.img_meta["analysis_error_cached"] = cached
    context.draft_record.img_meta["analysis_error_code"] = str(
        getattr(error, "error_code", "IMAGE_ANALYSIS_ERROR")
    )
    context.draft_record.img_meta["analysis_error_message"] = str(error)


class _TextEmbeddingBatcher:
    def __init__(
        self,
        client: OfflineImageModelClient,
        semaphore: asyncio.Semaphore,
        *,
        model_gate: Optional[_VisualPriorityModelGate] = None,
    ) -> None:
        self._client = client
        self._semaphore = semaphore
        self._model_gate = model_gate
        self._pending: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._batch_size = _positive_setting("IMAGE_TEXT_EMBED_BATCH_SIZE", 32)
        self._delay_sec = _positive_setting("IMAGE_TEXT_EMBED_BATCH_DELAY_MS", 10) / 1000
        self._active_dispatches = 0
        self._delayed_dispatch_task: Optional[asyncio.Task[None]] = None
        self._max_in_flight_batches = max(
            1,
            _positive_setting("IMAGE_TEXT_EMBED_MAX_IN_FLIGHT_BATCHES", 8),
        )

    async def embed_text(
        self,
        text: str,
        *,
        context: Optional[ImageFeaturePipelineContext] = None,
        feature_name: str = "text_embed",
    ) -> list[float]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        async with self._lock:
            self._pending.append(
                {
                    "text": str(text or ""),
                    "future": future,
                    "context": context,
                    "feature_name": feature_name,
                }
            )
            self._schedule_dispatch_locked()
        return await future

    def _schedule_dispatch_locked(self) -> None:
        if not self._pending:
            return
        if len(self._pending) >= self._batch_size and self._delayed_dispatch_task is not None:
            self._delayed_dispatch_task.cancel()
            self._delayed_dispatch_task = None
        while (
            len(self._pending) >= self._batch_size
            and self._active_dispatches < self._max_in_flight_batches
        ):
            self._active_dispatches += 1
            asyncio.create_task(self._dispatch_batch())
        if (
            self._pending
            and self._active_dispatches < self._max_in_flight_batches
            and self._delayed_dispatch_task is None
        ):
            self._delayed_dispatch_task = asyncio.create_task(self._dispatch_batch_after_delay())

    async def _dispatch_batch_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._delay_sec)
            async with self._lock:
                if self._delayed_dispatch_task is asyncio.current_task():
                    self._delayed_dispatch_task = None
                if not self._pending or self._active_dispatches >= self._max_in_flight_batches:
                    self._schedule_dispatch_locked()
                    return
                self._active_dispatches += 1
            await self._dispatch_batch()
        except asyncio.CancelledError:
            pass
        finally:
            async with self._lock:
                if self._delayed_dispatch_task is asyncio.current_task():
                    self._delayed_dispatch_task = None

    async def _dispatch_batch(self) -> None:
        async with self._lock:
            pending = self._pending[: self._batch_size]
            self._pending = self._pending[self._batch_size :]
        try:
            if not pending:
                return

            texts = [str(item["text"]) for item in pending]
            try:
                async with self._semaphore:
                    request = self._client.embed_texts(texts)
                    if self._model_gate is None:
                        vectors = await request
                    else:
                        vectors = await self._model_gate.run(request, priority="text")
                if len(vectors) != len(pending):
                    raise ImageProviderError(
                        IMAGE_PROVIDER_RESPONSE_INVALID,
                        (
                            "text embedding batch response count mismatch: "
                            f"expected={len(pending)} actual={len(vectors)}"
                        ),
                        retryable=True,
                    )
            except Exception as batch_error:
                if len(pending) <= 1:
                    _finish_pending_with_error(pending, batch_error)
                else:
                    await self._fallback_to_single_requests(pending, batch_error)
                return

            for item, vector in zip(pending, vectors):
                future = item["future"]
                if not future.done():
                    future.set_result(vector)
        finally:
            async with self._lock:
                self._active_dispatches = max(self._active_dispatches - 1, 0)
                self._schedule_dispatch_locked()

    async def _fallback_to_single_requests(
        self,
        pending: list[dict[str, Any]],
        batch_error: Exception,
    ) -> None:
        for item in pending:
            context = item.get("context")
            if isinstance(context, ImageFeaturePipelineContext):
                _record_text_batch_fallback(
                    context,
                    feature_name=str(item.get("feature_name") or "text_embed"),
                    error=batch_error,
                )
            future = item["future"]
            try:
                async with self._semaphore:
                    request = self._client.embed_texts([str(item["text"])])
                    if self._model_gate is None:
                        vectors = await request
                    else:
                        vectors = await self._model_gate.run(request, priority="text")
                if not vectors:
                    raise ImageProviderError(
                        IMAGE_PROVIDER_RESPONSE_INVALID,
                        "text embedding single fallback returned no vectors",
                        retryable=True,
                    )
            except Exception as single_error:
                if not future.done():
                    future.set_exception(single_error)
            else:
                if not future.done():
                    future.set_result(vectors[0])


def _finish_pending_with_error(
    pending: list[dict[str, Any]],
    error: Exception,
) -> None:
    for item in pending:
        future = item["future"]
        if not future.done():
            future.set_exception(error)


def _record_text_batch_fallback(
    context: ImageFeaturePipelineContext,
    *,
    feature_name: str,
    error: Exception,
) -> None:
    img_meta = context.draft_record.img_meta
    img_meta["text_embedding_batch_fallback"] = True
    fallbacks = img_meta.setdefault("text_embedding_batch_fallback_features", [])
    if isinstance(fallbacks, list) and feature_name not in fallbacks:
        fallbacks.append(feature_name)
    img_meta["text_embedding_batch_fallback_error"] = str(error)


async def _run_with_semaphore(
    semaphore: asyncio.Semaphore,
    awaitable: Awaitable[Any],
) -> Any:
    async with semaphore:
        return await awaitable


async def _run_in_thread_with_semaphore(
    semaphore: asyncio.Semaphore,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    async with semaphore:
        return await asyncio.to_thread(func, *args, **kwargs)


async def _context_data_url(
    context: ImageFeaturePipelineContext,
    max_pixels: int,
    *,
    semaphore: asyncio.Semaphore,
) -> str:
    key = _data_url_cache_key(max_pixels)
    if key in context.scratch:
        context.draft_record.img_meta["data_url_cache_hit"] = True
        return str(context.scratch[key])

    task_key = f"{key}:task"
    pending = context.scratch.get(task_key)
    if not isinstance(pending, asyncio.Task):
        context.draft_record.img_meta.setdefault("data_url_cache_hit", False)
        pending = asyncio.create_task(
            _build_context_data_url(
                context,
                max_pixels,
                key=key,
                task_key=task_key,
                semaphore=semaphore,
            )
        )
        context.scratch[task_key] = pending
    else:
        context.draft_record.img_meta["data_url_cache_hit"] = True
    return str(await asyncio.shield(pending))


async def _build_context_data_url(
    context: ImageFeaturePipelineContext,
    max_pixels: int,
    *,
    key: str,
    task_key: str,
    semaphore: asyncio.Semaphore,
) -> str:
    queued_at = time.perf_counter()
    try:
        async with semaphore:
            started = time.perf_counter()
            queue_wait = started - queued_at
            data_url = await asyncio.to_thread(
                image_bytes_to_data_url,
                context.image_bytes,
                max_pixels=max_pixels,
            )
        elapsed = time.perf_counter() - started
        image_stage_latency_seconds.labels(stage="data_url").observe(elapsed)
        context.scratch[key] = data_url
        context.draft_record.img_meta["data_url_chars"] = len(data_url)
        context.draft_record.img_meta["model_input_format"] = _image_model_input_format()
        context.draft_record.img_meta["model_input_max_pixels"] = _model_input_max_pixels(max_pixels)
        timings = context.draft_record.img_meta.setdefault("feature_stage_timings_ms", {})
        if isinstance(timings, dict):
            timings["data_url"] = round(elapsed * 1000.0, 3)
            timings["data_url_queue_wait"] = round(queue_wait * 1000.0, 3)
        return data_url
    finally:
        context.scratch.pop(task_key, None)


def _positive_setting(name: str, default: int) -> int:
    value = getattr(settings, name, default)
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return max(int(default), 1)
    return max(resolved, 1)


def _same_endpoint(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return str(value or "").strip().rstrip("/").casefold()

    normalized_left = normalize(left)
    return bool(normalized_left and normalized_left == normalize(right))


def image_bytes_to_data_url(image_bytes: bytes, *, max_pixels: int) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImageProviderError(
            IMAGE_PROVIDER_DEPENDENCY_MISSING,
            "Pillow is required for image data URL conversion",
            retryable=False,
        ) from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            converted = ImageOps.exif_transpose(image)
            if converted.mode in {"RGBA", "LA"} or (
                converted.mode == "P" and "transparency" in converted.info
            ):
                background = Image.new("RGB", converted.convert("RGBA").size, "white")
                background.paste(converted.convert("RGBA"), mask=converted.convert("RGBA").split()[-1])
                converted = background
            else:
                converted = converted.convert("RGB")
            converted = _resize_for_model_input(converted, max_pixels=max_pixels)
            buffer, mime_type = _encode_model_input_image(converted)
    except Exception as exc:
        if isinstance(exc, ImageProviderError):
            raise
        raise ImageProviderError(
            IMAGE_PROVIDER_RESPONSE_INVALID,
            f"image bytes could not be decoded: {exc}",
            retryable=False,
        ) from exc
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _data_url_cache_key(max_pixels: int) -> str:
    return (
        f"data_url:{_image_model_input_format()}:"
        f"{_model_input_max_pixels(max_pixels)}:"
        f"{_positive_setting('IMAGE_MODEL_INPUT_MAX_SIDE', 1280)}:"
        f"{_positive_setting('IMAGE_MODEL_INPUT_JPEG_QUALITY', 82)}:"
        f"{_positive_setting('IMAGE_MODEL_INPUT_MAX_BYTES', 512_000)}"
    )


def _resize_for_model_input(image, *, max_pixels: int):
    from PIL import Image

    target_max_pixels = _model_input_max_pixels(max_pixels)
    target_max_side = _positive_setting("IMAGE_MODEL_INPUT_MAX_SIDE", 1280)
    pixels = max(int(image.width) * int(image.height), 1)
    scale_by_pixels = math.sqrt(target_max_pixels / pixels) if pixels > target_max_pixels else 1.0
    longest_side = max(int(image.width), int(image.height), 1)
    scale_by_side = target_max_side / longest_side if longest_side > target_max_side else 1.0
    scale = min(scale_by_pixels, scale_by_side, 1.0)
    if scale >= 1.0:
        return image
    new_size = (
        max(int(image.width * scale), 1),
        max(int(image.height * scale), 1),
    )
    return image.resize(new_size, Image.LANCZOS)


def _encode_model_input_image(image) -> tuple[BytesIO, str]:
    image_format = _image_model_input_format()
    if image_format in {"jpg", "jpeg"}:
        return _encode_jpeg_model_input(image), "image/jpeg"
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    _ensure_model_input_size(buffer)
    return buffer, "image/png"


def _encode_jpeg_model_input(image) -> BytesIO:
    quality = _bounded_int_setting("IMAGE_MODEL_INPUT_JPEG_QUALITY", 82, lower=1, upper=95)
    min_quality = _bounded_int_setting(
        "IMAGE_MODEL_INPUT_MIN_JPEG_QUALITY",
        70,
        lower=1,
        upper=quality,
    )
    max_bytes = _positive_setting("IMAGE_MODEL_INPUT_MAX_BYTES", 512_000)
    max_pixels = _positive_setting("IMAGE_MODEL_INPUT_MAX_PIXELS", 1_200_000)
    max_side = _positive_setting("IMAGE_MODEL_INPUT_MAX_SIDE", 1280)
    working = image
    while True:
        for candidate_quality in range(quality, min_quality - 1, -6):
            buffer = BytesIO()
            working.save(
                buffer,
                format="JPEG",
                quality=candidate_quality,
                optimize=True,
            )
            if buffer.tell() <= max_bytes:
                return buffer
        longest_side = max(int(working.width), int(working.height), 1)
        if longest_side <= 512:
            _ensure_model_input_size(buffer)
            return buffer
        next_side = max(int(longest_side * 0.85), 512)
        scale = min(next_side / longest_side, 1.0)
        next_size = (
            max(int(working.width * scale), 1),
            max(int(working.height * scale), 1),
        )
        if next_size == (working.width, working.height):
            _ensure_model_input_size(buffer)
            return buffer
        from PIL import Image

        working = working.resize(next_size, Image.LANCZOS)
        if working.width * working.height > max_pixels:
            working = _resize_for_model_input(working, max_pixels=max_pixels)
        if max(working.width, working.height) > max_side:
            working = _resize_for_model_input(working, max_pixels=max_pixels)


def _ensure_model_input_size(buffer: BytesIO) -> None:
    max_bytes = _positive_setting("IMAGE_MODEL_INPUT_MAX_BYTES", 512_000)
    if buffer.tell() > max_bytes:
        raise ImageProviderError(
            IMAGE_VL_EMBED_INPUT_TOO_LARGE,
            (
                "model input image is too large after compression: "
                f"bytes={buffer.tell()} max_bytes={max_bytes}"
            ),
            retryable=False,
        )


def _image_model_input_format() -> str:
    value = str(getattr(settings, "IMAGE_MODEL_INPUT_FORMAT", "jpeg") or "jpeg").strip().lower()
    return value if value in {"jpeg", "jpg", "png"} else "jpeg"


def _model_input_max_pixels(max_pixels: int) -> int:
    configured = getattr(settings, "IMAGE_MODEL_INPUT_MAX_PIXELS", 0)
    try:
        resolved = int(configured or 0)
    except (TypeError, ValueError):
        resolved = 0
    if resolved <= 0:
        resolved = int(max_pixels or 1)
    return max(min(resolved, int(max_pixels or resolved or 1)), 1)


def _bounded_int_setting(name: str, default: int, *, lower: int, upper: int) -> int:
    value = getattr(settings, name, default)
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return min(max(resolved, lower), upper)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _trim_context_text(value: str, max_chars: int) -> tuple[str, bool]:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _extract_embedding(data: Mapping[str, Any]) -> Any:
    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ImageProviderError(
            IMAGE_PROVIDER_RESPONSE_INVALID,
            f"unexpected embedding API response: {_safe_response_preview(data)}",
            retryable=True,
        ) from exc


def _extract_embeddings(
    data: Mapping[str, Any],
    expected_count: int,
) -> list[Any]:
    try:
        items = data["data"]
    except (KeyError, TypeError) as exc:
        raise ImageProviderError(
            IMAGE_PROVIDER_RESPONSE_INVALID,
            f"unexpected embedding API response: {_safe_response_preview(data)}",
            retryable=True,
        ) from exc
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ImageProviderError(
            IMAGE_PROVIDER_RESPONSE_INVALID,
            f"unexpected embedding API response: {_safe_response_preview(data)}",
            retryable=True,
        )
    embeddings: list[Any] = []
    for item in items:
        if not isinstance(item, Mapping) or "embedding" not in item:
            raise ImageProviderError(
                IMAGE_PROVIDER_RESPONSE_INVALID,
                f"unexpected embedding API response: {_safe_response_preview(data)}",
                retryable=True,
            )
        embeddings.append(item["embedding"])
    if len(embeddings) != expected_count:
        raise ImageProviderError(
            IMAGE_PROVIDER_RESPONSE_INVALID,
            (
                "embedding API response count mismatch: "
                f"expected={expected_count} actual={len(embeddings)}; "
                f"preview={_safe_response_preview(data)}"
            ),
            retryable=True,
        )
    return embeddings


def _normalized_vector(raw: Any) -> list[float]:
    vector = np.asarray(raw, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ImageProviderError(
            IMAGE_PROVIDER_RESPONSE_INVALID,
            "embedding response must be a non-empty 1D vector",
            retryable=True,
        )
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return [float(item) for item in vector.astype(np.float16)]


def _loads_json_object(value: Any, label: str) -> dict[str, Any]:
    raw_text = _chat_content_to_text(value)
    candidates = _json_text_candidates(raw_text)
    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            raise ImageProviderError(
                IMAGE_PROVIDER_RESPONSE_INVALID,
                _invalid_json_message(label, raw_text, "JSON must be an object"),
                retryable=True,
            )
        return parsed

    reason = str(last_error) if last_error is not None else "no JSON object found"
    raise ImageProviderError(
        IMAGE_PROVIDER_RESPONSE_INVALID,
        _invalid_json_message(label, raw_text, reason),
        retryable=True,
    )


def _json_text_candidates(text: str) -> list[str]:
    cleaned = _extract_json_text(text)
    candidates = [cleaned]
    balanced = _extract_first_balanced_json_object(cleaned)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    return [candidate for candidate in candidates if candidate]


def _extract_json_text(text: Any) -> str:
    raw_text = _chat_content_to_text(text).strip()
    match = _JSON_FENCE_RE.match(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text


def _extract_first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1].strip()
        start = text.find("{", start + 1)
    return ""


def _chat_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                if "text" in item:
                    parts.append(_chat_content_to_text(item.get("text")))
                elif "content" in item:
                    parts.append(_chat_content_to_text(item.get("content")))
                else:
                    parts.append(_stringify_for_preview(item))
            else:
                parts.append(_stringify_for_preview(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        if "text" in content:
            return _chat_content_to_text(content.get("text"))
        if "content" in content:
            return _chat_content_to_text(content.get("content"))
    return _stringify_for_preview(content)


def _invalid_json_message(label: str, raw_text: str, reason: str) -> str:
    return (
        f"{label} returned invalid JSON: {reason}; "
        f"raw_len={_text_length(raw_text)}; preview={_safe_text_preview(raw_text)}"
    )


def _safe_response_preview(data: Any) -> str:
    return _safe_text_preview(data)


def _safe_text_preview(value: Any, limit: Optional[int] = None) -> str:
    text = _stringify_for_preview(value)
    redacted = _DATA_IMAGE_RE.sub("data:image/<redacted>;base64,<redacted>", text)
    preview_limit = settings.IMAGE_VL_RESPONSE_PREVIEW_CHARS if limit is None else limit
    preview_limit = max(int(preview_limit or 0), 0)
    if not preview_limit:
        return ""
    suffix = "..." if len(redacted) > preview_limit else ""
    return redacted[:preview_limit] + suffix


def _text_length(value: Any) -> int:
    return len(_stringify_for_preview(value))


def _stringify_for_preview(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _safe_http_error_message(error: httpx.HTTPStatusError) -> str:
    status = error.response.status_code
    preview = _safe_text_preview(error.response.text, limit=200)
    if preview:
        return f"HTTP {status} from image model provider: {preview}"
    return f"HTTP {status} from image model provider"


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _backoff_delay(attempt: int) -> float:
    delay = min(settings.RETRY_BASE_DELAY_SEC * (2 ** attempt), settings.RETRY_MAX_DELAY_SEC)
    return delay * random.uniform(0.75, 1.25)


_RETRYABLE_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)
