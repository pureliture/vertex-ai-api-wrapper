from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

import google.auth
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
TOKEN_REFRESH_SKEW_SECONDS = int(os.getenv("TOKEN_REFRESH_SKEW_SECONDS", "300"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
VERTEX_AUTO_TRUNCATE = os.getenv("VERTEX_AUTO_TRUNCATE", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_MAX_INSTANCES = int(os.getenv("DEFAULT_MAX_INSTANCES", "1"))

# ---------------------------------------------------------------------------
# Model registry (config-driven)
# ---------------------------------------------------------------------------

_SUPPORTED_APIS = {"predict", "embedContent", "generateContent"}

# kind 기본값 결정: api별 default kind
_API_DEFAULT_KIND: dict[str, str] = {
    "predict": "embedding",
    "embedContent": "embedding",
    "generateContent": "chat",
}

_BUILTIN_REGISTRY: dict[str, dict[str, Any]] = {
    "text-embedding-005": {"api": "predict", "max_instances": 5},
    "text-multilingual-embedding-002": {"api": "predict", "max_instances": 5},
    "gemini-embedding-001": {"api": "predict", "max_instances": 1},
    "gemini-embedding-2": {"api": "embedContent", "location": "global", "max_instances": 1},
    # Chat models
    # thinking 모델은 thinking_budget=0으로 사고 토큰을 끈다 -> 작은 max_tokens에도 본문이 비지 않음
    # (RAGFlow verify가 작은 max_tokens로 호출해 빈 응답->실패하던 문제 방지).
    "gemini-2.5-flash": {"api": "generateContent", "kind": "chat", "location": "us-central1", "thinking_budget": 0},
    "gemini-2.5-pro": {"api": "generateContent", "kind": "chat", "location": "us-central1"},
    "gemini-3.5-flash": {"api": "generateContent", "kind": "chat", "location": "global", "thinking_budget": 0},
}

def _build_registry() -> dict[str, dict[str, Any]]:
    """환경변수를 반영한 최종 모델 레지스트리를 빌드한다."""
    registry: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _BUILTIN_REGISTRY.items()}

    # MODEL_REGISTRY_JSON: 기본값 위에 merge
    registry_json_str = os.getenv("MODEL_REGISTRY_JSON", "").strip()
    if registry_json_str:
        # invalid JSON이면 loudly fail
        override = json.loads(registry_json_str)
        if not isinstance(override, dict):
            raise ValueError("MODEL_REGISTRY_JSON must be a JSON object")
        for model_id, cfg in override.items():
            if model_id in registry:
                registry[model_id] = {**registry[model_id], **cfg}
            else:
                registry[model_id] = dict(cfg)

    # EXTRA_MODELS: comma-separated backward compat
    extra_models_str = os.getenv("EXTRA_MODELS", "").strip()
    if extra_models_str:
        for name in (m.strip() for m in extra_models_str.split(",") if m.strip()):
            if name not in registry:
                registry[name] = {"api": "predict", "max_instances": DEFAULT_MAX_INSTANCES}

    # 모든 엔트리의 api가 지원되는 값인지 검증 (누락/오타/대소문자 등은 loudly fail)
    for model_id, cfg in registry.items():
        api = cfg.get("api")
        if api not in _SUPPORTED_APIS:
            raise ValueError(
                f"Model '{model_id}' has invalid api={api!r}; "
                f"must be one of {sorted(_SUPPORTED_APIS)}"
            )

    return registry


MODEL_REGISTRY: dict[str, dict[str, Any]] = _build_registry()

# Backward-compat: 기존 코드가 KNOWN_MAX_INSTANCES를 직접 참조하는 경우를 위해 유지
KNOWN_MAX_INSTANCES: dict[str, int] = {
    k: v["max_instances"] for k, v in MODEL_REGISTRY.items() if v.get("api") == "predict"
}


def model_config(model: str) -> dict[str, Any] | None:
    """모델의 resolved config dict를 반환한다.

    반환 dict는 최소 api, kind, location, max_instances 키를 포함한다.
    - location: 엔트리에 명시된 경우 그 값, embedContent면 "global", predict면 VERTEX_LOCATION.
    - kind: 엔트리에 명시된 경우 그 값, 없으면 api에 따라 결정 (_API_DEFAULT_KIND).
    """
    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        return None
    cfg = dict(entry)
    if "location" not in cfg:
        if cfg.get("api") == "embedContent":
            cfg["location"] = "global"
        else:
            cfg["location"] = VERTEX_LOCATION
    if "max_instances" not in cfg:
        cfg["max_instances"] = DEFAULT_MAX_INSTANCES
    if "kind" not in cfg:
        cfg["kind"] = _API_DEFAULT_KIND.get(cfg.get("api", ""), "embedding")
    return cfg


def allowed_models() -> set[str]:
    """레지스트리에 등록된 모든 모델 id를 반환한다."""
    return set(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class VertexAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None, raw: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.raw = raw


def _parse_vertex_error(resp: httpx.Response) -> VertexAPIError:
    """Vertex AI HTTP 에러 응답(4xx/5xx)을 VertexAPIError로 변환한다.

    predict / embedContent 등 어댑터가 공유하는 에러 파싱 로직.
    """
    try:
        payload = resp.json()
    except Exception:
        payload = {"error": {"message": resp.text}}
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = err.get("message") or resp.text or "Vertex AI request failed"
    code = err.get("status") or err.get("code") or str(resp.status_code)
    return VertexAPIError(resp.status_code, message=message, code=str(code), raw=payload)


def _vertex_host(location: str) -> str:
    """Vertex AI 호스트. location이 'global'이면 region prefix가 없다."""
    if location == "global":
        return "aiplatform.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i : i + size]

class GoogleAccessTokenProvider:
    """서비스 계정(ADC) 기반 access token을 만료 전 선갱신하며 캐시한다."""
    def __init__(self) -> None:
        creds, detected_project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        self._creds = creds
        self.project_id = VERTEX_PROJECT or detected_project
        if not self.project_id:
            raise RuntimeError(
                "Google Cloud project를 결정할 수 없습니다. VERTEX_PROJECT 환경변수를 설정하거나 ADC project를 구성하세요."
            )
        self._lock = threading.Lock()
        self._request = GoogleAuthRequest()

    def _valid_with_skew(self) -> bool:
        token = getattr(self._creds, "token", None)
        expiry = getattr(self._creds, "expiry", None)
        if not token:
            return False
        if expiry is None:
            return bool(self._creds.valid)
        remaining = (expiry - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        return bool(self._creds.valid) and remaining > TOKEN_REFRESH_SKEW_SECONDS

    def _get_token_sync(self) -> str:
        with self._lock:
            if not self._valid_with_skew():
                self._creds.refresh(self._request)
            token = getattr(self._creds, "token", None)
            if not token:
                raise RuntimeError("Google access token 발급에 실패했습니다.")
            return token

    async def get_token(self) -> str:
        return await asyncio.to_thread(self._get_token_sync)


# ---------------------------------------------------------------------------
# Vertex AI HTTP client
# ---------------------------------------------------------------------------

class VertexEmbeddingClient:
    def __init__(self, token_provider: GoogleAccessTokenProvider | None = None) -> None:
        self.token_provider = token_provider or GoogleAccessTokenProvider()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def close(self) -> None:
        await self.http.aclose()

    def _predict_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:predict"
        )

    def _embed_content_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:embedContent"
        )

    async def _predict(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        auto_truncate: bool,
        location: str,
    ) -> list[dict[str, Any]]:
        """predict API를 호출하고 예측 목록을 반환한다."""
        token = await self.token_provider.get_token()
        url = self._predict_url(model, location)

        instances: list[dict[str, Any]] = []
        for text in texts:
            item: dict[str, Any] = {"content": text}
            if task_type and task_type != "UNSPECIFIED":
                item["task_type"] = task_type
            if title:
                item["title"] = title
            instances.append(item)

        parameters: dict[str, Any] = {"autoTruncate": auto_truncate}
        if dimensions is not None:
            parameters["outputDimensionality"] = dimensions

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = await self.http.post(
                url, headers=headers, json={"instances": instances, "parameters": parameters}
            )
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        predictions = data.get("predictions")
        if not isinstance(predictions, list):
            raise VertexAPIError(502, "Malformed Vertex AI response: missing predictions[]", code="bad_gateway")
        if len(predictions) != len(texts):
            raise VertexAPIError(
                502,
                f"Malformed predict response: prediction count mismatch "
                f"(expected {len(texts)}, got {len(predictions)})",
                code="bad_gateway",
            )
        return predictions

    async def _embed_content_single(
        self,
        *,
        model: str,
        text: str,
        dimensions: int | None,
        task_type: str,
        location: str,
    ) -> dict[str, Any]:
        """embedContent API를 1개 텍스트에 대해 호출하고 raw 응답을 반환한다."""
        token = await self.token_provider.get_token()
        url = self._embed_content_url(model, location)

        body: dict[str, Any] = {
            "content": {"parts": [{"text": text}]},
        }
        if dimensions is not None:
            body["outputDimensionality"] = dimensions
        if task_type and task_type != "UNSPECIFIED":
            body["taskType"] = task_type

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        return data

    async def embed(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
    ) -> list[dict[str, Any]]:
        """모든 텍스트에 대한 embedding을 반환한다.

        Returns:
            list[dict] in input order, each dict = {"values": list[float], "token_count": int}
        """
        cfg = model_config(model) or {
            "api": "predict",
            "location": VERTEX_LOCATION,
            "max_instances": DEFAULT_MAX_INSTANCES,
        }
        api = cfg.get("api", "predict")
        location = cfg.get("location", VERTEX_LOCATION)
        batch_size = cfg.get("max_instances", DEFAULT_MAX_INSTANCES)

        if api == "embedContent":
            return await self._embed_all_embed_content(
                model=model,
                texts=texts,
                dimensions=dimensions,
                task_type=task_type,
                location=location,
            )
        else:
            return await self._embed_all_predict(
                model=model,
                texts=texts,
                dimensions=dimensions,
                task_type=task_type,
                title=title,
                batch_size=batch_size,
                location=location,
            )

    async def _embed_all_predict(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        batch_size: int,
        location: str,
    ) -> list[dict[str, Any]]:
        """predict API를 사용하여 배치 처리 후 flat list로 반환."""
        async def one_chunk(chunk: list[str]) -> list[dict[str, Any]]:
            async with self.semaphore:
                return await self._predict(
                    model=model,
                    texts=chunk,
                    dimensions=dimensions,
                    task_type=task_type,
                    title=title,
                    auto_truncate=VERTEX_AUTO_TRUNCATE,
                    location=location,
                )

        chunk_results = await asyncio.gather(*(one_chunk(chunk) for chunk in chunked(texts, batch_size)))

        # flat list로 변환
        results: list[dict[str, Any]] = []
        for predictions in chunk_results:
            for pred in predictions:
                emb = pred.get("embeddings", {}) if isinstance(pred, dict) else {}
                values = emb.get("values", []) if isinstance(emb, dict) else []
                stats = emb.get("statistics", {}) if isinstance(emb, dict) else {}
                try:
                    token_count = int(stats.get("token_count", 0))
                except (TypeError, ValueError):
                    token_count = 0
                results.append({"values": values, "token_count": token_count})
        return results

    # ------------------------------------------------------------------
    # Private helpers for embed_content path
    # ------------------------------------------------------------------

    async def _embed_all_embed_content(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        location: str,
    ) -> list[dict[str, Any]]:
        """embedContent API를 사용하여 1개씩 호출 후 flat list로 반환. 입력 순서 보장."""
        async def one_text(text: str) -> dict[str, Any]:
            async with self.semaphore:
                return await self._embed_content_single(
                    model=model,
                    text=text,
                    dimensions=dimensions,
                    task_type=task_type,
                    location=location,
                )

        raw_results = await asyncio.gather(*(one_text(text) for text in texts))

        results: list[dict[str, Any]] = []
        for data in raw_results:
            embedding = data.get("embedding", {})
            values = embedding.get("values", []) if isinstance(embedding, dict) else []
            if not values:
                raise VertexAPIError(
                    502,
                    "Malformed embedContent response: embedding.values missing",
                    code="bad_gateway",
                )
            usage = data.get("usageMetadata", {}) or {}
            # usageMetadata 키는 다양할 수 있어 방어적으로 파싱
            token_count = 0
            if isinstance(usage, dict):
                for key in ("tokenCount", "token_count", "inputTokens", "input_tokens"):
                    val = usage.get(key)
                    if val is not None:
                        try:
                            token_count = int(val)
                        except (TypeError, ValueError):
                            token_count = 0
                        break
            results.append({"values": values, "token_count": token_count})
        return results


# ---------------------------------------------------------------------------
# Vertex AI Chat client (generateContent)
# ---------------------------------------------------------------------------

def _map_finish_reason(vertex_reason: str) -> str:
    """Vertex finishReason -> OpenAI finish_reason 매핑."""
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }.get(vertex_reason, "stop")


def _extract_message_text(content: Any) -> str:
    """OpenAI 메시지 content(str 또는 parts list)에서 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content is not None else ""


def _parse_stream_chunk(chunk: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None]:
    """Vertex streamGenerateContent SSE chunk를 (delta_text, finish_reason, usage)로 파싱한다.

    - delta_text: candidates[0].content.parts[*].text 연결 (없으면 "").
    - finish_reason: candidates[0].finishReason가 있으면 OpenAI 매핑값, 없으면 None.
    - usage: usageMetadata가 있으면 usage dict, 없으면 None.
    """
    delta_text = ""
    finish_reason: str | None = None

    candidates = chunk.get("candidates", []) if isinstance(chunk, dict) else []
    if candidates:
        candidate = candidates[0] or {}
        content_obj = candidate.get("content", {}) or {}
        parts_list = content_obj.get("parts", []) or []
        delta_text = "".join(
            p.get("text", "") for p in parts_list if isinstance(p, dict)
        )
        raw_finish = candidate.get("finishReason")
        if raw_finish:
            finish_reason = _map_finish_reason(raw_finish)

    usage: dict[str, Any] | None = None
    usage_meta = chunk.get("usageMetadata") if isinstance(chunk, dict) else None
    if isinstance(usage_meta, dict) and usage_meta:
        usage = {
            "prompt_tokens": int(usage_meta.get("promptTokenCount", 0) or 0),
            "completion_tokens": int(usage_meta.get("candidatesTokenCount", 0) or 0),
            "total_tokens": int(usage_meta.get("totalTokenCount", 0) or 0),
        }

    return delta_text, finish_reason, usage


class VertexChatClient:
    """Vertex AI generateContent API를 사용하는 채팅 클라이언트."""

    def __init__(self, token_provider: GoogleAccessTokenProvider | None = None) -> None:
        self.token_provider = token_provider or GoogleAccessTokenProvider()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def close(self) -> None:
        await self.http.aclose()

    def _generate_content_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:generateContent"
        )

    def _stream_generate_content_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:streamGenerateContent?alt=sse"
        )

    @staticmethod
    def _build_request_body(
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        """OpenAI messages/params -> Vertex generateContent request body.

        비스트림(generate)과 스트림(stream_chat)이 공유하는 매핑 로직.
        """
        # --- 메시지 매핑 ---
        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            text = _extract_message_text(raw_content)

            if role == "system":
                system_parts.append({"text": text})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        body: dict[str, Any] = {"contents": contents}

        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        # --- generationConfig 매핑 ---
        gen_cfg: dict[str, Any] = {}
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_cfg["temperature"] = temperature
        if top_p is not None:
            gen_cfg["topP"] = top_p
        if stop is not None:
            gen_cfg["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        if thinking_budget is not None:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        if gen_cfg:
            body["generationConfig"] = gen_cfg

        return body

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """generateContent를 호출하고 정규화된 결과를 반환한다.

        Returns:
            {"text": str, "finish_reason": str, "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
        """
        cfg = model_config(model) or {"location": VERTEX_LOCATION}
        location = cfg.get("location", VERTEX_LOCATION)

        token = await self.token_provider.get_token()
        url = self._generate_content_url(model, location)

        body = self._build_request_body(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            thinking_budget=cfg.get("thinking_budget"),
        )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            async with self.semaphore:
                resp = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        # --- 응답 파싱 ---
        candidates = data.get("candidates", [])
        if not candidates:
            raise VertexAPIError(502, "Malformed Vertex AI response: no candidates", code="bad_gateway")

        candidate = candidates[0]
        content_obj = candidate.get("content", {}) or {}
        parts_list = content_obj.get("parts", []) or []
        text_out = "".join(p.get("text", "") for p in parts_list if isinstance(p, dict))

        finish_reason = _map_finish_reason(candidate.get("finishReason", "STOP"))

        usage_meta = data.get("usageMetadata", {}) or {}
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
        completion_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0)
        total_tokens = int(usage_meta.get("totalTokenCount", 0) or 0)

        return {
            "text": text_out,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ):
        """streamGenerateContent(SSE)를 호출하고 델타를 순차 yield하는 async generator.

        각 yield는 dict:
            {"delta_text": str, "finish_reason": str | None, "usage": dict | None}
        - delta_text: 이 chunk의 델타 텍스트 (없으면 "").
        - finish_reason: 마지막(혹은 finishReason이 있는) chunk에서 OpenAI 매핑값, 아니면 None.
        - usage: usageMetadata가 있는 chunk에서 usage dict, 아니면 None.

        스트림 시작 전 Vertex 4xx/5xx면 VertexAPIError를 raise한다.
        스트림 도중 끊김/파싱 실패는 방어적으로 해당 줄을 건너뛴다.
        """
        cfg = model_config(model) or {"location": VERTEX_LOCATION}
        location = cfg.get("location", VERTEX_LOCATION)

        token = await self.token_provider.get_token()
        url = self._stream_generate_content_url(model, location)

        body = self._build_request_body(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            thinking_budget=cfg.get("thinking_budget"),
        )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # 세마포어는 연결+헤더 수신(및 4xx 에러 처리)까지만 점유하고,
        # aiter_lines 스트림 루프에 진입하기 전에 해제한다. 그렇지 않으면 긴
        # 스트림들이 MAX_CONCURRENCY 슬롯을 점유한 채 비스트림/후속 요청을 굶긴다.
        stream_ctx = self.http.stream("POST", url, headers=headers, json=body)
        await self.semaphore.acquire()
        entered = False
        try:
            try:
                resp = await stream_ctx.__aenter__()
                entered = True
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

            if resp.status_code >= 400:
                # 에러 본문을 읽어 VertexAPIError로 변환 (스트림 시작 전 에러).
                try:
                    await resp.aread()
                except Exception:
                    pass
                # raise 전에 열린 stream_ctx를 정리한다.
                await stream_ctx.__aexit__(None, None, None)
                entered = False
                raise _parse_vertex_error(resp)
        finally:
            self.semaphore.release()
            # __aenter__ 자체가 실패해 진입하지 못한 경우엔 닫을 컨텍스트가 없다.
            # 4xx로 위에서 이미 닫은 경우엔 entered=False라 중복 호출하지 않는다.

        # 여기부터 세마포어는 해제된 상태. resp는 열린 채로 스트리밍한다.
        try:
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    stripped = line.strip()
                    if not stripped.startswith("data:"):
                        continue
                    payload = stripped[len("data:"):].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except (ValueError, TypeError):
                        # 부분 줄/파싱 실패는 방어적으로 건너뜀
                        continue

                    delta_text, finish_reason, usage = _parse_stream_chunk(chunk)
                    yield {
                        "delta_text": delta_text,
                        "finish_reason": finish_reason,
                        "usage": usage,
                    }
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc
        finally:
            # 정상 종료/에러/클라이언트 끊김(GeneratorExit) 모든 경로에서 닫는다.
            if entered:
                await stream_ctx.__aexit__(None, None, None)
