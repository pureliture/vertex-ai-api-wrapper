from __future__ import annotations

import asyncio
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

KNOWN_MAX_INSTANCES = {
    "gemini-embedding-001": 1,
    "text-embedding-005": 5,
    "text-multilingual-embedding-002": 5,
}

class VertexAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None, raw: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.raw = raw

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

class VertexEmbeddingClient:
    def __init__(self, token_provider: GoogleAccessTokenProvider | None = None) -> None:
        self.token_provider = token_provider or GoogleAccessTokenProvider()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def close(self) -> None:
        await self.http.aclose()

    async def _predict(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        auto_truncate: bool,
    ) -> list[dict[str, Any]]:
        token = await self.token_provider.get_token()
        url = (
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/"
            f"v1/projects/{self.token_provider.project_id}/locations/{VERTEX_LOCATION}/"
            f"publishers/google/models/{model}:predict"
        )

        instances: list[dict[str, Any]] = []
        for text in texts:
            item: dict[str, Any] = {"content": text}
            if task_type:
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
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": {"message": resp.text}}
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = err.get("message") or resp.text or "Vertex AI request failed"
            code = err.get("status") or err.get("code") or str(resp.status_code)
            raise VertexAPIError(resp.status_code, message=message, code=str(code), raw=payload)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        predictions = data.get("predictions")
        if not isinstance(predictions, list):
            raise VertexAPIError(502, "Malformed Vertex AI response: missing predictions[]", code="bad_gateway")
        return predictions

    async def embed(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
    ) -> list[list[dict[str, Any]]]:
        batch_size = KNOWN_MAX_INSTANCES.get(model, DEFAULT_MAX_INSTANCES)
        
        async def one_chunk(chunk: list[str]) -> list[dict[str, Any]]:
            async with self.semaphore:
                return await self._predict(
                    model=model,
                    texts=chunk,
                    dimensions=dimensions,
                    task_type=task_type,
                    title=title,
                    auto_truncate=VERTEX_AUTO_TRUNCATE,
                )
        
        chunk_results = await asyncio.gather(*(one_chunk(chunk) for chunk in chunked(texts, batch_size)))
        return list(chunk_results)
