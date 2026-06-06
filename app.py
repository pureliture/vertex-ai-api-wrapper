"""OpenAI-compatible embedding proxy for Google Cloud Vertex AI (Gemini Enterprise Agent Platform).

RAGFlow 0.25.6의 "OpenAI-API-Compatible" 임베딩 provider가 호출하는
POST /v1/embeddings 를 받아서 Vertex AI :predict 엔드포인트로 통역한다.

- 인증: 서비스 계정(ADC)에서 short-lived OAuth2 access token을 자동 발급/캐시/갱신.
- 배치 분할: RAGFlow는 요청당 16개 텍스트를 보내지만, Vertex 모델별 요청당 instance
  한도(gemini-embedding-001=1, text-embedding-005/multilingual-002=5)에 맞춰 쪼개 병렬 호출.
"""

from __future__ import annotations

import asyncio
import base64
import os
import struct
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

import google.auth
import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from pydantic import BaseModel, ConfigDict


CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

SUPPORTED_TASK_TYPES = {
    "UNSPECIFIED",
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
}

# Vertex :predict 요청당 최대 instance 개수 (Google 공식문서 기준, 2026-06 확인).
# gemini-embedding-001: "Each request can only include a single input text." -> 1
# text-embedding-005 / text-multilingual-embedding-002: "five texts ..." -> 5
# 미등록 모델은 안전하게 1로 폴백.
KNOWN_MAX_INSTANCES = {
    "gemini-embedding-001": 1,
    "text-embedding-005": 5,
    "text-multilingual-embedding-002": 5,
}
DEFAULT_MAX_INSTANCES = int(os.getenv("DEFAULT_MAX_INSTANCES", "1"))

# 허용 모델 화이트리스트 (SSRF/path 주입 방지). KNOWN + env로 추가한 모델만 Vertex URL 경로에 들어간다.
EXTRA_MODELS = {m.strip() for m in os.getenv("EXTRA_MODELS", "").split(",") if m.strip()}
ALLOWED_MODELS = set(KNOWN_MAX_INSTANCES) | EXTRA_MODELS

VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_TASK_TYPE_DEFAULT = os.getenv("VERTEX_TASK_TYPE_DEFAULT", "RETRIEVAL_DOCUMENT")
VERTEX_AUTO_TRUNCATE = os.getenv("VERTEX_AUTO_TRUNCATE", "true").lower() in {"1", "true", "yes", "on"}
WRAPPER_API_KEY = os.getenv("WRAPPER_API_KEY")
TOKEN_REFRESH_SKEW_SECONDS = int(os.getenv("TOKEN_REFRESH_SKEW_SECONDS", "300"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))


class OpenAIEmbeddingsRequest(BaseModel):
    # RAGFlow가 extra_body={"drop_params": True} 등 비표준 필드를 실어 보낼 수 있어 허용 후 무시.
    model_config = ConfigDict(extra="allow")

    input: str | list[str]
    model: str
    # openai SDK는 encoding_format을 명시 전송(기본 base64)하므로 Literal로 받으면 충분.
    # 미지정 시에만 기본값 float이 적용된다.
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class VertexAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None, raw: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.raw = raw


def openai_error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": param, "code": code}},
    )


def map_vertex_status_to_openai_type(status_code: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status_code, "api_error" if status_code >= 500 else "invalid_request_error")


def coerce_string_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError("This wrapper supports only string or array[string] inputs.")


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def encode_embedding(values: list[float], fmt: str) -> list[float] | str:
    """OpenAI encoding_format에 맞춰 임베딩을 직렬화한다.

    - "float": list[float] 그대로.
    - "base64": float32 little-endian 패킹 후 base64 문자열 (OpenAI SDK 디코드 호환).
    """
    if fmt == "base64":
        return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")
    return values


class GoogleAccessTokenProvider:
    """서비스 계정(ADC) 기반 access token을 만료 전 선갱신하며 캐시한다.

    refresh()는 동기 blocking I/O라 asyncio.to_thread로 별도 스레드에서 실행하고,
    threading.Lock으로 동시 refresh stampede를 막는다.
    """

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
    def __init__(self, token_provider: GoogleAccessTokenProvider) -> None:
        self.token_provider = token_provider
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    async def predict(
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.token_provider = GoogleAccessTokenProvider()
    app.state.vertex_client = VertexEmbeddingClient(app.state.token_provider)
    # 전역 동시성 상한: 모든 요청이 공유 -> 다중 요청 동시 fan-out 시에도 Vertex로 가는
    # 동시 호출 수를 MAX_CONCURRENCY로 제한해 429 폭주를 막는다.
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    try:
        yield
    finally:
        await app.state.vertex_client.close()


app = FastAPI(title="Vertex AI OpenAI-Compatible Embeddings Wrapper", version="0.1.0", lifespan=lifespan)
# trailing slash 307 리다이렉트 시 일부 HTTP 클라이언트가 POST 본문을 잃는다 -> 비활성화.
app.router.redirect_slashes = False
# 브라우저 클라이언트(Open WebUI 등)의 CORS preflight(OPTIONS) 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI 기본 422 {"detail":[...]} 는 OpenAI 클라이언트가 파싱하다 깨질 수 있어
    # OpenAI 에러 포맷으로 변환한다.
    detail = exc.errors()
    msg = detail[0].get("msg", "Invalid request") if detail else "Invalid request"
    loc = detail[0].get("loc") if detail else None
    param = ".".join(str(p) for p in loc[1:]) if loc and len(loc) > 1 else None
    return openai_error_response(
        message=str(msg),
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_request",
        param=param,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _model_object(model_id: str) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "google"}


@app.get("/v1/models", response_model=None)
async def list_models(authorization: str | None = Header(default=None)) -> JSONResponse | dict[str, Any]:
    if WRAPPER_API_KEY and authorization != f"Bearer {WRAPPER_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    return {"object": "list", "data": [_model_object(m) for m in sorted(ALLOWED_MODELS)]}


@app.get("/v1/models/{model_id}", response_model=None)
async def retrieve_model(
    model_id: str, authorization: str | None = Header(default=None)
) -> JSONResponse | dict[str, Any]:
    if WRAPPER_API_KEY and authorization != f"Bearer {WRAPPER_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    if model_id not in ALLOWED_MODELS:
        return openai_error_response(
            message=f"The model '{model_id}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )
    return _model_object(model_id)


@app.post("/v1/embeddings", response_model=None)
async def create_embeddings(
    payload: OpenAIEmbeddingsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_vertex_task_type: str | None = Header(default=None, alias="X-Vertex-Task-Type"),
    x_vertex_title: str | None = Header(default=None, alias="X-Vertex-Title"),
) -> JSONResponse | dict[str, Any]:
    if WRAPPER_API_KEY and authorization != f"Bearer {WRAPPER_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    # model은 Vertex URL 경로에 직접 들어가므로 화이트리스트로만 허용(path 주입/SSRF 방지).
    if payload.model not in ALLOWED_MODELS:
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist. Allowed: {sorted(ALLOWED_MODELS)}",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    if payload.dimensions is not None and payload.dimensions < 1:
        return openai_error_response(
            message="dimensions must be >= 1.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_dimensions",
            param="dimensions",
        )

    try:
        texts = coerce_string_inputs(payload.input)
    except ValueError as exc:
        return openai_error_response(
            message=str(exc),
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_input_shape",
            param="input",
        )
    if not texts:
        return openai_error_response(
            message="input must not be empty.",
            status_code=400,
            error_type="invalid_request_error",
            code="empty_input",
            param="input",
        )

    task_type = x_vertex_task_type or VERTEX_TASK_TYPE_DEFAULT
    if task_type not in SUPPORTED_TASK_TYPES:
        return openai_error_response(
            message=f"Unsupported task type: {task_type}",
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_task_type",
            param="X-Vertex-Task-Type",
        )

    batch_size = KNOWN_MAX_INSTANCES.get(payload.model, DEFAULT_MAX_INSTANCES)
    semaphore: asyncio.Semaphore = request.app.state.semaphore
    vertex_client: VertexEmbeddingClient = request.app.state.vertex_client

    async def one_chunk(chunk: list[str]) -> list[dict[str, Any]]:
        async with semaphore:
            return await vertex_client.predict(
                model=payload.model,
                texts=chunk,
                dimensions=payload.dimensions,
                task_type=task_type,
                title=x_vertex_title,
                auto_truncate=VERTEX_AUTO_TRUNCATE,
            )

    try:
        # 입력 순서를 보존하기 위해 chunk 결과를 순서대로 모은다.
        chunk_results = await asyncio.gather(*(one_chunk(chunk) for chunk in chunked(texts, batch_size)))
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    data: list[dict[str, Any]] = []
    total_tokens = 0
    index = 0
    for predictions in chunk_results:
        for pred in predictions:
            emb = pred.get("embeddings", {}) if isinstance(pred, dict) else {}
            values = emb.get("values") if isinstance(emb, dict) else None
            if not isinstance(values, list):
                return openai_error_response(
                    message="Malformed Vertex AI response: embeddings.values missing.",
                    status_code=502,
                    error_type="api_error",
                    code="bad_gateway",
                )
            stats = emb.get("statistics", {}) if isinstance(emb, dict) else {}
            try:
                total_tokens += int(stats.get("token_count", 0))
            except (TypeError, ValueError):
                pass
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": encode_embedding(values, payload.encoding_format),
                }
            )
            index += 1

    return {
        "object": "list",
        "data": data,
        "model": payload.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }
