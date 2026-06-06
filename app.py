"""OpenAI-compatible embedding proxy for Google Cloud Vertex AI (Gemini Enterprise Agent Platform).

RAGFlow 0.25.6의 "OpenAI-API-Compatible" 임베딩 provider가 호출하는
POST /v1/embeddings 를 받아서 Vertex AI :predict 엔드포인트로 통역한다.

- 인증: 서비스 계정(ADC)에서 short-lived OAuth2 access token을 자동 발급/캐시/갱신.
- 배치 분할: RAGFlow는 요청당 16개 텍스트를 보내지만, Vertex 모델별 요청당 instance
  한도에 맞춰 쪼개 병렬 호출.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from vertex import (
    SUPPORTED_RESPONSE_FORMAT_TYPES,
    VertexAPIError,
    VertexChatClient,
    VertexEmbeddingClient,
    allowed_models,
    model_config,
)

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

ALLOWED_MODELS = allowed_models()

VERTEX_TASK_TYPE_DEFAULT = os.getenv("VERTEX_TASK_TYPE_DEFAULT", "RETRIEVAL_DOCUMENT")
WRAPPER_API_KEY = os.getenv("WRAPPER_API_KEY")


class OpenAIEmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any]


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None


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


def encode_embedding(values: list[float], fmt: str) -> list[float] | str:
    if fmt == "base64":
        import base64
        import struct
        return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")
    return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vertex_client = VertexEmbeddingClient()
    app.state.vertex_chat_client = VertexChatClient()
    try:
        yield
    finally:
        await app.state.vertex_client.close()
        await app.state.vertex_chat_client.close()


app = FastAPI(title="Vertex AI OpenAI-Compatible Embeddings Wrapper", version="0.1.0", lifespan=lifespan)
app.router.redirect_slashes = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
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

    if payload.model not in ALLOWED_MODELS:
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist. Allowed: {sorted(ALLOWED_MODELS)}",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    _embed_cfg = model_config(payload.model)
    if _embed_cfg and _embed_cfg.get("kind") != "embedding":
        return openai_error_response(
            message=f"The model '{payload.model}' is not an embedding model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
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

    vertex_client: VertexEmbeddingClient = request.app.state.vertex_client

    try:
        chunk_results = await vertex_client.embed(
            model=payload.model,
            texts=texts,
            dimensions=payload.dimensions,
            task_type=task_type,
            title=x_vertex_title,
        )
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    data: list[dict[str, Any]] = []
    total_tokens = 0
    for index, item in enumerate(chunk_results):
        values = item.get("values")
        if not isinstance(values, list):
            return openai_error_response(
                message="Malformed Vertex AI response: embeddings.values missing.",
                status_code=502,
                error_type="api_error",
                code="bad_gateway",
            )
        try:
            total_tokens += int(item.get("token_count", 0))
        except (TypeError, ValueError):
            pass
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": encode_embedding(values, payload.encoding_format),
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": payload.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


def _new_chat_completion_id() -> str:
    """매 요청 고유한 OpenAI 호환 chat completion id를 생성한다."""
    return f"chatcmpl-{uuid.uuid4().hex[:16]}"


def _chat_completions_stream(
    chat_client: VertexChatClient,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
) -> StreamingResponse:
    """stream=true 요청을 OpenAI 호환 SSE로 변환하는 StreamingResponse를 만든다."""
    completion_id = _new_chat_completion_id()

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        obj = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": payload.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_generator():
        first = True
        final_finish_reason: str | None = None
        try:
            async for event in chat_client.stream_chat(
                model=payload.model,
                messages=messages,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                stop=payload.stop,
                response_format=payload.response_format,
            ):
                delta_text = event.get("delta_text", "") or ""
                fr = event.get("finish_reason")
                if fr is not None:
                    final_finish_reason = fr

                delta: dict[str, Any] = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if delta_text:
                    delta["content"] = delta_text

                # 내용 또는 role이 있는 청크만 델타로 내보낸다.
                if delta:
                    yield _chunk(delta, None)
        except VertexAPIError as exc:
            # 스트림 시작 전/도중 에러: OpenAI 에러 형태를 SSE data로 흘려보낸 뒤 종료.
            err_obj = {
                "error": {
                    "message": exc.message,
                    "type": map_vertex_status_to_openai_type(exc.status_code),
                    "param": None,
                    "code": exc.code,
                }
            }
            yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 첫 청크가 한 번도 안 나갔다면(빈 스트림) role 청크라도 보낸다.
        if first:
            yield _chunk({"role": "assistant"}, None)

        # 종료 청크: finish_reason 담기 (없으면 stop으로 폴백).
        yield _chunk({}, final_finish_reason or "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    if WRAPPER_API_KEY and authorization != f"Bearer {WRAPPER_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    if payload.model not in ALLOWED_MODELS:
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    _chat_cfg = model_config(payload.model)
    if _chat_cfg and _chat_cfg.get("kind") != "chat":
        return openai_error_response(
            message=f"The model '{payload.model}' is not a chat model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )

    # response_format.type 유효성 검사: 지원하지 않는 type은 400으로 거부한다.
    # 허용 타입은 vertex.SUPPORTED_RESPONSE_FORMAT_TYPES 단일 출처를 공유한다(드리프트 방지).
    if payload.response_format is not None:
        rf_type = payload.response_format.get("type")
        if rf_type not in SUPPORTED_RESPONSE_FORMAT_TYPES:
            return openai_error_response(
                message=f"Unsupported response_format.type: {rf_type!r}. "
                        f"Allowed: {sorted(SUPPORTED_RESPONSE_FORMAT_TYPES)}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
                param="response_format",
            )

    chat_client: VertexChatClient = request.app.state.vertex_chat_client

    messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    if payload.stream:
        return _chat_completions_stream(chat_client, payload, messages)

    try:
        result = await chat_client.generate(
            model=payload.model,
            messages=messages,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop=payload.stop,
            response_format=payload.response_format,
        )
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    return {
        "id": _new_chat_completion_id(),
        "object": "chat.completion",
        "created": 0,
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
    }
