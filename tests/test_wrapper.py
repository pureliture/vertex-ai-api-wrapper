"""래퍼의 변환/라우팅 로직 단위 테스트 (실제 Vertex/GCP 호출 없이).

lifespan을 실제로 돌리되 VertexEmbeddingClient를 가짜로
monkeypatch해서, app.state가 TestClient의 이벤트 루프에서 정상 구성되게 한다.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest
from fastapi.testclient import TestClient

import app as wrapper
import vertex


# ---- 순수 함수 ----

def test_coerce_string_inputs_str():
    assert wrapper.coerce_string_inputs("hi") == ["hi"]


def test_coerce_string_inputs_list():
    assert wrapper.coerce_string_inputs(["a", "b"]) == ["a", "b"]


def test_coerce_string_inputs_rejects_tokens():
    with pytest.raises(ValueError):
        wrapper.coerce_string_inputs([[1, 2, 3]])


def test_chunked_splits_by_size():
    assert list(vertex.chunked(["a", "b", "c", "d", "e"], 2)) == [["a", "b"], ["c", "d"], ["e"]]


def test_chunked_size_floor_is_one():
    assert list(vertex.chunked(["a", "b"], 0)) == [["a"], ["b"]]


def test_status_mapping():
    assert wrapper.map_vertex_status_to_openai_type(429) == "rate_limit_error"
    assert wrapper.map_vertex_status_to_openai_type(503) == "api_error"
    assert wrapper.map_vertex_status_to_openai_type(400) == "invalid_request_error"


def test_encode_embedding_float_passthrough():
    assert wrapper.encode_embedding([0.1, 0.2], "float") == [0.1, 0.2]


def test_encode_embedding_base64_roundtrip():
    vals = [0.1, -0.2, 0.3]
    s = wrapper.encode_embedding(vals, "base64")
    assert isinstance(s, str)
    back = list(struct.unpack(f"<{len(vals)}f", base64.b64decode(s)))
    assert all(abs(a - b) < 1e-6 for a, b in zip(vals, back))


# ---- Model Registry Tests ----

def test_registry_defaults_present():
    """기본 모델 4개 모두 레지스트리에 있어야 한다."""
    reg = vertex.MODEL_REGISTRY
    assert "text-embedding-005" in reg
    assert "text-multilingual-embedding-002" in reg
    assert "gemini-embedding-001" in reg
    assert "gemini-embedding-2" in reg


def test_registry_defaults_api_types():
    """기본 모델들의 api 타입이 올바른지 확인."""
    reg = vertex.MODEL_REGISTRY
    assert reg["text-embedding-005"]["api"] == "predict"
    assert reg["text-multilingual-embedding-002"]["api"] == "predict"
    assert reg["gemini-embedding-001"]["api"] == "predict"
    assert reg["gemini-embedding-2"]["api"] == "embedContent"


def test_registry_defaults_max_instances():
    """기본 모델들의 max_instances가 올바른지 확인."""
    reg = vertex.MODEL_REGISTRY
    assert reg["text-embedding-005"]["max_instances"] == 5
    assert reg["text-multilingual-embedding-002"]["max_instances"] == 5
    assert reg["gemini-embedding-001"]["max_instances"] == 1
    assert reg["gemini-embedding-2"]["max_instances"] == 1


def test_model_config_returns_resolved_dict():
    """model_config()가 api, location, max_instances 키를 포함한 dict를 반환해야 한다."""
    cfg = vertex.model_config("text-embedding-005")
    assert cfg is not None
    assert "api" in cfg
    assert "location" in cfg
    assert "max_instances" in cfg


def test_model_config_returns_none_for_unknown():
    """알 수 없는 모델은 None을 반환해야 한다."""
    assert vertex.model_config("nonexistent-model-xyz") is None


def test_model_config_predict_uses_vertex_location():
    """predict API 모델은 VERTEX_LOCATION을 location으로 사용해야 한다."""
    cfg = vertex.model_config("text-embedding-005")
    assert cfg["location"] == vertex.VERTEX_LOCATION


def test_model_config_embedcontent_uses_global():
    """embedContent API 모델은 'global'을 location으로 사용해야 한다."""
    cfg = vertex.model_config("gemini-embedding-2")
    assert cfg["location"] == "global"


def test_model_config_embedcontent_explicit_location_overrides():
    """embedContent 모델에 location이 명시되면 그것을 사용해야 한다."""
    # 임시로 레지스트리를 수정하여 테스트
    old_reg = vertex.MODEL_REGISTRY.copy()
    try:
        vertex.MODEL_REGISTRY["test-embed-custom"] = {
            "api": "embedContent",
            "location": "us-east1",
            "max_instances": 1,
        }
        cfg = vertex.model_config("test-embed-custom")
        assert cfg["location"] == "us-east1"
    finally:
        vertex.MODEL_REGISTRY.clear()
        vertex.MODEL_REGISTRY.update(old_reg)


def test_allowed_models_returns_set():
    """allowed_models()가 set[str]을 반환해야 한다."""
    models = vertex.allowed_models()
    assert isinstance(models, set)


def test_allowed_models_contains_defaults():
    """allowed_models()에 기본 모델들이 포함되어야 한다."""
    models = vertex.allowed_models()
    assert "text-embedding-005" in models
    assert "text-multilingual-embedding-002" in models
    assert "gemini-embedding-001" in models
    assert "gemini-embedding-2" in models


def test_model_registry_json_env_adds_model(monkeypatch):
    """MODEL_REGISTRY_JSON 환경변수로 새 모델을 추가할 수 있어야 한다."""
    custom = json.dumps({"custom-model-v1": {"api": "predict", "max_instances": 3}})
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    # 모듈을 재로드하여 환경변수 반영
    import importlib
    importlib.reload(vertex)
    try:
        assert "custom-model-v1" in vertex.MODEL_REGISTRY
        assert vertex.MODEL_REGISTRY["custom-model-v1"]["max_instances"] == 3
        # 기존 모델도 유지되어야 한다
        assert "text-embedding-005" in vertex.MODEL_REGISTRY
    finally:
        monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
        importlib.reload(vertex)


def test_model_registry_json_env_overrides_existing(monkeypatch):
    """MODEL_REGISTRY_JSON으로 기존 모델의 설정을 덮어쓸 수 있어야 한다."""
    custom = json.dumps({"text-embedding-005": {"api": "predict", "max_instances": 10}})
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    import importlib
    importlib.reload(vertex)
    try:
        assert vertex.MODEL_REGISTRY["text-embedding-005"]["max_instances"] == 10
    finally:
        monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
        importlib.reload(vertex)


def test_model_registry_json_invalid_raises(monkeypatch):
    """MODEL_REGISTRY_JSON이 유효하지 않은 JSON이면 임포트 시 raise해야 한다."""
    monkeypatch.setenv("MODEL_REGISTRY_JSON", "not-valid-json{{{")
    import importlib
    with pytest.raises((ValueError, json.JSONDecodeError)):
        importlib.reload(vertex)
    monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
    importlib.reload(vertex)


def test_extra_models_env_backward_compat(monkeypatch):
    """EXTRA_MODELS 환경변수의 모델이 레지스트리에 predict API로 추가되어야 한다."""
    monkeypatch.setenv("EXTRA_MODELS", "my-extra-model-1,my-extra-model-2")
    import importlib
    importlib.reload(vertex)
    try:
        assert "my-extra-model-1" in vertex.MODEL_REGISTRY
        assert "my-extra-model-2" in vertex.MODEL_REGISTRY
        assert vertex.MODEL_REGISTRY["my-extra-model-1"]["api"] == "predict"
    finally:
        monkeypatch.delenv("EXTRA_MODELS", raising=False)
        importlib.reload(vertex)


def test_extra_models_does_not_override_registry(monkeypatch):
    """EXTRA_MODELS에 이미 레지스트리에 있는 모델을 넣어도 덮어쓰지 않아야 한다."""
    monkeypatch.setenv("EXTRA_MODELS", "gemini-embedding-2")
    import importlib
    importlib.reload(vertex)
    try:
        # gemini-embedding-2는 embedContent여야 하며, EXTRA_MODELS로 predict로 바뀌면 안 된다
        assert vertex.MODEL_REGISTRY["gemini-embedding-2"]["api"] == "embedContent"
    finally:
        monkeypatch.delenv("EXTRA_MODELS", raising=False)
        importlib.reload(vertex)


# ---- embedContent API unit tests (httpx mock) ----

@pytest.fixture
def mock_httpx_client(monkeypatch):
    """VertexEmbeddingClient의 httpx 클라이언트를 모킹한다."""
    import asyncio
    import httpx

    posted_requests = []

    class MockResponse:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        def json(self):
            return self._data

        @property
        def text(self):
            return json.dumps(self._data)

    class MockAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            posted_requests.append({"url": url, "headers": headers, "json": json})
            # embedContent 엔드포인트인지 확인
            if ":embedContent" in url:
                return MockResponse(200, {
                    "embedding": {"values": [0.1, 0.2, 0.3]},
                    "usageMetadata": {"tokenCount": 5},
                })
            else:
                # predict 엔드포인트
                texts = json.get("instances", [])
                predictions = [
                    {"embeddings": {"values": [0.1, 0.2, 0.3], "statistics": {"token_count": 2}}}
                    for _ in texts
                ]
                return MockResponse(200, {"predictions": predictions})

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    return posted_requests


@pytest.fixture
def mock_token_provider(monkeypatch):
    """GoogleAccessTokenProvider를 가짜로 교체한다."""
    class FakeTokenProvider:
        def __init__(self):
            self.project_id = "test-project"

        async def get_token(self):
            return "fake-token"

    monkeypatch.setattr(vertex, "GoogleAccessTokenProvider", FakeTokenProvider)
    return FakeTokenProvider()


@pytest.mark.anyio
async def test_embed_content_url_has_no_region_prefix(mock_httpx_client, mock_token_provider):
    """gemini-embedding-2의 embedContent URL은 region prefix가 없어야 한다."""
    import asyncio

    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="gemini-embedding-2",
        texts=["hello"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    assert len(mock_httpx_client) == 1
    url = mock_httpx_client[0]["url"]
    # host는 정확히 aiplatform.googleapis.com (region prefix 없음)
    assert "aiplatform.googleapis.com" in url
    assert url.startswith("https://aiplatform.googleapis.com/"), f"URL should start with https://aiplatform.googleapis.com/ but got: {url}"
    # global이 URL 경로에 있어야 함
    assert "/locations/global/" in url
    assert "gemini-embedding-2:embedContent" in url


@pytest.mark.anyio
async def test_embed_content_request_body_shape(mock_httpx_client, mock_token_provider):
    """embedContent 요청 body가 올바른 형태여야 한다."""
    client = vertex.VertexEmbeddingClient()
    await client.embed(
        model="gemini-embedding-2",
        texts=["test text"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    assert len(mock_httpx_client) == 1
    body = mock_httpx_client[0]["json"]
    assert "content" in body
    assert "parts" in body["content"]
    assert body["content"]["parts"][0]["text"] == "test text"
    assert body.get("taskType") == "RETRIEVAL_DOCUMENT"


@pytest.mark.anyio
async def test_embed_content_with_dimensions(mock_httpx_client, mock_token_provider):
    """outputDimensionality가 embedContent 요청에 포함되어야 한다."""
    client = vertex.VertexEmbeddingClient()
    await client.embed(
        model="gemini-embedding-2",
        texts=["test text"],
        dimensions=256,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    body = mock_httpx_client[0]["json"]
    assert body.get("outputDimensionality") == 256


@pytest.mark.anyio
async def test_embed_content_response_parse(mock_httpx_client, mock_token_provider):
    """embedContent 응답에서 embedding.values를 올바르게 파싱해야 한다."""
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="gemini-embedding-2",
        texts=["hello"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    assert len(results) == 1
    assert results[0]["values"] == [0.1, 0.2, 0.3]


@pytest.mark.anyio
async def test_embed_content_multiple_texts_one_call_each(mock_httpx_client, mock_token_provider):
    """embedContent 모델은 텍스트 1개당 1번 호출해야 한다."""
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="gemini-embedding-2",
        texts=["a", "b", "c"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    # 3개의 텍스트 = 3번의 API 호출
    assert len(mock_httpx_client) == 3
    # 결과는 3개여야 함
    assert len(results) == 3


@pytest.mark.anyio
async def test_embed_content_preserves_input_order(mock_httpx_client, mock_token_provider):
    """embedContent embed() 결과가 입력 순서대로 반환되어야 한다."""
    import asyncio
    import httpx

    call_count = 0
    responses = [
        {"embedding": {"values": [1.0, 0.0]}, "usageMetadata": {}},
        {"embedding": {"values": [0.0, 1.0]}, "usageMetadata": {}},
        {"embedding": {"values": [0.5, 0.5]}, "usageMetadata": {}},
    ]

    class OrderedMockResponse:
        def __init__(self, data):
            self.status_code = 200
            self._data = data
        def json(self):
            return self._data

    class OrderedMockClient:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            nonlocal call_count
            resp = OrderedMockResponse(responses[call_count % len(responses)])
            call_count += 1
            return resp

        async def aclose(self):
            pass

    monkeypatch_attr = httpx.AsyncClient
    httpx.AsyncClient = OrderedMockClient
    try:
        client = vertex.VertexEmbeddingClient()
        results = await client.embed(
            model="gemini-embedding-2",
            texts=["first", "second", "third"],
            dimensions=None,
            task_type="RETRIEVAL_DOCUMENT",
            title=None,
        )
        assert results[0]["values"] == [1.0, 0.0]
        assert results[1]["values"] == [0.0, 1.0]
        assert results[2]["values"] == [0.5, 0.5]
    finally:
        httpx.AsyncClient = monkeypatch_attr


# ---- Unified embed() return contract ----

@pytest.mark.anyio
async def test_embed_predict_returns_flat_list(mock_httpx_client, mock_token_provider):
    """predict API embed()가 flat list[dict]를 반환해야 한다."""
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="text-embedding-005",
        texts=["a", "b"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert "values" in r
        assert "token_count" in r
        assert isinstance(r["values"], list)
        assert isinstance(r["token_count"], int)


@pytest.mark.anyio
async def test_embed_embedcontent_returns_flat_list(mock_httpx_client, mock_token_provider):
    """embedContent API embed()도 flat list[dict]를 반환해야 한다."""
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="gemini-embedding-2",
        texts=["a"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert "values" in results[0]
    assert "token_count" in results[0]


@pytest.mark.anyio
async def test_embed_predict_token_count_from_statistics(mock_httpx_client, mock_token_provider):
    """predict API의 token_count가 statistics.token_count에서 읽혀야 한다."""
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="text-embedding-005",
        texts=["a"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )
    assert results[0]["token_count"] == 2


@pytest.mark.anyio
async def test_embed_embedcontent_token_count_from_usage_metadata(monkeypatch, mock_token_provider):
    """embedContent의 token_count가 usageMetadata에서 읽혀야 한다."""
    import httpx

    class MockClientWithUsage:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "embedding": {"values": [1.0, 2.0]},
                    "usageMetadata": {"tokenCount": 42},
                },
                "text": "{}",
            })()

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockClientWithUsage)
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="gemini-embedding-2",
        texts=["hello"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )
    assert results[0]["token_count"] == 42


@pytest.mark.anyio
async def test_embed_embedcontent_token_count_defaults_zero_if_missing(monkeypatch, mock_token_provider):
    """usageMetadata가 없거나 tokenCount가 없으면 0으로 기본값 처리해야 한다."""
    import httpx

    class MockClientNoUsage:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "embedding": {"values": [1.0]},
                    # usageMetadata 없음
                },
                "text": "{}",
            })()

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockClientNoUsage)
    client = vertex.VertexEmbeddingClient()
    results = await client.embed(
        model="gemini-embedding-2",
        texts=["hello"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )
    assert results[0]["token_count"] == 0


# ---- 기존 KNOWN_MAX_INSTANCES 호환성 ----

def test_known_max_instances_backward_compat():
    """KNOWN_MAX_INSTANCES는 여전히 접근 가능해야 한다 (backward compat)."""
    assert hasattr(vertex, "KNOWN_MAX_INSTANCES")
    assert vertex.KNOWN_MAX_INSTANCES.get("gemini-embedding-001") == 1
    assert vertex.KNOWN_MAX_INSTANCES.get("text-embedding-005") == 5


# ---- 엔드포인트 (Vertex 호출은 가짜로 대체) ----

class _FakeVertexService:
    """VertexEmbeddingClient 흉내. embed()가 flat list[dict]를 반환하는 새 계약."""

    def __init__(self, *_a, **_k):
        self.calls: list[list[str]] = []
        self._model = None

    async def embed(self, *, model, texts, dimensions, task_type, title):
        self._model = model
        # 배치 로직 테스트를 위해 청크 크기를 여기서 흉내 냄
        cfg = vertex.model_config(model)
        batch_size = cfg["max_instances"] if cfg else vertex.DEFAULT_MAX_INSTANCES
        self.calls.extend(list(vertex.chunked(texts, batch_size)))

        # 새 flat contract: list[{"values": ..., "token_count": ...}]
        return [
            {"values": [0.1, 0.2, 0.3], "token_count": 2}
            for _ in texts
        ]

    async def close(self):
        pass


class _FakeVertexChatService:
    """VertexChatClient 기본 흉내 (embedding 테스트에서 lifespan 오류 방지용)."""

    def __init__(self, *_a, **_k):
        pass

    async def generate(self, **kw):
        return {
            "text": "fake",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def close(self):
        pass


@pytest.fixture
def client_with_fake(monkeypatch):
    fake = _FakeVertexService()
    fake_chat = _FakeVertexChatService()
    # app.py의 lifespan에서 생성되는 VertexEmbeddingClient / VertexChatClient 교체
    monkeypatch.setattr(wrapper, "VertexEmbeddingClient", lambda: fake)
    monkeypatch.setattr(wrapper, "VertexChatClient", lambda: fake_chat)
    with TestClient(wrapper.app) as c:
        yield c, fake


def test_embeddings_order_and_usage(client_with_fake):
    client, _ = client_with_fake
    r = client.post("/v1/embeddings", json={"model": "text-embedding-005", "input": ["a", "b", "c"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert all(d["embedding"] == [0.1, 0.2, 0.3] for d in body["data"])
    assert body["usage"]["total_tokens"] == 6  # 3 텍스트 x 2 토큰


def test_gemini_001_splits_into_single_instance_calls(client_with_fake):
    client, fake = client_with_fake
    client.post("/v1/embeddings", json={"model": "gemini-embedding-001", "input": ["x", "y", "z"]})
    assert sorted(len(c) for c in fake.calls) == [1, 1, 1]


def test_text_005_splits_by_five(client_with_fake):
    client, fake = client_with_fake
    client.post("/v1/embeddings", json={"model": "text-embedding-005", "input": [str(i) for i in range(12)]})
    assert sorted((len(c) for c in fake.calls), reverse=True) == [5, 5, 2]


def test_base64_response_roundtrip(client_with_fake):
    client, _ = client_with_fake
    r = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-005", "input": ["a"], "encoding_format": "base64"},
    )
    assert r.status_code == 200
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, str)
    back = list(struct.unpack("<3f", base64.b64decode(emb)))
    assert all(abs(a - b) < 1e-6 for a, b in zip([0.1, 0.2, 0.3], back))


def test_float_still_returns_list(client_with_fake):
    client, _ = client_with_fake
    r = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-005", "input": ["a"], "encoding_format": "float"},
    )
    assert isinstance(r.json()["data"][0]["embedding"], list)


def test_unknown_model_rejected(client_with_fake):
    client, fake = client_with_fake
    r = client.post("/v1/embeddings", json={"model": "../evil", "input": "a"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    assert fake.calls == []  # Vertex 호출 전에 차단


def test_invalid_encoding_format_is_openai_error(client_with_fake):
    client, _ = client_with_fake
    r = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-005", "input": "a", "encoding_format": "xml"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"  # 422 default 아님


def test_list_models(client_with_fake):
    client, _ = client_with_fake
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert {"gemini-embedding-001", "text-embedding-005", "gemini-embedding-2"} <= ids


def test_retrieve_model_known_and_unknown(client_with_fake):
    client, _ = client_with_fake
    assert client.get("/v1/models/text-embedding-005").status_code == 200
    assert client.get("/v1/models/nope").status_code == 404


def test_wrapper_api_key_enforced(client_with_fake, monkeypatch):
    client, _ = client_with_fake
    monkeypatch.setattr(wrapper, "WRAPPER_API_KEY", "secret-key")
    bad = client.post("/v1/embeddings", json={"model": "text-embedding-005", "input": "a"})
    assert bad.status_code == 401
    ok = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer secret-key"},
        json={"model": "text-embedding-005", "input": "a"},
    )
    assert ok.status_code == 200


def test_gemini_embedding_2_allowed(client_with_fake):
    """gemini-embedding-2가 허용된 모델이어야 한다."""
    client, _ = client_with_fake
    r = client.post("/v1/embeddings", json={"model": "gemini-embedding-2", "input": ["test"]})
    assert r.status_code == 200


def test_gemini_embedding_2_in_models_list(client_with_fake):
    """gemini-embedding-2가 /v1/models 목록에 있어야 한다."""
    client, _ = client_with_fake
    r = client.get("/v1/models")
    ids = {m["id"] for m in r.json()["data"]}
    assert "gemini-embedding-2" in ids


def test_gemini_embedding_2_post_returns_embeddings(client_with_fake):
    """gemini-embedding-2로 POST하면 embedding data가 반환되어야 한다."""
    client, _ = client_with_fake
    r = client.post("/v1/embeddings", json={"model": "gemini-embedding-2", "input": ["hello", "world"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert body["usage"]["total_tokens"] == 4  # 2 텍스트 x 2 토큰


def test_allowed_models_comes_from_vertex(client_with_fake):
    """app.py의 ALLOWED_MODELS가 vertex.allowed_models()에서 와야 한다."""
    # vertex.allowed_models()와 wrapper.ALLOWED_MODELS가 동일해야 한다
    assert vertex.allowed_models() == wrapper.ALLOWED_MODELS


# ---- [Important 1] predict 경로 길이 불일치 silent 손실 방지 ----

@pytest.mark.anyio
async def test_predict_prediction_count_mismatch_raises_502(monkeypatch, mock_token_provider):
    """predict 응답의 prediction 개수가 chunk 텍스트 개수와 다르면 502를 raise해야 한다."""
    import httpx

    class MismatchClient:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            # 텍스트 3개를 보냈지만 prediction은 2개만 반환 (silent 손실 시나리오)
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "predictions": [
                        {"embeddings": {"values": [0.1], "statistics": {"token_count": 1}}},
                        {"embeddings": {"values": [0.2], "statistics": {"token_count": 1}}},
                    ]
                },
                "text": "{}",
            })()

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MismatchClient)
    client = vertex.VertexEmbeddingClient()
    with pytest.raises(vertex.VertexAPIError) as excinfo:
        await client.embed(
            model="text-embedding-005",
            texts=["a", "b", "c"],
            dimensions=None,
            task_type="RETRIEVAL_DOCUMENT",
            title=None,
        )
    assert excinfo.value.status_code == 502


# ---- [Important 2] embedContent 빈 응답 silent 방지 ----

@pytest.mark.anyio
async def test_embed_content_empty_response_raises_502(monkeypatch, mock_token_provider):
    """embedContent 응답에 embedding/values가 없으면 502를 raise해야 한다."""
    import httpx

    class EmptyClient:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {},  # embedding 키 없음
                "text": "{}",
            })()

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", EmptyClient)
    client = vertex.VertexEmbeddingClient()
    with pytest.raises(vertex.VertexAPIError) as excinfo:
        await client.embed(
            model="gemini-embedding-2",
            texts=["hello"],
            dimensions=None,
            task_type="RETRIEVAL_DOCUMENT",
            title=None,
        )
    assert excinfo.value.status_code == 502


@pytest.mark.anyio
async def test_embed_content_empty_values_raises_502(monkeypatch, mock_token_provider):
    """embedContent 응답의 values가 빈 리스트이면 502를 raise해야 한다."""
    import httpx

    class EmptyValuesClient:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {"embedding": {"values": []}},
                "text": "{}",
            })()

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", EmptyValuesClient)
    client = vertex.VertexEmbeddingClient()
    with pytest.raises(vertex.VertexAPIError) as excinfo:
        await client.embed(
            model="gemini-embedding-2",
            texts=["hello"],
            dimensions=None,
            task_type="RETRIEVAL_DOCUMENT",
            title=None,
        )
    assert excinfo.value.status_code == 502


# ---- [Important 3] UNSPECIFIED task_type 생략 ----

@pytest.mark.anyio
async def test_embed_content_unspecified_task_type_omitted(mock_httpx_client, mock_token_provider):
    """task_type이 UNSPECIFIED이면 embedContent body에 taskType 키가 없어야 한다."""
    client = vertex.VertexEmbeddingClient()
    await client.embed(
        model="gemini-embedding-2",
        texts=["hello"],
        dimensions=None,
        task_type="UNSPECIFIED",
        title=None,
    )
    body = mock_httpx_client[0]["json"]
    assert "taskType" not in body


@pytest.mark.anyio
async def test_predict_unspecified_task_type_omitted(mock_httpx_client, mock_token_provider):
    """task_type이 UNSPECIFIED이면 predict instance에 task_type 키가 없어야 한다."""
    client = vertex.VertexEmbeddingClient()
    await client.embed(
        model="text-embedding-005",
        texts=["hello"],
        dimensions=None,
        task_type="UNSPECIFIED",
        title=None,
    )
    body = mock_httpx_client[0]["json"]
    instance = body["instances"][0]
    assert "task_type" not in instance


# ---- [Important 4] embedContent taskType 검증 ----

@pytest.mark.anyio
async def test_embed_content_task_type_in_body(mock_httpx_client, mock_token_provider):
    """embedContent 요청 body에 taskType이 올바르게 포함되어야 한다."""
    client = vertex.VertexEmbeddingClient()
    await client.embed(
        model="gemini-embedding-2",
        texts=["hello"],
        dimensions=None,
        task_type="RETRIEVAL_DOCUMENT",
        title=None,
    )
    body = mock_httpx_client[0]["json"]
    assert body.get("taskType") == "RETRIEVAL_DOCUMENT"


# ---- [Important 5] registry api 값 검증 ----

def test_registry_invalid_api_raises(monkeypatch):
    """MODEL_REGISTRY_JSON 엔트리의 api가 알 수 없는 값이면 build 시 ValueError."""
    import importlib
    custom = json.dumps({"typo-model": {"api": "Predict", "max_instances": 1}})
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    with pytest.raises(ValueError):
        importlib.reload(vertex)
    monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
    importlib.reload(vertex)


def test_registry_missing_api_raises(monkeypatch):
    """MODEL_REGISTRY_JSON 엔트리에 api 키가 없으면 build 시 ValueError."""
    import importlib
    custom = json.dumps({"no-api-model": {"max_instances": 1}})
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    with pytest.raises(ValueError):
        importlib.reload(vertex)
    monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
    importlib.reload(vertex)


# ===========================================================================
# Chat Completions — Registry & kind field
# ===========================================================================

def test_chat_models_in_registry():
    """gemini-2.5-flash, gemini-2.5-pro가 레지스트리에 kind='chat'으로 있어야 한다."""
    reg = vertex.MODEL_REGISTRY
    assert "gemini-2.5-flash" in reg
    assert "gemini-2.5-pro" in reg
    assert reg["gemini-2.5-flash"].get("kind") == "chat"
    assert reg["gemini-2.5-pro"].get("kind") == "chat"


def test_chat_models_api_is_generate_content():
    """채팅 모델의 api는 'generateContent'여야 한다."""
    reg = vertex.MODEL_REGISTRY
    assert reg["gemini-2.5-flash"]["api"] == "generateContent"
    assert reg["gemini-2.5-pro"]["api"] == "generateContent"


def test_embedding_models_resolve_kind_embedding():
    """기존 임베딩 모델은 model_config()에서 kind='embedding'을 반환해야 한다."""
    cfg = vertex.model_config("text-embedding-005")
    assert cfg is not None
    assert cfg.get("kind") == "embedding"

    cfg2 = vertex.model_config("gemini-embedding-2")
    assert cfg2 is not None
    assert cfg2.get("kind") == "embedding"


def test_chat_model_config_resolves_kind():
    """채팅 모델의 model_config()에서 kind='chat'이 포함되어야 한다."""
    cfg = vertex.model_config("gemini-2.5-flash")
    assert cfg is not None
    assert cfg.get("kind") == "chat"
    assert cfg.get("location") == "us-central1"


def test_api_validation_allows_generate_content(monkeypatch):
    """MODEL_REGISTRY_JSON으로 generateContent api 모델을 추가할 수 있어야 한다."""
    import importlib
    custom = json.dumps({
        "my-chat-model": {"api": "generateContent", "kind": "chat", "location": "us-central1"}
    })
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    importlib.reload(vertex)
    try:
        assert "my-chat-model" in vertex.MODEL_REGISTRY
        assert vertex.MODEL_REGISTRY["my-chat-model"]["api"] == "generateContent"
    finally:
        monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
        importlib.reload(vertex)


def test_model_registry_json_can_add_chat_model(monkeypatch):
    """MODEL_REGISTRY_JSON 환경변수로 kind='chat' 모델을 추가할 수 있어야 한다."""
    import importlib
    custom = json.dumps({
        "custom-chat-v1": {"api": "generateContent", "kind": "chat", "location": "us-east1"}
    })
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    importlib.reload(vertex)
    try:
        cfg = vertex.model_config("custom-chat-v1")
        assert cfg is not None
        assert cfg.get("kind") == "chat"
        assert cfg.get("location") == "us-east1"
        # 기존 모델도 유지되어야 한다
        assert "text-embedding-005" in vertex.MODEL_REGISTRY
    finally:
        monkeypatch.delenv("MODEL_REGISTRY_JSON", raising=False)
        importlib.reload(vertex)


def test_allowed_models_includes_chat_models():
    """allowed_models()에 chat 모델도 포함되어야 한다."""
    models = vertex.allowed_models()
    assert "gemini-2.5-flash" in models
    assert "gemini-2.5-pro" in models


# ===========================================================================
# Chat Completions — Message mapping (VertexChatClient)
# ===========================================================================

@pytest.fixture
def chat_client(monkeypatch):
    """VertexChatClient 인스턴스를 반환 (httpx mock 없이)."""
    import httpx

    class MockChatHttpClient:
        def __init__(self, *a, **kw):
            self.last_request: dict = {}

        async def post(self, url, *, headers=None, json=None):
            self.last_request = {"url": url, "headers": headers, "json": json}
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": "Hello!"}]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 8,
                    },
                },
                "text": "{}",
            })()

        async def aclose(self):
            pass

    mock_http = MockChatHttpClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)
    monkeypatch.setattr(vertex, "GoogleAccessTokenProvider", lambda: type("FTP", (), {
        "project_id": "test-project",
        "get_token": lambda self: __import__("asyncio").coroutine(lambda: "fake-token")(),
    })())

    class FakeTokenProvider:
        def __init__(self):
            self.project_id = "test-project"

        async def get_token(self):
            return "fake-token"

    client = vertex.VertexChatClient(token_provider=FakeTokenProvider())
    client.http = mock_http
    return client, mock_http


@pytest.mark.anyio
async def test_chat_system_message_becomes_system_instruction(chat_client):
    """system role 메시지는 Vertex systemInstruction으로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    await client.generate(
        model="gemini-2.5-flash",
        messages=messages,
    )
    body = mock_http.last_request["json"]
    assert "systemInstruction" in body
    assert body["systemInstruction"]["parts"][0]["text"] == "You are helpful."


@pytest.mark.anyio
async def test_chat_user_message_maps_to_user_role(chat_client):
    """user role 메시지는 Vertex contents의 role='user'로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hello"}]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    contents = body.get("contents", [])
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello"


@pytest.mark.anyio
async def test_chat_assistant_message_maps_to_model_role(chat_client):
    """assistant role 메시지는 Vertex contents의 role='model'로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    contents = body.get("contents", [])
    assert len(contents) == 2
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi there"


@pytest.mark.anyio
async def test_chat_multiple_messages_preserve_order(chat_client):
    """여러 메시지가 순서대로 contents에 들어가야 한다."""
    client, mock_http = chat_client
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user", "content": "Third"},
    ]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    contents = body.get("contents", [])
    assert len(contents) == 3
    assert contents[0]["parts"][0]["text"] == "First"
    assert contents[1]["parts"][0]["text"] == "Second"
    assert contents[2]["parts"][0]["text"] == "Third"


@pytest.mark.anyio
async def test_chat_multiple_system_messages_concatenated(chat_client):
    """여러 system 메시지는 하나의 systemInstruction으로 합쳐져야 한다."""
    client, mock_http = chat_client
    messages = [
        {"role": "system", "content": "Part1."},
        {"role": "system", "content": "Part2."},
        {"role": "user", "content": "Hi"},
    ]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    assert "systemInstruction" in body
    parts = body["systemInstruction"]["parts"]
    combined = " ".join(p["text"] for p in parts)
    assert "Part1." in combined
    assert "Part2." in combined


@pytest.mark.anyio
async def test_chat_no_system_message_omits_system_instruction(chat_client):
    """system 메시지가 없으면 systemInstruction 키가 없어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hello"}]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    assert "systemInstruction" not in body


@pytest.mark.anyio
async def test_chat_content_as_list_of_parts_extracts_text(chat_client):
    """content가 {type:'text', text:...} 리스트 형태여도 텍스트를 추출해야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello from list"}]}]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    contents = body.get("contents", [])
    assert contents[0]["parts"][0]["text"] == "Hello from list"


# ===========================================================================
# Chat Completions — generationConfig mapping
# ===========================================================================

@pytest.mark.anyio
async def test_chat_max_tokens_maps_to_max_output_tokens(chat_client):
    """max_tokens는 generationConfig.maxOutputTokens로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-flash", messages=messages, max_tokens=100)
    body = mock_http.last_request["json"]
    assert body.get("generationConfig", {}).get("maxOutputTokens") == 100


@pytest.mark.anyio
async def test_chat_temperature_maps(chat_client):
    """temperature는 generationConfig.temperature로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-flash", messages=messages, temperature=0.7)
    body = mock_http.last_request["json"]
    assert body.get("generationConfig", {}).get("temperature") == 0.7


@pytest.mark.anyio
async def test_chat_top_p_maps(chat_client):
    """top_p는 generationConfig.topP로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-flash", messages=messages, top_p=0.9)
    body = mock_http.last_request["json"]
    assert body.get("generationConfig", {}).get("topP") == 0.9


@pytest.mark.anyio
async def test_chat_stop_string_maps_to_list(chat_client):
    """stop이 문자열이면 stopSequences 단일 원소 리스트로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-flash", messages=messages, stop="END")
    body = mock_http.last_request["json"]
    assert body.get("generationConfig", {}).get("stopSequences") == ["END"]


@pytest.mark.anyio
async def test_chat_stop_list_maps(chat_client):
    """stop이 리스트이면 stopSequences로 그대로 매핑되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-flash", messages=messages, stop=["END", "STOP"])
    body = mock_http.last_request["json"]
    assert body.get("generationConfig", {}).get("stopSequences") == ["END", "STOP"]


@pytest.mark.anyio
async def test_chat_omitted_params_absent_from_generation_config(chat_client):
    """제공되지 않은 파라미터는 generationConfig에 없어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-flash", messages=messages)
    body = mock_http.last_request["json"]
    gen_cfg = body.get("generationConfig", {})
    assert "maxOutputTokens" not in gen_cfg
    assert "temperature" not in gen_cfg
    assert "topP" not in gen_cfg
    assert "stopSequences" not in gen_cfg


@pytest.mark.anyio
async def test_chat_no_generation_config_when_all_omitted(chat_client):
    """생성 파라미터가 모두 없으면 generationConfig 자체가 없어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-pro", messages=messages)
    body = mock_http.last_request["json"]
    assert "generationConfig" not in body


# ===========================================================================
# Chat Completions — Response parsing
# ===========================================================================

@pytest.mark.anyio
async def test_chat_response_text_extracted(chat_client):
    """응답에서 텍스트가 올바르게 추출되어야 한다."""
    client, _ = chat_client
    result = await client.generate(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result["text"] == "Hello!"


@pytest.mark.anyio
async def test_chat_response_usage_extracted(chat_client):
    """응답의 usageMetadata가 usage dict로 변환되어야 한다."""
    client, _ = chat_client
    result = await client.generate(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    usage = result["usage"]
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == 8


@pytest.mark.anyio
async def test_chat_finish_reason_stop_mapped(monkeypatch):
    """finishReason 'STOP'은 'stop'으로 매핑되어야 한다."""
    import httpx

    class FakeTokenProvider:
        def __init__(self):
            self.project_id = "test-project"
        async def get_token(self):
            return "fake-token"

    class MockHttp:
        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                },
                "text": "{}",
            })()
        async def aclose(self): pass

    client = vertex.VertexChatClient(token_provider=FakeTokenProvider())
    client.http = MockHttp()
    result = await client.generate(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_chat_finish_reason_max_tokens_mapped(monkeypatch):
    """finishReason 'MAX_TOKENS'은 'length'로 매핑되어야 한다."""
    import httpx

    class FakeTokenProvider:
        def __init__(self):
            self.project_id = "test-project"
        async def get_token(self):
            return "fake-token"

    class MockHttp:
        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{"content": {"role": "model", "parts": [{"text": "truncated"}]}, "finishReason": "MAX_TOKENS"}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15},
                },
                "text": "{}",
            })()
        async def aclose(self): pass

    client = vertex.VertexChatClient(token_provider=FakeTokenProvider())
    client.http = MockHttp()
    result = await client.generate(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result["finish_reason"] == "length"


@pytest.mark.anyio
async def test_chat_finish_reason_safety_mapped(monkeypatch):
    """finishReason 'SAFETY'는 'content_filter'로 매핑되어야 한다."""
    import httpx

    class FakeTokenProvider:
        def __init__(self):
            self.project_id = "test-project"
        async def get_token(self):
            return "fake-token"

    class MockHttp:
        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "SAFETY"}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 0, "totalTokenCount": 3},
                },
                "text": "{}",
            })()
        async def aclose(self): pass

    client = vertex.VertexChatClient(token_provider=FakeTokenProvider())
    client.http = MockHttp()
    result = await client.generate(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result["finish_reason"] == "content_filter"
    assert result["text"] == ""


@pytest.mark.anyio
async def test_chat_candidate_without_parts_returns_empty_text(monkeypatch):
    """candidate에 parts가 없으면 text=''으로 처리해야 한다."""
    import httpx

    class FakeTokenProvider:
        def __init__(self):
            self.project_id = "test-project"
        async def get_token(self):
            return "fake-token"

    class MockHttp:
        async def post(self, url, *, headers=None, json=None):
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{"content": {"role": "model"}, "finishReason": "SAFETY"}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 0, "totalTokenCount": 3},
                },
                "text": "{}",
            })()
        async def aclose(self): pass

    client = vertex.VertexChatClient(token_provider=FakeTokenProvider())
    client.http = MockHttp()
    result = await client.generate(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result["text"] == ""


# ===========================================================================
# Chat Completions — /v1/chat/completions endpoint
# ===========================================================================

class _FakeChatService:
    """VertexChatClient 흉내."""

    def __init__(self, *_a, **_k):
        self.last_call: dict = {}

    async def generate(self, *, model, messages, max_tokens=None, temperature=None, top_p=None, stop=None, response_format=None):
        self.last_call = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "response_format": response_format,
        }
        return {
            "text": "Hello, I am Gemini!",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }

    async def stream_chat(self, *, model, messages, max_tokens=None, temperature=None, top_p=None, stop=None, response_format=None):
        self.last_call = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
        }
        yield {
            "delta_text": "Hello, I am Gemini!",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }

    async def close(self):
        pass


@pytest.fixture
def chat_app_client(monkeypatch):
    """chat completions 엔드포인트 테스트용 TestClient."""
    fake_chat = _FakeChatService()
    fake_embed = _FakeVertexService()
    monkeypatch.setattr(wrapper, "VertexEmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr(wrapper, "VertexChatClient", lambda: fake_chat)
    with TestClient(wrapper.app) as c:
        yield c, fake_chat


def test_chat_completions_returns_openai_shape(chat_app_client):
    """POST /v1/chat/completions가 올바른 OpenAI ChatCompletion 형태를 반환해야 한다."""
    client, _ = chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "id" in body
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "gemini-2.5-flash"
    choices = body["choices"]
    assert len(choices) == 1
    assert choices[0]["index"] == 0
    assert choices[0]["message"]["role"] == "assistant"
    assert choices[0]["message"]["content"] == "Hello, I am Gemini!"
    assert choices[0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == 11


def test_chat_completions_unknown_model_returns_404(chat_app_client):
    """존재하지 않는 모델은 404 model_not_found를 반환해야 한다."""
    client, _ = chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_chat_completions_embedding_model_returns_error(chat_app_client):
    """임베딩 모델을 chat 엔드포인트에 사용하면 에러를 반환해야 한다."""
    client, _ = chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "text-embedding-005",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "not a chat model" in error["message"].lower() or "embedding" in error["message"].lower()


def test_chat_completions_auth_enforced(chat_app_client, monkeypatch):
    """WRAPPER_API_KEY가 설정된 경우 인증이 강제되어야 한다."""
    client, _ = chat_app_client
    monkeypatch.setattr(wrapper, "WRAPPER_API_KEY", "secret-key")
    bad = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert bad.status_code == 401

    ok = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-key"},
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert ok.status_code == 200


def test_chat_model_rejected_on_embeddings_endpoint(chat_app_client):
    """채팅 모델을 /v1/embeddings 엔드포인트에 사용하면 에러를 반환해야 한다."""
    client, _ = chat_app_client
    r = client.post("/v1/embeddings", json={
        "model": "gemini-2.5-flash",
        "input": ["hello"],
    })
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"


def test_list_models_includes_chat_models(chat_app_client):
    """GET /v1/models 응답에 chat 모델도 포함되어야 한다."""
    client, _ = chat_app_client
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "gemini-2.5-flash" in ids
    assert "gemini-2.5-pro" in ids


# ===========================================================================
# Chat Completions — Streaming (SSE) VertexChatClient.stream_chat
# ===========================================================================

class _FakeStreamTokenProvider:
    def __init__(self):
        self.project_id = "test-project"

    async def get_token(self):
        return "fake-token"


def _make_stream_chat_client(sse_lines, status_code=200, error_body=None, capture=None):
    """httpx.AsyncClient.stream을 mock한 VertexChatClient를 만든다.

    sse_lines: aiter_lines가 yield할 줄들의 리스트.
    capture: 스트림 요청을 기록할 dict (url, json, headers).
    """
    class MockStreamResponse:
        def __init__(self):
            self.status_code = status_code

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

        async def aread(self):
            return (error_body or "").encode("utf-8") if isinstance(error_body, str) else (error_body or b"")

        @property
        def text(self):
            return error_body if isinstance(error_body, str) else ""

        def json(self):
            if isinstance(error_body, dict):
                return error_body
            return json.loads(error_body) if error_body else {}

    class MockStreamCtx:
        def __init__(self, method, url, *, headers=None, json=None):
            if capture is not None:
                capture["method"] = method
                capture["url"] = url
                capture["headers"] = headers
                capture["json"] = json
            self._resp = MockStreamResponse()

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *exc):
            return False

    class MockHttp:
        def stream(self, method, url, *, headers=None, json=None):
            return MockStreamCtx(method, url, headers=headers, json=json)

        async def aclose(self):
            pass

    client = vertex.VertexChatClient(token_provider=_FakeStreamTokenProvider())
    client.http = MockHttp()
    return client


@pytest.mark.anyio
async def test_stream_chat_yields_deltas_and_finish_reason():
    """stream_chat이 델타 텍스트와 마지막 finish_reason/usage를 순차 yield해야 한다."""
    sse_lines = [
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}',
        '',
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":", world"}]}}]}',
        '',
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":3,"totalTokenCount":6}}',
        '',
    ]
    client = _make_stream_chat_client(sse_lines)
    events = []
    async for ev in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        events.append(ev)

    # 델타 텍스트들을 이으면 전체 텍스트 복원
    full = "".join(ev["delta_text"] for ev in events)
    assert full == "Hello, world!"
    # 마지막 이벤트에 finish_reason과 usage
    last = events[-1]
    assert last["finish_reason"] == "stop"
    assert last["usage"]["prompt_tokens"] == 3
    assert last["usage"]["completion_tokens"] == 3
    assert last["usage"]["total_tokens"] == 6


@pytest.mark.anyio
async def test_stream_chat_uses_stream_generate_content_url():
    """stream_chat은 :streamGenerateContent?alt=sse URL을 사용해야 한다."""
    capture = {}
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{}}',
        '',
    ]
    client = _make_stream_chat_client(sse_lines, capture=capture)
    async for _ in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        pass

    assert capture["method"] == "POST"
    assert ":streamGenerateContent" in capture["url"]
    assert "alt=sse" in capture["url"]


@pytest.mark.anyio
async def test_stream_chat_maps_messages_and_generation_config():
    """stream_chat 요청 body가 비스트림과 동일한 매핑(contents/systemInstruction/generationConfig)을 써야 한다."""
    capture = {}
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{}}',
        '',
    ]
    client = _make_stream_chat_client(sse_lines, capture=capture)
    async for _ in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=50,
        temperature=0.5,
        top_p=0.8,
        stop="END",
    ):
        pass

    body = capture["json"]
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][0]["parts"][0]["text"] == "Hi"
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    gen = body["generationConfig"]
    assert gen["maxOutputTokens"] == 50
    assert gen["temperature"] == 0.5
    assert gen["topP"] == 0.8
    assert gen["stopSequences"] == ["END"]


@pytest.mark.anyio
async def test_stream_chat_ignores_done_and_blank_lines():
    """[DONE] 또는 비 data: 줄은 무시하고 크래시하지 않아야 한다."""
    sse_lines = [
        ': comment line',
        'data: {"candidates":[{"content":{"parts":[{"text":"A"}]}}]}',
        '',
        'data: [DONE]',
        '',
    ]
    client = _make_stream_chat_client(sse_lines)
    events = []
    async for ev in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        events.append(ev)
    full = "".join(ev["delta_text"] for ev in events)
    assert full == "A"


@pytest.mark.anyio
async def test_stream_chat_error_before_stream_raises():
    """스트림 시작 전 Vertex 4xx/5xx면 VertexAPIError를 raise해야 한다."""
    client = _make_stream_chat_client(
        sse_lines=[],
        status_code=429,
        error_body={"error": {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"}},
    )
    with pytest.raises(vertex.VertexAPIError) as excinfo:
        async for _ in client.stream_chat(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            pass
    assert excinfo.value.status_code == 429


# ===========================================================================
# Chat Completions — Streaming (SSE) endpoint
# ===========================================================================

class _FakeStreamingChatService:
    """stream_chat을 흉내내는 VertexChatClient 대역."""

    def __init__(self, *_a, **_k):
        self.last_call: dict = {}
        # generate() 비스트림도 지원 (회귀 안전)
        self._nonstream_result = {
            "text": "non-stream",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def generate(self, **kw):
        return self._nonstream_result

    async def stream_chat(self, *, model, messages, max_tokens=None, temperature=None, top_p=None, stop=None, response_format=None):
        self.last_call = {"model": model, "messages": messages, "response_format": response_format}
        deltas = [
            {"delta_text": "Hel", "finish_reason": None, "usage": None},
            {"delta_text": "lo!", "finish_reason": None, "usage": None},
            {"delta_text": "", "finish_reason": "stop",
             "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        ]
        for d in deltas:
            yield d

    async def close(self):
        pass


@pytest.fixture
def streaming_chat_app_client(monkeypatch):
    fake_chat = _FakeStreamingChatService()
    fake_embed = _FakeVertexService()
    monkeypatch.setattr(wrapper, "VertexEmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr(wrapper, "VertexChatClient", lambda: fake_chat)
    with TestClient(wrapper.app) as c:
        yield c, fake_chat


def _parse_sse(raw: str):
    """SSE 본문에서 data: 라인들의 payload(str)를 순서대로 추출."""
    out = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            out.append(line[len("data: "):])
    return out


def test_chat_completions_stream_returns_sse(streaming_chat_app_client):
    """stream=true이면 text/event-stream으로 OpenAI 청크 + [DONE]을 반환해야 한다."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    payloads = _parse_sse(r.text)
    # 마지막은 [DONE]
    assert payloads[-1] == "[DONE]"
    # 나머지는 JSON 청크
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["model"] == "gemini-2.5-flash" for c in chunks)
    assert all(c["id"].startswith("chatcmpl-") for c in chunks)


def test_chat_completions_stream_first_chunk_has_role(streaming_chat_app_client):
    """첫 청크의 delta에 role=assistant가 포함되어야 한다."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    payloads = _parse_sse(r.text)
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # 이후 청크엔 role 없이 content만 (content 델타가 있는 청크 기준)
    for c in chunks[1:]:
        assert "role" not in c["choices"][0]["delta"]


def test_chat_completions_stream_content_reconstructs(streaming_chat_app_client):
    """content 델타들을 이으면 전체 텍스트가 복원되어야 한다."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    payloads = _parse_sse(r.text)
    chunks = [json.loads(p) for p in payloads[:-1]]
    full = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert full == "Hello!"


def test_chat_completions_stream_last_chunk_has_finish_reason(streaming_chat_app_client):
    """마지막 청크(직전 [DONE])에 finish_reason 매핑값이 담겨야 한다."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    payloads = _parse_sse(r.text)
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_nonstream_still_works(streaming_chat_app_client):
    """stream=false(기본) 경로는 그대로 JSON ChatCompletion을 반환해야 한다 (회귀 안전)."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["object"] == "chat.completion"


def test_chat_completions_stream_unknown_model_404(streaming_chat_app_client):
    """stream=true라도 알 수 없는 모델은 스트림 시작 전 404를 반환해야 한다."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "nonexistent-xyz",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_chat_completions_stream_embedding_model_rejected(streaming_chat_app_client):
    """stream=true라도 임베딩 모델은 400으로 거부되어야 한다."""
    client, _ = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "text-embedding-005",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_stream_auth_enforced(streaming_chat_app_client, monkeypatch):
    """stream=true라도 WRAPPER_API_KEY 인증이 강제되어야 한다."""
    client, _ = streaming_chat_app_client
    monkeypatch.setattr(wrapper, "WRAPPER_API_KEY", "secret-key")
    bad = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert bad.status_code == 401


# ===========================================================================
# [Must-fix 2] 엔드포인트: stream_chat이 VertexAPIError를 raise해도
# 200 text/event-stream으로 data:{error} + data:[DONE]로 끝나야 한다.
# ===========================================================================

class _RealChatClientRaising4xx:
    """진짜 VertexChatClient를 쓰되 httpx.stream만 4xx로 mock한 래퍼.

    raise; yield 안티패턴 대신 실제 stream_chat 경로(연결 -> 4xx -> VertexAPIError)
    를 태워 엔드포인트의 에러 SSE 처리를 검증한다.
    """

    def __init__(self, status_code=429, message="rate limited", code="RESOURCE_EXHAUSTED"):
        self._inner = vertex.VertexChatClient(token_provider=_FakeStreamTokenProvider())
        err_body = {"error": {"message": message, "status": code}}

        class MockStreamResponse:
            def __init__(self):
                self.status_code = status_code

            async def aiter_lines(self):
                if False:
                    yield ""  # 도달하지 않음

            async def aread(self):
                return b""

            @property
            def text(self):
                return ""

            def json(self):
                return err_body

        class MockStreamCtx:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *exc):
                return False

        class MockHttp:
            def stream(self, method, url, *, headers=None, json=None):
                return MockStreamCtx()

            async def aclose(self):
                pass

        self._inner.http = MockHttp()

    async def generate(self, **kw):
        return await self._inner.generate(**kw)

    def stream_chat(self, **kw):
        return self._inner.stream_chat(**kw)

    async def close(self):
        await self._inner.close()


@pytest.fixture
def raising_stream_app_client(monkeypatch):
    fake_chat = _RealChatClientRaising4xx()
    fake_embed = _FakeVertexService()
    monkeypatch.setattr(wrapper, "VertexEmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr(wrapper, "VertexChatClient", lambda: fake_chat)
    with TestClient(wrapper.app) as c:
        yield c, fake_chat


def test_chat_completions_stream_error_is_sse_not_crash(raising_stream_app_client):
    """stream_chat이 VertexAPIError를 raise하면 200 SSE로 error 청크 + [DONE]을 내보내야 한다."""
    client, _ = raising_stream_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    # 서버 크래시/깨진 SSE가 아니라 정상 200 event-stream
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    payloads = _parse_sse(r.text)
    # 마지막은 [DONE]
    assert payloads[-1] == "[DONE]"
    # [DONE] 직전은 error payload
    err_payload = json.loads(payloads[-2])
    assert "error" in err_payload
    assert err_payload["error"]["message"] == "rate limited"
    assert err_payload["error"]["type"] == "rate_limit_error"


# ===========================================================================
# [Minor] chatcmpl id는 매 요청 고유(uuid 기반)여야 한다.
# ===========================================================================

def test_chat_completions_nonstream_id_is_unique(chat_app_client):
    """비스트림 chat completion id가 요청마다 고유해야 한다 (chatcmpl-vertex 상수 아님)."""
    client, _ = chat_app_client
    r1 = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    r2 = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]
    assert id1.startswith("chatcmpl-")
    assert id2.startswith("chatcmpl-")
    assert id1 != "chatcmpl-vertex"
    assert id1 != id2


def test_chat_completions_stream_id_is_unique(streaming_chat_app_client):
    """스트림 chat completion id가 요청마다 고유하고 모든 청크에서 동일해야 한다."""
    client, _ = streaming_chat_app_client
    r1 = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    r2 = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    chunks1 = [json.loads(p) for p in _parse_sse(r1.text)[:-1]]
    chunks2 = [json.loads(p) for p in _parse_sse(r2.text)[:-1]]
    ids1 = {c["id"] for c in chunks1}
    ids2 = {c["id"] for c in chunks2}
    # 한 응답 내 모든 청크는 동일 id
    assert len(ids1) == 1
    assert len(ids2) == 1
    # 두 요청은 서로 다른 id
    assert ids1 != ids2
    assert "chatcmpl-vertex" not in ids1


# ===========================================================================
# [Minor] mid-stream 깨진 JSON 줄은 스킵하고 정상 델타는 계속 처리.
# ===========================================================================

@pytest.mark.anyio
async def test_stream_chat_skips_broken_json_line_midstream():
    """스트림 중간에 깨진 JSON 줄이 와도 스킵하고 나머지를 정상 처리해야 한다."""
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"A"}]}}]}',
        '',
        'data: {this is not valid json',  # 깨진 줄
        '',
        'data: {"candidates":[{"content":{"parts":[{"text":"B"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}',
        '',
    ]
    client = _make_stream_chat_client(sse_lines)
    events = []
    async for ev in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        events.append(ev)
    full = "".join(ev["delta_text"] for ev in events)
    assert full == "AB"
    assert events[-1]["finish_reason"] == "stop"


# ===========================================================================
# [Must-fix 1] 동시성: stream_chat은 연결+헤더 수신까지만 세마포어를 잡고
# aiter_lines 루프 진입 전에 해제해야 한다. 또한 stream_ctx는 모든 경로에서
# 반드시 닫혀야(__aexit__ 호출) 한다.
# ===========================================================================

def _make_tracking_stream_client(sse_lines, status_code=200, error_body=None,
                                 semaphore_holders=None):
    """세마포어 점유 추적 + __aexit__ 호출 여부를 추적하는 stream client.

    semaphore_holders: aiter_lines 각 줄을 yield하기 직전의 세마포어 _value를
    기록할 리스트. 세마포어가 루프 진입 전에 해제됐다면 값이 회복돼 있어야 한다.
    """
    state = {"aexit_called": False, "aenter_called": False}

    class MockStreamResponse:
        def __init__(self, sem):
            self.status_code = status_code
            self._sem = sem

        async def aiter_lines(self):
            for line in sse_lines:
                if semaphore_holders is not None and self._sem is not None:
                    semaphore_holders.append(self._sem._value)
                yield line

        async def aread(self):
            return b""

        @property
        def text(self):
            return error_body if isinstance(error_body, str) else ""

        def json(self):
            if isinstance(error_body, dict):
                return error_body
            return {}

    class MockStreamCtx:
        def __init__(self, sem):
            self._sem = sem
            self._resp = MockStreamResponse(sem)

        async def __aenter__(self):
            state["aenter_called"] = True
            return self._resp

        async def __aexit__(self, *exc):
            state["aexit_called"] = True
            return False

    client = vertex.VertexChatClient(token_provider=_FakeStreamTokenProvider())
    sem = client.semaphore

    class MockHttp:
        def stream(self, method, url, *, headers=None, json=None):
            return MockStreamCtx(sem)

        async def aclose(self):
            pass

    client.http = MockHttp()
    return client, state


@pytest.mark.anyio
async def test_stream_chat_releases_semaphore_before_iteration():
    """aiter_lines 루프를 도는 동안 세마포어가 해제되어 있어야 한다 (동시성 굶음 방지)."""
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"A"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"B"}]},"finishReason":"STOP"}],"usageMetadata":{}}',
    ]
    holders = []
    client, _state = _make_tracking_stream_client(sse_lines, semaphore_holders=holders)
    initial = client.semaphore._value
    async for _ in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        pass
    # 루프를 도는 내내 세마포어 _value가 initial(완전 해제 상태)로 회복돼 있어야 한다.
    assert holders, "aiter_lines가 호출되지 않음"
    assert all(v == initial for v in holders), (
        f"세마포어가 스트림 루프 동안 점유됨: holders={holders}, initial={initial}"
    )


@pytest.mark.anyio
async def test_stream_chat_closes_context_on_normal_completion():
    """정상 종료 시 stream_ctx.__aexit__이 호출되어야 한다."""
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"A"}]},"finishReason":"STOP"}],"usageMetadata":{}}',
    ]
    client, state = _make_tracking_stream_client(sse_lines)
    async for _ in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        pass
    assert state["aexit_called"] is True


@pytest.mark.anyio
async def test_stream_chat_closes_context_on_client_disconnect():
    """소비자가 중도에 끊어도(GeneratorExit) stream_ctx.__aexit__이 호출되어야 한다."""
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"A"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"B"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"C"}]},"finishReason":"STOP"}],"usageMetadata":{}}',
    ]
    client, state = _make_tracking_stream_client(sse_lines)
    gen = client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    # 첫 이벤트만 받고 중단
    await gen.__anext__()
    await gen.aclose()  # GeneratorExit 유발
    assert state["aexit_called"] is True
    # 세마포어가 누수 없이 회복됐는지
    assert client.semaphore._value == 8


@pytest.mark.anyio
async def test_stream_chat_error_before_stream_still_closes_context():
    """4xx 응답으로 VertexAPIError를 raise해도 stream_ctx가 닫혀야 한다."""
    client, state = _make_tracking_stream_client(
        sse_lines=[],
        status_code=500,
        error_body={"error": {"message": "boom", "status": "INTERNAL"}},
    )
    with pytest.raises(vertex.VertexAPIError):
        async for _ in client.stream_chat(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            pass
    assert state["aexit_called"] is True
    assert client.semaphore._value == 8


# ---- gemini-3.5-flash (global generateContent) ----

def test_gemini_35_flash_registered_as_global_chat():
    cfg = vertex.model_config("gemini-3.5-flash")
    assert cfg is not None
    assert cfg["api"] == "generateContent"
    assert cfg["kind"] == "chat"
    assert cfg["location"] == "global"


def test_chat_generate_url_global_has_no_region_prefix(mock_token_provider):
    client = vertex.VertexChatClient()
    url = client._generate_content_url("gemini-3.5-flash", "global")
    assert url.startswith("https://aiplatform.googleapis.com/"), url
    assert "/locations/global/" in url
    assert "gemini-3.5-flash:generateContent" in url


def test_chat_stream_url_global_has_no_region_prefix(mock_token_provider):
    client = vertex.VertexChatClient()
    url = client._stream_generate_content_url("gemini-3.5-flash", "global")
    assert url.startswith("https://aiplatform.googleapis.com/"), url
    assert ":streamGenerateContent?alt=sse" in url


def test_chat_generate_url_regional_keeps_prefix(mock_token_provider):
    client = vertex.VertexChatClient()
    url = client._generate_content_url("gemini-2.5-flash", "us-central1")
    assert url.startswith("https://us-central1-aiplatform.googleapis.com/"), url


# ---- thinking budget (thinking 모델이 작은 max_tokens에 본문 비는 것 방지) ----

def test_chat_body_includes_thinking_budget_when_set():
    body = vertex.VertexChatClient._build_request_body(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=20, temperature=None, top_p=None, stop=None, thinking_budget=0,
    )
    assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


def test_chat_body_omits_thinking_config_when_none():
    body = vertex.VertexChatClient._build_request_body(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=20, temperature=None, top_p=None, stop=None, thinking_budget=None,
    )
    assert "thinkingConfig" not in body.get("generationConfig", {})


def test_gemini_35_flash_has_thinking_budget_zero():
    assert vertex.model_config("gemini-3.5-flash").get("thinking_budget") == 0


def test_gemini_25_pro_has_no_thinking_budget():
    assert vertex.model_config("gemini-2.5-pro").get("thinking_budget") is None


# ===========================================================================
# response_format — vertex.py 단위 테스트 (_build_request_body)
# ===========================================================================

@pytest.mark.anyio
async def test_response_format_json_object_sets_mime_type(chat_client):
    """response_format={"type":"json_object"}이면 generationConfig.responseMimeType이 application/json이어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(
        model="gemini-2.5-flash",
        messages=messages,
        response_format={"type": "json_object"},
    )
    body = mock_http.last_request["json"]
    gen_cfg = body.get("generationConfig", {})
    assert gen_cfg.get("responseMimeType") == "application/json"
    assert "responseJsonSchema" not in gen_cfg


@pytest.mark.anyio
async def test_response_format_json_schema_sets_mime_type_and_schema(chat_client):
    """response_format={"type":"json_schema",...}이면 responseMimeType + responseJsonSchema가 설정되어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        "required": ["color"],
    }
    await client.generate(
        model="gemini-2.5-flash",
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {"name": "MySchema", "schema": schema}},
    )
    body = mock_http.last_request["json"]
    gen_cfg = body.get("generationConfig", {})
    assert gen_cfg.get("responseMimeType") == "application/json"
    actual_schema = gen_cfg.get("responseJsonSchema")
    assert actual_schema is not None
    # enum은 그대로 보존되어야 한다
    assert actual_schema["properties"]["color"]["enum"] == ["red", "green", "blue"]


@pytest.mark.anyio
async def test_response_format_does_not_set_response_schema_old_field(chat_client):
    """responseSchema (구 필드)는 절대 body에 포함되지 않아야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    for rf in [
        {"type": "json_object"},
        {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    ]:
        await client.generate(
            model="gemini-2.5-flash",
            messages=messages,
            response_format=rf,
        )
        body = mock_http.last_request["json"]
        gen_cfg = body.get("generationConfig", {})
        assert "responseSchema" not in gen_cfg, f"responseSchema found for {rf}"


@pytest.mark.anyio
async def test_response_format_absent_no_mime_type(chat_client):
    """response_format 미제공 시 responseMimeType이 없고, generationConfig도 없어야 한다 (회귀)."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(model="gemini-2.5-pro", messages=messages)
    body = mock_http.last_request["json"]
    # generationConfig 자체가 없어야 한다 (기존 test_chat_no_generation_config_when_all_omitted 보완)
    assert "generationConfig" not in body
    # 혹시라도 있다면 responseMimeType 없어야 한다
    assert "responseMimeType" not in body.get("generationConfig", {})


@pytest.mark.anyio
async def test_response_format_text_treated_as_no_structured_output(chat_client):
    """response_format={"type":"text"}이면 구조적 출력 설정이 없어야 한다."""
    client, mock_http = chat_client
    messages = [{"role": "user", "content": "Hi"}]
    await client.generate(
        model="gemini-2.5-pro",
        messages=messages,
        response_format={"type": "text"},
    )
    body = mock_http.last_request["json"]
    # generationConfig 자체가 없어야 한다 (다른 gen 파라미터 없을 때)
    assert "generationConfig" not in body


# ===========================================================================
# response_format — _sanitize_schema 단위 테스트
# ===========================================================================

def test_sanitize_schema_strips_dollar_schema_but_keeps_enum():
    """_sanitize_schema는 $schema를 제거하고 enum/type/properties/required는 유지해야 한다."""
    from vertex import _sanitize_schema
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "my-schema",
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "fail"]},
        },
        "required": ["status"],
        "$defs": {"helper": {"type": "string"}},
    }
    result = _sanitize_schema(schema)
    # 제거되어야 하는 키
    assert "$schema" not in result
    assert "$id" not in result
    assert "$defs" not in result
    # 보존되어야 하는 키
    assert result["type"] == "object"
    assert result["required"] == ["status"]
    assert result["properties"]["status"]["enum"] == ["ok", "fail"]


def test_sanitize_schema_does_not_mutate_original():
    """_sanitize_schema는 원본 dict를 변경하지 않아야 한다."""
    from vertex import _sanitize_schema
    original = {"$schema": "http://...", "type": "string"}
    _ = _sanitize_schema(original)
    assert "$schema" in original  # 원본 불변


# ===========================================================================
# response_format — 엔드포인트 테스트
# ===========================================================================

def test_chat_completions_unsupported_response_format_returns_400(chat_app_client):
    """지원하지 않는 response_format.type은 HTTP 400 invalid_request_error를 반환해야 한다."""
    client, _ = chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "xml"},
    })
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"


def test_chat_completions_json_schema_flows_to_fake_service(chat_app_client):
    """json_schema response_format이 엔드포인트를 통해 fake service에 전달되어야 한다."""
    client, fake_chat = chat_app_client
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
    })
    assert r.status_code == 200
    # fake service의 last_call에 response_format이 기록되어야 한다
    assert fake_chat.last_call.get("response_format") is not None
    assert fake_chat.last_call["response_format"]["type"] == "json_schema"


# ===========================================================================
# response_format — 스트리밍 경로 전달 테스트
# ===========================================================================

def test_chat_completions_stream_response_format_forwarded(streaming_chat_app_client):
    """streaming 경로에서도 response_format이 fake stream_chat에 전달되어야 한다."""
    client, fake_chat = streaming_chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "response_format": {"type": "json_object"},
    })
    assert r.status_code == 200
    # fake stream_chat last_call에 response_format이 기록되어야 한다
    assert fake_chat.last_call.get("response_format") is not None
    assert fake_chat.last_call["response_format"]["type"] == "json_object"


@pytest.mark.anyio
async def test_stream_chat_json_schema_reaches_vertex_body():
    """stream_chat 실제 Vertex body에 responseJsonSchema가 enum 보존된 채 들어가야 한다."""
    capture = {}
    sse_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{}}',
        '',
    ]
    client = _make_stream_chat_client(sse_lines, capture=capture)
    schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        "required": ["color"],
    }
    # async generator라 한 번 소비해야 요청이 나간다.
    async for _ in client.stream_chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    ):
        pass
    body = capture["json"]
    gen_cfg = body.get("generationConfig", {})
    assert gen_cfg.get("responseMimeType") == "application/json"
    actual_schema = gen_cfg.get("responseJsonSchema")
    assert actual_schema is not None
    assert actual_schema["properties"]["color"]["enum"] == ["red", "green", "blue"]
