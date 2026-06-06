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
    await client.generate(model="gemini-2.5-flash", messages=messages)
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

    async def generate(self, *, model, messages, max_tokens=None, temperature=None, top_p=None, stop=None):
        self.last_call = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        return {
            "text": "Hello, I am Gemini!",
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


def test_chat_completions_stream_true_returns_error(chat_app_client):
    """stream=true이면 400 streaming_not_supported 에러를 반환해야 한다."""
    client, _ = chat_app_client
    r = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "streaming_not_supported"
    assert error["type"] == "invalid_request_error"


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
