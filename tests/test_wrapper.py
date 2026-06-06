"""래퍼의 변환/라우팅 로직 단위 테스트 (실제 Vertex/GCP 호출 없이).

lifespan을 실제로 돌리되 GoogleAccessTokenProvider/VertexEmbeddingClient를 가짜로
monkeypatch해서, app.state(semaphore 포함)가 TestClient의 이벤트 루프에서 정상 구성되게 한다.
"""

from __future__ import annotations

import base64
import struct

import pytest
from fastapi.testclient import TestClient

import app as wrapper


# ---- 순수 함수 ----

def test_coerce_string_inputs_str():
    assert wrapper.coerce_string_inputs("hi") == ["hi"]


def test_coerce_string_inputs_list():
    assert wrapper.coerce_string_inputs(["a", "b"]) == ["a", "b"]


def test_coerce_string_inputs_rejects_tokens():
    with pytest.raises(ValueError):
        wrapper.coerce_string_inputs([[1, 2, 3]])


def test_chunked_splits_by_size():
    assert list(wrapper.chunked(["a", "b", "c", "d", "e"], 2)) == [["a", "b"], ["c", "d"], ["e"]]


def test_chunked_size_floor_is_one():
    assert list(wrapper.chunked(["a", "b"], 0)) == [["a"], ["b"]]


def test_gemini_001_max_instances_is_one():
    assert wrapper.KNOWN_MAX_INSTANCES["gemini-embedding-001"] == 1


def test_text_005_max_instances_is_five():
    assert wrapper.KNOWN_MAX_INSTANCES["text-embedding-005"] == 5


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


# ---- 엔드포인트 (Vertex 호출은 가짜로 대체) ----

class _FakeVertex:
    """predict()를 흉내. instance(text)당 prediction 하나씩 반환."""

    def __init__(self, *_a, **_k):
        self.calls: list[list[str]] = []

    async def predict(self, *, model, texts, dimensions, task_type, title, auto_truncate):
        self.calls.append(list(texts))
        return [
            {"embeddings": {"values": [0.1, 0.2, 0.3], "statistics": {"token_count": 2}}}
            for _ in texts
        ]

    async def close(self):
        pass


class _FakeTokenProvider:
    project_id = "test-project"

    def __init__(self, *_a, **_k):
        pass


@pytest.fixture
def client_with_fake(monkeypatch):
    fake = _FakeVertex()
    # 실제 lifespan이 돌되 GCP 의존성만 가짜로 교체 -> semaphore 등 app.state가 in-loop 구성됨.
    monkeypatch.setattr(wrapper, "GoogleAccessTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(wrapper, "VertexEmbeddingClient", lambda _tp: fake)
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
    assert {"gemini-embedding-001", "text-embedding-005"} <= ids


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
