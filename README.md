# vertex-ai-api-wrapper

Google Cloud **Vertex AI (Gemini Enterprise Agent Platform)** 임베딩을 **OpenAI 호환 `/v1/embeddings`** 로 노출하는 얇은 프록시.
RAGFlow 0.25.6의 **"OpenAI-API-Compatible"** 임베딩 provider에서 Vertex 임베딩을 쓰기 위해 만든다.

RAGFlow 0.25.6에는 Vertex 임베딩 네이티브 메뉴가 없고(“Google Cloud” provider는 채팅 전용, “Gemini”는 AI Studio API 키 전용),
AI Studio를 쓸 수 없는 환경에서는 이 래퍼가 유일한 경로다.

## 동작

```
RAGFlow ──POST /v1/embeddings (OpenAI)──▶ 래퍼 ──:predict + Bearer token──▶ Vertex AI
        ◀──{data:[{embedding}]}─────────       ◀──{predictions:[{embeddings}]}──
```

- 인증: 서비스 계정(ADC)에서 1시간짜리 OAuth2 access token을 `google-auth`가 자동 발급·캐시·갱신. **리프레시 토큰을 직접 다루지 않는다.**
- 배치 분할: RAGFlow는 요청당 **16개** 텍스트를 보낸다. Vertex 요청당 instance 한도에 맞춰 자동으로 쪼개 병렬 호출한다.

| 모델 | API | 요청당 한도 | 기본 차원 |
|---|---|---|---|
| `gemini-embedding-2` | embedContent(global) | **1개** | 3072 (축소 가능, 멀티모달 최신) |
| `gemini-embedding-001` | predict | **1개** | 3072 (축소 가능) |
| `text-embedding-005` | predict | **5개** | 768 |
| `text-multilingual-embedding-002` | predict | **5개** | 768 |

> 한도 출처: Google Vertex AI 공식문서 + 실 API 호출 확인 (2026-06).

### 새 모델 추가 = 코드 수정 없이 설정만
모델 라우팅은 `vertex.py`의 config-driven 레지스트리가 담당한다. 새 모델은 env로 추가:
- `EXTRA_MODELS=text-embedding-006` — predict 계열 모델 한 줄 추가
- `MODEL_REGISTRY_JSON='{"새모델":{"api":"embedContent","location":"global","max_instances":1}}'` — 다른 API 계열까지 전체 지정
Vertex 임베딩 API 계열은 `predict`/`embedContent` 둘뿐이라, 두 어댑터로 모든 모델을 설정만으로 커버한다.

## 실행 (로컬, uv)

```bash
cp .env.example .env        # 값 채우기 (VERTEX_PROJECT 등)
export GOOGLE_APPLICATION_CREDENTIALS=/path/vertex-sa.json
export VERTEX_PROJECT=your-gcp-project
export VERTEX_LOCATION=us-central1

uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8930
```

동작 확인:

```bash
curl -s http://localhost:8930/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-005","input":["hello","world"]}' | head
```

## 실행 (Docker)

```bash
docker build -t vertex-embed-wrapper .
docker run -p 8930:8930 \
  -e VERTEX_PROJECT=your-gcp-project \
  -e VERTEX_LOCATION=us-central1 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/vertex-sa.json \
  -v /path/vertex-sa.json:/app/secrets/vertex-sa.json:ro \
  vertex-embed-wrapper
```

## RAGFlow 등록 (Settings → Model providers → OpenAI-API-Compatible)

| 항목 | 값 |
|---|---|
| Model type | `Embedding` |
| Model name | `gemini-embedding-001` (또는 `text-embedding-005`) — Vertex 공식 모델명 그대로 |
| Base url | `http://<래퍼-호스트>:8930` ⚠️ **끝 슬래시 금지** |
| API-Key | `.env`의 `WRAPPER_API_KEY` 값 (비웠으면 빈칸) |
| Max tokens | `2048` |

### ⚠️ Base url 함정
RAGFlow 0.25.6은 입력 주소 끝에 `v1`을 자동으로 붙인다(`urljoin(base_url, "v1")`).

| 입력 | 실제 호출 | |
|---|---|---|
| `http://wrapper:8930` | `/v1/embeddings` | ✅ |
| `http://wrapper:8930/v1` | `/v1/embeddings` | ✅ |
| `http://wrapper:8930/v1/` | `/v1/v1/embeddings` | ❌ 끝 슬래시 금지 |

같은 docker-compose 망이면 컨테이너 이름으로 `http://vertex-embed-wrapper:8930`.
래퍼가 호스트에서 단독 실행이고 RAGFlow가 컨테이너면 `http://host.docker.internal:8930`.

## 범용 OpenAI 호환 (RAGFlow 외 다른 클라이언트)

이 래퍼는 OpenAI 임베딩 표준을 구현하므로 RAGFlow뿐 아니라 OpenAI 호환 임베딩 클라이언트
전반(LangChain, LlamaIndex, Dify, Open WebUI, LiteLLM, 직접 `openai` SDK)에 그대로 붙는다.

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8930/v1", api_key="<WRAPPER_API_KEY 또는 아무값>")
# encoding_format 미지정 -> openai SDK 기본 base64 전송 -> 래퍼가 처리 (drop-in)
r = client.embeddings.create(model="text-embedding-005", input=["hello", "world"])
print(len(r.data), len(r.data[0].embedding))
```

지원 항목:
- `POST /v1/embeddings` — `encoding_format`: `float`(list) / `base64`(float32 LE, SDK 기본) 둘 다.
- `GET /v1/models`, `GET /v1/models/{id}` — 모델 목록/조회(일부 클라이언트가 프로브).
- 에러는 OpenAI 포맷(`{"error":{"message","type","code","param"}}`). 요청 검증 실패도 422가 아닌 OpenAI 400으로 변환.
- CORS 허용(브라우저 클라이언트 OPTIONS preflight), trailing-slash 리다이렉트 비활성.
- `model`은 레지스트리(`EXTRA_MODELS`/`MODEL_REGISTRY_JSON`로 확장)에 등록된 것만 허용 — 미등록 모델은 404(경로 주입 방지).

## 옵션 (요청 헤더)

- `X-Vertex-Task-Type`: 호출별 task_type 덮어쓰기 (기본 `RETRIEVAL_DOCUMENT`).
  - 쿼리 임베딩 품질을 따로 올리려면 RAGFlow 쿼리 경로에 `RETRIEVAL_QUERY`를 보내야 하나,
    순수 OpenAI 인터페이스엔 문서/쿼리 구분 필드가 없다(현재 한계).
- `X-Vertex-Title`: 문서 임베딩 title.

## 주의

- **차원 고정**: RAGFlow는 지식베이스 첫 임베딩의 차원으로 벡터DB 스키마를 고정한다. KB 생성 후 `dimensions`를 바꾸면 충돌 → 인덱스 재생성 필요.
- **할당량(429)**: 대량 색인 시 Vertex 리전 quota에 걸릴 수 있다. 필요 시 백오프 재시도/quota 증설.
- **IAM 최소 권한**: 서비스 계정에 `roles/aiplatform.user`만 부여.

## 설계 근거

`Docs/` 의 딥리서치 2건 + RAGFlow v0.25.6 소스(`rag/llm/embedding_model.py`) + Google 공식문서 교차검증 결과를 반영했다.
