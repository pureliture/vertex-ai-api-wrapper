<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=vertex-ai-api-wrapper&fontSize=40" width="100%"/>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAGFlow-FF4F00?style=for-the-badge&logo=ragflow&logoColor=white"/>
</div>

<br/>
<div align="center">
  <b>Google Cloud Vertex AI 임베딩 및 Rerank API를 RAGFlow(OpenAI/LocalAI 규격)에서 사용할 수 있게 해주는 프록시 서버입니다.</b>
</div>
<br/>

<div align="center">
  <a href="#-시스템-아키텍처">🏛️ 시스템 아키텍처</a> &nbsp;|&nbsp;
  <a href="#-빠른-시작">🚀 빠른 시작</a> &nbsp;|&nbsp;
  <a href="#-환경-변수-설정">⚙️ 환경 변수 설정</a> &nbsp;|&nbsp;
  <a href="#-ragflow-연동-가이드">🎯 RAGFlow 연동</a> &nbsp;|&nbsp;
  <a href="#-api-참조">📡 API 참조</a>
</div>

---


<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 🏛️ 시스템 아키텍처

<!-- docs/images/architecture.svg 위치에 다이어그램 삽입 -->

### 🎨 핵심 설계 포인트

<table width="100%">
  <tr>
    <td width="50%">
      <h3>🟦 Drop-in Replacement</h3>
      <p>기존 OpenAI/LocalAI 생태계 코드 변경 없이 Vertex AI를 그대로 사용 가능합니다.</p>
    </td>
    <td width="50%">
      <h3>🟩 Native Reranking</h3>
      <p>RAGFlow의 <code>LocalAI</code> provider 규격을 통해 Vertex AI Search Ranking API를 완벽하게 연결합니다.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🟪 Auto-Batching</h3>
      <p>RAGFlow의 고정 배치(16개)를 Vertex AI의 모델별 한도(1~5개)에 맞춰 자동 분할 및 병렬 처리합니다.</p>
    </td>
    <td width="50%">
      <h3>🟧 Auth Abstraction</h3>
      <p>리프레시 토큰 관리 없이 ADC(Application Default Credentials) 서비스 계정을 통해 자동으로 OAuth2 토큰을 발급받습니다.</p>
    </td>
  </tr>
</table>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## ⚙️ 환경 변수 설정 (Configuration)

`.env` 파일을 생성하거나 컨테이너 환경 변수로 다음 값을 주입합니다.

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | `""` | GCP 서비스 계정 JSON 키 경로 (필요 권한: `roles/aiplatform.user`). |
| `VERTEX_PROJECT` | *(Required)* | GCP 프로젝트 ID. |
| `VERTEX_LOCATION` | `us-central1` | Vertex API를 호출할 GCP 리전. |
| `WRAPPER_API_KEY` | `""` | 래퍼 서버를 보호하기 위한 선택적 API 키 (Bearer Token). |
| `VERTEX_TASK_TYPE_DEFAULT` | `RETRIEVAL_DOCUMENT` | 텍스트 임베딩을 위한 기본 Task Type. |
| `VERTEX_AUTO_TRUNCATE` | `true` | 토큰 제한 초과 시 400 에러 대신 자동으로 입력 텍스트를 자를지 여부. |
| `MAX_CONCURRENCY` | `8` | Vertex API에 대한 최대 동시 HTTP 요청 수. |
| `HTTP_TIMEOUT_SECONDS` | `60` | Vertex API HTTP 요청 타임아웃. |
| `TOKEN_REFRESH_SKEW_SECONDS` | `300` | Google OAuth 토큰 만료 전 사전 갱신 시간 (초). |
| `EXTRA_MODELS` | `""` | 콤마(,)로 구분된 추가 지원 모델 목록. |
| `MODEL_REGISTRY_JSON` | `""` | 복잡한 모델 라우팅을 위한 JSON 설정. |
| `DEFAULT_MAX_INSTANCES` | `1` | 알 수 없는 모델에 대한 병렬 호출 시 기본 청크 크기. |

<details>
<summary><b>💡 복잡한 모델 라우팅 추가 방법</b></summary>
<p>새로운 모델은 <code>EXTRA_MODELS</code> 환경 변수에 콤마로 구분하여 추가하거나, <code>MODEL_REGISTRY_JSON</code>을 통해 구체적인 API 타입, 리전, max_instances 등을 제어할 수 있습니다.</p>
</details>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 🎯 RAGFlow 연동 가이드

래퍼 서버를 RAGFlow에 연결하기 위한 필수 설정입니다.

### 1. Rerank 연동 (**LocalAI Provider** 사용)

Vertex AI의 검색 랭킹 모델을 사용하려면, 반드시 **LocalAI** 프로바이더로 등록해야 합니다.

| 항목 | 설정 값 | 비고 |
|---|---|---|
| Model Type | `Rerank` | 반드시 **Rerank** 선택 |
| Model Name | `semantic-ranker-512` 등 | Vertex API의 Rank 모델명 |
| Base URL | `http://wrapper-vertex-ai-api` | ⚠️ **끝에 슬래시(`/`) 금지** |

> **💡 Base URL 함정 주의**: RAGFlow는 내부적으로 `urljoin(base_url, "v1")`을 수행합니다. Base URL 끝에 슬래시가 있으면 `/v1/v1/rerank`로 잘못 호출되어 에러가 발생하므로 절대 포함하지 마세요.

### 2. Embedding 연동 (**OpenAI-API-Compatible** 사용)

임베딩 모델은 **OpenAI-API-Compatible** 프로바이더로 등록합니다.

| 항목 | 설정 값 | 비고 |
|---|---|---|
| Model Type | `Embedding` | |
| Model Name | `text-embedding-004` 등 | |
| Base URL | `http://wrapper-vertex-ai-api` | ⚠️ **끝에 슬래시(`/`) 금지** |

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 🚀 빠른 시작

### ⚡ 요구사항
- uv
- Docker 및 Docker Compose
- GCP 서비스 계정 JSON 키 파일 (`roles/aiplatform.user`)

### 🧪 로컬 환경 (uv 사용)

```bash
# 의존성 설치 및 백엔드 서버 실행
uv run uvicorn app:app --reload --port 8000
```

### 🐳 Docker Compose 환경

```bash
# 백그라운드 컨테이너 빌드 및 실행
docker compose up -d --build
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 📡 API 참조

이 래퍼는 아래와 같은 호환 엔드포인트를 제공합니다.

| Method | Endpoint | 호환 규격 | 반환 형식 |
|---|---|---|---|
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square"/> | `/v1/embeddings` | OpenAI 호환 | `encoding_format` 지원 |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square"/> | `/v1/chat/completions` | OpenAI 호환 | SSE Stream 지원 |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square"/> | `/v1/rerank` | Cohere / LocalAI 호환 | `results[{index, relevance_score}]` |

<br/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer"/>
</div>

<div align="center">
  <a href="#-시스템-아키텍처">🏛️ 아키텍처</a> &nbsp;|&nbsp;
  <a href="#-빠른-시작">🚀 빠른 시작</a> &nbsp;|&nbsp;
  <a href="#top">⬆️ 맨 위로</a>
</div>
