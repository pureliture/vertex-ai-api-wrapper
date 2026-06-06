# OpenAI 임베딩 drop-in 확장 — 구현 기록

vertex-ai-api-wrapper를 RAGFlow 전용에서 **범용 OpenAI 임베딩 호환**으로 확장한 작업 기록.
계획 → antigravity 멀티에이전트 리뷰 → 정정 → 구현 → 검증 순으로 진행했다.

## 적용한 변경 (app.py)

1. **`encoding_format="base64"` 지원** — `Literal["float","base64"]="float"` 필드.
   openai SDK가 기본 base64를 명시 전송하므로 Literal로 받으면 충분. (계획의 "raw body 판별"은
   과설계라 폐기 — Pydantic은 전달된 값을 기본값으로 덮어쓰지 않음.) base64는 float32 little-endian
   `struct.pack` 후 base64 인코딩(`encode_embedding`).
2. **`GET /v1/models`, `GET /v1/models/{id}`** — 일부 클라이언트가 모델 목록을 프로브함.
3. **422 → OpenAI 에러 변환** — `RequestValidationError` 핸들러로 FastAPI 기본 422
   `{"detail":...}`를 `{"error":{...}}` 포맷으로 변환(openai SDK 파싱 호환).
4. **CORS 허용** — `CORSMiddleware`로 브라우저 클라이언트(Open WebUI) OPTIONS preflight 통과.
5. **trailing-slash 리다이렉트 비활성** — `redirect_slashes=False`로 `/v1/embeddings/` 307 시
   POST 본문 소실 방지(미등록 경로는 404로 명확히 실패).
6. **model 화이트리스트** — `ALLOWED_MODELS = KNOWN + EXTRA_MODELS`. model이 Vertex URL 경로에
   직접 들어가므로 미등록 모델은 404(path 주입/SSRF 방지).
7. **전역 동시성 상한** — semaphore를 요청당이 아닌 app.state 공유로 변경(다중 요청 fan-out 시
   Vertex 동시 호출을 `MAX_CONCURRENCY`로 제한, 429 폭주 완화).

## antigravity 리뷰 반영 결과

- 채택(계획에 없던 것): 422 핸들러, CORS, trailing-slash, model 화이트리스트, 전역 semaphore.
- 정정: `encoding_format` raw-body 판별 → Literal 필드로 단순화.
- 오탐(이미 app.py에 있던 것): `usage`/`object`/`index`/에러 래핑/세마포어/429 매핑.
  (리뷰어가 app.py 본문을 받지 못해 계획서만 보고 누락으로 추정한 항목.)
- 미적용(과범위): 429 지수백오프(tenacity), 로그 마스킹 — 현재 input/토큰을 로깅하지 않아 시급도 낮음.
  추후 운영에서 대량 색인 시 백오프 추가 검토.

## 검증

- 단위테스트 20개 통과(분할/순서/usage/base64 라운드트립/모델 화이트리스트/`/v1/models`/검증에러/auth).
- 실 Vertex 왕복: `text-embedding-005` 768d, `gemini-embedding-001` 3072d.
- **공식 `openai` SDK 기본 호출**(encoding_format 미지정 → base64)이 실 Vertex까지 관통,
  SDK가 base64를 float로 디코드 성공 → drop-in 실증.
- `GET /v1/models` 3개 반환, `/v1/embeddings/`(trailing slash) 404, `model="../evil"` 404.
