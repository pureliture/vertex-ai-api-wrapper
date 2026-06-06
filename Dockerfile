FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv

# 의존성 먼저 설치 (레이어 캐시)
COPY pyproject.toml /app/pyproject.toml
RUN uv sync --no-dev

COPY app.py vertex.py /app/

EXPOSE 8930

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8930"]
