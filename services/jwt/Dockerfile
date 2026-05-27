# Build from bioagent project root:
#   docker build --platform linux/amd64 -t jwt-server -f services/jwt/Dockerfile .
#
# Lightweight JWT token issuance service (~120 MB).
# No GPU, no heavy deps — just FastAPI + PyJWT + cryptography.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

WORKDIR /opt/jwt
RUN uv venv .venv --python python3.11

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        fastapi "uvicorn[standard]" "PyJWT[crypto]"

COPY services/jwt/app.py    /opt/jwt/app.py
COPY services/jwt/jwks.json /opt/jwt/jwks.json
COPY services/jwt/private_keys/ /opt/jwt/private_keys/

ENV PORT=9000
EXPOSE 9000

CMD [".venv/bin/python", "-m", "uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000"]
