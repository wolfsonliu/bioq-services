# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .
#
# RFantibody source must be at opensource/RFantibody/ (weights baked into image).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# --- Install RFantibody ---
WORKDIR /opt/rfantibody

# Dependencies first (cache-friendly layer order)
COPY opensource/RFantibody/pyproject.toml opensource/RFantibody/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Full RFantibody source + weights
COPY opensource/RFantibody/ .
RUN uv sync --frozen --no-dev

# Server dependencies (into the same venv)
RUN uv pip install fastapi uvicorn python-multipart

# --- Copy server code as "server" package ---
COPY services/rfantibody-server/ /opt/rfantibody/server/

ENV RFANTIBODY_ROOT=/opt/rfantibody
ENV RFANTIBODY_WEIGHTS=/opt/rfantibody/weights
ENV RFANTIBODY_SCRIPTS=/opt/rfantibody/scripts

# --- Alibaba Cloud FC configuration ---
# CAPort: FC default listening port is 9000
# Keep-alive: FC requires >= 15 minutes (900s)
# Startup: HTTP server must be ready within 120s
ENV PORT=9000
EXPOSE 9000

CMD ["uv", "run", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
