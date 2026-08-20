# Build from bioq-services repo root:
#   docker build --platform linux/amd64 -t seqkit-server -f services/seqkit-server/Dockerfile .
#
# Pre-build (once; re-run to bump the pinned SeqKit release):
#   ./services/seqkit-server/scripts/vendor.sh
#
# SeqKit is CPU-only — a single static Go binary, no CUDA, no model weights,
# no compiled algorithm stack. So unlike GPU/conda services the image only
# needs a Python venv for the bioq service framework + HTTP layer; the tool
# itself is the vendored binary COPYed from upstream/.

# Debian bookworm (12): system python is 3.11.2. Pinned to bookworm on purpose
# (see plip-server Dockerfile for the floating-tag / keyring rationale).
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# A writable HOME for tools that assume one when run headless.
ENV HOME=/tmp
# Huawei Cloud pip mirror (project-wide convention, ~10x faster from China).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# Minimal OS layer: the CPython that owns the venv + TLS certs for httpx/OSS.
RUN apt-get update && apt-get install --no-install-recommends -y \
        python3 python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv binary (fast installs) from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Plain venv (no --system-site-packages: SeqKit has no apt python deps).
WORKDIR /opt/seqkit
RUN uv venv .venv --python /usr/bin/python3

# Shared bioagent service framework + HTTP/remote-fetch deps into the venv.
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.4" \
        "httpx>=0.28" \
        python-multipart \
        alibabacloud-oss-v2

# Vendored SeqKit binary (static, linux/amd64). Prerequisite: scripts/vendor.sh.
COPY services/seqkit-server/upstream/seqkit /opt/seqkit/bin/seqkit
RUN chmod +x /opt/seqkit/bin/seqkit

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/seqkit-server/__init__.py \
     services/seqkit-server/__main__.py \
     services/seqkit-server/VERSION \
     services/seqkit-server/app.py \
     services/seqkit-server/adapter.py \
     services/seqkit-server/models.py \
     services/seqkit-server/settings.py \
     services/seqkit-server/tools.py \
     /opt/seqkit/server/

ENV SEQKIT_BIN=/opt/seqkit/bin/seqkit
ENV SEQKIT_JOBS_BASE_DIR=/data/seqkit_jobs
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir here. Requires the data-plane OSS bucket mounted at /mnt/oss in the
# FC console. Absent mount => no-op.
ENV SEQKIT_OSS_OUTPUT_MOUNT=/mnt/oss
# FC session affinity header (also needs FC console config).
ENV SEQKIT_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/seqkit_jobs

# Build smoke: the binary must run and the app module must import. Catches a
# broken vendor or venv wiring at build time.
RUN /opt/seqkit/bin/seqkit version \
    && .venv/bin/python -c "import server.app; print('server.app OK')"

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
