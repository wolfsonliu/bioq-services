# Build from bioagent project root:
#   docker build --platform linux/amd64 -t lightdock-server -f services/lightdock-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/lightdock-server/scripts/vendor.sh
#
# LightDock source lives at services/lightdock-server/upstream/ (vendored at a
# pinned SHA via scripts/vendor.sh). CPU-only Python + C/Cython extensions, no
# NN weights. Final image ~450 MB.

# Base: uv's pre-built python3.11 slim image (Debian bookworm + uv pre-installed).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10x faster from China).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# build-essential: LightDock compiles pre-generated C + Cython extensions during
# `pip install -e .`. Kept in the (single-stage) image; a future optimization
# could multistage it out.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the venv up front so subsequent COPYs do not overlay it.
WORKDIR /opt/lightdock
RUN uv venv .venv --python python3.11

# Heavy deps first (rarely changes). numpy<2.0 is pinned because LightDock's
# extensions are shipped as pre-generated C compiled against the numpy 1.x
# C-API; numpy 2.x would risk an ABI mismatch at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "numpy<2.0" "scipy>=1.7.0" "cython>=0.29.13" setuptools \
        "prody>=2.0.1" "freesasa>=2.0.3"

# LightDock source — copied from vendored upstream/ (see scripts/vendor.sh) into
# /opt/lightdock/ (overlay style so the existing .venv survives).
# --no-build-isolation forces the compile to use the numpy<2.0 installed above
# (setup.py imports numpy at build time); build isolation would otherwise pull a
# fresh, possibly numpy-2 build env and compile against the wrong C-API.
COPY services/lightdock-server/upstream/ /opt/lightdock/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-build-isolation -e .

# Shared bioagent service framework + remote-fetch deps (late layer so framework
# changes don't invalidate the LightDock install layer above).
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/lightdock-server/__init__.py \
     services/lightdock-server/__main__.py \
     services/lightdock-server/VERSION \
     services/lightdock-server/app.py \
     services/lightdock-server/adapter.py \
     services/lightdock-server/docking.py \
     services/lightdock-server/models.py \
     services/lightdock-server/settings.py \
     services/lightdock-server/tools.py \
     /opt/lightdock/server/

ENV PYTHONPATH=/opt/lightdock
ENV LIGHTDOCK_ROOT=/opt/lightdock
ENV LIGHTDOCK_PYTHON=/opt/lightdock/.venv/bin/python
ENV LIGHTDOCK_DRIVER_SCRIPT=/opt/lightdock/server/docking.py
ENV LIGHTDOCK_BIN_DIR=/opt/lightdock/.venv/bin
ENV LIGHTDOCK_JOBS_BASE_DIR=/data/lightdock_jobs
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir here. Requires the data-plane OSS bucket mounted at /mnt/oss in the FC
# console. Absent mount => no-op (NAS only).
ENV LIGHTDOCK_OSS_OUTPUT_MOUNT=/mnt/oss
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV LIGHTDOCK_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/lightdock_jobs

ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/lightdock (set above) — CPython prepends CWD to sys.path,
# making the `server/` subdir importable as `server.app`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
