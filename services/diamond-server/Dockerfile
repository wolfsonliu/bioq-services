# Build from bioagent project root:
#   docker build --platform linux/amd64 -t diamond-server -f services/diamond-server/Dockerfile .
#
# Pre-build (once; re-run to bump the pinned DIAMOND release):
#   ./services/diamond-server/scripts/vendor.sh
#
# DIAMOND is CPU-only — no CUDA. The prebuilt linux64 binary is vendored on the
# host by scripts/vendor.sh into services/diamond-server/upstream/ (gitignored)
# and extracted here. No model weights are baked in; reference `.dmnd` databases
# live on NAS at /data/models/diamond/ (weights-externalization convention).
# Final image ~250 MB — diamond binary + framework + fastapi/httpx wheels.

# Base: uv's prebuilt python3.11 slim image (Debian bookworm + uv preinstalled).
# Same base as dockq-server (a CPU binary service); glibc 2.36 runs the DIAMOND
# linux64 release fine.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster from China).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# libgomp1: DIAMOND links OpenMP dynamically. ca-certificates: httpx/OSS TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# DIAMOND binary — extracted from the vendored prebuilt tarball (top-level
# `diamond` entry). Prerequisite: run scripts/vendor.sh before building.
COPY services/diamond-server/upstream/diamond-linux64.tar.gz /tmp/diamond.tgz
RUN tar xz -C /usr/local/bin -f /tmp/diamond.tgz diamond \
    && rm /tmp/diamond.tgz \
    && chmod +x /usr/local/bin/diamond \
    && /usr/local/bin/diamond version

# Create the venv up front so subsequent COPYs do not overlay it.
WORKDIR /opt/diamond
RUN uv venv .venv --python python3.11

# Shared bioagent service framework + HTTP/remote-fetch deps.
COPY services/_framework /tmp/service-framework
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

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/diamond-server/__init__.py \
     services/diamond-server/__main__.py \
     services/diamond-server/VERSION \
     services/diamond-server/app.py \
     services/diamond-server/adapter.py \
     services/diamond-server/a3m.py \
     services/diamond-server/diamond_driver.py \
     services/diamond-server/models.py \
     services/diamond-server/settings.py \
     services/diamond-server/tools.py \
     /opt/diamond/server/

# PYTHONPATH lets `server` (and `server.diamond_driver`) import under any CWD
# (Docker WORKDIR covers Docker; Apptainer/Singularity ignore WORKDIR).
ENV PYTHONPATH=/opt/diamond
ENV DIAMOND_BINARY=/usr/local/bin/diamond
ENV DIAMOND_JOBS_BASE_DIR=/data/diamond_jobs
ENV DIAMOND_DB_DIR=/data/models/diamond
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir here. Requires the data-plane OSS bucket mounted at /mnt/oss in the FC
# console. Absent mount => no-op (NAS only). Default already /mnt/oss; explicit
# for self-documentation.
ENV DIAMOND_OSS_OUTPUT_MOUNT=/mnt/oss
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV DIAMOND_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/diamond_jobs /data/models/diamond

ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/diamond (set above) — CPython prepends CWD to sys.path, making
# the `server/` subdir importable as `server.app`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
