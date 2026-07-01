# Build from bioagent project root:
#   docker build --platform linux/amd64 -t mmseqs2-server -f services/mmseqs2-server/Dockerfile .
#
# mmseqs2-server — FastAPI wrapper around the GPU-enabled `mmseqs` binary,
# exposing the ColabFold MSA HTTP protocol so any client that talks to
# `https://api.colabfold.com` (notably boltz-server's `--msa_server_url`)
# can be redirected to an internal endpoint.
#
# Base: nvidia/cuda 12.4 runtime + cuDNN — needed because mmseqs2 GPU mode
# loads CUDA libs from the system, not from PyPI wheels. (Unlike boltz-server,
# which gets CUDA via PyTorch's bundled nvidia-* wheels, mmseqs is a standalone
# binary linked against the system libcuda + libcudart.)
#
# The MSA pipeline orchestration is NOT pip-installed from upstream ColabFold;
# it is vendored into server/orchestrator.py — see VENDOR_INFO.md + the design
# doc's "Vendor 约定" section. Keeps the image lean and decouples us from
# ColabFold's research-grade release cadence.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster from China than
# pypi.org).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# Python 3.11 + curl + ca-certs. python3.11 is the project's pinned minimum
# (pyproject.toml `requires-python = ">=3.11"`).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# uv (pinned via the Astral image — copy just the binaries, no Python install).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Project venv up front so subsequent COPYs do not overlay it.
WORKDIR /opt/mmseqs2
RUN uv venv /opt/mmseqs2/.venv --python python3.11
ENV PATH=/opt/mmseqs2/.venv/bin:$PATH

# -----------------------------------------------------------------------------
# MMseqs2 GPU binary — vendored on host by scripts/vendor.sh.
#
# Prerequisite (run once before `docker build`, re-run to bump version):
#   ./services/mmseqs2-server/scripts/vendor.sh
#
# The script drops the GPU tarball at
# `services/mmseqs2-server/upstream/mmseqs-linux-gpu.tar.gz` (gitignored).
# Docker build is now 0-network for this step, which avoids the CN-side
# GitHub Releases connection resets we hit trying to download in-image.
# The version is pinned in `scripts/vendor.sh` (default 18-8cc5c).
# -----------------------------------------------------------------------------
COPY services/mmseqs2-server/upstream/mmseqs-linux-gpu.tar.gz /tmp/mmseqs.tgz
RUN tar xz -C /opt -f /tmp/mmseqs.tgz \
    && rm /tmp/mmseqs.tgz \
    && mv /opt/mmseqs /opt/mmseqs-gpu
ENV PATH=/opt/mmseqs-gpu/bin:$PATH

# DO NOT pip-install ColabFold — orchestrator is vendored (see Task 2.0).

# -----------------------------------------------------------------------------
# Shared bioagent service framework + runtime deps. Bind-mount the framework
# source so each rebuild of the framework doesn't bloat the layer.
# -----------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/mmseqs2/.venv/bin/python \
        /tmp/service-framework \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.4" \
        "httpx>=0.28" \
        python-multipart

# -----------------------------------------------------------------------------
# Server code — copied last so iterating on routes/adapters doesn't bust the
# heavier mmseqs + framework layers above.
# -----------------------------------------------------------------------------
COPY services/mmseqs2-server /opt/mmseqs2/server
ENV PYTHONPATH=/opt:/opt/mmseqs2

# -----------------------------------------------------------------------------
# NAS mount conventions — follows the `/data/models/<svc>/` externalization
# pattern (engineering/decisions/2026-06-26-service-weights-externalization.md).
# /data/models/mmseqs2 — pre-built ColabFold DB dir, mounted from NAS.
# /data/mmseqs2_jobs   — per-job working directories (NAS-backed so jobs
#                        survive instance reclamation).
# -----------------------------------------------------------------------------
ENV MMSEQS2_DB_DIR=/data/models/mmseqs2
ENV MMSEQS2_JOBS_BASE_DIR=/data/mmseqs2_jobs
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV MMSEQS2_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/models/mmseqs2 /data/mmseqs2_jobs

ENV PORT=9000
EXPOSE 9000
CMD ["python", "-m", "server"]
