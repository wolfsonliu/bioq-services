# Build from bioagent project root:
#   docker build --platform linux/amd64 -t qligfep-server -f services/qligfep-server/Dockerfile .
#
# Prerequisites (one-time):
#   ./services/qligfep-server/scripts/vendor.sh   # vendor upstream/qligfep + upstream/Q6
#
# See engineering/decisions/2026-07-06-qligfep-server-design.md §6.4.

# ---- Q6 builder ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS q6-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake gfortran openmpi-bin libopenmpi-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY services/qligfep-server/upstream/Q6 /src/Q6
WORKDIR /src/Q6/src
# Q6 (esguerra/Q6 @ 202d90c) has no CUDA build target — only default (gfortran)
# + mpi (via `make mpi COMP=gcc`).  device=gpu in run-fep is therefore not
# supported by v0.0.1; qligfep-server /healthz/detail will show qdyn_cuda
# as missing, and RunFepRequest(device=gpu) will fail with a clear
# "binary not found" from _select_binary.  See engineering/decisions/
# 2026-07-06-qligfep-server-design.md §10 risk #1 for context.
RUN mkdir -p /opt/Q6/bin \
    && make all COMP=gcc \
    && cp ../bin/qdyn ../bin/qprep ../bin/qfep ../bin/qcalc /opt/Q6/bin/ \
    && make clean \
    && make mpi COMP=gcc \
    && cp ../bin/qdynp /opt/Q6/bin/

# ---- runtime ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv ca-certificates curl \
        openmpi-bin libopenmpi3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
COPY --from=q6-builder /opt/Q6/bin /opt/Q6/bin
ENV PATH="/opt/Q6/bin:$PATH"

WORKDIR /opt/qligfep-server
RUN uv venv .venv --python python3.10

# Heavy deps first (cache-friendly layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        rdkit "openff-toolkit>=0.15" "openff-interchange>=0.3" mdtraj numpy

# Upstream qligfep (vendor.sh has already applied settings.py shim)
COPY services/qligfep-server/upstream/qligfep /opt/qligfep-server/upstream/qligfep

# Service framework
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code
COPY services/qligfep-server/ /opt/qligfep-server/server/

ENV PYTHONPATH=/opt/qligfep-server
ENV QLIGFEP_ROOT=/opt/qligfep-server
ENV QLIGFEP_Q_BIN_DIR=/opt/Q6/bin
ENV QLIGFEP_UPSTREAM_DIR=/opt/qligfep-server/upstream/qligfep
ENV QLIGFEP_JOBS_BASE_DIR=/data/qligfep_jobs
RUN mkdir -p /data/qligfep_jobs

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
