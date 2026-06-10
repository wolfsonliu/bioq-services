# Build from bioagent project root:
#   docker build --platform linux/amd64 -t boltzgen-server -f services/boltzgen-server/Dockerfile .
#
# Prerequisites (run once before build):
#   ./services/boltzgen-server/scripts/fetch_weights.sh
#
# BoltzGen source must be at opensource/boltzgen/ (GPU-required, no nvcc needed).
# Weights in services/boltzgen-server/weights/ (~10 GB, 5 checkpoints).
# Molecule data in services/boltzgen-server/moldir/ (~6 GB, CCD .pkl files).
#
# Base: uv's pre-built python3.12 slim image (Debian bookworm + uv pre-installed).
# Same rationale as boltz-server: modern PyTorch (>=2.0) bundles CUDA / cuDNN libs
# via nvidia-* PyPI sub-packages, so the nvidia/cuda base's libs are redundant.
# Slim + PyPI-bundled CUDA saves ~3 GB on the final image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (~10× faster from China than pypi.org).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/boltzgen
RUN uv venv .venv --python python3.12

# Heavy deps first. BoltzGen pins numpy==2.0.2 and requires torch>=2.4.1.
# torch's wheel transitively pulls nvidia-cuda-runtime-cu12 / nvidia-cudnn-cu12 /
# nvidia-cublas-cu12 / etc. (~2.7 GB total) — these provide GPU runtime libs.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "torch>=2.4.1" "numpy==2.0.2"

# BoltzGen source + editable install. cuequivariance deps (cu12 wheels, no nvcc
# needed) are listed as direct dependencies in pyproject.toml.
COPY opensource/boltzgen/pyproject.toml opensource/boltzgen/PYPI_DESCRIPTION.md /opt/boltzgen/
COPY opensource/boltzgen/src /opt/boltzgen/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e .

# Pre-staged model weights (5 checkpoints, ~10 GB total). Pre-stage once with:
#   ./services/boltzgen-server/scripts/fetch_weights.sh
COPY services/boltzgen-server/weights/ /opt/boltzgen/weights/

# Pre-staged CCD molecule library (.pkl files, ~6 GB extracted from mols.zip).
# Required: load_canonicals() is called unconditionally at pipeline start.
# For protein-only protocols only the 20 canonical amino acid .pkl files are
# needed; the full moldir is required for protein-small_molecule protocol.
COPY services/boltzgen-server/moldir/ /opt/boltzgen/moldir/

# Build-time import smoke — catches torch / cuequivariance ABI mismatches at
# build, not at FC deploy time.
RUN .venv/bin/python -c "import torch, cuequivariance_torch, boltzgen; \
    print('torch', torch.__version__, 'compiled cuda', torch.version.cuda)"

# Shared bioagent service framework + remote-fetch deps (late layer so the
# framework changing doesn't invalidate the BoltzGen install/weight layers).
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 pyyaml

# Server code — explicit file list to avoid re-copying weights/moldir/tests.
COPY services/boltzgen-server/__init__.py \
     services/boltzgen-server/__main__.py \
     services/boltzgen-server/VERSION \
     services/boltzgen-server/app.py \
     services/boltzgen-server/adapter.py \
     services/boltzgen-server/models.py \
     services/boltzgen-server/settings.py \
     services/boltzgen-server/tools.py \
     services/boltzgen-server/uris.py \
     /opt/boltzgen/server/

# Service settings — mirror services/boltzgen-server/settings.py (BoltzGenSettings).
ENV PYTHONPATH=/opt/boltzgen
ENV BOLTZGEN_ROOT=/opt/boltzgen
ENV BOLTZGEN_CLI=/opt/boltzgen/.venv/bin/boltzgen
ENV BOLTZGEN_PYTHON=/opt/boltzgen/.venv/bin/python
ENV BOLTZGEN_WEIGHTS_DIR=/opt/boltzgen/weights
ENV BOLTZGEN_MOLDIR=/opt/boltzgen/moldir
ENV BOLTZGEN_JOBS_BASE_DIR=/data/boltzgen_jobs
RUN mkdir -p /data/boltzgen_jobs

# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV BOLTZGEN_KEEPALIVE_URL=https://fc-boltzgen-karzusdoih.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV BOLTZGEN_SESSION_HEADER_NAME=bioagent-session-id
ENV BOLTZGEN_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/boltzgen — CPython prepends CWD to sys.path, making the
# `server/` subdir importable as `server.app`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
