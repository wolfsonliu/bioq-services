# Build from bioagent project root:
#   docker build --platform linux/amd64 -t boltz-server -f services/boltz-server/Dockerfile .
#
# Boltz source must be at opensource/boltz/ (GPU-required, but built without nvcc).
# Final image ~11.1 GB — code stack ~5 GB + Boltz-2 weights/cache ~6 GB.
#
# Base: uv's pre-built python3.12 slim image (Debian bookworm + uv pre-installed).
# Chosen over `nvidia/cuda:*-cudnn-runtime` because modern PyTorch (>=2.0)
# bundles CUDA / cuDNN libs via its own `nvidia-*` PyPI sub-packages, so the
# nvidia/cuda base image's CUDA libs are duplicated (and never loaded). slim +
# PyPI-bundled CUDA saves ~3 GB on the final image. See
# engineering/decisions/2026-05-19-boltz-server-design.md "Docker" section
# for the trade-off analysis and the rollback path if FC's GPU driver is too
# old to support the bundled CUDA 12.x runtime.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster from China than
# pypi.org).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

# Create the venv up front so subsequent COPYs do not overlay it. uv image
# ships uv + python3.12 system-wide, so no apt python install needed.
WORKDIR /opt/boltz
RUN uv venv .venv --python python3.12

# Heavy deps first (torch + numpy<2.0). torch's wheel transitively pulls
# nvidia-cuda-runtime-cu12 / nvidia-cudnn-cu12 / nvidia-cublas-cu12 / etc.
# (~2.7 GB total) — these provide the GPU runtime libs at the Python level.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "torch>=2.2" "numpy<2.0"

# Boltz source + editable install. `.[cuda]` extras pull cuequivariance kernels
# (NVIDIA equivariant ops; cu12 wheels, no nvcc needed). Only the files needed
# for `pip install -e` are copied here — pyproject references README.md, and
# the `boltz` package lives under src/.
COPY opensource/boltz/pyproject.toml opensource/boltz/README.md /opt/boltz/
COPY opensource/boltz/src /opt/boltz/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e ".[cuda]"

# Pre-staged Boltz-2 weights, copied directly from `opensource/boltz/weights/`
# so `docker build` does no network download (HuggingFace can be slow/blocked
# from CN build hosts). Pre-stage once with:
#   python -c "from pathlib import Path; from boltz.main import download_boltz2; \
#       download_boltz2(Path('opensource/boltz/weights'))"
#   rm -f opensource/boltz/weights/mols.tar
#
# Only Boltz-2 files are COPYed — `boltz1_conf.ckpt` and `ccd.pkl` are
# deliberately not pulled in (boltz-server is Boltz-2 only, and the boltz2
# codepath loads CCD via `mols/`+`load_canonicals()`, not `ccd.pkl` — see
# opensource/boltz/src/boltz/main.py:765). Listing each path explicitly here
# (instead of `COPY weights /opt/boltz/weights`) is what enforces the
# exclusion; an empty `mols.tar` marker is created in the same RUN so that
# `download_boltz2` (called by every `boltz predict`) skips its existence
# check and doesn't re-download 1.5 GB.
COPY opensource/boltz/weights/boltz2_conf.ckpt /opt/boltz/weights/boltz2_conf.ckpt
COPY opensource/boltz/weights/boltz2_aff.ckpt /opt/boltz/weights/boltz2_aff.ckpt
COPY opensource/boltz/weights/mols /opt/boltz/weights/mols
RUN touch /opt/boltz/weights/mols.tar

# Build-time import smoke — catches torch / cuequivariance ABI mismatches at
# build, not at FC deploy time. Note buildx doesn't expose GPUs, so we can only
# verify `import`, not `torch.cuda.is_available()` (which only confirms a GPU
# is reachable, not that the libs are correct).
RUN .venv/bin/python -c "import torch, cuequivariance_torch, boltz; \
    print('torch', torch.__version__, 'compiled cuda', torch.version.cuda)"

# Shared bioagent service framework + remote-fetch deps (late layer so the
# framework changing doesn't invalidate the Boltz install/weight layers above).
# Bind mount: framework dir is available during RUN but never stored in a layer.
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 pyyaml

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/boltz-server/__init__.py \
     services/boltz-server/VERSION \
     services/boltz-server/app.py \
     services/boltz-server/adapter.py \
     services/boltz-server/models.py \
     services/boltz-server/settings.py \
     services/boltz-server/tools.py \
     services/boltz-server/uris.py \
     /opt/boltz/server/

# Service settings; mirror `services/boltz-server/settings.py` (BoltzSettings).
ENV BOLTZ_ROOT=/opt/boltz
ENV BOLTZ_BINARY=/opt/boltz/.venv/bin/boltz
ENV BOLTZ_CACHE_DIR=/opt/boltz/weights
ENV BOLTZ_JOBS_BASE_DIR=/data/boltz_jobs
RUN mkdir -p /data/boltz_jobs

ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/boltz (set above) — CPython prepends CWD to sys.path,
# making the `server/` subdir importable as `server.app`. Without this
# uvicorn fails with `ModuleNotFoundError: No module named 'server'`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
