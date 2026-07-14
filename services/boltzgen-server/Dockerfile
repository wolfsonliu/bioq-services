# Build from bioagent project root:
#   docker build --platform linux/amd64 -t boltzgen-server -f services/boltzgen-server/Dockerfile .
#
# Prerequisites (run once before build):
#   ./services/boltzgen-server/scripts/vendor.sh           # vendor upstream/
#
# BoltzGen source lives at services/boltzgen-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  Model weights (5 checkpoints, ~10 GB)
# and CCD moldir (~6 GB) are NOT baked into the image — they live on NAS
# at /data/models/boltzgen/{weights,moldir}/ (FC NAS mount; SIF / HPC users
# `apptainer run --bind` to inject).
#
# Pre-stage / upload to NAS once:
#   ./services/boltzgen-server/scripts/fetch_weights.sh    # → ./weights/
#   rsync -a services/boltzgen-server/weights/ <NAS>:/data/models/boltzgen/weights/
#   rsync -a services/boltzgen-server/moldir/  <NAS>:/data/models/boltzgen/moldir/
# (Or run fetch_weights.sh with WEIGHTS_DST pointing at the final dir.)
# Weights in services/boltzgen-server/weights/ (~10 GB, 5 checkpoints).
# Molecule data in services/boltzgen-server/moldir/ (~6 GB, CCD .pkl files).
#
# Base: nvidia/cuda 12.4.1 cudnn-runtime — provides a CUDA 12.4 runtime that
# matches the HPC node drivers (driver reports max CUDA 12.4.0).  Earlier
# slim base let pip resolve nvidia-*-cu12 wheels whose versions could drift to
# CUDA 12.5/12.6 runtimes, triggering "NVIDIA driver too old" on HPC.  The
# CUDA base image guarantees a consistent, validated 12.4 runtime baseline.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (~10× faster from China than pypi.org).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# gcc/g++ required at runtime: cuequivariance Triton kernels JIT-compile CUDA
# code via triton.runtime.build, which shells out to `cc`. Blackwell (compute
# capability ≥ 12) has no pre-built cuequivariance kernel so JIT is the only
# path. Without gcc the design step fails with "Failed to find C compiler".
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# uv binary + uv-managed Python 3.12 (avoids deadsnakes PPA — Ubuntu 22.04
# ships Python 3.10).  uv downloads a standalone CPython 3.12 build.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

WORKDIR /opt/boltzgen
RUN uv venv .venv --python 3.12

# Heavy deps first. BoltzGen pins numpy==2.0.2 and requires torch>=2.4.1.
# torch's wheel transitively pulls nvidia-cuda-runtime-cu12 / nvidia-cudnn-cu12 /
# nvidia-cublas-cu12 / etc. (~2.7 GB total) — these provide GPU runtime libs.
#
# Pin CUDA 12.4 torch builds — HPC nodes may have older drivers (CUDA 12.4 /
# driver 550.x) that reject cu126+ runtime.  Use PyTorch's cu124 index as
# primary so uv resolves torch from there; Huawei mirror stays as extra for
# numpy, nvidia-* sub-packages, and other deps.
# Override at build time: --build-arg TORCH_INDEX_URL=.../cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# Pin torch to 2.4.1 (not >=2.4.1): newer cu124 wheels (torch 2.5+ / 2.6+)
# transitively pull nvidia-cublas-cu12 / nvidia-cudnn-cu12 versions that
# need a CUDA runtime newer than driver 12.4 supports, producing
# "NVIDIA driver too old (found version 12040)" on HPC nodes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "torch==2.4.1" "numpy==2.0.2" \
        --index-url "$TORCH_INDEX_URL" \
        --extra-index-url "$UV_INDEX_URL"

# BoltzGen source — from vendored upstream/ (see scripts/vendor.sh).
# cuequivariance deps (cu12 wheels, no nvcc needed) are listed as direct
# dependencies in pyproject.toml.
COPY services/boltzgen-server/upstream/pyproject.toml         /opt/boltzgen/pyproject.toml
COPY services/boltzgen-server/upstream/PYPI_DESCRIPTION.md    /opt/boltzgen/PYPI_DESCRIPTION.md
COPY services/boltzgen-server/upstream/src                    /opt/boltzgen/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e .

# Weights + moldir are NOT bundled — they live on a NAS volume mounted by FC
# at /data/models/boltzgen/{weights,moldir}/.  For SIF / HPC, bind the dirs
# at run time:
#   apptainer run --bind /scratch/models/boltzgen:/data/models/boltzgen \
#       boltzgen-server.sif python -m server design ...
# Local download to a stage dir (~16 GB; default ./services/boltzgen-server/
# weights+moldir) is via:
#   ./services/boltzgen-server/scripts/fetch_weights.sh
# After download, rsync the contents to the NAS / HPC target.
# See engineering/decisions/2026-06-26-service-weights-externalization.md.

# Build-time import smoke — catches torch / cuequivariance ABI mismatches at
# build, not at FC deploy time.
RUN .venv/bin/python -c "import torch, cuequivariance_torch, boltzgen; \
    print('torch', torch.__version__, 'compiled cuda', torch.version.cuda)"

# Shared bioagent service framework + remote-fetch deps (late layer so the
# framework changing doesn't invalidate the BoltzGen install/weight layers).
#
# COPY (not bind-mount): BuildKit does not include bind-mount content in the
# layer cache key, so a bind would ship a stale framework even after the
# framework source is updated (would silently drop the output-sink fix).
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
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
     /opt/boltzgen/server/

# Service settings — mirror services/boltzgen-server/settings.py (BoltzGenSettings).
ENV PYTHONPATH=/opt/boltzgen
ENV BOLTZGEN_ROOT=/opt/boltzgen
ENV BOLTZGEN_CLI=/opt/boltzgen/.venv/bin/boltzgen
ENV BOLTZGEN_PYTHON=/opt/boltzgen/.venv/bin/python
ENV BOLTZGEN_WEIGHTS_DIR=/data/models/boltzgen/weights
ENV BOLTZGEN_MOLDIR=/data/models/boltzgen/moldir
ENV BOLTZGEN_JOBS_BASE_DIR=/data/boltzgen_jobs
RUN mkdir -p /data/boltzgen_jobs

# Output-sink: when the gateway sends X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console (storage config). Absent mount => no-op (NAS-only).
ENV BOLTZGEN_OSS_OUTPUT_MOUNT=/mnt/oss

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
