# Build from bioagent project root:
#   docker build --platform linux/amd64 -t boltz-server -f services/boltz-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/boltz-server/scripts/vendor.sh   # vendor upstream/
# Pre-stage Boltz-2 weights (HuggingFace download) before build:
#   python -c "from pathlib import Path; from boltz.main import download_boltz2; \
#       download_boltz2(Path('services/boltz-server/weights'))"
#   rm -f services/boltz-server/weights/mols.tar
#
# Boltz upstream source lives at services/boltz-server/upstream/ (vendored at
# pinned SHA; see scripts/vendor.sh).  Weights are pre-staged at
# services/boltz-server/weights/ (HuggingFace; not in upstream repo).
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

# Boltz source — from vendored upstream/ (see scripts/vendor.sh) + editable
# install.  `.[cuda]` extras pull cuequivariance kernels (NVIDIA equivariant
# ops; cu12 wheels, no nvcc needed).
COPY services/boltz-server/upstream/pyproject.toml /opt/boltz/pyproject.toml
COPY services/boltz-server/upstream/README.md      /opt/boltz/README.md
COPY services/boltz-server/upstream/src            /opt/boltz/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e ".[cuda]"

# Boltz-2 weights are NOT bundled — they live on NAS at /data/models/boltz/
# (FC mount).  For SIF / HPC, bind the dir at run time:
#   apptainer run --bind /scratch/models/boltz:/data/models/boltz \
#       boltz-server.sif python -m server cofold ...
#
# Pre-stage weights (HuggingFace download) into either the local stage dir or
# directly into the NAS target:
#   python -c "from pathlib import Path; from boltz.main import download_boltz2; \
#       download_boltz2(Path('services/boltz-server/weights'))"
#   rm -f services/boltz-server/weights/mols.tar
#   rsync -a services/boltz-server/weights/ <NAS>:/data/models/boltz/
#
# Only Boltz-2 files are needed at /data/models/boltz/:
#   boltz2_conf.ckpt, boltz2_aff.ckpt, mols/
# (`boltz1_conf.ckpt` and `ccd.pkl` deliberately omitted — boltz2 codepath
# loads CCD via `mols/`+`load_canonicals()`, not `ccd.pkl`.)
# See engineering/decisions/2026-06-26-service-weights-externalization.md.

# Build-time import smoke — catches torch / cuequivariance ABI mismatches at
# build, not at FC deploy time. Note buildx doesn't expose GPUs, so we can only
# verify `import`, not `torch.cuda.is_available()` (which only confirms a GPU
# is reachable, not that the libs are correct).
RUN .venv/bin/python -c "import torch, cuequivariance_torch, boltz; \
    print('torch', torch.__version__, 'compiled cuda', torch.version.cuda)"

# Shared bioagent service framework + remote-fetch deps (late layer so the
# framework changing doesn't invalidate the Boltz install/weight layers above).
# COPY (not bind): framework source changes must bust this layer's cache so the
# output-sink fix ships in the image (BuildKit bind-mount content isn't a cache key).
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 pyyaml

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/boltz-server/__init__.py \
     services/boltz-server/__main__.py \
     services/boltz-server/VERSION \
     services/boltz-server/app.py \
     services/boltz-server/adapter.py \
     services/boltz-server/models.py \
     services/boltz-server/settings.py \
     services/boltz-server/tools.py \
     /opt/boltz/server/

# Service settings; mirror `services/boltz-server/settings.py` (BoltzSettings).
ENV PYTHONPATH=/opt/boltz
ENV BOLTZ_ROOT=/opt/boltz
ENV BOLTZ_BINARY=/opt/boltz/.venv/bin/boltz
ENV BOLTZ_CACHE_DIR=/data/models/boltz
ENV BOLTZ_JOBS_BASE_DIR=/data/boltz_jobs
RUN mkdir -p /data/boltz_jobs

# Output-sink: 当 gateway 传 X-Bioagent-Oss-Prefix 时，framework 把 job dir 镜像到此挂载点。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss。缺挂载 => no-op（仅 NAS）。
ENV BOLTZ_OSS_OUTPUT_MOUNT=/mnt/oss

# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV BOLTZ_KEEPALIVE_URL=https://fc-boltz-kbioniejif.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV BOLTZ_SESSION_HEADER_NAME=bioagent-session-id
ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/boltz (set above) — CPython prepends CWD to sys.path,
# making the `server/` subdir importable as `server.app`. Without this
# uvicorn fails with `ModuleNotFoundError: No module named 'server'`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
