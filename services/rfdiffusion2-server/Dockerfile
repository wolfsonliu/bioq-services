# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfdiffusion2-server -f services/rfdiffusion2-server/Dockerfile .
#
# Prereqs (both gitignored, populate before build):
#   ./services/rfdiffusion2-server/scripts/vendor.sh         # syncs upstream/
#   ./services/rfdiffusion2-server/scripts/fetch_weights.sh  # syncs weights/
# See engineering/decisions/2026-05-19-rfdiffusion2-server-vendor.md.
#
# Multistage layout
# -----------------
#   builder = heavy: bzip2/curl/wget/git + micromamba + conda env create
#             (~10 GB env including pyrosetta, torch+cu124, dgl, openbabel).
#             Final RUN strips jupyterlab/*.a/share/doc/tests/__pycache__.
#   runtime = lean: only build-essential + ca-certificates. The finished
#             conda env + vendored source + weights + server code are copied
#             over.
#
# Why two stages: bzip2/curl/wget/git/micromamba aren't needed once the env
# is built, but in a single Dockerfile `apt-get remove` after install can't
# reclaim the bytes from the install layer. Multistage is the only way to
# leave those packages behind. Same playbook as rfdiffusion-server.
#
# Base: CUDA 12.4 -devel (REQUIRED, do not switch to -runtime). rf2aa imports
# ipd.dev.cuda.cudabuild which calls `torch.utils.cpp_extension.load(...)` —
# JIT-compiles a CUDA kernel on first model construction. That path needs
# nvcc + cuda_runtime_api.h, only present in -devel. We also keep
# build-essential in the runtime stage so the JIT compile succeeds.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


# ===========================================================================
# Stage 1: builder — install the rfd2 conda env + vendored source.
# ===========================================================================
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pin torch's CUDA discovery to the system toolkit.
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# Build-only apt deps + micromamba bootstrap.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && /usr/local/bin/micromamba --version

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda.
#
# pip: Aliyun mirror. Configure BEFORE micromamba env-create — env.yml's
# `pip:` section invokes pip during env create and would otherwise hit pypi.org.
#
# conda: TUNA (Tsinghua) for the standard channels. Three upstream channels
# we leave alone (fall through to conda.anaconda.org):
#   - conda.rosettacommons.org    pyrosetta (proprietary, no Chinese mirror)
#   - nvidia/label/cuda-12.4.0    TUNA has `cloud/nvidia` (main) but NOT
#                                 `label/*` — mapping it 404s the build.
#   - dglteam/label/th24_cu124    same limitation.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Create the rfd2 conda env from the vendored env files.
# ---------------------------------------------------------------------------
COPY services/rfdiffusion2-server/upstream/envs/cuda124_env.yml /tmp/rfd2-env/cuda124_env.yml
COPY services/rfdiffusion2-server/upstream/envs/requirements_cuda124.txt /tmp/rfd2-env/requirements_cuda124.txt

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    cd /tmp/rfd2-env \
    && micromamba env create -y -n rfd2 -f cuda124_env.yml \
    && rm -rf /tmp/rfd2-env

# ---------------------------------------------------------------------------
# Pip extras from upstream's apptainer .spec.
#
# NOTE: `baker-laboratory/ipd` is NOT pip-installed here. Its github HEAD has
# moved past the RFdiffusion2 cut and now imports `evn` at module load, which
# isn't in the conda env. We instead vendor opensource/RFdiffusion2/ipd/ into
# upstream/ipd/ (see scripts/vendor.sh) so the contemporary copy lives on
# PYTHONPATH alongside rf_diffusion/.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/rfd2/bin/pip install \
        git+https://github.com/RalphMao/PyTimer.git

# ---------------------------------------------------------------------------
# Pull in the vendored source + weights + framework. Strip dead weight from
# the conda env in the SAME layer so cleanup actually shrinks the image.
#
# Cleanup targets (cumulative ~1.2 GB):
#   1. jupyterlab + ipywidgets — pulled in by env.yml / requirements but never
#      reached by the inference path.
#   2. *.a static archives under the env's lib/. Conda's pytorch-cuda /
#      cudatoolkit ship static libs alongside .so; only .so are dlopened.
#   3. share/{doc,man,locale} — documentation + i18n catalogs.
#   4. test/ + tests/ dirs inside site-packages.
#   5. __pycache__ + *.pyc — conda packages ship pre-compiled bytecode.
# ---------------------------------------------------------------------------
COPY services/rfdiffusion2-server/upstream /opt/rfdiffusion2-server/upstream
COPY services/rfdiffusion2-server/weights /opt/rfdiffusion2-server/upstream/rf_diffusion/model_weights

# Patch SE3Transformer NVTX profiling — conda's PyTorch 2.4.0 doesn't compile
# the _C._nvtx binding, so `from torch.cuda.nvtx import range` raises
# RuntimeError at call time. Replace with a try/except fallback to
# contextlib.nullcontext (no-op context manager). Done here in the Dockerfile
# (not vendor.sh) so the patch is guaranteed present regardless of vendor state.
RUN sed -i 's/from torch\.cuda\.nvtx import range as nvtx_range/try:\n    from torch.cuda.nvtx import range as nvtx_range\nexcept (RuntimeError, AttributeError):\n    from contextlib import nullcontext as nvtx_range/' \
    /opt/rfdiffusion2-server/upstream/rf2aa/SE3Transformer/se3_transformer/model/basis.py \
    /opt/rfdiffusion2-server/upstream/rf2aa/SE3Transformer/se3_transformer/model/layers/attention.py \
    /opt/rfdiffusion2-server/upstream/rf2aa/SE3Transformer/se3_transformer/model/layers/norm.py \
    /opt/rfdiffusion2-server/upstream/rf2aa/SE3Transformer/se3_transformer/model/layers/convolution.py

COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/rfd2/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 \
    && rm -rf /tmp/service-framework \
    && /opt/conda/envs/rfd2/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/rfd2/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/rfd2/share/doc \
        /opt/conda/envs/rfd2/share/man \
        /opt/conda/envs/rfd2/share/locale \
    && find /opt/conda/envs/rfd2/lib/python3.11/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/rfd2 -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/rfd2 -name '*.pyc' -delete


# ===========================================================================
# Stage 2: runtime — keep the JIT-compile toolchain, drop the rest.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH
# Checkpoint configs embed ${oc.env:USER} for wandb_dir; resolve needs this set.
ENV USER=root

# Runtime apt is minimal:
#   build-essential   — REQUIRED. ipd.dev.cuda.cudabuild does runtime JIT compile
#                       via torch.cpp_extension.load on first model construction.
#   ca-certificates   — httpx HTTPS for OSS / signed URL inputs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring over the conda env + vendored source (with weights baked in) as
# separate COPYs so the big stable layers cache independently.
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/rfdiffusion2-server /opt/rfdiffusion2-server

# WORKDIR must be set in the runtime stage; uvicorn imports `server.app:app`
# from CWD.
WORKDIR /opt/rfdiffusion2-server

# Server code last so iterating on the FastAPI layer only rebuilds the tail.
COPY services/rfdiffusion2-server/ /opt/rfdiffusion2-server/server/

# Settings (env-driven; see services/rfdiffusion2-server/settings.py).
ENV RFDIFFUSION2_ROOT=/opt/rfdiffusion2-server
ENV RFDIFFUSION2_MODELS_DIR=/opt/rfdiffusion2-server/upstream/rf_diffusion/model_weights
ENV RFDIFFUSION2_INFERENCE_SCRIPT=/opt/rfdiffusion2-server/upstream/rf_diffusion/run_inference.py
ENV RFDIFFUSION2_PYTHON=/opt/conda/envs/rfd2/bin/python
ENV RFDIFFUSION2_PYTHONPATH=/opt/rfdiffusion2-server/upstream
ENV RFDIFFUSION2_JOBS_BASE_DIR=/data/rfdiffusion2_jobs
RUN mkdir -p /data/rfdiffusion2_jobs

# Alibaba Cloud FC: port 9000, keep-alive ≥ 15 min, /healthz ready < 120 s.
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/rfd2/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
