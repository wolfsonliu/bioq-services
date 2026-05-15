# Build from bioagent project root:
#   docker build --platform linux/amd64 -t ppiflow-server -f services/ppiflow-server/Dockerfile .
#
# PPIFlow source must be at opensource/PPIFlow/ (weights baked into image).
# This image only supports structure generation (sample_*.py). Sequence design,
# scoring, packing, and Rosetta steps are NOT included — they belong in
# separate bioagent services.
#
# Base: cudnn-runtime variant of nvidia/cuda 12.4.1. Downgraded from `-devel` on
# 2026-05-14 to shave ~3.6 GB off the image (devel = 6.7 GB vs runtime = 3.1 GB).
# The conda env install path doesn't compile any CUDA kernel from source — every
# pip wheel (torch, pyg-lib, torch-{cluster,scatter,sparse}, deepspeed, etc.) is
# pre-built; bioconda/ostrokach conda packages are also pre-built. nvcc + CUDA
# headers (the only meaningful extras in `-devel`) are therefore unused in the
# install chain, and PPIFlow's inference path doesn't trigger JIT CUDA compilation.
# If a future dep does need nvcc at install time, revert to `-devel` or split
# into a multi-stage build (see git history for the previous Dockerfile).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention). UV_INDEX_URL controls the
# uv pip layer below; pip.conf (further down) controls the pip invocations
# micromamba makes while resolving env.yml's pip: section.
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

# ---------------------------------------------------------------------------
# Install micromamba on top of the nvidia/cuda base. We pin via the public
# tarball URL rather than apt to avoid pulling in conda's whole channel cache.
#
# uv is brought in for the final framework-install layer only (replaces a
# slower pip-on-conda-env install). The conda env itself stays conda-managed
# because PPIFlow's env.yml leans on bioconda-only packages (pymol-open-source,
# pyqt5, cuda-nvcc) that uv/PyPI can't supply.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && /usr/local/bin/micromamba --version

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/ppiflow/bin:$PATH

# ---------------------------------------------------------------------------
# Configure pip to use the Huawei Cloud mirror **before** micromamba runs
# `env create` (which itself invokes pip for the pip: section of env.yml).
# Without this, CN build hosts hit flaky pypi.org and get "(from versions: none)"
# errors on random packages — we observed this on aiohappyeyeballs==2.5.0 even
# though absl-py from the same install batch succeeded.
#
# Huawei Cloud is the project-wide pip mirror (matches genie3-server and
# rfantibody-server). The pytorch wheel index (download.pytorch.org) stays as
# the explicit --extra-index-url inside env.yml's pip: section — we don't
# override that.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip && \
    printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
    > /root/.config/pip/pip.conf

# ---------------------------------------------------------------------------
# Create the `ppiflow` conda env from upstream's environment.yml.
# Cache-friendly layer: env.yml rarely changes vs the PPIFlow source itself.
# ---------------------------------------------------------------------------
WORKDIR /opt/ppiflow

COPY opensource/PPIFlow/tool/PPIFlow/environment.yml /tmp/ppiflow-env.yml
RUN micromamba env create -y -n ppiflow -f /tmp/ppiflow-env.yml \
    && micromamba clean --all --yes \
    && rm /tmp/ppiflow-env.yml

# ---------------------------------------------------------------------------
# Copy the PPIFlow tool source + pretrained checkpoints.
# The four .ckpt files baked into the image are heavy (~100s MB each); make
# sure opensource/PPIFlow/checkpoint/ is populated before building.
# ---------------------------------------------------------------------------
COPY opensource/PPIFlow/tool/PPIFlow /opt/ppiflow
COPY opensource/PPIFlow/checkpoint /opt/ppiflow/checkpoint

# ---------------------------------------------------------------------------
# Shared bioagent service framework + remote-fetch deps in the same env.
# Keeping this as a late layer so the framework re-install doesn't invalidate
# the (slow) conda env build cache. uv pip resolves and installs into the
# conda env's interpreter — faster than conda env's bundled pip and benefits
# from Huawei mirror (UV_INDEX_URL set above).
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/ppiflow/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code as `server` package — uvicorn imports `server.app:app`. Mounted
# alongside PPIFlow's tool directory so subprocess_cwd resolves correctly.
COPY services/ppiflow-server/ /opt/ppiflow/server/

# Settings via env (see services/ppiflow-server/settings.py).
ENV PPIFLOW_ROOT=/opt/ppiflow
ENV PPIFLOW_CKPT_DIR=/opt/ppiflow/checkpoint
ENV PPIFLOW_CONFIG_DIR=/opt/ppiflow/configs
ENV PPIFLOW_JOBS_BASE_DIR=/data/ppiflow_jobs
RUN mkdir -p /data/ppiflow_jobs

# Alibaba Cloud FC plumbing.
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/ppiflow/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
