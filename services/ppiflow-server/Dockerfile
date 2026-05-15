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
# Redirect conda channels to TUNA (Tsinghua) mirror via `.condarc`.
#
# Upstream env.yml lists 8 境外 channels (defaults / conda-forge / nvidia /
# pytorch / schrodinger / bioconda / ostrokach / anaconda) BEFORE the 3 TUNA
# fallback URLs at the bottom. micromamba probes in order, so most metadata
# requests hit the slow upstream channels first → 5–10 min of round-trips
# before falling back to TUNA.
#
# `~/.condarc` with `custom_channels` remaps channel-name → mirror-URL on
# every micromamba request. We use this instead of sed-rewriting env.yml
# because (a) it works even when env.yml channels list updates upstream and
# (b) it doesn't need to know which YAML lines to delete.
#
# Mirror coverage (verified 2026-05-15):
#   - conda-forge / pytorch / bioconda / pkgs/main → TUNA ✅
#   - nvidia → NOT mirrored by TUNA/USTC/Aliyun/Huawei (404). The `nvidia`
#     channel only supplies `cuda-nvcc=12.1.66` (~50MB) in this env.yml, so
#     we leave it to fall through to upstream `conda.anaconda.org/nvidia`
#     by NOT listing it in custom_channels. Slow once, then cache mount
#     catches it.
#   - schrodinger / ostrokach / anaconda: also fall through to upstream;
#     no packages in env.yml's `dependencies:` actually originate there.
#
# Previously tried Huawei mirror (commit before this one) — it kept failing
# with "Could not load repodata.json for conda-forge". TUNA is more reliable
# for cloud channels.
# ---------------------------------------------------------------------------
RUN cat > /root/.condarc <<'EOF'
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
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
COPY opensource/PPIFlow/tool/PPIFlow/environment.yml /tmp/ppiflow-env.yml

# ---------------------------------------------------------------------------
# Create the `ppiflow` conda env from upstream's environment.yml.
#
# BuildKit cache mounts (the big optimization):
#   - /opt/conda/pkgs        — conda package cache (extracted tarballs).
#                              Saves ~1–2 GB re-download on env.yml-changing
#                              rebuilds.
#   - /root/.cache/pip       — pip wheel cache for the env.yml `pip:` section.
#                              The ~2 GB torch wheels from download.pytorch.org
#                              (no China CDN — ~10 min download) get cached
#                              across rebuilds. First build is unavoidably
#                              slow; rebuilds drop from ~25 min to ~3 min.
#
# We DELIBERATELY DROP `micromamba clean --all`. The cache-mounted directories
# are excluded from the image layer by BuildKit, so cleaning inside the RUN
# would only purge the persistent cache — defeating the optimization. Image
# size is unaffected: only `/opt/conda/envs/ppiflow/` (the realized env, ~10 GB)
# ends up in the layer; `pkgs/` and `~/.cache/pip` do not.
#
# Cache-friendly layer ordering: env.yml change invalidates this layer, but
# the cache mount still hits → most package downloads are re-used from cache,
# only newly-added packages get fetched.
# ---------------------------------------------------------------------------
WORKDIR /opt/ppiflow

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba env create -y -n ppiflow -f /tmp/ppiflow-env.yml \
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
