# Build from bioagent project root:
#   docker build --platform linux/amd64 -t alphafold-server -f services/alphafold-server/Dockerfile .
#
# AlphaFold source is fetched from GitHub at a pinned SHA (see ALPHAFOLD_SHA
# below) — no opensource/ checkout required.  Databases and weights are NOT
# bundled — mount NAS at /data/models/alphafold/.
#
# Multistage layout
# -----------------
#   builder = micromamba env (Python + openmm/pdbfixer via conda-forge) +
#             uv pip extras (tensorflow, jax, framework) + AlphaFold source
#             cloned to /opt/alphafold-src.
#   runtime = minimal apt + conda env from builder + AlphaFold source +
#             server code.
#
# Base: nvidia/cuda 12.2 + cuDNN8, matching the official AlphaFold Dockerfile.
ARG CUDA=12.2.2
ARG BASE_IMAGE=nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2

# Pinned AlphaFold upstream. Bump SHA to upgrade; the build verifies that the
# remote actually checked out the requested commit so a moved-tag scenario
# can't silently change the image contents.
ARG ALPHAFOLD_REPO=https://github.com/google-deepmind/alphafold.git
ARG ALPHAFOLD_SHA=c77e5d2a8961d1a353632c462914ff0a32a950f6


# ===========================================================================
# Stage 1: builder — conda env + pip deps.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder
ARG CUDA

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl ca-certificates \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        hmmer kalign tzdata \
    && rm -rf /var/lib/apt/lists/*

# HH-suite from source.
RUN git clone --branch v3.3.0 --single-branch https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/af2/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
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
# Conda env: Python 3.12 + openmm + pdbfixer (conda-only packages).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n af2 -c conda-forge \
        python=3.12 \
        openmm=8.2.0 \
        pdbfixer=1.12.0

# ---------------------------------------------------------------------------
# AlphaFold source — clone at pinned SHA (treeless clone, then drop .git).
# Doing it inside the builder stage means git is already installed (above)
# and the runtime stage stays slim.
# ---------------------------------------------------------------------------
ARG ALPHAFOLD_REPO
ARG ALPHAFOLD_SHA
RUN set -eux; \
    git clone --filter=blob:none --no-checkout "$ALPHAFOLD_REPO" /opt/alphafold-src; \
    cd /opt/alphafold-src; \
    git checkout "$ALPHAFOLD_SHA"; \
    test "$(git rev-parse HEAD)" = "$ALPHAFOLD_SHA"; \
    rm -rf .git

# ---------------------------------------------------------------------------
# Pip deps via uv: AlphaFold requirements (minus openmm/pdbfixer) + JAX CUDA.
# Versions are hardcoded below (not read from requirements.txt) — kept in sync
# with /opt/alphafold-src/requirements.txt when bumping ALPHAFOLD_SHA.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/af2/bin/python \
        absl-py==1.0.0 \
        biopython==1.84 \
        dm-haiku==0.0.12 \
        "flax<0.8.6" \
        docker==5.0.0 \
        matplotlib==3.8.0 \
        ml-collections==0.1.0 \
        "pytest<8.5.0" \
        "setuptools<72.0.0" \
        tensorflow-cpu==2.16.1 \
    && uv pip install --python /opt/conda/envs/af2/bin/python \
        jax==0.4.26 \
        jaxlib==0.4.26+cuda12.cudnn89 \
        --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && uv pip install --python /opt/conda/envs/af2/bin/python \
        "nvidia-cuda-nvcc-cu12==12.3.*"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN find /opt/conda/envs/af2/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/af2/share/doc \
        /opt/conda/envs/af2/share/man \
        /opt/conda/envs/af2/share/locale \
    && find /opt/conda/envs/af2 -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/af2 -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/af2/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + AlphaFold source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}
ARG CUDA

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/af2/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        hmmer kalign wget ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HH-suite binaries from builder.
COPY --from=builder /opt/hhsuite /opt/hhsuite
RUN ln -s /opt/hhsuite/bin/* /usr/bin

# Conda env from builder.
COPY --from=builder /opt/conda/envs/af2 /opt/conda/envs/af2

# AlphaFold source — copied from the builder stage's clone.
COPY --from=builder /opt/alphafold-src /opt/alphafold
RUN wget -q -P /opt/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Fix SETUID for ldconfig (GPU visibility in non-root).
RUN chmod u+s /sbin/ldconfig.real
RUN ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /opt/conda/envs/af2/lib/libffi.so.7

# Server code — explicit file list so test/doc changes don't bust cache.
COPY services/alphafold-server/__init__.py \
     services/alphafold-server/settings.py \
     services/alphafold-server/models.py \
     services/alphafold-server/adapter.py \
     services/alphafold-server/tools.py \
     services/alphafold-server/uris.py \
     services/alphafold-server/app.py \
     services/alphafold-server/__main__.py \
     services/alphafold-server/VERSION \
     /opt/alphafold/server/

# Environment.
ENV ALPHAFOLD_ROOT=/opt/alphafold
ENV ALPHAFOLD_PYTHON=/opt/conda/envs/af2/bin/python
ENV ALPHAFOLD_JOBS_BASE_DIR=/data/alphafold_jobs
ENV ALPHAFOLD_DATA_DIR=/data/models/alphafold
ENV ALPHAFOLD_N_CPU=8
ENV ALPHAFOLD_MAX_CONCURRENT_JOBS=1
ENV TF_FORCE_UNIFIED_MEMORY=1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
ENV OPENMM_CPU_THREADS=8
ENV PYTHONPATH=/opt/alphafold

# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV ALPHAFOLD_KEEPALIVE_URL=https://fc-alphafold-fcizetpnjc.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV ALPHAFOLD_SESSION_HEADER_NAME=bioagent-session-id

WORKDIR /opt/alphafold

# ldconfig must run before GPU access.
RUN echo $'#!/bin/bash\nldconfig\nexec "$@"' > /opt/entrypoint.sh \
    && chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]

EXPOSE 9000
CMD ["/opt/conda/envs/af2/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000"]
