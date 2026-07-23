# qligfep-server — Q + qligfep FEP/LIE full pipeline (HPC-primary).
#
# Build from bioagent project root:
#   docker build --platform linux/amd64 -t qligfep-server -f services/qligfep-server/Dockerfile .
#
# Prerequisites (one-time):
#   ./services/qligfep-server/scripts/vendor.sh   # vendor upstream/qligfep + upstream/Q6
#
# See engineering/decisions/2026-07-06-qligfep-server-design.md.
#
# Multistage layout
# -----------------
#   q6-builder    = nvidia/cuda -devel; compile Q6 (gfortran + OpenMPI). No CUDA
#                   build target exists in Q6 upstream, so only qdyn + qdynp.
#   conda-builder = micromamba conda env with the OpenFF stack. OpenFF is a
#                   conda-first ecosystem — openff-toolkit's charge backend
#                   (ambertools/antechamber) has NO PyPI wheel, so pip/uv install
#                   is unreliable. conda-forge is the officially supported path.
#   runtime       = lean: conda env + Q6 binaries + qligfep source + server code.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: q6-builder — compile Q6 (gfortran + MPI). No CUDA.
# ===========================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS q6-builder
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
# as missing, and RunFepRequest(device=gpu) fails with a clear rc=3
# "binary not found" from run_fep_cli._select_binary.  See design doc §10.
RUN mkdir -p /opt/Q6/bin \
    && make all COMP=gcc \
    && cp ../bin/qdyn ../bin/qprep ../bin/qfep ../bin/qcalc /opt/Q6/bin/ \
    && make clean \
    && make mpi COMP=gcc \
    && cp ../bin/qdynp /opt/Q6/bin/


# ===========================================================================
# Stage 2: conda-builder — OpenFF stack via conda-forge + service framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS conda-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# NB: no `git` in apt — vendor.sh ran on the host; the image never clones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/qligfep/bin:$PATH

# Mirror config (Huawei pip + TUNA conda) for CN-friendly builds.
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
show_channel_urls: true
# Strict priority + single channel (conda-forge) keeps the OpenFF solve fast
# and avoids cross-channel SAT explosion.
channel_priority: strict
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — OpenFF stack (all conda-forge):
#   - openff-toolkit + openff-interchange : ligand parameterization → offxml/top
#   - ambertools                          : antechamber AM1-BCC charge backend
#                                           (NO PyPI wheel — the reason we use
#                                            conda instead of uv/pip)
#   - rdkit                               : OpenFF toolkit wrapper (mol I/O)
#   - openmm                              : interchange dependency
#   - mdtraj + numpy                      : used by qligfep protprep.py etc.
# Python 3.11 — openff-toolkit 0.18+ requires >=3.11.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n qligfep -c conda-forge \
        python=3.11 \
        openff-toolkit \
        openff-interchange \
        ambertools \
        rdkit \
        openmm \
        mdtraj \
        numpy

# Build-time smoke: verify the OpenFF import chain resolves.
RUN /opt/conda/envs/qligfep/bin/python -c \
    "from openff.toolkit import ForceField, Molecule; \
     from openff.interchange import Interchange; \
     from rdkit import Chem; import mdtraj, numpy; \
     print('openff/rdkit/mdtraj import ok')"

# ---------------------------------------------------------------------------
# Shared service framework — separate layer so framework bumps don't
# invalidate the (heavy) conda env layer above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/qligfep/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# ---------------------------------------------------------------------------
# Trim dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN find /opt/conda/envs/qligfep/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/qligfep/share/doc \
        /opt/conda/envs/qligfep/share/man \
    && find /opt/conda/envs/qligfep/lib/python3.11/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + 2>/dev/null; \
    find /opt/conda/envs/qligfep -depth -type d \
        -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    find /opt/conda/envs/qligfep -name '*.pyc' -delete 2>/dev/null; true


# ===========================================================================
# Stage 3: runtime — conda env + Q6 binaries + qligfep source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda

# libgomp1 for numeric libs; openmpi runtime so qdynp resolves libmpi at run;
# ca-certificates for OSS / HTTP URI fetches.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 openmpi-bin libopenmpi3 \
    && rm -rf /var/lib/apt/lists/*

# Conda env + Q6 binaries from the builder stages.
COPY --from=conda-builder /opt/conda /opt/conda
COPY --from=q6-builder /opt/Q6/bin /opt/Q6/bin
ENV PATH=/opt/conda/envs/qligfep/bin:/opt/Q6/bin:$PATH

# Upstream qligfep (vendor.sh already applied the settings.py env shim).
COPY services/qligfep-server/upstream/qligfep /opt/qligfep-server/upstream/qligfep

# Server code as `server` package — uvicorn imports `server.app:app`.
# Explicit file list keeps the nested upstream/ + tests/ out of the image.
COPY services/qligfep-server/__init__.py \
     services/qligfep-server/__main__.py \
     services/qligfep-server/VERSION \
     services/qligfep-server/app.py \
     services/qligfep-server/adapter.py \
     services/qligfep-server/models.py \
     services/qligfep-server/settings.py \
     services/qligfep-server/tools.py \
     services/qligfep-server/vendor_shim.py \
     services/qligfep-server/cog_cli.py \
     services/qligfep-server/protprep_cli.py \
     services/qligfep-server/ligprep_cli.py \
     services/qligfep-server/setup_ligfep_cli.py \
     services/qligfep-server/setup_resfep_cli.py \
     services/qligfep-server/setup_lie_cli.py \
     services/qligfep-server/run_fep_cli.py \
     services/qligfep-server/analyze_fep_cli.py \
     services/qligfep-server/analyze_lie_cli.py \
     /opt/qligfep-server/server/

WORKDIR /opt/qligfep-server

# PYTHONPATH so `server` package imports under any CWD (Apptainer ignores WORKDIR).
ENV PYTHONPATH=/opt/qligfep-server
ENV QLIGFEP_ROOT=/opt/qligfep-server
ENV QLIGFEP_Q_BIN_DIR=/opt/Q6/bin
ENV QLIGFEP_UPSTREAM_DIR=/opt/qligfep-server/upstream/qligfep
ENV QLIGFEP_PYTHON=/opt/conda/envs/qligfep/bin/python
ENV QLIGFEP_JOBS_BASE_DIR=/data/qligfep_jobs
RUN mkdir -p /data/qligfep_jobs

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/qligfep/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
