# Build from bioagent project root:
#   docker build --platform linux/amd64 -t openbpmd-server \
#       -f services/openbpmd-server/Dockerfile .
#
# Pre-build (once; re-run to bump the pinned upstream SHA):
#   ./services/openbpmd-server/scripts/vendor.sh   # → upstream/
#
# Upstream (Gervasiolab/OpenBPMD) is MIT.
#
# GPU service. Primary entry is FC async task mode (/api/tasks/score); HPC
# apptainer CLI (`python -m server score`) is the long-run escape hatch for
# full 10x10 ns runs that can exceed FC's 24 h ceiling.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + openmm 8.1.1 CUDA + mdanalysis + mdtraj
#             + parmed + numpy + pandas + bioq-service-framework), plus the
#             vendored OpenBPMD source with patches/ applied.
#   runtime = lean: conda env + patched upstream + server code.
#
# Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 — provides the CUDA 12
# runtime libs OpenMM's CUDA platform loads via the host driver.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env + patched OpenBPMD source + framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# `patch` is needed to apply patches/*.patch onto the vendored upstream.
# No `git` — vendor.sh ran on the host; the image never clones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates patch \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/openbpmd/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for CN-friendly builds.
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
show_channel_urls: true
channel_priority: flexible
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# ---------------------------------------------------------------------------
# Create conda env. Only ~7 packages (all conda-forge) — a single solve stays
# well under the OOM threshold, no need to split into layers.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n openbpmd \
        python=3.10 pip \
        openmm=8.1.1 \
        mdanalysis=2.7.0 \
        mdtraj=1.10.0 \
        parmed=4.2.2 \
        numpy=1.26 \
        pandas=2.2

# `environment.yml` retained as the authoritative pin source (not consumed
# directly by the build — see the explicit create above).
COPY services/openbpmd-server/environment.yml /opt/openbpmd/environment.yml

# ---------------------------------------------------------------------------
# Vendored OpenBPMD source + patch (parametrize hardcoded MD lengths).
# ---------------------------------------------------------------------------
COPY services/openbpmd-server/upstream /opt/openbpmd/upstream
COPY services/openbpmd-server/patches /opt/openbpmd/patches
RUN patch -p1 -d /opt/openbpmd/upstream < /opt/openbpmd/patches/0001-parametrize-md-lengths.patch

# ---------------------------------------------------------------------------
# Framework — into the same conda env.
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/openbpmd/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# ---------------------------------------------------------------------------
# Build-time smoke #1 — filesystem sanity (vendor.sh + patch ran).
# ---------------------------------------------------------------------------
RUN set -e; \
    for f in /opt/openbpmd/upstream/openbpmd.py \
             /opt/openbpmd/upstream/LICENSE; do \
        [ -f "$f" ] || { echo "ERROR: $f missing; run scripts/vendor.sh"; exit 1; }; \
    done; \
    grep -q 'EQUIL_STEPS' /opt/openbpmd/upstream/openbpmd.py \
        || { echo "ERROR: patch not applied to openbpmd.py"; exit 1; }

# ---------------------------------------------------------------------------
# Build-time smoke #2 — simtk->openmm shim + vendored module import chain.
# (No GPU at build time, so we only assert import works, not CUDA presence.)
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/openbpmd/bin/python -c "\
import sys, types; \
import openmm, openmm.app, openmm.app.metadynamics, openmm.unit; \
simtk = types.ModuleType('simtk'); simtk.openmm = openmm; simtk.unit = openmm.unit; \
sys.modules['simtk'] = simtk; \
sys.modules['simtk.openmm'] = openmm; \
sys.modules['simtk.openmm.app'] = openmm.app; \
sys.modules['simtk.unit'] = openmm.unit; \
sys.modules['simtk.openmm.app.metadynamics'] = openmm.app.metadynamics; \
sys.path.insert(0, '/opt/openbpmd/upstream'); \
import openbpmd; \
assert hasattr(openbpmd, 'PROD_SIM_NS') and hasattr(openbpmd, 'EQUIL_STEPS'); \
print('smoke ok: openmm', openmm.version.version)"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN find /opt/conda/envs/openbpmd/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/openbpmd/share/doc \
        /opt/conda/envs/openbpmd/share/man \
    && find /opt/conda/envs/openbpmd/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/openbpmd -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/openbpmd -name '*.pyc' -delete


# ===========================================================================
# Stage 2: runtime — conda env + patched upstream + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/openbpmd/bin:$PATH

# LANG=C.UTF-8: FC containers default to an ascii locale; any upstream open()
# without encoding= that meets a UTF-8 byte would crash. See cookbook.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/openbpmd /opt/openbpmd

# Server code as `server` package — uvicorn imports `server.app:app`,
# `python -m server` triggers server/__main__.py.
COPY services/openbpmd-server/__init__.py \
     services/openbpmd-server/__main__.py \
     services/openbpmd-server/VERSION \
     services/openbpmd-server/app.py \
     services/openbpmd-server/adapter.py \
     services/openbpmd-server/inference.py \
     services/openbpmd-server/models.py \
     services/openbpmd-server/settings.py \
     services/openbpmd-server/tools.py \
     /opt/openbpmd/server/

WORKDIR /opt/openbpmd

# PYTHONPATH lets `server.app` / `server.__main__` import regardless of the
# runner's WORKDIR (Apptainer does not honor WORKDIR).
ENV PYTHONPATH=/opt/openbpmd
ENV OPENBPMD_ROOT=/opt/openbpmd
ENV OPENBPMD_PYTHON=/opt/conda/envs/openbpmd/bin/python
ENV OPENBPMD_INFERENCE_SCRIPT=/opt/openbpmd/server/inference.py
ENV OPENBPMD_UPSTREAM_DIR=/opt/openbpmd/upstream
ENV OPENBPMD_JOBS_BASE_DIR=/data/openbpmd_jobs
ENV OPENBPMD_PLATFORM=CUDA

# Output-sink: 当 gateway 传 X-Bioagent-Oss-Prefix 时，framework 把 job dir 镜像到此挂载点。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss。缺挂载 => no-op（仅 NAS）。
ENV OPENBPMD_OSS_OUTPUT_MOUNT=/mnt/oss

# Weights placeholder — OpenBPMD has no NN weights; NAS mount optional.
ENV OPENBPMD_WEIGHTS_DIR=/data/models/openbpmd
ENV OPENBPMD_MAX_CONCURRENT_JOBS=1
# Modern GPU service — FC async task endpoints enabled.
ENV OPENBPMD_TASK_ENDPOINTS_ENABLED=true
# FC session affinity: route a job's polls to the same instance (see
# 2026-06-10-fc-session-affinity.md). Without this the affinity header is a
# no-op and concurrent GPU jobs can't be scheduled onto parallel instances.
ENV OPENBPMD_SESSION_HEADER_NAME=bioagent-session-id

RUN mkdir -p /data/openbpmd_jobs

ENV PORT=9000
EXPOSE 9000
# HTTP mode is the container default. For CLI batch mode (HPC):
#   apptainer exec --nv image.sif python -m server score ...
CMD ["/opt/conda/envs/openbpmd/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
