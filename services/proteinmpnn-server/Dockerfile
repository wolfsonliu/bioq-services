# Build from bioagent project root:
#   docker build --platform linux/amd64 -t proteinmpnn-server -f services/proteinmpnn-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/proteinmpnn-server/scripts/vendor.sh          # vendor upstream/
#   ./services/proteinmpnn-server/scripts/fetch_weights.sh   # AbMPNN weights
#
# ProteinMPNN upstream source + 3 of the 4 weight folders (vanilla / soluble
# / ca) come from services/proteinmpnn-server/upstream/, which vendor.sh
# populates at a pinned SHA — Dockerfile does no network access.
# AbMPNN_model_weights/ is NOT in the upstream repo and is pre-staged via
# fetch_weights.sh.
# Total weight bake-in ~1 GB.
#
# Base: same nvidia/cuda 12.4.1 image as ppiflow-server / genie3-server to share CI
# layer cache. CUDA 12.x runtime is forward-compatible with the torch wheels we install.
# `-runtime` (not `-devel`) — ProteinMPNN install path is wheel-only (torch + numpy +
# framework), no nvcc / CUDA headers required. Save ~3.6 GB vs `-devel`.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror via uv (project-wide convention, ~10× faster from
# China than download.pytorch.org which has no China CDN).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv binary (matches rfantibody-server's version pin).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Create isolated venv up front (mirrors rfantibody-server's .venv layout —
# CMD references `.venv/bin/python`). ProteinMPNN has no pyproject.toml of
# its own, so we drive installs with `uv pip install` instead of `uv sync`.
WORKDIR /opt/proteinmpnn
RUN uv venv .venv --python python3.10

# Torch first (heavy ~2 GB layer, rarely changes). PyPI's default linux torch
# wheel is `+cu121` (CUDA 12.1 bundled), which works on the 12.4 base runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python torch==2.4.1 numpy

# ProteinMPNN source + weights — copied from the vendored upstream tree at
# services/proteinmpnn-server/upstream/.  Run vendor.sh first to populate it.
# We skip non-runtime dirs (training/, inputs/, outputs/, colab_notebooks/,
# examples/) by cherry-picking only what's needed.
# Weights tracked in upstream git: vanilla / soluble / ca.
COPY services/proteinmpnn-server/upstream/vanilla_model_weights/ ./vanilla_model_weights/
COPY services/proteinmpnn-server/upstream/soluble_model_weights/ ./soluble_model_weights/
COPY services/proteinmpnn-server/upstream/ca_model_weights/      ./ca_model_weights/
COPY services/proteinmpnn-server/upstream/protein_mpnn_run.py    ./protein_mpnn_run.py
COPY services/proteinmpnn-server/upstream/protein_mpnn_utils.py  ./protein_mpnn_utils.py
COPY services/proteinmpnn-server/upstream/helper_scripts/        ./helper_scripts/

# AbMPNN_model_weights/ — upstream-untracked; pre-staged via fetch_weights.sh.
COPY services/proteinmpnn-server/AbMPNN_model_weights/  ./AbMPNN_model_weights/

# Shared bioagent service framework + remote-fetch deps (late layer).
# COPY (not --mount=type=bind) so a framework SOURCE change busts this layer's
# cache: BuildKit does not include bind-mount content in the cache key, which
# would otherwise ship a stale framework even after the framework is updated.
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/proteinmpnn-server/__init__.py \
     services/proteinmpnn-server/__main__.py \
     services/proteinmpnn-server/VERSION \
     services/proteinmpnn-server/app.py \
     services/proteinmpnn-server/adapter.py \
     services/proteinmpnn-server/models.py \
     services/proteinmpnn-server/settings.py \
     services/proteinmpnn-server/tools.py \
     /opt/proteinmpnn/server/

ENV PYTHONPATH=/opt/proteinmpnn
ENV PROTEINMPNN_ROOT=/opt/proteinmpnn
ENV PROTEINMPNN_WEIGHTS_DIR=/opt/proteinmpnn
ENV PROTEINMPNN_JOBS_BASE_DIR=/data/proteinmpnn_jobs
RUN mkdir -p /data/proteinmpnn_jobs

# Output-sink: when the gateway passes X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console (storage config). Absent mount => no-op (NAS-only).
ENV PROTEINMPNN_OSS_OUTPUT_MOUNT=/mnt/oss


# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV PROTEINMPNN_KEEPALIVE_URL=https://fc-proteinmpnn-pahlhbttzb.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV PROTEINMPNN_SESSION_HEADER_NAME=bioagent-session-id
ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/proteinmpnn (set above) — CPython prepends CWD to sys.path,
# making the `server/` subdir importable as `server.app`. Mirrors rfantibody-
# server / ppiflow-server layout.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
