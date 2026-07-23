# Build from bioagent project root:
#   docker build --platform linux/amd64 -t haddock3-server -f services/haddock3-server/Dockerfile .
#
# Pre-build (once; re-run to bump the pinned upstream commit):
#   ./services/haddock3-server/scripts/vendor.sh
#
# HADDOCK3 source lives at services/haddock3-server/upstream/ (vendored at a
# pinned SHA). CPU-only Python; `pip install` compiles two small C/C++ helpers
# (contact_fcc, fast-rmsdmatrix) so build-essential is required.
#
# The CNS engine is license-gated and NOT baked in — it is staged to NAS at
# /data/models/haddock3/cns/cns (CNS_EXEC) and needs libgfortran5 at runtime.
# Docking + scoring require it; restraints utilities do not. See README (## CNS).

# Base: uv's pre-built python3.11 slim image (Debian bookworm + uv pre-installed),
# same choice as dockq-server — skips the `COPY --from=ghcr.io/astral-sh/uv` step
# and gives us apt for build-essential + libgfortran5.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster from China).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# build-essential: compiles HADDOCK3's contact_fcc (g++) + fast-rmsdmatrix (gcc)
#   during `uv pip install -e ./upstream`.
# libgfortran5: runtime shared lib for the externally-staged CNS fortran binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Create the venv up front so subsequent COPYs do not overlay it.
WORKDIR /opt/haddock3
RUN uv venv .venv --python python3.11

# Heavy deps first (large layer, rarely changes) to warm the build cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "numpy==2.*" scipy "pandas==2.*" "biopython==1.*"

# HADDOCK3 source — copied from vendored upstream/ into /opt/haddock3/ (overlay
# style so the .venv survives). Editable install pulls the rest of the deps +
# compiles the C helpers + exposes the haddock3* console scripts in .venv/bin.
COPY services/haddock3-server/upstream/ /opt/haddock3/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e .

# Shared bioagent service framework + remote-fetch deps (late layer so framework
# changes don't invalidate the haddock3 install layer above).
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/haddock3-server/__init__.py \
     services/haddock3-server/__main__.py \
     services/haddock3-server/VERSION \
     services/haddock3-server/app.py \
     services/haddock3-server/adapter.py \
     services/haddock3-server/configs.py \
     services/haddock3-server/inference.py \
     services/haddock3-server/models.py \
     services/haddock3-server/settings.py \
     services/haddock3-server/tools.py \
     /opt/haddock3/server/

# Build smoke: package imports + console scripts resolve + wrapper parses args.
# (Restraints path is CNS-free; we do NOT import haddock.core.defaults here since
# that would sys.exit without a CNS binary — that's expected only at dock time.)
RUN .venv/bin/python -c "import haddock; from importlib.metadata import version; print('haddock3', version('haddock3'))" \
    && test -x .venv/bin/haddock3 \
    && test -x .venv/bin/haddock3-score \
    && test -x .venv/bin/haddock3-restraints \
    && .venv/bin/python -m server --help >/dev/null

ENV PYTHONPATH=/opt/haddock3
ENV HADDOCK3_ROOT=/opt/haddock3
ENV HADDOCK3_PYTHON=/opt/haddock3/.venv/bin/python
ENV HADDOCK3_INFERENCE_SCRIPT=/opt/haddock3/server/inference.py
ENV HADDOCK3_JOBS_BASE_DIR=/data/haddock3_jobs
ENV HADDOCK3_WEIGHTS_DIR=/data/models/haddock3
# License-gated CNS engine, NAS-mounted (FC) / --bind (SIF). Docking + scoring
# read this; a missing file makes haddock3 exit cleanly at dock time.
ENV CNS_EXEC=/data/models/haddock3/cns/cns
# Output-sink: framework mirrors the job dir here when the gateway sends
# X-Bioagent-Oss-Prefix. Requires the data-plane OSS bucket mounted at /mnt/oss.
ENV HADDOCK3_OSS_OUTPUT_MOUNT=/mnt/oss
# Force UTF-8 (FC containers default to ascii locale → open() crashes on some PDBs).
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN mkdir -p /data/haddock3_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up polls to the same instance (requires FC console config too).
ENV HADDOCK3_SESSION_HEADER_NAME=bioagent-session-id
ENV PORT=9000
EXPOSE 9000
# WORKDIR /opt/haddock3 (set above) — CPython prepends CWD to sys.path so the
# server/ subdir imports as `server.app`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
