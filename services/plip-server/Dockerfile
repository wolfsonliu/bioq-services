# Build from bioagent project root:
#   docker build --platform linux/amd64 -t plip-server -f services/plip-server/Dockerfile .
#
# Pre-build (once; re-run to bump the pinned PLIP SHA):
#   ./services/plip-server/scripts/vendor.sh
#
# PLIP is CPU-only and rule-based — no CUDA, no model weights. Its algorithm
# dependencies (numpy / lxml / openbabel + PyMOL for visualization) are compiled
# extensions best obtained from Debian's apt (this mirrors upstream's official
# Dockerfile, which apt-installs the same stack and passes PLIP's full test
# suite). A uv venv created with --system-site-packages reuses those system
# packages while keeping the bioagent service framework isolated in the venv.

# Debian bookworm (stable): system python is 3.11.2 (NOT ubuntu jammy's 3.11.0rc1
# which breaks torch/str-int limits — see engineering guides). Same base family
# as upstream's `FROM debian:stable`.
FROM debian:stable-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# PyMOL (and openbabel tmp) need a writable HOME when run headless.
ENV HOME=/tmp
# Huawei Cloud pip mirror (project-wide convention, ~10x faster from China).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# Algorithm stack from apt (mirrors upstream PLIP Dockerfile):
#   pymol / python3-pymol  — visualization (.pse / .png), library mode, headless
#   python3-openbabel      — hydrogen bonds, aromaticity, SMILES (required always)
#   python3-lxml           — XML report writer
#   python3-numpy          — geometry
#   python3 / python3-venv — system CPython 3.11 that owns the dist-packages above
#   ca-certificates        — httpx / OSS TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv \
        pymol python3-pymol python3-openbabel python3-lxml python3-numpy \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv binary (fast installs) from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create the venv FROM the system python so --system-site-packages exposes the
# apt-installed openbabel / pymol / lxml / numpy under /usr/lib/python3/dist-packages.
WORKDIR /opt/plip
RUN uv venv .venv --python /usr/bin/python3 --system-site-packages

# Shared bioagent service framework + HTTP/remote-fetch deps into the venv.
# numpy/lxml/openbabel/pymol resolve from system site-packages (not reinstalled).
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.4" \
        "httpx>=0.28" \
        python-multipart \
        alibabacloud-oss-v2

# Vendored PLIP source (flat `plip/` package). Prerequisite: run scripts/vendor.sh.
COPY services/plip-server/upstream/plip /opt/plip/upstream/plip

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/plip-server/__init__.py \
     services/plip-server/__main__.py \
     services/plip-server/VERSION \
     services/plip-server/app.py \
     services/plip-server/adapter.py \
     services/plip-server/models.py \
     services/plip-server/settings.py \
     services/plip-server/tools.py \
     /opt/plip/server/

# PYTHONPATH lets the subprocess `import plip` under any CWD; WORKDIR /opt/plip
# makes `server` importable (CPython prepends CWD to sys.path).
ENV PYTHONPATH=/opt/plip/upstream
ENV PLIP_PYTHON=/opt/plip/.venv/bin/python
ENV PLIP_UPSTREAM_DIR=/opt/plip/upstream
ENV PLIP_JOBS_BASE_DIR=/data/plip_jobs
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir here. Requires the data-plane OSS bucket mounted at /mnt/oss in the FC
# console. Absent mount => no-op (NAS only).
ENV PLIP_OSS_OUTPUT_MOUNT=/mnt/oss
# FC session affinity header (also needs FC console config).
ENV PLIP_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/plip_jobs

# Build smoke: exercise the full openbabel-backed import chain + confirm pymol
# imports (visualization) — catches a broken apt/venv wiring at build time.
RUN PYTHONPATH=/opt/plip/upstream /opt/plip/.venv/bin/python -m plip.plipcmd --help > /dev/null \
    && /opt/plip/.venv/bin/python -c "import openbabel, pymol; print('openbabel + pymol OK')"

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
