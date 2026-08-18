# Dockerfile — build skeleton

English | [中文](dockerfile.zh.md)

> ← Back to the [Adding a service cookbook overview](./index.md)

This page covers `vendor.sh` / `fetch_weights.sh` / the uv venv and conda/micromamba Dockerfile
skeletons / the wrapper vs patch decision for modifying upstream. The pitfalls of wrapping
conda-based upstream are in [conda-pitfalls](./conda-pitfalls.md).

### 8. `services/<svc>/Dockerfile`

#### 8.0 Prerequisite: vendor.sh + fetch_weights.sh (must read)

**New convention (from 2026-06)**: the Docker build no longer clones upstream and no longer bakes
weights. Both are done by host scripts before the build:

| Step | Script | Purpose |
|---|---|---|
| 1. Vendor upstream source | `scripts/vendor.sh` | `git clone` the upstream repo at a pinned SHA into `services/<svc>/upstream/` |
| 2. (optional) Download weights | `scripts/fetch_weights.sh` | download into `services/<svc>/weights/` or directly to NAS (`WEIGHTS_DST=` env) |

**Why**:

- ✅ Zero github network access at build time → stable in-region builds (not affected by flaky TLS)
- ✅ Service self-containment (`services/<svc>/` holds upstream + scripts + framework wrapper)
- ✅ The upstream SHA is written explicitly in the script, visible + retried + verified
- ✅ The image holds only the code stack → shrinks to 1.5-3 GB (vs 5-18 GB), FC pulls faster, weights version independently

#### 8.1 vendor.sh template

```bash
#!/usr/bin/env bash
# Vendor the upstream <name> source into services/<svc>/upstream/ at a
# pinned SHA, so `docker build` does no network access.
#
#   ./services/<svc>/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   <NAME>_REPO=https://ghproxy.cn/https://github.com/<owner>/<repo>.git \
#       ./services/<svc>/scripts/vendor.sh
#
# To bump the upstream pin, edit <NAME>_SHA below.

set -euo pipefail

<NAME>_REPO="${<NAME>_REPO:-https://github.com/<owner>/<repo>.git}"
<NAME>_SHA="${<NAME>_SHA:-<40-char-sha>}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/<svc>/upstream"
TMP="$(mktemp -d -t <svc>-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# retry 5 times: CN build machines get occasional GnuTLS TLS reset
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "${<NAME>_REPO}" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "${<NAME>_SHA}"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "${<NAME>_SHA}" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected ${<NAME>_SHA})" >&2
    exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored ${<NAME>_REPO} @ ${<NAME>_SHA}"
echo "  -> $DST"
du -sh "$DST"
```

**Multi-upstream** (e.g. promera-server needs tinyprot + promera + LigandMPNN): extract the
`vendor_one()` function and clone serially. Full example:
[promera-server/scripts/vendor.sh](../../services/promera-server/scripts/vendor.sh).

Add `services/<svc>/upstream/` to the repo-root `.gitignore`.

#### 8.2 fetch_weights.sh template (if the service needs local weights)

```bash
#!/usr/bin/env bash
# Download <svc> weights.
#
# Weights are no longer baked into the image — they live on NAS (FC) or via
# apptainer --bind (SIF). This script only downloads to a staging dir; production
# deploys rsync them to NAS / HPC scratch afterwards.
#
# Default (local staging):
#   ./services/<svc>/scripts/fetch_weights.sh
#       → services/<svc>/weights/
#
# Download straight to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/<svc> \
#       ./services/<svc>/scripts/fetch_weights.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
mkdir -p "$DST"

# wget -c supports resume
for f in <list-of-files>; do
    wget -c -O "$DST/$f" "${BASE_URL}/$f"
done
```

##### Two common weight-externalization pitfalls

**Pitfall 1: `args.yaml` / `config.json` next to the checkpoint dir must be stashed too**

Most pytorch-lightning / in-house training frameworks keep a copy of the training hyperparameters
(`args.yaml` / `config.json` / `hparams.yaml`) **next to** the checkpoint dir; the inference-side
model loader **reads it first** to know how to construct the network. DiffDock-PP, boltz, and
DeepRank-Ab all follow this pattern.

`fetch_weights.sh` must cp these auxiliary files to NAS too, and the `/healthz/detail` probe should
also list them in its `expected` dict:

```bash
# e.g. DiffDock-PP fold_0 layout
CKPT_DIR="$SRC/large_model_dips/fold_0"
DST_DIR="$DST/large_model_dips/fold_0"
mkdir -p "$DST_DIR"
cp "$CKPT_DIR"/model_best_*.pth "$DST_DIR/"    # main checkpoint
cp "$SRC/large_model_dips/args.yaml" "$DST/large_model_dips/"  # training hyperparams — don't forget!
```

Missing `args.yaml` makes the service fail at **first inference** with "missing key in checkpoint" or
"unexpected keyword argument", while build / import / healthz are all green — very hard to diagnose.

**Pitfall 2: upstream pulls auxiliary weights via `torch.hub` / `transformers` AutoModel (ESM / T5 / etc.)**

If the upstream pipeline contains `torch.hub.load("facebookresearch/esm:main", ...)` or
`AutoModel.from_pretrained("...")`, the container **will attempt a network download at runtime** — FC
egress is restricted and SIF often has no network, so it must be pre-cached to NAS. See
deeprank-ab-server (ESM-2) and alphafold-server (parameter tarball) for reference:

```bash
# add a section to fetch_weights.sh: stash ESM-2 into the cache layout torch.hub expects
mkdir -p "$DST/esm_cache/hub/checkpoints"
wget -c -O "$DST/esm_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt" \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
# the upstream torch.hub API also needs hubconf.py — stash a copy too
git clone --depth 1 https://github.com/facebookresearch/esm.git \
    "$DST/esm_cache/hub/facebookresearch_esm_main"
```

Point `TORCH_HOME` / `HF_HOME` / `TRANSFORMERS_CACHE` at that NAS subdir from `settings.py` /
`Dockerfile` so runtime takes the offline cache branch:

```python
# append to settings.py
torchhub_dir: Path = Field(default=Path("/data/models/<svc>/esm_cache"))
```

```dockerfile
# append to the Dockerfile (make the offline cache visible to upstream libraries)
ENV TORCH_HOME=/data/models/<svc>/esm_cache
ENV HF_HOME=/data/models/<svc>/hf_cache
```

List these auxiliary checkpoints in the `/healthz/detail` probe's `expected` too:

```python
expected = {
    "score_ckpt": settings.weights_dir / "large_model_dips/fold_0/model_best.pth",
    "score_args": settings.weights_dir / "large_model_dips/args.yaml",
    "esm2_weight": settings.weights_dir / "esm_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt",
}
```

#### 8.3 Dockerfile

The default is the "uv venv + Huawei mirror" skeleton. For other scenarios (multi-stage build /
system Python / conda deps) see "Conda / micromamba alternative skeleton" below.

```dockerfile
# Build from the bioq-services repo root:
#   docker build --platform linux/amd64 -t <svc> -f services/<svc>/Dockerfile .
#
# Prerequisites (one-time; re-run to bump SHA / weights):
#   ./services/<svc>/scripts/vendor.sh           # vendor upstream/
#   ./services/<svc>/scripts/fetch_weights.sh    # download to weights/, then rsync to NAS
#
# Upstream source is injected from services/<svc>/upstream/ (vendor.sh artifacts);
# weights are loaded from NAS at /data/models/<svc>/ (not baked into the image).
# SIF / HPC inject them via `apptainer run --bind /scratch/models/<svc>:/data/models/<svc>`.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04   # pick the CUDA major; GPU algorithms use runtime, not devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror — required for in-region builds; the default PyPI torch wheel already bundles CUDA 12.x.
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# Note: **usually do NOT install git** in apt — vendor.sh runs on the host, the image doesn't need it.
# **Exception**: if a later `uv pip install` has a `"<pkg> @ git+https://..."`
# VCS specifier (e.g. openfold, dllogger, a github fork), you must add `git` to the apt
# list — uv 0.11+ no longer bundles libgit2 and shells out to the `git` CLI at runtime.
# The missing-git failure signal is: "Git executable not found".

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# create the venv under the algorithm dir — uv venv must run before COPYing source;
# Docker COPY won't delete an existing .venv/
WORKDIR /opt/<svc>
RUN uv venv .venv --python python3.10

# install heavy deps first (torch / numpy / other heavy wheels) — this layer rarely changes, so cache hits are high
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python torch numpy

# upstream algorithm source — **COPY from vendor.sh artifacts**, no in-image git clone
# (if upstream is a Python package, you can editable-install it)
COPY services/<svc>/upstream /opt/<svc>/upstream
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e ./upstream

# shared service framework + remote-fetch deps — a separate layer so framework changes
# don't invalidate the algorithm-layer cache
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/<svc>/ /opt/<svc>/server/

# ---- Weights: NAS mount, **do NOT COPY into the image** ----
# /data/models/<svc>/ is auto-mounted by FC; SIF / HPC inject it via --bind.
# When weights are missing the /healthz/detail probe reports weights_loaded=false (see app.py healthz_detail).

# Settings via env (see services/<svc>/settings.py).
# PYTHONPATH makes the `server` package importable from any CWD (Docker's WORKDIR
# covers Docker itself, but Apptainer/Singularity ignores WORKDIR).
ENV PYTHONPATH=/opt/<svc>
ENV <SVC>_ROOT=/opt/<svc>
ENV <SVC>_JOBS_BASE_DIR=/data/<svc>_jobs
ENV <SVC>_WEIGHTS_DIR=/data/models/<svc>   # keep in sync with the settings.py default
RUN mkdir -p /data/<svc>_jobs

# ---- Output-sink (required for gateway-invoked services) ----
# The gateway sends X-Bioagent-Oss-Prefix on every run; after the job finishes the framework
# mirrors the whole NAS job dir to <this-mount>/users/<user>/<job_id>/ and writes results.zip,
# which the gateway uses to serve the 302 download from OSS. The framework default is already
# /mnt/oss; it's declared explicitly here for self-documentation.
# Mount the data-plane OSS bucket to /mnt/oss in the FC console (see "FC console config"); a
# missing mount => no-op (NAS only).
ENV <SVC>_OSS_OUTPUT_MOUNT=/mnt/oss

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
```

Key constraints:

- **`-runtime` not `-devel`** — algorithms run on prebuilt wheels and don't need nvcc / CUDA headers, saving ~3.6 GB
- **Usually don't install git in apt** — vendor.sh runs on the host, the image doesn't need it. Installing it misleads future contributors into thinking an in-build clone is allowed. **Exception**: `uv pip install "<pkg> @ git+https://..."` (openfold / dllogger / github fork, etc.) needs the `git` CLI, in which case you **must** apt-install git — uv 0.11+ no longer bundles libgit2, and the missing-git error is "Git executable not found" (see the [diffdock-server Dockerfile](../../services/diffdock-server/Dockerfile) for reference)
- **WORKDIR must be the parent of `server/`** — otherwise `server.app` import fails under Docker
- **`ENV PYTHONPATH=<workdir>` is required** — Apptainer/Singularity ignores WORKDIR, so PYTHONPATH must make the `server` package importable from any CWD
- **`uv venv` before `COPY services/<svc>/upstream`** — so `.venv/` isn't overwritten by the source
- **CMD uses the absolute `.venv/bin/python` path** — no dependency on PATH
- **Add `--mount=type=cache,target=/root/.cache/uv` to every RUN** — speeds up rebuilds; the cache isn't written as an image layer
- **Do not `COPY services/<svc>/weights/`** — weights live on NAS. If you must bake small weights (< 100 MB and no version-management need), it's allowed as an exception, but state the reason in a Dockerfile comment

#### 8.4 When upstream needs modifying: wrapper vs patch decision

Upstream code almost always needs "a small tweak" to plug into the service framework (change
argparse, expose hard-coded paths, disable training-only wandb branches, etc.). Three approaches
coexist in this project today; picking the wrong one makes every vendor upgrade → patch rebase
painful. **Decision criteria**:

| Approach | Typical scenario | Pros | Cons | Example |
|---|---|---|---|---|
| **A. `server/inference.py` wrapper** (preferred) | need to expose parameterized paths / add custom argparse / glue front and back (data-layout shim / result post-processing) | 0 modification to upstream; the wrapper is fully auditable under `services/<svc>/` | one extra file; the wrapper must understand the upstream entry signature | [diffusion-hopping](../../services/diffusion-hopping-server/inference.py), [turbohopp](../../services/turbohopp-server/inference.py), [drughive](../../services/drughive-server/) |
| **B. `patches/*.patch` + Dockerfile apply** | multiple small upstream changes (sed is too brittle / the wrapper can't cover source-internal paths), and the changes are systematic and cross-file | explicit patches, reviewable via `git diff` | upstream SHA upgrades require rebasing patches; more patches = more brittle | [genie3-server/patches/](../../services/genie3-server/patches/) |
| **C. runtime monkey-patch** (`importlib` + attribute replacement) | an upstream internal function has an environment-related bug (e.g. `os.getenv` expecting a file in the CWD); and the source **can't** be patched (e.g. upstream uses a binary that depends on `__file__`-relative paths) | upstream source untouched, patch logic is testable in server code | one extra Python layer; the patch depends on the upstream module structure and breaks on refactor | [deeprank-ab-server/run_inference.py](../../services/deeprank-ab-server/) |

**Default to A**. Upgrade to B only when "parameterization + pre/post-processing" can't cover the deep
changes, and use C only when "the upstream code physically can't be moved" (e.g. relative paths of a
binary resource).

**Common wrapper template for A**:

```python
# services/<svc>/inference.py
import argparse
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    # ... parameterize upstream's hard-coded paths / disable training-only options
    return p.parse_args()

def validate(args):
    # weight / input existence + suffix checks → fail early with a clean error at the tail of stderr
    ...

def main():
    args = parse_args()
    validate(args)
    # **critical: re-import upstream modules must happen after validate()**, so that a missing
    # path/weight doesn't surface to the user as "ModuleNotFoundError: upstream lib missing CUDA
    # extension" instead of "checkpoint not found: ..."
    from <upstream_pkg> import ...
    ...

if __name__ == "__main__":
    sys.exit(main())
```

**Deferred import is the core of A**: upstream libraries often have heavy imports (pytorch + PyG +
rdkit + openbabel); if the `import` sits at the top of the file, any input-validation error pays a
20-30s import cost first. Put the import after `validate()` and a bad argument errors back instantly.

**How to do B** (put it in the Dockerfile, not vendor.sh):

```dockerfile
COPY services/<svc>/upstream /opt/<svc>/upstream
COPY services/<svc>/patches /tmp/<svc>-patches
RUN cd /opt/<svc>/upstream && for p in /tmp/<svc>-patches/*.patch; do \
        echo "Applying $p"; patch -p1 < "$p"; \
    done && rm -rf /tmp/<svc>-patches
```

Do not apply patches in vendor.sh — then changing a patch forces a re-vendor (slow + network). Putting
it in the Dockerfile makes patch iteration trigger only one layer rebuild.

#### Conda / micromamba alternative skeleton

When upstream dependencies **must** be managed by conda (e.g. PyTorch + PyG combos, special CUDA
version constraints, conda-only packages in `environment*.yml`), use **micromamba + multistage**
instead of uv venv. Typical scenarios: DeepRank-Ab (Python 3.9 + PyTorch 2.0.1 + CUDA 11.8 + PyG),
PPIFlow. Conda mirror mapping has been consolidated into `deploy/conda/mirrors.condarc` (COPYed and
`cat >>`-appended to each `.condarc` at build time); after changing a Dockerfile run
`python3 scripts/check_conda_mirrors.py` for regression — see the conda-mirror consolidation design
doc `docs/specs/2026-08-18-conda-mirror-consolidation-design.md`.

```dockerfile
# ---- builder ----
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Huawei pip mirror + the shared conda mirror mapping (deploy/conda/mirrors.condarc — single source).
# Do not inline TUNA/Ali conda mirror URLs in individual Dockerfiles; change the one file instead.
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
show_channel_urls: true
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# micromamba + conda env from the vendored upstream yml
# (vendor.sh must have run on the host first; artifacts are in services/<svc>/upstream/)
RUN ... install micromamba ...
COPY services/<svc>/upstream/environment*.yml /tmp/
RUN micromamba env create -n <env-name> -f /tmp/environment*.yml

# additional pip deps not in the yml
RUN micromamba run -n <env-name> pip install anarci  # example

# algorithm source + sed patches (if any)
COPY services/<svc>/upstream /opt/<algo>
RUN sed -i 's/NUM_WORKERS = 96/NUM_WORKERS = 8/' /opt/<algo>/scripts/inference.py  # example

# Framework
COPY framework /tmp/service-framework
RUN micromamba run -n <env-name> pip install "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# ---- runtime ----
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY --from=builder /opt/conda/envs/<env-name> /opt/conda/envs/<env-name>
COPY --from=builder /opt/<algo> /opt/<algo>
COPY services/<svc>/ /opt/<algo>/server/

WORKDIR /opt/<algo>
ENV PYTHONPATH=/opt/<algo>
ENV PATH="/opt/conda/envs/<env-name>/bin:$PATH"
ENV <SVC>_PYTHON=/opt/conda/envs/<env-name>/bin/python
ENV <SVC>_WEIGHTS_DIR=/data/models/<svc>   # NAS mount, not baked
ENV <SVC>_OSS_OUTPUT_MOUNT=/mnt/oss        # output-sink mount point (see the uv skeleton comment)
# ... other <SVC>_ env vars ...

# **required**: FC containers have LANG unset by default, so Python's locale.getpreferredencoding()
# returns 'ascii'. Any upstream code that reads a file containing UTF-8 characters via `open()`
# without encoding= crashes with UnicodeDecodeError. Set C.UTF-8 to disable the whole class of
# problems. See conda-pitfalls.md (common pitfalls when wrapping conda-based upstream).
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
```

**When to choose conda**: (1) upstream ships an `environment*.yml` that includes pytorch/pyg/nvidia
channels; (2) Python < 3.10 (the uv wheel ecosystem is incomplete); (3) deps on conda-only packages
(certain bioconda tools). In all other cases prefer the uv venv skeleton.