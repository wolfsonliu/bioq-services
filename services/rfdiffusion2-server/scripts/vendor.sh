#!/usr/bin/env bash
# Re-vendor RFdiffusion2 source into services/rfdiffusion2-server/upstream/ at
# a pinned SHA, so `docker build` does no network access to github and does NOT
# depend on a local opensource/RFdiffusion2/ checkout (the old path).
#
#   ./services/rfdiffusion2-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   RFDIFFUSION2_REPO=https://ghproxy.cn/https://github.com/RosettaCommons/RFdiffusion2.git \
#       ./services/rfdiffusion2-server/scripts/vendor.sh
#
# Excludes test/dev/data files per the vendor design doc; keeps all *.py +
# config/ + benchmark/input/ + envs/cuda124_env.yml + requirements_cuda124.txt.
# Weights (model_weights/) are NOT vendored here — use fetch_weights.sh.
#
# RFdiffusion2 is not a pip package — upstream's install pattern is
# `export PYTHONPATH=/path/to/RFdiffusion2` so every top-level dir is
# importable as a sibling. rf_diffusion/__init__.py imports rf2aa at module
# load, rf2aa pulls from ipd at module load, and individual modules pull
# from openfold/ and se3_flow_matching/ too. Four siblings (rf2aa, ipd,
# openfold, se3_flow_matching) must be vendored alongside rf_diffusion/.
#
# Why vendor `ipd` instead of `pip install` from github: baker-laboratory/ipd
# HEAD has progressed past the version RFdiffusion2 was built against and
# now imports `evn` at module load, which isn't in the conda env. Vendoring
# the contemporary copy from the pinned RFdiffusion2 checkout's ipd/ avoids
# the version skew. (`lib/` and `fused_mpnn/` are not imported by rf_diffusion
# at runtime, so they're skipped.)
set -euo pipefail

RFDIFFUSION2_REPO="${RFDIFFUSION2_REPO:-https://github.com/RosettaCommons/RFdiffusion2.git}"
RFDIFFUSION2_SHA="${RFDIFFUSION2_SHA:-d365cbf4db3958814a9f8e4f6f94fa309dfebc2b}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/rfdiffusion2-server/upstream"
TMP="$(mktemp -d -t rfdiffusion2-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

# Clone at the pinned SHA directly from GitHub — no dependency on the old
# local opensource/RFdiffusion2/ checkout. --filter=blob:none defers blob
# download until checkout, keeping the initial fetch light on CN networks.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$RFDIFFUSION2_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

git -C "$TMP/repo" checkout "$RFDIFFUSION2_SHA"
actual="$(git -C "$TMP/repo" rev-parse HEAD)"
[[ "$actual" = "$RFDIFFUSION2_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $RFDIFFUSION2_SHA)" >&2
    exit 1
}
rm -rf "$TMP/repo/.git"

SRC="$TMP/repo"

if [[ ! -d "$SRC/rf_diffusion" ]]; then
    echo "ERROR: $SRC/rf_diffusion not found after clone." >&2
    echo "Check RFDIFFUSION2_REPO / RFDIFFUSION2_SHA point at RosettaCommons/RFdiffusion2." >&2
    exit 1
fi

mkdir -p "$DST/envs"

# Truly universal junk — safe to drop from every sibling we vendor.
#
# IMPORTANT: rsync exclude patterns without a leading slash match at ANY
# depth. So a pattern like 'dev/' would strip *both* an upstream top-level
# `dev/` *and* nested packages named `dev` (e.g. ipd/dev/, rf_diffusion/dev/).
# The latter two are runtime-required, so 'dev/' is NOT in this list. Anchor
# RFdiffusion2-specific top-level junk to the rf_diffusion rsync below.
COMMON_EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.pkl'
    --exclude='*.pse'
    --exclude='tests/'
    --exclude='test_pickles/'
    --exclude='media/'
    --exclude='archived/'
)

# rf_diffusion/ — drop RFdiffusion2-specific top-level junk (anchored with
# leading slash so we don't accidentally strip rf_diffusion/dev/, which has
# `idealize_backbone` and friends that run_inference.py imports) plus the
# bundled weights (separate `weights/` flow).
rsync -a --delete \
    "${COMMON_EXCLUDES[@]}" \
    --exclude='/test_data/' \
    --exclude='/goldens/' \
    --exclude='/exec/' \
    --exclude='/datahub_pipelines/' \
    --exclude='/benchmark/rotamer_library/' \
    --exclude='/model_weights/' \
    --exclude='/third_party_model_weights/' \
    "$SRC/rf_diffusion/" "$DST/rf_diffusion/"

# Sibling Python packages on PYTHONPATH alongside rf_diffusion/.
#   rf2aa            — imported at rf_diffusion package load
#   ipd              — imported transitively by rf2aa at module load. Extra
#                      excludes drop the 98 MB spacegroup_data.pickle (only
#                      used by ipd.sym.xtal, which the AME / binder /
#                      unconditional configs do not reach) and ipd/data/tests.
#   openfold         — namespace package (no top-level __init__.py); imported
#                      via `from openfold.utils import ...` etc.
#   se3_flow_matching — namespace package; imported via
#                       `from se3_flow_matching.data import ...`
for sibling in rf2aa openfold se3_flow_matching; do
    if [[ ! -d "$SRC/$sibling" ]]; then
        echo "ERROR: $SRC/$sibling not found — required sibling package." >&2
        exit 1
    fi
    rsync -a --delete \
        "${COMMON_EXCLUDES[@]}" \
        "$SRC/$sibling/" "$DST/$sibling/"
done

# ipd — extra exclude for the 98 MB crystallography pickle. The universal
# tests/ exclude already drops ipd/data/tests/.
#
# The 94 MB ipd/data/spacegroup_data.pickle is excluded to keep vendor size
# down. However ipd.sym.xtal.spacegroup_deriveddata is imported transitively
# at module load (rf2aa → ipd.sym) and calls load_package_data('spacegroup_data')
# which looks in ipd/data/. A tiny stub (.xz, ~1 KB) lives in ipd/dev/data/
# containing only P1 — enough to pass module-level KeyError checks. We copy
# that stub into ipd/data/ so package_data_path() finds it.
if [[ ! -d "$SRC/ipd" ]]; then
    echo "ERROR: $SRC/ipd not found — required sibling package." >&2
    exit 1
fi
rsync -a --delete \
    "${COMMON_EXCLUDES[@]}" \
    --exclude='data/spacegroup_data.pickle' \
    "$SRC/ipd/" "$DST/ipd/"

# Place the spacegroup stub where load_package_data() can find it.
if [[ -f "$SRC/ipd/dev/data/spacegroup_data.pickle.xz" ]]; then
    cp "$SRC/ipd/dev/data/spacegroup_data.pickle.xz" "$DST/ipd/data/spacegroup_data.pickle.xz"
fi

# Top-level .py modules imported by rf_diffusion at runtime.
#   paths.py — `from paths import evaluate_path` in model_runners.py; resolves
#              REPO_ROOT placeholder in Hydra config paths. __file__-based, so
#              repo_root correctly becomes upstream/ after vendoring.
cp "$SRC/paths.py" "$DST/paths.py"

# Env files (only CUDA 12.4 — others not used by FC deployment).
cp "$SRC/envs/cuda124_env.yml" "$DST/envs/"
cp "$SRC/envs/requirements_cuda124.txt" "$DST/envs/"

# ---------------------------------------------------------------------------
# Patch: SE3Transformer NVTX compatibility.
#
# PyTorch builds without libnvToolsExt (common in conda envs) define
# torch.cuda.nvtx.range as a pure-Python stub that only raises at CALL
# time, not import time. The four SE3Transformer files that use it would
# crash during the first model forward pass.
#
# Fix: drop a small nvtx_compat.py shim next to the SE3Transformer model
# package (probes NVTX at import time, falls back to nullcontext), then
# rewrite the import lines to use it.
# ---------------------------------------------------------------------------
SE3_MODEL="$DST/rf2aa/SE3Transformer/se3_transformer/model"
PATCHES="$PROJECT_ROOT/services/rfdiffusion2-server/patches"

# Place the shim in both model/ and model/layers/ so the relative import
# `from .nvtx_compat import nvtx_range` resolves from either package.
cp "$PATCHES/nvtx_compat.py" "$SE3_MODEL/nvtx_compat.py"
cp "$PATCHES/nvtx_compat.py" "$SE3_MODEL/layers/nvtx_compat.py"

for f in \
    "$SE3_MODEL/basis.py" \
    "$SE3_MODEL/layers/attention.py" \
    "$SE3_MODEL/layers/norm.py" \
    "$SE3_MODEL/layers/convolution.py"
do
    sed -i 's|^from torch\.cuda\.nvtx import range as nvtx_range|from .nvtx_compat import nvtx_range|' "$f"
done

echo "Vendored to $DST"
du -sh "$DST"
du -sh "$DST"/{rf_diffusion,rf2aa,ipd,openfold,se3_flow_matching,envs} 2>/dev/null
echo
echo "Review with: git -C $PROJECT_ROOT status -- services/rfdiffusion2-server/upstream/"
