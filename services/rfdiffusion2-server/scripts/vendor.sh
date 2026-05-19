#!/usr/bin/env bash
# Re-vendor RFdiffusion2 source from opensource/RFdiffusion2/ into upstream/.
# Run this whenever you pull new upstream changes.
#
# Excludes test/dev/data files per the vendor design doc; keeps all *.py +
# config/ + benchmark/input/ + envs/cuda124_env.yml + requirements_cuda124.txt.
# Weights (model_weights/) are NOT vendored here — use fetch_weights.sh.
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
SRC="$PROJECT_ROOT/opensource/RFdiffusion2"
DST="$PROJECT_ROOT/services/rfdiffusion2-server/upstream"

if [[ ! -d "$SRC/rf_diffusion" ]]; then
    echo "ERROR: $SRC/rf_diffusion not found." >&2
    echo "Make sure opensource/RFdiffusion2/ is checked out." >&2
    exit 1
fi

mkdir -p "$DST/envs"

# Source tree (excluding test/dev/data + model_weights).
rsync -a --delete \
    --exclude='test_data/' \
    --exclude='goldens/' \
    --exclude='dev/' \
    --exclude='exec/' \
    --exclude='datahub_pipelines/' \
    --exclude='benchmark/rotamer_library/' \
    --exclude='model_weights/' \
    --exclude='*.pkl' \
    --exclude='*.pse' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$SRC/rf_diffusion/" "$DST/rf_diffusion/"

# Env files (only CUDA 12.4 — others not used by FC deployment).
cp "$SRC/envs/cuda124_env.yml" "$DST/envs/"
cp "$SRC/envs/requirements_cuda124.txt" "$DST/envs/"

echo "Vendored to $DST"
du -sh "$DST"
echo
echo "Review with: git -C $PROJECT_ROOT status -- services/rfdiffusion2-server/upstream/"
