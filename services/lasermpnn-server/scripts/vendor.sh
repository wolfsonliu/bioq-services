#!/usr/bin/env bash
# Vendor the upstream LASErMPNN source into services/lasermpnn-server/upstream/
# at a pinned SHA, so `docker build` does no network access. Run once before
# each build (and again to upgrade):
#
#   ./services/lasermpnn-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   LASERMPNN_REPO=https://ghproxy.cn/https://github.com/polizzilab/LASErMPNN.git \
#       ./services/lasermpnn-server/scripts/vendor.sh
#
# To bump the upstream pin, edit LASERMPNN_SHA below.
#
# The big model_weights/ (~260 MB) are EXCLUDED — they load from NAS at runtime
# (see scripts/fetch_weights.sh). The small files/*.pt reference tensors ARE
# vendored (inference loads them relative to the package).

set -euo pipefail

LASERMPNN_REPO="${LASERMPNN_REPO:-https://github.com/polizzilab/LASErMPNN.git}"
LASERMPNN_SHA="${LASERMPNN_SHA:-5df210fced6764d83f01425d1fc4319a22b70c2a}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/lasermpnn-server/upstream"
TMP="$(mktemp -d -t lasermpnn-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$LASERMPNN_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$LASERMPNN_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$LASERMPNN_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $LASERMPNN_SHA)" >&2
    exit 1
}
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# Exclude big/unused dirs: model_weights (NAS, ~260 MB), databases (training
# splits), images/notebooks (docs). Keep utils/, files/ (inference reference
# tensors), run_*.py, and the tiny example_pdbs/ (used by FC tests via file://).
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='model_weights' \
    --exclude='databases' \
    --exclude='images' \
    --exclude='*.ipynb' \
    --exclude='.git' \
    "$TMP/repo/" "$DST/"

echo "Vendored $LASERMPNN_REPO @ $LASERMPNN_SHA"
echo "  -> $DST"
du -sh "$DST"
