#!/usr/bin/env bash
# Vendor the upstream IgGM source into services/iggm-server/upstream/ at a
# pinned SHA, so `docker build` does no network access.
#
#   ./services/iggm-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   IGGM_REPO=https://ghproxy.cn/https://github.com/TencentAI4S/IgGM.git \
#       ./services/iggm-server/scripts/vendor.sh
#
# To bump the upstream pin, edit IGGM_SHA below.

set -euo pipefail

IGGM_REPO="${IGGM_REPO:-https://github.com/TencentAI4S/IgGM.git}"
IGGM_SHA="${IGGM_SHA:-06abc563b3fc8c7ea020543add16b69b6f8a1c8d}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/iggm-server/upstream"
TMP="$(mktemp -d -t iggm-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$IGGM_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$IGGM_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$IGGM_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $IGGM_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# Keep: IgGM/ package, design.py (the CLI our run_design.py wraps),
#   scripts/merge_chains.py, and a couple of native examples for test fixtures.
# Drop: docs/ (gif), the huge Merge_output.ipynb, humanization example tree,
#   and the collect_plot/plotting deps (v0.0.2).
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.ipynb' \
    --exclude='docs/' \
    --exclude='outputs/' \
    --exclude='examples/humanization/' \
    --exclude='scripts/collect_plot.py' \
    --exclude='scripts/Merge_output.ipynb' \
    --exclude='scripts/multiple_runs.sh' \
    "$TMP/repo/" "$DST/"

echo "Vendored $IGGM_REPO @ $IGGM_SHA"
echo "  -> $DST"
du -sh "$DST"
