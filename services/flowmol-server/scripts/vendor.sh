#!/usr/bin/env bash
# Vendor the upstream FlowMol source into services/flowmol-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/flowmol-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   FLOWMOL_REPO=https://ghproxy.cn/https://github.com/Dunni3/FlowMol.git \
#       ./services/flowmol-server/scripts/vendor.sh
#
# To bump the upstream pin, edit FLOWMOL_SHA below.

set -euo pipefail

FLOWMOL_REPO="${FLOWMOL_REPO:-https://github.com/Dunni3/FlowMol.git}"
FLOWMOL_SHA="${FLOWMOL_SHA:-77cae22174b7792b0e25e9e0414038420736d841}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/flowmol-server/upstream"
TMP="$(mktemp -d -t flowmol-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$FLOWMOL_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$FLOWMOL_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$FLOWMOL_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $FLOWMOL_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# Exclude the notebook + large-ish sample dirs — these aren't needed at
# runtime. The `flowmol/trained_models/` directory is NOT excluded but is
# effectively empty until fetch_weights.sh runs; upstream ships only a
# readme there.
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.ipynb' \
    --exclude='images/' --exclude='data/' \
    "$TMP/repo/" "$DST/"

echo "Vendored $FLOWMOL_REPO @ $FLOWMOL_SHA"
echo "  -> $DST"
du -sh "$DST"
