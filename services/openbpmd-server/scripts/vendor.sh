#!/usr/bin/env bash
# Vendor the upstream OpenBPMD source into services/openbpmd-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/openbpmd-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   OPENBPMD_REPO=https://ghproxy.cn/https://github.com/Gervasiolab/OpenBPMD.git \
#       ./services/openbpmd-server/scripts/vendor.sh
#
# To bump the upstream pin, edit OPENBPMD_SHA below.

set -euo pipefail

OPENBPMD_REPO="${OPENBPMD_REPO:-https://github.com/Gervasiolab/OpenBPMD.git}"
OPENBPMD_SHA="${OPENBPMD_SHA:-62c719555729a8e7c850dd29b841f38623c7ad70}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/openbpmd-server/upstream"
TMP="$(mktemp -d -t openbpmd-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry 5x: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$OPENBPMD_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$OPENBPMD_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$OPENBPMD_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $OPENBPMD_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# Exclude the heavy example/test trajectory artifacts — the runtime only needs
# `openbpmd.py` (imported by our wrapper) + LICENSE.  The multi-MB
# examples/*.{prm7,rst7,gro,top,pdb} + tests/files/*.{dcd,pdb} fixtures are NOT
# needed inside the image (integration-test fixtures live in
# services/openbpmd-server/tests/data/, copied separately).
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.ipynb' \
    --exclude='examples/' \
    --exclude='tests/' \
    "$TMP/repo/" "$DST/"

echo "Vendored $OPENBPMD_REPO @ $OPENBPMD_SHA"
echo "  -> $DST"
du -sh "$DST"
