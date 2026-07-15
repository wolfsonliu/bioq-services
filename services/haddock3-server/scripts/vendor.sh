#!/usr/bin/env bash
# Vendor the upstream HADDOCK3 source into services/haddock3-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/haddock3-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   HADDOCK3_REPO=https://ghproxy.cn/https://github.com/haddocking/haddock3.git \
#       ./services/haddock3-server/scripts/vendor.sh
#
# To bump the upstream pin, edit HADDOCK3_SHA below.

set -euo pipefail

HADDOCK3_REPO="${HADDOCK3_REPO:-https://github.com/haddocking/haddock3.git}"
HADDOCK3_SHA="${HADDOCK3_SHA:-7ad1148493fc3a96bbd9c1f16cfc7d7d1e73416c}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/haddock3-server/upstream"
TMP="$(mktemp -d -t haddock3-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry 5x: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$HADDOCK3_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$HADDOCK3_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$HADDOCK3_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $HADDOCK3_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# We keep varia/ (the HADDOCK CNS patches) but drop the heavy example/test
# fixtures + docs + notebooks — the image only needs the installable package.
# Integration-test fixtures live in services/haddock3-server/tests/data/.
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='examples/' \
    --exclude='integration_tests/' \
    --exclude='end-to-end_tests/' \
    --exclude='docs/' \
    --exclude='notebooks/' \
    "$TMP/repo/" "$DST/"

echo "Vendored $HADDOCK3_REPO @ $HADDOCK3_SHA"
echo "  -> $DST"
du -sh "$DST"
