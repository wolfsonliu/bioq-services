#!/usr/bin/env bash
# Vendor the upstream BoltzGen source into services/boltzgen-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/boltzgen-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   BOLTZGEN_REPO=https://ghproxy.cn/https://github.com/HannesStark/boltzgen.git \
#       ./services/boltzgen-server/scripts/vendor.sh
#
# To bump the upstream pin, edit BOLTZGEN_SHA below.

set -euo pipefail

BOLTZGEN_REPO="${BOLTZGEN_REPO:-https://github.com/HannesStark/boltzgen.git}"
BOLTZGEN_SHA="${BOLTZGEN_SHA:-31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/boltzgen-server/upstream"
TMP="$(mktemp -d -t boltzgen-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$BOLTZGEN_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$BOLTZGEN_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$BOLTZGEN_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $BOLTZGEN_SHA)" >&2
    exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $BOLTZGEN_REPO @ $BOLTZGEN_SHA"
echo "  -> $DST"
du -sh "$DST"
