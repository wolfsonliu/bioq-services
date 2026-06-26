#!/usr/bin/env bash
# Vendor the upstream RFantibody source into
# services/rfantibody-server/upstream/ at a pinned SHA, so `docker build`
# does no network access.
#
#   ./services/rfantibody-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   RFANTIBODY_REPO=https://ghproxy.cn/https://github.com/RosettaCommons/RFantibody.git \
#       ./services/rfantibody-server/scripts/vendor.sh
set -euo pipefail

RFANTIBODY_REPO="${RFANTIBODY_REPO:-https://github.com/RosettaCommons/RFantibody.git}"
RFANTIBODY_SHA="${RFANTIBODY_SHA:-8fe311415754e0276d1a39c87c57e69c88927a2d}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/rfantibody-server/upstream"
TMP="$(mktemp -d -t rfantibody-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$RFANTIBODY_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$RFANTIBODY_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$RFANTIBODY_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $RFANTIBODY_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $RFANTIBODY_REPO @ $RFANTIBODY_SHA"
echo "  -> $DST"
du -sh "$DST"
