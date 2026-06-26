#!/usr/bin/env bash
# Vendor the upstream DockQ source into services/dockq-server/upstream/ at a
# pinned SHA, so `docker build` does no network access.
#
#   ./services/dockq-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   DOCKQ_REPO=https://ghproxy.cn/https://github.com/wallnerlab/DockQ.git \
#       ./services/dockq-server/scripts/vendor.sh
set -euo pipefail

DOCKQ_REPO="${DOCKQ_REPO:-https://github.com/wallnerlab/DockQ.git}"
DOCKQ_SHA="${DOCKQ_SHA:-75db7ab4f6b824c70d120c5f620582e164ed5479}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/dockq-server/upstream"
TMP="$(mktemp -d -t dockq-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$DOCKQ_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$DOCKQ_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$DOCKQ_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $DOCKQ_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $DOCKQ_REPO @ $DOCKQ_SHA"
echo "  -> $DST"
du -sh "$DST"
