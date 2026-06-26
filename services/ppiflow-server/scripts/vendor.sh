#!/usr/bin/env bash
# Vendor the upstream PPIFlow source into services/ppiflow-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/ppiflow-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   PPIFLOW_REPO=https://ghproxy.cn/https://github.com/Mingchenchen/PPIFlow.git \
#       ./services/ppiflow-server/scripts/vendor.sh
set -euo pipefail

PPIFLOW_REPO="${PPIFLOW_REPO:-https://github.com/Mingchenchen/PPIFlow.git}"
PPIFLOW_SHA="${PPIFLOW_SHA:-697411f0686f6b26280d3801fee7ac96e4247bac}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/ppiflow-server/upstream"
TMP="$(mktemp -d -t ppiflow-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$PPIFLOW_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$PPIFLOW_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$PPIFLOW_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $PPIFLOW_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $PPIFLOW_REPO @ $PPIFLOW_SHA"
echo "  -> $DST"
du -sh "$DST"
