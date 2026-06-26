#!/usr/bin/env bash
# Vendor the upstream DeepRank-Ab source into
# services/deeprank-ab-server/upstream/ at a pinned SHA, so `docker build`
# does no network access to github.
#
#   ./services/deeprank-ab-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   DEEPRANK_AB_REPO=https://ghproxy.cn/https://github.com/haddocking/DeepRank-Ab.git \
#       ./services/deeprank-ab-server/scripts/vendor.sh
set -euo pipefail

DEEPRANK_AB_REPO="${DEEPRANK_AB_REPO:-https://github.com/haddocking/DeepRank-Ab.git}"
DEEPRANK_AB_SHA="${DEEPRANK_AB_SHA:-5204621bacc7df62e9b0b8fe28acbd3bca5fbacf}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/deeprank-ab-server/upstream"
TMP="$(mktemp -d -t deeprank-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$DEEPRANK_AB_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$DEEPRANK_AB_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$DEEPRANK_AB_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $DEEPRANK_AB_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $DEEPRANK_AB_REPO @ $DEEPRANK_AB_SHA"
echo "  -> $DST"
du -sh "$DST"
