#!/usr/bin/env bash
# Vendor the upstream Genie3 source into services/genie3-server/upstream/
# at a pinned SHA, so `docker build` does no network access to github.
#
# Patches under services/genie3-server/patches/ are applied at build time by
# the Dockerfile (NOT here) so iterating on a patch doesn't require re-cloning.
#
#   ./services/genie3-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   GENIE3_REPO=https://ghproxy.cn/https://github.com/aqlaboratory/genie3.git \
#       ./services/genie3-server/scripts/vendor.sh
set -euo pipefail

GENIE3_REPO="${GENIE3_REPO:-https://github.com/aqlaboratory/genie3.git}"
GENIE3_SHA="${GENIE3_SHA:-5214459c42e69b01fadfc7d7ebda09d8e5082115}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/genie3-server/upstream"
TMP="$(mktemp -d -t genie3-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$GENIE3_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$GENIE3_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$GENIE3_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $GENIE3_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $GENIE3_REPO @ $GENIE3_SHA"
echo "  -> $DST"
du -sh "$DST"
