#!/usr/bin/env bash
# Vendor the upstream ESM (Biohub fork) source into
# services/esmfold2-server/upstream/ at a pinned SHA, so `docker build` does
# no network access.
#
#   ./services/esmfold2-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   ESM_REPO=https://ghproxy.cn/https://github.com/Biohub/esm.git \
#       ./services/esmfold2-server/scripts/vendor.sh
set -euo pipefail

ESM_REPO="${ESM_REPO:-https://github.com/Biohub/esm.git}"
ESM_SHA="${ESM_SHA:-af8ef5cead4388c4d7af04168157c98898c78805}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/esmfold2-server/upstream"
TMP="$(mktemp -d -t esm-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$ESM_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$ESM_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$ESM_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $ESM_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $ESM_REPO @ $ESM_SHA"
echo "  -> $DST"
du -sh "$DST"
