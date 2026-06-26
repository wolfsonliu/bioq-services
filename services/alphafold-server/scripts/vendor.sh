#!/usr/bin/env bash
# Vendor the upstream AlphaFold source into
# services/alphafold-server/upstream/ at a pinned SHA, so `docker build`
# does no network access to github.
#
#   ./services/alphafold-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   ALPHAFOLD_REPO=https://ghproxy.cn/https://github.com/google-deepmind/alphafold.git \
#       ./services/alphafold-server/scripts/vendor.sh
set -euo pipefail

ALPHAFOLD_REPO="${ALPHAFOLD_REPO:-https://github.com/google-deepmind/alphafold.git}"
ALPHAFOLD_SHA="${ALPHAFOLD_SHA:-c77e5d2a8961d1a353632c462914ff0a32a950f6}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/alphafold-server/upstream"
TMP="$(mktemp -d -t alphafold-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$ALPHAFOLD_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$ALPHAFOLD_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$ALPHAFOLD_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $ALPHAFOLD_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $ALPHAFOLD_REPO @ $ALPHAFOLD_SHA"
echo "  -> $DST"
du -sh "$DST"
