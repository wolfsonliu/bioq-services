#!/usr/bin/env bash
# Vendor the upstream ODesign source into services/odesign-server/upstream/
# at a pinned SHA, so `docker build` does no network access to github.
#
#   ./services/odesign-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   ODESIGN_REPO=https://ghproxy.cn/https://github.com/OTeam-AI4S/ODesign.git \
#       ./services/odesign-server/scripts/vendor.sh
set -euo pipefail

ODESIGN_REPO="${ODESIGN_REPO:-https://github.com/OTeam-AI4S/ODesign.git}"
ODESIGN_SHA="${ODESIGN_SHA:-cc95c2f8f915af4d87b1db3f44c4db2df8566a41}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/odesign-server/upstream"
TMP="$(mktemp -d -t odesign-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$ODESIGN_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$ODESIGN_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$ODESIGN_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $ODESIGN_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $ODESIGN_REPO @ $ODESIGN_SHA"
echo "  -> $DST"
du -sh "$DST"
