#!/usr/bin/env bash
# Vendor the upstream ImmuneBuilder source into
# services/immunebuilder-server/upstream/ at a pinned SHA, so `docker build`
# does no network access.
#
#   ./services/immunebuilder-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   IMMUNEBUILDER_REPO=https://ghproxy.cn/https://github.com/brennanaba/ImmuneBuilder.git \
#       ./services/immunebuilder-server/scripts/vendor.sh
set -euo pipefail

IMMUNEBUILDER_REPO="${IMMUNEBUILDER_REPO:-https://github.com/brennanaba/ImmuneBuilder.git}"
IMMUNEBUILDER_SHA="${IMMUNEBUILDER_SHA:-0df4e2ad82a1aa60f37ea9dae335d1198159ef78}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/immunebuilder-server/upstream"
TMP="$(mktemp -d -t immunebuilder-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$IMMUNEBUILDER_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$IMMUNEBUILDER_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$IMMUNEBUILDER_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $IMMUNEBUILDER_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $IMMUNEBUILDER_REPO @ $IMMUNEBUILDER_SHA"
echo "  -> $DST"
du -sh "$DST"
