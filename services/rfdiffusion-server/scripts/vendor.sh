#!/usr/bin/env bash
# Vendor the upstream RFdiffusion source into
# services/rfdiffusion-server/upstream/ at a pinned SHA, so `docker build`
# does no network access to github.
#
#   ./services/rfdiffusion-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   RFDIFFUSION_REPO=https://ghproxy.cn/https://github.com/RosettaCommons/RFdiffusion.git \
#       ./services/rfdiffusion-server/scripts/vendor.sh
set -euo pipefail

RFDIFFUSION_REPO="${RFDIFFUSION_REPO:-https://github.com/RosettaCommons/RFdiffusion.git}"
RFDIFFUSION_SHA="${RFDIFFUSION_SHA:-9535f1938203a24937d7dadf0cb831d02cb5fc0e}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/rfdiffusion-server/upstream"
TMP="$(mktemp -d -t rfdiffusion-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$RFDIFFUSION_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$RFDIFFUSION_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$RFDIFFUSION_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $RFDIFFUSION_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $RFDIFFUSION_REPO @ $RFDIFFUSION_SHA"
echo "  -> $DST"
du -sh "$DST"
