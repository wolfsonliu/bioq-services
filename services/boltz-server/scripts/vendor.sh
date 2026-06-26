#!/usr/bin/env bash
# Vendor the upstream Boltz source into services/boltz-server/upstream/ at a
# pinned SHA, so `docker build` does no network access.  Run once before each
# build (and again to upgrade):
#
#   ./services/boltz-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   BOLTZ_REPO=https://ghproxy.cn/https://github.com/jwohlwend/boltz.git \
#       ./services/boltz-server/scripts/vendor.sh
#
# To bump the upstream pin, edit BOLTZ_SHA below.

set -euo pipefail

BOLTZ_REPO="${BOLTZ_REPO:-https://github.com/jwohlwend/boltz.git}"
BOLTZ_SHA="${BOLTZ_SHA:-cb04aeccdd480fd4db707f0bbafde538397fa2ac}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/boltz-server/upstream"
TMP="$(mktemp -d -t boltz-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$BOLTZ_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$BOLTZ_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$BOLTZ_SHA" ]] || {
    echo "ERROR: HEAD mismatch after checkout (got $actual, expected $BOLTZ_SHA)" >&2
    exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $BOLTZ_REPO @ $BOLTZ_SHA"
echo "  -> $DST"
du -sh "$DST"
