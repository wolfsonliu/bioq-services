#!/usr/bin/env bash
# Vendor the upstream LightDock source into services/lightdock-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/lightdock-server/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   LIGHTDOCK_REPO=https://ghproxy.cn/https://github.com/lightdock/lightdock.git \
#       ./services/lightdock-server/scripts/vendor.sh
set -euo pipefail

LIGHTDOCK_REPO="${LIGHTDOCK_REPO:-https://github.com/lightdock/lightdock.git}"
# v1.0.0-pre2
LIGHTDOCK_SHA="${LIGHTDOCK_SHA:-4654ee58542e99174776b2f80dbcfb673e918da1}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/lightdock-server/upstream"
TMP="$(mktemp -d -t lightdock-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$LIGHTDOCK_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$LIGHTDOCK_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$LIGHTDOCK_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $LIGHTDOCK_SHA)" >&2; exit 1
}
rm -rf .git

# Exclude lightdock/test (~145 MB of golden-data PDBs) — pure test fixtures, not
# imported at runtime; keeping them would bloat the image for no benefit. NOTE:
# after changing this exclude list, `rm -rf services/lightdock-server/upstream/`
# before re-running so Docker COPY doesn't reuse a stale, larger vendor.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='lightdock/test' \
    "$TMP/repo/" "$DST/"

echo "Vendored $LIGHTDOCK_REPO @ $LIGHTDOCK_SHA"
echo "  -> $DST"
du -sh "$DST"
