#!/usr/bin/env bash
# Vendor the upstream ChemBounce source into
# services/chembounce-server/upstream/ at a pinned SHA, so `docker build`
# does no network access to github.
#
#   ./services/chembounce-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   CHEMBOUNCE_REPO=https://ghproxy.cn/https://github.com/jyryu3161/chembounce.git \
#       ./services/chembounce-server/scripts/vendor.sh
#
# To bump the upstream pin, edit CHEMBOUNCE_SHA below.
#
# *** LICENSE NOTE ***
# ChemBounce ships without a LICENSE file (as of SHA below).  This image
# is for INTERNAL RESEARCH USE ONLY — do not redistribute, do not push to
# public registries.  See engineering/decisions/2026-06-28-chembounce-server-design.md §10.1.

set -euo pipefail

CHEMBOUNCE_REPO="${CHEMBOUNCE_REPO:-https://github.com/jyryu3161/chembounce.git}"
CHEMBOUNCE_SHA="${CHEMBOUNCE_SHA:-eaadfa725c649fb2d28315f20de7100bda688694}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/chembounce-server/upstream"
TMP="$(mktemp -d -t chembounce-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$CHEMBOUNCE_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$CHEMBOUNCE_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$CHEMBOUNCE_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $CHEMBOUNCE_SHA)" >&2
    exit 1
fi
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $CHEMBOUNCE_REPO @ $CHEMBOUNCE_SHA"
echo "  -> $DST"
du -sh "$DST"
