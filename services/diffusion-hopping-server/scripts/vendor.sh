#!/usr/bin/env bash
# Vendor the upstream DiffHopp source into
# services/diffusion-hopping-server/upstream/ at a pinned SHA, so
# `docker build` does no network access.
#
#   ./services/diffusion-hopping-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   DIFFHOPP_REPO=https://ghproxy.cn/https://github.com/jostorge/diffusion-hopping.git \
#       ./services/diffusion-hopping-server/scripts/vendor.sh
#
# To bump the upstream pin, edit DIFFHOPP_SHA below.

set -euo pipefail

DIFFHOPP_REPO="${DIFFHOPP_REPO:-https://github.com/jostorge/diffusion-hopping.git}"
DIFFHOPP_SHA="${DIFFHOPP_SHA:-94aaca24339d021cf78b90c0ca18294b531b5766}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/diffusion-hopping-server/upstream"
TMP="$(mktemp -d -t diffhopp-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$DIFFHOPP_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$DIFFHOPP_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$DIFFHOPP_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $DIFFHOPP_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# DiffHopp ships its 4 ckpts in upstream/checkpoints/ (~189 MB) — those come
# along; fetch_weights.sh copies them to the weights/ stage dir.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $DIFFHOPP_REPO @ $DIFFHOPP_SHA"
echo "  -> $DST"
du -sh "$DST"
