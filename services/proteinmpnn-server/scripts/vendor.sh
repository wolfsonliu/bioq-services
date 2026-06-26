#!/usr/bin/env bash
# Vendor the upstream ProteinMPNN source into
# services/proteinmpnn-server/upstream/ at a pinned SHA, so `docker build`
# does no network access.  Run once before each build (and again to upgrade):
#
#   ./services/proteinmpnn-server/scripts/vendor.sh
#
# To use a github mirror (CN networks, flaky TLS):
#
#   PROTEINMPNN_REPO=https://ghproxy.cn/https://github.com/dauparas/ProteinMPNN \
#       ./services/proteinmpnn-server/scripts/vendor.sh
#
# To bump the upstream pin, edit PROTEINMPNN_SHA below.

set -euo pipefail

PROTEINMPNN_REPO="${PROTEINMPNN_REPO:-https://github.com/dauparas/ProteinMPNN}"
PROTEINMPNN_SHA="${PROTEINMPNN_SHA:-8907e6671bfbfc92303b5f79c4b5e6ce47cdef57}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/proteinmpnn-server/upstream"
TMP="$(mktemp -d -t pmpnn-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$PROTEINMPNN_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && {
        echo "ERROR: git clone failed after 5 attempts" >&2
        exit 1
    }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$PROTEINMPNN_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$PROTEINMPNN_SHA" ]]; then
    echo "ERROR: HEAD mismatch after checkout (got $actual, expected $PROTEINMPNN_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# We vendor the full repo (excluding __pycache__/.pyc) — the Dockerfile picks
# only the runtime-required subset via per-path COPYs.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $PROTEINMPNN_REPO @ $PROTEINMPNN_SHA"
echo "  -> $DST"
du -sh "$DST"
