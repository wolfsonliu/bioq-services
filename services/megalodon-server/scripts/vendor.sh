#!/usr/bin/env bash
# Vendor the upstream Megalodon source into services/megalodon-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/megalodon-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   MEGALODON_REPO=https://ghproxy.cn/https://github.com/NVIDIA-BioNeMo/megalodon.git \
#       ./services/megalodon-server/scripts/vendor.sh
#
# To bump the upstream pin, edit MEGALODON_SHA below.
set -euo pipefail

MEGALODON_REPO="${MEGALODON_REPO:-https://github.com/NVIDIA-BioNeMo/megalodon.git}"
MEGALODON_SHA="${MEGALODON_SHA:-7cf61f4e97c0c15d5b3dc5781a30448af6e67b81}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/megalodon-server/upstream"
TMP="$(mktemp -d -t megalodon-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$MEGALODON_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$MEGALODON_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$MEGALODON_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $MEGALODON_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# Keep src/ (the `megalodon` package, PYTHONPATH=/opt/megalodon/src),
# scripts/ (conf/ YAML variants the service rewrites at runtime), and
# data_processing/ (kept for reference / rebuilding the statistics bundle).
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.ipynb' \
    --exclude='notebooks/' \
    --exclude='images/' \
    --exclude='.github/' \
    "$TMP/repo/" "$DST/"

echo "Vendored $MEGALODON_REPO @ $MEGALODON_SHA"
echo "  -> $DST"
du -sh "$DST"
