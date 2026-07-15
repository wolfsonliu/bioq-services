#!/usr/bin/env bash
# Vendor the upstream PLIP source into services/plip-server/upstream/ at a
# pinned SHA, so `docker build` does no network access. Run once before each
# build (and again to upgrade):
#
#   ./services/plip-server/scripts/vendor.sh
#
# To use a github mirror (CN networks, flaky TLS):
#
#   PLIP_REPO=https://ghproxy.cn/https://github.com/pharmai/plip \
#       ./services/plip-server/scripts/vendor.sh
#
# To bump the upstream pin, edit PLIP_SHA below.

set -euo pipefail

PLIP_REPO="${PLIP_REPO:-https://github.com/pharmai/plip}"
# v3.0.1 (PLIP 2025). Provenance: github.com/pharmai/plip.
PLIP_SHA="${PLIP_SHA:-2f4911d307490479ac023b22d6faa8f59b577ca8}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/plip-server/upstream"
TMP="$(mktemp -d -t plip-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$PLIP_REPO" "$TMP/repo"; then
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
git checkout "$PLIP_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$PLIP_SHA" ]]; then
    echo "ERROR: HEAD mismatch after checkout (got $actual, expected $PLIP_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sanity: the flat `plip/` package + its CLI entry must be present.
if [[ ! -f "$TMP/repo/plip/plipcmd.py" ]]; then
    echo "ERROR: vendored source missing plip/plipcmd.py — wrong ref?" >&2
    exit 1
fi

# Sync into DST. --delete drops stale files from previous vendor runs. We vendor
# the runtime subset; the Dockerfile COPYs the plip/ package subtree. The
# upstream plip/test/ dir (~30 MB of PDB fixtures) is excluded so it isn't baked
# into the image — it is not imported at runtime.
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='plip/test' \
    "$TMP/repo/" "$DST/"

echo "Vendored $PLIP_REPO @ $PLIP_SHA"
echo "  -> $DST"
du -sh "$DST"
