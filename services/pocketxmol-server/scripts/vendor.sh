#!/usr/bin/env bash
# Vendor the upstream PocketXMol source into
# services/pocketxmol-server/upstream/ at a pinned SHA, so `docker build`
# does no network access.
#
#   ./services/pocketxmol-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   POCKETXMOL_REPO=https://ghproxy.cn/https://github.com/pengxingang/PocketXMol.git \
#       ./services/pocketxmol-server/scripts/vendor.sh
#
# To bump the upstream pin, edit POCKETXMOL_SHA below.
#
# Upstream: MIT-licensed (Peng et al., Cell 2026).

set -euo pipefail

POCKETXMOL_REPO="${POCKETXMOL_REPO:-https://github.com/pengxingang/PocketXMol.git}"
POCKETXMOL_SHA="${POCKETXMOL_SHA:-65488cf635c856101dbe703ac97e2f10f58e005c}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/pocketxmol-server/upstream"
TMP="$(mktemp -d -t pocketxmol-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$POCKETXMOL_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$POCKETXMOL_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$POCKETXMOL_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $POCKETXMOL_SHA)" >&2
    exit 1
fi
rm -rf .git

# Keep the full source tree — scripts/sample_use.py imports scripts.train_pl
# at module scope (for DataModule), utils.reconstruct pulls in
# process.utils_process, etc.  The tree is small (~ few MB) and pruning
# individual files is fragile.
#
# data/ccd/ (~1.9 MB) and data/examples/ (~1.7 MB) are kept in-tree for
# test fixture reuse; fetch_weights.sh optionally re-stages them to NAS.

# Sync into DST. --delete drops stale files from previous vendor runs.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $POCKETXMOL_REPO @ $POCKETXMOL_SHA"
echo "  -> $DST"
du -sh "$DST"
