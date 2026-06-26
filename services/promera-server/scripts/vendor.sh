#!/usr/bin/env bash
# Vendor 3 upstreams (tinyprot, promera, LigandMPNN) into
# services/promera-server/upstream/{tinyprot,promera,LigandMPNN}/ at pinned
# SHAs, so `docker build` does no network access.
#
#   ./services/promera-server/scripts/vendor.sh
#
# Github mirror override (CN networks) — applies to all 3 unless overridden
# individually:
#
#   GITHUB_PREFIX=https://ghproxy.cn/ \
#       ./services/promera-server/scripts/vendor.sh
set -euo pipefail

GITHUB_PREFIX="${GITHUB_PREFIX:-}"

TINYPROT_REPO="${TINYPROT_REPO:-${GITHUB_PREFIX}https://github.com/bjing2016/tinyprot.git}"
TINYPROT_SHA="${TINYPROT_SHA:-e33866bc474c64a9cfc324e0cbb7f9f82ad1e855}"
PROMERA_REPO="${PROMERA_REPO:-${GITHUB_PREFIX}https://github.com/bjing2016/promera.git}"
PROMERA_SHA="${PROMERA_SHA:-5c7d6a88bbf7f13906274fdc7c7f84e7a666017d}"
LIGANDMPNN_REPO="${LIGANDMPNN_REPO:-${GITHUB_PREFIX}https://github.com/dauparas/LigandMPNN}"
LIGANDMPNN_SHA="${LIGANDMPNN_SHA:-26ec57ac976ade5379920dbd43c7f97a91cf82de}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/promera-server/upstream"
TMP="$(mktemp -d -t promera-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

vendor_one() {
    local name="$1" repo="$2" sha="$3"
    echo "=== vendoring $name @ $sha ==="
    local tmprepo="$TMP/$name"

    for i in 1 2 3 4 5; do
        rm -rf "$tmprepo"
        if git clone --filter=blob:none --no-checkout "$repo" "$tmprepo"; then break; fi
        [ "$i" = "5" ] && { echo "ERROR: $name clone failed after 5 attempts" >&2; exit 1; }
        echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
    done

    cd "$tmprepo"
    git checkout "$sha"
    actual="$(git rev-parse HEAD)"
    [[ "$actual" = "$sha" ]] || {
        echo "ERROR: $name HEAD mismatch (got $actual, expected $sha)" >&2; exit 1
    }
    rm -rf .git

    rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
        "$tmprepo/" "$DST/$name/"
}

mkdir -p "$DST"
vendor_one tinyprot   "$TINYPROT_REPO"   "$TINYPROT_SHA"
vendor_one promera    "$PROMERA_REPO"    "$PROMERA_SHA"
vendor_one LigandMPNN "$LIGANDMPNN_REPO" "$LIGANDMPNN_SHA"

echo
echo "Vendored 3 upstreams into $DST:"
du -sh "$DST"/*/
