#!/usr/bin/env bash
# Vendor the upstream TurboHopp source into
# services/turbohopp-server/upstream/ at a pinned SHA, so `docker build`
# does no network access.
#
#   ./services/turbohopp-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   TURBOHOPP_REPO=https://ghproxy.cn/https://github.com/orgw/TurboHopp.git \
#       ./services/turbohopp-server/scripts/vendor.sh
#
# To bump the upstream pin, edit TURBOHOPP_SHA below.

set -euo pipefail

TURBOHOPP_REPO="${TURBOHOPP_REPO:-https://github.com/orgw/TurboHopp.git}"
TURBOHOPP_SHA="${TURBOHOPP_SHA:-e342350be5b83ad6456ef5d52fb3882328cf5ea1}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/turbohopp-server/upstream"
TMP="$(mktemp -d -t turbohopp-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$TURBOHOPP_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$TURBOHOPP_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$TURBOHOPP_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $TURBOHOPP_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# TurboHopp upstream does NOT ship model checkpoints — no weights/ directory
# to copy along. Users must obtain the consistency-model .ckpt separately
# and rsync to NAS (see README.md ## Weights).
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $TURBOHOPP_REPO @ $TURBOHOPP_SHA"
echo "  -> $DST"
du -sh "$DST"
