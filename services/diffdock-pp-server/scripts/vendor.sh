#!/usr/bin/env bash
# Vendor the upstream DiffDock-PP source into
# services/diffdock-pp-server/upstream/ at a pinned SHA, so
# `docker build` does no network access.
#
#   ./services/diffdock-pp-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   DIFFDOCK_PP_REPO=https://ghproxy.cn/https://github.com/ketatam/DiffDock-PP.git \
#       ./services/diffdock-pp-server/scripts/vendor.sh
#
# To bump the upstream pin, edit DIFFDOCK_PP_SHA below.

set -euo pipefail

DIFFDOCK_PP_REPO="${DIFFDOCK_PP_REPO:-https://github.com/ketatam/DiffDock-PP.git}"
DIFFDOCK_PP_SHA="${DIFFDOCK_PP_SHA:-25a28900736c0730821e45265ee8e409751c358a}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/diffdock-pp-server/upstream"
TMP="$(mktemp -d -t diffdock-pp-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$DIFFDOCK_PP_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$DIFFDOCK_PP_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$DIFFDOCK_PP_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $DIFFDOCK_PP_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# DiffDock-PP upstream ships score/confidence ckpts under checkpoints/ (~22 MB);
# fetch_weights.sh copies them (plus args.yaml + ESM-2) to the NAS layout.
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='baselines/rmsd_plots/' \
    --exclude='src/notebooks/' \
    "$TMP/repo/" "$DST/"

echo "Vendored $DIFFDOCK_PP_REPO @ $DIFFDOCK_PP_SHA"
echo "  -> $DST"
du -sh "$DST"
