#!/usr/bin/env bash
# Vendor the upstream DrugHIVE source into
# services/drughive-server/upstream/ at a pinned SHA, so `docker build`
# does no network access.
#
#   ./services/drughive-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   DRUGHIVE_REPO=https://ghproxy.cn/https://github.com/jssweller/DrugHIVE.git \
#       ./services/drughive-server/scripts/vendor.sh
#
# To bump the upstream pin, edit DRUGHIVE_SHA below.
#
# NOTE: DrugHIVE is USC-RL v2.0 licensed (non-commercial academic research
# only).  See LICENSE in the vendored tree and engineering/decisions/
# 2026-07-02-drughive-server-design.md §Risks §1.

set -euo pipefail

DRUGHIVE_REPO="${DRUGHIVE_REPO:-https://github.com/jssweller/DrugHIVE.git}"
DRUGHIVE_SHA="${DRUGHIVE_SHA:-d965edf6e6770bc15c38860e0f7e773bdf28975b}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/drughive-server/upstream"
TMP="$(mktemp -d -t drughive-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$DRUGHIVE_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$DRUGHIVE_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$DRUGHIVE_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $DRUGHIVE_SHA)" >&2
    exit 1
fi
rm -rf .git

# Strip non-inference bits.  We keep dock.py / ff_optimize.py because
# generate_optimize.py imports helpers from them; the standalone /api/dock
# and /api/ff_optimize endpoints are out of v0.0.1 scope but the modules
# stay in the tree.
rm -f  "$TMP/repo/train.py" \
       "$TMP/repo/process_pdbbind_data.py" \
       "$TMP/repo/process_zinc_dataset.py"
rm -rf "$TMP/repo/img"
rm -f  "$TMP/repo/config/train.yml"

# Sync into DST. --delete drops stale files from previous vendor runs.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $DRUGHIVE_REPO @ $DRUGHIVE_SHA"
echo "  -> $DST"
du -sh "$DST"
