#!/usr/bin/env bash
# Vendor the upstream SemlaFlow source into services/semlaflow-server/upstream/
# at a pinned SHA, so `docker build` does no network access.
#
#   ./services/semlaflow-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   SEMLAFLOW_REPO=https://ghproxy.cn/https://github.com/rssrwn/semla-flow.git \
#       ./services/semlaflow-server/scripts/vendor.sh
#
# To bump the upstream pin, edit SEMLAFLOW_SHA below.

set -euo pipefail

SEMLAFLOW_REPO="${SEMLAFLOW_REPO:-https://github.com/rssrwn/semla-flow.git}"
SEMLAFLOW_SHA="${SEMLAFLOW_SHA:-3f43103d3af138b86dbe9f29fe8085e83f9a6283}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/semlaflow-server/upstream"
TMP="$(mktemp -d -t semlaflow-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$SEMLAFLOW_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$SEMLAFLOW_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$SEMLAFLOW_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $SEMLAFLOW_SHA)" >&2
    exit 1
fi
rm -rf .git

# Sync into DST. --delete drops stale files from previous vendor runs.
# Exclude notebooks/images — not needed at runtime. The `semlaflow/` package
# is what `python .../server/inference.py` imports (PYTHONPATH=/opt/semlaflow).
#
# Unlike flowmol, SemlaFlow does NOT bake any per-dataset summary artifacts:
# molecule sizes are sampled at runtime from the NAS-staged reference .smol
# split, and coord_std / bucket_limits are hardcoded constants in
# semlaflow/scriptutil.py. So there is no data/ dir to preserve here — the
# vendored tree is just the source package.
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.ipynb' \
    --exclude='notebooks/' \
    --exclude='images/' \
    "$TMP/repo/" "$DST/"

echo "Vendored $SEMLAFLOW_REPO @ $SEMLAFLOW_SHA"
echo "  -> $DST"
du -sh "$DST"
