#!/usr/bin/env bash
# Vendor the upstream BindFlow source into
# services/bindflow-server/upstream/ at a pinned SHA, so `docker build`
# does no network access to github.
#
#   ./services/bindflow-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   BINDFLOW_REPO=https://ghproxy.cn/https://github.com/ale94mleon/BindFlow.git \
#       ./services/bindflow-server/scripts/vendor.sh
#
# To bump the upstream pin, edit BINDFLOW_SHA below.

set -euo pipefail

BINDFLOW_REPO="${BINDFLOW_REPO:-https://github.com/ale94mleon/BindFlow.git}"
# Pinned to the head of main at design time (2026-07-06).  BindFlow is
# pre-alpha and does not cut tagged releases; we track main-tip.
BINDFLOW_SHA="${BINDFLOW_SHA:-3dfe07c2121a81f2e6e3a1e1568f3d50298ada59}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/bindflow-server/upstream"
TMP="$(mktemp -d -t bindflow-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$BINDFLOW_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$BINDFLOW_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$BINDFLOW_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $BINDFLOW_SHA)" >&2
    exit 1
fi

# BindFlow uses versioningit → the version is derived from `git describe`.
# We drop .git for a self-contained vendor, so first write the resolved
# version to _version.py (upstream's `[tool.versioningit.write]` target)
# so the pip install still has a sane __version__ instead of "1+unknown".
{
    echo "__version__ = \"0.0.0+bindflow-server-vendor-${BINDFLOW_SHA:0:12}\""
    echo "__version_tuple__ = (0, 0, 0)"
} > src/bindflow/_version.py

rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $BINDFLOW_REPO @ $BINDFLOW_SHA"
echo "  -> $DST"
du -sh "$DST"
