#!/usr/bin/env bash
# Vendor the upstream protein-design-mcp source into
# edge/protein-design-mcp/upstream/ at a pinned SHA, so `docker
# build` does no network access for THIS upstream.  (The Dockerfile still
# pulls RFdiffusion / ProteinMPNN / ColabFold / OpenFold from github at
# build time via plain `git clone`; those are out of scope for this script.)
#
#   ./edge/protein-design-mcp/scripts/vendor.sh
#
# Github mirror override (CN networks):
#
#   PROTEIN_DESIGN_MCP_REPO=https://ghproxy.cn/https://github.com/jasonkim8652/protein-design-mcp.git \
#       ./edge/protein-design-mcp/scripts/vendor.sh
set -euo pipefail

PROTEIN_DESIGN_MCP_REPO="${PROTEIN_DESIGN_MCP_REPO:-https://github.com/jasonkim8652/protein-design-mcp.git}"
PROTEIN_DESIGN_MCP_SHA="${PROTEIN_DESIGN_MCP_SHA:-7a45f13d5c7667513f4b3cfc47e472f3209b1be1}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/edge/protein-design-mcp/upstream"
TMP="$(mktemp -d -t pdm-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$PROTEIN_DESIGN_MCP_REPO" "$TMP/repo"; then break; fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."; sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$PROTEIN_DESIGN_MCP_SHA"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "$PROTEIN_DESIGN_MCP_SHA" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected $PROTEIN_DESIGN_MCP_SHA)" >&2; exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored $PROTEIN_DESIGN_MCP_REPO @ $PROTEIN_DESIGN_MCP_SHA"
echo "  -> $DST"
du -sh "$DST"
