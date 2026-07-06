#!/usr/bin/env bash
# Vendor the upstream DiffDock source into
# services/diffdock-server/upstream/ at a pinned SHA, so
# `docker build` does no network access.
#
#   ./services/diffdock-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   DIFFDOCK_REPO=https://ghproxy.cn/https://github.com/gcorso/DiffDock.git \
#       ./services/diffdock-server/scripts/vendor.sh
#
# To bump the upstream pin, edit DIFFDOCK_SHA below.

set -euo pipefail

DIFFDOCK_REPO="${DIFFDOCK_REPO:-https://github.com/gcorso/DiffDock.git}"
DIFFDOCK_SHA="${DIFFDOCK_SHA:-85c49b60d3e0b0182a59ee43a34a6d7036981284}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/diffdock-server/upstream"
TMP="$(mktemp -d -t diffdock-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# Retry: CN build hosts sometimes hit transient TLS resets on github.com.
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$DIFFDOCK_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$DIFFDOCK_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$DIFFDOCK_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $DIFFDOCK_SHA)" >&2
    exit 1
fi
rm -rf .git

# Strip training / evaluation / GUI bits we do not need at inference time.
# Keep: inference.py, utils/, datasets/, models/, spyrmsd/, confidence/,
#       default_inference_args.yaml, data/1a0q/, data/protein_ligand_example.csv,
#       environment.yml (reference), Dockerfile (reference)
rm -f  "$TMP/repo/train.py" "$TMP/repo/evaluate.py"
rm -rf "$TMP/repo/app"                                           # Gradio Web UI
rm -f  "$TMP/repo/datasets/pdbbind_lm_embedding_preparation.py" \
       "$TMP/repo/datasets/esm_embedding_preparation.py" \
       "$TMP/repo/datasets/esm_embeddings_to_pt.py"
rm -f  "$TMP/repo/overview.png"

# Sync into DST. --delete drops stale files from previous vendor runs.
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.github/' --exclude='.gitignore' \
    "$TMP/repo/" "$DST/"

echo "Vendored $DIFFDOCK_REPO @ $DIFFDOCK_SHA"
echo "  -> $DST"
du -sh "$DST"
