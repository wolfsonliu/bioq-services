#!/usr/bin/env bash
# Fetch OpenADMET pre-staged weights: 6 chemprop-chemeleon models from
# HuggingFace + CheMeleon foundation from Zenodo.
#
# Weights are NOT baked into the Docker image — they live on NAS (FC) or get
# bound via apptainer (SIF).  This script fetches to a stage dir; upload the
# result to NAS afterward (or set WEIGHTS_DST=/mnt/nas/... to write direct).
#
# Layout produced (matching engineering/decisions/2026-07-05-openadmet-server-design.md §6.7):
#
#   $DST/
#   ├── foundations/
#   │   └── .chemprop/
#   │       └── chemeleon_mp.pt          ← Zenodo record 15460715
#   └── models/
#       ├── herg-chemeleon-baseline/
#       ├── cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1/
#       ├── cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline/
#       ├── microsomal-clearance-chemeleon-v1/
#       ├── permeability-logd-ppb-chemeleon-baseline/
#       └── pxr-chemeleon-baseline/
#
# HF download layout is `<name>/anvil_training/{model.pth,model.json,recipe_components/,...}`;
# this script flattens the anvil_training/ level so NAS layout is `<name>/{model.pth,...}` directly.
#
# Default (local stage dir):
#   ./services/openadmet-server/scripts/fetch_weights.sh
#       → services/openadmet-server/weights/
#
# Direct download to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/openadmet \
#       ./services/openadmet-server/scripts/fetch_weights.sh
#
# If opensource/openadmet-models/hc/ already contains the HF-downloaded
# model dirs (via the download.sh in that folder), we rsync from there
# instead of re-fetching from HuggingFace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
HC_SRC="$PROJECT_ROOT/opensource/openadmet-models/hc"

mkdir -p "$DST/models" "$DST/foundations/.chemprop"

MODELS=(
    "herg-chemeleon-baseline"
    "cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1"
    "cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline"
    "microsomal-clearance-chemeleon-v1"
    "permeability-logd-ppb-chemeleon-baseline"
    "pxr-chemeleon-baseline"
)

# ---- 1. Model dirs ----
for name in "${MODELS[@]}"; do
    hc_dir="$HC_SRC/$name/anvil_training"
    dst_dir="$DST/models/$name"

    if [[ -d "$hc_dir" ]] && [[ -f "$hc_dir/model.pth" ]]; then
        echo "[hc] $name  <-  $hc_dir  (flatten anvil_training/)"
        # Trailing slash on source: rsync copies **contents** of anvil_training/
        # into dst_dir/, not the anvil_training/ dir itself.
        mkdir -p "$dst_dir"
        rsync -a --delete \
            --exclude='__pycache__' --exclude='*.pyc' \
            --exclude='logs/' --exclude='.cache/' \
            "$hc_dir/" "$dst_dir/"
    else
        echo "[hf] $name  <-  huggingface.co/openadmet/$name  (hc/ missing, fetching)"
        # Fall back to HuggingFace CLI.  Requires the `huggingface_hub[cli]` package
        # (aliased as `hf` since v0.29) to be on PATH.
        tmp="$(mktemp -d -t oa-model.XXXX)"
        hf download --local-dir "$tmp" "openadmet/$name"
        if [[ ! -f "$tmp/anvil_training/model.pth" ]]; then
            echo "ERROR: expected $tmp/anvil_training/model.pth after hf download" >&2
            exit 1
        fi
        rsync -a --delete --exclude='logs/' --exclude='.cache/' \
            "$tmp/anvil_training/" "$dst_dir/"
        rm -rf "$tmp"
    fi
done

# ---- 2. CheMeleon foundation ----
FOUNDATION="$DST/foundations/.chemprop/chemeleon_mp.pt"
ZENODO_URL="https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

if [[ -f "$FOUNDATION" ]]; then
    echo "[skip] chemeleon_mp.pt already present at $FOUNDATION"
else
    # Prefer the hc/ copy if present (single source of truth already downloaded).
    if [[ -f "$HC_SRC/chemeleon_mp.pt" ]]; then
        echo "[hc] chemeleon_mp.pt  <-  $HC_SRC/chemeleon_mp.pt"
        cp "$HC_SRC/chemeleon_mp.pt" "$FOUNDATION"
    else
        echo "[zenodo] chemeleon_mp.pt  <-  $ZENODO_URL"
        wget -c -O "$FOUNDATION" "$ZENODO_URL"
    fi
fi

# Sanity check — 0-byte file means wget silently failed on redirect chain.
size=$(stat -c '%s' "$FOUNDATION")
if [ "$size" -lt 1000000 ]; then
    echo "ERROR: chemeleon_mp.pt is only ${size} bytes — expected > 1 MB." >&2
    echo "  Path: $FOUNDATION" >&2
    echo "  Zenodo URL: $ZENODO_URL" >&2
    exit 1
fi

# ---- Summary ----
echo ""
echo "Done. Layout in: $DST"
du -sh "$DST/foundations" "$DST/models"/* 2>/dev/null | sort -k2
echo ""
echo "Total: $(du -sh "$DST" | cut -f1)"
echo ""
echo "Next: rsync -av \"$DST/\" <NAS-host>:/data/models/openadmet/"
