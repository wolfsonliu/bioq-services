#!/usr/bin/env bash
# Stage DiffHopp model checkpoints for upload to NAS / HPC scratch.
#
# Unlike most services, DiffHopp ships its 4 checkpoints (~189 MB) **in the
# upstream git repo** under checkpoints/.  vendor.sh already brought them
# into services/diffusion-hopping-server/upstream/checkpoints/; this script
# just copies them into a stage dir / NAS for the externalized layout.
#
# Default (local stage dir for inspection):
#   ./services/diffusion-hopping-server/scripts/fetch_weights.sh
#       → services/diffusion-hopping-server/weights/  (~189 MB, 4 .ckpt)
#
# Direct to NAS / HPC scratch (skip the upload step):
#   WEIGHTS_DST=/mnt/nas/data/models/diffusion-hopping/checkpoints \
#       ./services/diffusion-hopping-server/scripts/fetch_weights.sh
#
# Requires vendor.sh to have run first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/../upstream/checkpoints"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: $SRC not found." >&2
    echo "Run ./services/diffusion-hopping-server/scripts/vendor.sh first." >&2
    exit 1
fi

CKPTS=(
    gvp_conditional.ckpt
    gvp_unconditional.ckpt
    egnn_conditional.ckpt
    egnn_unconditional.ckpt
)

mkdir -p "$DST"
for f in "${CKPTS[@]}"; do
    if [[ ! -f "$SRC/$f" ]]; then
        echo "ERROR: $SRC/$f missing — upstream layout may have changed?" >&2
        exit 1
    fi
    if [[ -f "$DST/$f" ]]; then
        echo "  skip (exists): $f"
    else
        echo "  copying:       $f"
        cp "$SRC/$f" "$DST/$f"
    fi
done

echo "Done. Weights in: $DST"
du -sh "$DST"/*
