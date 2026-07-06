#!/usr/bin/env bash
# Fetch PocketXMol model weights from Zenodo.
#
# Weights are NOT baked into the image — they live on NAS at
# /data/models/pocketxmol/ (FC mount) or bound in via
# `apptainer run --bind /scratch/models/pocketxmol:/data/models/pocketxmol`
# for SIF / HPC.  See
# engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Default (local stage dir for inspection):
#   ./services/pocketxmol-server/scripts/fetch_weights.sh
#       → services/pocketxmol-server/weights/  (~500 MB after extract)
#
# Direct to NAS / HPC scratch (skip the local stage upload step):
#   WEIGHTS_DST=/mnt/nas/data/models/pocketxmol \
#       ./services/pocketxmol-server/scripts/fetch_weights.sh
#
# The archive `model_weights.tar.gz` bundles three checkpoints:
#   pxm/checkpoints/pocketxmol.ckpt          — main foundation model
#   tuned_ranker/checkpoints/tuned_ranker.ckpt   — /api/confidence tuned_cfd
#   flex_cfd/checkpoints/flex_cfd.ckpt            — /api/confidence flex_cfd
# plus per-checkpoint train_config/*.yml that sample_use.py reads to
# reconstruct network shape.
#
# The CCD dictionary (data/ccd/, ~1.9 MB) is optionally re-staged to NAS
# so the future sdf2pdb_robust endpoint (not in v0.0.1) can pick it up.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"

ZENODO_RECORD_URL="${ZENODO_RECORD_URL:-https://zenodo.org/records/17801271/files/model_weights.tar.gz}"

mkdir -p "$DST"

TAR="$DST/model_weights.tar.gz"
if [[ -f "$TAR" ]]; then
    echo "  skip download (exists): $TAR"
else
    echo "  downloading: $ZENODO_RECORD_URL"
    # wget -c supports resume.
    wget -c -O "$TAR" "$ZENODO_RECORD_URL"
fi

echo "  extracting: $TAR"
# Upstream tarball root is `data/trained_models/<name>/checkpoints/…`.
# We strip the leading `data/trained_models/` so the extracted layout
# lives directly under $DST (matching the settings.py defaults:
# /data/models/pocketxmol/pxm/checkpoints/pocketxmol.ckpt).
tar -xzf "$TAR" -C "$DST" --strip-components=2

echo "  extracted layout:"
find "$DST" -mindepth 1 -maxdepth 3 -type d | head -20

# Verify the three checkpoint files we care about are present.
REQUIRED_CKPTS=(
    "pxm/checkpoints/pocketxmol.ckpt"
    "tuned_ranker/checkpoints/tuned_ranker.ckpt"
    "flex_cfd/checkpoints/flex_cfd.ckpt"
)
for rel in "${REQUIRED_CKPTS[@]}"; do
    if [[ ! -f "$DST/$rel" ]]; then
        echo "WARNING: expected checkpoint missing: $DST/$rel" >&2
        echo "         upstream layout may have changed; inspect $DST and update settings.py" >&2
    fi
done

# Also stage the CCD dictionary (small, used by future sdf2pdb_robust endpoint).
SRC_CCD="$SCRIPT_DIR/../upstream/data/ccd"
DST_CCD="$DST/ccd"
if [[ -d "$SRC_CCD" && ! -d "$DST_CCD" ]]; then
    echo "  copying ccd/: $SRC_CCD -> $DST_CCD"
    cp -r "$SRC_CCD" "$DST_CCD"
fi

echo "Done. Weights in: $DST"
du -sh "$DST"/*
