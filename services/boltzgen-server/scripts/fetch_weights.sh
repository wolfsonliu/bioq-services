#!/usr/bin/env bash
# Download BoltzGen model weights + molecule data from HuggingFace.
#
# Run before Docker build:
#   ./services/boltzgen-server/scripts/fetch_weights.sh
#
# Outputs (all gitignored):
#   services/boltzgen-server/weights/   — 5 model checkpoints (~10 GB)
#   services/boltzgen-server/moldir/    — CCD molecule .pkl files (~6 GB)
#
# Uses wget -c for resumable downloads.
#
# Models:
#   design-diverse   — boltzgen1_diverse.ckpt     (structure-conditioned design, diverse)
#   design-adherence — boltzgen1_adherence.ckpt   (structure-conditioned design, adherence)
#   inverse-fold     — boltzgen1_ifold.ckpt       (inverse folding)
#   folding          — boltz2_conf_final.ckpt     (structure prediction / folding)
#   affinity         — boltz2_aff.ckpt            (affinity prediction)
#
# Data:
#   mols.zip → moldir/  — CCD molecule library (required for non-canonical ligands)

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WEIGHTS_DIR="$SCRIPT_DIR/../weights"
MOLDIR="$SCRIPT_DIR/../moldir"

MODEL_REPO="boltzgen/boltzgen-1"
MODEL_BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"

DATA_REPO="boltzgen/inference-data"
DATA_BASE_URL="https://huggingface.co/datasets/${DATA_REPO}/resolve/main"

mkdir -p "$WEIGHTS_DIR" "$MOLDIR"

# --- Model checkpoints ---
CKPT_FILES=(
    boltzgen1_diverse.ckpt
    boltzgen1_adherence.ckpt
    boltzgen1_ifold.ckpt
    boltz2_conf_final.ckpt
    boltz2_aff.ckpt
)

failed=()
for f in "${CKPT_FILES[@]}"; do
    echo "[download] weights/$f"
    if ! wget -c -q --show-progress -O "$WEIGHTS_DIR/$f" "$MODEL_BASE_URL/$f"; then
        echo "[fail] $f" >&2
        failed+=("$f")
    fi
done

# --- Molecule data (mols.zip) ---
MOLS_ZIP="$MOLDIR/mols.zip"
if [ -d "$MOLDIR" ] && ls "$MOLDIR"/*.pkl >/dev/null 2>&1; then
    echo "[skip] moldir already has .pkl files"
else
    echo "[download] mols.zip (~6 GB)"
    if wget -c -q --show-progress -O "$MOLS_ZIP" "$DATA_BASE_URL/mols.zip"; then
        echo "[extract] mols.zip → moldir/"
        unzip -o -q "$MOLS_ZIP" -d "$MOLDIR"
        rm -f "$MOLS_ZIP"
    else
        echo "[fail] mols.zip" >&2
        failed+=("mols.zip")
    fi
fi

if [ "${#failed[@]}" -ne 0 ]; then
    echo
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi

echo
echo "Done."
echo "Weights:"
ls -lh "$WEIGHTS_DIR"
echo "Moldir:"
ls "$MOLDIR" | head -5
echo "  ... ($(ls "$MOLDIR" | wc -l) files total)"
