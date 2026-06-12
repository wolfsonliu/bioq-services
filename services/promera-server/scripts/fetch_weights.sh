#!/usr/bin/env bash
# Download Promera + LigandMPNN weights before Docker build.
#
# Usage:
#   ./services/promera-server/scripts/fetch_weights.sh
#
# Weights are saved to services/promera-server/weights/ (gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/../weights"

# --- Promera checkpoint ---
PROMERA_DIR="$WEIGHTS_DIR/promera"
mkdir -p "$PROMERA_DIR"
echo "Downloading Promera checkpoint..."
wget -c -O "$PROMERA_DIR/promera_2606.ckpt" \
    "https://huggingface.co/bjing-mit/promera/resolve/main/promera_2606.ckpt"

# --- LigandMPNN model_params ---
LMPNN_DIR="$WEIGHTS_DIR/ligandmpnn"
mkdir -p "$LMPNN_DIR"
echo "Downloading LigandMPNN model params..."
LMPNN_BASE="https://github.com/dauparas/LigandMPNN/raw/main/model_params"
for f in proteinmpnn_v_48_020.pt \
         solublempnn_v_48_020.pt \
         ligandmpnn_v_32_010_25.pt \
         ligandmpnn_sc_v_32_002_16.pt \
         per_residue_label_membrane_mpnn_v_48_020.pt \
         global_label_membrane_mpnn_v_48_020.pt; do
    wget -c -O "$LMPNN_DIR/$f" "$LMPNN_BASE/$f"
done

echo "All weights downloaded to $WEIGHTS_DIR"
ls -lh "$PROMERA_DIR"
ls -lh "$LMPNN_DIR"
