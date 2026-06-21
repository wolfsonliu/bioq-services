#!/usr/bin/env bash
# Download Promera + LigandMPNN weights and tinyprot LMDB caches before Docker build.
#
# Usage:
#   ./services/promera-server/scripts/fetch_weights.sh
#
# Weights are saved to services/promera-server/weights/ (gitignored).
#
# The tinyprot caches (ccd.lmdb, taxonomy.lmdb) are required because
# tinyprot.msa opens taxonomy.lmdb at module import time — without them
# baked into the image the FC instance crashes before the cofold/design
# pipelines can start.  See engineering/guides/promera-tinyprot-cache.md.
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

# --- tinyprot LMDB caches (CCD + taxonomy) ---
#
# Each LMDB is a directory containing a single data.mdb file.  tinyprot
# expects <TINYPROT_CACHE>/{ccd.lmdb,taxonomy.lmdb}/data.mdb and refuses
# to import if taxonomy.lmdb is missing (see tinyprot/msa.py L16).
TINYPROT_DIR="$WEIGHTS_DIR/tinyprot"
mkdir -p "$TINYPROT_DIR/ccd.lmdb" "$TINYPROT_DIR/taxonomy.lmdb"
TINYPROT_BASE="https://huggingface.co/datasets/bjing-mit/tinyprot/resolve/main"
echo "Downloading tinyprot CCD LMDB..."
wget -c -O "$TINYPROT_DIR/ccd.lmdb/data.mdb" "$TINYPROT_BASE/ccd.lmdb/data.mdb"
echo "Downloading tinyprot taxonomy LMDB (~2-5 GB, may take a while)..."
wget -c -O "$TINYPROT_DIR/taxonomy.lmdb/data.mdb" "$TINYPROT_BASE/taxonomy.lmdb/data.mdb"

echo "All weights downloaded to $WEIGHTS_DIR"
ls -lh "$PROMERA_DIR"
ls -lh "$LMPNN_DIR"
ls -lh "$TINYPROT_DIR"/*/
