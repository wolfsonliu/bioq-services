#!/usr/bin/env bash
# Download AbMPNN model weights + dataset splits from Zenodo.
# Run before Docker build:
#   ./services/proteinmpnn-server/scripts/fetch_weights.sh
#
# Source: https://zenodo.org/records/8164693 (arXiv:2310.19513).
#
# Downloads ~23 MB into services/proteinmpnn-server/AbMPNN_model_weights/:
#   - abmpnn.pt           — AbMPNN model weights (~20 MB)
#   - oas_splits.csv      — OAS train/val/test splits (~3.3 MB)
#   - sabdab_splits.csv   — SAbDab train/val/test splits (~84 KB)
#
# The other 3 ProteinMPNN weight sets (vanilla/soluble/ca) are NOT downloaded
# here — they live in the upstream ProteinMPNN git repo at the SHA pinned in
# services/proteinmpnn-server/Dockerfile (ARG PROTEINMPNN_SHA), so the
# Docker build supplies them directly from the upstream clone.
set -euo pipefail

DST="$(cd "$(dirname "$0")/.." && pwd)/AbMPNN_model_weights"
mkdir -p "$DST"

ZENODO="https://zenodo.org/records/8164693/files"
FILES=(
    "abmpnn.pt"
    "oas_splits.csv"
    "sabdab_splits.csv"
)

echo "Downloading AbMPNN weights to $DST ..."

for f in "${FILES[@]}"; do
    dest="$DST/$f"
    if [[ -f "$dest" ]]; then
        echo "  skip (exists): $f"
        continue
    fi
    echo "  downloading: $f"
    wget -c -O "$dest" "${ZENODO}/${f}?download=1" \
        || curl -L -# -o "$dest" "${ZENODO}/${f}?download=1"
done

echo "Done. Weights in: $DST"
ls -lh "$DST"
