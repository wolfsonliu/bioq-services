#!/usr/bin/env bash
# Download ESM-2 weights for DeepRank-Ab inference.
# Run this BEFORE Docker build:
#
#   ./services/deeprank-ab-server/scripts/fetch_esm_weights.sh
#
# Downloads ~2.6 GB total (model + contact regression weights) into
# services/deeprank-ab-server/weights/esm/. The Dockerfile COPYs them
# into the image so inference.py never needs network access at runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="$(cd "$SCRIPT_DIR/.." && pwd)/weights/esm"
mkdir -p "$DST"

ESM_MODEL="esm2_t33_650M_UR50D"
BASE_URL="https://dl.fbaipublicfiles.com/fair-esm"

echo "Downloading ESM-2 model weights..."
wget -c -q --show-progress \
    -O "$DST/${ESM_MODEL}.pt" \
    "${BASE_URL}/models/${ESM_MODEL}.pt"

echo "Downloading ESM-2 contact regression weights..."
wget -c -q --show-progress \
    -O "$DST/${ESM_MODEL}-contact-regression.pt" \
    "${BASE_URL}/regression/${ESM_MODEL}-contact-regression.pt"

echo
echo "ESM-2 weights downloaded to $DST"
du -sh "$DST"/*
