#!/usr/bin/env bash
# Download ESMFold2 weights from HuggingFace.
# Run before Docker build:
#   ./services/esmfold2-server/scripts/fetch_weights.sh
#
# Requires: pip install huggingface-hub
# Weights are downloaded to services/esmfold2-server/weights/.
# Two models required:
#   1. biohub/ESMFold2    — main model (config + safetensors + ccd.pkl)
#   2. biohub/ESMC-6B     — ESMC 6B language model (loaded internally by ESMFold2)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DST_ESMFOLD2="$SERVICE_DIR/weights/esmfold2"
DST_ESMC="$SERVICE_DIR/weights/esmc-6b"

mkdir -p "$DST_ESMFOLD2" "$DST_ESMC"

echo "=== Downloading ESMFold2 weights ==="
hf download biohub/ESMFold2 \
    --local-dir "$DST_ESMFOLD2"

echo ""
echo "=== Downloading ESMC-6B weights (required by ESMFold2) ==="
hf download biohub/ESMC-6B \
    --local-dir "$DST_ESMC"

echo ""
echo "Done! Weights in: $SERVICE_DIR/weights/"
echo "ESMFold2:"
ls -lh "$DST_ESMFOLD2"
echo "ESMC-6B:"
ls -lh "$DST_ESMC"
