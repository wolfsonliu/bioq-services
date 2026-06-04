#!/usr/bin/env bash
# Download ESMFold2 weights from HuggingFace.
# Run before Docker build:
#   ./services/esmfold2-server/scripts/fetch_weights.sh
#
# Requires: pip install huggingface-hub
# Weights are downloaded to services/esmfold2-server/weights/esmfold2/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DST="$SERVICE_DIR/weights/esmfold2"

mkdir -p "$DST"

echo "Downloading ESMFold2 weights to $DST ..."
huggingface-cli download biohub/ESMFold2 \
    --local-dir "$DST" \
    --local-dir-use-symlinks False

echo ""
echo "Done! Weights in: $DST"
echo "Contents:"
ls -lh "$DST"
