#!/usr/bin/env bash
# Download ESMFold2 weights from HuggingFace.
# Run before Docker build:
#   ./services/esmfold2-server/scripts/fetch_weights.sh
#
# Requires: pip install huggingface-hub
# Weights are downloaded to weights/esmfold2/ (relative to project root).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DST="$PROJECT_ROOT/weights/esmfold2"

mkdir -p "$DST"

echo "Downloading ESMFold2 weights to $DST ..."
huggingface-cli download biohub/ESMFold2 \
    --local-dir "$DST" \
    --local-dir-use-symlinks False

echo ""
echo "Done! Weights in: $DST"
echo "Contents:"
ls -lh "$DST"
