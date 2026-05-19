#!/usr/bin/env bash
# Populate services/rfdiffusion2-server/weights/ from
# opensource/RFdiffusion2/rf_diffusion/model_weights/. Run this once before
# `make build-rfdiffusion2-server` on a fresh checkout.
#
# Prereq: opensource/RFdiffusion2/rf_diffusion/model_weights/ must be populated
# (upstream provides `setup.py` to download them; run it once in that dir).
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
SRC="$PROJECT_ROOT/opensource/RFdiffusion2/rf_diffusion/model_weights"
DST="$PROJECT_ROOT/services/rfdiffusion2-server/weights"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: $SRC not found." >&2
    echo "Populate it first by running:" >&2
    echo "    cd $PROJECT_ROOT/opensource/RFdiffusion2 && python setup.py" >&2
    exit 1
fi

if [[ -z "$(ls -A "$SRC" 2>/dev/null)" ]]; then
    echo "ERROR: $SRC is empty." >&2
    exit 1
fi

mkdir -p "$DST"
rsync -a --delete "$SRC/" "$DST/"

echo "Synced weights to $DST"
du -sh "$DST"
ls -lh "$DST"
