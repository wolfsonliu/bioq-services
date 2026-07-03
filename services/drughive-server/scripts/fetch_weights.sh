#!/usr/bin/env bash
# Download DrugHIVE model checkpoint from Zenodo.
#
# Weights are NOT baked into the Docker image — they live on NAS (FC) or get
# bound via apptainer (SIF).  This script fetches to a stage dir; upload the
# result to NAS / HPC scratch afterward.
#
# Zenodo record: 10.5281/zenodo.12668687
#
# Default (local stage dir):
#   ./services/drughive-server/scripts/fetch_weights.sh
#       → services/drughive-server/weights/drughive_model_ch9.ckpt
#
# Direct download to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/drughive/checkpoints \
#     ./services/drughive-server/scripts/fetch_weights.sh
set -euo pipefail

DST="${WEIGHTS_DST:-$(cd "$(dirname "$0")/.." && pwd)/weights}"
mkdir -p "$DST"

ZENODO_URL="https://zenodo.org/records/12668687/files/drughive_model_ch9.ckpt"

echo "Downloading DrugHIVE checkpoint to $DST ..."
wget -c -O "$DST/drughive_model_ch9.ckpt" "$ZENODO_URL"

# Sanity check — 0-byte file means wget silently failed on redirect chain.
size=$(stat -c '%s' "$DST/drughive_model_ch9.ckpt")
if [ "$size" -lt 1000000 ]; then
    echo "ERROR: downloaded checkpoint is only ${size} bytes — expected > 1 MB." >&2
    echo "  Path: $DST/drughive_model_ch9.ckpt" >&2
    echo "  Zenodo URL: $ZENODO_URL" >&2
    exit 1
fi

echo "Done: $DST/drughive_model_ch9.ckpt ($(du -h "$DST/drughive_model_ch9.ckpt" | cut -f1))"
