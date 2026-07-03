#!/usr/bin/env bash
# Download ChemBounce scaffold database + fingerprint .npz files from Zenodo
# (upstream install.sh).
#
# Data is NOT baked into the Docker image — it lives on NAS at
# /data/models/chembounce/data/ (FC mount) or is bound via apptainer (SIF).
# This script fetches into a stage dir; upload the result afterward.
#
# Default (local stage dir):
#   ./services/chembounce-server/scripts/fetch_data.sh
#       → services/chembounce-server/data/
#
# Direct to NAS / HPC scratch:
#   DATA_DST=/mnt/nas/data/models/chembounce/data \
#       ./services/chembounce-server/scripts/fetch_data.sh
#
# Contents (~ multi-GB; full DB + 250mw subset — upstream Zenodo names):
#   Scaffolds_processed.txt                (~hundreds of MB)
#   Scaffolds_processed_mw250.txt          (~MB)
#   scaffold_fingerprints.npz              (~GB)
#   scaffold_fingerprints_mw250.npz        (~tens of MB)
#   ... plus SAscore data if upstream includes it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${DATA_DST:-$SCRIPT_DIR/../data}"
mkdir -p "$DST"

URL_PRIMARY="https://zenodo.org/records/16741967/files/data.zip"
URL_FALLBACK="https://www.dropbox.com/scl/fi/1wlp71fdvjycee8r52wp6/data.zip?rlkey=o328bgjyj2mtyf71khmzceh72&st=89du7ywu&dl=1"

ZIP="$DST/data.zip"
if [[ -f "$DST/scaffold_fingerprints_mw250.npz" ]]; then
    echo "[skip] $DST already populated"
    du -sh "$DST"/* 2>/dev/null | head -10
    exit 0
fi

if [[ ! -f "$ZIP" ]]; then
    echo "[download] data.zip (primary: Zenodo)"
    if ! wget -O "$ZIP" "$URL_PRIMARY"; then
        echo "[fallback] data.zip (Dropbox)"
        wget -O "$ZIP" "$URL_FALLBACK"
    fi
fi

echo "[extract] data.zip → $DST"
unzip -o "$ZIP" -d "$DST"

# Upstream zip has nested structure: data/data/* — flatten.
if [[ -d "$DST/data" ]]; then
    mv "$DST/data/"* "$DST/"
    rmdir "$DST/data"
fi
rm -f "$ZIP"

echo "Done. Data in: $DST"
du -sh "$DST"/* | head -10
