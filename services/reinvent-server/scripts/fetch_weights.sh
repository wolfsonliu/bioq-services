#!/usr/bin/env bash
# Download REINVENT4 prior models (Zenodo 10.5281/zenodo.15641296).
#
# Priors are NOT baked into the image — they load from NAS (FC) or
# apptainer --bind (SIF) at /data/models/reinvent via REINVENT_PRIOR_BASE.
# This script downloads to a stage dir; deploy by rsync to NAS / HPC scratch.
#
# Default (local stage):
#   ./services/reinvent-server/scripts/fetch_weights.sh   -> services/reinvent-server/weights/
# Direct to NAS:
#   WEIGHTS_DST=/mnt/nas/data/models/reinvent ./services/reinvent-server/scripts/fetch_weights.sh
#
# NOTE: the exact Zenodo file URLs must be confirmed on first run (README gives
# only the DOI). Resolve the record, then fill ZENODO_BASE + FILES below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${WEIGHTS_DST:-${HERE}/weights}"
ZENODO_BASE="${ZENODO_BASE:-https://zenodo.org/records/20701824/files}"

FILES=(
    libinvent.prior
    libinvent_transformer_pubchem.prior
    linkinvent.prior
    linkinvent_transformer_pubchem.prior
    pepinvent.prior
    pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior
    reinvent_pubchem.prior
)

mkdir -p "${DST}"
for f in "${FILES[@]}"; do
    echo "Fetching ${f} ..."
    wget -c -O "${DST}/${f}" "${ZENODO_BASE}/${f}?download=1"
done
echo "Priors staged at ${DST}. rsync to NAS: rsync -av ${DST}/ <nas>:/data/models/reinvent/"
