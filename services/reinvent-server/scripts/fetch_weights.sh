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
ZENODO_BASE="${ZENODO_BASE:-https://zenodo.org/records/15641296/files}"

FILES=(
    reinvent.prior
    libinvent.prior
    linkinvent.prior
    mol2mol_high_similarity.prior
    mol2mol_medium_similarity.prior
    mol2mol_mmp.prior
    mol2mol_scaffold.prior
    mol2mol_scaffold_generic.prior
    pepinvent.prior
)

mkdir -p "${DST}"
for f in "${FILES[@]}"; do
    echo "Fetching ${f} ..."
    wget -c -O "${DST}/${f}" "${ZENODO_BASE}/${f}?download=1"
done
echo "Priors staged at ${DST}. rsync to NAS: rsync -av ${DST}/ <nas>:/data/models/reinvent/"
