#!/usr/bin/env bash
# Download ImmuneBuilder model weights from Zenodo.
#
# Weights are NOT baked into the Docker image — they live on NAS (FC) or get
# bound via apptainer (SIF).  This script fetches to a stage dir; upload the
# result to NAS / HPC scratch afterward.
#
# Default (local stage dir):
#   ./services/immunebuilder-server/scripts/fetch_weights.sh
#       → services/immunebuilder-server/trained_model/  (~600 MB, 16 files)
#
# Direct download to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/immunebuilder/trained_model \
#     ./services/immunebuilder-server/scripts/fetch_weights.sh
set -euo pipefail

DST="${WEIGHTS_DST:-$(cd "$(dirname "$0")/.." && pwd)/trained_model}"
mkdir -p "$DST"

ZENODO_V1="https://zenodo.org/record/7258553/files"
ZENODO_TCR_PLUS="https://zenodo.org/records/10892159/files"

echo "Downloading ImmuneBuilder weights to $DST ..."

# ABodyBuilder2 (4 models)
for i in 1 2 3 4; do
    wget -c -O "$DST/antibody_model_$i" "${ZENODO_V1}/antibody_model_${i}?download=1"
done

# NanoBodyBuilder2 (4 models)
for i in 1 2 3 4; do
    wget -c -O "$DST/nanobody_model_$i" "${ZENODO_V1}/nanobody_model_${i}?download=1"
done

# TCRBuilder2+ (default, 4 models)
for i in 1 2 3 4; do
    wget -c -O "$DST/tcr_model_$i" "${ZENODO_TCR_PLUS}/tcr_model_${i}?download=1"
done

# TCRBuilder2 original (4 models)
for i in 1 2 3 4; do
    wget -c -O "$DST/tcr2_model_$i" "${ZENODO_V1}/tcr_model_${i}?download=1"
done

echo "Done. $(find "$DST" -type f ! -name '.gitkeep' | wc -l) weight files downloaded."
