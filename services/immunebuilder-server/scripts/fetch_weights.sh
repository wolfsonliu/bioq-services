#!/usr/bin/env bash
# Download ImmuneBuilder model weights from Zenodo.
# Run before Docker build:
#   ./services/immunebuilder-server/scripts/fetch_weights.sh
#
# Downloads 16 model files (~600 MB total) into
# services/immunebuilder-server/trained_model/
set -euo pipefail

DST="$(cd "$(dirname "$0")/.." && pwd)/trained_model"
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
