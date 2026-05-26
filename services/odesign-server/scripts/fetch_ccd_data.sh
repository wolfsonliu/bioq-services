#!/usr/bin/env bash
# Download CCD component data required by ODesign.
# Run before Docker build:
#   ./services/odesign-server/scripts/fetch_ccd_data.sh [data_dir]
#
# The two files (~2 GB total) must be placed in DATA_ROOT_DIR:
#   - components.v20240608.cif
#   - components.v20240608.cif.rdkit_mol.pkl
#
# Source: Google Drive folder linked in ODesign README:
#   https://drive.google.com/drive/folders/1wPmwIrC3G52q1JFY0RXY95tjKDl7YEln

set -euo pipefail

DST="${1:-$(dirname "$0")/../weights/data}"
mkdir -p "$DST"

echo "CCD data must be downloaded manually from Google Drive:"
echo "  https://drive.google.com/drive/folders/1wPmwIrC3G52q1JFY0RXY95tjKDl7YEln"
echo ""
echo "Place the following files in: $DST"
echo "  - components.v20240608.cif"
echo "  - components.v20240608.cif.rdkit_mol.pkl"
echo ""

for f in "components.v20240608.cif" "components.v20240608.cif.rdkit_mol.pkl"; do
    if [[ -f "$DST/$f" ]]; then
        echo "  found: $f"
    else
        echo "  MISSING: $f"
    fi
done
