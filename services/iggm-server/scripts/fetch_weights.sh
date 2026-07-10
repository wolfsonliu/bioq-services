#!/usr/bin/env bash
# Download IgGM pretrained checkpoints (5 .pth) into the NAS layout the service
# expects.
#
# Target NAS layout (matches settings.IgGMSettings + design doc §6):
#   <DST>/esm_ppi_650m_ab.pth
#   <DST>/antibody_design_trunk.pth
#   <DST>/antibody_inverse_design_trunk.pth
#   <DST>/antibody_fr_design_trunk.pth
#   <DST>/igso3_buffer.pth
#
# The service expects this at IGGM_WEIGHTS_DIR (default /data/models/iggm),
# symlinked to /opt/iggm/checkpoints in the image so upstream's hardcoded
# ./checkpoints/<name>.pth resolves without a runtime torch.hub download.
#
# Source: Zenodo record 16909543
#   https://zenodo.org/records/16909543/files/<name>.pth?download=1
#
# Default (local stage):
#   ./services/iggm-server/scripts/fetch_weights.sh
#       → services/iggm-server/weights/
#
# Direct to NAS:
#   WEIGHTS_DST=/mnt/nas/data/models/iggm \
#       ./services/iggm-server/scripts/fetch_weights.sh
#
# Subset (only the checkpoints you need):
#   IGGM_CKPTS="esm_ppi_650m_ab,antibody_design_trunk,igso3_buffer" \
#       ./services/iggm-server/scripts/fetch_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
ZENODO_RECORD="${IGGM_ZENODO_RECORD:-16909543}"
REQ="${IGGM_CKPTS:-esm_ppi_650m_ab,antibody_design_trunk,antibody_inverse_design_trunk,antibody_fr_design_trunk,igso3_buffer}"

mkdir -p "$DST"

IFS=',' read -r -a CKPTS <<< "$REQ"

echo "Downloading IgGM checkpoints from Zenodo record $ZENODO_RECORD → $DST"
for name in "${CKPTS[@]}"; do
    name="$(echo "$name" | xargs)"  # trim
    [[ -z "$name" ]] && continue
    dst="$DST/$name.pth"
    if [[ -f "$dst" ]]; then
        echo "  $name.pth already present, skipping"
        continue
    fi
    url="https://zenodo.org/records/$ZENODO_RECORD/files/$name.pth?download=1"
    echo "  fetching $name.pth ..."
    # -C - resumes partial downloads; --fail turns HTTP errors into nonzero rc.
    curl -fL -C - -o "$dst" "$url" || {
        echo "ERROR: failed to download $url" >&2
        rm -f "$dst"
        exit 1
    }
done

echo ""
echo "Done. NAS layout under $DST:"
ls -la "$DST"/*.pth 2>/dev/null || true
echo ""
echo "Verify with the running service: GET /healthz/detail → weights_loaded=true"
