#!/usr/bin/env bash
# Download SemlaFlow pretrained checkpoints + reference .smol datasets into
# services/semlaflow-server/weights/ or directly to NAS.
#
# Target NAS layout (matches settings.SemlaFlowSettings + design doc §6):
#   <DST>/<model>/model.ckpt
#   <DST>/<model>/smol/{train,val,test}.smol
#   <DST>/<model>/manifest.yaml           # {dataset: qm9|geom-drugs}
#
# The service expects this at SEMLAFLOW_WEIGHTS_DIR — see
# services/semlaflow-server/settings.py.
#
# Source: upstream provides a Google Drive folder containing per-dataset
# `smol/` data + headline `.ckpt`:
#   https://drive.google.com/drive/folders/1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT
#
# Default (local stage, both models):
#   ./services/semlaflow-server/scripts/fetch_weights.sh
#       → services/semlaflow-server/weights/<model>/
#
# Direct to NAS:
#   WEIGHTS_DST=/mnt/nas/data/models/semlaflow \
#       ./services/semlaflow-server/scripts/fetch_weights.sh
#
# Subset (qm9 only — fast to validate):
#   SEMLAFLOW_MODELS="qm9" ./services/semlaflow-server/scripts/fetch_weights.sh
#
# NOTE: Google Drive folder structure is not a stable public API. This script
# pulls the whole shared folder with `gdown`, then you must place files into
# the NAS layout above. GEOM-drugs `smol/` is GB-scale. The owner has already
# staged NAS per §6; this script exists for rebuild/DR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
DRIVE_FOLDER="${SEMLAFLOW_DRIVE_FOLDER:-1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT}"
REQ="${SEMLAFLOW_MODELS:-qm9,geom-drugs}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "ERROR: gdown not found. Install with: pip install gdown" >&2
    exit 1
fi

mkdir -p "$DST"
STAGING="$DST/_gdrive_raw"
mkdir -p "$STAGING"

echo "Downloading SemlaFlow Google Drive folder $DRIVE_FOLDER → $STAGING"
echo "(this includes QM9 + GEOM-drugs smol/ data + checkpoints; GEOM is large)"
gdown --folder "https://drive.google.com/drive/folders/$DRIVE_FOLDER" \
    -O "$STAGING" --remaining-ok

echo ""
echo "Raw download complete. Contents:"
find "$STAGING" -maxdepth 3 -type f | sort

cat <<EOF

------------------------------------------------------------------------
Next: reorganize the raw download into the NAS layout expected by the
service (per model in "$REQ"):

  $DST/<model>/model.ckpt
  $DST/<model>/smol/{train,val,test}.smol
  $DST/<model>/manifest.yaml    # echo 'dataset: <qm9|geom-drugs>'

Google Drive foldering varies; inspect $STAGING and move the QM9 /
GEOM-drugs 'smol' folders + headline .ckpt into place. Example for qm9:

  mkdir -p "$DST/qm9/smol"
  cp "$STAGING"/.../qm9/smol/*.smol "$DST/qm9/smol/"
  cp "$STAGING"/.../qm9*.ckpt       "$DST/qm9/model.ckpt"
  echo 'dataset: qm9' > "$DST/qm9/manifest.yaml"

Verify with the running service: GET /healthz/detail → each model .ready=true.
------------------------------------------------------------------------
EOF
