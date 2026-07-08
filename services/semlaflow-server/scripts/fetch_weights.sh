#!/usr/bin/env bash
# Download SemlaFlow pretrained checkpoints + reference .smol datasets and
# reorganize them into the NAS layout the service expects.
#
# Target NAS layout (matches settings.SemlaFlowSettings + design doc §6):
#   <DST>/<model>/model.ckpt
#   <DST>/<model>/smol/{train,val,test}.smol
#   <DST>/<model>/manifest.yaml           # {dataset: qm9|geom-drugs}
#
# The service expects this at SEMLAFLOW_WEIGHTS_DIR — see
# services/semlaflow-server/settings.py.
#
# Source: upstream Google Drive folder, whose raw layout is:
#   _gdrive_raw/data/<model>/smol/{train,val,test}.smol
#   _gdrive_raw/models/<model>/<NNNepochs>.ckpt
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
# Reorganize an already-downloaded _gdrive_raw (skip the gdown step):
#   SEMLAFLOW_SKIP_DOWNLOAD=1 ./services/semlaflow-server/scripts/fetch_weights.sh
#
# NOTE: GEOM-drugs `smol/` is GB-scale. The owner has already staged NAS per
# §6; this script exists for rebuild/DR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
STAGING="${SEMLAFLOW_STAGING:-$DST/_gdrive_raw}"
DRIVE_FOLDER="${SEMLAFLOW_DRIVE_FOLDER:-1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT}"
REQ="${SEMLAFLOW_MODELS:-qm9,geom-drugs}"
SKIP_DOWNLOAD="${SEMLAFLOW_SKIP_DOWNLOAD:-0}"

SPLITS=(train val test)

# ---------------------------------------------------------------------------
# 1. Download the Google Drive folder into STAGING (unless skipped).
# ---------------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    if ! command -v gdown >/dev/null 2>&1; then
        echo "ERROR: gdown not found. Install with: pip install gdown" >&2
        echo "       (or set SEMLAFLOW_SKIP_DOWNLOAD=1 to reorganize an" >&2
        echo "        already-downloaded $STAGING)" >&2
        exit 1
    fi
    mkdir -p "$STAGING"
    echo "Downloading SemlaFlow Google Drive folder $DRIVE_FOLDER → $STAGING"
    echo "(includes QM9 + GEOM-drugs smol/ data + checkpoints; GEOM is large)"
    gdown --folder "https://drive.google.com/drive/folders/$DRIVE_FOLDER" \
        -O "$STAGING" --continue
else
    echo "SEMLAFLOW_SKIP_DOWNLOAD=1 — reorganizing existing $STAGING"
fi

if [[ ! -d "$STAGING" ]]; then
    echo "ERROR: staging dir not found: $STAGING" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Reorganize STAGING (data/<m>/smol + models/<m>/*.ckpt) into NAS layout.
#
# Raw layout from the Drive folder:
#   $STAGING/data/<model>/smol/{train,val,test}.smol
#   $STAGING/models/<model>/<something>.ckpt   (e.g. 300epochs.ckpt)
# Target:
#   $DST/<model>/model.ckpt
#   $DST/<model>/smol/{train,val,test}.smol
#   $DST/<model>/manifest.yaml
# For the two headline models the directory name IS the dataset kind
# (qm9 | geom-drugs), so manifest `dataset:` == model name.
# ---------------------------------------------------------------------------
IFS=',' read -r -a MODELS <<< "$REQ"

echo ""
echo "Reorganizing into NAS layout under $DST"
for m in "${MODELS[@]}"; do
    m="$(echo "$m" | xargs)"  # trim whitespace
    [[ -z "$m" ]] && continue
    echo ""
    echo "===  $m  ==="

    src_smol="$STAGING/data/$m/smol"
    src_ckpt_dir="$STAGING/models/$m"
    dst_model="$DST/$m"

    if [[ ! -d "$src_smol" ]]; then
        echo "ERROR: reference splits not found: $src_smol" >&2
        exit 1
    fi
    if [[ ! -d "$src_ckpt_dir" ]]; then
        echo "ERROR: checkpoint dir not found: $src_ckpt_dir" >&2
        exit 1
    fi

    mkdir -p "$dst_model/smol"

    # Copy the .smol splits (train.smol is mandatory — novelty reference).
    for s in "${SPLITS[@]}"; do
        if [[ -f "$src_smol/$s.smol" ]]; then
            cp -f "$src_smol/$s.smol" "$dst_model/smol/$s.smol"
        else
            echo "WARN: $src_smol/$s.smol missing" >&2
        fi
    done
    if [[ ! -f "$dst_model/smol/train.smol" ]]; then
        echo "ERROR: train.smol is required (novelty reference) but missing" >&2
        exit 1
    fi

    # Pick the checkpoint (upstream ships a single <NNNepochs>.ckpt per model).
    ckpt="$(find "$src_ckpt_dir" -maxdepth 1 -type f -name '*.ckpt' | sort | head -n1)"
    if [[ -z "$ckpt" ]]; then
        echo "ERROR: no .ckpt found in $src_ckpt_dir" >&2
        exit 1
    fi
    cp -f "$ckpt" "$dst_model/model.ckpt"

    # dataset kind == model name for the two headline models.
    echo "dataset: $m" > "$dst_model/manifest.yaml"

    echo "  ckpt: $(basename "$ckpt") -> model.ckpt"
    echo "  smol: $(ls "$dst_model/smol")"
    du -sh "$dst_model"
done

echo ""
echo "Done. NAS layout under $DST:"
for m in "${MODELS[@]}"; do
    m="$(echo "$m" | xargs)"; [[ -z "$m" ]] && continue
    echo "  $m/  ($(du -sh "$DST/$m" | cut -f1))"
done
echo ""
echo "Staging ($STAGING) left in place — remove it manually once verified:"
echo "  rm -rf \"$STAGING\""
echo ""
echo "Verify with the running service: GET /healthz/detail → each model .ready=true"
