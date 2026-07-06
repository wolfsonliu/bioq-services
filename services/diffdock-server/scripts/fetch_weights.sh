#!/usr/bin/env bash
# Stage DiffDock model weights for upload to NAS / HPC scratch.
#
# Three sources to reconcile:
#
#   1. DiffDock-L score + confidence checkpoints — from a github release
#      zip.  We extract into workdir/v1.1/{score_model,confidence_model}/.
#
#   2. ESM-2 t33_650M_UR50D checkpoint (~2.5 GB) — pulled from fair-esm's
#      CDN into the torch.hub cache layout so runtime's ESM loader finds
#      it offline.
#
#   3. ESMFold 3B v1 checkpoint (~5 GB) — same CDN, same directory.  Only
#      required for the protein_sequence input branch; DIFFDOCK_SKIP_ESMFOLD=1
#      skips this file (saves ~5 GB when confidently not needed).
#
# Default (local stage dir for inspection):
#   ./services/diffdock-server/scripts/fetch_weights.sh
#       → services/diffdock-server/weights/
#
# Direct to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/diffdock \
#       ./services/diffdock-server/scripts/fetch_weights.sh
#
# Skip ESMFold weights (protein_sequence branch will fail if not staged):
#   DIFFDOCK_SKIP_ESMFOLD=1 ./services/diffdock-server/scripts/fetch_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"

mkdir -p "$DST"

# --------------------------------------------------------------------------
# 1. DiffDock-L score + confidence checkpoints from github release zip.
#    Layout inside the zip:
#      workdir/v1.1/score_model/best_ema_inference_epoch_model.pt
#      workdir/v1.1/score_model/model_parameters.yml
#      workdir/v1.1/confidence_model/best_model_epoch75.pt
#      workdir/v1.1/confidence_model/model_parameters.yml
# --------------------------------------------------------------------------
DIFFDOCK_ZIP_URL="${DIFFDOCK_ZIP_URL:-https://github.com/gcorso/DiffDock/releases/latest/download/diffdock_models.zip}"
SCORE_CKPT="$DST/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt"
CONF_CKPT="$DST/workdir/v1.1/confidence_model/best_model_epoch75.pt"

if [[ -f "$SCORE_CKPT" && -f "$CONF_CKPT" ]]; then
    echo "==> DiffDock-L checkpoints already present"
else
    echo "==> Downloading diffdock_models.zip from $DIFFDOCK_ZIP_URL ..."
    tmp_dir="$(mktemp -d -t diffdock-weights.XXXXXX)"
    trap "rm -rf '$tmp_dir'" EXIT
    zip_path="$tmp_dir/diffdock_models.zip"
    for i in 1 2 3 4 5; do
        if wget -c -O "$zip_path" "$DIFFDOCK_ZIP_URL"; then
            break
        fi
        [ "$i" = "5" ] && { echo "ERROR: download failed after 5 attempts" >&2; exit 1; }
        echo "  download failed, retrying in $((i*10))s ..."
        sleep $((i*10))
    done
    mkdir -p "$DST"
    ( cd "$DST" && unzip -o "$zip_path" >/dev/null )
    echo "  extracted DiffDock-L checkpoints"
fi

# --------------------------------------------------------------------------
# 2. ESM-2 t33 650M UR50D checkpoint (~2.5 GB) via wget -c (resumable).
#    Also fetch the tiny contact-regression sidecar the fair-esm loader
#    checks for when constructing the model.
# --------------------------------------------------------------------------
ESM_URL="${ESM_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt}"
ESM_CONTACT_URL="${ESM_CONTACT_URL:-https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt}"
ESM_DST_DIR="$DST/esm_cache/hub/checkpoints"
ESM_DST="$ESM_DST_DIR/esm2_t33_650M_UR50D.pt"
ESM_CONTACT_DST="$ESM_DST_DIR/esm2_t33_650M_UR50D-contact-regression.pt"

mkdir -p "$ESM_DST_DIR"
if [[ -f "$ESM_DST" ]]; then
    echo "==> ESM-2 checkpoint already present"
else
    echo "==> Downloading ESM-2 t33_650M checkpoint (~2.5 GB) ..."
    wget -c -O "$ESM_DST" "$ESM_URL"
fi
if [[ -f "$ESM_CONTACT_DST" ]]; then
    echo "==> ESM-2 contact-regression sidecar already present"
else
    echo "==> Downloading ESM-2 contact-regression sidecar (~4 MB) ..."
    wget -c -O "$ESM_CONTACT_DST" "$ESM_CONTACT_URL"
fi

# --------------------------------------------------------------------------
# 3. ESMFold 3B v1 checkpoint (~5 GB).  Only needed when protein_sequence
#    input is used.  Skippable via DIFFDOCK_SKIP_ESMFOLD=1.
# --------------------------------------------------------------------------
if [[ "${DIFFDOCK_SKIP_ESMFOLD:-0}" == "1" ]]; then
    echo "==> DIFFDOCK_SKIP_ESMFOLD=1 — skipping ESMFold checkpoint"
else
    ESMFOLD_URL="${ESMFOLD_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt}"
    ESMFOLD_DST="$ESM_DST_DIR/esmfold_3B_v1.pt"
    if [[ -f "$ESMFOLD_DST" ]]; then
        echo "==> ESMFold checkpoint already present"
    else
        echo "==> Downloading ESMFold 3B v1 checkpoint (~5 GB) ..."
        wget -c -O "$ESMFOLD_DST" "$ESMFOLD_URL"
    fi
fi

echo
echo "Done. Weights + ESM cache in: $DST"
du -sh "$DST"/*
