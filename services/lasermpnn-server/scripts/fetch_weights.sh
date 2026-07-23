#!/usr/bin/env bash
# Stage LASErMPNN model weights.
#
# Weights are NOT baked into the image — they load from NAS (FC) or via
# apptainer --bind (SIF). This script only stages them; deploy then rsyncs to
# NAS / HPC scratch.
#
# The weights are committed in the upstream git repo's model_weights/ dir
# (~260 MB), so we do a blob-filtered sparse checkout of just that dir and copy
# the four inference checkpoints out.
#
# Default (local stage):
#   ./services/lasermpnn-server/scripts/fetch_weights.sh
#       -> services/lasermpnn-server/weights/
#
# Straight to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/lasermpnn \
#       ./services/lasermpnn-server/scripts/fetch_weights.sh

set -euo pipefail

LASERMPNN_REPO="${LASERMPNN_REPO:-https://github.com/polizzilab/LASErMPNN.git}"
LASERMPNN_SHA="${LASERMPNN_SHA:-5df210fced6764d83f01425d1fc4319a22b70c2a}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
TMP="$(mktemp -d -t lasermpnn-weights.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

FILES=(
    "laser_weights_0p1A_nothing_heldout.pt"
    "laser_weights_0p1A_noise_ligandmpnn_split.pt"
    "soluble_weights_no_heldout_drop_clusters_optstep_65000.pt"
    "pretrained_ligand_encoder_weights.pt"
)

# Skip the clone entirely if every file is already staged.
missing=0
for f in "${FILES[@]}"; do
    [[ -f "$DST/$f" ]] || missing=1
done
if [[ "$missing" -eq 0 ]]; then
    echo "All weights already staged in $DST — nothing to do."
    ls -lh "$DST"
    exit 0
fi

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$LASERMPNN_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git sparse-checkout init --cone
git sparse-checkout set model_weights
git checkout "$LASERMPNN_SHA"

echo "Staging weights to $DST ..."
for f in "${FILES[@]}"; do
    src="$TMP/repo/model_weights/$f"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: expected weight not found in upstream: model_weights/$f" >&2
        exit 1
    fi
    cp -v "$src" "$DST/$f"
done

echo "Done. Weights in: $DST"
ls -lh "$DST"
