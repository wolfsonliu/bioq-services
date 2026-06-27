#!/usr/bin/env bash
# Download ODesign model checkpoints + supply grnade.h5 from the vendored
# upstream tree.
#
# Weights are NOT baked into the Docker image since v0.0.5 — they live on
# NAS at /data/models/odesign/ckpt/ (FC mount) or bound via apptainer (SIF).
# This script fetches into a stage dir; upload the result afterward.
#
# Default (local stage dir):
#   ./services/odesign-server/scripts/fetch_weights.sh
#       → services/odesign-server/weights/ckpt/
#
# Custom destination (e.g. direct to NAS):
#   ./services/odesign-server/scripts/fetch_weights.sh /mnt/nas/data/models/odesign/ckpt
# or via env:
#   WEIGHTS_DST=/mnt/nas/data/models/odesign/ckpt \
#     ./services/odesign-server/scripts/fetch_weights.sh

set -euo pipefail

DST="${1:-${WEIGHTS_DST:-$(dirname "$0")/../weights/ckpt}}"
mkdir -p "$DST"

CKPTS=(
    # ODesign backbone models (HuggingFace)
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_prot_flex.pt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_prot_rigid.pt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_ligand_rigid.pt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_na_rigid.pt?download=true"
    # OInvFold inverse folding models
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_protein.ckpt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_ligand.ckpt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_dna.ckpt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_rna.ckpt?download=true"
    # ProteinMPNN
    "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt"
)

for url in "${CKPTS[@]}"; do
    fname=$(basename "$url" | sed 's/?download=true//')
    dest="$DST/$fname"
    if [[ -f "$dest" ]]; then
        echo "  skip (exists): $fname"
        continue
    fi
    echo "  downloading: $fname"
    wget -q --show-progress -O "$dest" "$url" || curl -L -# -o "$dest" "$url"
done

# grnade.h5 ships in upstream git, but weights are now externalized so we
# need it in $DST alongside the HF checkpoints.  Copy from the vendored
# upstream tree.  Vendor first if not present:
#   ./services/odesign-server/scripts/vendor.sh
GRNADE_SRC="$(cd "$(dirname "$0")/.." && pwd)/upstream/ckpt/grnade.h5"
if [[ ! -f "$GRNADE_SRC" ]]; then
    echo "WARNING: $GRNADE_SRC not found — run scripts/vendor.sh first" >&2
elif [[ ! -f "$DST/grnade.h5" ]]; then
    echo "  copying: grnade.h5 from vendored upstream"
    cp "$GRNADE_SRC" "$DST/grnade.h5"
fi

echo "Done. Weights in: $DST"
