#!/usr/bin/env bash
# Download ODesign model checkpoints.
# Run before Docker build:
#   ./services/odesign-server/scripts/fetch_weights.sh [ckpt_dir]

set -euo pipefail

DST="${1:-$(dirname "$0")/../weights/ckpt}"
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

# grnade.h5 is committed to the upstream ODesign repo at the SHA pinned in
# services/odesign-server/Dockerfile (ARG ODESIGN_SHA), so it is supplied
# automatically by the Docker build via `cp /opt/odesign/ODesign/ckpt/grnade.h5
# /opt/odesign/ckpt/grnade.h5` and does NOT need to be pre-staged here.

echo "Done. Weights in: $DST"
