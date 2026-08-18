#!/usr/bin/env bash
# Download the RFdiffusion2 diffusion checkpoints into
# services/rfdiffusion2-server/weights/. Run this once before
# `make build-rfdiffusion2-server` on a fresh checkout.
#
#   ./services/rfdiffusion2-server/scripts/fetch_weights.sh
#
# Download straight to NAS (recommended for FC staging — weights are baked
# into the image today, see Dockerfile):
#
#   WEIGHTS_DST=/mnt/nas/data/models/rfdiffusion2 \
#     ./services/rfdiffusion2-server/scripts/fetch_weights.sh
#
# Sources the same UW file host that upstream's setup.py reads
# (https://files.ipd.uw.edu/pub/rfdiffusion2/), so this no longer depends on a
# local opensource/RFdiffusion2/ checkout having run `setup.py` beforehand.
#
# Upstream publishes no checksums; `wget -c` resumes partial downloads and we
# only require each file to be non-empty afterward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
BASE_URL="https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights"

# RFdiffusion2 ships two diffusion checkpoints today. The default config
# (aa.yaml) points at RFD_140.pt; RFD_173.pt is the newer benchmark model,
# selected via `model=rfd_173`. The LigandMPNN weights under
# rf_diffusion/third_party_model_weights/ are consumed by downstream sequence
# design, not by run_inference.py itself, so they are NOT fetched here.
FILES=(RFD_140.pt RFD_173.pt)

mkdir -p "$DST"

for f in "${FILES[@]}"; do
    dest="$DST/$f"
    if [[ -s "$dest" ]]; then
        echo "  skip (already present): $f"
        continue
    fi
    echo "  downloading: $f"
    wget -c -O "$dest" "$BASE_URL/$f" \
        || curl -L -# -o "$dest" "$BASE_URL/$f"
done

echo "Weights in: $DST"
ls -lh "$DST"