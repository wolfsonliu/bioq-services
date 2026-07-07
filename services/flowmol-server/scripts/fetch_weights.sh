#!/usr/bin/env bash
# Download FlowMol3 pretrained weights (v3.1) into services/flowmol-server/weights/
# or directly to NAS.
#
# Layout (matches upstream flowmol/trained_models/<name>/):
#   <DST>/trained_models/<variant>/checkpoints/last.ckpt
#   <DST>/trained_models/<variant>/config.yaml
#
# The service expects this layout at FLOWMOL_WEIGHTS_DIR — see
# services/flowmol-server/settings.py.
#
# Default (local stage):
#   ./services/flowmol-server/scripts/fetch_weights.sh
#       → services/flowmol-server/weights/trained_models/<variant>/
#
# Direct to NAS:
#   WEIGHTS_DST=/mnt/nas/data/models/flowmol \
#       ./services/flowmol-server/scripts/fetch_weights.sh
#
# Variant subset (default = 4 primaries):
#   FLOWMOL_VARIANTS="flowmol3,fm3_nodistort,fm3_none,fm3_ahigh" \
#       ./services/flowmol-server/scripts/fetch_weights.sh
#
# Full 22-variant set:
#   FLOWMOL_VARIANTS=all \
#       ./services/flowmol-server/scripts/fetch_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
BASE_URL="https://bits.csb.pitt.edu/files/FlowMol/trained_models_v3.1"

ALL_VARIANTS=(
    flowmol3
    fm3_nodistort fm3_none
    fm3_ahigh fm3_alow fm3_chigh fm3_clow
    fm3_ehigh fm3_elow fm3_xhigh fm3_xlow
    fm3_distort_extreme fm3_distort_highp fm3_distort_hight
    fm3_distort_lowp fm3_distort_lowt
    fm3_fa_highp fm3_fa_highstd fm3_fa_lowp fm3_fa_lowstd
    fm3_scprop_high fm3_scprop_low
)

DEFAULT_VARIANTS="flowmol3,fm3_nodistort,fm3_none,fm3_ahigh"
REQ="${FLOWMOL_VARIANTS:-$DEFAULT_VARIANTS}"

if [[ "$REQ" == "all" ]]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
else
    IFS=',' read -r -a VARIANTS <<< "$REQ"
fi

mkdir -p "$DST/trained_models"

echo "Fetching ${#VARIANTS[@]} variant(s) into $DST/trained_models/"
for v in "${VARIANTS[@]}"; do
    v="$(echo "$v" | xargs)"  # trim whitespace
    [[ -z "$v" ]] && continue
    echo ""
    echo "===  $v  ==="
    # Recursive mirror of the upstream directory. --cut-dirs=3 strips the
    # /files/FlowMol/trained_models_v3.1/ prefix so files land under
    # <DST>/trained_models/<variant>/.
    wget -c -r -np -nH --cut-dirs=3 \
        --reject 'index.html*' \
        -P "$DST/trained_models/" \
        "$BASE_URL/$v/"

    ckpt="$DST/trained_models/$v/checkpoints/last.ckpt"
    cfg="$DST/trained_models/$v/config.yaml"
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: $ckpt missing after download" >&2
        exit 1
    fi
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: $cfg missing after download" >&2
        exit 1
    fi
    du -sh "$DST/trained_models/$v"
done

echo ""
echo "Done."
echo "Total NAS layout:"
du -sh "$DST/trained_models/"
ls -la "$DST/trained_models/"
