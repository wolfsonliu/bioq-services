#!/usr/bin/env bash
# Stage Megalodon checkpoints (+ optional statistics bundle) into the NAS
# layout the service expects.
#
# Target NAS layout (matches settings.MegalodonSettings + design doc §6):
#   <DST>/ckpts/<dataset>/*.ckpt          # <dataset> = qm9 | drugs
#   <DST>/stats/<dataset>/<statistics files>
#
# The service reads this at MEGALODON_WEIGHTS_DIR — see
# services/megalodon-server/settings.py.
#
# ---------------------------------------------------------------------------
# CHECKPOINTS (default step) — HuggingFace, checkpoints only
# ---------------------------------------------------------------------------
# Source repos (each contains only *.ckpt + README — NO statistics).
# NOTE: the diffusion checkpoint filename DIFFERS per dataset.
#   nvidia/NV-Megalodon-QM9-v1        -> ckpts/qm9/
#     megalodon_diffusion.ckpt, megalodon_fm.ckpt, megalodon_small_diffusion.ckpt
#   nvidia/NV-Megalodon-GEOM-Drugs-v1 -> ckpts/drugs/
#     megalodon_large_diffusion.ckpt, megalodon_fm.ckpt, megalodon_small_diffusion.ckpt
#
# The service's model_name -> (dataset, ckpt file) mapping is:
#   qm9_diffusion   -> qm9/megalodon_diffusion.ckpt
#   drugs_diffusion -> drugs/megalodon_large_diffusion.ckpt
#   {qm9,drugs}_fm    -> <dataset>/megalodon_fm.ckpt
#   {qm9,drugs}_quick -> <dataset>/megalodon_small_diffusion.ckpt
#
# Two ways to provide the checkpoints:
#   (a) point at pre-downloaded HF snapshot dirs (recommended — you already
#       downloaded on the server):
#         MEGALODON_QM9_SRC=/path/to/NV-Megalodon-QM9-v1 \
#         MEGALODON_DRUGS_SRC=/path/to/NV-Megalodon-GEOM-Drugs-v1 \
#             ./scripts/fetch_weights.sh
#   (b) let the script download via hf (needs `pip install -U
#       huggingface_hub` and network):
#         ./scripts/fetch_weights.sh          # downloads both repos
#
# Direct to NAS:
#   WEIGHTS_DST=/mnt/nas/data/models/megalodon ./scripts/fetch_weights.sh
#
# Subset (qm9 only — drugs still downloading):
#   MEGALODON_MODELS="qm9" MEGALODON_QM9_SRC=/path/to/qm9 ./scripts/fetch_weights.sh
#
# ---------------------------------------------------------------------------
# STATISTICS (optional step, MEGALODON_FETCH_STATS=1)
# ---------------------------------------------------------------------------
# The HF repos ship checkpoints ONLY. Sampling + the full metric suite need a
# per-dataset statistics bundle, which is NOT distributed — it is produced by
# the upstream preprocessing scripts (data_processing/). This is a heavy,
# one-time offline job (downloads GB-scale raw GEOM-Drugs / QM9, runs the
# `megalodon` conda env). Run it explicitly with MEGALODON_FETCH_STATS=1 and
# MEGALODON_STATS_SRC pointing at an already-processed `processed/` dir, OR
# read the documented commands below and run preprocessing yourself.
#
# See design doc §6 / §12 risk 1 for the full rationale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
REQ="${MEGALODON_MODELS:-qm9,drugs}"
FETCH_STATS="${MEGALODON_FETCH_STATS:-0}"

# HF repo per dataset kind.
declare -A HF_REPO=(
    [qm9]="nvidia/NV-Megalodon-QM9-v1"
    [drugs]="nvidia/NV-Megalodon-GEOM-Drugs-v1"
)
# Pre-downloaded HF snapshot dir per dataset (skip download if set).
declare -A SRC_DIR=(
    [qm9]="${MEGALODON_QM9_SRC:-}"
    [drugs]="${MEGALODON_DRUGS_SRC:-}"
)

# Canonical checkpoint file names per dataset (diffusion name differs).
declare -A EXPECTED_CKPTS=(
    [qm9]="megalodon_diffusion.ckpt megalodon_fm.ckpt megalodon_small_diffusion.ckpt"
    [drugs]="megalodon_large_diffusion.ckpt megalodon_fm.ckpt megalodon_small_diffusion.ckpt"
)

IFS=',' read -r -a MODELS <<< "$REQ"

# ---------------------------------------------------------------------------
# 1. Checkpoints -> <DST>/ckpts/<dataset>/
# ---------------------------------------------------------------------------
echo "Staging Megalodon checkpoints under $DST/ckpts"
for ds in "${MODELS[@]}"; do
    ds="$(echo "$ds" | xargs)"; [[ -z "$ds" ]] && continue
    if [[ -z "${HF_REPO[$ds]:-}" ]]; then
        echo "ERROR: unknown dataset '$ds' (want qm9 | drugs)" >&2; exit 1
    fi
    echo ""
    echo "===  $ds  (${HF_REPO[$ds]})  ==="

    src="${SRC_DIR[$ds]}"
    if [[ -z "$src" ]]; then
        # Download the HF snapshot into a staging dir.
        if ! command -v hf >/dev/null 2>&1; then
            echo "ERROR: hf not found and MEGALODON_${ds^^}_SRC unset." >&2
            echo "       Install (pip install -U huggingface_hub) or point at a" >&2
            echo "       pre-downloaded snapshot dir." >&2
            exit 1
        fi
        src="$DST/_hf_$ds"
        echo "Downloading ${HF_REPO[$ds]} -> $src"
        hf download "${HF_REPO[$ds]}" \
            --local-dir "$src"
    else
        echo "Using pre-downloaded snapshot: $src"
    fi
    if [[ ! -d "$src" ]]; then
        echo "ERROR: source dir not found: $src" >&2; exit 1
    fi

    dst_ckpt="$DST/ckpts/$ds"
    mkdir -p "$dst_ckpt"

    # Copy every *.ckpt (preserve names; the service maps names -> variants).
    shopt -s nullglob
    found=0
    for f in "$src"/*.ckpt "$src"/**/*.ckpt; do
        [[ -f "$f" ]] || continue
        cp -f "$f" "$dst_ckpt/$(basename "$f")"
        echo "  + $(basename "$f")"
        found=$((found + 1))
    done
    shopt -u nullglob
    if [[ "$found" -eq 0 ]]; then
        echo "ERROR: no *.ckpt found under $src" >&2; exit 1
    fi

    # Warn on any missing canonical name for this dataset (non-fatal).
    for want in ${EXPECTED_CKPTS[$ds]}; do
        [[ -f "$dst_ckpt/$want" ]] || echo "  WARN: expected $want not present in $ds" >&2
    done
    du -sh "$dst_ckpt"
done

# ---------------------------------------------------------------------------
# 2. Statistics bundle -> <DST>/stats/<dataset>/  (optional, heavy)
# ---------------------------------------------------------------------------
# save_statistics (data_processing/utils_data.py) writes ALL of these into
# <processed>/ automatically, including train_charges_prior_h.npy and
# train_n_h.pickle. The only extra step is the drugs flow-matching config,
# which references train_charges_prior.npy (no _h) — that is the SAME marginal
# as train_charges_prior_h.npy (identical (charge_types*atom_types).sum(0)
# formula), so we copy it.
STATS_FILES=(
    train_n_h.pickle
    train_atom_types_h.npy train_bond_types_h.npy train_charges_h.npy
    train_charges_prior_h.npy
    train_is_aromatic_h.npy train_is_in_ring_h.npy train_hybridization_h.npy
    train_valency_h.pickle train_smiles.pickle
    train_bond_lengths_h.pickle train_angles_h.pickle train_dihedrals_h.pickle
)

stage_stats() {
    local ds="$1" processed="$2"
    local dst_stats="$DST/stats/$ds"
    mkdir -p "$dst_stats"
    for f in "${STATS_FILES[@]}"; do
        if [[ -f "$processed/$f" ]]; then
            cp -f "$processed/$f" "$dst_stats/$f"
        else
            echo "  WARN: $processed/$f missing" >&2
        fi
    done
    # drugs flow-matching variant expects train_charges_prior.npy (no _h).
    if [[ "$ds" == "drugs" && -f "$dst_stats/train_charges_prior_h.npy" ]]; then
        cp -f "$dst_stats/train_charges_prior_h.npy" "$dst_stats/train_charges_prior.npy"
        echo "  + train_charges_prior.npy (copy of _h; drugs_fm config)"
    fi
    echo "  stats staged: $(ls "$dst_stats" | wc -l) files"
    du -sh "$dst_stats"
}

if [[ "$FETCH_STATS" == "1" ]]; then
    echo ""
    echo "Staging statistics bundle under $DST/stats"
    for ds in "${MODELS[@]}"; do
        ds="$(echo "$ds" | xargs)"; [[ -z "$ds" ]] && continue
        echo ""
        echo "===  stats: $ds  ==="
        # MEGALODON_STATS_SRC_<DS> points at an already-processed `processed/`
        # dir. If unset, print the preprocessing recipe and skip.
        var="MEGALODON_STATS_SRC_${ds^^}"
        processed="${!var:-}"
        if [[ -n "$processed" && -d "$processed" ]]; then
            stage_stats "$ds" "$processed"
        else
            echo "  $var unset — no processed/ dir to stage from." >&2
            echo "  Produce it with the upstream env (one-time, heavy):" >&2
            if [[ "$ds" == "drugs" ]]; then
                cat >&2 <<'EOF'
    conda activate megalodon
    mkdir -p drugs_data/raw && cd drugs_data/raw
    wget -r -np -nH --cut-dirs=2 --reject "index.html*" \
        https://bits.csb.pitt.edu/files/geom_raw/
    cd - && python opensource/megalodon/data_processing/process_geom.py \
        --raw_data_dir drugs_data/raw --save_data_folder drugs_data
    # then re-run: MEGALODON_FETCH_STATS=1 MEGALODON_STATS_SRC_DRUGS=drugs_data/processed \
    #     MEGALODON_MODELS=drugs ./scripts/fetch_weights.sh
EOF
            else
                cat >&2 <<'EOF'
    conda activate megalodon
    mkdir -p qm9_data/raw && cd qm9_data/raw
    wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
    unzip qm9.zip && rm qm9.zip
    cd - && python opensource/megalodon/data_processing/process_qm9.py \
        --qm9_sdf_path qm9_data/raw/gdb9.sdf --save_data_folder qm9_data
    # then re-run: MEGALODON_FETCH_STATS=1 MEGALODON_STATS_SRC_QM9=qm9_data/processed \
    #     MEGALODON_MODELS=qm9 ./scripts/fetch_weights.sh
EOF
            fi
        fi
    done
else
    echo ""
    echo "Statistics NOT staged (MEGALODON_FETCH_STATS!=1)."
    echo "Sampling + metrics need <DST>/stats/<dataset>/ — see this script's header."
fi

echo ""
echo "Done. NAS layout under $DST:"
[[ -d "$DST/ckpts" ]] && find "$DST/ckpts" -name '*.ckpt' | sort | sed 's/^/  /'
[[ -d "$DST/stats" ]] && du -sh "$DST"/stats/* 2>/dev/null | sed 's/^/  /'
echo ""
echo "rsync to NAS:  rsync -av $DST/ <nas>:/data/models/megalodon/"
echo "Verify:        GET /healthz/detail -> weights_loaded=true"
