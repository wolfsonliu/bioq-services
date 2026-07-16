#!/usr/bin/env bash
# Stage EVERYTHING Megalodon needs (checkpoints + statistics) into the NAS
# layout the service expects — with ONE command:
#
#     ./scripts/fetch_weights.sh
#
# By default this fetches BOTH datasets (qm9 + drugs) AND BOTH artifact kinds
# (checkpoints + statistics). The statistics bundle is NOT distributed as files,
# so this script produces it the only supported way: download the raw dataset
# and run the upstream preprocessing (data_processing/process_{qm9,geom}.py),
# then copy the resulting statistics into place. Everything lands under <DST>.
#
# Target NAS layout (matches settings.MegalodonSettings + design doc §6):
#   <DST>/ckpts/<dataset>/*.ckpt          # <dataset> = qm9 | drugs
#   <DST>/stats/<dataset>/<statistics files>
#   <DST>/_staging/<dataset>/{raw,proc}   # intermediates (raw + processed/)
#
# The service reads this at MEGALODON_WEIGHTS_DIR — see
# services/megalodon-server/settings.py. Verify with:
#   GET /healthz/detail -> models.<name>.ready == true
#
# ---------------------------------------------------------------------------
# QUICK START
# ---------------------------------------------------------------------------
#   # Everything (ckpts + stats, qm9 + drugs) straight to NAS, using the
#   # megalodon conda env's python for the preprocessing step:
#   MEGALODON_PY=/opt/conda/envs/megalodon/bin/python \
#   WEIGHTS_DST=/mnt/nas/data/models/megalodon \
#       ./scripts/fetch_weights.sh
#
#   # Only qm9 (drugs still downloading elsewhere):
#   MEGALODON_MODELS=qm9 ./scripts/fetch_weights.sh
#
#   # Checkpoints only (skip the heavy stats preprocessing):
#   MEGALODON_FETCH_STATS=0 ./scripts/fetch_weights.sh
#
#   # Stats only (checkpoints already staged):
#   MEGALODON_FETCH_CKPTS=0 ./scripts/fetch_weights.sh
#
#   # Delete the multi-GB raw/processed intermediates after staging:
#   MEGALODON_CLEAN_STAGING=1 ./scripts/fetch_weights.sh
#
# ---------------------------------------------------------------------------
# WHAT EACH STEP NEEDS
# ---------------------------------------------------------------------------
# CHECKPOINTS (MEGALODON_FETCH_CKPTS=1, default) — HuggingFace, checkpoints only:
#   nvidia/NV-Megalodon-QM9-v1        -> ckpts/qm9/
#     megalodon_diffusion.ckpt, megalodon_fm.ckpt, megalodon_small_diffusion.ckpt
#   nvidia/NV-Megalodon-GEOM-Drugs-v1 -> ckpts/drugs/
#     megalodon_large_diffusion.ckpt, megalodon_fm.ckpt, megalodon_small_diffusion.ckpt
#   The model_name -> (dataset, ckpt file) mapping (see models.MODEL_REGISTRY):
#     qm9_diffusion -> qm9/megalodon_diffusion.ckpt
#     drugs_diffusion -> drugs/megalodon_large_diffusion.ckpt
#     {qm9,drugs}_fm -> <dataset>/megalodon_fm.ckpt
#     {qm9,drugs}_quick -> <dataset>/megalodon_small_diffusion.ckpt
#   Needs `hf` (pip install -U huggingface_hub) OR a pre-downloaded snapshot dir
#   via MEGALODON_QM9_SRC / MEGALODON_DRUGS_SRC.
#
# STATISTICS (MEGALODON_FETCH_STATS=1, default) — raw download + preprocessing:
#   qm9:   qm9.zip (deepchem S3) -> gdb9.sdf -> process_qm9.py  -> processed/
#   drugs: geom_raw/ (bits.csb.pitt.edu, tens of GB) -> process_geom.py -> processed/
#   Preprocessing runs with MEGALODON_PY (a python that can import torch, rdkit,
#   torch_geometric, pandas — i.e. the `megalodon` conda env). Needs `wget` and,
#   for qm9, `unzip`. Model init itself loads train_atom_types_h.npy as a prior,
#   so stats are REQUIRED, not just for metrics.
#   Shortcut: if you already have a processed/ dir, skip the download+compute:
#     MEGALODON_STATS_SRC_QM9=/path/to/qm9_data/processed \
#     MEGALODON_STATS_SRC_DRUGS=/path/to/drugs_data/processed \
#         ./scripts/fetch_weights.sh
#
# See design doc §6 / §12 risk 1 for the full rationale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
REQ="${MEGALODON_MODELS:-qm9,drugs}"
FETCH_CKPTS="${MEGALODON_FETCH_CKPTS:-1}"
FETCH_STATS="${MEGALODON_FETCH_STATS:-1}"
CLEAN_STAGING="${MEGALODON_CLEAN_STAGING:-0}"

# Python that can run the upstream preprocessing (torch + rdkit + torch_geometric
# + pandas). Prefers MEGALODON_PY, then the image's MEGALODON_PYTHON (set in the
# Dockerfile to the megalodon conda env), then `python`.
PREPROCESS_PY="${MEGALODON_PY:-${MEGALODON_PYTHON:-python}}"
# Upstream preprocessing scripts (process_qm9.py / process_geom.py) live in the
# vendored upstream tree. Resolve to the repo checkout when run from the repo, or
# to /opt/megalodon/data_processing when run inside the image — both are the same
# vendored upstream. Override with MEGALODON_DATA_PROC_DIR.
if [[ -n "${MEGALODON_DATA_PROC_DIR:-}" ]]; then
    DATA_PROC_DIR="$MEGALODON_DATA_PROC_DIR"
else
    DATA_PROC_DIR="$SCRIPT_DIR/../upstream/data_processing"
    for _cand in "$SCRIPT_DIR/../upstream/data_processing" \
                 "${MEGALODON_ROOT:-/opt/megalodon}/data_processing"; do
        if [[ -d "$_cand" ]]; then DATA_PROC_DIR="$_cand"; break; fi
    done
fi
# Where raw + processed intermediates live (under DST so it's all in one place).
STAGING="${MEGALODON_STAGING_DIR:-$DST/_staging}"

# Raw dataset sources (documented upstream recipe).
QM9_ZIP_URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
GEOM_RAW_URL="https://bits.csb.pitt.edu/files/geom_raw/"

# HF repo per dataset kind.
declare -A HF_REPO=(
    [qm9]="nvidia/NV-Megalodon-QM9-v1"
    [drugs]="nvidia/NV-Megalodon-GEOM-Drugs-v1"
)
# Pre-downloaded HF snapshot dir per dataset (skip ckpt download if set).
declare -A SRC_DIR=(
    [qm9]="${MEGALODON_QM9_SRC:-}"
    [drugs]="${MEGALODON_DRUGS_SRC:-}"
)
# Pre-processed `processed/` dir per dataset (skip raw download + preprocess).
declare -A STATS_SRC=(
    [qm9]="${MEGALODON_STATS_SRC_QM9:-}"
    [drugs]="${MEGALODON_STATS_SRC_DRUGS:-}"
)

# Canonical checkpoint file names per dataset (diffusion name differs).
declare -A EXPECTED_CKPTS=(
    [qm9]="megalodon_diffusion.ckpt megalodon_fm.ckpt megalodon_small_diffusion.ckpt"
    [drugs]="megalodon_large_diffusion.ckpt megalodon_fm.ckpt megalodon_small_diffusion.ckpt"
)

# save_statistics (data_processing/utils_data.py) writes ALL of these into
# processed/. The service reads them from <DST>/stats/<dataset>/.
STATS_FILES=(
    train_n_h.pickle
    train_atom_types_h.npy train_bond_types_h.npy train_charges_h.npy
    train_charges_prior_h.npy
    train_is_aromatic_h.npy train_is_in_ring_h.npy train_hybridization_h.npy
    train_valency_h.pickle train_smiles.pickle
    train_bond_lengths_h.pickle train_angles_h.pickle train_dihedrals_h.pickle
)

IFS=',' read -r -a MODELS <<< "$REQ"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Checkpoints -> <DST>/ckpts/<dataset>/
# ---------------------------------------------------------------------------
stage_ckpts() {
    local ds="$1"
    echo ""
    echo "=== ckpts: $ds  (${HF_REPO[$ds]}) ==="

    local src="${SRC_DIR[$ds]}"
    if [[ -z "$src" ]]; then
        if ! command -v hf >/dev/null 2>&1; then
            die "hf not found and MEGALODON_${ds^^}_SRC unset. Install (pip install
       -U huggingface_hub) or point at a pre-downloaded snapshot dir."
        fi
        src="$DST/_hf_$ds"
        echo "Downloading ${HF_REPO[$ds]} -> $src"
        hf download "${HF_REPO[$ds]}" --local-dir "$src"
    else
        echo "Using pre-downloaded snapshot: $src"
    fi
    [[ -d "$src" ]] || die "source dir not found: $src"

    local dst_ckpt="$DST/ckpts/$ds"
    mkdir -p "$dst_ckpt"

    shopt -s nullglob
    local found=0 f
    for f in "$src"/*.ckpt "$src"/**/*.ckpt; do
        [[ -f "$f" ]] || continue
        cp -f "$f" "$dst_ckpt/$(basename "$f")"
        echo "  + $(basename "$f")"
        found=$((found + 1))
    done
    shopt -u nullglob
    [[ "$found" -gt 0 ]] || die "no *.ckpt found under $src"

    local want
    for want in ${EXPECTED_CKPTS[$ds]}; do
        [[ -f "$dst_ckpt/$want" ]] || echo "  WARN: expected $want not present in $ds" >&2
    done
    du -sh "$dst_ckpt"
}

# ---------------------------------------------------------------------------
# Raw dataset download (idempotent) -> <raw_dir>
# ---------------------------------------------------------------------------
download_qm9_raw() {
    local raw_dir="$1"
    if [[ -f "$raw_dir/gdb9.sdf" ]]; then
        echo "  raw present: $raw_dir/gdb9.sdf"; return
    fi
    command -v wget >/dev/null 2>&1 || die "wget not found (needed for qm9 raw download)"
    command -v unzip >/dev/null 2>&1 || die "unzip not found (needed for qm9.zip)"
    mkdir -p "$raw_dir"
    echo "  downloading qm9.zip -> $raw_dir"
    ( cd "$raw_dir" && wget -q --show-progress -O qm9.zip "$QM9_ZIP_URL" \
        && unzip -o qm9.zip && rm -f qm9.zip )
    [[ -f "$raw_dir/gdb9.sdf" ]] || die "gdb9.sdf missing after unzip in $raw_dir"
}

download_geom_raw() {
    local raw_dir="$1"
    if [[ -f "$raw_dir/train_data.pickle" ]]; then
        echo "  raw present: $raw_dir/train_data.pickle"; return
    fi
    command -v wget >/dev/null 2>&1 || die "wget not found (needed for geom raw download)"
    mkdir -p "$raw_dir"
    echo "  downloading GEOM raw (tens of GB) -> $raw_dir"
    ( cd "$raw_dir" && wget -r -np -nH --cut-dirs=2 --reject "index.html*" "$GEOM_RAW_URL" )
    local want
    for want in train_data.pickle val_data.pickle test_data.pickle; do
        [[ -f "$raw_dir/$want" ]] || die "GEOM raw missing $want in $raw_dir"
    done
}

# ---------------------------------------------------------------------------
# Preprocessing (idempotent) -> <save_folder>/processed/
# ---------------------------------------------------------------------------
_preprocess_done() { [[ -f "$1/processed/train_atom_types_h.npy" ]]; }

run_preprocess_qm9() {
    local raw_dir="$1" save_folder="$2"
    if _preprocess_done "$save_folder"; then
        echo "  processed present: $save_folder/processed"; return
    fi
    echo "  preprocessing qm9 (process_qm9.py) ..."
    ( cd "$DATA_PROC_DIR" && "$PREPROCESS_PY" process_qm9.py \
        --qm9_sdf_path "$raw_dir/gdb9.sdf" --save_data_folder "$save_folder" )
}

run_preprocess_geom() {
    local raw_dir="$1" save_folder="$2"
    if _preprocess_done "$save_folder"; then
        echo "  processed present: $save_folder/processed"; return
    fi
    echo "  preprocessing drugs (process_geom.py) ..."
    ( cd "$DATA_PROC_DIR" && "$PREPROCESS_PY" process_geom.py \
        --raw_data_dir "$raw_dir" --save_data_folder "$save_folder" )
}

# ---------------------------------------------------------------------------
# Statistics staging -> <DST>/stats/<dataset>/
# ---------------------------------------------------------------------------
_stats_staged() {
    local ds="$1" dst_stats="$DST/stats/$ds" f
    for f in "${STATS_FILES[@]}"; do
        [[ -f "$dst_stats/$f" ]] || return 1
    done
    return 0
}

stage_stats() {
    local ds="$1" processed="$2"
    local dst_stats="$DST/stats/$ds" f
    mkdir -p "$dst_stats"
    for f in "${STATS_FILES[@]}"; do
        if [[ -f "$processed/$f" ]]; then
            cp -f "$processed/$f" "$dst_stats/$f"
        else
            echo "  WARN: $processed/$f missing" >&2
        fi
    done
    # drugs flow-matching variant expects train_charges_prior.npy (no _h) — the
    # SAME marginal as train_charges_prior_h.npy.
    if [[ "$ds" == "drugs" && -f "$dst_stats/train_charges_prior_h.npy" ]]; then
        cp -f "$dst_stats/train_charges_prior_h.npy" "$dst_stats/train_charges_prior.npy"
        echo "  + train_charges_prior.npy (copy of _h; drugs_fm config)"
    fi
    echo "  stats staged: $(ls "$dst_stats" | wc -l) files"
    du -sh "$dst_stats"
}

# Orchestrate the stats path for one dataset: pre-processed override -> stage;
# already staged -> skip; otherwise download raw + preprocess + stage.
ensure_stats() {
    local ds="$1"
    echo ""
    echo "=== stats: $ds ==="

    local processed="${STATS_SRC[$ds]}"
    if [[ -n "$processed" ]]; then
        [[ -d "$processed" ]] || die "MEGALODON_STATS_SRC_${ds^^}=$processed not a dir"
        echo "  using pre-processed dir: $processed"
        stage_stats "$ds" "$processed"
        return
    fi

    if _stats_staged "$ds"; then
        echo "  already staged under $DST/stats/$ds — skipping"
        return
    fi

    [[ -d "$DATA_PROC_DIR" ]] || die "data_processing dir not found: $DATA_PROC_DIR
       (set MEGALODON_DATA_PROC_DIR to the upstream data_processing path)."

    local base="$STAGING/$ds"
    local raw_dir="$base/raw" proc_dir="$base/proc"
    mkdir -p "$raw_dir" "$proc_dir"

    case "$ds" in
        qm9)
            download_qm9_raw "$raw_dir"
            run_preprocess_qm9 "$raw_dir" "$proc_dir"
            ;;
        drugs)
            download_geom_raw "$raw_dir"
            run_preprocess_geom "$raw_dir" "$proc_dir"
            ;;
        *) die "unknown dataset '$ds'";;
    esac

    stage_stats "$ds" "$proc_dir/processed"

    if [[ "$CLEAN_STAGING" == "1" ]]; then
        echo "  cleaning staging: $base"
        rm -rf "$base"
    fi
}

# Preflight: fail fast BEFORE a multi-GB download if the preprocessing env is
# wrong (only when we actually need to preprocess).
preflight_preprocess_env() {
    local need_preprocess=0 ds
    for ds in "${MODELS[@]}"; do
        ds="$(echo "$ds" | xargs)"; [[ -z "$ds" ]] && continue
        [[ -n "${STATS_SRC[$ds]:-}" ]] && continue
        _stats_staged "$ds" && continue
        need_preprocess=1
    done
    [[ "$need_preprocess" == "0" ]] && return
    if ! "$PREPROCESS_PY" -c "import torch, rdkit, torch_geometric, pandas" 2>/dev/null; then
        die "MEGALODON_PY=$PREPROCESS_PY cannot import torch/rdkit/torch_geometric/pandas.
       Point MEGALODON_PY at the megalodon conda env, e.g.
       MEGALODON_PY=/opt/conda/envs/megalodon/bin/python, or provide a
       pre-processed dir via MEGALODON_STATS_SRC_<DS> / skip with
       MEGALODON_FETCH_STATS=0."
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for ds in "${MODELS[@]}"; do
    ds="$(echo "$ds" | xargs)"; [[ -z "$ds" ]] && continue
    [[ -n "${HF_REPO[$ds]:-}" ]] || die "unknown dataset '$ds' (want qm9 | drugs)"
done

echo "Staging Megalodon artifacts under $DST"
echo "  datasets: $REQ | ckpts: $FETCH_CKPTS | stats: $FETCH_STATS"

[[ "$FETCH_STATS" == "1" ]] && preflight_preprocess_env

for ds in "${MODELS[@]}"; do
    ds="$(echo "$ds" | xargs)"; [[ -z "$ds" ]] && continue
    [[ "$FETCH_CKPTS" == "1" ]] && stage_ckpts "$ds"
    [[ "$FETCH_STATS" == "1" ]] && ensure_stats "$ds"
done

echo ""
echo "Done. NAS layout under $DST:"
[[ -d "$DST/ckpts" ]] && find "$DST/ckpts" -name '*.ckpt' | sort | sed 's/^/  /'
[[ -d "$DST/stats" ]] && du -sh "$DST"/stats/* 2>/dev/null | sed 's/^/  /'
echo ""
echo "rsync to NAS:  rsync -av $DST/ <nas>:/data/models/megalodon/"
echo "Verify:        GET /healthz/detail -> models.<name>.ready == true"
