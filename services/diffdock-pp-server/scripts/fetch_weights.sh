#!/usr/bin/env bash
# Stage DiffDock-PP model weights for upload to NAS / HPC scratch.
#
# Three sources to reconcile:
#
#   1. Score + confidence checkpoints — shipped in the upstream repo under
#      checkpoints/; vendor.sh already brought them into
#      services/diffdock-pp-server/upstream/checkpoints/. We copy them out
#      into the externalized layout, along with the args.yaml files (each
#      fold dir needs its args.yaml — upstream args.py loads it back to
#      construct the network).
#
#   2. ESM-2 t33_650M_UR50D checkpoint (~2.5 GB) — pulled from
#      fair-esm's CDN into the torch.hub cache layout so runtime's
#      `torch.hub.load("facebookresearch/esm:main", ...)` finds it offline.
#
#   3. facebookresearch/esm source dir — torch.hub also needs the repo
#      cloned (hubconf.py). We clone it once here.
#
# Default (local stage dir for inspection):
#   ./services/diffdock-pp-server/scripts/fetch_weights.sh
#       → services/diffdock-pp-server/weights/
#
# Direct to NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/diffdock-pp \
#       ./services/diffdock-pp-server/scripts/fetch_weights.sh
#
# Requires vendor.sh to have run first (for #1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UP_SRC="$SCRIPT_DIR/../upstream"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"

if [[ ! -d "$UP_SRC/checkpoints" ]]; then
    echo "ERROR: $UP_SRC/checkpoints not found." >&2
    echo "Run ./services/diffdock-pp-server/scripts/vendor.sh first." >&2
    exit 1
fi

mkdir -p "$DST"

# --------------------------------------------------------------------------
# 1. Score + confidence checkpoints (from vendored upstream/checkpoints/).
#    Each model dir has fold_0/model_best_*.pth + args.yaml (in the parent dir).
# --------------------------------------------------------------------------
copy_model() {
    local name="$1"        # e.g. large_model_dips
    local src="$UP_SRC/checkpoints/$name"
    local dst="$DST/$name"
    if [[ ! -d "$src" ]]; then
        echo "ERROR: $src missing — upstream layout may have changed?" >&2
        exit 1
    fi
    mkdir -p "$dst/fold_0"
    # args.yaml at parent-of-fold_0 level
    if [[ -f "$src/args.yaml" ]]; then
        cp -n "$src/args.yaml" "$dst/args.yaml"
    else
        echo "WARN: $src/args.yaml not found — model construction may fail" >&2
    fi
    # model_best_*.pth (also model_last.pth if present, but only score model has it)
    for f in "$src/fold_0/"*.pth; do
        [[ -e "$f" ]] || continue
        cp -n "$f" "$dst/fold_0/$(basename "$f")"
    done
    echo "  staged $name"
}

echo "==> Staging score + confidence checkpoints"
copy_model "large_model_dips"
copy_model "confidence_model_dips"

# --------------------------------------------------------------------------
# 2. ESM-2 checkpoint (~2.5 GB) via wget -c (resumable).
# --------------------------------------------------------------------------
ESM_URL="${ESM_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt}"
ESM_DST="$DST/esm_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt"
mkdir -p "$(dirname "$ESM_DST")"
if [[ -f "$ESM_DST" ]]; then
    echo "==> ESM-2 checkpoint already present: $ESM_DST"
else
    echo "==> Downloading ESM-2 checkpoint (~2.5 GB) ..."
    wget -c -O "$ESM_DST" "$ESM_URL"
fi

# --------------------------------------------------------------------------
# 3. facebookresearch/esm source (needed by torch.hub in offline mode).
#    torch.hub caches the repo at ~/.cache/torch/hub/<owner>_<repo>_<ref>/;
#    with `TORCH_HOME=<weights_dir>/esm_cache` it looks at
#    <esm_cache>/hub/facebookresearch_esm_main/.
# --------------------------------------------------------------------------
ESM_REPO_DST="$DST/esm_cache/hub/facebookresearch_esm_main"
ESM_REPO_URL="${ESM_REPO_URL:-https://github.com/facebookresearch/esm.git}"
if [[ -f "$ESM_REPO_DST/hubconf.py" ]]; then
    echo "==> ESM source already staged: $ESM_REPO_DST"
else
    echo "==> Cloning ESM source repo ..."
    tmp_esm="$(mktemp -d -t diffdock-pp-esm.XXXXXX)"
    trap "rm -rf '$tmp_esm'" EXIT
    for i in 1 2 3 4 5; do
        rm -rf "$tmp_esm/esm"
        if git clone --depth 1 "$ESM_REPO_URL" "$tmp_esm/esm"; then
            break
        fi
        [ "$i" = "5" ] && { echo "ERROR: esm repo clone failed after 5 attempts" >&2; exit 1; }
        echo "  clone failed, retrying in $((i*10))s ..."
        sleep $((i*10))
    done
    rm -rf "$tmp_esm/esm/.git"
    mkdir -p "$(dirname "$ESM_REPO_DST")"
    rsync -a --exclude='__pycache__' --exclude='*.pyc' \
        "$tmp_esm/esm/" "$ESM_REPO_DST/"
fi

echo
echo "Done. Weights + ESM cache in: $DST"
du -sh "$DST"/*
