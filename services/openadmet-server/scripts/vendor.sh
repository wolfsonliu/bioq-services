#!/usr/bin/env bash
# Vendor the upstream OpenADMET-models source into
# services/openadmet-server/upstream/ at a pinned SHA, so `docker build`
# does no network access to github.
#
#   ./services/openadmet-server/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   OPENADMET_REPO=https://ghproxy.cn/https://github.com/OpenADMET/openadmet-models.git \
#       ./services/openadmet-server/scripts/vendor.sh
#
# To bump the upstream pin, edit OPENADMET_SHA below.
#
# In addition to the main repo, four pip-git dependencies referenced by
# devtools/conda-envs/openadmet-models-gpu.yaml are vendored to
# services/openadmet-server/pip_deps/ so `docker build` can install them
# offline (see engineering/decisions/2026-07-05-openadmet-server-design.md §6.8).

set -euo pipefail

OPENADMET_REPO="${OPENADMET_REPO:-https://github.com/OpenADMET/openadmet-models.git}"
OPENADMET_SHA="${OPENADMET_SHA:-b6571905ac1d76a557f6c4795a278ea8643c336e}"

# Pip-git deps referenced by openadmet-models-gpu.yaml.  These pins are
# looser than the main repo (the yaml uses @main / @master — we snapshot
# them here at whatever HEAD they're at when this script runs, and update
# manually if needed).
declare -A PIP_DEPS=(
    ["molfeat"]="https://github.com/OpenADMET/molfeat.git"
    ["TabPFN"]="https://github.com/PriorLabs/TabPFN.git"
    ["neural-pairwise-regression"]="https://github.com/JacksonBurns/neural-pairwise-regression.git"
    ["useful_rdkit_utils"]="https://github.com/PatWalters/useful_rdkit_utils.git"
)
declare -A PIP_BRANCHES=(
    ["molfeat"]="main"
    ["TabPFN"]="main"
    ["neural-pairwise-regression"]="master"
    ["useful_rdkit_utils"]="master"
)

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/openadmet-server/upstream"
PIP_DST="$PROJECT_ROOT/services/openadmet-server/pip_deps"
TMP="$(mktemp -d -t openadmet-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST" "$PIP_DST"

_clone_retry() {
    local repo="$1" dst="$2" ref="${3:-}"
    for i in 1 2 3 4 5; do
        rm -rf "$dst"
        if git clone --filter=blob:none --no-checkout "$repo" "$dst"; then
            break
        fi
        [ "$i" = "5" ] && { echo "ERROR: git clone $repo failed after 5 attempts" >&2; exit 1; }
        echo "  clone failed, retrying in $((i*10))s ..."
        sleep $((i*10))
    done
    if [[ -n "$ref" ]]; then
        (cd "$dst" && git checkout "$ref")
    else
        (cd "$dst" && git checkout HEAD)
    fi
}

# ---- Main upstream repo ----
echo "[1/2] Vendoring $OPENADMET_REPO @ $OPENADMET_SHA"
_clone_retry "$OPENADMET_REPO" "$TMP/openadmet-models" "$OPENADMET_SHA"

actual="$(cd "$TMP/openadmet-models" && git rev-parse HEAD)"
if [[ "$actual" != "$OPENADMET_SHA" ]]; then
    echo "ERROR: HEAD mismatch (got $actual, expected $OPENADMET_SHA)" >&2
    exit 1
fi
rm -rf "$TMP/openadmet-models/.git"

# Sync into DST. --delete drops stale files from previous vendor runs.
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/openadmet-models/" "$DST/"

echo "  -> $DST"

# ---- Pip-git deps ----
echo "[2/2] Vendoring pip-git deps into $PIP_DST"
for name in "${!PIP_DEPS[@]}"; do
    repo="${PIP_DEPS[$name]}"
    branch="${PIP_BRANCHES[$name]}"
    echo "  * $name  <- $repo @ $branch"
    _clone_retry "$repo" "$TMP/$name" "$branch"
    (cd "$TMP/$name" && git checkout "$branch")
    rm -rf "$TMP/$name/.git"
    rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
        "$TMP/$name/" "$PIP_DST/$name/"
done

echo ""
echo "Vendored OpenADMET @ $OPENADMET_SHA"
echo "  main: $DST  ($(du -sh "$DST" | cut -f1))"
echo "  pip_deps: $PIP_DST  ($(du -sh "$PIP_DST" | cut -f1))"
