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

# Pip-git deps referenced by openadmet-models-gpu.yaml.  The upstream yaml
# uses @main / @master — we snapshot each repo's *current default branch*
# HEAD (resolved via `git ls-remote --symref HEAD`), so this stays working
# even if a repo renames master→main later.  The resolved SHA is printed
# at the end so you can lock it in vendor.sh if reproducibility matters.
declare -A PIP_DEPS=(
    ["molfeat"]="https://github.com/OpenADMET/molfeat.git"
    ["TabPFN"]="https://github.com/PriorLabs/TabPFN.git"
    ["neural-pairwise-regression"]="https://github.com/JacksonBurns/neural-pairwise-regression.git"
    ["useful_rdkit_utils"]="https://github.com/PatWalters/useful_rdkit_utils.git"
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
    # Populate worktree.  If a specific ref is requested we check it out;
    # otherwise the working tree is left detached at whatever HEAD points to
    # (the remote's default branch — origin/HEAD is set by clone).
    if [[ -n "$ref" ]]; then
        (cd "$dst" && git checkout "$ref")
    else
        (cd "$dst" && git checkout FETCH_HEAD 2>/dev/null || git -C "$dst" checkout "$(git -C "$dst" symbolic-ref --short refs/remotes/origin/HEAD | sed 's|^origin/||')")
    fi
}

# Resolve <owner>/<repo>.git's default branch via ls-remote (network call,
# but cheap — one line). Returns the branch name ('main' / 'master' / ...).
_default_branch() {
    local repo="$1"
    git ls-remote --symref "$repo" HEAD 2>/dev/null \
        | awk '/^ref:/ {sub("refs/heads/", "", $2); print $2; exit}'
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
declare -A RESOLVED_SHA
for name in "${!PIP_DEPS[@]}"; do
    repo="${PIP_DEPS[$name]}"
    branch="$(_default_branch "$repo")"
    if [[ -z "$branch" ]]; then
        echo "ERROR: could not resolve default branch for $repo" >&2
        exit 1
    fi
    echo "  * $name  <- $repo @ $branch (default)"
    _clone_retry "$repo" "$TMP/$name" "$branch"
    RESOLVED_SHA[$name]="$(cd "$TMP/$name" && git rev-parse HEAD)"
    rm -rf "$TMP/$name/.git"
    rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
        "$TMP/$name/" "$PIP_DST/$name/"
done

echo ""
echo "Vendored OpenADMET @ $OPENADMET_SHA"
echo "  main:     $DST  ($(du -sh "$DST" | cut -f1))"
echo "  pip_deps: $PIP_DST  ($(du -sh "$PIP_DST" | cut -f1))"
echo ""
echo "Resolved pip_deps snapshot SHAs (record here if pin needed):"
for name in "${!RESOLVED_SHA[@]}"; do
    printf "  %-32s  %s\n" "$name" "${RESOLVED_SHA[$name]}"
done
