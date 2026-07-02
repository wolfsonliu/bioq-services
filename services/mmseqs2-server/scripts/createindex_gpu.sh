#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-mmseqs-createindex-%j.out
#
# Build GPU-searchable indexes (.idx) for the ColabFold MMseqs2 databases
# so mmseqs2-server can run in GPU mode (MMSEQS2_GPU_ENABLED=true, default).
#
# Mirrors the createindex + taxonomy-binary + idx-symlink steps that
# upstream `opensource/ColabFold/setup_databases.sh` performs when `GPU=1`
# is set (its `GPU_INDEX_PAR` block + the `UNIREF30_READY` / `COLABDB_READY`
# stanzas).  Two source modes:
#
#   MODE=symlink  (default) — reuse the HPC public-dataset mount, symlink
#                             `${SOURCE_DIR}/<db>*` into a writable stage
#                             dir.  Zero download cost.
#
#   MODE=download           — fetch `${db}.db.tar.gz` from opendata.mmseqs.org
#                             (mirrors setup_databases.sh's
#                             FAST_PREBUILT_DATABASES=1 branch) and extract
#                             into the stage dir.  Also fetches the 2025-08-04
#                             `uniref30_2302_newtaxonomy.tar.gz` overlay so
#                             pairing uses the corrected taxonomy.
#
# Sources (MODE=symlink):
#   /data/public/datasets/colabfold/{uniref30_2302_db,colabfold_envdb_202108_db}*
#
# Sources (MODE=download):
#   https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz
#   https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz
#   https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz
#
# Staging (writable — symlinks/extracts + new .idx / .idx_* land here):
#   ${OUTPUT_DIR:-$HOME/mmseqs2_gpu_index}/<db>.stage/
#
# Submit:
#   sbatch services/mmseqs2-server/scripts/createindex_gpu.sh
#   MODE=download sbatch services/mmseqs2-server/scripts/createindex_gpu.sh
#
# After the job completes, rsync each <db>.stage/ directory to the Aliyun
# NAS mount used by FC.  `rsync -aL` follows symlinks so both the source
# DB files and the freshly built .idx sidecars land at the destination:
#   /data/models/mmseqs2/uniref30_2302_db*
#   /data/models/mmseqs2/colabfold_envdb_202108_db*
# (matches MMSEQS2_DB_DIR default in services/mmseqs2-server/settings.py).
#
# Env overrides:
#   MODE=symlink|download               default: symlink
#   SOURCE_DIR=/other/path              default: /data/public/datasets/colabfold  (symlink mode)
#   DOWNLOAD_DIR=/scratch/downloads     default: $OUTPUT_DIR/downloads            (download mode)
#   OUTPUT_DIR=/scratch/mmseqs2_index   default: $HOME/mmseqs2_gpu_index
#   DBS="db1 db2"                       default: uniref30_2302_db colabfold_envdb_202108_db
#   MMSEQS_MODULE=mmseqs/git-e42b142    default: mmseqs/git-e42b142
#   THREADS=8                           default: 8 (should match --cpus-per-task)
#   APPLY_NEWTAXONOMY=1|0               default: 1 in download mode, 0 in symlink mode
#                                       Enable in symlink mode too if you want the
#                                       2025-08-04 corrected uniref30_2302 taxonomy.
#
# GPU indexing parameters (per setup_databases.sh lines 117-131):
#   --split 1              — single contiguous shard (required for GPU search)
#   --index-subset N       — the GPU-friendly subset ID.  N=2 for older mmseqs,
#                            N=10 if the binary supports "no sequence lookup"
#                            (detected at runtime).
#   --remove-tmp-files 1   — clean intermediate files inside the tmp dir

set -euo pipefail

module load "${MMSEQS_MODULE:-mmseqs/git-e42b142}"

MODE="${MODE:-symlink}"
SOURCE_DIR="${SOURCE_DIR:-/data/public/datasets/colabfold}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/mmseqs2_gpu_index}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$OUTPUT_DIR/downloads}"
DBS="${DBS:-uniref30_2302_db colabfold_envdb_202108_db}"
THREADS="${THREADS:-8}"

# Newtaxonomy overlay defaults: on in download mode (parity with upstream
# setup_databases.sh), off in symlink mode (assumes public dataset is up-to-date).
if [ -z "${APPLY_NEWTAXONOMY:-}" ]; then
    if [ "$MODE" = "download" ]; then
        APPLY_NEWTAXONOMY=1
    else
        APPLY_NEWTAXONOMY=0
    fi
fi

# Upstream mmseqs opendata bucket (mirrors setup_databases.sh line 80-88).
OPENDATA_BASE="${OPENDATA_BASE:-https://opendata.mmseqs.org/colabfold}"

# Match upstream: force merged DB output to keep the directory tidy.
export MMSEQS_FORCE_MERGE=1

echo "=== mmseqs createindex (GPU) on $(hostname) ==="
echo "GPU        : $(nvidia-smi -L 2>/dev/null | head -1 || echo '(none reported)')"
echo "mmseqs     : $(which mmseqs) — $(mmseqs version 2>/dev/null || echo '?')"
echo "MODE       : $MODE"
echo "OUTPUT_DIR : $OUTPUT_DIR"
if [ "$MODE" = "symlink" ]; then
    echo "SOURCE_DIR : $SOURCE_DIR"
else
    echo "DOWNLOAD_DIR: $DOWNLOAD_DIR"
fi
echo "DBS        : $DBS"
echo "THREADS    : $THREADS"
echo "APPLY_NEWTAXONOMY: $APPLY_NEWTAXONOMY"
echo ""

# ---------------------------------------------------------------------------
# GPU / version checks
# ---------------------------------------------------------------------------

# Sanity: the module must support GPU (mmseqs release 16+).
if ! mmseqs --help 2>&1 | grep -q 'gpuserver'; then
    echo "ERROR: this mmseqs build has no GPU support (release 16+ required)." >&2
    echo "       module=${MMSEQS_MODULE:-mmseqs/git-e42b142}" >&2
    exit 1
fi

# Detect --index-subset value: newer mmseqs versions support subset 10
# ("no sequence lookup") which is smaller + faster; older ones need 2.
if mmseqs indexdb --help 2>&1 | grep -q "8: no sequence lookup"; then
    INDEX_SUBSET=10
else
    INDEX_SUBSET=2
fi
echo "using --index-subset $INDEX_SUBSET"
echo ""

GPU_INDEX_PAR="--split 1 --index-subset $INDEX_SUBSET"

mkdir -p "$OUTPUT_DIR"


# ---------------------------------------------------------------------------
# Download helpers (used only in MODE=download)
# ---------------------------------------------------------------------------

has_cmd() { command -v "$1" >/dev/null 2>&1; }

# Pick the first available download tool.  aria2c > curl > wget, matching
# upstream setup_databases.sh strategy resolution.
DOWNLOADER=""
if [ "$MODE" = "download" ]; then
    if has_cmd aria2c;   then DOWNLOADER=aria2c
    elif has_cmd curl;   then DOWNLOADER=curl
    elif has_cmd wget;   then DOWNLOADER=wget
    else
        echo "ERROR: MODE=download requires one of aria2c / curl / wget in PATH" >&2
        exit 1
    fi
    echo "downloader : $DOWNLOADER"
    echo ""
    mkdir -p "$DOWNLOAD_DIR"
fi


download_file() {
    local url="$1"
    local out="$2"
    if [ -s "$out" ]; then
        echo "  (already downloaded: $out)"
        return 0
    fi
    case "$DOWNLOADER" in
        aria2c)
            aria2c --max-connection-per-server=8 --allow-overwrite=true \
                -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
            ;;
        curl)  curl -L --fail --retry 5 -o "$out" "$url" ;;
        wget)  wget -O "$out" "$url" ;;
    esac
}


# Fetch + extract the DB tarball into $STAGE_DIR.  Accepts either
# opendata layout (per upstream setup_databases.sh):
#
#   ${db_base}.db.tar.gz  — FAST_PREBUILT_DATABASES=1 (new): unpacks to
#                           uniref30_2302_db* (already a profile DB, no
#                           post-processing).
#   ${db_base}.tar.gz     — FAST_PREBUILT_DATABASES=0 (old TSV): unpacks
#                           to uniref30_2302* (base name, needs
#                           `mmseqs tsv2exprofiledb` to convert to _db).
#
# If either tarball is already present at $DOWNLOAD_DIR (e.g. pre-staged
# by the user), the download is skipped and the existing file is reused.
# The `.db.tar.gz` variant is preferred when both are present.
fetch_and_extract_db() {
    local DB="$1"            # e.g. uniref30_2302_db
    local STAGE_DIR="$2"
    local DB_BASE="${DB%_db}"       # e.g. uniref30_2302

    local TARBALL_NEW="$DOWNLOAD_DIR/${DB_BASE}.db.tar.gz"
    local TARBALL_OLD="$DOWNLOAD_DIR/${DB_BASE}.tar.gz"
    local TARBALL="" IS_OLD_FORMAT=0

    # 1) Reuse pre-staged tarball if present (either variant).
    if   [ -s "$TARBALL_NEW" ]; then
        TARBALL="$TARBALL_NEW"
        echo "  reusing pre-staged $TARBALL_NEW"
    elif [ -s "$TARBALL_OLD" ]; then
        TARBALL="$TARBALL_OLD"
        IS_OLD_FORMAT=1
        echo "  reusing pre-staged $TARBALL_OLD (old TSV format — will run tsv2exprofiledb)"
    else
        # 2) Otherwise fetch the new FAST_PREBUILT tarball from opendata.
        echo "  fetching ${DB_BASE}.db.tar.gz from $OPENDATA_BASE ..."
        download_file "$OPENDATA_BASE/${DB_BASE}.db.tar.gz" "$TARBALL_NEW"
        TARBALL="$TARBALL_NEW"
    fi

    echo "  extracting $(basename "$TARBALL") into $STAGE_DIR ..."
    tar -xzf "$TARBALL" -C "$STAGE_DIR"

    # 3) Old-format tarball → run tsv2exprofiledb to produce ${DB} files
    #    (matches upstream setup_databases.sh line 147).  --gpu 1 makes the
    #    resulting DB GPU-searchable (upstream's ${GPU_PAR} expands to
    #    `--gpu 1` when GPU=1).
    if [ "$IS_OLD_FORMAT" = "1" ]; then
        echo "  running mmseqs tsv2exprofiledb (${DB_BASE} → ${DB}) ..."
        ( cd "$STAGE_DIR" && mmseqs tsv2exprofiledb "$DB_BASE" "$DB" --gpu 1 )
    elif [ ! -e "$STAGE_DIR/${DB}.dbtype" ]; then
        # Tarball claimed to be new-format but didn't produce ${DB}.dbtype.
        # Try tsv2exprofiledb as a fallback if the base-name files exist.
        if [ -e "$STAGE_DIR/${DB_BASE}.dbtype" ] || [ -e "$STAGE_DIR/${DB_BASE}" ]; then
            echo "  tarball produced base-name files only — running tsv2exprofiledb ..."
            ( cd "$STAGE_DIR" && mmseqs tsv2exprofiledb "$DB_BASE" "$DB" --gpu 1 )
        fi
    fi

    # 4) Optional taxonomy overlay for uniref30_2302 (upstream lines 156-172).
    if [ "$APPLY_NEWTAXONOMY" = "1" ] && [ "$DB_BASE" = "uniref30_2302" ]; then
        local TAX_TARBALL="$DOWNLOAD_DIR/uniref30_2302_newtaxonomy.tar.gz"
        if [ ! -s "$TAX_TARBALL" ]; then
            echo "  fetching newtaxonomy overlay ..."
            download_file "$OPENDATA_BASE/uniref30_2302_newtaxonomy.tar.gz" "$TAX_TARBALL"
        else
            echo "  reusing pre-staged $TAX_TARBALL"
        fi
        echo "  applying newtaxonomy overlay into $STAGE_DIR ..."
        tar -xzf "$TAX_TARBALL" -C "$STAGE_DIR"
    fi
}


# Symlink the source-mode files.  Optionally overlay newtaxonomy (which
# requires ability to download — mmseqs public dataset does not ship
# newtaxonomy).
symlink_source_db() {
    local DB="$1"
    local STAGE_DIR="$2"
    local n=0 src
    for src in "$SOURCE_DIR/${DB}"*; do
        [ -e "$src" ] || continue
        ln -sfn "$src" "$STAGE_DIR/$(basename "$src")"
        n=$((n+1))
    done
    if [ "$n" -eq 0 ]; then
        echo "  ERROR: no ${DB}* files under $SOURCE_DIR" >&2
        return 1
    fi
    echo "  staged $n file(s) via symlink into $STAGE_DIR/"

    if [ "$APPLY_NEWTAXONOMY" = "1" ] && [ "$DB" = "uniref30_2302_db" ]; then
        if [ -z "$DOWNLOADER" ]; then
            echo "  WARNING: APPLY_NEWTAXONOMY=1 but no downloader available — skipping" >&2
            return 0
        fi
        local TAX_TARBALL="$DOWNLOAD_DIR/uniref30_2302_newtaxonomy.tar.gz"
        mkdir -p "$DOWNLOAD_DIR"
        echo "  fetching newtaxonomy overlay (small, ~few MB) ..."
        download_file "$OPENDATA_BASE/uniref30_2302_newtaxonomy.tar.gz" "$TAX_TARBALL"
        # The overlay's _mapping / _taxonomy files must physically OVERWRITE
        # the symlinks to the read-only source ones — extract to a temp dir
        # first, then rm the stale symlinks and mv files into place.
        local TAX_TMP="$STAGE_DIR/_newtax_tmp"
        rm -rf "$TAX_TMP"
        mkdir -p "$TAX_TMP"
        tar -xzf "$TAX_TARBALL" -C "$TAX_TMP"
        local f
        for f in "$TAX_TMP"/*; do
            [ -e "$f" ] || continue
            local base
            base="$(basename "$f")"
            rm -f "$STAGE_DIR/$base"
            mv "$f" "$STAGE_DIR/$base"
        done
        rm -rf "$TAX_TMP"
        echo "  newtaxonomy overlay applied over symlinks"
    fi
}


# ---------------------------------------------------------------------------
# Per-DB indexing
# ---------------------------------------------------------------------------


index_one_db() {
    local DB="$1"
    local STAGE_DIR="$OUTPUT_DIR/$DB.stage"
    local TMP_DIR="$OUTPUT_DIR/_tmp/$DB"

    echo "==================================================================="
    echo "  Indexing: $DB"
    echo "==================================================================="

    mkdir -p "$STAGE_DIR" "$TMP_DIR"

    # ---- Populate STAGE_DIR according to MODE ----
    if [ "$MODE" = "download" ]; then
        fetch_and_extract_db "$DB" "$STAGE_DIR"
    else
        if [ ! -d "$SOURCE_DIR" ]; then
            echo "  ERROR: SOURCE_DIR $SOURCE_DIR not readable (MODE=symlink)" >&2
            return 1
        fi
        symlink_source_db "$DB" "$STAGE_DIR" || return 1
    fi

    # Sanity: DB should have the mandatory core files.
    if [ ! -e "$STAGE_DIR/${DB}.dbtype" ]; then
        echo "  ERROR: $STAGE_DIR/${DB}.dbtype missing — is $DB a valid mmseqs DB?" >&2
        return 1
    fi

    # ---- Convert *_db_mapping to binary (mmap-able, saves load time) ----
    # Upstream setup_databases.sh lines 160-168: if the mapping isn't already
    # binary (magic header != 0c170013), convert it.  Safe no-op otherwise.
    if [ -e "$STAGE_DIR/${DB}_mapping" ]; then
        local TAXHEADER
        TAXHEADER=$(od -An -N4 -t x4 "$STAGE_DIR/${DB}_mapping" | tr -d ' ')
        if [ "$TAXHEADER" != "0c170013" ]; then
            echo "  converting ${DB}_mapping to binary…"
            mmseqs createbintaxmapping \
                "$STAGE_DIR/${DB}_mapping" \
                "$STAGE_DIR/${DB}_mapping.bin"
            mv -f -- "$STAGE_DIR/${DB}_mapping.bin" "$STAGE_DIR/${DB}_mapping"
        else
            echo "  ${DB}_mapping already binary — skipping conversion"
        fi
    fi

    # ---- Build the GPU index ----
    echo "  running mmseqs createindex …"
    time mmseqs createindex \
        "$STAGE_DIR/$DB" \
        "$TMP_DIR" \
        --remove-tmp-files 1 \
        --threads "$THREADS" \
        $GPU_INDEX_PAR

    # ---- Post-createindex symlinks for pairing ----
    # Upstream setup_databases.sh lines 168-172: mmseqs pairing expects
    # `<db>.idx_mapping` / `<db>.idx_taxonomy` sidecars.
    if [ -e "$STAGE_DIR/${DB}_mapping" ]; then
        ln -sf "${DB}_mapping" "$STAGE_DIR/${DB}.idx_mapping"
        echo "  linked ${DB}.idx_mapping -> ${DB}_mapping"
    fi
    if [ -e "$STAGE_DIR/${DB}_taxonomy" ]; then
        ln -sf "${DB}_taxonomy" "$STAGE_DIR/${DB}.idx_taxonomy"
        echo "  linked ${DB}.idx_taxonomy -> ${DB}_taxonomy"
    fi

    echo ""
    echo "  generated index files in $STAGE_DIR/:"
    ls -lh "$STAGE_DIR/${DB}.idx"* 2>/dev/null || echo "  WARNING: no .idx generated for $DB"
    echo ""

    # Free tmp scratch (potentially huge).
    rm -rf "$TMP_DIR"
    return 0
}


rc=0
for DB in $DBS; do
    if ! index_one_db "$DB"; then
        rc=1
    fi
done

echo "=== All done (exit $rc) ==="
echo ""
echo "Next step — upload each stage dir to the Aliyun NAS mount used by FC."
echo "\`rsync -aL\` follows symlinks so both the immutable source DB files and"
echo "the freshly-built .idx sidecars land at the destination:"
echo ""
for DB in $DBS; do
    echo "  rsync -aL --info=progress2 $OUTPUT_DIR/$DB.stage/ \\"
    echo "      <NAS_HOST>:/nas/path/to/models/mmseqs2/"
done
echo ""
echo "FC-side container mounts that NAS path at /data/models/mmseqs2/."
echo "Verify with:  curl \$FC_URL/healthz/detail | jq .db_loaded"

exit $rc
