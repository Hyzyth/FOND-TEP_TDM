#!/bin/bash
# =============================================================================
# DualwaveSAM 3-Class LMDB Cache Builder
# Author  : Ethan
# =============================================================================

set -e

echo "╔═══════════════════════════════════════════════╗"
echo "║   DualwaveSAM 3-Class LMDB Database Builder   ║"
echo "╚═══════════════════════════════════════════════╝"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Environment ────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.12
[ ! -d "dualwave_env" ] && uv venv dualwave_env --python 3.12
source dualwave_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt || \
    { echo "[WARN] requirements.txt not found — ensure deps are installed"; }

# ── 2. Configuration ──────────────────────────────────────────────────────
PPDATA_FOLDER="/data/ethan/PP_hecktor2026_kfold_npz"
JSON_PREFIX="dataset_swincross_2026kfold"
OUT_DIR="/data/ethan/DualwaveSAM3c/lmdb_cache"
K_FOLDS=5
BUILD_CLASSIC=true
BUILD_KFOLD=true
BUILD_FULL=true
BUILD_EVAL=true

mkdir -p "$OUT_DIR"

# ── 3. Helper Function ────────────────────────────────────────────────────
build_cache() {
    local JSON_FILE=$1
    if [ -f "$PPDATA_FOLDER/$JSON_FILE" ]; then
        echo "------------------------------------------------------------"
        echo "▶ Building Cache for: $JSON_FILE"
        python3.12 adaptation/build_lmdb_cache.py \
            --data_dir "$PPDATA_FOLDER" \
            --json_list "$JSON_FILE" \
            --out_dir "$OUT_DIR"
    else
        echo "[SKIP] JSON not found: $PPDATA_FOLDER/$JSON_FILE"
    fi
}

# ── 4. Execution ──────────────────────────────────────────────────────────

if [ "$BUILD_CLASSIC" = true ]; then
    build_cache "${JSON_PREFIX}_classic.json"
fi

if [ "$BUILD_KFOLD" = true ]; then
    for fold in $(seq 0 $((K_FOLDS - 1))); do
        build_cache "${JSON_PREFIX}_fold${fold}.json"
    done
fi

if [ "$BUILD_FULL" = true ]; then
    build_cache "${JSON_PREFIX}_full.json"
fi

if [ "$BUILD_EVAL" = true ]; then
    build_cache "${JSON_PREFIX}_test.json"
    build_cache "${JSON_PREFIX}_classic_train.json"
    build_cache "${JSON_PREFIX}_classic_val.json"
fi

echo "============================================================"
echo "[✓] All LMDB caches built successfully in: $OUT_DIR"
echo "You are now ready to run DualwaveSAM3c_training.sh!"
