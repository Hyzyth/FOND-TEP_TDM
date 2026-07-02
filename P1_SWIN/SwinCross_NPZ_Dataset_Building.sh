#!/bin/bash
# =============================================================================
# SwinCross Dataset Building — NPZ offline preprocessing
# Author : Ethan
# Project : ProjetMaster / StageM1_IA
# Updated : 2026-05 (NPZ pipeline)
# =============================================================================

set -e

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          GLOBAL CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── 1. Execution Toggles (Set to true to build, false to skip) ─────────────
BUILD_HECKTOR_2025=false
BUILD_HECKTOR_2026_KFOLD=true
BUILD_TEMPORAL=false

# ── 2. Parameters ──────────────────────────────────────────────────────────
TRAIN_RATIO=0.8
K_FOLDS=5
SEED=42

# ── 3. Paths ───────────────────────────────────────────────────────────────
DATA_2025="/data/santiago/HECKTOR_data/2025/Task_1_segmentation"
OUT_2025="/data/ethan/PP_hecktor_swincross_npz"

DATA_2026="/data/santiago/HECKTOR_data/2026/HECKTOR 2026 Training Data"
OUT_2026="/data/ethan/PP_hecktor2026_kfold_npz"

DATA_TEMPORAL="/data/santiago/Database_nifti_TEMPORAL"
OUT_TEMPORAL="/data/ethan/PP_temporal_swincross_npz"


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                              ENVIRONMENT                               ║
# ╚════════════════════════════════════════════════════════════════════════╝
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.12
[ ! -d "swincross_env" ] && uv venv swincross_env --python 3.12
source swincross_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt || { echo "requirements.txt missing"; exit 1; }


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                               EXECUTION                                ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── A. HECKTOR 2025 (Classic) ─────────────────────────────────────────────
if [ "$BUILD_HECKTOR_2025" = true ]; then
    echo "╔══════════════════════════════════╗"
    echo "║  HECKTOR 2025 → SwinCross NPZ    ║"
    echo "╚══════════════════════════════════╝"
    mkdir -p $OUT_2025
    python3.12 adaptation/prepare_hecktor_npz_swincross.py \
        --data_dir   $DATA_2025 \
        --output_dir $OUT_2025 \
        --json_name  dataset_swincross.json \
        --val_split  1-$TRAIN_RATIO \
        --seed       $SEED \
        2>&1 | tee $OUT_2025/preprocessing.log
    echo ""
fi

# ── B. HECKTOR 2026 (K-Fold) ──────────────────────────────────────────────
if [ "$BUILD_HECKTOR_2026_KFOLD" = true ]; then
    echo "╔═════════════════════════════════════════════════╗"
    echo "║  HECKTOR 2026 → SwinCross k-fold NPZ build      ║"
    echo "╚═════════════════════════════════════════════════╝"
    mkdir -p $OUT_2026
    python3.12 adaptation/prepare_hecktor2026_kfold_npz.py \
        --data_dir    "$DATA_2026" \
        --output_dir  $OUT_2026 \
        --train_ratio $TRAIN_RATIO \
        --k_folds     $K_FOLDS \
        --json_prefix dataset_swincross_2026kfold \
        --seed        $SEED \
        2>&1 | tee $OUT_2026/preprocessing.log
    echo ""
fi

# ── C. TemPoRAL (Zero-Shot) ───────────────────────────────────────────────
if [ "$BUILD_TEMPORAL" = true ]; then
    echo "╔══════════════════════════════════════╗"
    echo "║  TemPoRAL → SwinCross NPZ build      ║"
    echo "╚══════════════════════════════════════╝"
    mkdir -p $OUT_TEMPORAL
    python3.12 adaptation/prepare_temporal_npz_swincross.py \
        --input_folder  $DATA_TEMPORAL \
        --output_folder $OUT_TEMPORAL \
        --json_name     dataset_swincross_temporal.json \
        --timepoints    all \
        2>&1 | tee $OUT_TEMPORAL/preprocessing.log
    echo ""
fi

echo "  Dataset building sequence complete."
