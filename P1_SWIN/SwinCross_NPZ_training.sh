#!/bin/bash
# =============================================================================
# SwinCross Training Script — SwinUNETR Cross-Modality Fusion
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Data source  : /data/ethan/PP_hecktor_2026_kfold_npz/   (NPZ, read by data_utils.py)
# All outputs  : /data/ethan/SwinCross/<MODEL_DIR>/
#
#     Run prepare_hecktor2026_kfold_npz.py (Dataset Building script) first
#     to generate the NPZ files and per-fold JSONs.
# =============================================================================

set -e

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          GLOBAL CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── 1. Execution Toggles (Set to true to run, false to skip) ───────────────
RUN_CLASSIC_TRAIN=false
RUN_CLASSIC_RESUME=false

RUN_KFOLD_TRAIN=true
RUN_KFOLD_PRODUCTION_FULL=false  # Trains a final model on 100% of the train pool

# ── 2. Hardware & Hyperparameters ──────────────────────────────────────────
GPU=0
EPOCH_NUMBER_CLASSIC=1000
EPOCH_NUMBER_KFOLD=400
BATCH_SIZE=2
CACHE_RATE=0.35  # Set to 0.0 if you lack RAM

# ── 3. Data Paths & Naming ─────────────────────────────────────────────────
PPDATA_FOLDER="/data/ethan/PP_hecktor2026_kfold_npz"
JSON_PREFIX="dataset_swincross_2026kfold"

CLASSIC_MODEL_DIR="HECKTOR_run_${EPOCH_NUMBER_CLASSIC}_epoch"
KFOLD_BASE_DIR="HECKTOR_kfold_${EPOCH_NUMBER_KFOLD}ep"
K_FOLDS=5


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                              ENVIRONMENT                               ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── STEP 0 — Environment ──────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
uv python install 3.12
[ ! -d "swincross_env" ] && uv venv swincross_env --python 3.12
source swincross_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt || { echo "requirements.txt missing"; exit 1; }

mkdir -p /data/ethan/SwinCross
ln -sfn /data/ethan/SwinCross ./runs


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                               EXECUTION                                ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── A. CLASSIC TRAINING ───────────────────────────────────────────────────
if [ "$RUN_CLASSIC_TRAIN" = true ]; then
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  CLASSIC TRAINING (${EPOCH_NUMBER_CLASSIC} Epochs)                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    mkdir -p /data/ethan/SwinCross/$CLASSIC_MODEL_DIR
    
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3.12 -u npz_version/train.py \
        --data_dir   $PPDATA_FOLDER \
        --logdir     $CLASSIC_MODEL_DIR \
        --json_list  ${JSON_PREFIX}_classic.json \
        --batch_size $BATCH_SIZE \
        --val_every  20 \
        --workers    4 \
        --cache_rate $CACHE_RATE \
        --max_epochs $EPOCH_NUMBER_CLASSIC \
        --warmup_epochs 50 \
        --RandFlipd_prob           0.5 \
        --RandRotate90d_prob       0.5 \
        --RandScaleIntensityd_prob 0.2 \
        --RandShiftIntensityd_prob 0.2 \
        --noamp \
        --save_checkpoint \
        2>&1 | tee /data/ethan/SwinCross/$CLASSIC_MODEL_DIR/training_from_scratch.log
fi

if [ "$RUN_CLASSIC_RESUME" = true ]; then
    echo "  ▶ Resuming Classic Training from model_last.pth."
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3.12 -u npz_version/train.py \
        --data_dir        $PPDATA_FOLDER \
        --json_list       ${JSON_PREFIX}_classic.json \
        --logdir          $CLASSIC_MODEL_DIR \
        --checkpoint      ./runs/$CLASSIC_MODEL_DIR/model_last.pth \
        --max_epochs      $EPOCH_NUMBER_CLASSIC \
        --warmup_epochs   50 \
        --batch_size      $BATCH_SIZE \
        --val_every       20 \
        --lrschedule      warmup_cosine \
        --noamp \
        --save_checkpoint \
        2>&1 | tee /data/ethan/SwinCross/$CLASSIC_MODEL_DIR/training_resume.log
fi

# ── B. K-FOLD CROSS VALIDATION ────────────────────────────────────────────
if [ "$RUN_KFOLD_TRAIN" = true ]; then
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  K-FOLD TRAINING (k=${K_FOLDS}, ${EPOCH_NUMBER_KFOLD} Epochs)                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    
    # Sanity check JSONs
    for fold in $(seq 0 $((K_FOLDS - 1))); do
        if [ ! -f "$PPDATA_FOLDER/${JSON_PREFIX}_fold${fold}.json" ]; then
            echo "  /!\ Missing JSON for Fold $fold. Run Dataset Builder first."
            exit 1
        fi
    done

    for fold in $(seq 0 $((K_FOLDS - 1))); do
        JSON_LIST="${JSON_PREFIX}_fold${fold}.json"
        MODEL_DIR="${KFOLD_BASE_DIR}_fold${fold}"
        mkdir -p /data/ethan/SwinCross/$MODEL_DIR

        echo "  ▶ Running Fold ${fold} / $((K_FOLDS - 1))"
        CUDA_VISIBLE_DEVICES=$GPU \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python3.12 -u npz_version/train.py \
            --data_dir      $PPDATA_FOLDER \
            --logdir        $MODEL_DIR \
            --json_list     $JSON_LIST \
            --batch_size    $BATCH_SIZE \
            --val_every     20 \
            --workers       4 \
            --cache_rate    $CACHE_RATE \
            --max_epochs    $EPOCH_NUMBER_KFOLD \
            --warmup_epochs 50 \
            --RandFlipd_prob           0.5 \
            --RandRotate90d_prob       0.5 \
            --RandScaleIntensityd_prob 0.2 \
            --RandShiftIntensityd_prob 0.2 \
            --noamp \
            --save_checkpoint \
            2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_fold${fold}.log
        
        echo "  [V] Fold ${fold} complete."
    done
fi

# ── C. K-FOLD PRODUCTION (100% Data) ──────────────────────────────────────
if [ "$RUN_KFOLD_PRODUCTION_FULL" = true ]; then
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  PRODUCTION MODEL (100% of Train Pool Data)                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    FULL_MODEL_DIR="${KFOLD_BASE_DIR}_full_production"
    mkdir -p /data/ethan/SwinCross/$FULL_MODEL_DIR
    
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3.12 -u npz_version/train.py \
        --data_dir      $PPDATA_FOLDER \
        --logdir        $FULL_MODEL_DIR \
        --json_list     ${JSON_PREFIX}_full.json \
        --batch_size    $BATCH_SIZE \
        --val_every     20 \
        --workers       4 \
        --cache_rate    $CACHE_RATE \
        --max_epochs    $EPOCH_NUMBER_KFOLD \
        --warmup_epochs 50 \
        --RandFlipd_prob           0.5 \
        --RandRotate90d_prob       0.5 \
        --RandScaleIntensityd_prob 0.2 \
        --RandShiftIntensityd_prob 0.2 \
        --noamp \
        --save_checkpoint \
        2>&1 | tee /data/ethan/SwinCross/$FULL_MODEL_DIR/training_production_full.log
fi

echo "  Training script complete."
